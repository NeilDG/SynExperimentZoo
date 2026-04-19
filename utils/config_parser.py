import yaml
import os
from yaml.loader import SafeLoader

class ConfigParser:
    def __init__(self, vcc_string):
        """
        Parses VCC string: problem_version.hyper.loss
        Example: mobisr_v02.06_div2k.12.3
        problem - mobisr (loads configs/mobisr.yaml)
        version - v02.06_div2k (lookup in versions: block)
        hyper   - 12 (lookup in hyperparams: block)
        loss    - 3 (lookup in loss_weights: block)
        """
        self.vcc_string = vcc_string
        self.problem, self.version, self.hyper_id, self.loss_id = self._parse_vcc(vcc_string)
        self.config = {}

    def _parse_vcc(self, vcc_string):
        # Handle optional V. prefix
        clean_string = vcc_string[2:] if vcc_string.startswith("V.") else vcc_string
        
        # Split by last two dots for hyper and loss
        parts = clean_string.rsplit('.', 2)
        if len(parts) != 3:
             raise ValueError(f"Invalid VCC format: {vcc_string}. Expected problem_version.hyper.loss")
        
        hyper_id = parts[1]
        loss_id = parts[2]
        
        # Split problem and version by first underscore
        sub_parts = parts[0].split('_', 1)
        if len(sub_parts) != 2:
             raise ValueError(f"Invalid VCC prefix: {parts[0]}. Expected problem_version")
        
        return sub_parts[0], sub_parts[1], hyper_id, loss_id

    def load_config(self, server_config_index=0):
        yaml_path = os.path.join("configs", f"{self.problem}.yaml")
        raw_config = self._read_yaml(yaml_path)
        if not raw_config:
            raise FileNotFoundError(f"Problem config {yaml_path} not found.")

        # 1. Start with common parameters
        self.config = raw_config.get('common', {}).copy()

        # 2. Get server path
        server_paths = raw_config.get('server_paths', {})
        server_path = server_paths.get(server_config_index, "")
        self.config['server_path'] = server_path

        # 3. Merge version-specific config
        version_info = raw_config.get('versions', {}).get(self.version, {})
        self.config = self._deep_merge(self.config, version_info)

        # 4. Merge hyperparams
        # Note: hyper_id might be string or int in YAML, we try both
        hyper_table = raw_config.get('hyperparams', {})
        hyper_info = hyper_table.get(self.hyper_id) or hyper_table.get(int(self.hyper_id) if self.hyper_id.isdigit() else -1, {})
        self.config['hyperparams'] = hyper_info

        # 5. Merge loss weights
        loss_table = raw_config.get('loss_weights', {})
        loss_info = loss_table.get(self.loss_id) or loss_table.get(int(self.loss_id) if self.loss_id.isdigit() else -1, {})
        self.config['losses'] = loss_info

        # 6. Resolve path placeholders
        self._resolve_paths(self.config, server_path)

        # Inject metadata
        self.config['vcc'] = {
            'string': self.vcc_string,
            'problem': self.problem,
            'version': self.version,
            'hyper_id': self.hyper_id,
            'loss_id': self.loss_id,
            'server_index': server_config_index
        }
        
        return self.config

    def _resolve_paths(self, config_dict, server_path):
        """Recursively replaces {server_path} in strings."""
        for key, value in config_dict.items():
            if isinstance(value, str) and "{server_path}" in value:
                config_dict[key] = value.replace("{server_path}", server_path)
            elif isinstance(value, dict):
                self._resolve_paths(value, server_path)

    def _read_yaml(self, path):
        if not os.path.exists(path):
            return {}
        with open(path, 'r') as f:
            return yaml.load(f, Loader=SafeLoader) or {}

    def _deep_merge(self, base, override):
        merged = base.copy()
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def get(self, key, default=None):
        keys = key.split('.')
        val = self.config
        try:
            for k in keys:
                if isinstance(val, dict):
                    val = val.get(k)
                else:
                    return default
            return val if val is not None else default
        except (KeyError, TypeError):
            return default
