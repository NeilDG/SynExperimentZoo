import yaml
import os
from yaml.loader import SafeLoader

class ConfigParser:
    def __init__(self, vcc_string):
        """
        Parses VCC string V.XX.YY.ZZ and loads corresponding configs.
        XX - Significant code changes (configs/models/vXX.yaml)
        YY - Network/Dataset/Hyperparams (configs/datasets/YY.yaml)
        ZZ - Loss weight adjustments (configs/experiments/ZZ.yaml)
        """
        self.vcc_string = vcc_string
        self.xx, self.yy, self.zz = self._parse_vcc(vcc_string)
        self.config = {}

    def _parse_vcc(self, vcc_string):
        # Handle "V." prefix if present
        if vcc_string.startswith("V."):
            parts = vcc_string[2:].split(".")
        else:
            parts = vcc_string.split(".")
        
        if len(parts) != 3:
            raise ValueError(f"Invalid VCC string format: {vcc_string}. Expected V.XX.YY.ZZ")
        
        return parts[0], parts[1], parts[2]

    def load_config(self, server_name="default"):
        # 1. Load Model Config (XX)
        model_path = os.path.join("configs", "models", f"v{self.xx}.yaml")
        model_config = self._read_yaml(model_path)

        # 2. Load Dataset/Hyperparam Config (YY)
        dataset_path = os.path.join("configs", "datasets", f"{self.yy}.yaml")
        dataset_config = self._read_yaml(dataset_path)

        # 3. Load Experiment/Loss Config (ZZ)
        experiment_path = os.path.join("configs", "experiments", f"{self.zz}.yaml")
        experiment_config = self._read_yaml(experiment_path)

        # 4. Load Server Config
        server_path = os.path.join("configs", "servers", f"{server_name}.yaml")
        server_config = self._read_yaml(server_path)

        # Merge configs (Server > Experiment > Dataset > Model)
        self.config = self._deep_merge(model_config, dataset_config)
        self.config = self._deep_merge(self.config, experiment_config)
        self.config = self._deep_merge(self.config, server_config)
        
        # Inject VCC info
        self.config['vcc'] = {
            'string': self.vcc_string,
            'xx': self.xx,
            'yy': self.yy,
            'zz': self.zz
        }
        
        return self.config

    def _read_yaml(self, path):
        if not os.path.exists(path):
            print(f"Warning: Config file {path} not found. Returning empty dict.")
            return {}
        with open(path, 'r') as f:
            return yaml.load(f, Loader=SafeLoader) or {}

    def _deep_merge(self, base, override):
        """Recursively merges override into base."""
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
