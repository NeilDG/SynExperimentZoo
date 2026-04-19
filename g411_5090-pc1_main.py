#Script to use for running heavy training with completion check.

import os
import torch
from utils.config_parser import ConfigParser

def is_training_complete(version_str, server_config):
    # 1. Load config to get max_epochs
    try:
        cp = ConfigParser(version_str)
        config = cp.load_config(server_config)
        max_epochs = config.get('max_epochs', 200)
    except Exception as e:
        print(f"Error loading config for {version_str}: {e}")
        return False

    # 2. Check for existing checkpoints
    # Logic matches PairedTrainer's load_saved_state: .pth then .pth.checkpt
    checkpath = os.path.join("checkpoint", f"{version_str}.pth")
    checkpt_path = checkpath + ".checkpt"
    
    found_path = None
    if os.path.exists(checkpath):
        found_path = checkpath
    elif os.path.exists(checkpt_path):
        found_path = checkpt_path
        
    if found_path:
        try:
            # Load on CPU to avoid VRAM overhead during the check
            checkpoint = torch.load(found_path, map_location='cpu', weights_only=True)
            current_epoch = checkpoint.get('epoch', 0)
            
            if current_epoch >= max_epochs:
                print(f"Skipping {version_str}: Already reached {current_epoch}/{max_epochs} epochs.")
                return True
            else:
                print(f"Resuming {version_str}: Current epoch {current_epoch}, Target {max_epochs}.")
                return False
        except Exception as e:
            print(f"Error reading checkpoint {found_path}: {e}")
            return False
            
    return False

def train_sr_main():
    server_config = 5
    model_ids = range(1, 15)
    hyper_ids = range(1, 13)
    loss_ids = range(1, 6)

    for m_id in model_ids:
        for h_id in hyper_ids:
            for l_id in loss_ids:
                version_str = f"mobisr_v02.{m_id:02d}_hypersim.{h_id}.{l_id}"
                
                if not is_training_complete(version_str, server_config):
                    command = (f"python \"train_sr_main.py\" --server_config={server_config} --img_to_load=-1 "
                               f"--plot_enabled=1 --save_per_iter=250 --network_version=\"{version_str}\"")
                    
                    print(f"Executing: {command}")
                    os.system(command)

def test_sr_main():
    server_config = 5
    model_ids = range(1, 15)
    
    for m_id in model_ids:
        version_str = f"mobisr_v02.{m_id:02d}_hypersim.12.3"
        os.system(f"python \"test_sr_main.py\" --server_config={server_config} --img_to_load=-1 "
                  f"--plot_enabled=0 --network_version=\"{version_str}\"")

def main():
    train_sr_main()

if __name__ == "__main__":
    main()
