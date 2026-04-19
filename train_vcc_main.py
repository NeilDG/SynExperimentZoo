import sys
import torch
import random
import numpy as np
from optparse import OptionParser
from utils.config_parser import ConfigParser
from core.factory_model import ModelFactory
from core.factory_dataset import DatasetFactory
from core.factory_loss import LossFactory
from trainers import paired_trainer
from tqdm import tqdm
import global_config

parser = OptionParser()
parser.add_option('--vcc', type=str, help="VCC version (V.XX.YY.ZZ.WW)", default="V.02.06div2k.12.3")
parser.add_option('--server', type=str, help="Server config name", default="default")
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--save_per_iter', type=int, default=500)

def main(argv):
    (opts, args) = parser.parse_args(argv)
    device = torch.device(opts.cuda_device if torch.cuda.is_available() else "cpu")
    
    # 1. Parse and Load Config
    cp = ConfigParser(opts.vcc)
    config = cp.load_config(opts.server)
    
    # Setup global_config for backward compatibility
    global_config.sr_network_version = f"{cp.problem}_{cp.version}"
    global_config.hyper_iteration = cp.hyper_id
    global_config.loss_iteration = cp.loss_id
    
    global_config.load_size = config.get('training.load_size', 1)
    global_config.batch_size = config.get('training.batch_size', 1)
    global_config.num_workers = config.get('training.num_workers', 4)
    global_config.save_per_iter = opts.save_per_iter
    
    # 2. Triple-Factory Pattern
    train_loader, test_loader, train_count = DatasetFactory.get_dataloaders(cp)
    
    # Initialize the trainer
    from config.network_config import ConfigHolder
    # Mock the old structures for ConfigHolder
    old_network_config = {
        "model_type": config.get('model_type'),
        "input_nc": config.get('input_nc'),
        "num_blocks": config.get('num_blocks'),
        "max_epochs": config.get('max_epochs', 200),
        "min_epochs": 10
    }
    # For backward compat, we keep iterations at 0 since YAML merges them
    old_hyperparam_data = {"hyperparams": {cp.hyper_id: config.get('hyperparams', {})}}
    old_weight_data = {"loss_weights": {cp.loss_id: config.get('losses', {})}}
    
    ConfigHolder.initialize(old_network_config, old_hyperparam_data, old_weight_data)
    
    trainer = paired_trainer.PairedTrainer(device)
    
    # Training Loop
    start_epoch = global_config.last_epoch_st
    max_epochs = old_network_config["max_epochs"]
    
    print(f"Starting training for {opts.vcc} on {opts.server}")
    
    # Handle possible 0 division if data not found
    iter_per_epoch = (train_count // global_config.load_size) if global_config.load_size > 0 else 1
    pbar = tqdm(total=max_epochs * iter_per_epoch)
    iteration = 0
    
    for epoch in range(start_epoch, max_epochs):
        if train_loader is None: break
        for i, (_, a_batch, b_batch) in enumerate(train_loader):
            a_batch = a_batch.to(device)
            b_batch = b_batch.to(device)
            input_map = {"img_a": a_batch, "img_b": b_batch}
            
            trainer.train(epoch, iteration, input_map)
            
            iteration += 1
            pbar.update(1)
            
            if iteration % opts.save_per_iter == 0:
                trainer.save_states(epoch, iteration, True)

    pbar.close()

if __name__ == "__main__":
    main(sys.argv)
