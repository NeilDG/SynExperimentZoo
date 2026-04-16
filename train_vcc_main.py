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
parser.add_option('--vcc', type=str, help="VCC version (V.XX.YY.ZZ)", default="V.02.05div2k.10")
parser.add_option('--server', type=str, help="Server config name", default="default")
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--save_per_iter', type=int, default=500)

def main(argv):
    (opts, args) = parser.parse_args(argv)
    device = torch.device(opts.cuda_device if torch.cuda.is_available() else "cpu")
    
    # 1. Parse and Load Config
    cp = ConfigParser(opts.vcc)
    config = cp.load_config(opts.server)
    
    # Setup global_config for backward compatibility (some trainers still use it)
    global_config.sr_network_version = opts.vcc
    global_config.load_size = config.get('training.load_size', 1)
    global_config.batch_size = config.get('training.batch_size', 1)
    global_config.num_workers = config.get('training.num_workers', 4)
    global_config.save_per_iter = opts.save_per_iter
    
    # 2. Triple-Factory Pattern
    # Note: Currently PairedTrainer handles its own instantiation of models via NetworkCreator
    # To keep it simple for now, we inject the config and let it use it.
    # In a full refactor, we would pass the instantiated objects.
    
    # For now, let's use the factories to show the design
    train_loader, test_loader, train_count = DatasetFactory.get_dataloaders(config)
    
    # 3. Model & Loss Factory (Demonstration of Triple-Factory)
    # netG, netD = ModelFactory.create_model(config, device)
    # loss_repo, weights = LossFactory.get_losses(config, device)
    
    # Initialize the trainer (we still use PairedTrainer for now but it will read from ConfigHolder)
    # We need to initialize ConfigHolder for backward compatibility with existing trainers
    from config.network_config import ConfigHolder
    # Mock the old structures for ConfigHolder
    # This is a bridge between the new VCC and old code
    old_network_config = {
        "model_type": config.get('model.type'),
        "input_nc": config.get('model.input_nc'),
        "num_blocks": config.get('model.num_blocks'),
        "max_epochs": config.get('experiment.training.epochs', 200),
        "min_epochs": 10
    }
    old_hyperparam_data = {"hyperparams": {0: config.get('experiment.hyperparams', {})}}
    old_weight_data = {"loss_weights": {0: config.get('experiment.losses', {})}}
    global_config.hyper_iteration = 0
    global_config.loss_iteration = 0
    
    ConfigHolder.initialize(old_network_config, old_hyperparam_data, old_weight_data)
    
    trainer = paired_trainer.PairedTrainer(device)
    
    # Training Loop
    start_epoch = global_config.last_epoch_st
    max_epochs = old_network_config["max_epochs"]
    
    print(f"Starting training for {opts.vcc} on {opts.server}")
    
    pbar = tqdm(total=max_epochs * (train_count // global_config.load_size))
    iteration = 0
    
    for epoch in range(start_epoch, max_epochs):
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
