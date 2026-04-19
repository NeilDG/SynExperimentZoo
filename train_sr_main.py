import sys
from optparse import OptionParser
import random
import torch
import numpy as np
from config.network_config import ConfigHolder
from loaders import dataset_loader
import global_config
from utils import plot_utils
from trainers import paired_trainer
from tqdm import tqdm
from utils.config_parser import ConfigParser

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Server config index", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--network_version', type=str, default="mobisr_v02.06_div2k.12.3")
parser.add_option('--plot_enabled', type=int, default=1)
parser.add_option('--save_per_iter', type=int, default=500)

def main(argv):
    (opts, args) = parser.parse_args(argv)
    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    manualSeed = 0
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

    # Use only new VCC parser
    cp = ConfigParser(opts.network_version)
    config = cp.load_config(opts.server_config) 
    
    global_config.sr_network_version = f"{cp.problem}_{cp.version}"
    global_config.hyper_iteration = cp.hyper_id
    global_config.loss_iteration = cp.loss_id
    
    global_config.plot_enabled = opts.plot_enabled
    global_config.img_to_load = opts.img_to_load
    global_config.cuda_device = opts.cuda_device
    global_config.save_per_iter = opts.save_per_iter
    global_config.server_config = opts.server_config
    global_config.test_size = 2
    
    # Mock the old structures for ConfigHolder
    old_network_config = {
        "model_type": config.get('model_type'),
        "input_nc": config.get('input_nc'),
        "patch_size": config.get('patch_size', 64),
        "num_blocks": config.get('num_blocks'),
        "max_epochs": config.get('max_epochs', 200),
        "min_epochs": config.get('min_epochs', 10),
        "dataset_version": config.get('dataset_version', "div2k"),
        "low_path": config.get('low_path_train'), 
        "high_path": config.get('high_path_train'),
        "batch_size": config.get('batch_size', [256]*4),
        "load_size": config.get('load_size', [128]*4)
    }
    old_hyperparam_data = {"hyperparams": {cp.hyper_id: config.get('hyperparams', {})}}
    old_weight_data = {"loss_weights": {cp.loss_id: config.get('losses', {})}}
    
    ConfigHolder.initialize(old_network_config, old_hyperparam_data, old_weight_data)
    
    # Setup global_config paths directly since they are resolved by new parser
    global_config.a_path_train = config.get('low_path_train')
    global_config.b_path_train = config.get('high_path_train')
    global_config.a_path_test = config.get('low_path_test')
    global_config.b_path_test = config.get('high_path_test')
    vram_index = global_config.get_vram_index(opts.server_config)
    global_config.batch_size = old_network_config["batch_size"][vram_index]
    global_config.load_size = old_network_config["load_size"][vram_index]
    global_config.num_workers = 8 # Default

    print(opts)
    print("=====================BEGIN============================")
    
    network_config = ConfigHolder.getInstance().get_network_config()
    hyperparams_table = ConfigHolder.getInstance().get_all_hyperparams()
    loss_config_table = ConfigHolder.getInstance().get_loss_weights()["loss_weights"][cp.loss_id]
    
    print("Network version:", opts.network_version, ". Hyper parameters: ", hyperparams_table, " Loss weights: ", loss_config_table, " model_type:", network_config["model_type"], " num_blocks:", network_config["num_blocks"], " batch_size:", global_config.batch_size, " load_size:", global_config.load_size, " min_epochs:", network_config["min_epochs"], " max_epochs:", network_config["max_epochs"])

    plot_utils.VisdomReporter.initialize()

    train_loader, train_count = dataset_loader.load_train_img2img_dataset(global_config.a_path_train, global_config.b_path_train)
    test_loader, test_count = dataset_loader.load_test_img2img_dataset(global_config.a_path_test, global_config.b_path_test)
    img2img_t = paired_trainer.PairedTrainer(device)

    iteration = 0
    start_epoch = global_config.last_epoch_st
    
    # compute total progress
    load_size = global_config.load_size
    needed_progress = int((network_config["max_epochs"]) * (train_count / (load_size if load_size > 0 else 1)))
    pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)

    for epoch in range(start_epoch, network_config["max_epochs"]):
        for i, (_, a_batch, b_batch) in enumerate(train_loader, 0):
            a_batch = a_batch.to(device)
            b_batch = b_batch.to(device)
            input_map = {"img_a" : a_batch, "img_b" : b_batch}
            img2img_t.train(epoch, iteration, input_map)

            iteration = iteration + 1
            pbar.update(1)

            if(iteration % opts.save_per_iter == 0):
                img2img_t.save_states(epoch, iteration, True)

                if global_config.plot_enabled == 1 and iteration % (opts.save_per_iter * 128) == 0:
                    img2img_t.visdom_plot(iteration)
                    img2img_t.visdom_visualize(input_map, "Train")

                    _, a_test_batch, b_test_batch = next(iter(test_loader))
                    a_test_batch = a_test_batch.to(device, non_blocking = True)
                    b_test_batch = b_test_batch.to(device, non_blocking = True)

                    input_map = {"img_a": a_test_batch, "img_b": b_test_batch}
                    img2img_t.visdom_visualize(input_map, "Test")

    pbar.close()

if __name__ == "__main__":
    main(sys.argv)
