import sys
from optparse import OptionParser
import random
import torch
import numpy as np
from config.network_config import ConfigHolder
from loaders import dataset_loader
import global_config
from utils import plot_utils
from trainers import img2imgtrainer
from tqdm import tqdm
from utils.config_parser import ConfigParser

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Server config index", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--network_version', type=str, default="fcg2cityscapes_v00.01.1.1")
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
    global_config.test_size = 16
    
    old_network_config = {
        "model_type": config.get('model_type'),
        "input_nc": config.get('input_nc'),
        "patch_size": config.get('patch_size', 64),
        "num_blocks": config.get('num_blocks'),
        "max_epochs": config.get('max_epochs', 200),
        "min_epochs": config.get('min_epochs', 10),
        "dataset_a_train": config.get('dataset_a_train'),
        "dataset_b_train": config.get('dataset_b_train'),
        "dataset_a_test": config.get('dataset_a_test'),
        "dataset_b_test": config.get('dataset_b_test'),
        "batch_size": config.get('batch_size', [256]*4),
        "load_size": config.get('load_size', [128]*4)
    }
    old_hyperparam_data = {"hyperparams": {cp.hyper_id: config.get('hyperparams', {})}}
    old_weight_data = {"loss_weights": {cp.loss_id: config.get('losses', {})}}
    
    ConfigHolder.initialize(old_network_config, old_hyperparam_data, old_weight_data)
    
    network_config = ConfigHolder.getInstance().get_network_config()
    hyperparams_table = ConfigHolder.getInstance().get_all_hyperparams()
    loss_config_table = ConfigHolder.getInstance().get_loss_weights()["loss_weights"][cp.loss_id]
    
    print("Network version:", opts.network_version, ". Hyper parameters: ", hyperparams_table, " Loss weights: ", loss_config_table, " model_type:", network_config["model_type"], " num_blocks:", network_config["num_blocks"], " batch_size:", global_config.batch_size, " load_size:", global_config.load_size, " min_epochs:", network_config["min_epochs"], " max_epochs:", network_config["max_epochs"])
    
    global_config.a_path_test = old_network_config["dataset_a_test"]
    global_config.b_path_test = old_network_config["dataset_b_test"]
    vram_index = global_config.get_vram_index(opts.server_config)
    global_config.batch_size = old_network_config["batch_size"][vram_index]
    global_config.load_size = old_network_config["load_size"][vram_index]
    global_config.num_workers = 8

    plot_utils.VisdomReporter.initialize()

    test_loader, test_count = dataset_loader.load_test_img2img_dataset(global_config.a_path_test, global_config.b_path_test)
    img2img_t = img2imgtrainer.Img2ImgTrainer(device)

    pbar = tqdm(total=test_count, disable=global_config.disable_progress_bar)

    for i, (_, a_batch, b_batch) in enumerate(test_loader, 0):
        a_batch = a_batch.to(device)
        b_batch = b_batch.to(device)
        input_map = {"img_a" : a_batch, "img_b" : b_batch}

        pbar.update(1)
        if global_config.plot_enabled == 1:
            img2img_t.visdom_visualize(input_map, "Test")

    pbar.close()

if __name__ == "__main__":
    main(sys.argv)
