import sys
from optparse import OptionParser
import random
import torch
import numpy as np
from config.network_config import ConfigHolder
from loaders import dataset_loader
import global_config
from utils import plot_utils
from testers import paired_tester
from tqdm import tqdm
from utils.config_parser import ConfigParser

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Server config index", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--network_version', type=str, default="synseg_v00.01_cityscapes.1.1")
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
    global_config.test_size = 2
    
    old_network_config = {
        "model_type": config.get('model_type'),
        "input_nc": config.get('input_nc'),
        "patch_size": config.get('patch_size', 512),
        "num_blocks": config.get('num_blocks'),
        "max_epochs": config.get('max_epochs', 200),
        "min_epochs": config.get('min_epochs', 10),
        "dataset_version": config.get('dataset_version', "CityScapes"),
        "img_path_test": config.get('img_path_test'),
        "mask_path_test": config.get('mask_path_test'),
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
    
    global_config.seg_path_rgb_path_test = old_network_config["img_path_test"]
    global_config.seg_path_mask_path_test = old_network_config["mask_path_test"]
    vram_index = global_config.get_vram_index(opts.server_config)
    global_config.batch_size = old_network_config["batch_size"][vram_index]
    global_config.load_size = old_network_config["load_size"][vram_index]
    global_config.num_workers = 8

    plot_utils.VisdomReporter.initialize()

    test_loader_a, test_count = dataset_loader.load_cityscapes_gan_dataset_test(global_config.seg_path_rgb_path_test, global_config.seg_path_mask_path_test)
    img2img_t = paired_tester.PairedTester(device)

    pbar = tqdm(total=test_count, disable=global_config.disable_progress_bar)

    for i, (file_name, img_batch, target_batch) in enumerate(test_loader_a, 0):
        img_batch = img_batch.to(device)
        target_batch = target_batch.to(device)
        input_map = {"file_name": file_name, "img_a": img_batch, "img_b": target_batch}

        img2img_t.measure_and_store(input_map)
        img2img_t.save_images(input_map)
        pbar.update(1)

        if ((i + 1) % 4 == 0):
            break

    if (global_config.plot_enabled == 1):
        img2img_t.visualize_results(input_map, "Test Dataset")
    img2img_t.report_metrics("Test Dataset")

    pbar.close()

if __name__ == "__main__":
    main(sys.argv)
