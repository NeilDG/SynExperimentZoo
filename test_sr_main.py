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
parser.add_option('--network_version', type=str, default="mobisr_v02.05_hypersim.10.1")
parser.add_option('--save_images', type=int, default=0)
parser.add_option('--plot_enabled', type=int, default=1)

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
    global_config.test_size = 16
    global_config.server_config = opts.server_config
    
    old_network_config = {
        "model_type": config.get('model_type'),
        "input_nc": config.get('input_nc'),
        "patch_size": config.get('patch_size', 64),
        "num_blocks": config.get('num_blocks'),
        "max_epochs": config.get('max_epochs', 200),
        "min_epochs": config.get('min_epochs', 10),
        "dataset_version": config.get('dataset_version', "div2k"),
        "low_path": config.get('low_path_test'), 
        "high_path": config.get('high_path_test'),
        "batch_size": config.get('batch_size', [256]*4),
        "load_size": config.get('load_size', [128]*4)
    }
    old_hyperparam_data = {"hyperparams": {cp.hyper_id: config.get('hyperparams', {})}}
    old_weight_data = {"loss_weights": {cp.loss_id: config.get('losses', {})}}
    
    ConfigHolder.initialize(old_network_config, old_hyperparam_data, old_weight_data)
    
    global_config.a_path_test = config.get('low_path_test')
    global_config.b_path_test = config.get('high_path_test')
    global_config.num_workers = 8
    
    # BurstSR/Div2K paths from consolidated YAML if available, else fallback
    global_config.burst_sr_lr_path = config.get('burst_sr_lr_path', global_config.a_path_test)
    global_config.burst_sr_hr_path = config.get('burst_sr_hr_path', global_config.b_path_test)
    global_config.div2k_lr_path = config.get('div2k_lr_path', global_config.a_path_test)
    global_config.div2k_hr_path = config.get('div2k_hr_path', global_config.b_path_test)

    plot_utils.VisdomReporter.initialize()

    test_loader_a, test_count = dataset_loader.load_test_img2img_dataset(global_config.a_path_test, global_config.b_path_test)
    test_loader_div2k, _ = dataset_loader.load_test_img2img_dataset(global_config.div2k_lr_path, global_config.div2k_hr_path)

    img2img_t = paired_tester.PairedTester(device)
    print("---------------------------------------------------------------------------")
    print("Started synth test loop for mode: ", ConfigHolder.getInstance().get_sr_version_name())
    print("---------------------------------------------------------------------------")

    with torch.no_grad():
        for i, (file_name, a_batch, b_batch) in enumerate(test_loader_a, 0):
            a_batch = a_batch.to(device)
            b_batch = b_batch.to(device)
            input_map = {"file_name": file_name, "img_a" : a_batch, "img_b" : b_batch}
            img2img_t.measure_and_store(input_map)
            img2img_t.save_images(input_map)
            if((i + 1) % 4 == 0): break

        img2img_t.report_metrics("Train Dataset")

        for i, (file_name, a_batch, b_batch) in enumerate(test_loader_div2k, 0):
            a_batch = a_batch.to(device)
            b_batch = b_batch.to(device)
            input_map = {"file_name": file_name, "img_a": a_batch, "img_b": b_batch}
            img2img_t.measure_and_store(input_map)
            img2img_t.save_images(input_map)
            if ((i + 1) % 4 == 0): break

        img2img_t.report_metrics("Test - Div2k")

if __name__ == "__main__":
    main(sys.argv)
