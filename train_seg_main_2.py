import sys
from optparse import OptionParser
import random
import torch
import numpy as np
from config.network_config import ConfigHolder
from loaders import dataset_loader
import global_config
from utils import plot_utils
from trainers import segmentation_trainer
from tqdm import tqdm
from utils.config_parser import ConfigParser

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Server config index", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--network_version', type=str, default="synseg_v00.00_fcg.1.1")
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
        "dataset_version_train": config.get('dataset_version', "CityScapes"),
        "dataset_version_test": config.get('dataset_version', "CityScapes"),
        "img_path_train": config.get('img_path_train'),
        "mask_path_train": config.get('mask_path_train'),
        "label_path_train": config.get('label_path_train'),
        "img_path_test": config.get('img_path_test'),
        "mask_path_test": config.get('mask_path_test'),
        "label_path_test": config.get('label_path_test'),
        "batch_size": config.get('batch_size', [256]*4),
        "load_size": config.get('load_size', [128]*4)
    }
    old_hyperparam_data = {"hyperparams": {cp.hyper_id: config.get('hyperparams', {})}}
    old_weight_data = {"loss_weights": {cp.loss_id: config.get('losses', {})}}
    
    ConfigHolder.initialize(old_network_config, old_hyperparam_data, old_weight_data)
    
    global_config.seg_path_rgb_path_train = old_network_config["img_path_train"]
    global_config.seg_path_mask_path_train = old_network_config["mask_path_train"]
    global_config.seg_path_label_path_train = old_network_config["label_path_train"]
    global_config.seg_path_rgb_path_test = old_network_config["img_path_test"]
    global_config.seg_path_mask_path_test = old_network_config["mask_path_test"]
    global_config.seg_path_label_path_test = old_network_config["label_path_test"]
    global_config.batch_size = old_network_config["batch_size"][0]
    global_config.load_size = old_network_config["load_size"][0]
    global_config.num_workers = 8

    plot_utils.VisdomReporter.initialize()

    train_loader, train_count = dataset_loader.load_cityscapes_dataset_train(global_config.seg_path_rgb_path_train, global_config.seg_path_label_path_train)
    test_loader, test_count = dataset_loader.load_cityscapes_dataset_test(global_config.seg_path_rgb_path_test, global_config.seg_path_label_path_test)
    seg_t = segmentation_trainer.SegmentationTrainer(device)

    iteration = 0
    start_epoch = global_config.last_epoch_st
    
    load_size = global_config.load_size
    needed_progress = int((old_network_config["max_epochs"]) * (train_count / (load_size if load_size > 0 else 1)))
    pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)

    for epoch in range(start_epoch, old_network_config["max_epochs"]):
        for i, (_, train_img, train_mask) in enumerate(train_loader, 0):
            train_img = train_img.to(device)
            train_mask = train_mask.to(device)
            train_map = {"train_img" : train_img, "train_mask" : train_mask}
            seg_t.train(epoch, iteration, train_map)
            iteration = iteration + 1
            pbar.update(1)

            if (iteration % opts.save_per_iter == 0):
                seg_t.save_states(epoch, iteration, True)
                if global_config.plot_enabled == 1:
                    seg_t.visdom_plot(iteration)
                    seg_t.visdom_visualize({"img": train_img, "mask": train_mask}, "Train")
                    _, val_img, val_mask = next(iter(test_loader))
                    seg_t.visdom_visualize({"img": val_img.to(device), "mask": val_mask.to(device)}, "Test")

    pbar.close()

if __name__ == "__main__":
    main(sys.argv)
