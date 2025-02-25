import itertools
import sys
from optparse import OptionParser
import random

import datasets
import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
from super_image import ImageLoader

from config.network_config import ConfigHolder
from loaders import dataset_loader
import global_config
from utils import plot_utils
from testers import paired_tester
from tqdm import tqdm
import yaml
from yaml.loader import SafeLoader
import util_script_main as utils_script

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--network_version', type=str, default="vXX.XX")
parser.add_option('--save_images', type=int, default=0)
parser.add_option('--plot_enabled', type=int, default=1)

def update_config(opts):
    global_config.server_config = opts.server_config
    global_config.plot_enabled = opts.plot_enabled
    global_config.img_to_load = opts.img_to_load
    global_config.cuda_device = opts.cuda_device
    global_config.test_size = 8

    network_config = ConfigHolder.getInstance().get_network_config()
    dataset_version = network_config["dataset_version"]
    low_path = network_config["low_path"]
    high_path = network_config["high_path"]

    if (global_config.server_config == 0):  #RTX 4060Ti PC
        global_config.num_workers = 8
        global_config.a_path_train = "C:/Datasets/SuperRes Dataset/{dataset_version}{low_path}"
        global_config.b_path_train = "C:/Datasets/SuperRes Dataset/{dataset_version}{high_path}"
        global_config.a_path_test = "C:/Datasets/SuperRes Dataset/{dataset_version}{low_path}"
        global_config.b_path_test = "C:/Datasets/SuperRes Dataset/{dataset_version}{high_path}"
        global_config.burst_sr_lr_path = "C:/Datasets/SuperRes Dataset/v02_burstsr/low/*.png"
        global_config.burst_sr_hr_path = "C:/Datasets/SuperRes Dataset/v02_burstsr/high/*.png"
        global_config.div2k_lr_path = "C:/Datasets/SuperRes Dataset/div2k/lr/*.png"
        global_config.div2k_hr_path = "C:/Datasets/SuperRes Dataset/div2k/bicubic_x4/*.png"
        global_config.batch_size = network_config["batch_size"][1]
        global_config.load_size = network_config["load_size"][1]
        print("Using G411-RTX4060Ti configuration. Workers: ", global_config.num_workers)

    elif (global_config.server_config == 1):  # CCS Cloud
        global_config.num_workers = 12
        global_config.a_path_train = "/home/jupyter-neil.delgallego/SuperRes Dataset/{dataset_version}/low/train_patches/*.jpg"
        global_config.b_path_train = "/home/jupyter-neil.delgallego/SuperRes Dataset/{dataset_version}/high/train_patches/*.jpg"
        global_config.a_path_test = "/home/jupyter-neil.delgallego/SuperRes Dataset/{dataset_version}/low/test_images/*.jpg"
        global_config.b_path_test = "/home/jupyter-neil.delgallego/SuperRes Dataset/{dataset_version}/high/test_images/*.jpg"
        global_config.batch_size = network_config["batch_size"][1]
        global_config.load_size = network_config["load_size"][1]
        print("Using CCS configuration. Workers: ", global_config.num_workers)

    elif (global_config.server_config == 2):  # RTX 2080Ti
        global_config.num_workers = 6
        global_config.a_path_train = "X:/SuperRes Dataset/{dataset_version}/low/train_patches/*.jpg"
        global_config.b_path_train = "X:/SuperRes Dataset/{dataset_version}/high/train_patches/*.jpg"
        global_config.a_path_test = "X:/SuperRes Dataset/{dataset_version}/low/test_images/*.jpg"
        global_config.b_path_test = "X:/SuperRes Dataset/{dataset_version}/high/test_images/*.jpg"
        global_config.batch_size = network_config["batch_size"][2]
        global_config.load_size = network_config["load_size"][2]
        print("Using RTX 2080Ti configuration. Workers: ", global_config.num_workers)

    elif (global_config.server_config == 3):  # RTX 3090 PC
        global_config.num_workers = 12
        global_config.a_path_train = "X:/SuperRes Dataset/{dataset_version}{low_path}"
        global_config.b_path_train = "X:/SuperRes Dataset/{dataset_version}{high_path}"
        global_config.a_path_test = "X:/SuperRes Dataset/{dataset_version}{low_path}"
        global_config.b_path_test = "X:/SuperRes Dataset/{dataset_version}{high_path}"
        global_config.burst_sr_lr_path = "X:/SuperRes Dataset/v02_burstsr/val/*/samsung_00/im_rgb_*.png"
        global_config.burst_sr_hr_path = "X:/SuperRes Dataset/v02_burstsr/val/*/canon/im_rgb_*.png"
        global_config.div2k_lr_path = "X:/SuperRes Dataset/div2k/lr/*.png"
        global_config.div2k_hr_path = "X:/SuperRes Dataset/div2k/bicubic_x4/*.png"
        global_config.batch_size = network_config["batch_size"][0]
        global_config.load_size = network_config["load_size"][0]
        print("Using RTX 3090 configuration. Workers: ", global_config.num_workers)

    elif (global_config.server_config == 4):  # @TITAN1 - 3
        global_config.num_workers = 4
        global_config.a_path_train = "/home/neildelgallego/SuperRes Dataset/{dataset_version}/low/train_patches/*.jpg"
        global_config.b_path_train = "/home/neildelgallego/SuperRes Dataset/{dataset_version}/high/train_patches/*.jpg"
        global_config.a_path_test = "/home/neildelgallego/SuperRes Dataset/{dataset_version}/low/test_images/*.jpg"
        global_config.b_path_test = "/home/neildelgallego/SuperRes Dataset/{dataset_version}/high/test_images/*.jpg"
        global_config.batch_size = network_config["batch_size"][2]
        global_config.load_size = network_config["load_size"][2]
        print("Using TITAN Workstation configuration. Workers: ", global_config.num_workers)

    elif (global_config.server_config == 5): #Titan RTX 2070
        global_config.num_workers = 4
        global_config.a_path_train = "/home/gamelab/Documents/SuperRes Dataset/{dataset_version}{low_path}"
        global_config.b_path_train = "/home/gamelab/Documents/SuperRes Dataset/{dataset_version}{high_path}"
        global_config.a_path_test = "/home/gamelab/Documents/SuperRes Dataset/{dataset_version}{low_path}"
        global_config.b_path_test = "/home/gamelab/Documents/SuperRes Dataset/{dataset_version}{high_path}"
        global_config.batch_size = network_config["batch_size"][3]
        global_config.load_size = network_config["load_size"][3]
        print("Using G411-RTX3060 Workstation configuration. Workers: ", global_config.num_workers)

    elif (global_config.server_config == 6): #G411 RTX 3060
        global_config.num_workers = 6
        global_config.a_path_train = "C:/Datasets/SuperRes Dataset/{dataset_version}{low_path}"
        global_config.b_path_train = "C:/Datasets/SuperRes Dataset/{dataset_version}{high_path}"
        global_config.a_path_test = "C:/Datasets/SuperRes Dataset/{dataset_version}{low_path}"
        global_config.b_path_test = "C:/Datasets/SuperRes Dataset/{dataset_version}{high_path}"
        global_config.burst_sr_lr_path = "C:/Datasets/SuperRes Dataset/v02_burstsr/val/*/samsung_00/im_rgb_*.png"
        global_config.burst_sr_hr_path = "C:/Datasets/SuperRes Dataset/v02_burstsr/val/*/canon/im_rgb_*.png"
        global_config.div2k_lr_path = "C:/Datasets/SuperRes Dataset/div2k/lr/*.png"
        global_config.div2k_hr_path = "C:/Datasets/SuperRes Dataset/div2k/bicubic_x4/*.png"
        global_config.batch_size = network_config["batch_size"][2]
        global_config.load_size = network_config["load_size"][2]
        print("Using G411-RTX3060 Workstation configuration. Workers: ", global_config.num_workers)

    global_config.a_path_train = global_config.a_path_train.format(dataset_version=dataset_version, low_path=low_path)
    global_config.b_path_train = global_config.b_path_train.format(dataset_version=dataset_version, high_path=high_path)
    global_config.a_path_test = global_config.a_path_test.format(dataset_version=dataset_version, low_path=low_path)
    global_config.b_path_test = global_config.b_path_test.format(dataset_version=dataset_version, high_path=high_path)

def main(argv):
    (opts, args) = parser.parse_args(argv)
    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    manualSeed = 0
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

    global_config.sr_network_version, global_config.hyper_iteration, global_config.loss_iteration = utils_script.parse_string(opts.network_version)

    yaml_config = "./hyperparam_tables/{network_version}.yaml"
    yaml_config = yaml_config.format(network_version=global_config.sr_network_version)
    hyperparam_path = "./hyperparam_tables/common_hyper.yaml"
    loss_weights_path = "./hyperparam_tables/common_weights.yaml"
    with open(yaml_config) as f, open(hyperparam_path) as h, open(loss_weights_path) as l:
        ConfigHolder.initialize(yaml.load(f, SafeLoader), yaml.load(h, SafeLoader), yaml.load(l, SafeLoader))

    update_config(opts)
    print(opts)
    print("=====================BEGIN============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (global_config.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    network_config = ConfigHolder.getInstance().get_network_config()
    hyperparams_table = ConfigHolder.getInstance().get_all_hyperparams()
    loss_config = ConfigHolder.getInstance().get_loss_weights()
    loss_iteration = global_config.loss_iteration

    loss_config_table = loss_config["loss_weights"][loss_iteration]
    print("Network version:", opts.network_version, ". Hyper parameters: ", hyperparams_table, " Loss weights: ", loss_config_table, " Learning rates: ", hyperparams_table["g_lr"], hyperparams_table["d_lr"])

    a_path_train = global_config.a_path_train
    b_path_train = global_config.b_path_train
    a_path_test = global_config.a_path_test
    b_path_test = global_config.b_path_test
    burst_sr_lr_path = global_config.burst_sr_lr_path
    burst_sr_hr_path = global_config.burst_sr_hr_path
    div2k_lr_path = global_config.div2k_lr_path
    div2k_hr_path = global_config.div2k_hr_path

    print("Dataset path A: ", a_path_train, a_path_test)
    print("Dataset path B: ", b_path_train, b_path_test)

    plot_utils.VisdomReporter.initialize()

    test_loader_a, test_count = dataset_loader.load_test_img2img_dataset(a_path_test, b_path_test)
    test_loader_b, test_count = dataset_loader.load_test_img2img_dataset(burst_sr_lr_path, burst_sr_hr_path)
    test_loader_div2k, test_count = dataset_loader.load_base_img2img_dataset(div2k_lr_path, div2k_hr_path)

    img2img_t = paired_tester.PairedTester(device)
    start_epoch = global_config.last_epoch_st
    print("---------------------------------------------------------------------------")
    print("Started synth test loop for mode: ", ConfigHolder.getInstance().get_sr_version_name(), " Set start epoch: ", start_epoch)
    print("---------------------------------------------------------------------------")

    # compute total progress
    steps = global_config.test_size
    needed_progress = int(test_count / steps) + 1
    current_progress = 0
    pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
    pbar.update(current_progress)

    with torch.no_grad():
        for i, (file_name, a_batch, b_batch) in enumerate(test_loader_a, 0):
            a_batch = a_batch.to(device)
            b_batch = b_batch.to(device)

            input_map = {"file_name": file_name, "img_a" : a_batch, "img_b" : b_batch}
            img2img_t.measure_and_store(input_map)
            img2img_t.save_images(input_map)
            pbar.update(1)

            if((i + 1) % 4 == 0):
                break

        if (global_config.plot_enabled == 1):
            img2img_t.visualize_results(input_map, "Train Dataset")
        img2img_t.report_metrics("Train Dataset")

        for i, (file_name, a_batch, b_batch) in enumerate(test_loader_b, 0):
            a_batch = a_batch.to(device)
            b_batch = b_batch.to(device)

            input_map = {"file_name": file_name, "img_a": a_batch, "img_b": b_batch}
            img2img_t.measure_and_store(input_map)
            img2img_t.save_images(input_map)
            pbar.update(1)

            if ((i + 1) % 4 == 0):
                break

        if (global_config.plot_enabled == 1):
            img2img_t.visualize_results(input_map, "Test - BurstSR")
        img2img_t.report_metrics("Test - BurstSR")

        pbar.close()

        for i, (file_name, a_batch, b_batch) in enumerate(test_loader_div2k, 0):
            a_batch = a_batch.to(device)
            b_batch = b_batch.to(device)

            # print("Shapes: ", np.shape(a_batch), np.shape(b_batch))
            input_map = {"file_name": file_name, "img_a": a_batch, "img_b": b_batch}
            img2img_t.measure_and_store(input_map)
            img2img_t.save_images(input_map)
            pbar.update(1)

            if ((i + 1) % 4 == 0):
                break

        if (global_config.plot_enabled == 1):
            img2img_t.visualize_results(input_map, "Test - Div2k")
        img2img_t.report_metrics("Test - Div2k")

        pbar.close()


if __name__ == "__main__":
    main(sys.argv)