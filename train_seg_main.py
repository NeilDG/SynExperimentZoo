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
import yaml
from yaml.loader import SafeLoader
import util_script_main as utils_script

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--network_version', type=str, default="vXX.XX")
parser.add_option('--plot_enabled', type=int, default=1)
parser.add_option('--save_per_iter', type=int, default=500)

def update_config(opts):
    global_config.server_config = opts.server_config
    global_config.plot_enabled = opts.plot_enabled
    global_config.img_to_load = opts.img_to_load
    global_config.cuda_device = opts.cuda_device
    global_config.save_per_iter = opts.save_per_iter
    global_config.test_size = 2

    network_config = ConfigHolder.getInstance().get_network_config()

    if(global_config.server_config == 0): #RTX 4060Ti PC
        global_config.num_workers = 8
        global_config.batch_size = network_config["batch_size"][1]
        global_config.load_size = network_config["load_size"][1]
        print("Using G411-RTX4060Ti configuration. ", global_config, network_config)

    elif(global_config.server_config == 1): #CCS Cloud
        global_config.num_workers = 12
        global_config.seg_path_root_train = "/home/jupyter-neil.delgallego/Segmentation Dataset/VOC/"
        global_config.batch_size = network_config["batch_size"][1]
        global_config.load_size = network_config["load_size"][1]
        print("Using CCS configuration.", global_config, network_config)

    elif(global_config.server_config == 2): #RTX 2080Ti
        global_config.num_workers = 6
        global_config.batch_size = network_config["batch_size"][2]
        global_config.load_size = network_config["load_size"][2]
        print("Using RTX 2080Ti configuration.", global_config, network_config)

    elif(global_config.server_config == 3): #RTX 3090 PC
        global_config.num_workers = 24
        global_config.seg_path_root_train = "X:/Segmentation Dataset/VOC/"
        global_config.batch_size = network_config["batch_size"][0]
        global_config.load_size = network_config["load_size"][0]
        print("Using RTX 3090 configuration. ", global_config, network_config)

    elif (global_config.server_config == 4):  #Titan RTX 3060
        global_config.num_workers = 6
        global_config.batch_size = network_config["batch_size"][2]
        global_config.load_size = network_config["load_size"][2]
        print("Using TITAN Workstation configuration. ", global_config, network_config)

    elif (global_config.server_config == 5): #Titan RTX 2070
        global_config.num_workers = 6
        global_config.batch_size = network_config["batch_size"][3]
        global_config.load_size = network_config["load_size"][3]
        print("Using G411-RTX3060 Workstation configuration. ", global_config, network_config)

    elif (global_config.server_config == 6): #G411 RTX 3060
        global_config.num_workers = 8
        global_config.batch_size = network_config["batch_size"][2]
        global_config.load_size = network_config["load_size"][2]
        print("Using G411-RTX3060 Workstation configuration. ", global_config, network_config)

    elif (global_config.server_config == 7): #RTX 3060 Laguna PCs
        global_config.num_workers = 6
        global_config.batch_size = network_config["batch_size"][2]
        global_config.load_size = network_config["load_size"][2]
        print("Using G411-RTX3060 Workstation configuration. ", global_config, network_config)

    elif (global_config.server_config == 8): #COARE
        global_config.num_workers = 6
        global_config.batch_size = network_config["batch_size"][0]
        global_config.load_size = network_config["load_size"][0]
        print("Using DOST-COARE Workstation configuration. ", global_config, network_config)

def main(argv):
    (opts, args) = parser.parse_args(argv)
    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    manualSeed = 0
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

    global_config.sr_network_version, global_config.hyper_iteration, global_config.loss_iteration = utils_script.parse_string(opts.network_version)

    yaml_config = "./hyperparam_tables/seg/{network_version}.yaml"
    yaml_config = yaml_config.format(network_version=global_config.sr_network_version)
    hyperparam_path = "./hyperparam_tables/seg/common_hyper.yaml"
    loss_weights_path = "./hyperparam_tables/seg/common_weights.yaml"
    with open(yaml_config) as f, open(hyperparam_path) as h, open(loss_weights_path) as l:
        ConfigHolder.initialize(yaml.load(f, SafeLoader), yaml.load(h, SafeLoader), yaml.load(l, SafeLoader))

    update_config(opts)
    print(opts)
    print("=====================BEGIN============================")
    print("Server config? %d GPU Count: %d" % (global_config.server_config, torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    network_config = ConfigHolder.getInstance().get_network_config()
    hyperparams_table = ConfigHolder.getInstance().get_all_hyperparams()
    loss_config = ConfigHolder.getInstance().get_loss_weights()
    loss_iteration = global_config.loss_iteration

    loss_config_table = loss_config["loss_weights"][loss_iteration]
    print("Network version:", opts.network_version, ". Hyper parameters: ", hyperparams_table, " Loss weights: ", loss_config_table, " Learning rates: ", hyperparams_table["g_lr"], hyperparams_table["d_lr"])

    plot_utils.VisdomReporter.initialize()

    train_loader, train_count = dataset_loader.load_voc_dataset()
    iteration = 0
    start_epoch = global_config.last_epoch_st
    print("---------------------------------------------------------------------------")
    print("Started Training loop for mode: synthseg", " Set start epoch: ", start_epoch)
    print("---------------------------------------------------------------------------")

    # compute total progress
    load_size = global_config.load_size
    needed_progress = int((network_config["max_epochs"]) * (train_count / load_size))
    current_progress = int(start_epoch * (train_count / load_size))
    pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
    pbar.update(current_progress)


    for epoch in range(start_epoch, network_config["max_epochs"]):
        for i, (img_batch, target_batch) in enumerate(train_loader, 0):
            img_batch = img_batch.to(device)
            target_batch = target_batch.to(device)
            input_map = {"img" : img_batch, "target" : target_batch}

            iteration = iteration + 1
            pbar.update(1)

    pbar.close()

if __name__ == "__main__":
    main(sys.argv)