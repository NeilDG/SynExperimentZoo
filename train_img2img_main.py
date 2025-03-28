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
    dataset_a_train = network_config["dataset_a_train"]
    dataset_b_train = network_config["dataset_b_train"]
    dataset_a_test = network_config["dataset_a_test"]
    dataset_b_test = network_config["dataset_b_test"]

    if(global_config.server_config == 0): #RTX 4060Ti PC
        global_config.num_workers = 8
        global_config.a_path_train = "C:/Datasets/{dataset_version}"
        global_config.b_path_train = "C:/Datasets/{dataset_version}"
        global_config.a_path_test = "C:/Datasets/{dataset_version}"
        global_config.b_path_test = "C:/Datasets/{dataset_version}"
        global_config.batch_size = network_config["batch_size"][1]
        global_config.load_size = network_config["load_size"][1]
        print("Using G411-RTX4060Ti configuration. ", global_config, network_config)

    elif(global_config.server_config == 1): #CCS Cloud
        global_config.num_workers = 12
        global_config.a_path_train = "/home/jupyter-neil.delgallego/SuperRes Dataset/{dataset_version}/low/train_patches/*.jpg"
        global_config.b_path_train = "/home/jupyter-neil.delgallego/SuperRes Dataset/{dataset_version}/high/train_patches/*.jpg"
        global_config.a_path_test = "/home/jupyter-neil.delgallego/SuperRes Dataset/{dataset_version}/low/test_images/*.jpg"
        global_config.b_path_test = "/home/jupyter-neil.delgallego/SuperRes Dataset/{dataset_version}/high/test_images/*.jpg"
        global_config.batch_size = network_config["batch_size"][1]
        global_config.load_size = network_config["load_size"][1]
        print("Using CCS configuration.", global_config, network_config)

    elif(global_config.server_config == 2): #RTX 2080Ti
        global_config.num_workers = 6
        global_config.a_path_train = "X:/{dataset_version}/low/train_patches/*.jpg"
        global_config.b_path_train = "X:/SuperRes Dataset/{dataset_version}/high/train_patches/*.jpg"
        global_config.a_path_test = "X:/SuperRes Dataset/{dataset_version}/low/test_images/*.jpg"
        global_config.b_path_test = "X:/SuperRes Dataset/{dataset_version}/high/test_images/*.jpg"
        global_config.batch_size = network_config["batch_size"][2]
        global_config.load_size = network_config["load_size"][2]
        print("Using RTX 2080Ti configuration.", global_config, network_config)

    elif(global_config.server_config == 3): #RTX 3090 PC
        global_config.num_workers = 12
        global_config.a_path_train = "X:/{dataset_version}"
        global_config.b_path_train = "X:/{dataset_version}"
        global_config.a_path_test = "X:/{dataset_version}"
        global_config.b_path_test = "X:/{dataset_version}"
        global_config.batch_size = network_config["batch_size"][0]
        global_config.load_size = network_config["load_size"][0]
        print("Using RTX 3090 configuration. ", global_config, network_config)

    elif (global_config.server_config == 4):  #Titan RTX 3060
        global_config.num_workers = 4
        global_config.a_path_train = "/home/gamelab/Documents/SuperRes Dataset/{dataset_version}{low_path}"
        global_config.b_path_train = "/home/gamelab/Documents/SuperRes Dataset/{dataset_version}{high_path}"
        global_config.a_path_test = "/home/gamelab/Documents/SuperRes Dataset/{dataset_version}{low_path}"
        global_config.b_path_test = "/home/gamelab/Documents/SuperRes Dataset/{dataset_version}{high_path}"
        global_config.batch_size = network_config["batch_size"][2]
        global_config.load_size = network_config["load_size"][2]
        print("Using TITAN Workstation configuration. ", global_config, network_config)

    elif (global_config.server_config == 5): #Titan RTX 2070
        global_config.num_workers = 4
        global_config.a_path_train = "/home/gamelab/Documents/SuperRes Dataset/{dataset_version}{low_path}"
        global_config.b_path_train = "/home/gamelab/Documents/SuperRes Dataset/{dataset_version}{high_path}"
        global_config.a_path_test = "/home/gamelab/Documents/SuperRes Dataset/{dataset_version}{low_path}"
        global_config.b_path_test = "/home/gamelab/Documents/SuperRes Dataset/{dataset_version}{high_path}"
        global_config.batch_size = network_config["batch_size"][3]
        global_config.load_size = network_config["load_size"][3]
        print("Using G411-RTX3060 Workstation configuration. ", global_config, network_config)

    elif (global_config.server_config == 6): #G411 RTX 3060
        global_config.num_workers = 8
        global_config.a_path_train = "C:/Datasets/{dataset_version}"
        global_config.b_path_train = "C:/Datasets/{dataset_version}"
        global_config.a_path_test = "C:/Datasets/{dataset_version}"
        global_config.b_path_test = "C:/Datasets/{dataset_version}"
        global_config.batch_size = network_config["batch_size"][2]
        global_config.load_size = network_config["load_size"][2]
        print("Using G411-RTX3060 Workstation configuration. ", global_config, network_config)

    elif (global_config.server_config == 7): #RTX 3060 Laguna PCs
        global_config.num_workers = 6
        global_config.a_path_train = "D:/Datasets/{dataset_version}"
        global_config.b_path_train = "D:/Datasets/{dataset_version}"
        global_config.a_path_test = "D:/Datasets/{dataset_version}"
        global_config.b_path_test = "D:/Datasets/{dataset_version}"
        global_config.batch_size = network_config["batch_size"][2]
        global_config.load_size = network_config["load_size"][2]
        print("Using RL202-RTX3060 Workstation configuration. ", global_config, network_config)

    elif (global_config.server_config == 8): #COARE
        global_config.num_workers = 6
        global_config.disable_progress_bar = True
        global_config.a_path_train = "/scratch3/neil.delgallego/SuperRes Dataset/{dataset_version}{low_path}"
        global_config.b_path_train = "/scratch3/neil.delgallego/SuperRes Dataset/{dataset_version}{high_path}"
        global_config.a_path_test = "/scratch3/neil.delgallego/SuperRes Dataset/{dataset_version}{low_path}"
        global_config.b_path_test = "/scratch3/neil.delgallego/SuperRes Dataset/{dataset_version}{high_path}"
        global_config.batch_size = network_config["batch_size"][0]
        global_config.load_size = network_config["load_size"][0]
        print("Using DOST-COARE Workstation configuration. ", global_config, network_config)

    global_config.a_path_train = global_config.a_path_train.format(dataset_version=dataset_a_train)
    global_config.a_path_test = global_config.a_path_test.format(dataset_version=dataset_a_test)

    global_config.b_path_train = global_config.b_path_train.format(dataset_version=dataset_b_train)
    global_config.b_path_test = global_config.b_path_test.format(dataset_version=dataset_b_test)

def main(argv):
    (opts, args) = parser.parse_args(argv)
    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    manualSeed = 0
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

    global_config.sr_network_version, global_config.hyper_iteration, global_config.loss_iteration = utils_script.parse_string(opts.network_version)

    yaml_config = "./hyperparam_tables/img2img/{network_version}.yaml"
    yaml_config = yaml_config.format(network_version=global_config.sr_network_version)
    hyperparam_path = "./hyperparam_tables/img2img/common_hyper.yaml"
    loss_weights_path = "./hyperparam_tables/img2img/common_weights.yaml"
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

    a_path_train = global_config.a_path_train
    b_path_train = global_config.b_path_train
    a_path_test = global_config.a_path_test
    b_path_test = global_config.b_path_test

    print("Dataset path A: ", a_path_train, a_path_test)
    print("Dataset path B: ", b_path_train, b_path_test)

    plot_utils.VisdomReporter.initialize()

    train_loader, train_count = dataset_loader.load_train_img2img_dataset(a_path_train, b_path_train)
    test_loader, test_count = dataset_loader.load_test_img2img_dataset(a_path_test, b_path_test)
    img2img_t = img2imgtrainer.Img2ImgTrainer(device)

    iteration = 0
    start_epoch = global_config.last_epoch_st
    print("---------------------------------------------------------------------------")
    print("Started Training loop for mode: synth2real", " Set start epoch: ", start_epoch)
    print("---------------------------------------------------------------------------")

    # compute total progress
    load_size = global_config.load_size
    needed_progress = int((network_config["max_epochs"]) * (train_count / load_size))
    current_progress = int(start_epoch * (train_count / load_size))
    pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
    pbar.update(current_progress)

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
                    a_test_batch = a_test_batch.to(device)
                    b_test_batch = b_test_batch.to(device)

                    input_map = {"img_a": a_test_batch, "img_b": b_test_batch}
                    img2img_t.visdom_visualize(input_map, "Test")

    pbar.close()

if __name__ == "__main__":
    main(sys.argv)