import glob
import random
from pathlib import Path

import kornia
import numpy as np
import torch
import torchvision.utils
from torch.utils import data
import cv2
import torchvision.transforms as transforms
from tqdm import tqdm

import global_config
from config.network_config import ConfigHolder
from loaders import image_datasets
import os

def load_single_test_dataset(path_a, opts):
    print("Dataset path: ", path_a)
    a_list = glob.glob(path_a)
    random.shuffle(a_list)
    if (opts.img_to_load > 0):
        a_list = a_list[0: opts.img_to_load]

    # a_list = a_list[100000:328497] #TODO: Temp only

    print("Length of images: %d" % len(a_list))

    data_loader = torch.utils.data.DataLoader(
        image_datasets.SingleImageDataset(a_list, 2),
        batch_size=128,
        num_workers=1,
        shuffle=True
    )

    return data_loader

def load_train_img2img_dataset(a_path, b_path):
    network_config = ConfigHolder.getInstance().get_network_config()
    a_list = glob.glob(a_path)
    b_list = glob.glob(b_path)
    a_list_dup = glob.glob(a_path)
    b_list_dup = glob.glob(b_path)

    if (global_config.img_to_load > 0):
        a_list = a_list[0: global_config.img_to_load]
        b_list = b_list[0: global_config.img_to_load]
        a_list_dup = a_list_dup[0: global_config.img_to_load]
        b_list_dup = b_list_dup[0: global_config.img_to_load]

    for i in range(0, network_config["dataset_repeats"]): #TEMP: formerly 0-1
        a_list += a_list_dup
        b_list += b_list_dup

    temp_list = list(zip(a_list, b_list))
    random.shuffle(temp_list)
    a_list, b_list = zip(*temp_list)

    img_length = len(a_list)
    print("Length of images: %d %d"  % (img_length, len(b_list)))

    num_workers = global_config.num_workers
    data_loader = torch.utils.data.DataLoader(
        image_datasets.PairedImageDataset(a_list, b_list, 1),
        batch_size=global_config.load_size,
        num_workers=num_workers
    )

    return data_loader, img_length

def load_test_img2img_dataset(a_path, b_path):
    a_list = glob.glob(a_path)
    b_list = glob.glob(b_path)

    if (global_config.img_to_load > 0):
        a_list = a_list[0: global_config.img_to_load]
        b_list = b_list[0: global_config.img_to_load]

    temp_list = list(zip(a_list, b_list))
    random.shuffle(temp_list)
    a_list, b_list = zip(*temp_list)

    img_length = len(a_list)
    print("Length of images: %d %d" % (img_length, len(b_list)))

    data_loader = torch.utils.data.DataLoader(
        image_datasets.PairedImageDataset(a_list, b_list, 2),
        batch_size=global_config.test_size,
        num_workers=1
    )

    return data_loader, img_length

def load_singleimg_dataset(a_path):
    a_list = glob.glob(a_path)

    if (global_config.img_to_load > 0):
        a_list = a_list[0: global_config.img_to_load]

    random.shuffle(a_list)

    img_length = len(a_list)
    print("Length of images: %d" % (img_length))

    data_loader = torch.utils.data.DataLoader(
        image_datasets.SingleImageDataset(a_list, 1),
        batch_size=global_config.test_size,
        num_workers=4
    )

    return data_loader, img_length

