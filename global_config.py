# -*- coding: utf-8 -*-
import os

PROJECT_PATH = ""
DATASET_PLACES_PATH = "X:/Places Dataset/"
TEST_IMAGE_SIZE = (256, 256)

#========================================================================#
OPTIMIZER_KEY = "optimizer"
GENERATOR_KEY = "generator"
DISCRIMINATOR_KEY = "discriminator"

LAST_METRIC_KEY = "last_metric"

plot_enabled = 1
early_stop_threshold = 500
disable_progress_bar = False

server_config = -1
num_workers = -1

a_path_base = ""
b_path_base = ""
a_path_train = "X:/SuperRes Dataset/{dataset_version}/low/train_patches/*.jpg"
b_path_train = "X:/SuperRes Dataset/{dataset_version}/high/train_patches/*.jpg"
a_path_test = "X:/SuperRes Dataset/{dataset_version}/low/test_images/*.jpg"
b_path_test = "X:/SuperRes Dataset/{dataset_version}/high/test_images/*.jpg"

seg_path_rgb_path_train = ""
seg_path_mask_path_train = ""
seg_path_rgb_path_test = ""
seg_path_mask_path_test = ""

burst_sr_lr_path = ""
burst_sr_hr_path = ""
div2k_lr_path = ""
div2k_hr_path = ""

sr_network_version = "VXX.XX"
hyper_iteration = -1
loss_iteration = -1

loaded_network_config = None
sm_network_config = None
ns_network_config = None

img_to_load = -1
load_size = -1
batch_size = -1
test_size = -1
train_mode = "all"
last_epoch_sm = 0
last_epoch_ns = 0
last_epoch_st = 0
last_iteration_ns = 0
dataset_target = ""
cuda_device = ""
save_images = 0
save_every_epoch = 5
save_per_iter = 50
epoch_to_load = 0
load_per_epoch = False
load_per_sample = False
load_best = False


