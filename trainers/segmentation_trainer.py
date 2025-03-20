# -*- coding: utf-8 -*-
# Template trainer. Do not use this for actual training.

import os

from config.network_config import ConfigHolder
from model import vanilla_cycle_gan as cg
import global_config
import torch
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.utils as vutils
from utils import plot_utils
import segmentation_models_pytorch as smp
import loaders.segmentation_datasets as segmentation_datasets
import torch.nn.functional as F

class SegmentationTrainer:

    def __init__(self, gpu_device):
        self.gpu_device = gpu_device
        self.iteration = global_config.loss_iteration
        self.visdom_reporter = plot_utils.VisdomReporter()
        self.initialize_dict()

        self.load_size = global_config.load_size
        self.batch_size = global_config.batch_size

        config_holder = ConfigHolder.getInstance()
        network_config = config_holder.get_network_config()
        model_type = network_config["model_type"]
        hyperparam_config = config_holder.get_all_hyperparams()

        self.backbone = "mobilenet_v2"
        self.num_classes = segmentation_datasets.color_to_class_len
        if model_type == 1:
            self.model = smp.Unet(encoder_name=self.backbone, encoder_weights='imagenet', classes=self.num_classes)
        else:
            self.model = smp.PSPNet(encoder_name=self.backbone, encoder_weights='imagenet', classes=self.num_classes)

        # Loss function for multi-class segmentation
        self.dice_loss = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE)

        self.optimizerG = torch.optim.Adam(itertools.chain(self.model.parameters()), lr=hyperparam_config["g_lr"], weight_decay=hyperparam_config["weight_decay"])
        self.model.to(self.gpu_device)
        # self.preprocess_fn = get_preprocessing_fn(self.backbone, "imagenet")

        self.NETWORK_VERSION = ConfigHolder.getInstance().get_sr_version_name()
        self.NETWORK_CHECKPATH = 'checkpoint/' + self.NETWORK_VERSION + '.pth'
        self.load_saved_state()

    def initialize_dict(self):
        # what to store in visdom?
        self.G_LOSS_KEY = "g_loss"
        self.losses_dict = {}
        self.losses_dict[self.G_LOSS_KEY] = []

        self.caption_dict = {}
        self.caption_dict[self.G_LOSS_KEY] = "Dice loss per iteration"

    def train(self, epoch, iteration, train_map):
        train_img = train_map["train_img"]
        train_mask = train_map["train_mask"]

        # train_img = self.preprocess_fn(train_img)
        # train_mask = self.preprocess_fn(train_mask)

        accum_batch_size = self.load_size * iteration

        self.optimizerG.zero_grad()
        self.model.train()
        prediction = self.model(train_img)

        # print("Shapes: ", train_img.shape, prediction.shape, train_mask.shape, " Type: ", prediction.dtype, train_mask.dtype)
        dice_loss = self.dice_loss(prediction, train_mask)
        dice_loss.backward()

        if (accum_batch_size % self.batch_size == 0):
            self.optimizerG.step()

            # what to put to losses dict for visdom reporting?
            if (iteration > 10):
                self.losses_dict[self.G_LOSS_KEY].append(dice_loss.item())


    def test(self, input_map):
        with torch.no_grad():
            img = input_map["img"]

            self.model.eval()
            prediction = self.model(img)
            return prediction

    def visdom_plot(self, iteration):
        style_transfer_version = global_config.sr_network_version
        self.visdom_reporter.plot_finegrain_loss("a2b_loss", iteration, self.losses_dict, self.caption_dict, style_transfer_version)

    def visdom_visualize(self, input_map, label = "Train"):
        with torch.no_grad():
            # report to visdom
            network_version = global_config.sr_network_version
            img = input_map["img"]
            mask = input_map["mask"]
            # mask_rgb = input_map["mask_rgb"]

            prediction = self.test(input_map)
            prediction = F.softmax(prediction, dim=1)
            prediction = prediction.argmax(dim=1)
            # print("Prediction shape: ", prediction.shape, " Mask shape: ", mask.shape)

            self.visdom_reporter.plot_image(img, str(label) + " RGB Images - " + network_version + str(self.iteration))
            self.visdom_reporter.plot_cmap(prediction, str(label) + " RGB->Mask Transfer " + network_version + str(self.iteration))
            self.visdom_reporter.plot_cmap(mask, str(label) + " Mask Images - " + network_version + str(self.iteration))
            # self.visdom_reporter.plot_image(mask_rgb, str(label) + " Mask RGB Images - " + network_version + str(self.iteration), normalize=False)

    def save_states(self, epoch, iteration, is_temp: bool):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        model_state_dict = self.model.state_dict()

        save_dict[global_config.GENERATOR_KEY + "A2B"] = model_state_dict

        if (is_temp):
            torch.save(save_dict, self.NETWORK_CHECKPATH + ".checkpt")
            print("Saved checkpoint state: %s Epoch: %d" % (self.NETWORK_VERSION, (epoch + 1)))
        else:
            torch.save(save_dict, self.NETWORK_CHECKPATH)
            print("Saved stable model state: %s Epoch: %d" % (self.NETWORK_VERSION, (epoch + 1)))

    def load_saved_state(self):
        try:
            checkpoint = torch.load(self.NETWORK_CHECKPATH, map_location=self.gpu_device, weights_only=True)
        except:
            # check if a .checkpt is available, load it
            try:
                checkpt_name = 'checkpoint/' + self.NETWORK_VERSION + ".pth.checkpt"
                checkpoint = torch.load(checkpt_name, map_location=self.gpu_device, weights_only=True)
            except:
                checkpoint = None
                print("No existing checkpoint file found. Creating new seg network: ", self.NETWORK_CHECKPATH)

        if (checkpoint != None):
            global_config.last_epoch_st = checkpoint["epoch"]

            self.model.load_state_dict(checkpoint[global_config.GENERATOR_KEY + "A2B"])
            print("Loaded seg network: ", self.NETWORK_CHECKPATH, "Epoch: ", global_config.last_epoch_st)