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
from segmentation_models_pytorch.encoders import get_preprocessing_fn

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

        self.backbone = "mobilenet_v2"
        self.num_classes = 4
        if(network_config == 1):
            self.model = smp.Unet(encoder_name=self.backbone, encoder_weights='imagenet', classes=self.num_classes)
        else:
            self.model = smp.PSPNet(encoder_name=self.backbone, encoder_weights='imagenet', classes=self.num_classes)

        # Loss function for multi-class segmentation
        self.dice_loss = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

        self.optimizerG = torch.optim.Adam(itertools.chain(self.model.parameters()), lr=network_config["g_lr"], weight_decay=network_config["weight_decay"])
        self.model.to(self.gpu_device)

    def initialize_dict(self):
        # what to store in visdom?
        self.losses_dict = {}

    def update_penalties(self):
        # what penalties to use for losses?
        self.cycle_weight = 10.0


    def train(self, epoch, iteration, train_map):
        train_img = train_map["train_img"]
        train_mask = train_map["train_mask"]
        preprocess_fn = get_preprocessing_fn(self.backbone, "imagenet")
        train_img = preprocess_fn(train_img)
        train_mask = preprocess_fn(train_mask)

        accum_batch_size = self.load_size * iteration

        self.optimizerG.zero_grad()
        self.model.train()
        prediction = self.model(train_img)

        dice_loss = self.dice_loss(prediction, train_mask)
        dice_loss.backward()

        if (accum_batch_size % self.batch_size == 0):
            self.optimizerG.step()


        # what to put to losses dict for visdom reporting?

    def visdom_report(self, train_a, train_b, test_a, test_b):
        with torch.no_grad():
            # infer
            print()

        # report to visdom

    def load_saved_state(self, iteration, checkpoint, model_key, optimizer_key):
        self.iteration = iteration
        # load model

    def save_states(self, epoch, iteration, path, model_key, optimizer_key):
        print()
        # save model