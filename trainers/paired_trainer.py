# -*- coding: utf-8 -*-
# Paired trainer used for training.
import numpy as np

from config.network_config import ConfigHolder
from losses import common_losses
import global_config
import torch
import torch.amp as amp
import itertools

from model.modules import image_pool
from trainers import early_stopper, abstract_iid_trainer
from utils import plot_utils, tensor_utils


class PairedTrainer:
    def __init__(self, gpu_device):
        self.gpu_device = gpu_device
        self.initialize_train_config()

    def initialize_train_config(self):
        config_holder = ConfigHolder.getInstance()
        network_config = config_holder.get_network_config()
        hyperparam_config = config_holder.get_all_hyperparams()
        self.iteration = global_config.loss_iteration
        self.common_losses = common_losses.LossRepository(self.gpu_device)

        self.D_B_pool = image_pool.ImagePool(50)
        self.fp16_scaler = amp.GradScaler()
        self.visdom_reporter = plot_utils.VisdomReporter.getInstance()

        self.load_size = global_config.load_size
        self.batch_size = global_config.batch_size

        self.stopper_method = early_stopper.EarlyStopper(network_config["min_epochs"], early_stopper.EarlyStopperMethod.L1_TYPE, 1000)
        self.stop_result = False

        self.initialize_dict()
        network_creator = abstract_iid_trainer.NetworkCreator(self.gpu_device)
        self.G_A2B, self.D_B = network_creator.initialize_img2img_network()

        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_A2B.parameters()), lr=hyperparam_config["g_lr"])
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_B.parameters()), lr=hyperparam_config["d_lr"])

        self.NETWORK_VERSION = ConfigHolder.getInstance().get_sr_version_name()
        self.NETWORK_CHECKPATH = 'checkpoint/' + self.NETWORK_VERSION + '.pth'
        self.load_saved_state()

        self.optimizerG.zero_grad()
        self.optimizerD.zero_grad()

    def initialize_dict(self):
        # dictionary keys
        self.G_LOSS_KEY = "g_loss"
        self.G_ADV_LOSS_KEY = "g_adv"
        self.L1_LOSS_KEY = "l1_loss"
        self.COLOR_LOSS_KEY = "color_loss"
        self.TV_LOSS_KEY = "tv_loss"
        self.SSIM_LOSS_KEY = "ssim_loss"
        self.PERCEPTUAL_LOSS_KEY = "perceptual_loss"
        self.BICUBIC_LOSS_KEY = "bicubic_loss"

        self.D_OVERALL_LOSS_KEY = "d_loss"
        self.D_B_LOSS_KEY = "d_b"

        # what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[self.G_LOSS_KEY] = []
        self.losses_dict[self.D_OVERALL_LOSS_KEY] = []
        self.losses_dict[self.L1_LOSS_KEY] = []
        self.losses_dict[self.G_ADV_LOSS_KEY] = []
        self.losses_dict[self.COLOR_LOSS_KEY] = []
        self.losses_dict[self.TV_LOSS_KEY] = []
        self.losses_dict[self.SSIM_LOSS_KEY] = []
        self.losses_dict[self.PERCEPTUAL_LOSS_KEY] = []
        self.losses_dict[self.BICUBIC_LOSS_KEY] = []
        self.losses_dict[self.D_B_LOSS_KEY] = []

        self.caption_dict = {}
        self.caption_dict[self.G_LOSS_KEY] = "Shadow G loss per iteration"
        self.caption_dict[self.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict[self.L1_LOSS_KEY] = "L1 loss per iteration"
        self.caption_dict[self.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict[self.COLOR_LOSS_KEY] = "Color loss per iteration"
        self.caption_dict[self.TV_LOSS_KEY] = "TV loss per iteration"
        self.caption_dict[self.SSIM_LOSS_KEY] = "SSIM loss per iteration"
        self.caption_dict[self.PERCEPTUAL_LOSS_KEY] = "Perceptual loss per iteration"
        self.caption_dict[self.BICUBIC_LOSS_KEY] = "Bicubic loss per iteration"
        self.caption_dict[self.D_B_LOSS_KEY] = "D(B) real loss per iteration"

        # what to store in visdom?
        self.losses_dict_t = {}

        self.TRAIN_LOSS_KEY = "TRAIN_LOSS_KEY"
        self.losses_dict_t[self.TRAIN_LOSS_KEY] = []
        self.TEST_LOSS_KEY = "TEST_LOSS_KEY"
        self.losses_dict_t[self.TEST_LOSS_KEY] = []

        self.caption_dict_t = {}
        self.caption_dict_t[self.TRAIN_LOSS_KEY] = "Train L1 loss per iteration"
        self.caption_dict_t[self.TEST_LOSS_KEY] = "Test L1 loss per iteration"

    def train(self, epoch, iteration, input_map):
        img_a = input_map["img_a"]
        img_b = input_map["img_b"]
        accum_batch_size = self.load_size * iteration

        with amp.autocast(device_type="cuda"):
            self.D_B.train()

            output = self.G_A2B(img_a)
            prediction = self.D_B(output)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_B_real_loss = self.common_losses.compute_adversarial_loss(self.D_B(img_b), real_tensor)
            D_B_fake_loss = self.common_losses.compute_adversarial_loss(self.D_B_pool.query(self.D_B(output.detach())), fake_tensor)

            errD = D_B_real_loss + D_B_fake_loss
            self.fp16_scaler.scale(errD).backward()
            # torch.nn.utils.clip_grad_norm_(self.D_B.parameters(), max_norm=1.0)  # gradient clip

            self.G_A2B.train()
            img_a2b = self.G_A2B(img_a)

            # print("Shapes of pred and target:", np.shape(img_a2b), np.shape(img_b))
            B_likeness_loss = self.common_losses.compute_l1_loss(img_a2b, img_b)
            B_perceptual_loss = self.common_losses.compute_perceptual_loss(img_a2b, img_b)
            B_color_loss = self.common_losses.compute_color_loss(img_a2b, img_b)
            B_tv_loss = self.common_losses.compute_total_variation_loss(img_a2b)
            B_bicubic_loss = self.common_losses.compute_bicubic_loss(img_a2b, img_b)

            prediction = self.D_B(img_a2b)
            real_tensor = torch.ones_like(prediction)
            B_adv_loss = self.common_losses.compute_adversarial_loss(prediction, real_tensor)

            errG = B_likeness_loss + B_perceptual_loss + B_color_loss + B_tv_loss + B_bicubic_loss + B_adv_loss
            self.fp16_scaler.scale(errG).backward()
            # torch.nn.utils.clip_grad_norm_(self.G_A2B.parameters(), max_norm=1.0)  # gradient clip

            if (accum_batch_size % self.batch_size == 0):
                self.fp16_scaler.step(self.optimizerD)
                self.fp16_scaler.step(self.optimizerG)
                self.fp16_scaler.update()

                self.optimizerD.zero_grad()
                self.optimizerG.zero_grad()

                # what to put to losses dict for visdom reporting?
                if (iteration > 10):
                    self.losses_dict[self.G_LOSS_KEY].append(errG.item())
                    self.losses_dict[self.D_OVERALL_LOSS_KEY].append(errD.item())
                    self.losses_dict[self.L1_LOSS_KEY].append(B_likeness_loss.item())
                    self.losses_dict[self.G_ADV_LOSS_KEY].append(B_adv_loss.item())
                    self.losses_dict[self.PERCEPTUAL_LOSS_KEY].append(B_perceptual_loss.item())
                    self.losses_dict[self.COLOR_LOSS_KEY].append(B_color_loss.item())
                    self.losses_dict[self.TV_LOSS_KEY].append(B_tv_loss.item())
                    self.losses_dict[self.BICUBIC_LOSS_KEY].append(B_bicubic_loss.item())
                    self.losses_dict[self.D_B_LOSS_KEY].append(D_B_fake_loss.item() + D_B_real_loss.item())

        a2b = self.test(input_map, "Train")
        self.stopper_method.register_metric(a2b, img_b, epoch)
        self.stop_result = self.stopper_method.test(epoch)

        if (self.stopper_method.has_reset()):
            self.save_states(epoch, iteration, False)

    def test(self, input_map, label="Test"):
        with torch.no_grad():
            img_a = input_map["img_a"]

            self.G_A2B.eval()
            if("Test" in label):
                # img_a2b = tensor_utils.patched_infer(img_a, self.G_A2B, 64, (536, 536, 32, 32))
                img_a2b = self.G_A2B(img_a)
                return img_a2b
            else:
                img_a2b = self.G_A2B(img_a)
                return img_a2b

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss", iteration, self.losses_dict, self.caption_dict, global_config.sr_network_version)

    def visdom_visualize(self, input_map, label = "Train"):
        with torch.no_grad():
            # report to visdom
            network_version = global_config.sr_network_version
            img_a = input_map["img_a"]
            img_b = input_map["img_b"]

            # print("Test shapes: ", np.shape(img_a), np.shape(img_b))
            img_a2b = self.test(input_map, label)

            self.visdom_reporter.plot_image(img_a, str(label) + " Input A Images - " + network_version + str(self.iteration))
            self.visdom_reporter.plot_image(img_a2b, str(label) + " A2B Transfer " + network_version + str(self.iteration))

            self.visdom_reporter.plot_image(img_b, str(label) + " Input B Images - " + network_version + str(self.iteration))

    def save_states(self, epoch, iteration, is_temp: bool):
        save_dict = {'epoch': epoch, 'iteration': iteration, global_config.LAST_METRIC_KEY: self.stopper_method.get_last_metric()}
        netGA2B_state_dict = self.G_A2B.state_dict()
        netDB_state_dict = self.D_B.state_dict()

        save_dict[global_config.GENERATOR_KEY + "A2B"] = netGA2B_state_dict
        save_dict[global_config.DISCRIMINATOR_KEY + "B"] = netDB_state_dict

        if (is_temp):
            torch.save(save_dict, self.NETWORK_CHECKPATH + ".checkpt")
            print("Saved checkpoint state: %s Epoch: %d" % (self.NETWORK_VERSION), (epoch + 1))
        else:
            torch.save(save_dict, self.NETWORK_CHECKPATH)
            print("Saved stable model state: %s Epoch: %d" % (self.NETWORK_VERSION), (epoch + 1))

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
                print("No existing checkpoint file found. Creating new SR network: ", self.NETWORK_CHECKPATH)

        if (checkpoint != None):
            global_config.last_epoch_st = checkpoint["epoch"]
            self.stopper_method.update_last_metric(checkpoint[global_config.LAST_METRIC_KEY])

            self.G_A2B.load_state_dict(checkpoint[global_config.GENERATOR_KEY + "A2B"])
            self.D_B.load_state_dict(checkpoint[global_config.DISCRIMINATOR_KEY + "B"])

            print("Loaded SR network: ", self.NETWORK_CHECKPATH, "Epoch: ", global_config.last_epoch_st)
