# -*- coding: utf-8 -*-
# Paired trainer used for training.
from config.network_config import ConfigHolder
from losses import common_losses
import global_config
import torch
import torch.cuda.amp as amp
import itertools

from model.modules import image_pool
from trainers import early_stopper, abstract_iid_trainer
from utils import plot_utils



class PairedTrainer:
    def __init__(self, gpu_device):
        self.gpu_device = gpu_device
        self.initialize_train_config()

    def initialize_train_config(self):
        config_holder = ConfigHolder.getInstance()
        network_config = config_holder.get_network_config()
        self.iteration = global_config.sr_iteration
        self.common_losses = common_losses.LossRepository(self.gpu_device, self.iteration)

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

        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_A2B.parameters()), lr=network_config["g_lr"])
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_B.parameters()), lr=network_config["d_lr"])
        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, patience=100000 / self.batch_size, threshold=0.00005)
        self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, patience=100000 / self.batch_size, threshold=0.00005)

        self.NETWORK_VERSION = ConfigHolder.getInstance().get_sr_version_name()
        self.NETWORK_CHECKPATH = 'checkpoint/' + self.NETWORK_VERSION + '.pt'
        self.load_saved_state()

    def initialize_dict(self):
        # dictionary keys
        self.G_LOSS_KEY = "g_loss"
        self.G_ADV_LOSS_KEY = "g_adv"
        self.LIKENESS_LOSS_KEY = "likeness"
        self.RMSE_LOSS_KEY = "rmse_loss"
        self.SSIM_LOSS_KEY = "ssim_loss"

        self.D_OVERALL_LOSS_KEY = "d_loss"
        self.D_B_LOSS_KEY = "d_b"

        # what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[self.G_LOSS_KEY] = []
        self.losses_dict[self.D_OVERALL_LOSS_KEY] = []
        self.losses_dict[self.LIKENESS_LOSS_KEY] = []
        self.losses_dict[self.G_ADV_LOSS_KEY] = []
        self.losses_dict[self.RMSE_LOSS_KEY] = []
        self.losses_dict[self.SSIM_LOSS_KEY] = []
        self.losses_dict[self.D_B_LOSS_KEY] = []

        self.caption_dict = {}
        self.caption_dict[self.G_LOSS_KEY] = "Shadow G loss per iteration"
        self.caption_dict[self.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict[self.LIKENESS_LOSS_KEY] = "L1 loss per iteration"
        self.caption_dict[self.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict[self.RMSE_LOSS_KEY] = "RMSE loss per iteration"
        self.caption_dict[self.SSIM_LOSS_KEY] = "SSIM loss per iteration"
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

        with amp.autocast():
            self.optimizerD.zero_grad()
            self.D_B.train()

            output = self.G_A2B(img_a)
            prediction = self.D_B(output)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)

            D_B_real_loss = self.common_losses.compute_adversarial_loss(self.D_B(img_b), real_tensor)
            D_B_fake_loss = self.common_losses.compute_adversarial_loss(self.D_B_pool.query(self.D_B(output.detach())), fake_tensor)

            errD = D_B_real_loss + D_B_fake_loss
            self.fp16_scaler.scale(errD).backward()
            if (accum_batch_size % self.batch_size == 0):
                self.schedulerD.step(errD)
                self.fp16_scaler.step(self.optimizerD)

            self.optimizerG.zero_grad()
            self.G_A2B.train()

            img_a2b = self.G_A2B(img_a)

            B_likeness_loss = self.common_losses.compute_l1_loss(img_a2b, img_b)
            prediction = self.D_B(img_a2b)
            real_tensor = torch.ones_like(prediction)
            B_adv_loss = self.common_losses.compute_adversarial_loss(prediction, real_tensor)

            errG = B_likeness_loss  + B_adv_loss
            self.fp16_scaler.scale(errG).backward()

            if (accum_batch_size % self.batch_size == 0):
                self.schedulerG.step(errG)
                self.fp16_scaler.step(self.optimizerG)
                self.fp16_scaler.update()

                # what to put to losses dict for visdom reporting?
                if (iteration > 10):
                    self.losses_dict[self.G_LOSS_KEY].append(errG.item())
                    self.losses_dict[self.D_OVERALL_LOSS_KEY].append(errD.item())
                    self.losses_dict[self.LIKENESS_LOSS_KEY].append(B_likeness_loss.item())
                    self.losses_dict[self.G_ADV_LOSS_KEY].append(B_adv_loss.item())
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

            img_a2b = self.test(input_map)

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
            print("Saved checkpoint state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))
        else:
            torch.save(save_dict, self.NETWORK_CHECKPATH)
            print("Saved stable model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

    def load_saved_state(self):
        try:
            checkpoint = torch.load(self.NETWORK_CHECKPATH, map_location=self.gpu_device)
        except:
            # check if a .checkpt is available, load it
            try:
                checkpt_name = 'checkpoint/' + self.NETWORK_VERSION + ".pt.checkpt"
                checkpoint = torch.load(checkpt_name, map_location=self.gpu_device)
            except:
                checkpoint = None
                print("No existing checkpoint file found. Creating new SR network: ", self.NETWORK_CHECKPATH)

        if (checkpoint != None):
            global_config.last_epoch_st = checkpoint["epoch"]
            self.stopper_method.update_last_metric(checkpoint[global_config.LAST_METRIC_KEY])

            self.G_A2B.load_state_dict(checkpoint[global_config.GENERATOR_KEY + "A2B"])
            self.D_B.load_state_dict(checkpoint[global_config.DISCRIMINATOR_KEY + "B"])

            print("Loaded SR network: ", self.NETWORK_CHECKPATH, "Epoch: ", global_config.last_epoch_st)
