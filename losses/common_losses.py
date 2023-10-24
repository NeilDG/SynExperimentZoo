import kornia.losses
import torch.nn as nn
import torch
from config.network_config import ConfigHolder
from losses import vgg_loss


#
# Class to contain common losses used for training networks
#
class LossRepository():
    def __init__(self, gpu_device, iteration):
        self.gpu_device = gpu_device
        self.iteration = iteration
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ssim_loss = kornia.losses.SSIMLoss(5)
        self.vgg_loss = vgg_loss.VGGPerceptualLoss()
        self.vgg_loss.to(self.gpu_device)

    def compute_adversarial_loss(self, pred, target):
        config_holder = ConfigHolder.getInstance()
        weight = config_holder.get_hyper_params_weight(self.iteration, "adv_weight")
        use_bce = config_holder.get_hyper_params_weight(self.iteration, "is_bce")

        if (weight > 0.0 and use_bce == 0):
            return self.mse_loss(pred, target) * weight
        elif (weight > 0.0 and use_bce == 1):
            return self.bce_loss(pred, target) * weight
        else:
            return torch.zeros_like(self.l1_loss(pred, target))

    def compute_l1_loss(self, pred, target):
        config_holder = ConfigHolder.getInstance()
        weight = config_holder.get_hyper_params_weight(self.iteration, "l1_weight")
        if (weight > 0.0):
            return self.l1_loss(pred, target) * weight
        else:
            return torch.zeros_like(self.l1_loss(pred, target))

    def compute_color_loss(self, pred, target):
        config_holder = ConfigHolder.getInstance()
        weight = config_holder.get_hyper_params_weight(self.iteration, "color_weight")
        pred_x = kornia.filters.gaussian_blur2d(pred, (5, 5), (3.0, 3.0))
        target_x = kornia.filters.gaussian_blur2d(target, (5, 5), (3.0, 3.0))

        if (weight > 0.0):
            return self.l1_loss(pred_x, target_x) * weight
        else:
            return torch.zeros_like(self.l1_loss(pred_x, target_x))

    def compute_perceptual_loss(self, pred, target):
        config_holder = ConfigHolder.getInstance()
        weight = config_holder.get_hyper_params_weight(self.iteration, "perceptual_weight")

        if (weight > 0.0):
            loss = torch.mean(self.vgg_loss(pred, target)) * weight
            return loss
        else:
            return torch.zeros_like(self.l1_loss(pred, target))

    def compute_total_variation_loss(self, pred):
        config_holder = ConfigHolder.getInstance()
        weight = config_holder.get_hyper_params_weight(self.iteration, "tv_weight")

        if (weight > 0.0):
            return torch.mean(kornia.losses.total_variation(pred))
        else:
            return torch.zeros_like(self.l1_loss(pred, pred))

    def compute_mse_loss(self, pred, target):
        config_holder = ConfigHolder.getInstance()
        weight = config_holder.get_hyper_params_weight(self.iteration, "mse_weight")
        if (weight > 0.0):
            return self.mse_loss(pred, target) * weight
        else:
            return torch.zeros_like(self.l1_loss(pred, target))

    def compute_ssim_loss(self, pred, target):
        config_holder = ConfigHolder.getInstance()
        weight = config_holder.get_hyper_params_weight(self.iteration, "ssim_weight")
        if (weight > 0.0):
            return self.ssim_loss(pred, target) * weight
        else:
            return torch.zeros_like(self.l1_loss(pred, target))