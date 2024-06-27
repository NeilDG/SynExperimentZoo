import os.path

import kornia.metrics.psnr
import torchvision

from config import network_config
from config.network_config import ConfigHolder
import global_config
import torch
from utils import plot_utils, tensor_utils
import torch.nn as nn
import numpy as np
from trainers import paired_trainer

class PairedTester():
    def __init__(self, gpu_device):
        self.gpu_device = gpu_device
        self.img2img_t = paired_trainer.PairedTrainer(self.gpu_device)
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.ssim_loss = kornia.losses.SSIMLoss(5)

        self.visdom_reporter = plot_utils.VisdomReporter.getInstance()

        self.l1_results = []
        self.mse_results = []
        self.psnr_results = []
        self.ssim_results = []

    def save_images(self, input_map):
        img_a2b = self.img2img_t.test(input_map)  # a2b --> real2synth, b2a --> synth2real
        file_name = input_map["file_name"]
        img_b = input_map["img_b"]

        version_name = network_config.ConfigHolder.getInstance().get_sr_version_name()
        img_path_like = "./reports/" + version_name +  "/high-like/"
        img_path = "./reports/" + version_name + "/high/"

        if not os.path.exists(img_path_like):
            os.makedirs(img_path_like, exist_ok=True)

        if not os.path.exists(img_path):
            os.makedirs(img_path, exist_ok=True)

        for i in range(0, len(img_b)):
            if(file_name is not None):
                impath = img_path_like + file_name[i] + "-high-like.png"
            torchvision.utils.save_image(img_a2b[i], impath, normalize = True)
            # print("Saved image (no shadows) : ", impath)

        for i in range(0, len(img_b)):
            impath = img_path + file_name[i] + ".png"
            torchvision.utils.save_image(img_b[i], impath, normalize = True)
            print("Saved image : ", impath)

    #measures the performance of a given batch and stores it
    def measure_and_store(self, input_map):
        use_tanh = ConfigHolder.getInstance().get_network_attribute("use_tanh", False)
        img_a2b = self.img2img_t.test(input_map) #a2b --> real2synth, b2a --> synth2real
        file_name = input_map["file_name"]
        target = input_map["img_b"]

        if(use_tanh):
            target = tensor_utils.normalize_to_01(target)

        psnr_result = kornia.metrics.psnr(img_a2b, target, torch.max(target).item())
        self.psnr_results.append(psnr_result.item())

        l1_result = self.l1_loss(img_a2b, target).cpu()
        self.l1_results.append(l1_result)

        mse_result = self.mse_loss(img_a2b, target).cpu()
        self.mse_results.append(mse_result)

        ssim_result = self.ssim_loss(img_a2b, target).cpu()
        self.ssim_results.append(ssim_result)

    def visualize_results(self, input_map, dataset_title):
        version_name = network_config.ConfigHolder.getInstance().get_sr_version_name()
        self.img2img_t.visdom_visualize(input_map, "Test - " + version_name + " " + dataset_title)

    def report_metrics(self, dataset_title):
        version_name = network_config.ConfigHolder.getInstance().get_sr_version_name()

        psnr_mean = np.round(np.mean(self.psnr_results), 4)
        self.psnr_results.clear()

        l1_mean = np.round(np.float32(np.mean(self.l1_results)), 4) #ISSUE: ROUND to 4 sometimes cause inf
        self.l1_results.clear()

        mse_mean = np.round(np.mean(self.mse_results), 4)
        self.mse_results.clear()

        ssim_mean = np.round(np.mean(self.ssim_results), 4)
        self.ssim_results.clear()

        last_epoch = global_config.last_epoch_st
        self.visdom_reporter.plot_text(dataset_title + " Results - " + version_name + " Last epoch: " + str(last_epoch) + "<br>"
                                        + "Dataset: " + str(dataset_title) + "<br>"
                                       + "PSNR: " +str(psnr_mean) + "<br>" 
                                       "Abs Rel: " + str(l1_mean) + "<br>"
                                        "Sqr Rel: " + str(mse_mean) + "<br>"
                                       "SSIM: " + str(ssim_mean) + "<br>")
