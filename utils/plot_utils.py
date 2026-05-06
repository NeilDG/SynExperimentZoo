# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:02:01 2020

@author: delgallegon
"""
import torch
from matplotlib.lines import Line2D
import time
import functools

import global_config
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import visdom

from loaders import segmentation_datasets
from loaders.segmentation_datasets import labels_to_mask

SALIKSIK_SERVER = "192.168.134.223" #IMPORTmsANT: No HTTP

class VisdomReporter:
    _sharedInstance = None

    @staticmethod
    def initialize():
        VisdomReporter._sharedInstance = VisdomReporter()

    @staticmethod
    def getInstance():
        return VisdomReporter._sharedInstance

    def __init__(self):
        if(global_config.plot_enabled == 0):
            self.vis = None
        elif(global_config.server_config == -99):
            self.vis = visdom.Visdom(SALIKSIK_SERVER, use_incoming_socket=False, port=8097) #TODO: Note that this is set to TRUE for observation.
        elif(global_config.server_config == 1):
            self.vis = None
        # elif(global_config.plot_enabled == 0):
        #     self.vis = None
        else:
            self.vis= visdom.Visdom()
        
        self.image_windows = {}
        self.loss_windows = {}
        self.text_windows = {}

    def log_profile(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            duration = time.time() - start_time
            # Only print if duration is significant (> 0.1s)
            if duration > 0.1:
                print(f"[Profiler] {func.__name__} took {duration:.4f}s")
            return result
        return wrapper
    
    @log_profile
    def plot_image(self, img_tensor, caption, normalize = True):
        if(global_config.plot_enabled == 0):
            return

        img_group = vutils.make_grid(img_tensor[:16], nrow = 8, padding=2, normalize=normalize).cpu()
        if hash(caption) not in self.image_windows:
            self.image_windows[hash(caption)] = self.vis.images(img_group, opts = dict(caption = caption))
        else:
            self.vis.images(img_group, win = self.image_windows[hash(caption)], opts = dict(caption = caption))

    def plot_cmap(self, mask, caption):
        if(global_config.plot_enabled == 0):
            return

        mask_label = mask[:16]
        all_mask_img = []
        for mask in mask_label:
            mask_img = labels_to_mask(mask)
            all_mask_img.append(mask_img)

        all_mask_img = torch.stack(all_mask_img).float()
        # print("All mask image shape: ", all_mask_img.shape, "Min: ", all_mask_img.min(), " Max: ", all_mask_img.max())

        self.plot_image(all_mask_img, caption, normalize=False)

    def plot_text(self, text):
        if(hash(text) not in self.text_windows):
            self.text_windows[hash(text)] = self.vis.text(text, opts = dict(caption = text))
        else:
            self.vis.text(text, win = self.text_windows[hash(text)], opts = dict(caption = text))

    def plot_grad_flow(self, named_parameters, caption):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="r")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="g")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="r", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="g", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

        if hash(caption) not in self.loss_windows:
            self.loss_windows[hash(caption)] = self.vis.matplot(plt, opts = dict(caption = caption))
        else:
            self.vis.matplot(plt, win = self.loss_windows[hash(caption)], opts = dict(caption = caption))

    @log_profile
    def plot_finegrain_loss(self, loss_key, iteration, losses_dict, caption_dict, label):
        if(global_config.plot_enabled == 0 or self.vis is None):
            return
        
        for key in losses_dict.keys():
            data = np.array(losses_dict[key])
            if len(data) == 0: continue
            
            # Calculate correct X-axis range for the circular buffer
            x = np.arange(iteration - len(data), iteration)
            
            opts = dict(
                caption=caption_dict.get(key, key),
                title=f"{label} - {caption_dict.get(key, key)}",
                ylabel='Loss',
                xlabel='Iteration'
            )
            
            # Use vis.line for native, high-performance plotting
            win_name = f"{label}_{key}"
            if win_name not in self.loss_windows:
                self.loss_windows[win_name] = self.vis.line(X=x, Y=data, opts=opts)
            else:
                self.vis.line(X=x, Y=data, win=self.loss_windows[win_name], opts=opts, update='replace')

    def plot_train_test_loss(self, loss_key, iteration, losses_dict, caption_dict, label):
        if (global_config.plot_enabled == 0 or self.vis is None):
            return

        for key in losses_dict.keys():
            data = np.array(losses_dict[key])
            if len(data) == 0: continue

            x = np.arange(iteration - len(data), iteration)
            opts = dict(
                caption=caption_dict.get(key, key),
                title=f"{label} - {caption_dict.get(key, key)}",
                ylabel='Loss',
                xlabel='Iteration'
            )

            win_name = f"{label}_{key}"
            if win_name not in self.loss_windows:
                self.loss_windows[win_name] = self.vis.line(X=x, Y=data, opts=opts)
            else:
                self.vis.line(X=x, Y=data, win=self.loss_windows[win_name], opts=opts, update='replace')

          
        plt.show()

    # def plot_train_test_loss(self, loss_key, iteration, train_losses, test_losses, train_caption, test_caption):
    #     colors = ['r', 'g', 'black', 'darkorange', 'olive', 'palevioletred', 'rosybrown', 'cyan', 'slategray', 'darkmagenta', 'linen', 'chocolate']
    #
    #     x1 = [i for i in range(iteration, iteration + len(train_losses))]
    #     x2 = [i for i in range(iteration, iteration + len(test_losses))]
    #
    #     plt.plot(x1, train_losses, color=colors[0], label=str(train_caption))
    #     plt.plot(x2, test_losses, color=colors[1], label=str(test_caption))
    #     plt.legend(loc='lower right')
    #
    #     if loss_key not in self.loss_windows:
    #         self.loss_windows[loss_key] = self.vis.matplot(plt, opts=dict(caption="Losses" + " " + str(global_config)))
    #     else:
    #         self.vis.matplot(plt, win=self.loss_windows[loss_key], opts=dict(caption="Losses" + " " + str(global_config)))
    #
    #     plt.show()

    def plot_train_test_loss(self, loss_key, iteration, losses_dict, caption_dict, label):
        if (global_config.plot_enabled == 0):
            return
        colors = ['r', 'g', 'black', 'darkorange', 'olive', 'palevioletred', 'rosybrown', 'cyan', 'slategray', 'darkmagenta', 'linen', 'chocolate']

        x = [i for i in range(iteration, iteration + len(losses_dict["TRAIN_LOSS_KEY"]))]
        loss_keys = list(losses_dict.keys())
        caption_keys = list(caption_dict.keys())

        plt.plot(x, losses_dict[loss_keys[0]], colors[0], label=str(caption_dict[caption_keys[0]]))
        plt.plot(x, losses_dict[loss_keys[1]], colors[1], label=str(caption_dict[caption_keys[1]]))
        plt.legend(loc='lower right')

        if loss_key not in self.loss_windows:
            self.loss_windows[loss_key] = self.vis.matplot(plt, opts = dict(caption = "Losses" + " " + str(label)))
        else:
            self.vis.matplot(plt, win = self.loss_windows[loss_key], opts = dict(caption = "Losses" + " " + str(label)))