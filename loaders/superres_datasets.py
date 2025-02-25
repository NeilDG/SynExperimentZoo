import torch
import cv2
import torchvision
from torch.utils import data
import torchvision.transforms as transforms

import global_config
from config.network_config import ConfigHolder

def normalize(light_angle):
    std = light_angle / 360.0
    min = -1.0
    max = 1.0
    scaled = std * (max - min) + min

    return scaled

class BasePairedImageDataset(data.Dataset):
    def __init__(self, a_list, b_list):
        self.a_list = a_list
        self.b_list = b_list

        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512), antialias=True),
            transforms.ToTensor()
        ])

        self.norm_op = transforms.Normalize((0.5, ), (0.5, ))
        # self.upsample_size = global_config.upsample_size

    def __getitem__(self, idx):
        file_name = self.b_list[idx % len(self.b_list)].split("\\")[-1].split(".")[0]

        a_img = cv2.imread(self.a_list[idx])
        a_img = cv2.cvtColor(a_img, cv2.COLOR_BGR2RGB)
        state = torch.get_rng_state()

        torch.set_rng_state(state)
        b_img = cv2.imread(self.b_list[(idx % len(self.b_list))])
        b_img = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)

        ref_h, ref_w = b_img.shape[:2]
        a_img = cv2.resize(a_img, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)

        a_img = self.initial_op(a_img)
        b_img = self.initial_op(b_img)

        return file_name, a_img, b_img

    def __len__(self):
        return len(self.a_list)

class PairedImageDataset(data.Dataset):
    def __init__(self, a_list, b_list, transform_config):
        self.a_list = a_list
        self.b_list = b_list
        self.transform_config = transform_config

        config_holder = ConfigHolder.getInstance()
        if(config_holder is not None):
            self.augment_mode = config_holder.get_network_attribute("augment_key", "none")
            self.use_tanh = config_holder.get_network_attribute("use_tanh", True)

        if (self.transform_config == 1):
            patch_size = config_holder.get_network_attribute("patch_size", 32)
            transform_list = [
                transforms.ToPILImage(),
                # transforms.Resize(patch_size, antialias=True),
                transforms.RandomCrop(patch_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip()
            ]
            if "random_sharpness_contrast" in self.augment_mode:
                transform_list.append(transforms.RandomAdjustSharpness(1.25))
                transform_list.append(transforms.RandomAutocontrast())
                print("Data augmentation: Added random sharpness and contrast")

            if "random_invert" in self.augment_mode:
                transform_list.append(transforms.RandomInvert())
                print("Data augmentation: Added random invert")

            if "augmix" in self.augment_mode:
                transform_list.append(transforms.AugMix())
                print("Data augmentation: Added augmix")

            transform_list.append(transforms.ToTensor())
            self.initial_op = transforms.Compose(transform_list)
        else:
            self.initial_op = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512, 512), antialias=True),
                transforms.ToTensor()
            ])

        self.norm_op = transforms.Compose([
            transforms.Normalize((0.5, ), (0.5, ))])

    def __getitem__(self, idx):
        file_name = self.b_list[idx % len(self.b_list)].split("\\")[-1].split(".")[0]

        a_img = cv2.imread(self.a_list[idx])
        a_img = cv2.cvtColor(a_img, cv2.COLOR_BGR2RGB)
        b_img = cv2.imread(self.b_list[(idx % len(self.b_list))])
        b_img = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
        # ref_h, ref_w = b_img.shape[:2]
        # a_img = cv2.resize(a_img, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)

        state = torch.get_rng_state()
        a_img = self.initial_op(a_img)
        torch.set_rng_state(state)
        b_img = self.initial_op(b_img)

        if(self.use_tanh):
            a_img = self.norm_op(a_img)
            b_img = self.norm_op(b_img)

        return file_name, a_img, b_img

    def __len__(self):
        return len(self.a_list)

class SingleImageDataset(data.Dataset):
    def __init__(self, a_list, transform_config):
        self.a_list = a_list
        self.transform_config = transform_config

        config_holder = ConfigHolder.getInstance()
        self.augment_mode = config_holder.get_network_attribute("augment_key", "none")
        self.use_tanh = config_holder.get_network_attribute("use_tanh", True)

        if (self.transform_config == 1):
            patch_size = config_holder.get_network_attribute("patch_size", 32)
        else:
            patch_size = 256

        self.patch_size = (patch_size, patch_size)
        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256), antialias=True),
            transforms.ToTensor()
        ])

        self.norm_op = transforms.Normalize((0.5, ), (0.5, ))

    def __getitem__(self, idx):
        file_name = self.a_list[idx].split("/")[-1].split(".")[0]

        a_img = cv2.imread(self.a_list[idx])
        a_img = cv2.cvtColor(a_img, cv2.COLOR_BGR2RGB)
        a_img = self.initial_op(a_img)

        if(self.use_tanh):
            a_img = self.norm_op(a_img)

        return file_name, a_img

    def __len__(self):
        return len(self.a_list)