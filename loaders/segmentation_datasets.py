import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data

import global_config
from config.network_config import ConfigHolder


# class CustomCityscapesDataset(torchvision.datasets.Cityscapes):
#     def __init__(self, transform_config, split):
#         super().__init__(global_config.seg_path_root_train, split=split, mode='fine', target_type='semantic')
#         self.transform_config = transform_config
#
#         config_holder = ConfigHolder.getInstance()
#         self.augment_mode = config_holder.get_network_attribute("augment_key", "none")
#
#         if self.transform_config == 1:
#             patch_size = config_holder.get_network_attribute("patch_size", 32)
#             transform_list = [
#                 # transforms.ToPILImage(),
#                 transforms.RandomCrop(patch_size),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomVerticalFlip()
#             ]
#             if "random_sharpness_contrast" in self.augment_mode:
#                 transform_list.append(transforms.RandomAdjustSharpness(1.25))
#                 transform_list.append(transforms.RandomAutocontrast())
#                 print("Data augmentation: Added random sharpness and contrast")
#
#             if "random_invert" in self.augment_mode:
#                 transform_list.append(transforms.RandomInvert())
#                 print("Data augmentation: Added random invert")
#
#             if "augmix" in self.augment_mode:
#                 transform_list.append(transforms.AugMix())
#                 print("Data augmentation: Added augmix")
#
#             transform_list.append(transforms.ToTensor())
#             self.rgb_op = transforms.Compose(transform_list)
#         else:
#             self.rgb_op = transforms.Compose([
#                 transforms.ToPILImage(),
#                 # transforms.RandomCrop(128),
#                 transforms.ToTensor()
#             ])
#
#         # self.norm_op = transforms.Normalize((0.5,), (0.5,))
#         self.norm_op = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #From ImageNet
#
#     def __getitem__(self, index):
#         image, target = super().__getitem__(index)
#
#         state = torch.get_rng_state()
#         image = self.rgb_op(image)
#         torch.set_rng_state(state)
#
#         image = self.norm_op(image)
#         target = self.rgb_op(target)
#
#         return image, target


class CityscapesGANDataset_Old(data.Dataset):
    def __init__(self, a_list, transform_config):
        self.a_list = a_list
        self.transform_config = transform_config

        config_holder = ConfigHolder.getInstance()
        self.augment_mode = config_holder.get_network_attribute("augment_key", "none")
        self.use_tanh = config_holder.get_network_attribute("use_tanh", True)

        if self.transform_config == 1:
            patch_size = config_holder.get_network_attribute("patch_size", 32)
            transform_list = [
                transforms.ToPILImage(),
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
                transforms.RandomCrop(128),
                transforms.ToTensor()
            ])

        self.norm_op = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,))])

    def __getitem__(self, idx):
        file_name = self.a_list[idx % len(self.a_list)].split("\\")[-1].split(".")[0]

        a_img = cv2.imread(self.a_list[idx])
        a_img = cv2.cvtColor(a_img, cv2.COLOR_BGR2RGB)

        # Split image into two parts
        height, width, _ = a_img.shape
        mid_point = width // 2  # Assuming vertical split

        left_image = a_img[:, :mid_point, :]
        right_image = a_img[:, mid_point:, :]

        state = torch.get_rng_state()
        left_image = self.initial_op(left_image)
        torch.set_rng_state(state)
        right_image = self.initial_op(right_image)

        if (self.use_tanh):
            left_image = self.norm_op(left_image)
            right_image = self.norm_op(right_image)

        return file_name, left_image, right_image

    def __len__(self):
        return len(self.a_list)

class CityscapesGANDataset(data.Dataset):
    def __init__(self, a_list, b_list, transform_config):
        self.a_list = a_list
        self.b_list = b_list
        self.transform_config = transform_config

        config_holder = ConfigHolder.getInstance()
        self.augment_mode = config_holder.get_network_attribute("augment_key", "none")
        self.use_tanh = config_holder.get_network_attribute("use_tanh", True)

        if self.transform_config == 1:
            patch_size = config_holder.get_network_attribute("patch_size", 32)
            transform_list = [
                transforms.ToPILImage(),
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
                transforms.RandomCrop(64),
                transforms.ToTensor()
            ])

        self.norm_op = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,))])

    def __getitem__(self, idx):
        file_name = self.a_list[idx % len(self.a_list)].split("\\")[-1].split(".")[0]

        a_img = cv2.imread(self.a_list[idx])
        a_img = cv2.cvtColor(a_img, cv2.COLOR_BGR2RGB)

        b_img = cv2.imread(self.b_list[idx])
        b_img = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)

        state = torch.get_rng_state()
        a_img = self.initial_op(a_img)
        torch.set_rng_state(state)
        b_img = self.initial_op(b_img)

        if (self.use_tanh):
            a_img = self.norm_op(a_img)
            b_img = self.norm_op(b_img)

        return file_name, a_img, b_img

    def __len__(self):
        return len(self.a_list)

class CityscapesDataset(data.Dataset):
    def __init__(self, rgb_list, mask_list, transform_config):
        self.rgb_list = rgb_list
        self.mask_list = mask_list
        self.transform_config = transform_config

        config_holder = ConfigHolder.getInstance()
        self.augment_mode = config_holder.get_network_attribute("augment_key", "none")
        self.use_tanh = config_holder.get_network_attribute("use_tanh", True)

        if self.transform_config == 1:
            patch_size = config_holder.get_network_attribute("patch_size", 32)
            transform_list = [
                transforms.ToPILImage(),
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
                transforms.RandomCrop(64),
                transforms.ToTensor()
            ])

        self.norm_op = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,))])

        self.color_to_class = {
            (0, 0, 0): 0, #background
            (0, 0, 142): 1, #vehicle
            (70, 70, 70): 2, #building
            #the rest are "others"
        }

        self.other_class = 3 # "others"

    def __getitem__(self, idx):
        file_name = self.rgb_list[idx % len(self.rgb_list)].split("\\")[-1].split(".")[0]

        rgb_img = cv2.imread(self.rgb_list[idx])
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        mask_img = cv2.imread(self.mask_list[idx])
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)

        state = torch.get_rng_state()
        rgb_img = self.initial_op(rgb_img)
        torch.set_rng_state(state)

        mask_img = self.initial_op(mask_img)
        mask_one_hot = self.mask_to_onehot(mask_img)

        self.print_class_counts(mask_one_hot)

        if self.use_tanh:
            rgb_img = self.norm_op(rgb_img)

        return file_name, rgb_img, mask_one_hot

    def __len__(self):
        return len(self.rgb_list)

    def mask_to_onehot(self, mask):
        """Converts to one-hot, handling "others" class."""
        mask = mask.permute(1, 2, 0).numpy()  # (H, W, 3) and numpy
        one_hot = np.zeros((mask.shape[0], mask.shape[1], len(self.color_to_class) + 1), dtype=np.uint8)  # (H, W, C)

        for color, class_id in self.color_to_class.items():
            color_mask = np.all(mask == np.array(color), axis=-1)
            one_hot[color_mask, class_id] = 1

        # Handle "others" class: Find pixels NOT in any defined class
        others_mask = np.ones((mask.shape[0], mask.shape[1]), dtype=bool)  # Start with all True
        for color, _ in self.color_to_class.items():
            color_mask = np.all(mask == np.array(color), axis=-1)
            others_mask = np.logical_and(others_mask, np.logical_not(color_mask))  # False if color is found

        one_hot[others_mask, self.other_class] = 1  # Assign "others" where needed
        one_hot = torch.from_numpy(one_hot).permute(2, 0, 1).float()  # (C, H, W) and tensor
        return one_hot

    def print_class_counts(self, mask_one_hot):
        """Prints the number of 1 values for each class in the one-hot mask."""

        print("Mask one hot shape: ", mask_one_hot.shape)
        for c in range(mask_one_hot.shape[0]):  # Iterate through classes
            class_count = torch.sum(mask_one_hot[c]).item()  # Count 1s in each channel
            print(f"Class {c}: {class_count} pixels")

class CustomVOCSegmentationDataset(torchvision.datasets.VOCSegmentation):
    def __init__(self, transform_config):
        super().__init__(global_config.seg_path_root_train, year="2012", download=True)
        self.transform_config = transform_config

        config_holder = ConfigHolder.getInstance()
        self.augment_mode = config_holder.get_network_attribute("augment_key", "none")

        self.transform_op = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #From ImageNet
                            ])

        self.target_op = transforms.Compose([
                            transforms.ToTensor()
                            ])

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        image = self.transform_op(image)
        target = self.target_op(target)

        return image, target