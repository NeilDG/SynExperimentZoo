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


# simplified classification. 0 = nature, 1 = vehicle, 2 = building, 3 = road, 4 = props, 5 = people
color_to_class = {
    (128, 64, 128): 3,  # road
    (244, 35, 232): 3,  # sidewalk
    (70, 70, 70): 2,  # building
    (250, 170, 160): 2,  # wall
    (230, 150, 140): 2,  # fence
    (102, 102, 156): 4,  # pole
    (190, 153, 153): 4,  # traffic light
    (153, 153, 153): 4,  # traffic sign
    (107, 142, 35): 0,  # vegetation
    (152, 251, 152): 0,  # terrain
    (150, 251, 152): 0,  # sky
    (220, 20, 60): 5,  # person
    (255, 0, 0): 5,  # rider
    (0, 0, 142): 1,  # car
    (0, 0, 70): 1,  # truck
    (0, 60, 100): 1,  # bus
    (0, 80, 100): 1,  # train
    (0, 0, 230): 1,  # motorcycle
    (119, 11, 32): 1,  # bicycle
    (250, 170, 30): 3,  # rail track
}
color_to_class_len = len(color_to_class)

#
# color_to_class = {
#     (128, 64, 128): 0,  # road
#     (244, 35, 232): 1,  # sidewalk
#     (70, 70, 70): 2,    # building
#     (250, 170, 160): 3, # wall
#     (230, 150, 140): 4, # fence
#     (102, 102, 156): 5, # pole
#     (190, 153, 153): 6, # traffic light
#     (153, 153, 153): 7, # traffic sign
#     (107, 142, 35): 8,  # vegetation
#     (152, 251, 152): 9,  # terrain
#     (150, 251, 152): 10, # sky
#     (220, 20, 60): 11,  # person
#     (255, 0, 0): 12,    # rider
#     (0, 0, 142): 13,    # car
#     (0, 0, 70): 14,     # truck
#     (0, 60, 100): 15,    # bus
#     (0, 80, 100): 16,    # train
#     (0, 0, 230): 17,    # motorcycle
#     (119, 11, 32): 18,   # bicycle
#     (250, 170, 30): 19,  # rail track
# }

def mask_to_labels(mask_img, color_to_class, other_class:int):
    """Encodes the mask to [0, 1, 2, 3] labels."""
    mask_img = mask_img.permute(1, 2, 0).numpy()  # (H, W, 3) and numpy
    encoded_mask = np.zeros((mask_img.shape[0], mask_img.shape[1]), dtype=np.uint8)  # (H, W)

    for color, class_id in color_to_class.items():
        color = np.full(mask_img.shape, color)
        # print("Color values: ", np.mean(color), " Mask values: ", np.mean(mask_img))
        color_mask = np.all(mask_img == color, axis=-1)
        encoded_mask[color_mask] = class_id

    # Handle "others" class
    # others_mask = np.ones((mask.shape[0], mask.shape[1]), dtype=bool)
    # for color, _ in color_to_class.items():
    #     color_mask = np.all(mask == np.array(color), axis=-1)
    #     others_mask = np.logical_and(others_mask, np.logical_not(color_mask))
    # encoded_mask[others_mask] = other_class

    encoded_mask = torch.from_numpy(encoded_mask).to(torch.uint8)  # To tensor and Long type
    # print(encoded_mask)
    return encoded_mask

def labels_to_mask(mask_labels:torch.uint8, color_to_class):
    device = mask_labels.device
    rgb_mask = torch.zeros((mask_labels.shape[0], mask_labels.shape[1], 3), dtype=torch.uint8, device=device)  # (H, W, 3)

    for color, class_id in color_to_class.items():
        label_tensor = torch.tensor(class_id, device=device, dtype=torch.uint8)
        color_tensor = torch.tensor(color, dtype=torch.uint8, device=device)  # Convert color to tensor with uint8 dtype
        # print("RGB mask shape: ", rgb_mask.shape, "Mask shape: ", mask_labels.shape, "Color shape: ", color_tensor.shape, " Non-zeros in mask:", torch.count_nonzero(mask_labels))

        rgb_mask[mask_labels == label_tensor] = color_tensor

    rgb_mask = rgb_mask.permute(2, 0, 1)
    return rgb_mask


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
                transforms.RandomCrop(256),
                transforms.ToTensor()
            ])

        self.norm_op = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,))])

        self.other_class = 3 # "others"

    def __getitem__(self, idx):
        file_name = self.rgb_list[idx % len(self.rgb_list)].split("\\")[-1].split(".")[0]

        rgb_img = cv2.imread(self.rgb_list[idx])
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)


        mask_img = cv2.imread(self.mask_list[idx])
        # mask_file_name = file_name + "_color.png"
        # print("Mask file name: ", mask_file_name)
        # mask_img = cv2.imread(mask_file_name)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)

        state = torch.get_rng_state()
        rgb_img = self.initial_op(rgb_img)

        torch.set_rng_state(state)
        mask_img = self.initial_op(mask_img)
        mask_img = (mask_img * 255.0).to(torch.uint8)

        # mask = self.mask_to_onehot(mask_img)
        mask = mask_to_labels(mask_img, color_to_class, self.other_class)

        # self.print_class_counts(mask_one_hot)

        if self.use_tanh:
            rgb_img = self.norm_op(rgb_img)

        return file_name, rgb_img, mask, mask_img

    def __len__(self):
        return len(self.rgb_list)

    # def mask_to_onehot(self, mask):
    #     """Converts to one-hot, handling "others" class."""
    #     mask = mask.permute(1, 2, 0).numpy()  # (H, W, 3) and numpy
    #     one_hot = np.zeros((mask.shape[0], mask.shape[1], len(self.color_to_class) + 1), dtype=np.uint8)  # (H, W, C)
    #
    #     for color, class_id in self.color_to_class.items():
    #         color_mask = np.all(mask == np.array(color), axis=-1)
    #         one_hot[color_mask, class_id] = 1
    #
    #     # Handle "others" class: Find pixels NOT in any defined class
    #     others_mask = np.ones((mask.shape[0], mask.shape[1]), dtype=bool)  # Start with all True
    #     for color, _ in self.color_to_class.items():
    #         color_mask = np.all(mask == np.array(color), axis=-1)
    #         others_mask = np.logical_and(others_mask, np.logical_not(color_mask))  # False if color is found
    #
    #     one_hot[others_mask, self.other_class] = 1  # Assign "others" where needed
    #     one_hot = torch.from_numpy(one_hot).permute(2, 0, 1).float()  # (C, H, W) and tensor
    #     return one_hot

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