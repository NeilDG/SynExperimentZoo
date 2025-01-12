import torch
import cv2
import torchvision
from torch.utils import data
import torchvision.transforms as transforms

import global_config
from config.network_config import ConfigHolder

class CustomCityscapesDataset(torchvision.datasets.Cityscapes):
    def __init__(self, transform_config):
        super().__init__(global_config.seg_path_root_train, split='train', mode='fine', target_type='semantic', download=True)
        self.transform_config = transform_config

        config_holder = ConfigHolder.getInstance()
        self.augment_mode = config_holder.get_network_attribute("augment_key", "none")

        self.transform_op = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # From ImageNet
                            ])

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        image = self.transform(image)

        return image, target

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