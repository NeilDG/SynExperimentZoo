# Copyright 2020 by Andrey Ignatov. All Rights Reserved.
import glob

# python dng_to_png.py path_to_my_dng_file.dng

import numpy as np
import imageio
import rawpy
import sys
import os

import torchvision
from torchvision.utils import save_image


# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import cv2
import numpy as np
import pickle as pkl
import torch.nn.functional as F
import zipfile
import shutil

import global_config
from loaders import dataset_loader


def load_txt(path):
    with open(path, 'r') as fh:
        out = [d.rstrip() for d in fh.readlines()]

    return out


class SamsungRAWImage:
    """ Custom class for RAW images captured from Samsung Galaxy S8 """
    @staticmethod
    def load(path):
        im_raw = cv2.imread('{}/im_raw.png'.format(path), cv2.IMREAD_UNCHANGED)
        im_raw = np.transpose(im_raw, (2, 0, 1)).astype(np.int16)
        im_raw = torch.from_numpy(im_raw)
        meta_data = pkl.load(open('{}/meta_info.pkl'.format(path), "rb", -1))

        return SamsungRAWImage(im_raw, meta_data['black_level'], meta_data['cam_wb'],
                               meta_data['daylight_wb'], meta_data['color_matrix'], meta_data['exif_data'],
                               meta_data.get('im_preview', None))

    def __init__(self, im_raw, black_level, cam_wb, daylight_wb, color_matrix, exif_data, im_preview=None):
        self.im_raw = im_raw
        self.black_level = black_level
        self.cam_wb = cam_wb
        self.daylight_wb = daylight_wb
        self.color_matrix = color_matrix
        self.exif_data = exif_data
        self.im_preview = im_preview

        self.norm_factor = 1023.0

    def get_all_meta_data(self):
        return {'black_level': self.black_level, 'cam_wb': self.cam_wb, 'daylight_wb': self.daylight_wb,
                'color_matrix': self.color_matrix.tolist()}

    def get_exposure_time(self):
        return self.exif_data['Image ExposureTime'].values[0].decimal()

    def get_noise_profile(self):
        noise = self.exif_data['Image Tag 0xC761'].values
        noise = [n[0] for n in noise]
        noise = np.array(noise).reshape(3, 2)
        return noise

    def get_f_number(self):
        return self.exif_data['Image FNumber'].values[0].decimal()

    def get_iso(self):
        return self.exif_data['Image ISOSpeedRatings'].values[0]

    def get_image_data(self, substract_black_level=False, white_balance=False, normalize=False):
        im_raw = self.im_raw.float()

        if substract_black_level:
            im_raw = im_raw - torch.tensor(self.black_level).view(4, 1, 1)

        if white_balance:
            im_raw = im_raw * torch.tensor(self.cam_wb).view(4, 1, 1)

        if normalize:
            im_raw = im_raw / self.norm_factor

        return im_raw

    def shape(self):
        shape = (4, self.im_raw.shape[1], self.im_raw.shape[2])
        return shape

    def crop_image(self, r1, r2, c1, c2):
        self.im_raw = self.im_raw[:, r1:r2, c1:c2]

    def get_crop(self, r1, r2, c1, c2):
        im_raw = self.im_raw[:, r1:r2, c1:c2]

        if self.im_preview is not None:
            im_preview = self.im_preview[2*r1:2*r2, 2*c1:2*c2]
        else:
            im_preview = None

        return SamsungRAWImage(im_raw, self.black_level, self.cam_wb, self.daylight_wb, self.color_matrix,
                               self.exif_data, im_preview=im_preview)

    def postprocess(self, return_np=True, norm_factor=None):
        raise NotImplementedError


class CanonImage:
    """ Custom class for RAW images captured from Canon DSLR """
    @staticmethod
    def load(path):
        im_raw = cv2.imread('{}/im_raw.png'.format(path), cv2.IMREAD_UNCHANGED)
        im_raw = np.transpose(im_raw, (2, 0, 1)).astype(np.int16)
        im_raw = torch.from_numpy(im_raw)
        meta_data = pkl.load(open('{}/meta_info.pkl'.format(path), "rb", -1))

        return CanonImage(im_raw.float(), meta_data['black_level'], meta_data['cam_wb'],
                          meta_data['daylight_wb'], meta_data['rgb_xyz_matrix'], meta_data['exif_data'])

    @staticmethod
    def generate_processed_image(im, meta_data, return_np=False, external_norm_factor=None, gamma=True, smoothstep=True,
                                 no_white_balance=False):
        im = im * meta_data.get('norm_factor', 1.0)

        if not meta_data.get('black_level_subtracted', False):
            im = (im - torch.tensor(meta_data['black_level'])[[0, 1, -1]].view(3, 1, 1))

        if not meta_data.get('while_balance_applied', False) and not no_white_balance:
            im = im * torch.tensor(meta_data['cam_wb'])[[0, 1, -1]].view(3, 1, 1) / torch.tensor(meta_data['cam_wb'])[1]

        im_out = im

        if external_norm_factor is None:
            im_out = im_out / (im_out.mean() * 5.0)
        else:
            im_out = im_out / external_norm_factor

        im_out = im_out.clamp(0.0, 1.0)

        if gamma:
            im_out = im_out ** (1.0 / 2.2)

        if smoothstep:
            # Smooth curve
            im_out = 3 * im_out ** 2 - 2 * im_out ** 3

        if return_np:
            im_out = im_out.permute(1, 2, 0).numpy() * 255.0
            im_out = im_out.astype(np.uint8)
        return im_out

    def __init__(self, im_raw, black_level, cam_wb, daylight_wb, rgb_xyz_matrix, exif_data):
        super(CanonImage, self).__init__()
        self.im_raw = im_raw

        if len(black_level) == 4:
            black_level = [black_level[0], black_level[1], black_level[3]]
        self.black_level = black_level

        if len(cam_wb) == 4:
            cam_wb = [cam_wb[0], cam_wb[1], cam_wb[3]]
        self.cam_wb = cam_wb

        if len(daylight_wb) == 4:
            daylight_wb = [daylight_wb[0], daylight_wb[1], daylight_wb[3]]
        self.daylight_wb = daylight_wb

        self.rgb_xyz_matrix = rgb_xyz_matrix

        self.exif_data = exif_data

        self.norm_factor = 16383

    def shape(self):
        shape = (3, self.im_raw.shape[1], self.im_raw.shape[2])
        return shape

    def get_all_meta_data(self):
        return {'black_level': self.black_level, 'cam_wb': self.cam_wb, 'daylight_wb': self.daylight_wb,
                'rgb_xyz_matrix': self.rgb_xyz_matrix.tolist(), 'norm_factor': self.norm_factor}

    def get_exposure_time(self):
        return self.exif_data['EXIF ExposureTime'].values[0].decimal()

    def get_f_number(self):
        return self.exif_data['EXIF FNumber'].values[0].decimal()

    def get_iso(self):
        return self.exif_data['EXIF ISOSpeedRatings'].values[0]

    def get_image_data(self, substract_black_level=False, white_balance=False, normalize=False):
        im_raw = self.im_raw.float()

        if substract_black_level:
            im_raw = im_raw - torch.tensor(self.black_level).view(3, 1, 1)

        if white_balance:
            im_raw = im_raw * torch.tensor(self.cam_wb).view(3, 1, 1) / 1024.0

        if normalize:
            im_raw = im_raw / self.norm_factor

        return im_raw

    def set_image_data(self, im_data):
        self.im_raw = im_data

    def crop_image(self, r1, r2, c1, c2):
        self.im_raw = self.im_raw[:, r1:r2, c1:c2]

    def get_crop(self, r1, r2, c1, c2):
        im_raw = self.im_raw[:, r1:r2, c1:c2]
        return CanonImage(im_raw, self.black_level, self.cam_wb, self.daylight_wb, self.rgb_xyz_matrix,
                          self.exif_data)

    def set_crop_info(self, crop_info):
        self.crop_info = crop_info

    def resize(self, size=None, scale_factor=None):
        self.im_raw = F.interpolate(self.im_raw.unsqueeze(0), size=size, scale_factor=scale_factor,
                                    mode='bilinear').squeeze(0)

    def postprocess(self, return_np=True):
        raise NotImplementedError

def extract_bayer_channels(raw):

    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    return ch_R, ch_Gr, ch_B, ch_Gb

def from_raw_to_png(file):
    print("Converting file " + file)

    if not os.path.isfile(file):
        print("The file doesn't exist!")
        sys.exit()

    raw = rawpy.imread(file)
    raw_image = raw.raw_image
    del raw

    # Use the following code to rotate the image (if needed)
    # raw_image = np.rot90(raw_image, k=2)

    raw_image = raw_image.astype(np.float32)
    ch_R, ch_Gr, ch_B, ch_Gb = extract_bayer_channels(raw_image)

    png_image = raw_image.astype(np.uint16)
    new_name = file.replace("_raw.png", "_rgb.png")
    imageio.imwrite(new_name, png_image)

def read_canon_raw(file, index):
    # print("Converting canon raw file to image " + file)

    image = CanonImage.load(file)
    file_name = file + "/im_raw.png"
    new_name = file_name.replace("_raw.png", "_rgb_" + str(index) + ".png")
    im_data = image.get_image_data(True, True, True) * 10.0
    print(new_name)
    save_image(im_data, new_name)
    # cv2.imwrite(new_name, im_data)

def read_samsung_raw(file, index):
    # print("Converting samsung raw file to image " + file)

    image = SamsungRAWImage.load(file)
    file_name = file + "/im_raw.png"
    new_name = file_name.replace("_raw.png", "_rgb_" + str(index) + ".png")
    im_data = image.get_image_data(True, True, True)
    print(new_name)
    save_image(im_data, new_name)


def convert_raw_dataset_to_png():
    gt_path = "X:/SuperRes Dataset/v02_burstsr/*/*/canon/"
    lr_path = "X:/SuperRes Dataset/v02_burstsr/*/*/samsung_**/"

    hr_path_list = glob.glob(gt_path)
    for i in range(0, len(hr_path_list)):
        read_canon_raw(hr_path_list[i], i)
        print(hr_path_list[i])

    lr_path_list = glob.glob(lr_path)
    for i in range(0, len(lr_path_list)):
        read_samsung_raw(lr_path_list[i], i)
        print(lr_path_list[i])

def organize_burstsr_files_for_train():
    lr_path = "X:/SuperRes Dataset/v02_burstsr/val/*/samsung_00/im_rgb_*.png"
    hr_path = "X:/SuperRes Dataset/v02_burstsr/val/*/canon/im_rgb_*.png"

    global_config.test_size = 64
    test_loader, test_count = dataset_loader.load_base_img2img_dataset(lr_path, hr_path)
    new_lr_path = "X:/SuperRes Dataset/v02_burstsr/lr/"
    new_hr_path = "X:/SuperRes Dataset/v02_burstsr/hr/"

    if not os.path.exists(new_lr_path):
        os.makedirs(new_lr_path, exist_ok=True)

    if not os.path.exists(new_hr_path):
        os.makedirs(new_hr_path, exist_ok=True)

    for i, (file_name, a_batch, b_batch) in enumerate(test_loader, 0):
        for j in range(0, len(a_batch)):
            im_path = new_lr_path + file_name[j] + ".png"
            torchvision.utils.save_image(a_batch[j], im_path, normalize=True)

            im_path = new_hr_path + file_name[j] + ".png"
            torchvision.utils.save_image(b_batch[j], im_path, normalize=True)


if __name__ == "__main__":
    convert_raw_dataset_to_png()
    organize_burstsr_files_for_train()
