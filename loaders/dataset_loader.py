import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import glob
import random
import torch
import global_config
from config.network_config import ConfigHolder
from loaders import superres_datasets, segmentation_datasets

def load_train_img2img_dataset(a_path, b_path):
    network_config = ConfigHolder.getInstance().get_network_config()
    a_list = glob.glob(a_path)
    b_list = glob.glob(b_path)

    if (global_config.img_to_load > 0):
        a_list = a_list[0: global_config.img_to_load]
        b_list = b_list[0: global_config.img_to_load]

    # Ensure a_list and b_list have at least X00,000 elements
    ideal_sample_size = 300000
    if len(a_list) < ideal_sample_size:
        extend_length = ideal_sample_size - len(a_list)
        a_list.extend(a_list * (extend_length // len(a_list) + 1))

    if len(b_list) < len(a_list):
        extend_length = len(a_list) - len(b_list)
        b_list.extend(b_list * (extend_length // len(b_list) + 1))

    num_workers = global_config.num_workers
    print("Length of train images: %d %d. Num workers: %d" % (len(a_list), len(b_list), num_workers))

    data_loader = torch.utils.data.DataLoader(
        superres_datasets.PairedImageDataset(a_list, b_list, 1),
        batch_size=global_config.load_size,
        num_workers=num_workers, pin_memory=True, prefetch_factor=2
    )

    return data_loader, len(a_list)

def load_test_img2img_dataset(a_path, b_path):
    a_list = glob.glob(a_path)
    b_list = glob.glob(b_path)

    if (global_config.img_to_load > 0):
        a_list = a_list[0: global_config.img_to_load]
        b_list = b_list[0: global_config.img_to_load]

    temp_list = list(zip(a_list, b_list))
    random.shuffle(temp_list)
    a_list, b_list = zip(*temp_list)

    img_length = len(a_list)
    print("Length of test images: %d %d" % (img_length, len(b_list)))

    data_loader = torch.utils.data.DataLoader(
        superres_datasets.PairedImageDataset(a_list, b_list, 2),
        batch_size=global_config.test_size,
        num_workers=1
    )

    return data_loader, img_length

def load_base_img2img_dataset(a_path, b_path):
    a_list = glob.glob(a_path)
    b_list = glob.glob(b_path)

    if (global_config.img_to_load > 0):
        a_list = a_list[0: global_config.img_to_load]
        b_list = b_list[0: global_config.img_to_load]

    temp_list = list(zip(a_list, b_list))
    random.shuffle(temp_list)
    a_list, b_list = zip(*temp_list)

    img_length = len(a_list)
    print("Length of images: %d %d" % (img_length, len(b_list)))

    data_loader = torch.utils.data.DataLoader(
        superres_datasets.BasePairedImageDataset(a_list, b_list),
        batch_size=global_config.test_size,
        num_workers=1
    )

    return data_loader, img_length

def load_singleimg_dataset(a_path):
    a_list = glob.glob(a_path)

    if (global_config.img_to_load > 0):
        a_list = a_list[0: global_config.img_to_load]

    random.shuffle(a_list)

    img_length = len(a_list)
    print("Length of images: %d" % (img_length))

    data_loader = torch.utils.data.DataLoader(
        superres_datasets.SingleImageDataset(a_list, 1),
        batch_size=global_config.test_size,
        num_workers=4
    )

    return data_loader, img_length

# def load_huggingface_sr_dataset(eval_dataset:EvalDataset):
#     data_loader = torch.utils.data.DataLoader(
#         eval_dataset,
#         batch_size=global_config.test_size,
#         num_workers=4
#     )
#
#     return data_loader

def load_cityscapes_dataset():
    dataset = segmentation_datasets.CustomCityscapesDataset(1)

    num_workers = global_config.num_workers
    print("Loading Cityscapes. Num workers: %d" % (num_workers))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=global_config.load_size,
        num_workers=global_config.num_workers,
        shuffle=True, pin_memory=True, prefetch_factor=4
    )

    return data_loader, len(dataset)

def load_cityscapes_gan_dataset_train(a_path, b_path):
    a_list = glob.glob(a_path)
    b_list = glob.glob(b_path)

    if (global_config.img_to_load > 0):
        a_list = a_list[0: global_config.img_to_load]
        b_list = b_list[0: global_config.img_to_load]

    # Ensure a_list and b_list have at least X00,000 elements
    ideal_sample_size = len(a_list) * 10
    if len(a_list) < ideal_sample_size:
        extend_length = ideal_sample_size - len(a_list)
        a_list.extend(a_list * (extend_length // len(a_list) + 1))
        b_list.extend(b_list * (extend_length // len(b_list) + 1))

    temp_list = list(zip(a_list, b_list))
    random.shuffle(temp_list)
    a_list, b_list = zip(*temp_list)
    img_length = len(a_list)

    print("Loading Cityscapes train. Length of images: %d. Num workers: %d" % (img_length, global_config.num_workers))

    dataset = segmentation_datasets.CityscapesGANDataset(a_list, b_list, 1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=global_config.load_size,
        num_workers=global_config.num_workers,
        shuffle=False, pin_memory=True, prefetch_factor=4
    )

    return data_loader, img_length

def load_cityscapes_gan_dataset_test(a_path, b_path):
    a_list = glob.glob(a_path)
    b_list = glob.glob(b_path)

    if (global_config.img_to_load > 0):
        a_list = a_list[0: global_config.img_to_load]
        b_list = b_list[0: global_config.img_to_load]

    # Ensure a_list and b_list have at least X00,000 elements
    # ideal_sample_size = 300000
    # if len(a_list) < ideal_sample_size:
    #     extend_length = ideal_sample_size - len(a_list)
    #     a_list.extend(a_list * (extend_length // len(a_list) + 1))
    #     b_list.extend(b_list * (extend_length // len(b_list) + 1))

    temp_list = list(zip(a_list, b_list))
    random.shuffle(temp_list)
    a_list, b_list = zip(*temp_list)
    img_length = len(a_list)

    print("Loading Cityscapes test. Length of images: %d. Num workers: 1" % (img_length))

    dataset = segmentation_datasets.CityscapesGANDataset(a_list, b_list, 2)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=global_config.load_size,
        num_workers=1,
        shuffle=False, pin_memory=True, prefetch_factor=4
    )

    return data_loader, img_length

def load_cityscapes_dataset_train(rgb_path, label_path):
    rgb_list = glob.glob(rgb_path)
    # mask_list = glob.glob(mask_path)
    label_list = glob.glob(label_path)

    if (global_config.img_to_load > 0):
        rgb_list = rgb_list[0: global_config.img_to_load]
        # mask_list = mask_list[0: global_config.img_to_load]
        label_list = label_list[0: global_config.img_to_load]

    # Ensure rgb_list and b_list have at least X00,000 elements
    # ideal_sample_size = len(rgb_list) * 10
    # if len(rgb_list) < ideal_sample_size:
    #     extend_length = ideal_sample_size - len(rgb_list)
    #     rgb_list.extend(rgb_list * (extend_length // len(rgb_list) + 1))
    #     mask_list.extend(mask_list * (extend_length // len(mask_list) + 1))

    temp_list = list(zip(rgb_list, label_list))
    random.shuffle(temp_list)
    rgb_list, label_list = zip(*temp_list)

    dataset = segmentation_datasets.CityscapesDataset(rgb_list, label_list, 1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=global_config.load_size,
        num_workers=global_config.num_workers,
        shuffle=False, pin_memory=True
    )

    print("Loading Cityscapes train with one-hot. Length of images: %d %d. Num workers: %d" % (len(rgb_list), len(label_list), global_config.num_workers))
    return data_loader, len(rgb_list)

def load_cityscapes_dataset_test(rgb_path, label_path):
    rgb_list = glob.glob(rgb_path)
    # mask_list = glob.glob(mask_path)
    label_list = glob.glob(label_path)

    if (global_config.img_to_load > 0):
        rgb_list = rgb_list[0: global_config.img_to_load]
        # mask_list = mask_list[0: global_config.img_to_load]
        label_list = label_list[0: global_config.img_to_load]

    # temp_list = list(zip(rgb_list, mask_list))
    # random.shuffle(temp_list)
    # rgb_list, mask_list = zip(*temp_list)

    dataset = segmentation_datasets.CityscapesDataset(rgb_list, label_list, 2)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=global_config.load_size,
        num_workers=1,
        shuffle=False, pin_memory=True
    )

    print("Loading Cityscapes test with one-hot. Length of n    images: %d %d. Num workers: 1" % (len(rgb_list), len(label_list)))
    return data_loader, len(rgb_list)

def load_voc_dataset():
    dataset = segmentation_datasets.CustomVOCSegmentationDataset(1)

    num_workers = global_config.num_workers
    print("Loading VOC segmentation Dataset. Samples: %d Num workers: %d" % (len(dataset), num_workers))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=global_config.load_size,
        num_workers=global_config.num_workers,
        shuffle=True, pin_memory=True, prefetch_factor=4
    )

    return data_loader, len(dataset)

