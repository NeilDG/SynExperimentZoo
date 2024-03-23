import os
import sys
from datasets import load_dataset, Dataset
from super_image.data import EvalDataset
from unittest.mock import patch
import torch
from super_image import ImageLoader
import numpy as np

def store_input_dataset_to_local(output_dir, initial_dataset: Dataset, eval_dataset:EvalDataset):
    for i in range(0, len(eval_dataset)):
        image_name = initial_dataset[i]["hr"]
        image_name = image_name.split("/")[-1]

        lr_image, _ = eval_dataset[i]
        lr_image = torch.unsqueeze(torch.from_numpy(lr_image), 0)
        print(image_name, np.shape(lr_image))

        ImageLoader.save_image(lr_image, output_dir + image_name)

def store_highres_dataset_to_local(output_dir, initial_dataset: Dataset, eval_dataset:EvalDataset):
    for i in range(0, len(eval_dataset)):
        image_name = initial_dataset[i]["hr"]
        image_name = image_name.split("/")[-1]

        _, hr_image = eval_dataset[i]
        # lr_image = torch.unsqueeze(torch.from_numpy(lr_image), 0)
        hr_image = torch.unsqueeze(torch.from_numpy(hr_image), 0)
        print(image_name, np.shape(hr_image))

        ImageLoader.save_image(hr_image, output_dir + image_name)

def prepare_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

def main(argv):
    with patch("datasets.config.USE_PARQUET_EXPORT", False):
        # DOWNLOAD DIV2K
        initial_dataset = load_dataset('eugenesiow/Div2k', 'bicubic_x4', split='validation', cache_dir="./div2k_dataset/", save_infos=True)
        eval_dataset = EvalDataset(initial_dataset)

        input_dir = "X:/GithubProjects/SynExperimentZoo/reports/div2k/lr/"
        prepare_dir(input_dir)
        store_input_dataset_to_local(input_dir, initial_dataset, eval_dataset)

        target_dir = "X:/GithubProjects/SynExperimentZoo/reports/div2k/bicubic_x4/"
        prepare_dir(input_dir)
        store_highres_dataset_to_local(target_dir, initial_dataset, eval_dataset)

        initial_dataset = load_dataset('eugenesiow/Div2k', 'bicubic_x2', split='validation', cache_dir="./div2k_dataset/", save_infos=True)
        eval_dataset = EvalDataset(initial_dataset)

        target_dir = "X:/GithubProjects/SynExperimentZoo/reports/div2k/bicubic_x2/"
        prepare_dir(input_dir)
        store_highres_dataset_to_local(target_dir, initial_dataset, eval_dataset)

        initial_dataset = load_dataset('eugenesiow/Div2k', 'bicubic_x8', split='validation', cache_dir="./div2k_dataset/", save_infos=True)
        eval_dataset = EvalDataset(initial_dataset)

        target_dir = "X:/GithubProjects/SynExperimentZoo/reports/div2k/bicubic_x8/"
        prepare_dir(input_dir)
        store_highres_dataset_to_local(target_dir, initial_dataset, eval_dataset)

        initial_dataset = load_dataset('eugenesiow/Div2k', 'realistic_wild_x4', split='validation', cache_dir="./div2k_dataset/", save_infos=True)
        eval_dataset = EvalDataset(initial_dataset)

        target_dir = "X:/GithubProjects/SynExperimentZoo/reports/div2k/realistic_wild_x4/"
        prepare_dir(input_dir)
        store_highres_dataset_to_local(target_dir, initial_dataset, eval_dataset)

        # DOWNLOAD URBAN100
        initial_dataset = load_dataset('eugenesiow/Urban100', 'bicubic_x4', split='validation', cache_dir="./urban100/", save_infos=True)
        eval_dataset = EvalDataset(initial_dataset)

        input_dir = "X:/GithubProjects/SynExperimentZoo/reports/urban100/lr/"
        prepare_dir(input_dir)
        store_input_dataset_to_local(input_dir, initial_dataset, eval_dataset)

        target_dir = "X:/GithubProjects/SynExperimentZoo/reports/urban100/bicubic_x4/"
        prepare_dir(input_dir)
        store_highres_dataset_to_local(target_dir, initial_dataset, eval_dataset)




if __name__ == "__main__":
    main(sys.argv)