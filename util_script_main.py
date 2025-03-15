# Scripts for running utility functions.
# E.g. cropping an image dataset into patches

import torch
import kornia
import cv2
import glob
import os
import numpy as np
import global_config
import cv2
import loaders.segmentation_datasets as segmentation_datasets

def parse_string(input_string):
    """
    Parses a string of the format "AA.BB.CC.DD" and returns a tuple containing:
        - a new string with "AA.BB"
        - the integer value of "CC"
        - the integer value of "DD"

    Args:
        input_string: The string to parse (format "AA.BB.CC.DD").

    Returns:
        A tuple containing the parsed string and the two integer values.

    Raises:
        ValueError: If the input string is not in the correct format.
    """
    # Split the string by "."
    parts = input_string.split(".")

    # Check if there are exactly four parts
    if len(parts) != 4:
        raise ValueError("Input string must be in the format 'AA.BB.CC.DD'")

    # Extract the first two parts as a string
    parsed_string = ".".join(parts[:2])

    # Convert the last two parts to integers
    try:
        int_cc = int(parts[2])
        int_dd = int(parts[3])
    except ValueError:
        raise ValueError("CC and DD must be integers")

    return parsed_string, int_cc, int_dd

def patchify(input_dir, reference_dir, output_dir):
    patch_size = (64, 64)  # Size of the patches
    stride = (64, 64)  # Stride for patching

    # Get all image paths
    image_paths = glob.glob(os.path.join(input_dir, "*.png"))
    for image_path in image_paths:
        # Read image
        img_name = os.path.basename(image_path).split('.')[0]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        ref_img = cv2.imread(reference_dir + img_name + ".png")
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        h, w, _ = ref_img.shape

        #resize img to be equal to ref_img
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

        # Convert to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Patchify
        patches = kornia.contrib.extract_tensor_patches(img_tensor, patch_size, stride, allow_auto_padding=True)

        # Get image name and subdirectory
        subdirectory = os.path.basename(os.path.dirname(image_path))

        # Create subdirectory in the output directory
        subdirectory_path = os.path.join(output_dir, subdirectory)
        os.makedirs(subdirectory_path, exist_ok=True)

        # Save patches
        for j in range(patches.shape[1]):
            patch = patches[0, j].squeeze().permute(1, 2, 0).numpy() * 255.0
            patch = patch.astype(np.uint8)
            patch_name = f"{img_name}_patched_{j}.png"
            patch_path = os.path.join(subdirectory_path, patch_name)
            cv2.imwrite(patch_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

        print("Saved patches to ", image_path, " Input image size: ", np.shape(img), " Ref image size: ", np.shape(ref_img))

def patchify_without_ref(input_dir, output_dir, patch_size, stride, start_index=0):
    # Get all image paths
    # image_paths = glob.glob(os.path.join(input_dir, "*.png"))
    image_paths = glob.glob(input_dir)
    image_paths = image_paths[start_index:]
    print("Image paths length: ", len(image_paths))

    for image_path in image_paths:
        # Read image
        img_name = os.path.basename(image_path).split('.')[0]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Convert to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Patchify
        patches = kornia.contrib.extract_tensor_patches(img_tensor, patch_size, stride, allow_auto_padding=True)

        # Get image name and subdirectory
        subdirectory = os.path.basename(os.path.dirname(image_path))

        # Create subdirectory in the output directory
        subdirectory_path = os.path.join(output_dir, subdirectory)
        os.makedirs(subdirectory_path, exist_ok=True)

        # Save patches
        for j in range(patches.shape[1]):
            patch = patches[0, j].squeeze().permute(1, 2, 0).numpy() * 255.0
            # patch = patch.astype(np.uint8)
            patch_name = f"{img_name}_patched_{j}.png"
            patch_path = os.path.join(subdirectory_path, patch_name)
            cv2.imwrite(patch_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

        print("Saved patches to ", image_path, " Input image size: ", np.shape(img))

def patchify_segmentation(input_dir, output_dir, patch_size, stride):

    # Get all image paths
    # image_paths = glob.glob(os.path.join(input_dir, "*.png"))
    image_paths = glob.glob(input_dir)
    for image_path in image_paths:
        # Read image
        img_name = os.path.basename(image_path).split('.')[0]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Convert to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Patchify
        patches = kornia.contrib.extract_tensor_patches(img_tensor, patch_size, stride, allow_auto_padding=True)

        # Get image name and subdirectory
        subdirectory = os.path.basename(os.path.dirname(image_path))

        # Create subdirectory in the output directory
        subdirectory_path = os.path.join(output_dir, subdirectory)
        os.makedirs(subdirectory_path, exist_ok=True)

        # Save patches
        for j in range(patches.shape[1]):
            patch_tensor = patches[0, j].squeeze().permute(1, 2, 0) * 255.0
            patch = patch_tensor.numpy()
            # patch = patch.astype(np.uint8)
            patch_name = f"{img_name}_patched_{j}.png"
            patch_path = os.path.join(subdirectory_path, patch_name)
            cv2.imwrite(patch_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

            #also save the class labels
            mask_label = segmentation_datasets.mask_to_labels(patch_tensor).numpy()
            mask_name = f"{img_name}_patched_label_{j}.txt"
            np.savetxt(os.path.join(subdirectory_path, mask_name), mask_label, fmt='%d')

        print("Saved patches to ", image_path, " Input image size: ", np.shape(img))


def main():
    patch_size = (256, 256)  # Size of the patches
    stride = (256, 256)  # Stride for patching

    input_dir = "D:/Datasets/Segmentation Dataset/FCG-Synth-01/sequence.0/*.camera.png"  # Path to the input dataset
    output_dir = "D:/Datasets/Segmentation Dataset/FCG-Synth-01-patched/train-rgb/"  # Directory to save the patches
    patchify_without_ref(input_dir, output_dir, patch_size, stride)

    input_dir = "D:/Datasets/Segmentation Dataset/CityScapes/leftImg8bit/train/*/*.png"  # Path to the input dataset
    output_dir = "D:/Datasets/Segmentation Dataset/CityScapes-01-patched/train-rgb/"  # Directory to save the patches
    patchify_without_ref(input_dir, output_dir, patch_size, stride)

    # input_dir = "X:/GithubProjects/NeuralNets-SynthWorkplace_V3/Dataset/solo_1/sequence.0/*.camera.png"  # Path to the input dataset
    # output_dir = "X:/Segmentation Dataset/FCG-Synth/train-rgb/"  # Directory to save the patches
    # patchify_without_ref(input_dir, output_dir, patch_size, stride)
    #
    # input_dir = "X:/GithubProjects/NeuralNets-SynthWorkplace_V3/Dataset/solo_1/sequence.0/*.segmentation.png"  # Path to the input dataset
    # output_dir = "X:/Segmentation Dataset/FCG-Synth/train-seg/"  # Directory to save the patches
    # patchify_segmentation(input_dir, output_dir, patch_size, stride)

if __name__=="__main__":
    main()