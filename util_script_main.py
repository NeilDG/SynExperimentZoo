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

def patchify():
    patch_size = (64, 64)  # Size of the patches
    stride = (64, 64)  # Stride for patching

    input_dir = "X:/SuperRes Dataset/div2k/*/"  # Path to the image dataset
    reference_dir = "X:/SuperRes Dataset/div2k/bicubic_x4/"  # Reference size of the HR image
    output_dir = "X:/SuperRes Dataset/div2k_patched/"  # Directory to save the patches

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


def main():
    patchify()

if __name__=="__main__":
    main()