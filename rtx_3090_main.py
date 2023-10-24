#Script to use for running heavy training.

import os

def train_img2img():
    os.system("python \"train_img2img_main.py\" --server_config=3 --img_to_load=-1 "
              "--plot_enabled=0 --save_per_iter=250 --network_version=\"mobisr_v01.00_mipd\" --iteration=3")

def main():
    train_img2img()
    # os.system("shutdown /s /t 1")


if __name__ == "__main__":
    main()
