#Script to use for running heavy training.

import os

def train_sr_main():
    os.system("python \"train_sr_main.py\" --server_config=3 --img_to_load=1000 "
              "--plot_enabled=1 --save_per_iter=250 --network_version=\"mobisr_v01.05_div2k\" --iteration=1")

def test_sr_main():
    os.system("python \"test_sr_main.py\" --server_config=3 --img_to_load=-1 "
              "--plot_enabled=0 --network_version=\"mobisr_v01.05_div2k\" --iteration=1")

def download_ml_hypersim():
    os.system("python \"utils/ml_hypersim_dl.py\" --contains scene_cam_00_final_preview --contains .color.jpg")

def main():
    # train_sr_main()
    test_sr_main()
    # download_ml_hypersim()
    # os.system("shutdown /s /t 1")


if __name__ == "__main__":
    main()
