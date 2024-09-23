#Script to use for running heavy training.

import os

def train_sr_main():
    os.system("python3 \"train_sr_main.py\" --server_config=4 --img_to_load=-1 "
              "--plot_enabled=0 --save_per_iter=250 --network_version=\"mobisr_v01.03_div2k\" --iteration=3")

    os.system("python3 \"train_sr_main.py\" --server_config=4 --img_to_load=-1 "
              "--plot_enabled=0 --save_per_iter=250 --network_version=\"mobisr_v01.04_div2k\" --iteration=3")

    os.system("python3 \"train_sr_main.py\" --server_config=4 --img_to_load=-1 "
              "--plot_enabled=0 --save_per_iter=250 --network_version=\"mobisr_v01.06_div2k\" --iteration=1")

    os.system("python3 \"train_sr_main.py\" --server_config=4 --img_to_load=-1 "
              "--plot_enabled=0 --save_per_iter=250 --network_version=\"mobisr_v01.06_div2k\" --iteration=2")

def test_sr_main():
    os.system("python3 \"test_sr_main.py\" --server_config=3 --img_to_load=-1 "
              "--plot_enabled=1 --network_version=\"mobisr_v01.00_burstsr\" --iteration=3")

def download_ml_hypersim():
    os.system("python3 \"utils/ml_hypersim_dl.py\" --contains scene_cam_00_final_preview --contains .color.jpg")

def main():
    train_sr_main()
    # test_sr_main()
    # download_ml_hypersim()
    # os.system("shutdown /s /t 1")


if __name__ == "__main__":
    main()
