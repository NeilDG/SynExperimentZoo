#Script to use for running heavy training.

import os

def train_sr_main():
    os.system("python \"train_sr_main.py\" --server_config=3 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=250 --network_version=\"mobisr_v02.06_div2k.12.3\"")

def test_sr_main():
    os.system("python \"test_sr_main.py\" --server_config=3 --img_to_load=-1 "
              "--plot_enabled=1 --network_version=\"mobisr_v02.06_div2k.02.1")

def train_seg_main():
    # os.system("python \"train_seg_main_2.py\" --server_config=3 --img_to_load=-1 "
    #           "--plot_enabled=1 --save_per_iter=500 --network_version=\"synseg_v00.00_cityscapes.01.1\"")

    os.system("python \"train_seg_main_2.py\" --server_config=3 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=500 --network_version=\"synseg_v00.00_fcg.01.1\"")

def train_img2img_main():
    os.system("python \"train_img2img_main.py\" --server_config=3 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=500 --network_version=\"fcg2cityscapes_v00.00.01.1\"")


def download_ml_hypersim():
    os.system("python \"utils/ml_hypersim_dl.py\" --contains scene_cam_00_final_preview --contains .color.jpg")

def main():
    # train_sr_main()
    # test_sr_main()
    train_img2img_main()
    # download_ml_hypersim()
    # os.system("shutdown /s /t 1")


if __name__ == "__main__":
    main()
