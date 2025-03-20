#Script to use for running heavy training.

import os

def train_sr_main():
    os.system("python \"train_sr_main.py\" --server_config=7 --img_to_load=-1 "
              "--plot_enabled=0 --save_per_iter=250 --network_version=\"mobisr_v02.05_div2k.09.1\"")

    os.system("python \"train_sr_main.py\" --server_config=7 --img_to_load=-1 "
              "--plot_enabled=0 --save_per_iter=250 --network_version=\"mobisr_v02.05_div2k.09.2\"")

def test_sr_main():
    os.system("python \"test_sr_main.py\" --server_config=7 --img_to_load=-1 "
              "--plot_enabled=1 --network_version=\"mobisr_v01.00_burstsr\" --iteration=3")

def train_seg_main():
    os.system("python \"train_seg_main_2.py\" --server_config=7 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=500 --network_version=\"synseg_v00.00_fcg.01.1\"")


def test_seg_main():
    os.system("python \"test_seg_main.py\" --server_config=7 --img_to_load=-1 "
              "--plot_enabled=1 --network_version=\"synseg_v00.00_cityscapes.01.1\"")


def train_img2img_main():
    os.system("python \"train_img2img_main.py\" --server_config=7 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=500 --network_version=\"fcg2cityscapes_v00.00.01.3\"")

def run_util_script_main():
    os.system("python \"util_script_main.py\"")

def download_ml_hypersim():
    os.system("python \"utils/ml_hypersim_dl.py\" --contains scene_cam_00_final_preview --contains .color.jpg")

def main():
    # train_sr_main()
    # test_sr_main()
    run_util_script_main()
    train_img2img_main()
    # download_ml_hypersim()
    # os.system("shutdown /s /t 1")


if __name__ == "__main__":
    main()
