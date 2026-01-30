#Script to use for running heavy training.

import os

def train_sr_main():
    os.system("python3 \"train_sr_main.py\" --server_config=1 --img_to_load=-1 "
              "--plot_enabled=0 --save_per_iter=250 --network_version=\"mobisr_v02.05_hypersim.10.1\" --cuda_device=\"cuda:2\"")

    os.system("python3 \"train_sr_main.py\" --server_config=1 --img_to_load=-1 "
              "--plot_enabled=0 --save_per_iter=250 --network_version=\"mobisr_v02.05_hypersim.10.2\" --cuda_device=\"cuda:2\"")

    os.system("python3 \"train_sr_main.py\" --server_config=1 --img_to_load=-1 "
              "--plot_enabled=0 --save_per_iter=250 --network_version=\"mobisr_v02.05_hypersim.10.3\" --cuda_device=\"cuda:2\"")

    os.system("python3 \"train_sr_main.py\" --server_config=1 --img_to_load=-1 "
              "--plot_enabled=0 --save_per_iter=250 --network_version=\"mobisr_v02.05_hypersim.10.4\" --cuda_device=\"cuda:2\"")

    os.system("python3 \"train_sr_main.py\" --server_config=1 --img_to_load=-1 "
              "--plot_enabled=0 --save_per_iter=250 --network_version=\"mobisr_v02.05_hypersim.10.5\" --cuda_device=\"cuda:2\"")

    # os.system("python3 \"train_sr_main.py\" --server_config=1 --img_to_load=-1 "
    #           "--plot_enabled=0 --save_per_iter=250 --network_version=\"mobisr_v02.05_hypersim.11.1\"")
    #
    # os.system("python3 \"train_sr_main.py\" --server_config=1 --img_to_load=-1 "
    #           "--plot_enabled=0 --save_per_iter=250 --network_version=\"mobisr_v02.05_hypersim.11.2\"")
    #
    # os.system("python3 \"train_sr_main.py\" --server_config=1 --img_to_load=-1 "
    #           "--plot_enabled=0 --save_per_iter=250 --network_version=\"mobisr_v02.05_hypersim.11.3\"")
    #
    # os.system("python3 \"train_sr_main.py\" --server_config=1 --img_to_load=-1 "
    #           "--plot_enabled=0 --save_per_iter=250 --network_version=\"mobisr_v02.05_hypersim.11.4\"")
    #
    # os.system("python3 \"train_sr_main.py\" --server_config=1 --img_to_load=-1 "
    #           "--plot_enabled=0 --save_per_iter=250 --network_version=\"mobisr_v02.05_hypersim.11.5\"")

def test_sr_main():
    os.system("python3 \"test_sr_main.py\" --server_config=1 --img_to_load=-1 "
              "--plot_enabled=0 --network_version=\"mobisr_v02.05_hypersim.10.1")

    os.system("python3 \"test_sr_main.py\" --server_config=1 --img_to_load=-1 "
              "--plot_enabled=0 --network_version=\"mobisr_v02.05_hypersim.10.2")

    os.system("python3 \"test_sr_main.py\" --server_config=1 --img_to_load=-1 "
              "--plot_enabled=0 --network_version=\"mobisr_v02.05_hypersim.10.3")

def train_seg_main():
    # os.system("python \"train_seg_main_2.py\" --server_config=3 --img_to_load=-1 "
    #           "--plot_enabled=1 --save_per_iter=500 --network_version=\"synseg_v00.00_cityscapes.01.1\"")

    os.system("python3 \"train_seg_main_2.py\" --server_config=1 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=500 --network_version=\"synseg_v00.00_fcg.01.1\"")

def train_img2img_main():
    os.system("python3 \"train_img2img_main.py\" --server_config=1 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=500 --network_version=\"fcg2cityscapes_v00.00.01.1\"")

def test_img2img_main():
    os.system("python3 \"test_img2img_main.py\" --server_config=1 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=500 --network_version=\"fcg2cityscapes_v00.00.01.1\"")


def download_ml_hypersim():
    os.system("python3 \"utils/ml_hypersim_dl.py\" --contains scene_cam_00_final_preview --contains .color.jpg")

def main():
    train_sr_main()
    # test_sr_main()
    # train_img2img_main()
    # test_img2img_main()
    # download_ml_hypersim()
    # os.system("shutdown /s /t 1")


if __name__ == "__main__":
    main()
