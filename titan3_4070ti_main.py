#Script to use for running heavy training.

import os

def train_sr_main():
    # os.system("python3 \"train_sr_main.py\" --server_config=4 --img_to_load=-1 "
    #           "--plot_enabled=0 --save_per_iter=500 --network_version=\"mobisr_v02.06_div2k.06.1\"")
    #
    # os.system("python3 \"train_sr_main.py\" --server_config=4 --img_to_load=-1 "
    #           "--plot_enabled=0 --save_per_iter=500 --network_version=\"mobisr_v02.06_div2k.06.2\"")

    os.system("python3 \"train_sr_main.py\" --server_config=4 --img_to_load=-1 "
              "--plot_enabled=0 --save_per_iter=500 --network_version=\"mobisr_v02.06_div2k.06.3\"")

    os.system("python3 \"train_sr_main.py\" --server_config=4 --img_to_load=-1 "
              "--plot_enabled=0 --save_per_iter=500 --network_version=\"mobisr_v02.06_div2k.06.4\"")
    #
    # os.system("python3 \"train_sr_main.py\" --server_config=4 --img_to_load=-1 "
    #           "--plot_enabled=0 --save_per_iter=500 --network_version=\"mobisr_v02.06_div2k.06.5\"")

def test_sr_main():
    os.system("python3 \"test_sr_main.py\" --server_config=4 --img_to_load=-1 "
              "--plot_enabled=1 --network_version=\"mobisr_v01.00_burstsr\" --iteration=3")

def train_img2img_main():
    os.system("python3 \"train_img2img_main.py\" --server_config=4 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=500 --network_version=\"fcg2cityscapes_v00.00.03.1\"")

    os.system("python3 \"train_img2img_main.py\" --server_config=4 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=500 --network_version=\"fcg2cityscapes_v00.00.03.2\"")

    os.system("python3 \"train_img2img_main.py\" --server_config=4 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=500 --network_version=\"fcg2cityscapes_v00.00.03.3\"")

    os.system("python3 \"train_img2img_main.py\" --server_config=4 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=500 --network_version=\"fcg2cityscapes_v00.00.03.4\"")

    os.system("python3 \"train_img2img_main.py\" --server_config=4 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=500 --network_version=\"fcg2cityscapes_v00.00.03.5\"")

    os.system("python3 \"train_img2img_main.py\" --server_config=4 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=500 --network_version=\"fcg2cityscapes_v00.00.03.6\"")

    os.system("python3 \"train_img2img_main.py\" --server_config=4 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=500 --network_version=\"fcg2cityscapes_v00.00.04.1\"")

    os.system("python3 \"train_img2img_main.py\" --server_config=4 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=500 --network_version=\"fcg2cityscapes_v00.00.04.2\"")

    os.system("python3 \"train_img2img_main.py\" --server_config=4 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=500 --network_version=\"fcg2cityscapes_v00.00.04.3\"")

    os.system("python3 \"train_img2img_main.py\" --server_config=4 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=500 --network_version=\"fcg2cityscapes_v00.00.04.4\"")

    os.system("python3 \"train_img2img_main.py\" --server_config=4 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=500 --network_version=\"fcg2cityscapes_v00.00.04.5\"")

    os.system("python3 \"train_img2img_main.py\" --server_config=4 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=500 --network_version=\"fcg2cityscapes_v00.00.04.6\"")

def run_util_script_main():
    os.system("python3 \"util_script_main.py\"")

def download_ml_hypersim():
    os.system("python3 \"utils/ml_hypersim_dl.py\" --contains scene_cam_00_final_preview --contains .color.jpg")

def main():
    # train_sr_main()
    # test_sr_main()
    # download_ml_hypersim()
    run_util_script_main()
    train_img2img_main()
    # os.system("shutdown /s /t 1")


if __name__ == "__main__":
    main()
