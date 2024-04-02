import sys

import gdown
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)

def main(argv):
    (opts, args) = parser.parse_args(argv)

    if (opts.server_config == 0):
        output_dir = "/scratch3/neil.delgallego/SuperRes Dataset/"
    elif (opts.server_config == 4):
        output_dir = "/Documents/SuperRes Dataset/"
    else:
        output_dir = "/home/jupyter-neil.delgallego/SuperRes Dataset/"

    # Flickr2K
    direct_link = "https://drive.google.com/file/d/1X6162AwOPt_zDlPreQJK1XU5Kmzw2igE/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id=" + id
    gdown.download(url, output=output_dir, use_cookies=False)

    #v_istd+srd
    # direct_link = "https://drive.google.com/file/d/1mmI14uOtZzXzVX3P2AwRmdfkmNvJCUQe/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False, quiet=False)

    # url = "https://drive.google.com/drive/folders/1mhKmxwDODP4aCccG39FyQLlR0QVBvNA2?usp=sharing"
    # gdown.download_folder(url, output=output_dir, use_cookies=False)

    # url = "https://drive.google.com/drive/folders/1mhLqCankhs2i2sH2MGHea9kUCAbsiDy_?usp=sharing"
    # gdown.download_folder(url, output=output_dir, use_cookies=False)

    #v66_usr dataset
    # url = "https://drive.google.com/drive/folders/1m2KBqGJGDl8vITVTp4tnjnrKX1hTEe1k?usp=sharing"
    # gdown.download_folder(url, output=output_dir, use_cookies=False)
    #

if __name__ == "__main__":
    main(sys.argv)

