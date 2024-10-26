#!/bin/bash
#SBATCH -J DOWNLOAD
#SBATCH --partition=serial
#SBATCH --qos=84c-1d_serial
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --output=script_download.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

#About this script:
#Download of dataset
SERVER_CONFIG=4

module load anaconda
module load cuda
source activate NeilZoo

#do fresh install
#pip-review --local --auto
#pip install -I torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#pip install scikit-learn
#pip install scikit-image
#pip install visdom
#pip install kornia
#pip install -I opencv-python==4.5.5.62
#pip install --upgrade pillow
#pip install gputil
#pip install matplotlib
#pip install --upgrade --no-cache-dir gdown
#pip install PyYAML
#pip install ITTR_pytorch
#pip install super_image
#pip install timm

if [ "$SERVER_CONFIG" == 0 ]
then
  srun python "gdown_download.py" --server_config="$SERVER_CONFIG"
elif [ "$SERVER_CONFIG" == 4 ]
then
  python3 "gdown_download.py" --server_config="$SERVER_CONFIG"
else
  python "gdown_download.py" --server_config="$SERVER_CONFIG"
fi


if [ "$SERVER_CONFIG" == 0 ]
then
  OUTPUT_DIR="/scratch3/neil.delgallego/SuperRes Dataset/"
elif [ "$SERVER_CONFIG" == 4 ]
then
  OUTPUT_DIR="~/Documents/SuperRes Dataset/"
else
  OUTPUT_DIR="/home/jupyter-neil.delgallego/SuperRes Dataset/"
fi


#DATASET_NAME="Flickr2K"
#echo "$OUTPUT_DIR/$DATASET_NAME.zip"
#zip -F "$OUTPUT_DIR/$DATASET_NAME.zip" --out "$OUTPUT_DIR/$DATASET_NAME+fixed.zip"
#unzip "$OUTPUT_DIR/$DATASET_NAME+fixed.zip" -d "$OUTPUT_DIR"
#rm -rf "$OUTPUT_DIR/$DATASET_NAME+fixed.zip"

#DATASET_NAME="SRD_Train"
#echo "$OUTPUT_DIR/$DATASET_NAME.zip"
#zip -F "$OUTPUT_DIR/$DATASET_NAME.zip" --out "$OUTPUT_DIR/$DATASET_NAME+fixed.zip"
#unzip "$OUTPUT_DIR/$DATASET_NAME+fixed.zip" -d "$OUTPUT_DIR"
#rm -rf "$OUTPUT_DIR/$DATASET_NAME+fixed.zip"

#DATASET_NAME="v87_istd_base/v87_istd"
#echo "$OUTPUT_DIR/$DATASET_NAME.zip"
#zip -F "$OUTPUT_DIR/$DATASET_NAME.zip" --out "$OUTPUT_DIR/$DATASET_NAME+fixed.zip"
#unzip "$OUTPUT_DIR/$DATASET_NAME+fixed.zip" -d "$OUTPUT_DIR"

#DATASET_NAME="v88_istd_base/v88_istd"
#echo "$OUTPUT_DIR/$DATASET_NAME.zip"
#zip -F "$OUTPUT_DIR/$DATASET_NAME.zip" --out "$OUTPUT_DIR/$DATASET_NAME+fixed.zip"
#unzip "$OUTPUT_DIR/$DATASET_NAME+fixed.zip" -d "$OUTPUT_DIR"

DATASET_NAME="Flickr2K"
echo "$OUTPUT_DIR/$DATASET_NAME.zip"
unzip -q "$OUTPUT_DIR/$DATASET_NAME.zip" -d "$OUTPUT_DIR"

#if [ $SERVER_CONFIG == 1 ]
#then
#  python "ccs1_main.py"
#elif [ $SERVER_CONFIG == 5 ]
#then
#  python3 "titan2_main.py"
#fi