#!/bin/bash
#SBATCH -J INSTALL
#SBATCH --partition=serial
#SBATCH --qos=84c-1d_serial
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --output=script_install.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

#About this script:

# Remove previous conda environment and install a new one
module load anaconda
conda remove -y --name NeilZoo --all
conda create -n NeilZoo python=3.14
conda deactivate

# Installation of necessary libraries
module load anaconda
conda activate NeilZoo

#do fresh install
python3 -m pip install --upgrade pip
pip-review --local --auto
pip install -I numpy==1.26.4
pip install -I torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install scikit-learn
pip install scikit-image
pip install visdom
pip install kornia
pip install opencv-python
pip install --upgrade pillow
pip install gputil
pip install matplotlib==3.9.0
pip install --upgrade --no-cache-dir gdown
pip install PyYAML
pip install ITTR_pytorch
pip install super_image
pip install timm
pip install datasets

python "util_script_main.py"
python "g411_5090-pc3_main.py"