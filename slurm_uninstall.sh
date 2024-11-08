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
conda create -n NeilZoo python=3.12
conda deactivate

# Installation of necessary libraries
module load anaconda
conda activate NeilZoo

#do fresh install
pip uninstall -I numpy==1.26.4
pip uninstall -I torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip uninstall scikit-learn
pip uninstall scikit-image
pip uninstall visdom
pip uninstall kornia
pip uninstall opencv-python
pip uninstall --upgrade pillow
pip uninstall gputil
pip uninstall matplotlib
pip uninstall --upgrade --no-cache-dir gdown
pip uninstall PyYAML
pip uninstall ITTR_pytorch
pip uninstall super_image
pip uninstall timm
pip uninstall datasets