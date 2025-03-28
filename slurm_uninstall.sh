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
pip uninstall --yes numpy
pip uninstall --yes torch
pip uninstall --yes torchvision
pip uninstall --yes torchaudio
pip uninstall --yes scikit-learn
pip uninstall --yes scikit-image
pip uninstall --yes visdom
pip uninstall --yes kornia
pip uninstall --yes opencv-python
pip uninstall --yes pillow
pip uninstall --yes gputil
pip uninstall --yes matplotlib
pip uninstall --yes gdown
pip uninstall --yes PyYAML
pip uninstall --yes ITTR_pytorch
pip uninstall --yes super_image
pip uninstall --yes timm
pip uninstall --yes datasets