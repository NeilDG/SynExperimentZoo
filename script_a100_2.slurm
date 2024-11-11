#!/bin/bash
#SBATCH -J A100_2
#SBATCH --partition=gpu_a100
#SBATCH --qos=gpu-a100_default
#SBATCH --cpus-per-task=8
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH --output=script_4.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

NETWORK_VERSION=$1
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda
module load cuda
source activate NeilZoo

srun python train_sr_main.py --server_config=8 --img_to_load=-1 \
--plot_enabled=0 --save_per_iter=500 --network_version=$NETWORK_VERSION

conda deactivate