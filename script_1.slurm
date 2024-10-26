#!/bin/bash
#SBATCH -J SR_1
#SBATCH --partition=gpu
#SBATCH --qos=12c-1h_2gpu
#SBATCH --cpus-per-task=6
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH --output=script_1.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

NETWORK_VERSION=$1
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda
module load cuda
source activate NeilZoo

srun python train_sr_main.py --server_config=0 --img_to_load=-1 \
--plot_enabled=0 --save_per_iter=500 --network_version=$NETWORK_VERSION

conda deactivate