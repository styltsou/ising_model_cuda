#!/bin/bash
#SBATCH --job-name=nlm_gpu
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00
#SBATCH --output=out.stdout 


module load gcc
module load cuda/10.1.243

#nvidia-smi
make test
./test 3
make clean
