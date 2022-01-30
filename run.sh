#!/bin/bash
#SBATCH --job-name=main
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00
#SBATCH --output=main.stdout 


module load gcc
module load cuda/10.1.243

#nvidia-smi
make main
./main 3000 10
make clean

