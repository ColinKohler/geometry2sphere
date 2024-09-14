#!/bin/bash

#SBATCH --output=scripts/outputs/%j.out
#SBATCH --error=scripts/outputs/%j.err

#SBATCH --nodes=1

#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kohler.c@northeastern.edu

conda activate torch

echo "node: $SLURM_NODELIST"
python scripts/train.py
