#!/bin/bash
#SBATCH --job-name=soccerstwos
#SBATCH -N 1 --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G 
#SBATCH --gres=gpu:v100:1
#SBATCH -t 12:00:00
#SBATCH --output=soccerstwos-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kagrawal74@gatech.edu


module load anaconda3
conda activate soccertwos

# CLEANUP: Kill any hanging ray processes and clear the temp folder
ray stop --force
rm -rf ~/scratch/ray_tmp/*

cd ~/scratch/DRL/soccer-twos-starter
python train_ray_selfplay.py