#!/bin/bash
#SBATCH --job-name=vqvae_test
#SBATCH --output=jupyter.log
#SBATCH --error=jupyter.err
#SBATCH --account=eshiryaeva
#SBATCH --partition=main
cd /mnt/stud/home/eshiryaeva/dll24/notebooks/reproduce_training_ekaterina/gen_model_test/
srun python vqvae_script.py

