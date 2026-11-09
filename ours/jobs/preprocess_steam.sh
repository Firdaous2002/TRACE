#!/bin/bash
#SBATCH --job-name=prep_steam
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

module purge
module load 2023 Python/3.11.3-GCCcore-12.3.0
source ~/venvs/trace/bin/activate
cd ~/TRACE
python -m data.preprocess --dataset steam
