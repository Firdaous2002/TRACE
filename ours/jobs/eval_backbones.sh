#!/bin/bash
#SBATCH --job-name=eval_backbones
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

module purge
module load 2023 Python/3.11.3-GCCcore-12.3.0
source ~/venvs/trace/bin/activate
cd ~/TRACE
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'; print(torch.__version__, torch.cuda.get_device_name(0))"
python eval_backbones.py
