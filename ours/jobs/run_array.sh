#!/bin/bash
#SBATCH --job-name=trace_grid
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out

# Usage: sbatch --array=1-54%8 ours/jobs/run_array.sh ours/jobs/list_ml-1m.txt
LIST=$1
module purge
module load 2023 Python/3.11.3-GCCcore-12.3.0
source ~/venvs/trace/bin/activate
cd ~/TRACE
export TQDM_DISABLE=1

ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
echo "[task $SLURM_ARRAY_TASK_ID] $ARGS"
python run_explain.py $ARGS
