#!/bin/bash
#SBATCH --job-name=trace_chunk
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

# Usage: sbatch ours/jobs/run_chunk.sh <list> <first line> <last line>
LIST=$1; FIRST=$2; LAST=$3
module purge
module load 2023 Python/3.11.3-GCCcore-12.3.0
source ~/venvs/trace/bin/activate
cd ~/TRACE
export TQDM_DISABLE=1

for i in $(seq $FIRST $LAST); do
  ARGS=$(sed -n "${i}p" "$LIST")
  # safety net: push-out must use the paper's cutoff K=50
  if [[ $ARGS == *push_out* && $ARGS != *top_k=50* ]]; then
    ARGS="$ARGS targeted.top_k=50"
  fi
  echo "=== [line $i] $(date +%T) $ARGS"
  python run_explain.py $ARGS || echo "!!! line $i FAILED"
done
echo "=== done $(date +%T)"
