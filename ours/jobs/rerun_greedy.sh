#!/bin/bash
#SBATCH --job-name=rerun_greedy
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=06:0:00
#SBATCH --output=logs/%x_%j.out

module purge
module load 2023 Python/3.11.3-GCCcore-12.3.0
source ~/venvs/trace/bin/activate
cd ~/TRACE
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'; print(torch.__version__, torch.cuda.get_device_name(0))"

FIX="${FIX:-false}"
SEED="${SEED:-42}"
if [ "$FIX" = "true" ]; then OUT=results_fix; else OUT=results_orig; fi
echo "FIX=$FIX SEED=$SEED -> $OUT"

SET="data.dataset=ml-100k explainer.name=greedy targeted.enabled=true explainer.max_edits=5 explainer.query_budget=10000 evaluation.n_users=500 evaluation.n_seeds=1 seed=$SEED explainer.forbid_constraint_items=$FIX evaluation.results_dir=$OUT"
BRING="targeted.constraint=bring_in targeted.rank_min=2 targeted.rank_max=50"
PUSH="targeted.constraint=push_out targeted.rank_min=1 targeted.rank_max=1 targeted.top_k=50"

python run_explain.py model.name=sasrec   $SET $BRING
python run_explain.py model.name=sasrec   $SET $PUSH
python run_explain.py model.name=bert4rec $SET $BRING
python run_explain.py model.name=bert4rec $SET $PUSH
python run_explain.py model.name=gru4rec  $SET $BRING
python run_explain.py model.name=gru4rec  $SET $PUSH