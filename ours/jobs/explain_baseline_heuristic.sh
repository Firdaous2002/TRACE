#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=exp_base_heuristic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=ours/logs/explain_baseline_heuristic_%A.out

module purge
module load 2023 Python/3.11.3-GCCcore-12.3.0
source ~/venvs/trace/bin/activate
cd ~/TRACE

python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'; print(torch.__version__, torch.cuda.get_device_name(0))"
python run_explain.py \
    model.name=sasrec \
    data.dataset=ml-1m \
    explainer.name=heuristic \
    targeted.enabled=true \
    targeted.constraint=bring_in \
    targeted.rank_min=2 \
    targeted.rank_max=50 \
    explainer.max_edits=5 \
    explainer.query_budget=10000 \
    explainer.forbid_constraint_items=false \
    evaluation.n_users=500 \
    evaluation.n_seeds=1 \
    seed=42

# python run_explain.py \
#     model.name=sasrec \
#     data.dataset=ml-1m \
#     explainer.name=heuristic \
#     targeted.enabled=true \
#     targeted.constraint=push_out \
#     targeted.rank_min=1 \
#     targeted.rank_max=1 \
#     explainer.max_edits=5 \
#     explainer.query_budget=10000 \
#     explainer.forbid_constraint_items=false \
#     evaluation.n_users=500 \
#     evaluation.n_seeds=1 \
#     seed=42

