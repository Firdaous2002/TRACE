#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=exp_base_heuristic_bring_in
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=ours/logs/explain_baseline_bring_in_heuristic_%A.out

module purge
module load 2023 Python/3.11.3-GCCcore-12.3.0
source ~/venvs/trace/bin/activate
cd ~/TRACE

models=(sasrec bert4rec gru4rec)
datasets=(ml-100k ml-1m steam beauty sports)

python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'; print(torch.__version__, torch.cuda.get_device_name(0))"

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "=========================================="
        echo "Model:   ${model}"
        echo "Dataset: ${dataset}"
        echo "=========================================="

        python run_explain.py \
            model.name="${model}" \
            data.dataset="${dataset}" \
            explainer.name=heuristic \
            targeted.enabled=true \
            targeted.constraint=bring_in \
            targeted.rank_min=2 \
            targeted.rank_max=50 \
            explainer.max_edits=5 \
            explainer.query_budget=10000 \
            explainer.forbid_constraint_items=true \
            evaluation.n_users=500 \
            evaluation.n_seeds=1 \
            seed=42 \
            targeted.top_k=1
    done
done