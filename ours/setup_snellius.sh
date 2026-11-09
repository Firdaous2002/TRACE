#!/bin/bash
# One-time Snellius setup for the TRACE reproduction (run on the login node).
set -e
module purge
module load 2023 Python/3.11.3-GCCcore-12.3.0

[ -d ~/venvs/trace ] || python -m venv ~/venvs/trace
source ~/venvs/trace/bin/activate
pip install --upgrade pip
cd ~/TRACE
pip install -r requirements.txt
pip uninstall -y torch triton
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

for ds in ml-100k ml-1m beauty sports; do
  python -m data.preprocess --dataset "$ds"
done
mkdir -p logs
python -c "import torch; print('torch', torch.__version__)"
echo "Setup done. Steam: download on login node, then sbatch ours/jobs/preprocess_steam.sh"
