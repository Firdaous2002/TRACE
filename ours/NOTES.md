# Setup notes (Snellius)

- Environment: Python 3.11.3 (module 2023), PyTorch 2.10.0+cu128 (same as authors), A100-SXM4-40GB / MIG.
- Missing dependency: models/trainer.py imports adam_mini, not listed in requirements.txt (added).
- Preprocessing reproduces paper statistics exactly for ML-100K, ML-1M, Beauty, Sports.
- Steam: code gives 281,210 users / 11,961 items; paper reports 334,536 / 13,046.
  Released Steam checkpoints have 11,961 items, so code and checkpoints are consistent;
  the paper's Steam statistics are a reporting error.
- Steam preprocessing: download raw files on the login node, then sbatch ours/jobs/preprocess_steam.sh
  (~7 min, 60 GB RAM).
- Table 1 reproduced: 14/15 combos match to 4 decimals (A100, torch 2.10+cu128).
  BERT4Rec/Steam: ours HR@10 .1064 / NDCG@10 .0547 = authors' repo files;
  the paper PDF reports .1249 / .0662 -> reporting error in the PDF.
