# TRACE: Targeted Ranking-Aware Counterfactual Explanation for Sequential Recommendation

[![ACM RecSys 2026](https://img.shields.io/badge/ACM%20RecSys-2026-0085CA)](https://recsys.acm.org/recsys26/)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3773078.3831796-blue)](https://doi.org/10.1145/3773078.3831796)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **TRACE**, published at the
**20th ACM Conference on Recommender Systems (RecSys '26)**.

**Authors:** [Ungsik Kim](https://orcid.org/0009-0001-3648-8437),
[Sang-Min Choi](https://orcid.org/0000-0001-5950-3081),
[Gun-Woo Kim](https://orcid.org/0000-0001-5643-4797), and
[Suwon Lee](https://orcid.org/0000-0003-2603-1385) (corresponding author)

Gyeongsang National University, Jinju-si, Republic of Korea

TRACE generates targeted, ranking-aware counterfactual explanations for
sequential recommenders. The repository includes the complete experimental
pipeline, pretrained checkpoints for all 15 model–dataset combinations, cached
paper results, statistical tests, and the figures used in the paper.

## At a glance

| Component | Included |
|---|---|
| Sequential recommenders | SASRec, BERT4Rec, GRU4Rec |
| Datasets | MovieLens-100K, MovieLens-1M, Steam, Amazon Beauty, Amazon Sports |
| Explanation methods | TRACE, GECE, Greedy, Random, gradient-based baseline |
| Target constraints | Bring-in and push-out |
| Evaluation | Validity, Hamming distance, query count, RSS, and ΔNLL |
| Reproducibility assets | 15 checkpoints, cached results, paper figures, executed notebook |

## View the paper results

Open [`reproduce.ipynb`](reproduce.ipynb) to inspect every reported table,
figure, case study, and significance test. The notebook contains saved outputs,
so GitHub renders the results without a GPU, dataset download, or local setup.

To execute the notebook again from a clean environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pandas matplotlib jupyter
jupyter nbconvert --to notebook --execute reproduce.ipynb --inplace
```

The notebook reads the committed snapshot in
[`results/paper_numbers.json`](results/paper_numbers.json) and the rendered
assets in [`figures/`](figures/).

## Installation

The full experiment pipeline requires Python 3.10 or newer. A CUDA-capable GPU
is strongly recommended for training and explanation runs.

```bash
git clone https://github.com/devUuung/TRACE.git
cd TRACE

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

make check_env
```

The paper experiments used PyTorch 2.10.0 with CUDA 12.8 on one NVIDIA RTX
5090. The declared dependencies support PyTorch 2.x and are not tied to that
exact hardware configuration.

## Reproduce the experiments

### 1. Download and preprocess the datasets

```bash
python -m data.preprocess --dataset ml-100k
python -m data.preprocess --dataset ml-1m
python -m data.preprocess --dataset steam
python -m data.preprocess --dataset beauty
python -m data.preprocess --dataset sports
```

Preprocessing applies iterative 5-core filtering, chronological ordering,
contiguous ID mapping, and a leave-two-out validation/test split. Raw and
processed data are stored under `data/` and are excluded from Git.

### 2. Use or retrain the recommender checkpoints

All 15 pretrained checkpoints are committed under [`checkpoints/`](checkpoints/),
so retraining is optional.

```bash
# Optional: retrain SASRec, BERT4Rec, and GRU4Rec on all five datasets.
bash scripts/train_backbones.sh

# Re-evaluate the committed checkpoints (paper Table 1).
python eval_backbones.py
```

### 3. Run the main explanation experiments

```bash
# TRACE, GECE, Greedy, and Random; bring-in and push-out; seeds 42, 43, 44.
bash scripts/reproduce_main.sh

# Optional cumulative ablation.
bash scripts/reproduce_ablation.sh
```

The complete main grid takes approximately 10 hours on one RTX 5090.
Retraining all backbones adds approximately 4 hours.

For a smaller run, override the script defaults:

```bash
N_USERS=50 Q_BUDGET=1000 K_MAX=3 bash scripts/reproduce_main.sh
```

To run one configuration directly:

```bash
python run_explain.py \
  model.name=sasrec \
  data.dataset=ml-1m \
  explainer.name=beam \
  targeted.enabled=true \
  targeted.constraint=bring_in \
  targeted.rank_min=2 \
  targeted.rank_max=50 \
  evaluation.n_users=50 \
  evaluation.n_seeds=1
```

### 4. Rebuild the paper artifacts

```bash
python build_paper_table.py
python run_stat_tests.py
python build_paper_numbers.py
python generate_figures.py
```

The same workflow is exposed through the Makefile:

```bash
make help
make data
make train
make explain_main
make explain_ablation
make backbone
make tables
make stats
make paper_numbers
make figures

# Full pipeline.
make all
```

## Configuration

[`config/default.yaml`](config/default.yaml) contains the complete experiment
configuration. Command-line overrides use OmegaConf dot-list syntax:

```bash
python run_train.py model.name=bert4rec data.dataset=beauty
python run_explain.py model.name=gru4rec data.dataset=steam explainer.name=ga
```

The reported TRACE setup uses a query budget of 10,000, at most five edits, a
beam width of five, six context neighbors, three target neighbors, three local
neighbors, and a shortlist size of 12 for bring-in. Stochastic experiments use
seeds 42, 43, and 44.

Internally, TRACE is implemented by
`explainers.beam_explainer.BeamSearchExplainer` with
`candidate_strategy=pacer_lite` and `position_strategy=influence`. The legacy
configuration tag `pacer` refers to the same method.

## Repository structure

```text
reproduce.ipynb            Executed notebook containing every paper result
config/default.yaml        Global, model-specific, and dataset-specific settings
data/                      Dataset download, preprocessing, and data loaders
models/                    SASRec, BERT4Rec, GRU4Rec, and training utilities
explainers/                TRACE and all comparison methods
evaluation/                Metrics and evaluation pipeline
checkpoints/               Pretrained models for all 15 combinations
results/                   Committed paper numbers and backbone evaluation
figures/                   Paper figures in PDF and PNG formats
scripts/                   End-to-end training, main-run, and ablation scripts
run_train.py               Train one recommender configuration
run_explain.py             Generate counterfactual explanations
eval_backbones.py          Recompute HR@10 and NDCG@10
run_stat_tests.py          Run paired significance tests
build_paper_numbers.py     Assemble the notebook's result snapshot
generate_figures.py        Regenerate paper figures
```

## Reporting details

- Validity, Queries, and Hamming distance are computed over the full evaluation
  user set.
- ΔNLL is a conditional average over users for whom at least one valid
  counterfactual is found, matching the paper definition.
- The GRU4Rec gradient-based explanation path temporarily enables the cuDNN
  training mode required for its backward pass; this is handled in
  `explainers/gradient_explainer.py`.
- Dataset files are downloaded from their original providers. Review and comply
  with each dataset's terms before use; the datasets themselves are not
  redistributed in this repository.

## Citation

If this repository is useful in your research, please cite:

```bibtex
@inproceedings{kim2026trace,
  author    = {Ungsik Kim and Sang-Min Choi and Gun-Woo Kim and Suwon Lee},
  title     = {{TRACE}: Targeted Ranking-Aware Counterfactual Explanation for
               Sequential Recommendation},
  booktitle = {Proceedings of the 20th ACM Conference on Recommender Systems},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3773078.3831796},
  isbn      = {979-8-4007-2284-4/2026/09}
}
```

The conference takes place September 27–October 2, 2026, in Minneapolis,
Minnesota, USA. The paper is available at
[https://doi.org/10.1145/3773078.3831796](https://doi.org/10.1145/3773078.3831796).

## License

The source code is released under the [MIT License](LICENSE). The datasets
remain subject to their respective providers' licenses and terms.
