# Full reproduction pipeline for TRACE.
#
# Typical use on a GPU-equipped machine:
#     make all                     # full sequence (data -> train -> explain -> tables -> figures)
#     make paper_numbers           # re-assemble results/paper_numbers.json after edits
#     make figures                 # regenerate figures from existing tables
#
# Individual targets below allow partial reruns.

PY ?= python
RESULTS_DIR := results
FIGURES_DIR := figures
DATASETS := ml-100k ml-1m steam beauty sports

.PHONY: all help data train explain_main explain_ablation aggregate tables stats ablation_tables backbone paper_numbers figures clean_figures check_env

help:
	@echo "TRACE reproducibility pipeline"
	@echo "  make all              - Full sequence (data -> train -> explain -> tables -> figures)"
	@echo "  make data             - Preprocess all five datasets"
	@echo "  make train            - Train all 15 (model, dataset) backbones"
	@echo "  make explain_main     - Run bring-in + push-out for all 15 combos, 4 methods, 3 seeds"
	@echo "  make explain_ablation - Run ablation variants on 3 representative combos"
	@echo "  make aggregate        - Aggregate per-seed JSON to summary JSON"
	@echo "  make tables           - Build paper_table_bringin.json + paper_table_pushout.json"
	@echo "  make stats            - Run statistical tests for bring-in + push-out"
	@echo "  make ablation_tables  - Aggregate ablation -> ablation_{bringin,pushout}.json"
	@echo "  make backbone         - Evaluate backbones -> backbone_eval.json"
	@echo "  make paper_numbers    - Assemble results/paper_numbers.json"
	@echo "  make figures          - Regenerate figures/ from pipeline artifacts"
	@echo "  make clean_figures    - Remove generated figures"

all: data train explain_main explain_ablation aggregate backbone tables stats ablation_tables paper_numbers figures

check_env:
	@command -v $(PY) >/dev/null 2>&1 || { echo "Python interpreter '$(PY)' not found"; exit 1; }
	@$(PY) -c "import torch, numpy, pandas, omegaconf, matplotlib" || { echo "Missing requirements — run: pip install -r requirements.txt"; exit 1; }

data: check_env
	@for ds in $(DATASETS); do \
		echo "[data] preprocess $$ds"; \
		$(PY) -m data.preprocess --dataset $$ds || exit 1; \
	done

train: check_env
	bash scripts/train_backbones.sh

explain_main: check_env
	bash scripts/reproduce_main.sh

explain_ablation: check_env
	bash scripts/reproduce_ablation.sh

aggregate: check_env
	$(PY) aggregate_summary.py --all --task bring_in --results-dir $(RESULTS_DIR)
	$(PY) aggregate_summary.py --all --task push_out --results-dir $(RESULTS_DIR)

tables: check_env aggregate
	$(PY) build_paper_table.py --task bring_in --results-dir $(RESULTS_DIR)
	$(PY) build_paper_table.py --task push_out --results-dir $(RESULTS_DIR)

stats: check_env
	$(PY) run_stat_tests.py --task bring_in
	$(PY) run_stat_tests.py --task push_out

ablation_tables: check_env
	$(PY) build_ablation_table.py --results-dir $(RESULTS_DIR)

backbone: check_env
	$(PY) eval_backbones.py

paper_numbers: check_env
	$(PY) build_paper_numbers.py --results-dir $(RESULTS_DIR)

figures: check_env
	$(PY) generate_figures.py --results-dir $(RESULTS_DIR) --figures-dir $(FIGURES_DIR)

clean_figures:
	rm -f $(FIGURES_DIR)/fig_trace_gece_greedy_triplet_combined.*
