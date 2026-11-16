#!/bin/bash
# Usage: bash ours/jobs/make_list.sh <dataset>  -> writes ours/jobs/list_<dataset>.txt
DS=$1
OUT=ours/jobs/list_${DS}.txt
TRACE_ARGS="explainer.name=beam beam.beam_width=5 beam.k_context=6 beam.k_target_nn=3 beam.k_local_nn=3 beam.candidate_strategy=pacer_lite beam.position_strategy=influence beam.position_top_m=5 beam.pacer_shortlist_size=12"
: > "$OUT"
for model in sasrec bert4rec gru4rec; do
  for seed in 42 43 44; do
    for task in "bring_in 2 50" "push_out 1 1"; do
      read -r c rmin rmax <<< "$task"
      COMMON="model.name=$model data.dataset=$DS targeted.enabled=true targeted.constraint=$c targeted.rank_min=$rmin targeted.rank_max=$rmax $( [ "$c" = push_out ] && echo targeted.top_k=50 ) explainer.max_edits=5 explainer.query_budget=10000 evaluation.n_users=500 evaluation.n_seeds=1 seed=$seed"
      echo "$COMMON $TRACE_ARGS" >> "$OUT"
      echo "$COMMON explainer.name=ga" >> "$OUT"
      echo "$COMMON explainer.name=greedy" >> "$OUT"
    done
  done
done
echo "$(wc -l < "$OUT") runs written to $OUT"
