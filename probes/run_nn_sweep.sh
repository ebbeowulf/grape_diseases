#!/usr/bin/env bash
# Sweep the linear probe across diseases and modality combinations.
#
# Every run appends one row to the results CSV and writes its full report to
# the log directory. Failures are logged and skipped rather than stopping the
# sweep -- a disease with too few boxes should not cost you the rest of the
# night.
#
# Split across two GPUs with MODE, which partitions the work exactly:
#   CUDA_VISIBLE_DEVICES=0 ./run_probe_sweep.sh single sweep_single.csv logs_single 0
#   CUDA_VISIBLE_DEVICES=1 ./run_probe_sweep.sh multi sweep_multi.csv logs_multi 0
#
# Separate results files matter: two processes appending to one CSV would
# each write a header.
#
set -u

# TRAIN="/data3/datasets/shady_lanes_10_10_25/grape_disease_scripts/datasets/grape_diseases.train.txt"
TRAIN="/data3/datasets/shady_lanes_10_10_25/grape_disease_scripts/datasets/grape_diseases.train_with_tracking.txt"
EVAL="/data3/datasets/shady_lanes_10_10_25/grape_disease_scripts/datasets/grape_diseases.test.txt"
CALIB="/data3/datasets/shady_lanes_10_10_25/grape_disease_scripts/alignment/thermal_color_matches_calib.json"
PROBE="leaf_nn_classifier.py"

if [ $# -ne 4 ]; then
  echo "Usage: $0 MODE RESULTS LOGDIR EARLY" >&2
  echo "  MODE     single, multi or all -- which modality combinations to run" >&2
  echo "  RESULTS  CSV the probe appends one row per run to" >&2
  echo "  LOGDIR   directory holding the full text report for each run" >&2
  echo "  EARLY    1 to also run early fusion on multi-modal combinations" >&2
  echo >&2
  echo "Example, splitting across two GPUs:" >&2
  echo "  CUDA_VISIBLE_DEVICES=0 $0 single sweep_single.csv logs_single 0" >&2
  echo "  CUDA_VISIBLE_DEVICES=1 $0 multi sweep_multi.csv logs_multi 0" >&2
  exit 1
fi

MODE="$1"
RESULTS="$2"
LOGDIR="$3"
UNFREEZE="$4"

# Each entry is one probe target. Space-separated names are merged into a
# single positive class, as esca and red_blotch were.
DISEASES=(
  "esca red_blotch"
  "esca"
  "red_blotch"
  "black_rot"
  "downy_mildew"
  "powdery_mildew"
  "phylloxera"
)

SINGLE=("color" "thermal" "depth")
MULTI=("color thermal" "color depth" "thermal depth" "color thermal depth")

case "$MODE" in
  single) COMBOS=("${SINGLE[@]}") ;;
  multi)  COMBOS=("${MULTI[@]}") ;;
  all)    COMBOS=("${SINGLE[@]}" "${MULTI[@]}") ;;
  *)      echo "MODE must be single, multi or all" >&2; exit 1 ;;
esac

mkdir -p "$LOGDIR"
START=$(date +%s)
TOTAL=0
FAILED=0

run_one() {
  local disease="$1" modalities="$2" unfreeze="$3"
  local tag
  tag=$(echo "${disease}__${modalities}" | tr ' ' '-')
  local log="$LOGDIR/${tag}.txt"

  local cmd="python $PROBE --train-samples \"$TRAIN\" --eval-samples \"$EVAL\" --calibration \"$CALIB\" --positive-classes $disease --negative-classes \"healthy leaf\" \"unhealthy leaf\" --modalities $modalities --unfreeze \"$unfreeze\" --results \"$RESULTS\" > \"$log\" "

  TOTAL=$((TOTAL + 1))
  echo $cmd

  eval "$cmd"
  STATUS=$?

  if [ "$STATUS" -ne 0 ]; then
    FAILED=$((FAILED + 1))
    echo "    FAILED (exit $STATUS) -- see $log"
    tail -n 3 "$log" | sed 's/^/    /'
  fi
}

echo "mode=$MODE  results=$RESULTS  logdir=$LOGDIR"

for disease in "${DISEASES[@]}"; do
  for modalities in "${COMBOS[@]}"; do
    run_one "$disease" "$modalities" "$UNFREEZE"
  done
done

ELAPSED=$(( $(date +%s) - START ))
printf '\n%d runs, %d failed, %dm%02ds\n' \
  "$TOTAL" "$FAILED" $((ELAPSED / 60)) $((ELAPSED % 60))
echo "metrics: $RESULTS"
echo "reports: $LOGDIR/"