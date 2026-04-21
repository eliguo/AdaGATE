#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="your project dir"
PYTHON_SCRIPT="$PROJECT_DIR/mixed_noise_variants.py"
DATASET_PATH="$PROJECT_DIR/dataset/hotpotqa/hotpot_dev_distractor_v1.json"
DATASET_NAME="hotpotqa"
OUTPUT_DIR="$PROJECT_DIR/generated_stress_tests/new_hotpotqa_mixed_noise"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/run_mixed_noise_demo_$(date '+%Y%m%d_%H%M%S').log"

NOISE_RATIOS=(0.5)
NOISE_SOURCE="syntax_random_partial"
SYNTAX_SOURCE="all_context"
SYNTAX_OPERATIONS=(delete swap distortion modification)
SYNTAX_EDIT_PASSES=3
SEED=13
PROGRESS_EVERY=1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  echo "Python script not found: $PYTHON_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "Dataset file not found: $DATASET_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
: > "$LOG_FILE"

log "Starting mixed noise generation."
log "Dataset: $DATASET_PATH"
log "Output directory: $OUTPUT_DIR"
log "Noise ratios: ${NOISE_RATIOS[*]}"
log "Noise source: $NOISE_SOURCE"
log "Syntax operations: ${SYNTAX_OPERATIONS[*]}"
log "Syntax edit passes: $SYNTAX_EDIT_PASSES"
log "Progress logging: every $PROGRESS_EVERY query"
log "Each progress line will show the current query number as: processing record X/Y, id=..."
log "Log file: $LOG_FILE"

python3 "$PYTHON_SCRIPT" \
  --input "$DATASET_PATH" \
  --dataset "$DATASET_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --noise-ratios "${NOISE_RATIOS[@]}" \
  --noise-source "$NOISE_SOURCE" \
  --syntax-source "$SYNTAX_SOURCE" \
  --syntax-operations "${SYNTAX_OPERATIONS[@]}" \
  --syntax-edit-passes "$SYNTAX_EDIT_PASSES" \
  --seed "$SEED" \
  --progress-every "$PROGRESS_EVERY" 2>&1 | tee -a "$LOG_FILE"

log "Mixed noise demo finished. Outputs written to: $OUTPUT_DIR"
