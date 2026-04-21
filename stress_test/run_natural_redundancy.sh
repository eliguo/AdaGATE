#!/bin/sh

set -eu

PROJECT_DIR="your project dir"
DATASET_PATH="$PROJECT_DIR/dataset/hotpotqa/hotpot_dev_distractor_v1.json"
DATASET_NAME="hotpotqa"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/run_natural_demo_$(date '+%Y%m%d_%H%M%S').log"

NATURAL_REDUNDANCY_SCRIPT="$PROJECT_DIR/natural_redundancy_variants.py"
NATURAL_REDUNDANCY_OUTPUT_DIR="$PROJECT_DIR/generated_stress_tests/hotpotqa_natural_redundancy"

REDUNDANCY_RATIOS="0.5"
SEED=13

mkdir -p "$LOG_DIR"
: > "$LOG_FILE"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

run_and_log() {
  log "Command: $*"
  "$@" >> "$LOG_FILE" 2>&1
}

log "Starting natural redundancy generation."
log "Dataset: $DATASET_PATH"
log "Dataset name: $DATASET_NAME"
log "Natural redundancy script: $NATURAL_REDUNDANCY_SCRIPT"
log "Natural redundancy output directory: $NATURAL_REDUNDANCY_OUTPUT_DIR"
log "Redundancy ratios: $REDUNDANCY_RATIOS"
log "Seed: $SEED"
log "Log file: $LOG_FILE"


if [ ! -f "$NATURAL_REDUNDANCY_SCRIPT" ]; then
  log "ERROR: Python script not found: $NATURAL_REDUNDANCY_SCRIPT"
  exit 1
fi

if [ ! -f "$DATASET_PATH" ]; then
  log "ERROR: Dataset file not found: $DATASET_PATH"
  exit 1
fi

mkdir -p "$NATURAL_REDUNDANCY_OUTPUT_DIR"

log "Generating natural redundancy variants..."
run_and_log python3 "$NATURAL_REDUNDANCY_SCRIPT" \
  --input "$DATASET_PATH" \
  --dataset "$DATASET_NAME" \
  --output-dir "$NATURAL_REDUNDANCY_OUTPUT_DIR" \
  --redundancy-ratios $REDUNDANCY_RATIOS \
  --seed "$SEED" \
  --progress-every 1
log "Finished natural redundancy generation."

log "Natural redundancy outputs written to: $NATURAL_REDUNDANCY_OUTPUT_DIR"
log "Generation complete."
