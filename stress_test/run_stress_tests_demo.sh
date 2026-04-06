#!/usr/bin/env bash

set -euo pipefail

PYTHON_SCRIPT="/Users/SaraWang/Desktop/DS1012 LLM/Final Project/stress_test_variants.py"
DATASET_PATH="/Users/SaraWang/Desktop/DS1012 LLM/Final Project/dataset/hotpotqa/hotpot_dev_distractor_demo.json"
DATASET_NAME="hotpotqa"
OUTPUT_DIR="/Users/SaraWang/Desktop/DS1012 LLM/Final Project/generated_stress_tests/hotpotqa_demo"

NOISE_RATIOS=(0.1)
REDUNDANCY_RATIOS=(0.1)
POSITION_MODES=(shuffle_all support_back)
SEED=13

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  echo "Python script not found: $PYTHON_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "Dataset file not found: $DATASET_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

python3 "$PYTHON_SCRIPT" \
  --input "$DATASET_PATH" \
  --dataset "$DATASET_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --noise-ratios "${NOISE_RATIOS[@]}" \
  --redundancy-ratios "${REDUNDANCY_RATIOS[@]}" \
  --position-modes "${POSITION_MODES[@]}" \
  --seed "$SEED"

echo "Demo finished. Outputs written to: $OUTPUT_DIR"
