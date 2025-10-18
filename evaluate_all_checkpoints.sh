#!/usr/bin/env bash
set -euo pipefail

PREFIX="runs/combined_experiment"
BATCH=16

PYTHON="python"
COMMON_FLAGS="PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=true HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1"

# 1536 checkpoints
echo "=== Evaluating 1536 checkpoints ==="
/usr/bin/env bash -c "${COMMON_FLAGS} ${PYTHON} scripts/eval_checkpoints.py \
  --checkpoint ${PREFIX}/checkpoint-396 \
  --checkpoint ${PREFIX}/checkpoint-792 \
  --checkpoint ${PREFIX}/checkpoint-1188 \
  --checkpoint ${PREFIX}/checkpoint-1584 \
  --checkpoint ${PREFIX}/checkpoint-1980 \
  --seq-len 1536 \
  --batch-size ${BATCH} \
  --output ${PREFIX}/eval_1536.jsonl" | tee ${PREFIX}/eval_1536.log

# 2048 checkpoints
echo "=== Evaluating 2048 checkpoints ==="
/usr/bin/env bash -c "${COMMON_FLAGS} ${PYTHON} scripts/eval_checkpoints.py \
  --checkpoint ${PREFIX}/checkpoint-410 \
  --checkpoint ${PREFIX}/checkpoint-820 \
  --checkpoint ${PREFIX}/checkpoint-1230 \
  --checkpoint ${PREFIX}/checkpoint-1640 \
  --checkpoint ${PREFIX}/checkpoint-2050 \
  --seq-len 2048 \
  --batch-size ${BATCH} \
  --output ${PREFIX}/eval_2048.jsonl" | tee ${PREFIX}/eval_2048.log

