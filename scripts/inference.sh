#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/i2i.py" ]]; then
  REPO_ROOT="$SCRIPT_DIR"
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

CONFIG_FILE="${REPO_ROOT}/configs/flowinone_training_demo.py"
NNET_PATH="${REPO_ROOT}/checkpoints/flowinone_256px.pth"

INPUT_IMAGE="${REPO_ROOT}/imgs/input"
OUTPUT_IMAGE="${REPO_ROOT}/imgs/output"
LOG_PATH="${OUTPUT_IMAGE}/inference.log"

# Hyperparameters
# CFG scale
CFG_SCALE=7.0
# Sampling steps
SAMPLE_STEPS=50
# Whether to skip cross attention for all images (true/false)
SKIP_CROSS_ATTEN=false
# Batch size
BATCH_SIZE=1

GPU_ID=0
echo "=========================================="
echo "Repo root: ${REPO_ROOT}"
echo "Config: ${CONFIG_FILE}"
echo "Checkpoint: ${NNET_PATH}"
echo "Input path: ${INPUT_IMAGE}"
echo "Output path: ${OUTPUT_IMAGE}"
echo "CFG: ${CFG_SCALE} | Steps: ${SAMPLE_STEPS} | Batch: ${BATCH_SIZE}"
echo "Skip cross attention: ${SKIP_CROSS_ATTEN}"
echo "==========================================="

mkdir -p "${OUTPUT_IMAGE}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python "${REPO_ROOT}/i2i.py" \
  --config=${CONFIG_FILE} \
  --config.sample.sample_steps=${SAMPLE_STEPS} \
  --nnet_path=${NNET_PATH} \
  --input_image_path=${INPUT_IMAGE} \
  --output_image_path=${OUTPUT_IMAGE} \
  --output_path=${LOG_PATH} \
  --cfg=${CFG_SCALE} \
  --batch_size=${BATCH_SIZE} \
  --skip_cross_atten=${SKIP_CROSS_ATTEN}

echo ""
echo "=========================================="
echo "Inference completed! Output directory: ${OUTPUT_IMAGE}"
echo "=========================================="
