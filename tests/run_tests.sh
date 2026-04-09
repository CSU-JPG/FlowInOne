#!/bin/bash
set -euo pipefail

##############################################################################
# FlowInOne Test Runner
#
# Runs all (or filtered) test cases from test_configs.json.
# Prepares visual prompt inputs, then runs inference in two passes:
#   1) Editing tasks (cross-attention enabled)
#   2) Text2Image tasks (cross-attention skipped)
#
# Usage:
#   bash tests/run_tests.sh                         # Run all tests, default settings
#   bash tests/run_tests.sh --fast                   # Quick run (25 steps)
#   bash tests/run_tests.sh --cfg 9.0 --steps 75    # Custom CFG and steps
#   bash tests/run_tests.sh --filter "carnival"      # Only tests matching pattern
#   bash tests/run_tests.sh --prepare-only           # Only generate visual prompts
#
# Environment variables (override defaults):
#   FLOWINONE_CHECKPOINT  Path to flowinone_256px.pth
#   JANUS_MODEL_PATH      Path to Janus-Pro-1B model directory
#   GPU_ID                GPU device ID (default: 0)
#
# Optimization tips:
#   --fast               Sets SAMPLE_STEPS=25 (~2x faster, slight quality loss)
#   --steps 25|50|75|100 Control quality/speed tradeoff
#   --cfg 3-5            More creative/diverse outputs
#   --cfg 9-12           More prompt-adherent (risk of artifacts)
#   --batch-size 4       Better GPU utilization (needs more VRAM)
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults (overridable via env vars)
CONFIG_FILE="${FLOWINONE_CONFIG:-${REPO_ROOT}/configs/flowinone_training_demo.py}"
NNET_PATH="${FLOWINONE_CHECKPOINT:-${REPO_ROOT}/checkpoints/flowinone_256px.pth}"
JANUS_PATH="${JANUS_MODEL_PATH:-}"
GPU_ID="${GPU_ID:-0}"

# Inference defaults
CFG_SCALE=7.0
SAMPLE_STEPS=50
BATCH_SIZE=1
FILTER_PATTERN=""
PREPARE_ONLY=false
TEST_CONFIG="${REPO_ROOT}/tests/test_configs.json"
PREPARED_DIR="${REPO_ROOT}/tests/prepared_inputs"
OUTPUT_DIR=""

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cfg)
            CFG_SCALE="$2"; shift 2 ;;
        --steps)
            SAMPLE_STEPS="$2"; shift 2 ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        --gpu)
            GPU_ID="$2"; shift 2 ;;
        --fast)
            SAMPLE_STEPS=25; shift ;;
        --filter)
            FILTER_PATTERN="$2"; shift 2 ;;
        --output)
            OUTPUT_DIR="$2"; shift 2 ;;
        --checkpoint)
            NNET_PATH="$2"; shift 2 ;;
        --janus-path)
            JANUS_PATH="$2"; shift 2 ;;
        --config)
            TEST_CONFIG="$2"; shift 2 ;;
        --prepare-only)
            PREPARE_ONLY=true; shift ;;
        -h|--help)
            head -35 "$0" | tail -30
            exit 0 ;;
        *)
            echo "Unknown option: $1"
            exit 1 ;;
    esac
done

# Generate timestamp-based output directory if not specified
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${REPO_ROOT}/tests/results/$(date +%Y%m%d_%H%M%S)"
fi

echo "=========================================="
echo "  FlowInOne Test Runner"
echo "=========================================="
echo "Config:       ${TEST_CONFIG}"
echo "Checkpoint:   ${NNET_PATH}"
echo "CFG Scale:    ${CFG_SCALE}"
echo "Steps:        ${SAMPLE_STEPS}"
echo "Batch Size:   ${BATCH_SIZE}"
echo "GPU:          ${GPU_ID}"
echo "Output:       ${OUTPUT_DIR}"
if [[ -n "$FILTER_PATTERN" ]]; then
    echo "Filter:       ${FILTER_PATTERN}"
fi
echo "=========================================="

# Validate prerequisites
if [[ ! -f "$TEST_CONFIG" ]]; then
    echo "ERROR: Test config not found: ${TEST_CONFIG}"
    exit 1
fi

if [[ "$PREPARE_ONLY" = false ]]; then
    if [[ ! -f "$NNET_PATH" ]]; then
        echo "ERROR: Checkpoint not found: ${NNET_PATH}"
        echo ""
        echo "Download it with:"
        echo "  mkdir -p checkpoints"
        echo "  wget -O checkpoints/flowinone_256px.pth \\"
        echo "    https://huggingface.co/CSU-JPG/FlowInOne/resolve/main/flowinone_256px.pth"
        exit 1
    fi

    # Check if Janus path is configured
    if [[ -z "$JANUS_PATH" ]]; then
        # Try to read from test_configs.json
        JANUS_PATH=$(python3 -c "
import json
with open('${TEST_CONFIG}') as f:
    print(json.load(f)['global_settings'].get('janus_model_path', ''))
" 2>/dev/null || echo "")
    fi

    # Verify Janus path in i2i.py
    CURRENT_JANUS=$(grep 'model_path = ' "${REPO_ROOT}/i2i.py" | head -1 | sed 's/.*= *"\(.*\)"/\1/')
    if [[ "$CURRENT_JANUS" == "/path/to/Janus-Pro-1B/" ]]; then
        if [[ -n "$JANUS_PATH" ]]; then
            echo ""
            echo "NOTE: JANUS_MODEL_PATH is set to: ${JANUS_PATH}"
            echo "      The env var will be used by the patched i2i.py."
            echo "      If inference fails, update i2i.py line 117 manually."
        else
            echo ""
            echo "ERROR: Janus-Pro-1B model path not configured!"
            echo ""
            echo "Either:"
            echo "  1) Set env var:  export JANUS_MODEL_PATH=/path/to/Janus-Pro-1B"
            echo "  2) Edit i2i.py line 117:  model_path = \"/your/path/to/Janus-Pro-1B/\""
            echo ""
            echo "Download preparation files with:"
            echo "  wget -O preparation.tar.gz \\"
            echo "    https://huggingface.co/CSU-JPG/FlowInOne/resolve/main/preparation.tar.gz"
            echo "  tar -xzf preparation.tar.gz -C preparation/"
            exit 1
        fi
    fi
fi

# ---- Step 1: Prepare visual prompts ----
echo ""
echo "Step 1: Preparing visual prompt input images..."

PREPARE_ARGS="--config ${TEST_CONFIG} --output-dir ${PREPARED_DIR}"
if [[ -n "$FILTER_PATTERN" ]]; then
    # Extract matching test IDs
    MATCHING_IDS=$(python3 -c "
import json
with open('${TEST_CONFIG}') as f:
    cases = json.load(f)['test_cases']
ids = [c['id'] for c in cases if '${FILTER_PATTERN}' in c['id']]
print(','.join(ids))
" 2>/dev/null || echo "")
    if [[ -z "$MATCHING_IDS" ]]; then
        echo "ERROR: No test cases match filter pattern '${FILTER_PATTERN}'"
        exit 1
    fi
    PREPARE_ARGS="${PREPARE_ARGS} --test-ids ${MATCHING_IDS}"
    echo "  Filtered to: ${MATCHING_IDS}"
fi

python3 "${REPO_ROOT}/tests/prepare_visual_prompts.py" ${PREPARE_ARGS}

if [[ "$PREPARE_ONLY" = true ]]; then
    echo ""
    echo "Preparation complete (--prepare-only). Inspect images in:"
    echo "  ${PREPARED_DIR}/editing/"
    echo "  ${PREPARED_DIR}/text2image/"
    exit 0
fi

# ---- Step 2: Run inference ----
mkdir -p "${OUTPUT_DIR}"

EDITING_DIR="${PREPARED_DIR}/editing"
T2I_DIR="${PREPARED_DIR}/text2image"

# Run editing tasks (cross-attention enabled)
if [[ -d "$EDITING_DIR" ]] && [[ "$(ls -A "$EDITING_DIR" 2>/dev/null | head -1)" ]]; then
    EDITING_COUNT=$(ls "$EDITING_DIR"/*.png 2>/dev/null | wc -l)
    echo ""
    echo "Step 2a: Running inference for ${EDITING_COUNT} editing tasks..."
    echo "  skip_cross_atten=false"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 "${REPO_ROOT}/i2i.py" \
        --config="${CONFIG_FILE}" \
        --config.sample.sample_steps=${SAMPLE_STEPS} \
        --nnet_path="${NNET_PATH}" \
        --input_image_path="${EDITING_DIR}" \
        --output_image_path="${OUTPUT_DIR}/editing" \
        --output_path="${OUTPUT_DIR}/editing.log" \
        --cfg="${CFG_SCALE}" \
        --batch_size="${BATCH_SIZE}" \
        --skip_cross_atten=false
else
    echo ""
    echo "Step 2a: No editing tasks to run."
fi

# Run text2image tasks (cross-attention skipped)
if [[ -d "$T2I_DIR" ]] && [[ "$(ls -A "$T2I_DIR" 2>/dev/null | head -1)" ]]; then
    T2I_COUNT=$(ls "$T2I_DIR"/*.png 2>/dev/null | wc -l)
    echo ""
    echo "Step 2b: Running inference for ${T2I_COUNT} text2image tasks..."
    echo "  skip_cross_atten=true"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 "${REPO_ROOT}/i2i.py" \
        --config="${CONFIG_FILE}" \
        --config.sample.sample_steps=${SAMPLE_STEPS} \
        --nnet_path="${NNET_PATH}" \
        --input_image_path="${T2I_DIR}" \
        --output_image_path="${OUTPUT_DIR}/text2image" \
        --output_path="${OUTPUT_DIR}/text2image.log" \
        --cfg="${CFG_SCALE}" \
        --batch_size="${BATCH_SIZE}" \
        --skip_cross_atten=true
else
    echo ""
    echo "Step 2b: No text2image tasks to run."
fi

# ---- Step 3: Summary ----
echo ""
echo "=========================================="
echo "  Test run complete!"
echo "=========================================="
EDITING_RESULTS=$(ls "${OUTPUT_DIR}/editing/"*.png 2>/dev/null | wc -l)
T2I_RESULTS=$(ls "${OUTPUT_DIR}/text2image/"*.png 2>/dev/null | wc -l)
echo "Editing results:    ${EDITING_RESULTS} images in ${OUTPUT_DIR}/editing/"
echo "Text2Image results: ${T2I_RESULTS} images in ${OUTPUT_DIR}/text2image/"
echo "Logs:               ${OUTPUT_DIR}/*.log"
echo ""
echo "Settings used: CFG=${CFG_SCALE}, Steps=${SAMPLE_STEPS}, BatchSize=${BATCH_SIZE}"
echo "=========================================="
