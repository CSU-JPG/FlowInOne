#!/bin/bash
set -x

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SEED_MODELS_LOGGING_LEVEL=WARN
export TOKENIZERS_PARALLELISM=false
export VESCALE_SINGLE_DEVICE_RAND=0
export TF_CPP_MIN_LOG_LEVEL=2

NNODES=1
NPROC_PER_NODE=8
NPROC=$((NNODES * NPROC_PER_NODE))
MASTER_PORT=${MASTER_PORT:=29500}

# add wandb api key
WANDB_API_KEY=""

# login wandb
echo "Logging in to wandb..."
export WANDB_API_KEY=$WANDB_API_KEY
wandb login --relogin $WANDB_API_KEY
if [ $? -ne 0 ]; then
    echo "Error: Failed to login to wandb. Exiting..."
    exit 1
fi
echo "Successfully logged in to wandb"

echo $NPROC
echo $MASTER_PORT

cd path/to/FlowInOne
accelerate launch \
    --main_process_port $MASTER_PORT \
    --multi_gpu \
    --num_processes $NPROC \
    --num_machines $NNODES \
    --mixed_precision bf16 \
    run.py \
    --config=configs/flowinone_training_demo.py \
    --workdir_base="path/to/workdir" \
    --vae_pretrained_path="path/to/autoencoder_kl.pth" \
    --model_pretrained_path="path/to/flowinone_256px.pth.pth" \
    --fid_stat_path="path/to/fid_stats_mscoco256_val.npz" \
    --inception_ckpt_path="path/to/pt_inception-2015-12-05-6726825d.pth" \
    --sample_path="path/to/save_test_samples" \
    --train_tar_pattern="path/to/train_tar_pattern" \
    --test_tar_pattern="path/to/test_tar_pattern" \
    --vis_image_root="path/to/test_vis" \
    --n_steps=1000000 \
    --batch_size=512 \
    --log_interval=10 \
    --eval_interval=1000 \
    --save_interval=100000 \
    --n_samples_eval=40 \
    --dataset_name=online_features \
    --task=visual_instruction \
    --resolution=256 \
    --shuffle_buffer=500 \
    --resampled=True \
    --split_data_by_node=True \
    --estimated_samples_per_shard=600 \
    --sampling_weights=0.xx,0.xx,0.xx,0.xx,0.xx,0.xx,0.xx,0.xx,0.xx,0.xx,0.xx \
    --sample_steps=50 \
    --n_samples=600 \
    --mini_batch_size=32 \
    --scale=7 \
    --optimizer_name=adamw \
    --lr=0.00001 \
    --weight_decay=0.03 \
    --betas=0.9,0.9 \
    --adamw_impl=AdamW \
    --use_cross_attention=true \
    --wandb_project=wandb_project_name \
    --wandb_mode=online \
    --num_workers=16