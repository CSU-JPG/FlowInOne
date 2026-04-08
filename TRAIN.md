# Training Guide

For **DDP training**, you can choose between single-machine or multi-machine training.

Use the script below directly (copy of `run_train.sh`) and change the config. Detailed parameter explanations are provided in the tables below.

```bash
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

WANDB_API_KEY=""

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
    --model_pretrained_path="path/to/flowinone_256px.pth" \
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
    --n_samples=3000 \
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
```

### Parameter Explanations

#### Environment & Distributed Launch
| Variable | Meaning |
| :--- | :--- |
| `TORCH_NCCL_*` | PyTorch NCCL hints to avoid certain stream recording paths and enable async error handling for stable multi-GPU communication. |
| `TOKENIZERS_PARALLELISM` | Set to `false` to avoid Hugging Face tokenizers forking warnings. |
| `WANDB_API_KEY` | Your Weights & Biases API key for online logging. |
| `NNODES` | Number of machines (nodes) in the job. |
| `NPROC_PER_NODE` | Number of processes (typically GPUs) per node. |
| `NPROC` | Total processes across all machines (`NNODES * NPROC_PER_NODE`). |
| `MASTER_PORT` | TCP port for the main process in distributed training. |

#### Accelerate Arguments
| Argument | Meaning |
| :--- | :--- |
| `--main_process_port` | Same as `MASTER_PORT`. |
| `--multi_gpu` | Run one process per GPU on this machine. |
| `--num_processes` | Total number of processes across all nodes (`$NPROC`). |
| `--num_machines` | Number of nodes (`$NNODES`). |
| `--mixed_precision` | Use `bf16` mixed precision in training. |

#### Paths & Assets
| Flag | Meaning |
| :--- | :--- |
| `--config` | Training config module, e.g. `configs/flowinone_training_demo.py`. |
| `--workdir_base` | Base directory for logs, checkpoints and training visualization. |
| `--vae_pretrained_path` | Path to the pretrained VAE weights (e.g. Stable Diffusion VAE checkpoint). |
| `--model_pretrained_path` | Path to the FlowInOne pretrained checkpoint for init/fine-tuning. |
| `--fid_stat_path` | Precomputed FID statistics file (e.g. `.npz`) for evaluation. |
| `--inception_ckpt_path` | Inception weights used for FID / related metrics. |
| `--sample_path` | Where to write generated samples during eval. |
| `--vis_image_root` | Root folder of images used for training visualization. |

For `--vae_pretrained_path`, `--model_pretrained_path`, `--fid_stat_path`, `--inception_ckpt_path`, you can download all necessary models directly from [here](https://huggingface.co/CSU-JPG/FlowInOne/blob/main/preparation.tar.gz)
#### Training Schedule & Logging
| Flag | Meaning |
| :--- | :--- |
| `--n_steps` | Total training steps (iterations). |
| `--batch_size` | **Global** batch size summed over **all** GPUs/processes. |
| `--log_interval` | Print/log training stats every this many steps. |
| `--eval_interval` | Run visual eval routines every this many steps. |
| `--save_interval` | Save checkpoints and eval every this many steps. |
| `--n_samples_eval` | Number of samples used in the periodic visual eval path in training. |

#### Dataset & WebDataset
| Flag | Meaning |
| :--- | :--- |
| `--dataset_name` | Dataset registry name (e.g. `online_features`). |
| `--task` | Task name passed to the data pipeline (e.g. `visual_instruction`). |
| `--resolution` | Target spatial resolution for training data (e.g. `256`). |
| `--shuffle_buffer` | Shuffle buffer size when streaming WebDataset. |
| `--resampled` | Whether to use resampled/weighted mixing in the WebDataset pipeline. |
| `--split_data_by_node` | Whether each node only reads a shard of the data (multi-node). |
| `--estimated_samples_per_shard` | Hint for samples per tar shard (should match packing, e.g. `600`). |
| `--sampling_weights` | Comma-separated mixing weights for multiple training data sources (sum must be 1). |

#### Sampling (Eval / Test)
| Flag | Meaning |
| :--- | :--- |
| `--sample_steps` | Number of ODE/flow-matching steps when generating samples. |
| `--n_samples` | How many samples to generate in the configured test pass. |
| `--mini_batch_size` | Micro-batch size for generation (memory vs. speed). |
| `--scale` | Classifier-free guidance (CFG) scale for sampling. |

#### Optimizer & Model
| Flag | Meaning |
| :--- | :--- |
| `--optimizer_name` | Optimizer type (e.g. `adamw`). |
| `--lr` | Learning rate. |
| `--weight_decay` | Weight decay coefficient. |
| `--betas` | Adam `beta1,beta2` as two comma-separated floats. |
| `--adamw_impl` | Which AdamW implementation to use: `AdamW`, or `AdamW8bit`. |
| `--use_cross_attention` | Overrides the first stage's `use_cross_attention` in the model config. |

#### Misc
| Flag | Meaning |
| :--- | :--- |
| `--wandb_project` | W&B project name. |
| `--wandb_mode` | `online`, `offline`, or `disabled`. |
| `--num_workers` | `DataLoader` worker processes per rank. |

### Tar pattern format
 (`--train_tar_pattern` / `--test_tar_pattern`)

- Use comma-separated entries when mixing multiple dataset roots. Each entry is one absolute path to a shard naming template.
- Inside each entry, use bash brace expansion for zero-padded indices: `pairs-{000000..001390}.tar` expands to `pairs-000000.tar`, ... , `pairs-001390.tar`.
- The number of comma-separated segments must match `--sampling_weights`: one weight per segment (e.g. 11 sources -> 11 comma-separated floats).
- `--test_tar_pattern` uses the same string syntax; often a single pattern or a shorter list for validation.

Example (11 mixed sources, one line):

```text
/c2i-860K/pairs-{000000..001390}.tar,/t2i-2M/pairs-{000000..003780}.tar, /GPT-Image-Edit1M/pairs-{000000..001730}.tar, /UnicEdit600K/pairs-{000000..000975}.tar, /pico-11K/pairs-{000002..000018}.tar, /PixWizard315K/pairs-{000000..000520}.tar, /vismarker250K/pairs-{000000..000415}.tar, /text_box_control24K/pairs-{000000..000038}.tar, /force-32K/pairs-{000000..000052}.tar, /trajectory1K/pairs-{000000..000002}.tar, /doodles-1K/pairs-{000000..000001}.tar
```

Run:

```bash
bash run_train.sh
```
