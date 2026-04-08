import os
import sys
from pathlib import Path

from absl import flags
from absl import app
from ml_collections import config_flags

from train import train


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.DEFINE_string("workdir_base", None, "Base directory for workdir. If not provided, uses default path.")
flags.DEFINE_string("vae_pretrained_path", None, "Path to pretrained VAE checkpoint.")
flags.DEFINE_string("model_pretrained_path", None, "Path to pretrained model checkpoint.")
flags.DEFINE_string("fid_stat_path", None, "Path to FID statistics file.")
flags.DEFINE_string("inception_ckpt_path", None, "Path to Inception checkpoint.")
flags.DEFINE_string("sample_path", None, "Path to save samples.")
flags.DEFINE_string("train_tar_pattern", None, "Training tar pattern for WebDataset.")
flags.DEFINE_string("test_tar_pattern", None, "Test tar pattern for WebDataset.")
flags.DEFINE_string("vis_image_root", None, "Path to visualization images root.")
flags.DEFINE_string("resume_ckpt_root", None, "Path to checkpoint root directory for resuming. If not provided, uses workdir/ckpts.")

# WandB parameters
flags.DEFINE_string("wandb_project", None, "WandB project name. If not provided, uses config.wandb_project or default naming.")
flags.DEFINE_enum("wandb_mode", None, ["online", "offline", "disabled"], "WandB mode: online (sync to cloud), offline (local only), or disabled.")

# Training parameters
flags.DEFINE_integer("n_steps", None, "Total training iterations.")
flags.DEFINE_integer("batch_size", None, "Overall batch size across ALL gpus.")
flags.DEFINE_integer("log_interval", None, "Iteration interval for logging.")
flags.DEFINE_integer("eval_interval", None, "Iteration interval for visual testing.")
flags.DEFINE_integer("save_interval", None, "Iteration interval for saving checkpoints.")
flags.DEFINE_integer("n_samples_eval", None, "Number of samples for evaluation.")

# Dataset parameters
flags.DEFINE_string("dataset_name", None, "Dataset name.")
flags.DEFINE_string("task", None, "Task name.")
flags.DEFINE_integer("resolution", None, "Dataset resolution.")
flags.DEFINE_integer("shuffle_buffer", None, "Shuffle buffer size for WebDataset.")
flags.DEFINE_boolean("resampled", None, "Whether to resample WebDataset.")
flags.DEFINE_boolean("split_data_by_node", None, "Whether to split data by node.")
flags.DEFINE_integer("estimated_samples_per_shard", None, "Estimated samples per shard.")
flags.DEFINE_string("sampling_weights", None, "Sampling weights for multiple tar patterns (format: '0.7,0.3').")

# Sample parameters
flags.DEFINE_integer("sample_steps", None, "Sample steps during inference/testing.")
flags.DEFINE_integer("n_samples", None, "Number of samples for testing.")
flags.DEFINE_integer("mini_batch_size", None, "Batch size for testing.")
flags.DEFINE_integer("scale", None, "CFG scale.")

# Optimizer parameters
flags.DEFINE_string("optimizer_name", None, "Optimizer name.")
flags.DEFINE_float("lr", None, "Learning rate.")
flags.DEFINE_float("weight_decay", None, "Weight decay.")
flags.DEFINE_string("betas", None, "Betas for optimizer (format: '0.9,0.9').")
flags.DEFINE_enum("adamw_impl", None, ["torch", "bitsandbytes", "AdamW", "AdamW8bit"], "Select AdamW backend.")

# DataLoader parameters
flags.DEFINE_integer("num_workers", None, "Number of workers for DataLoader.")

# Model parameters
flags.DEFINE_boolean("use_cross_attention", None, "Whether to use cross attention in the first stage config.")


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem


def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        assert '=' in argv[i]
        if argv[i].startswith('--config.') and not argv[i].startswith('--config.dataset.path'):
            hparam, val = argv[i].split('=')
            hparam = hparam.split('.')[-1]
            if hparam.endswith('path'):
                val = Path(val).stem
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    if hparams == '':
        hparams = 'default'
    return hparams


def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.hparams = get_hparams()

    if FLAGS.workdir:
        config.workdir = FLAGS.workdir
    else:
        default_workdir_base = '/path/to/workdir_base'
        workdir_base = FLAGS.workdir_base or default_workdir_base
        config.workdir = os.path.join(workdir_base, config.config_name, config.hparams)

    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    # resume_ckpt_root is used for resuming; if specified, the specified path is used; otherwise, ckpt_root is used.
    if FLAGS.resume_ckpt_root:
        config.resume_ckpt_root = FLAGS.resume_ckpt_root
    else:
        config.resume_ckpt_root = config.ckpt_root
    config.sample_dir = os.path.join(config.workdir, 'samples')

    # WandB
    if FLAGS.wandb_project:
        config.wandb_project = FLAGS.wandb_project
    if FLAGS.wandb_mode:
        config.wandb_mode = FLAGS.wandb_mode

    if FLAGS.vae_pretrained_path:
        config.autoencoder.pretrained_path = FLAGS.vae_pretrained_path
    if FLAGS.model_pretrained_path:
        config.pretrained_path = FLAGS.model_pretrained_path
    if FLAGS.fid_stat_path:
        config.fid_stat_path = FLAGS.fid_stat_path
    if FLAGS.inception_ckpt_path:
        config.inception_ckpt_path = FLAGS.inception_ckpt_path
    if FLAGS.sample_path:
        config.sample.path = FLAGS.sample_path
    if FLAGS.train_tar_pattern:
        config.dataset.train_tar_pattern = FLAGS.train_tar_pattern
    if FLAGS.test_tar_pattern:
        config.dataset.test_tar_pattern = FLAGS.test_tar_pattern
    if FLAGS.vis_image_root:
        config.dataset.vis_image_root = FLAGS.vis_image_root

    # Training parameters
    if FLAGS.n_steps is not None:
        config.train.n_steps = FLAGS.n_steps
    if FLAGS.batch_size is not None:
        config.train.batch_size = FLAGS.batch_size
    if FLAGS.log_interval is not None:
        config.train.log_interval = FLAGS.log_interval
    if FLAGS.eval_interval is not None:
        config.train.eval_interval = FLAGS.eval_interval
    if FLAGS.save_interval is not None:
        config.train.save_interval = FLAGS.save_interval
    if FLAGS.n_samples_eval is not None:
        config.train.n_samples_eval = FLAGS.n_samples_eval

    # Dataset parameters
    if FLAGS.dataset_name is not None:
        config.dataset.name = FLAGS.dataset_name
    if FLAGS.task is not None:
        config.dataset.task = FLAGS.task
    if FLAGS.resolution is not None:
        config.dataset.resolution = FLAGS.resolution
    if FLAGS.shuffle_buffer is not None:
        config.dataset.shuffle_buffer = FLAGS.shuffle_buffer
    if FLAGS.resampled is not None:
        config.dataset.resampled = FLAGS.resampled
    if FLAGS.split_data_by_node is not None:
        config.dataset.split_data_by_node = FLAGS.split_data_by_node
    if FLAGS.estimated_samples_per_shard is not None:
        config.dataset.estimated_samples_per_shard = FLAGS.estimated_samples_per_shard
    if FLAGS.sampling_weights is not None:
        sampling_weights_values = [float(x.strip()) for x in FLAGS.sampling_weights.split(',')]
        config.dataset.sampling_weights = sampling_weights_values

    # Sample parameters
    if FLAGS.sample_steps is not None:
        config.sample.sample_steps = FLAGS.sample_steps
    if FLAGS.n_samples is not None:
        config.sample.n_samples = FLAGS.n_samples
    if FLAGS.mini_batch_size is not None:
        config.sample.mini_batch_size = FLAGS.mini_batch_size
    if FLAGS.scale is not None:
        config.sample.scale = FLAGS.scale

    # Optimizer parameters
    if FLAGS.optimizer_name is not None:
        config.optimizer.name = FLAGS.optimizer_name
    if FLAGS.lr is not None:
        config.optimizer.lr = FLAGS.lr
    if FLAGS.weight_decay is not None:
        config.optimizer.weight_decay = FLAGS.weight_decay
    if FLAGS.betas is not None:
        betas_values = [float(x.strip()) for x in FLAGS.betas.split(',')]
        config.optimizer.betas = tuple(betas_values)
    if FLAGS.adamw_impl is not None:
        config.optimizer.adamw_impl = FLAGS.adamw_impl

    # DataLoader parameters
    if FLAGS.num_workers is not None:
        config.num_workers = FLAGS.num_workers

    # Model parameters
    if FLAGS.use_cross_attention is not None:
        if hasattr(config.nnet.model_args, 'stage_configs') and len(config.nnet.model_args.stage_configs) > 0:
            config.nnet.model_args.stage_configs[0].use_cross_attention = FLAGS.use_cross_attention

    train(config)


if __name__ == "__main__":
    app.run(main)