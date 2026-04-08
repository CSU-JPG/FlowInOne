"""
    Image-to-Image
"""
import os
import sys
import glob
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ml_collections
import torch
from torch import multiprocessing as mp
import accelerate
import utils
from absl import logging
import builtins
import einops
import numpy as np
from PIL import Image
from torchvision.utils import save_image

from diffusion.flow_matching import ODEEulerFlowMatchingSolver
import libs.autoencoder

from libs.janus.models import MultiModalityCausalLM, VLChatProcessor
from transformers import AutoModelForCausalLM


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def unpreprocess(x):
    x = 0.5 * (x + 1.)
    x.clamp_(0., 1.)
    return x


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def load_and_preprocess_image(image_path, image_size, device):
    input_pil = Image.open(image_path).convert("RGB")
    input_arr = center_crop_arr(input_pil, image_size=image_size)
    input_arr = (input_arr / 127.5 - 1.0).astype(np.float32)
    input_tensor = torch.from_numpy(einops.rearrange(input_arr, 'h w c -> c h w')).to(device)
    input_tensor = input_tensor.unsqueeze(0)
    return input_tensor


def collect_images(input_path):
    """Collect all images under the input path (supports single files, single directories, and recursive subdirectories)."""
    if os.path.isfile(input_path):
        return [input_path]

    image_paths = []
    for root, _, files in os.walk(input_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(root, f))
    return sorted(image_paths)


def evaluate(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    if 'ACCELERATE_MIXED_PRECISION' not in os.environ and 'ACCELERATE_TORCH_DEVICE' not in os.environ:
        for key in ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'LOCAL_RANK', 'WORLD_SIZE']:
            os.environ.pop(key, None)
        logging.info("Running in single GPU mode (non-distributed)")
    else:
        logging.info("Running with accelerate launch (multi-GPU or distributed mode)")

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)
    if accelerator.is_main_process:
        if config.output_path:
            os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
        utils.set_logger(log_level='info', fname=config.output_path)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    nnet = utils.get_nnet(**config.nnet)
    nnet = accelerator.prepare(nnet)
    logging.info(f'Loading nnet from {config.nnet_path}')
    accelerator.unwrap_model(nnet).load_state_dict(torch.load(config.nnet_path, map_location='cpu'))
    nnet.eval()

    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    model_path = "/path/to/Janus-Pro-1B/"
    logging.info(f'Loading Janus VLM from {model_path}')
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, use_safetensors=True
    )
    vl_gpt = vl_gpt.half().eval().to(device)

    question = ""
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=[
            {"role": "<|User|>", "content": f"<image_placeholder>\n{question}"},
            {"role": "<|Assistant|>", "content": ""},
        ],
        sft_format=vl_chat_processor.sft_format,
        system_prompt=vl_chat_processor.system_prompt,
    )
    cached_input_ids = vl_chat_processor.tokenizer.encode(sft_format)
    logging.info(f'Tokenizer pre-encoded, input_ids length: {len(cached_input_ids)}')

    image_paths = collect_images(config.input_image_path)
    if not image_paths:
        raise ValueError(f'No images found in: {config.input_image_path}')
    logging.info(f'Found {len(image_paths)} images')

    output_dir = config.output_image_path or './inference_results'
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f'Output directory: {output_dir}')

    bdv_nnet = None
    vae_image_size = config.dataset.resolution if hasattr(config, 'dataset') else 256
    skip_cross_atten = config.get('skip_cross_atten', False)

    def ode_fm_solver_sample(nnet_ema, _n_samples, context=None, token_mask=None, image_latent=None, use_cross_atten_mask=None):
        with torch.no_grad():
            _z_gaussian = torch.randn(_n_samples, *config.z_shape, device=device)

            if 'dimr' in config.nnet.name or 'dit' in config.nnet.name:
                _z_x0, _mu, _log_var = nnet_ema(context, text_encoder=True, shape=_z_gaussian.shape, mask=token_mask, use_cross_atten_mask=use_cross_atten_mask)
                _z_init = _z_x0.reshape(_z_gaussian.shape)
            else:
                raise NotImplementedError

            assert config.sample.scale > 1
            _cfg = config.cfg if config.cfg != -1 else config.sample.scale
            has_null_indicator = hasattr(config.nnet.model_args, "cfg_indicator")
            _sample_steps = config.sample.sample_steps

            ode_solver = ODEEulerFlowMatchingSolver(
                nnet_ema,
                bdv_model_fn=bdv_nnet,
                step_size_type="step_in_dsigma",
                guidance_scale=_cfg
            )
            _z, _ = ode_solver.sample(
                x_T=_z_init,
                batch_size=_n_samples,
                sample_steps=_sample_steps,
                unconditional_guidance_scale=_cfg,
                has_null_indicator=has_null_indicator,
                image_latent=image_latent,
                use_cross_atten_mask=use_cross_atten_mask
            )

            return decode(_z)

    logging.info(f'Starting batch processing of {len(image_paths)} images...')
    batch_size = config.get('inference_batch_size', 4)
    success_count = 0
    error_count = 0

    for batch_start in range(0, len(image_paths), batch_size):
        batch_end = min(batch_start + batch_size, len(image_paths))
        batch_paths = image_paths[batch_start:batch_end]
        current_batch_size = len(batch_paths)

        logging.info(f'Processing batch [{batch_start+1}-{batch_end}/{len(image_paths)}]')

        try:
            batch_tensors_vae = []
            for img_path in batch_paths:
                t = load_and_preprocess_image(img_path, vae_image_size, device)
                batch_tensors_vae.append(t)
            batch_tensors_vae = torch.cat(batch_tensors_vae, dim=0)

            if skip_cross_atten:
                use_cross_atten_mask = torch.ones(current_batch_size, dtype=torch.bool, device=device)
            else:
                mask_list = []
                for img_path in batch_paths:
                    parts = img_path.split(os.sep)
                    mask_list.append('class2image' in parts or 'text2image' in parts)
                use_cross_atten_mask = torch.tensor(mask_list, dtype=torch.bool, device=device)

            contexts, token_mask = utils.get_input_image_embeddings_and_masks(
                batch_input_images=batch_paths,
                vl_chat_processor=vl_chat_processor,
                vl_gpt=vl_gpt,
                device=device,
                question=question,
                num_image_tokens=576,
                output_tokens=576,
                accelerator=accelerator,
                cached_input_ids=cached_input_ids
            )

            with torch.no_grad():
                input_moments = autoencoder(batch_tensors_vae, fn='encode_moments')
                input_latent = autoencoder.sample(input_moments)

            samples = ode_fm_solver_sample(
                nnet,
                _n_samples=current_batch_size,
                context=contexts,
                token_mask=token_mask,
                image_latent=input_latent,
                use_cross_atten_mask=use_cross_atten_mask
            )
            samples = unpreprocess(samples)

            if accelerator.is_main_process:
                for i in range(current_batch_size):
                    out_name = os.path.basename(batch_paths[i])
                    save_image(samples[i], os.path.join(output_dir, out_name))
                success_count += current_batch_size
                logging.info(f'  Batch saved successfully ({current_batch_size} images)')

        except Exception as e:
            logging.error(f'  Batch error: {str(e)}')
            logging.error(traceback.format_exc())
            error_count += current_batch_size
            continue

    if accelerator.is_main_process:
        logging.info('')
        logging.info('=' * 60)
        logging.info('Processing completed!')
        logging.info(f'Total: {len(image_paths)} | Success: {success_count} | Errors: {error_count}')
        logging.info(f'Output directory: {output_dir}')
        logging.info('=' * 60)


from absl import flags
from absl import app
from ml_collections import config_flags

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Inference configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("nnet_path", None, "The nnet checkpoint path.")
flags.DEFINE_string("input_image_path", None, "Input image or directory path.")
flags.DEFINE_string("output_image_path", None, "Output directory path.")
flags.DEFINE_string("output_path", None, "Log file path.")
flags.DEFINE_float("cfg", -1, 'CFG scale (-1 to use config default)')
flags.DEFINE_integer("batch_size", 4, 'Batch size for inference')
flags.DEFINE_boolean("skip_cross_atten", False, 'Skip cross attention for all images')


def main(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.input_image_path = FLAGS.input_image_path
    config.output_image_path = FLAGS.output_image_path
    config.output_path = FLAGS.output_path
    config.cfg = FLAGS.cfg
    config.inference_batch_size = FLAGS.batch_size
    config.skip_cross_atten = FLAGS.skip_cross_atten
    evaluate(config)


if __name__ == "__main__":
    app.run(main)