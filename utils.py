"""
This file contains some tools
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import einops
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torchvision.transforms import ToTensor
from absl import logging
from PIL import Image

def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


def get_nnet(name, **kwargs):
    if name == 'dimr':
        from libs.model.dimr import MRModel
        return MRModel(kwargs["model_args"])
    else:
        raise NotImplementedError(name)


def get_optimizer(params, name, adamw_impl=None, **kwargs):
    if name == 'adam':
        from torch.optim import Adam
        return Adam(params, **kwargs)
    elif name == 'adamw':
        impl = (adamw_impl or 'bitsandbytes').lower()
        if impl in ('torch', 'adamw'):
            from torch.optim import AdamW
            return AdamW(params, **kwargs)
        elif impl in ('bitsandbytes', 'adamw8bit'):
            from bitsandbytes.optim import AdamW8bit
            return AdamW8bit(params, **kwargs)
        else:
            raise ValueError(f'Unsupported AdamW implementation: {impl}')
    elif name == 'adafactor':
        from torch.optim import Adafactor
        return Adafactor(params, **kwargs)
    else:
        raise NotImplementedError(name)


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'customized':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


class TrainState(object):
    def __init__(self, optimizer, lr_scheduler, step, nnet=None, nnet_ema=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.nnet = nnet
        self.nnet_ema = nnet_ema

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        import shutil
        import time
        import logging
        
        max_retries = 2
        retry_delay = 60  # s
        
        for attempt in range(max_retries):
            temp_path = path + f'.tmp_{int(time.time())}'
            backup_path = path + '.backup'
            
            try:
                if os.path.exists(path):
                    try:
                        if not os.path.exists(os.path.join(path, 'step.pth')):
                            logging.warning(f'Incomplete checkpoint detected at {path}, removing...')
                            shutil.rmtree(path)
                    except Exception as e:
                        logging.warning(f'Error checking checkpoint integrity: {e}')
                
                if os.path.exists(temp_path):
                    shutil.rmtree(temp_path)
                if os.path.exists(backup_path):
                    shutil.rmtree(backup_path)
                
                os.makedirs(temp_path, exist_ok=True)
                
                torch.save(self.step, os.path.join(temp_path, 'step.pth'))
                for key, val in self.__dict__.items():
                    if key != 'step' and val is not None:
                        torch.save(val.state_dict(), os.path.join(temp_path, f'{key}.pth'))
                
                if os.path.exists(path):
                    shutil.move(path, backup_path)
                    try:
                        shutil.move(temp_path, path)
                        shutil.rmtree(backup_path)
                    except Exception as e:
                        if os.path.exists(backup_path):
                            shutil.move(backup_path, path)
                        raise
                else:
                    shutil.move(temp_path, path)
                
                logging.info(f'Successfully saved checkpoint to {path}')
                return  
                
            except Exception as e:
                logging.warning(f'Save attempt {attempt + 1}/{max_retries} failed: {e}')
                
                for tmp in [temp_path, backup_path]:
                    if os.path.exists(tmp):
                        try:
                            shutil.rmtree(tmp)
                        except:
                            pass
                
                if attempt < max_retries - 1:
                    logging.info(f'Retrying in {retry_delay} seconds...')
                    time.sleep(retry_delay)
                else:
                    logging.error(f'Failed to save checkpoint after {max_retries} attempts')
                    raise

    def load(self, path):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'), weights_only=True)
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu', weights_only=True))

    def resume(self, ckpt_root, step=None):
        if not os.path.exists(ckpt_root):
            return
        if step is None:
            ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
            if not ckpts:
                return
            steps = map(lambda x: int(x.split(".")[0]), ckpts)
            step = max(steps)
        ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def trainable_parameters(nnet):
    params_decay = []
    params_nodecay = []
    for name, param in nnet.named_parameters():
        if name.endswith(".nodecay_weight") or name.endswith(".nodecay_bias"):
            params_nodecay.append(param)
        else:
            params_decay.append(param)
    print("params_decay", len(params_decay))
    print("params_nodecay", len(params_nodecay))
    params = [
        {'params': params_decay},
        {'params': params_nodecay, 'weight_decay': 0.0}
    ]
    return params


def initialize_train_state(config, device):

    nnet = get_nnet(**config.nnet)

    if hasattr(config, 'pretrained_path') and config.pretrained_path:
        try:
            print(f"Loading pretrained weights from {config.pretrained_path}...")
            pretrained_dict = torch.load(config.pretrained_path, map_location='cpu', weights_only=True)
            model_dict = nnet.state_dict()
            
            matched_dict = {}
            size_mismatch_keys = []
            missing_keys = []
            
            for k, v in pretrained_dict.items():
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        matched_dict[k] = v
                    else:
                        size_mismatch_keys.append(k)
                        print(f"  ⚠ Size mismatch: {k}")
                        print(f"    pretrained: {v.shape}, current model: {model_dict[k].shape}")
            
            for k in model_dict.keys():
                if k not in pretrained_dict:
                    missing_keys.append(k)
            
            nnet.load_state_dict(matched_dict, strict=False)
            
            print(f"\n{'='*60}")
            print(f"Pretrained weight loading report:")
            print(f"{'='*60}")
            print(f"✓ Successfully loaded parameters: {len(matched_dict)}")
            
            if size_mismatch_keys:
                print(f"\n⚠ Size mismatch ({len(size_mismatch_keys)} keys) - skipped, using random init:")
                for key in size_mismatch_keys[:10]:
                    print(f"  • {key}")
                if len(size_mismatch_keys) > 10:
                    print(f"  ... and {len(size_mismatch_keys)-10} more")
            
            if missing_keys:
                print(f"\n⚠ Missing keys ({len(missing_keys)}):")
                adapter_keys = [k for k in missing_keys if "adapter" in k]
                other_missing = [k for k in missing_keys if "adapter" not in k]
                
                if adapter_keys:
                    print(f"  - Adapter-related keys ({len(adapter_keys)}): random init")
                    for key in adapter_keys[:5]:
                        print(f"    • {key}")
                    if len(adapter_keys) > 5:
                        print(f"    ... and {len(adapter_keys)-5} more")
                
                if other_missing:
                    print(f"  - Other missing keys ({len(other_missing)}): default init")
                    for key in other_missing[:5]:
                        print(f"    • {key}")
                    if len(other_missing) > 5:
                        print(f"    ... and {len(other_missing)-5} more")
            
            print(f"{'='*60}\n")
            
            if hasattr(nnet, 'adapter'):
                nn.init.xavier_uniform_(nnet.adapter[0].weight)
                if hasattr(nnet.adapter[0], 'bias') and nnet.adapter[0].bias is not None:
                    nn.init.zeros_(nnet.adapter[0].bias)
                print("✓ Adapter layer initialized (Xavier uniform)")
            
        except FileNotFoundError:
            print(f"\n❌ Error: pretrained weights file not found '{config.pretrained_path}'")
            print("Check the path, or comment out config.pretrained_path to train from scratch")
            raise
        
        except Exception as e:
            print(f"\n❌ Error loading pretrained weights: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    else:
        print("⚠ No pretrained path set; training from scratch (random init)")
        

    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()

    optimizer = get_optimizer(trainable_parameters(nnet), **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.to(device)
    return train_state


def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


def sample2dir_with_gt(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None, config=None):
    """
    Save generated images, inputs, and ground-truth side by side (order: input, generated, GT).

    Args:
        accelerator: accelerate.Accelerator instance
        path: output directory
        n_samples: total number of samples
        mini_batch_size: per-process batch size
        sample_fn: sampling function returning (generated_samples, gt_images, input_images)
        unpreprocess_fn: inverse preprocessing function
        config: config object
    """
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes

    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir_with_gt'):
        samples, gt_images, input_images = sample_fn(mini_batch_size, config=config)
        
        samples = unpreprocess_fn(samples)
        
        gt_images = unpreprocess_fn(gt_images)
        
        input_images = unpreprocess_fn(input_images)
        
        samples = accelerator.gather(samples.contiguous())[:_batch_size]
        gt_images = accelerator.gather(gt_images.contiguous())[:_batch_size]
        input_images = accelerator.gather(input_images.contiguous())[:_batch_size]
        
        if accelerator.is_main_process:
            target_size = 256  
            
            for input_img, sample, gt in zip(input_images, samples, gt_images):
                if input_img.shape[1] != target_size or input_img.shape[2] != target_size:
                    input_img = input_img.unsqueeze(0)
                    input_img = F.interpolate(input_img, size=(target_size, target_size), mode='bilinear', align_corners=False)
                    input_img = input_img.squeeze(0)  
                
                if sample.shape[1] != target_size or sample.shape[2] != target_size:
                    sample = sample.unsqueeze(0)
                    sample = F.interpolate(sample, size=(target_size, target_size), mode='bilinear', align_corners=False)
                    sample = sample.squeeze(0)  
                
                if gt.shape[1] != target_size or gt.shape[2] != target_size:
                    gt = gt.unsqueeze(0)
                    gt = F.interpolate(gt, size=(target_size, target_size), mode='bilinear', align_corners=False)
                    gt = gt.squeeze(0) 
                
                images_triplet = torch.stack([input_img, sample, gt], dim=0)  
                concatenated = make_grid(images_triplet, nrow=3, padding=2, pad_value=1.0)  
                save_image(concatenated, os.path.join(path, f"{idx}.png"))
                idx += 1
        


# Global cache to avoid repeated tokenizer.encode calls
_tokenizer_cache = {}
_tokenizer_cache_lock = None

def _get_tokenizer_cache_key(vl_chat_processor, question):
    """Build cache key for tokenizer output."""
    # Unique key from question and processor settings
    cache_key = (
        question,
        vl_chat_processor.sft_format,
        vl_chat_processor.system_prompt,
        id(vl_chat_processor.tokenizer)  # tokenizer object id for uniqueness
    )
    return cache_key

def _get_or_encode_tokenizer(vl_chat_processor, question, device):
    """Get or encode tokenizer output (cached)."""
    global _tokenizer_cache, _tokenizer_cache_lock
    import threading
    
    if _tokenizer_cache_lock is None:
        _tokenizer_cache_lock = threading.Lock()
    
    cache_key = _get_tokenizer_cache_key(vl_chat_processor, question)
    
    with _tokenizer_cache_lock:
        if cache_key in _tokenizer_cache:
            return _tokenizer_cache[cache_key]
    
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=[
            {"role": "<|User|>", "content": f"<image_placeholder>\n{question}"},
            {"role": "<|Assistant|>", "content": ""},
        ],
        sft_format=vl_chat_processor.sft_format,
        system_prompt=vl_chat_processor.system_prompt,
    )
    
    input_ids = vl_chat_processor.tokenizer.encode(sft_format)
    
    with _tokenizer_cache_lock:
        _tokenizer_cache[cache_key] = input_ids
    
    return input_ids

def get_input_image_embeddings_and_masks(
    batch_input_images, 
    vl_chat_processor,
    vl_gpt,
    device,
    question="",
    num_image_tokens=576,
    output_tokens=None,
    accelerator=None,
    cached_input_ids=None 
):
    """
    Batch-process input images and obtain token embeddings and masks.

    Args:
        batch_input_images: One of:
            - torch.Tensor: preprocessed tensor [batch_size, 3, H, W] (WebDataset mode)
            - list of str: image paths (filesystem mode, backward compatible)
        vl_chat_processor: Janus VLChatProcessor instance
        vl_gpt: Janus MultiModalityCausalLM. If wrapped by accelerator.prepare(), unwrap first:
                vl_gpt = accelerator.unwrap_model(vl_gpt) if hasattr(accelerator, 'unwrap_model') else vl_gpt
        device: torch.device
        question: optional text prompt (default empty)
        num_image_tokens: tokens per image (default 576)
        output_tokens: if set, keep first N tokens (default None = all)
        accelerator: optional, for rank in error logs
        cached_input_ids: optional pre-encoded input_ids on CPU; skips tokenizer.encode if set

    Returns:
        batch_embeddings: [batch_size, output_tokens or num_image_tokens, hidden_dim] on device
        batch_attention_masks: [batch_size, output_tokens or num_image_tokens] on device
    """
    batch_embeddings_list = []
    batch_attention_masks_list = []
    
    if isinstance(batch_input_images, torch.Tensor):
        if batch_input_images.device != device:
            batched_pixel_values = batch_input_images.to(device, non_blocking=True)
        else:
            batched_pixel_values = batch_input_images
        batch_size = batched_pixel_values.shape[0]
    else:
        import concurrent.futures
        
        def load_image(image_input):
            """Load one image; supports path strings."""
            if isinstance(image_input, str):
                try:
                    pil_img = Image.open(image_input)
                    pil_img.load()
                    return pil_img.convert('RGB')
                except Exception as e:
                    rank_info = f"[Rank {accelerator.process_index}] " if accelerator is not None else ""
                    print(f"{rank_info}Warning: failed to load input image {image_input}: {e}")
                    return Image.new('RGB', (384, 384), color='black')
            else:
                rank_info = f"[Rank {accelerator.process_index}] " if accelerator is not None else ""
                print(f"{rank_info}Warning: unsupported type {type(image_input)}")
                return Image.new('RGB', (384, 384), color='black')
        
        if len(batch_input_images) > 0:
            max_workers = min(len(batch_input_images), os.cpu_count() or 1)
            if max_workers > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    all_pil_images = list(executor.map(load_image, batch_input_images))
            else:
                all_pil_images = [load_image(path) for path in batch_input_images]
        else:
            all_pil_images = []
        
        images_outputs = vl_chat_processor.image_processor(all_pil_images, return_tensors="pt")
        batched_pixel_values = images_outputs.pixel_values.to(device, non_blocking=True)  # [batch_size, 3, H, W]
        batch_size = len(all_pil_images)
    
    if cached_input_ids is not None:
        input_ids = cached_input_ids
    else:
        input_ids = _get_or_encode_tokenizer(vl_chat_processor, question, device)
    
    batched_input_ids = torch.tensor([input_ids] * batch_size, dtype=torch.long, device=device)  
    
    image_token_mask = batched_input_ids == vl_chat_processor.image_id
    batched_images_seq_mask = image_token_mask
    
    batched_images_emb_mask = torch.zeros((batch_size, 1, num_image_tokens), dtype=torch.bool, device=device)
    batched_images_emb_mask[:, :, :num_image_tokens] = True
    
    batched_pixel_values = batched_pixel_values.unsqueeze(1)  
    
    with torch.no_grad():
        inputs_embeds = vl_gpt.prepare_inputs_embeds(
            input_ids=batched_input_ids,
            pixel_values=batched_pixel_values,
            images_seq_mask=batched_images_seq_mask,
            images_emb_mask=batched_images_emb_mask
        )
    
    inputs_embeds = inputs_embeds.detach().float()
    
    if inputs_embeds.shape[1] == num_image_tokens:
        batch_embeddings = inputs_embeds  # [batch_size, num_image_tokens, hidden_dim]
    else:
        num_image_tokens_per_sample = batched_images_seq_mask.sum(dim=1)  # [batch_size]
        
        if (num_image_tokens_per_sample == num_image_tokens).all():
            batch_embeddings_list = []
            for i in range(batch_size):
                image_mask = batched_images_seq_mask[i]  # [seq_len]
                image_embeddings = inputs_embeds[i][image_mask]  # [num_image_tokens, hidden_dim]
                batch_embeddings_list.append(image_embeddings)
            batch_embeddings = torch.stack(batch_embeddings_list, dim=0)  # [batch_size, num_image_tokens, hidden_dim]
        else:
            batch_embeddings_list = []
            for i in range(batch_size):
                image_mask = batched_images_seq_mask[i]  # [seq_len]
                image_embeddings = inputs_embeds[i][image_mask]  # [actual_tokens, hidden_dim]
                
                if image_embeddings.shape[0] > num_image_tokens:
                    image_embeddings = image_embeddings[:num_image_tokens]
                elif image_embeddings.shape[0] < num_image_tokens:
                    padding = torch.zeros(
                        (num_image_tokens - image_embeddings.shape[0], image_embeddings.shape[1]),
                        device=device,
                        dtype=image_embeddings.dtype
                    )
                    image_embeddings = torch.cat([image_embeddings, padding], dim=0)
                
                batch_embeddings_list.append(image_embeddings)
            batch_embeddings = torch.stack(batch_embeddings_list, dim=0)  # [batch_size, num_image_tokens, hidden_dim]
    
    batch_attention_masks = torch.ones(
        (batch_size, num_image_tokens), 
        device=device, 
        dtype=torch.long
    )  # [batch_size, num_image_tokens]
    
    if output_tokens is not None and output_tokens < num_image_tokens:
        batch_embeddings = batch_embeddings[:, :output_tokens, :]  # [batch_size, output_tokens, hidden_dim]
        batch_attention_masks = batch_attention_masks[:, :output_tokens]  # [batch_size, output_tokens]
    
    return batch_embeddings, batch_attention_masks


# Visualization helpers

def resize_tensor_image(img, target_size, device=None):
    """Resize a [C, H, W] image tensor to target_size × target_size."""
    if device is not None and img.device != device:
        img = img.to(device)
    if img.shape[1] != target_size or img.shape[2] != target_size:
        img = F.interpolate(
            img.unsqueeze(0),
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
    return img


def build_cross_atten_mask_from_batch_type(batch_type, batch_size, device):
    """Build cross-attention bool mask from a list of type bytes.

    Elements equal to ``b"t2i"`` map to ``True`` (skip cross-attention).
    If *batch_type* is ``None``, returns an all-False tensor of length *batch_size*.
    """
    if batch_type is not None:
        return torch.tensor(
            [t == b"t2i" if t is not None else False for t in batch_type],
            dtype=torch.bool,
            device=device,
        )
    return torch.zeros(batch_size, dtype=torch.bool, device=device)


def build_cross_atten_mask_from_paths(image_paths, device):
    """Return a bool tensor indicating which images are t2i (skip cross-attention).

    Images whose filename starts with ``t2i_`` map to ``True``.
    """
    mask = [os.path.basename(p).startswith("t2i_") for p in image_paths]
    return torch.tensor(mask, dtype=torch.bool, device=device)


def load_images_as_latents(image_paths, resolution, autoencoder, device):
    """Load images from *image_paths*, center-crop, and encode through autoencoder.

    Returns the sampled latent tensor of shape ``[N, 4, H, W]``.
    """
    from data import center_crop_arr

    tensors = []
    for path in image_paths:
        pil = Image.open(path).convert("RGB")
        arr = center_crop_arr(pil, image_size=resolution)
        arr = (arr / 127.5 - 1.0).astype(np.float32)
        tensors.append(torch.from_numpy(einops.rearrange(arr, 'h w c -> c h w')).to(device))

    stacked = torch.stack(tensors, dim=0)
    moments = autoencoder(stacked, fn='encode_moments').detach()
    return autoencoder.sample(moments)


def save_vis_grid_and_log(
    samples_unpreprocessed,
    input_images_pil,
    gt_images_pil,
    sample_dir,
    step,
    wandb_module,
    device,
    samples_per_group=10,
    target_size=256,
):
    """Build [input | gt | generated] image grids, save to disk, and log to wandb.

    Args:
        samples_unpreprocessed: list/tensor of generated images [C, H, W] in [0, 1].
        input_images_pil: list of PIL input images.
        gt_images_pil: list of PIL ground-truth images (may be empty).
        sample_dir: directory to save grid images.
        step: current training step (used in filenames and wandb step).
        wandb_module: the wandb module (passed to avoid importing it here).
        device: torch device for tensor operations.
        samples_per_group: number of rows per saved grid file.
        target_size: spatial size each image is resized to before gridding.
    """
    total = len(samples_unpreprocessed)
    num_groups = (total + samples_per_group - 1) // samples_per_group
    wandb_images = []

    for group_id in range(num_groups):
        start = group_id * samples_per_group
        end = min(start + samples_per_group, total)
        group_imgs = []

        for i in range(start, end):
            if i < len(input_images_pil):
                group_imgs.append(resize_tensor_image(ToTensor()(input_images_pil[i]), target_size, device))
            else:
                group_imgs.append(torch.zeros((3, target_size, target_size), device=device))

            if i < len(gt_images_pil):
                group_imgs.append(resize_tensor_image(ToTensor()(gt_images_pil[i]), target_size, device))
            else:
                group_imgs.append(torch.zeros((3, target_size, target_size), device=device))

            group_imgs.append(resize_tensor_image(samples_unpreprocessed[i], target_size, device))

        save_path = os.path.join(sample_dir, f'{step}_{group_id + 1}.png')
        if group_imgs:
            grid = make_grid(torch.stack(group_imgs, dim=0), nrow=3, padding=2, pad_value=1.0)
            save_image(grid, save_path)
            wandb_images.append(wandb_module.Image(grid))
        else:
            fallback = samples_unpreprocessed[start:end]
            if len(fallback) > 0:
                grid = make_grid(fallback, 5)
                save_image(grid, save_path)
                wandb_images.append(wandb_module.Image(grid))

    if wandb_images:
        wandb_module.log({'samples': wandb_images}, step=step)


def clean_stale_ckpt_files(ckpt_root):
    """Remove temporary/backup files in *ckpt_root* older than one hour."""
    import shutil
    import time as _time

    try:
        for item in os.listdir(ckpt_root):
            if '.tmp_' not in item and not item.endswith('.backup'):
                continue
            tmp_path = os.path.join(ckpt_root, item)
            try:
                if os.path.isdir(tmp_path) and _time.time() - os.path.getmtime(tmp_path) > 3600:
                    logging.info(f'Removing stale temporary directory: {tmp_path}')
                    shutil.rmtree(tmp_path)
            except Exception as e:
                logging.warning(f'Error cleaning temporary file {tmp_path}: {e}')
    except Exception as e:
        logging.warning(f'Error scanning checkpoint directory: {e}')