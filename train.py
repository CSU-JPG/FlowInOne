import ml_collections
import torch
from torch import multiprocessing as mp
from data.data_factory import OnlineFeatures
import utils
import accelerate
from tqdm.auto import tqdm
import tempfile
from absl import logging
import builtins
import os
import wandb
from PIL import Image

if 'NCCL_ASYNC_ERROR_HANDLING' in os.environ:
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = os.environ['NCCL_ASYNC_ERROR_HANDLING']

import libs.autoencoder
from diffusion.flow_matching import FlowMatching, ODEEulerFlowMatchingSolver
from tools.fid_score import calculate_fid_given_paths

from libs.janus.models import MultiModalityCausalLM, VLChatProcessor
from transformers import AutoModelForCausalLM

def train(config):
    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb_mode = (
            getattr(config, 'wandb_mode', None) or
            os.environ.get('WANDB_MODE', None) or
            'online'
        )
        
        wandb_project = (
            getattr(config, 'wandb_project', None) or
            os.environ.get('WANDB_PROJECT', None) or
            f'{config.config_name}_{config.dataset.name}'
        )
        
        wandb.init(dir=os.path.abspath(config.workdir), project=wandb_project, config=config.to_dict(),
                   name=config.hparams, job_type='train', mode=wandb_mode)
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
        logging.info(f'Optimizer config: {config.optimizer}')
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None
    logging.info(f'Run on {accelerator.num_processes} devices')

    model_path = "deepseek-ai/Janus-Pro-1B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)

    # pre-encode tokenizer
    training_question = ""
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=[
            {"role": "<|User|>", "content": f"<image_placeholder>\n{training_question}"},
            {"role": "<|Assistant|>", "content": ""},
        ],
        sft_format=vl_chat_processor.sft_format,
        system_prompt=vl_chat_processor.system_prompt,
    )
    cached_training_input_ids = vl_chat_processor.tokenizer.encode(sft_format)
    logging.info(f'Pre-encoded tokenizer completed, input_ids length: {len(cached_training_input_ids)}')

    dataset = OnlineFeatures(
        train_tar_pattern=config.dataset.train_tar_pattern,
        test_tar_pattern=config.dataset.test_tar_pattern,
        vis_image_root=config.dataset.vis_image_root,
        task=config.dataset.task,
        resolution=config.dataset.resolution,
        shuffle_buffer=config.dataset.shuffle_buffer,
        resampled=config.dataset.resampled,
        split_data_by_node=config.dataset.split_data_by_node,
        estimated_samples_per_shard=config.dataset.estimated_samples_per_shard,
        cfg=config.dataset.cfg,
        fid_stat_path=getattr(config, 'fid_stat_path', None),
        num_workers=getattr(config, 'num_workers', 8),
        batch_size=mini_batch_size,
        test_batch_size=config.sample.mini_batch_size,
        test_num_workers=3,
        vl_chat_processor=vl_chat_processor,
        device=device,
        sampling_weights=getattr(config.dataset, 'sampling_weights', None),
    )

    test_dataset = dataset.test 
    train_dataset_loader, test_dataset_loader = dataset.train_dataloader, dataset.test_dataloader

    train_state = utils.initialize_train_state(config, device)
    nnet, nnet_ema, optimizer = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer
    )
    lr_scheduler = train_state.lr_scheduler
    resume_path = config.resume_ckpt_root
    if resume_path and resume_path.endswith('.ckpt') and os.path.isdir(resume_path):
        logging.info(f'Load from checkpoint directory: {resume_path}')
        train_state.load(resume_path)
    else:
        train_state.resume(resume_path)

    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)
    
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, use_safetensors=True
    )
    vl_gpt = vl_gpt.half().eval().to(device)

    def decode(_batch):
        return autoencoder.decode(_batch)

    def get_data_generator():
        while True:
            remaining_steps = config.train.n_steps - train_state.step
            for data in tqdm(
                train_dataset_loader,
                disable=not accelerator.is_main_process,
                desc=f'step {train_state.step}/{config.train.n_steps}',
                unit=' its',
                ncols=120,
                dynamic_ncols=True,
                total=remaining_steps,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ):
                yield data

    data_generator = get_data_generator()

    def get_context_generator():
        while True:
            for data in tqdm(
                test_dataset_loader,
                disable=not accelerator.is_main_process,
                desc='step',
                unit=' its'
            ):
                yield data

    context_generator = get_context_generator()

    world_size = accelerator.num_processes
    rank = accelerator.process_index
    
    _flow_mathcing_model = FlowMatching(world_size=world_size, rank=rank)
    
    if accelerator.is_main_process:
        logging.info(f"FlowMatching initialized with world_size={world_size}, rank={rank}")
        logging.info(f"ClipLoss will use multi-GPU feature gathering: {world_size > 1}")

    def train_step(_batch):
        _metrics = dict()
        optimizer.zero_grad()

        assert len(_batch) == 4
        assert not config.dataset.cfg
        _batch_input_img, _batch_output_img, _batch_input_img_tensor, _batch_type = _batch

        if isinstance(_batch_output_img, torch.Tensor):
            _batch_output_img = _batch_output_img.to(device, non_blocking=True)
            if _batch_output_img.dim() == 3:
                _batch_output_img = _batch_output_img.unsqueeze(0)
        if isinstance(_batch_input_img_tensor, torch.Tensor):
            _batch_input_img_tensor = _batch_input_img_tensor.to(device, non_blocking=True)
            if _batch_input_img_tensor.dim() == 3:
                _batch_input_img_tensor = _batch_input_img_tensor.unsqueeze(0)

        batch_size = _batch_output_img.shape[0]
        use_cross_atten_mask = utils.build_cross_atten_mask_from_batch_type(_batch_type, batch_size, device)

        moments_256 = autoencoder(_batch_output_img, fn='encode_moments').detach()
        _z = autoencoder.sample(moments_256)

        input_moments_256 = autoencoder(_batch_input_img_tensor, fn='encode_moments').detach()
        _input_image_latent = autoencoder.sample(input_moments_256)

        _batch_con, _batch_mask = utils.get_input_image_embeddings_and_masks(
            batch_input_images=_batch_input_img,
            vl_chat_processor=vl_chat_processor,
            vl_gpt=vl_gpt,
            device=device,
            question="",
            num_image_tokens=576,
            output_tokens=576,
            accelerator=accelerator,
            cached_input_ids=cached_training_input_ids,
        )
        loss, loss_dict = _flow_mathcing_model(
            _z, nnet,
            loss_coeffs=config.loss_coeffs,
            cond=_batch_con, con_mask=_batch_mask,
            batch_img_clip=_batch_output_img,
            nnet_style=config.nnet.name,
            text_token=None,
            model_config=config.nnet.model_args,
            all_config=config,
            training_step=train_state.step,
            image_latent=_input_image_latent,
            use_cross_atten_mask=use_cross_atten_mask,
        )

        _metrics['loss'] = accelerator.gather(loss.detach()).mean()
        for key in loss_dict.keys():
            _metrics[key] = accelerator.gather(loss_dict[key].detach()).mean()
        accelerator.backward(loss.mean())
        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    def ode_fm_solver_sample(nnet_ema, _n_samples, _sample_steps, context=None, token_mask=None, image_latent=None, use_cross_atten_mask=None):
        with torch.no_grad():
            _z_gaussian = torch.randn(_n_samples, *config.z_shape, device=device)
                
            _z_x0, _mu, _log_var = nnet_ema(context, text_encoder = True, shape = _z_gaussian.shape, mask=token_mask)
            _z_init = _z_x0.reshape(_z_gaussian.shape)
            
            assert config.sample.scale > 1
            _cfg = config.sample.scale

            has_null_indicator = hasattr(config.nnet.model_args, "cfg_indicator")

            ode_solver = ODEEulerFlowMatchingSolver(nnet_ema, step_size_type="step_in_dsigma", guidance_scale=_cfg)
            _z, _ = ode_solver.sample(
                x_T=_z_init, batch_size=_n_samples, sample_steps=_sample_steps,
                unconditional_guidance_scale=_cfg, has_null_indicator=has_null_indicator,
                image_latent=image_latent, use_cross_atten_mask=use_cross_atten_mask,
            )

            image_unprocessed = decode(_z)
            return image_unprocessed

    def eval_step(n_samples, sample_steps):
        # ensure n_samples is not greater than the size of the test dataset
        if hasattr(test_dataset, 'num_samples'):
            test_dataset_size = test_dataset.num_samples
            if n_samples > test_dataset_size:
                logging.warning(f"n_samples ({n_samples}) is greater than the size of the test dataset ({test_dataset_size}), using the test dataset size")
                n_samples = test_dataset_size
        else:
            logging.info(f"Skip dataset size check, using n_samples={n_samples}")
            
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm=ODE_Euler_Flow_Matching_Solver, '
                     f'mini_batch_size={config.sample.mini_batch_size}')
        def sample_fn(_n_samples, return_caption=False, config=None):
            assert not return_caption

            _batch_data = next(context_generator)
            if len(_batch_data) == 4:
                _input_img, _output_img, _input_img_tensor, _batch_type = _batch_data
            else:
                _input_img, _output_img, _input_img_tensor = _batch_data
                _batch_type = None

            # keep a copy for visualisation before moving to device
            _input_img_for_vis = _input_img_tensor.clone() if isinstance(_input_img_tensor, torch.Tensor) else _input_img_tensor

            if isinstance(_input_img_tensor, torch.Tensor):
                _input_img_tensor = _input_img_tensor.to(device, non_blocking=True)
                if _input_img_tensor.dim() == 3:
                    _input_img_tensor = _input_img_tensor.unsqueeze(0)

            input_moments_256 = autoencoder(_input_img_tensor, fn='encode_moments').detach()
            _input_image_latent = autoencoder.sample(input_moments_256)

            _context, _token_mask = utils.get_input_image_embeddings_and_masks(
                batch_input_images=_input_img,
                vl_chat_processor=vl_chat_processor,
                vl_gpt=vl_gpt,
                device=device,
                question="",
                num_image_tokens=576,
                output_tokens=576,
                accelerator=accelerator,
                cached_input_ids=cached_training_input_ids,
            )

            assert _context.size(0) == _n_samples
            use_cross_atten_mask = utils.build_cross_atten_mask_from_batch_type(_batch_type, _n_samples, device)

            generated_samples = ode_fm_solver_sample(
                nnet_ema, _n_samples, sample_steps,
                context=_context, token_mask=_token_mask,
                image_latent=_input_image_latent, use_cross_atten_mask=use_cross_atten_mask,
            )

            if isinstance(_output_img, torch.Tensor):
                _output_img = _output_img.to(device, non_blocking=True)
                if _output_img.dim() == 3:
                    _output_img = _output_img.unsqueeze(0)

            if isinstance(_input_img_for_vis, torch.Tensor):
                _input_img_for_vis = _input_img_for_vis.to(device, non_blocking=True)
                if _input_img_for_vis.dim() == 3:
                    _input_img_for_vis = _input_img_for_vis.unsqueeze(0)
                if _input_img_for_vis.size(0) != _n_samples:
                    if _input_img_for_vis.size(0) < _n_samples:
                        pad = _input_img_for_vis[-1:].expand(_n_samples - _input_img_for_vis.size(0), -1, -1, -1)
                        _input_img_for_vis = torch.cat([_input_img_for_vis, pad], dim=0)
                    else:
                        _input_img_for_vis = _input_img_for_vis[:_n_samples]

            return generated_samples, _output_img, _input_img_for_vis
        
        with tempfile.TemporaryDirectory() as temp_path:
            path = config.sample.path or temp_path
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)
            utils.sample2dir_with_gt(accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess, config=config)
            _fid = 0
            if accelerator.is_main_process:
                inception_ckpt_path = getattr(config, 'inception_ckpt_path', None)
                _fid = calculate_fid_given_paths((dataset.fid_stat, path), inception_ckpt_path=inception_ckpt_path)

                logging.info(f'step={train_state.step} fid{n_samples}={_fid}')
                with open(os.path.join(config.workdir, 'eval.log'), 'a') as f:
                    print(f'step={train_state.step} fid{n_samples}={_fid}', file=f)
                wandb.log({f'fid{n_samples}': _fid}, step=train_state.step)
                eval_images = []
                for i in range(n_samples):
                    img_path = os.path.join(path, f"{i}.png")
                    if os.path.exists(img_path):
                        img_pil = Image.open(img_path).convert("RGB")
                        eval_images.append(wandb.Image(img_pil, caption=f"eval_sample_{i}_step_{train_state.step}"))
                
                if eval_images:
                    wandb.log({f'eval_samples_{n_samples}_step_{train_state.step}': eval_images}, step=train_state.step)
                    logging.info(f'Uploaded {len(eval_images)} evaluation samples to wandb at step {train_state.step}')
            _fid = torch.tensor(_fid, device=device)
            _fid = accelerator.reduce(_fid, reduction='sum')

        return _fid.item()

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')

    step_fid = []
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = next(data_generator)
        if len(batch) == 3:
            batch = (batch[0], batch[1], batch[2], None)
        elif len(batch) != 4:
            raise ValueError(f"Unexpected batch length: {len(batch)}, expected 3 or 4")
        
        if isinstance(batch[1], torch.Tensor) and batch[1].device != device:
            batch = (batch[0], batch[1].to(device, non_blocking=True), 
                    batch[2].to(device, non_blocking=True) if isinstance(batch[2], torch.Tensor) else batch[2],
                    batch[3])
        elif isinstance(batch[2], torch.Tensor) and batch[2].device != device:
            batch = (batch[0], batch[1], batch[2].to(device, non_blocking=True), batch[3])

        metrics = train_step(batch)

        nnet.eval()
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
            wandb.log(metrics, step=train_state.step)

        # save rigid image
        if train_state.step % config.train.eval_interval == 0:
            torch.cuda.empty_cache()
            logging.info('Save a grid of images...')
            if not hasattr(dataset, "vis_image_paths"):
                raise NotImplementedError()

            vis_image_paths = dataset.vis_image_paths[:config.train.n_samples_eval]
            use_cross_atten_mask = utils.build_cross_atten_mask_from_paths(vis_image_paths, device)
            vis_input_image_latent = utils.load_images_as_latents(
                vis_image_paths, config.dataset.resolution, autoencoder, device
            )
            contexts, token_mask = utils.get_input_image_embeddings_and_masks(
                batch_input_images=vis_image_paths,
                vl_chat_processor=vl_chat_processor,
                vl_gpt=vl_gpt,
                device=device,
                question="",
                num_image_tokens=576,
                output_tokens=576,
                accelerator=accelerator,
                cached_input_ids=cached_training_input_ids,
            )

            samples = ode_fm_solver_sample(
                nnet_ema, _n_samples=config.train.n_samples_eval, _sample_steps=50,
                context=contexts, token_mask=token_mask,
                image_latent=vis_input_image_latent, use_cross_atten_mask=use_cross_atten_mask,
            )
            samples_unpreprocessed = dataset.unpreprocess(samples)

            if accelerator.is_main_process:
                input_images_pil = dataset.get_vis_images_as_pil(max_images=config.train.n_samples_eval)
                gt_images_pil = (
                    dataset.get_vis_output_images_as_pil(max_images=config.train.n_samples_eval)
                    if hasattr(dataset, "get_vis_output_images_as_pil") else []
                )
                target_device = samples_unpreprocessed[0].device if len(samples_unpreprocessed) > 0 else accelerator.device
                utils.save_vis_grid_and_log(
                    samples_unpreprocessed, input_images_pil, gt_images_pil,
                    config.sample_dir, train_state.step, wandb, target_device,
                )
            accelerator.wait_for_everyone()
            torch.cuda.empty_cache()

        ############ save checkpoint and evaluate results
        if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
            torch.cuda.empty_cache()
            logging.info(f'Save and eval checkpoint {train_state.step}...')
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                utils.clean_stale_ckpt_files(config.ckpt_root)
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                ckpt_path = os.path.join(config.ckpt_root, f'{train_state.step}.ckpt')
                try:
                    train_state.save(ckpt_path)
                except Exception as e:
                    logging.error(f'Failed to save checkpoint at step {train_state.step}: {e}')
                    logging.warning('Continuing training despite checkpoint save failure')

            accelerator.wait_for_everyone()
            fid = eval_step(n_samples=30000, sample_steps=50)  # calculate fid of the saved checkpoint
            step_fid.append((train_state.step, fid))

            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
    logging.info(f'step_fid: {step_fid}')
    step_best = sorted(step_fid, key=lambda x: x[1])[0][0]
    logging.info(f'step_best: {step_best}')
    train_state.load(os.path.join(config.ckpt_root, f'{step_best}.ckpt'))
    del metrics
    accelerator.wait_for_everyone()
    eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps)