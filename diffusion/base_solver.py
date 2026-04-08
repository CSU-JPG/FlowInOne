"""
This file contains the solver base class, including the cfg indicator
"""

import enum
import logging
from collections import defaultdict
from typing import Callable, Dict, List, Union

import numpy as np
import torch

import random


logger = logging.getLogger(__name__)

_default_cfg_processor = {"caption": lambda x, T, t: x}


class ConditionTypes(enum.Enum):
    IMAGE_EMBED: str = "image_conditioning"  # not implemented yet
    TEXT_EMBED: str = "caption"
    HINT_EMBED: str = "hint"  # not implemented yet

class Solver:
    def __init__(
        self,
        model_fn,
        bdv_model_fn=None,
        schedule="linear",
        conditioning_types: List[str] = ["caption"],
        guidance_scale: Union[float, Dict[ConditionTypes, float]] = 1.0,
        cfg_processor: Callable = _default_cfg_processor,
        **kwargs,
    ):
        self.model = model_fn
        self.bdv_model = bdv_model_fn
        self.schedule = schedule
        self.condition_types = [ConditionTypes(el) for el in conditioning_types]

        self.unconditional_guidance_scale = guidance_scale
        if isinstance(guidance_scale, dict):
            self.unconditional_guidance_scale = {
                ConditionTypes(k): v for k, v in guidance_scale.items()
            }
        else:
            self.unconditional_guidance_scale = {
                ConditionTypes.TEXT_EMBED: guidance_scale
            }
        assert all(
            [
                el in self.unconditional_guidance_scale.keys()
                for el in self.condition_types
            ]
        )
        self.cfg_processor = cfg_processor
        if self.cfg_processor is None:
            self.cfg_processor = _default_cfg_processor
        if isinstance(self.cfg_processor, dict):
            assert all(callable(v) for k, v in self.cfg_processor.items())
            self.cfg_processor = {
                ConditionTypes(k): v for k, v in self.cfg_processor.items()
            }
        else:
            assert callable(self.cfg_processor)
            self.cfg_processor = {ConditionTypes.TEXT_EMBED: cfg_processor}

        if self.cfg_processor is not None:
            assert all([el in self.cfg_processor.keys() for el in self.condition_types])
        self.inf_steps_completed = 0

    @property
    def device(self):
        return self.model.device

    def register_buffer(self, name, attr):
        if isinstance(attr, torch.Tensor):
            attr = attr.to(self.device)
        setattr(self, name, attr)

    def _check_the_conditioning(self, conditioning, batch_size):
        # Checks if batch sizes match
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list):
                    ctmp = ctmp[0]
                if isinstance(ctmp, dict):
                    if isinstance(ctmp["c"], list):
                        cbs = ctmp["c"][0].shape[0]
                    else:
                        cbs = ctmp["c"].shape[0]
                else:
                    cbs = ctmp.shape[0]
                if cbs != batch_size:
                    logger.info(
                        f"Warning: Got {cbs} conditionings but batch-size is {batch_size}"
                    )

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        logger.info(
                            f"Warning: Got {ctmp.shape[0]} conditionings but batch-size is {batch_size}"
                        )

            else:
                if conditioning.shape[0] != batch_size:
                    logger.info(
                        f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}"
                    )

    def sample(
        self,
        sample_steps,
        batch_size,
        sampling_method,
        unconditional_guidance_scale,
        has_null_indicator,
        shape=None, # no longer use it
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        verbose=True,
        x_T=None,
        log_every_t=100,
        dynamic_threshold=None,
        ucg_schedule=None,
        t_schedule=None,  # Default value is set below
        skip_type=None,  # Deprecated, kept for backward compatibility. Use `t_schedule` instead.
        start_timestep=None,
        num_timesteps=None,
        do_make_schedule=True,
        **kwargs,
    ):
        self.num_inf_timesteps = sample_steps
        assert skip_type is None

        t_schedule = t_schedule or "time_uniform"

        if self.unconditional_guidance_scale is None:
            self.unconditional_guidance_scale = unconditional_guidance_scale

        assert isinstance(sampling_method, Callable)
        image_latent = kwargs.get("image_latent", None)
        use_cross_atten_mask = kwargs.get("use_cross_atten_mask", None)
        samples, intermediates = sampling_method(
            x_T=x_T,
            # Hardcoded in PLMS file
            ddim_use_original_steps=False,
            callback=callback,
            num_timesteps=num_timesteps,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            img_callback=img_callback,
            log_every_t=log_every_t,
            temperature=temperature,
            noise_dropout=noise_dropout,
            unconditional_guidance_scale=unconditional_guidance_scale,
            has_null_indicator=has_null_indicator,
            dynamic_threshold=dynamic_threshold,
            verbose=verbose,
            ucg_schedule=ucg_schedule,
            start_timestep=start_timestep,
            image_latent=image_latent,  
            use_cross_atten_mask=use_cross_atten_mask,  
        )
        return samples, intermediates

    @torch.no_grad()
    def get_model_output_dimr(
        self,
        x,
        t_continuous,
        unconditional_guidance_scale,
        has_null_indicator,
        image_latent=None,
        use_cross_atten_mask=None,
    ):

        log_snr = 4 - t_continuous * 8 # inversed

        if has_null_indicator:
            _cond = self.model(x, t=t_continuous, log_snr=log_snr, null_indicator=torch.tensor([False] * x.shape[0]).to(x.device), image_latent=image_latent, use_cross_atten_mask=use_cross_atten_mask)[-1]
            _uncond = self.model(x, t=t_continuous, log_snr=log_snr, null_indicator=torch.tensor([True] * x.shape[0]).to(x.device), image_latent=image_latent, use_cross_atten_mask=use_cross_atten_mask)[-1]
            assert unconditional_guidance_scale > 1
            return _uncond + unconditional_guidance_scale * (_cond - _uncond)
        else:          
            _cond = self.model(x, log_snr=log_snr, image_latent=image_latent, use_cross_atten_mask=use_cross_atten_mask)[-1]
            return _cond