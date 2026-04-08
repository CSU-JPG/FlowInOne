import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchdiffeq
import random

from sde import multi_scale_targets
from diffusion.base_solver import Solver
from torchvision import transforms


def check_zip(*args):
    args = [list(arg) for arg in args]
    length = len(args[0])
    for arg in args:
        assert len(arg) == length
    return zip(*args)


def gather_features(
    image_features,
    text_features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    use_horovod=False
):
    if use_horovod:
        try:
            import horovod.torch as hvd
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
            return all_image_features, all_text_features
        except ImportError:
            logging.warning("Horovod not available, falling back to torch.distributed")
    
    try:
        import torch.distributed as dist
        
        if not dist.is_available() or not dist.is_initialized():
            logging.warning("torch.distributed not initialized, returning original features")
            return image_features, text_features
        
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)
        
        return all_image_features, all_text_features
    except Exception as e:
        logging.warning(f"Error in gather_features: {e}, returning original features")
        return image_features, text_features


class TimeStepSampler:
    """
    Abstract class to sample timesteps for flow matching.
    """

    def sample_time(self, x_start):
        # In flow matching, time is in range [0, 1] and 1 indicates the original image; 0 is pure noise
        # this convention is *REVERSE* of diffusion
        raise NotImplementedError

class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class LogitNormalSampler(TimeStepSampler):
    def __init__(self, normal_mean: float = 0, normal_std: float = 1):
        # follows https://arxiv.org/pdf/2403.03206.pdf
        # sample from a normal distribution
        # pass the output through standard logistic function, i.e., sigmoid
        self.normal_mean = float(normal_mean)
        self.normal_std = float(normal_std)

    @torch.no_grad()
    def sample_time(self, x_start):
        x_normal = torch.normal(
            mean=self.normal_mean,
            std=self.normal_std,
            size=(x_start.shape[0],),
            device=x_start.device,
        )
        x_logistic = torch.nn.functional.sigmoid(x_normal)
        return x_logistic


class FlowMatching(nn.Module):  
    def __init__(
        self,
        sigma_min: float = 1e-5,
        sigma_max: float = 1.0,
        timescale: float = 1.0,
        world_size: int = 1,
        rank: int = 0,
        **kwargs,
    ):
        # LatentDiffusion/DDPM will create too many class variables we do not need
        super().__init__(**kwargs)
        self.time_step_sampler = LogitNormalSampler()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.timescale = timescale

        self.clip_loss = ClipLoss(
            world_size=world_size, 
            rank=rank,
            local_loss=True 
        )
        self.resizer = transforms.Resize(256) # for clip

    def mos(self, err, start_dim=1, con_mask=None):  # mean of square
        if con_mask is not None:
            return (err.pow(2).mean(dim=-1) * con_mask).sum(dim=-1) / con_mask.sum(dim=-1)
        else:
            return err.pow(2).flatten(start_dim=start_dim).mean(dim=-1)

    # model forward and prediction
    def forward(
        self,
        x,
        nnet,
        loss_coeffs,
        cond,
        con_mask,
        nnet_style,
        training_step,
        cond_ori=None,  # not using
        con_mask_ori=None,  # not using
        batch_img_clip=None, # not using
        model_config=None,
        all_config=None,
        text_token=None,
        return_raw_loss=False,
        additional_embeddings=None,
        timesteps: Optional[Tuple[int, int]] = None,
        *args,
        **kwargs,
    ):
        assert timesteps is None, "timesteps must be None"

        timesteps = self.time_step_sampler.sample_time(x)

        if nnet_style == 'dimr':
            if hasattr(model_config, "standard_diffusion") and model_config.standard_diffusion:
                standard_diffusion=True
            else:
                standard_diffusion=False
            return self.p_losses_VAE(
                x, cond, con_mask, timesteps, nnet, batch_img_clip=batch_img_clip, text_token=text_token, loss_coeffs=loss_coeffs, return_raw_loss=return_raw_loss, nnet_style=nnet_style, standard_diffusion=standard_diffusion, all_config=all_config, training_step=training_step, *args, **kwargs
            )
        else:
            raise NotImplementedError

    

    def p_losses_VAE(
        self,
        x_start,
        cond,
        con_mask,
        t,
        nnet,
        loss_coeffs,
        training_step,
        text_token=None,
        nnet_style=None,
        all_config=None,
        batch_img_clip=None,
        return_raw_loss=False,
        additional_embeddings=None,
        standard_diffusion=False,
        noise=None,
        image_latent=None,
        use_cross_atten_mask=None,
    ):

        assert noise is None
        x0, mu, log_var = nnet(cond, text_encoder = True, shape = x_start.shape, mask = con_mask)

        if batch_img_clip.shape[-1] == 512:
            recon_gt = self.resizer(batch_img_clip)
        else:
            recon_gt = batch_img_clip
        recon_gt_clip, logit_scale = nnet(recon_gt, image_clip = True) 
        image_features = recon_gt_clip / recon_gt_clip.norm(dim=-1, keepdim=True)
        text_features = x0 / x0.norm(dim=-1, keepdim=True)
        recons_loss = self.clip_loss(image_features, text_features, logit_scale)

        # kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)
        kld_loss = -0.5 * torch.sum(1 + log_var - (0.3 * mu) ** 6 - log_var.exp(), dim = 1) # slightly different KL loss function: mu -> 0 [(0.3*mu) ** 6] and var -> 1
        kld_loss_weight = 1e-2

        loss_mlp = recons_loss + kld_loss * kld_loss_weight
        
        
        noise = x0.reshape(x_start.shape)
        
        if hasattr(all_config.nnet.model_args, "cfg_indicator"):
            null_indicator = torch.tensor([random.random() < all_config.nnet.model_args.cfg_indicator for _ in range(x_start.shape[0])], dtype=torch.bool, device=x_start.device)
            if null_indicator.sum()<=1:
                null_indicator[null_indicator==True] = False
                assert null_indicator.sum() == 0
                pass
            else:
                target_null = x_start[null_indicator]
                target_null = torch.cat((target_null[1:], target_null[:1]))
                x_start[null_indicator] = target_null
        else:
            null_indicator = None
        

        x_noisy = self.psi(t, x=noise, x1=x_start)
        target_velocity = self.Dt_psi(t, x=noise, x1=x_start)
        log_snr = 4 - t * 8 # compute from timestep : inversed

        prediction = nnet(x_noisy, log_snr = log_snr, null_indicator=null_indicator, image_latent=image_latent, use_cross_atten_mask=use_cross_atten_mask)

        target = multi_scale_targets(target_velocity, levels = len(prediction), scale_correction = True)

        loss_diff = 0
        for pred, coeff in check_zip(prediction, loss_coeffs):
            loss_diff = loss_diff + coeff * self.mos(pred - target[pred.shape[-1]])


        loss = loss_diff + loss_mlp
        
        return loss, {'loss_diff': loss_diff, 'clip_loss': recons_loss, 'kld_loss': kld_loss, 'kld_loss_weight': torch.tensor(kld_loss_weight, device=kld_loss.device), 'clip_logit_scale': logit_scale}
        

    ## flow matching specific functions
    def psi(self, t, x, x1):
        assert (
            t.shape[0] == x.shape[0]
        ), f"Batch size of t and x does not agree {t.shape[0]} vs. {x.shape[0]}"
        assert (
            t.shape[0] == x1.shape[0]
        ), f"Batch size of t and x1 does not agree {t.shape[0]} vs. {x1.shape[0]}"
        assert t.ndim == 1
        t = self.expand_t(t, x)
        return (t * (self.sigma_min / self.sigma_max - 1) + 1) * x + t * x1

    def Dt_psi(self, t: torch.Tensor, x: torch.Tensor, x1: torch.Tensor):
        assert x.shape[0] == x1.shape[0]
        return (self.sigma_min / self.sigma_max - 1) * x + x1

    def expand_t(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t_expanded = t
        while t_expanded.ndim < x.ndim:
            t_expanded = t_expanded.unsqueeze(-1)
        return t_expanded.expand_as(x)

class ODEEulerFlowMatchingSolver(Solver):
    """
    ODE Solver for Flow matching that uses an Euler discretization
    Supports number of time steps at inference
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_size_type = kwargs.get("step_size_type", "step_in_dsigma")
        assert self.step_size_type in ["step_in_dsigma", "step_in_dt"]
        self.sample_timescale = 1.0 - 1e-5

    @torch.no_grad()
    def sample_euler(
        self,
        x_T,
        unconditional_guidance_scale,
        has_null_indicator,
        t=[0, 1.0],
        image_latent=None,
        use_cross_atten_mask=None,
        **kwargs,
    ):
        """
        Euler solver for flow matching.
        Based on https://github.com/VinAIResearch/LFM/blob/main/sampler/karras_sample.py
        """
        t = torch.tensor(t)
        t = t * self.sample_timescale
        sigma_min = 1e-5
        sigma_max = 1.0
        sigma_steps = torch.linspace(
            sigma_min, sigma_max, self.num_time_steps + 1, device=x_T.device
        )
        discrete_time_steps_for_step = torch.linspace(
            t[0], t[1], self.num_time_steps + 1, device=x_T.device
        )
        discrete_time_steps_to_eval_model_at = torch.linspace(
            t[0], t[1], self.num_time_steps, device=x_T.device
        )

        print("num_time_steps : " + str(self.num_time_steps))

        for i in range(self.num_time_steps):
            t_i = discrete_time_steps_to_eval_model_at[i]
            velocity = self.get_model_output_dimr(
                x_T,
                has_null_indicator = has_null_indicator,
                t_continuous = t_i.repeat(x_T.shape[0]),
                unconditional_guidance_scale = unconditional_guidance_scale,
                image_latent=image_latent,
                use_cross_atten_mask=use_cross_atten_mask,
            )
            if self.step_size_type == "step_in_dsigma":
                step_size = sigma_steps[i + 1] - sigma_steps[i]
            elif self.step_size_type == "step_in_dt":
                step_size = (
                    discrete_time_steps_for_step[i + 1]
                    - discrete_time_steps_for_step[i]
                )
            x_T = x_T + velocity * step_size

        intermediates = None
        return x_T, intermediates

    @torch.no_grad()
    def sample(
        self,
        *args,
        **kwargs,
    ):
        assert kwargs.get("ucg_schedule", None) is None
        assert kwargs.get("skip_type", None) is None
        assert kwargs.get("dynamic_threshold", None) is None
        assert kwargs.get("x0", None) is None
        assert kwargs.get("x_T") is not None
        assert kwargs.get("score_corrector", None) is None
        assert kwargs.get("normals_sequence", None) is None
        assert kwargs.get("callback", None) is None
        assert kwargs.get("quantize_x0", False) is False
        assert kwargs.get("eta", 0.0) == 0.0
        assert kwargs.get("mask", None) is None
        assert kwargs.get("noise_dropout", 0.0) == 0.0

        self.num_time_steps = kwargs.get("sample_steps")
        self.x_T_uncon = kwargs.get("x_T_uncon")
        
        image_latent = kwargs.get("image_latent", None)
        use_cross_atten_mask = kwargs.get("use_cross_atten_mask", None)

        if image_latent is not None:
            kwargs["image_latent"] = image_latent
        if use_cross_atten_mask is not None:
            kwargs["use_cross_atten_mask"] = use_cross_atten_mask

        samples, intermediates = super().sample(
            *args,
            sampling_method=self.sample_euler,
            do_make_schedule=False,
            **kwargs,
        )
        return samples, intermediates


class ODEFlowMatchingSolver(Solver):
    """
    ODE Solver for Flow matching that uses `dopri5`
    Does not support number of time steps based control
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_timescale = 1.0 - 1e-5

    # sampling for inference
    @torch.no_grad()
    def sample_transport(
        self,
        x_T,
        unconditional_guidance_scale,
        has_null_indicator,
        t=[0, 1.0],
        ode_opts={},
        **kwargs,
    ):
        num_evals = 0
        t = torch.tensor(t, device=x_T.device)
        if "options" not in ode_opts:
            ode_opts["options"] = {}
        ode_opts["options"]["step_t"] = [self.sample_timescale + 1e-6]

        def ode_func(t, x_T):
            nonlocal num_evals
            num_evals += 1
            model_output = self.get_model_output_dimr(
                x_T,
                has_null_indicator = has_null_indicator,
                t_continuous = t.repeat(x_T.shape[0]),
                unconditional_guidance_scale = unconditional_guidance_scale,
            )
            return model_output

        z = torchdiffeq.odeint(
            ode_func,
            x_T,
            t * self.sample_timescale,
            **{"atol": 1e-5, "rtol": 1e-5, "method": "dopri5", **ode_opts},
        )
        # first dimension of z contains solutions to different timepoints
        # only need the last one (corresponding to t=1, i.e., image)
        z = z[-1]
        intermediates = None
        return z, intermediates

    @torch.no_grad()
    def sample(
        self,
        *args,
        **kwargs,
    ):
        assert kwargs.get("ucg_schedule", None) is None
        assert kwargs.get("skip_type", None) is None
        assert kwargs.get("dynamic_threshold", None) is None
        assert kwargs.get("x0", None) is None
        assert kwargs.get("x_T") is not None
        assert kwargs.get("score_corrector", None) is None
        assert kwargs.get("normals_sequence", None) is None
        assert kwargs.get("callback", None) is None
        assert kwargs.get("quantize_x0", False) is False
        assert kwargs.get("eta", 0.0) == 0.0
        assert kwargs.get("mask", None) is None
        assert kwargs.get("noise_dropout", 0.0) == 0.0
        samples, intermediates = super().sample(
            *args,
            sampling_method=self.sample_transport,
            do_make_schedule=False,
            **kwargs,
        )
        return samples, intermediates