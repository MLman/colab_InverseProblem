"""
Based on: https://github.com/crowsonkb/k-diffusion
"""
import random

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from piq import LPIPS
from torchvision.transforms import RandomCrop
import torchvision.utils as vtils
import wandb
from . import dist_util, logger

from .nn import mean_flat, append_dims, append_zero
from .random_util import get_generator

from PIL import Image

def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = th.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = th.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings


class KarrasDenoiser:
    def __init__(
        self,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        distillation=False,
        loss_norm="lpips",
    ):
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.distillation = distillation
        self.loss_norm = loss_norm
        if loss_norm == "lpips":
            self.lpips_loss = LPIPS(replace_pooling=True, reduction="none")
        self.rho = rho
        self.num_timesteps = 40

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_sigmas(self, sigmas):
        return sigmas

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def training_losses(self, model, x_start, sigmas, augment_labels = None, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        terms = {}

        dims = x_start.ndim
        x_t = x_start + noise * append_dims(sigmas, dims)
        # model_output, denoised = self.denoise(model, x_t, sigmas, **model_kwargs)
        model_output, denoised = self.denoise(model, x_t, sigmas, augment_labels, **model_kwargs)

        snrs = self.get_snr(sigmas)
        weights = append_dims(
            get_weightings(self.weight_schedule, snrs, self.sigma_data), dims
        )
        terms["xs_mse"] = mean_flat((denoised - x_start) ** 2)
        terms["mse"] = mean_flat(weights * (denoised - x_start) ** 2)

        if "vb" in terms:
            terms["loss"] = terms["mse"] + terms["vb"]
        else:
            terms["loss"] = terms["mse"]

        return terms

    def consistency_losses(
        self,
        model,
        x_start,
        num_scales,
        model_kwargs=None,
        target_model=None,
        teacher_model=None,
        teacher_diffusion=None,
        noise=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t):
            return self.denoise(model, x, t, **model_kwargs)[1]

        if target_model:

            @th.no_grad()
            def target_denoise_fn(x, t):
                return self.denoise(target_model, x, t, **model_kwargs)[1]

        else:
            raise NotImplementedError("Must have a target model")

        if teacher_model:

            @th.no_grad()
            def teacher_denoise_fn(x, t):
                return teacher_diffusion.denoise(teacher_model, x, t, **model_kwargs)[1]

        @th.no_grad()
        def heun_solver(samples, t, next_t, x0):
            x = samples
            if teacher_model is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(x, t)

            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)
            if teacher_model is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(samples, next_t)

            next_d = (samples - denoiser) / append_dims(next_t, dims)
            samples = x + (d + next_d) * append_dims((next_t - t) / 2, dims)

            return samples

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            if teacher_model is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(x, t)
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t)

        if teacher_model is None:
            x_t2 = euler_solver(x_t, t, t2, x_start).detach()
        else:
            x_t2 = heun_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)
        if self.loss_norm == "l1":
            diffs = th.abs(distiller - distiller_target)
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2":
            diffs = (distiller - distiller_target) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2-32":
            distiller = F.interpolate(distiller, size=32, mode="bilinear")
            distiller_target = F.interpolate(
                distiller_target,
                size=32,
                mode="bilinear",
            )
            diffs = (distiller - distiller_target) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "lpips":
            if x_start.shape[-1] < 256:
                distiller = F.interpolate(distiller, size=224, mode="bilinear")
                distiller_target = F.interpolate(
                    distiller_target, size=224, mode="bilinear"
                )

            loss = (
                self.lpips_loss(
                    (distiller + 1) / 2.0,
                    (distiller_target + 1) / 2.0,
                )
                * weights
            )
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")

        terms = {}
        terms["loss"] = loss

        return terms

    def progdist_losses(
        self,
        model,
        x_start,
        num_scales,
        model_kwargs=None,
        teacher_model=None,
        teacher_diffusion=None,
        noise=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t):
            return self.denoise(model, x, t, **model_kwargs)[1]

        @th.no_grad()
        def teacher_denoise_fn(x, t):
            return teacher_diffusion.denoise(teacher_model, x, t, **model_kwargs)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t):
            x = samples
            denoiser = teacher_denoise_fn(x, t)
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        @th.no_grad()
        def euler_to_denoiser(x_t, t, x_next_t, next_t):
            denoiser = x_t - append_dims(t, dims) * (x_next_t - x_t) / append_dims(
                next_t - t, dims
            )
            return denoiser

        indices = th.randint(0, num_scales, (x_start.shape[0],), device=x_start.device)

        t = self.sigma_max ** (1 / self.rho) + indices / num_scales * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 0.5) / num_scales * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        t3 = self.sigma_max ** (1 / self.rho) + (indices + 1) / num_scales * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t3 = t3**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        denoised_x = denoise_fn(x_t, t)

        x_t2 = euler_solver(x_t, t, t2).detach()
        x_t3 = euler_solver(x_t2, t2, t3).detach()

        target_x = euler_to_denoiser(x_t, t, x_t3, t3).detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)
        if self.loss_norm == "l1":
            diffs = th.abs(denoised_x - target_x)
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2":
            diffs = (denoised_x - target_x) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "lpips":
            if x_start.shape[-1] < 256:
                denoised_x = F.interpolate(denoised_x, size=224, mode="bilinear")
                target_x = F.interpolate(target_x, size=224, mode="bilinear")
            loss = (
                self.lpips_loss(
                    (denoised_x + 1) / 2.0,
                    (target_x + 1) / 2.0,
                )
                * weights
            )
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")

        terms = {}
        terms["loss"] = loss

        return terms

    # def denoise(self, model, x_t, sigmas, **model_kwargs):
    def denoise(self, model, x_t, sigmas, augment_labels = None, **model_kwargs):
        import torch.distributed as dist

        if not self.distillation:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim) for x in self.get_scalings(sigmas)
            ]
        else:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim)
                for x in self.get_scalings_for_boundary_condition(sigmas)
            ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # model_output = model(c_in * x_t, rescaled_t, **model_kwargs)
        model_output = model(c_in * x_t, rescaled_t, augment_labels, **model_kwargs)
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised


def karras_sample(
    diffusion,
    model,
    shape,
    steps,
    sample_idx=0,
    save_dir='./',
    toy_exp=None,
    dataloader=None,
    lpips_loss=None,
    use_wandb=False,
    clip_denoised=True,
    progress=False,
    callback=None,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80,  # higher for highres?
    rho=7.0,
    sampler="heun",
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    generator=None,
    ts=None,
):  
    if generator is None:
        generator = get_generator("dummy")

    if sampler == "progdist":
        sigmas = get_sigmas_karras(steps + 1, sigma_min, sigma_max, rho, device=device)
    else:
        sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)

    x_T = generator.randn(*shape, device=device) * sigma_max

    if toy_exp is not None:
        sampler = f'{sampler}_{toy_exp}' # heun_forwardbackward

    sample_fn = {
        "heun": sample_heun,
        "dpm": sample_dpm,
        "ancestral": sample_euler_ancestral,
        "onestep": sample_onestep,
        "progdist": sample_progdist,
        "euler": sample_euler,
        "multistep": stochastic_iterative_sampler,
        "heun_forward_backward": sample_heun_forward_backward,
    }[sampler]

    if sampler in ["heun", "dpm"]:
        sampler_args = dict(
            s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise
        )
    elif sampler == "multistep":
        sampler_args = dict(
            ts=ts, t_min=sigma_min, t_max=sigma_max, rho=diffusion.rho, steps=steps
        )
    else:
        sampler_args = {}

    toy_args = dict(
            sample_idx=sample_idx, save_dir=save_dir,
        )
    sampler_args.update(toy_args)
    
    if toy_exp is not None:
        batch, cond = next(dataloader)
        batch_size = batch[0].shape[0]
        lpips_loss = LPIPS(replace_pooling=True, reduction="none")
        
        sampler_args.update(dict(batch=batch, cond=cond, batch_size=batch_size, device=device,\
                                 lpips_loss=lpips_loss, use_wandb=use_wandb,
                                 ))

        '''Visualize sanity check of dataloader'''
        # sharp_sample, blur_sample = batch[:][0], batch[:][1]
        # vtils.save_image(sharp_sample, f'{save_dir}/sharp_samples.png', normalize=True)
        # vtils.save_image(blur_sample, f'{save_dir}/blur_samples.png', normalize=True)
        
        '''Visualize each sample respectively'''
        # sample = th.cat(batch, 0)
        # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # sample = sample.permute(0, 2, 3, 1)
        # sample = sample.contiguous()
        # sample = sample.cpu().numpy()

        # for idx in range(batch_size // 2):
        #     img_sharp = Image.fromarray(sample[idx])
        #     img_blur = Image.fromarray(sample[idx + batch_size])
        #     sample_name = sample_idx + idx

        #     img_sharp.save(f'{save_dir}/{sample_name}_sharp_step{0}.png')
        #     img_blur.save(f'{save_dir}/{sample_name}_blur_step{0}.png')
            

    def denoiser(x_t, sigma):
        _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    x_0 = sample_fn(
        denoiser,
        x_T,
        sigmas,
        generator,
        progress=progress,
        callback=callback,
        **sampler_args,
    )
    return x_0.clamp(-1, 1)


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = th.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)



@th.no_grad()
def sample_heun_forward_backward(
    denoiser,
    x,
    sigmas,
    generator,
    batch, 
    cond, 
    batch_size,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    sample_idx=0,
    save_dir='./',
    device=None,
    lpips_loss=None,
    use_wandb=False,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)
    
    x_sharp = batch[:][0].to(device)
    x_blur = batch[:][1].to(device)
    vtils.save_image(x_sharp, f'{save_dir}/{sample_idx}_original_sharp.png', normalize=True)
    vtils.save_image(x_blur, f'{save_dir}/{sample_idx}_original_blur.png', normalize=True)

    x_sharp_ori = x_sharp.clone()
    x_blur_ori = x_blur.clone()

    x_sharp_f = x_sharp.clone()
    x_blur_f = x_blur.clone()

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )

        f_i = len(sigmas) - 1 - i
        gamma_f = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[f_i] <= s_tmax
            else 0.0
        )

        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        sigma_hat_f = sigmas[f_i] * (gamma_f + 1)

        logger.log(f'sigma_hat {sigma_hat}')
        logger.log(f'sigma_hat_f {sigma_hat_f}')

        # if gamma > 0:
        #     x_sharp = x_sharp + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        #     x_blur = x_blur + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

        # if gamma_f > 0:
        #     x_sharp_f = x_sharp_f + eps * (sigma_hat_f**2 - sigmas[f_i] ** 2) ** 0.5
        #     x_blur_f = x_blur_f + eps * (sigma_hat_f**2 - sigmas[f_i] ** 2) ** 0.5
            
        # denoised_sharp = denoiser(x_sharp, sigma_hat * s_in)
        # denoised_blur = denoiser(x_blur, sigma_hat * s_in)
        # d_sharp = to_d(x_sharp, sigma_hat, denoised_sharp)
        # d_blur = to_d(x_blur, sigma_hat, denoised_blur)

        denoised_sharp = denoiser(x_sharp, sigma_hat * s_in)
        denoised_blur = denoiser(x_blur, sigma_hat * s_in)
        d_sharp = to_d(x_sharp, sigma_hat, denoised_sharp)
        d_blur = to_d(x_blur, sigma_hat, denoised_blur)
        vtils.save_image(denoised_sharp, f'{save_dir}/{sample_idx}_denoised_sharp_step{i}.png', normalize=True)
        vtils.save_image(denoised_blur, f'{save_dir}/{sample_idx}_denoised_blur_step{i}.png', normalize=True)
        
        denoised_sharp_f = denoiser(x_sharp_f, sigma_hat_f * s_in)
        denoised_blur_f = denoiser(x_blur_f, sigma_hat_f * s_in)
        d_sharp_f = to_d(x_sharp_f, sigma_hat_f, denoised_sharp_f)
        d_blur_f = to_d(x_blur_f, sigma_hat_f, denoised_blur_f)
        vtils.save_image(denoised_sharp_f, f'{save_dir}/{sample_idx}_denoised_sharp_f_step{i}.png', normalize=True)
        vtils.save_image(denoised_blur_f, f'{save_dir}/{sample_idx}_denoised_blur_f_step{i}.png', normalize=True)
        
        dt = sigmas[i + 1] - sigma_hat
        dt_f = sigmas[f_i-1] - sigma_hat_f

        if sigmas[i + 1] == 0:
            # Euler method
            x_sharp = x_sharp + d_sharp * dt
            x_blur = x_blur + d_sharp * dt
        else:
            # Heun's method
            x_2_sharp = x_sharp + d_sharp * dt
            x_2_blur = x_blur + d_blur * dt
            denoised_2_sharp = denoiser(x_2_sharp, sigmas[i + 1] * s_in)
            denoised_2_blur = denoiser(x_2_blur, sigmas[i + 1] * s_in)
            d_2_sharp = to_d(x_2_sharp, sigmas[i + 1], denoised_2_sharp)
            d_2_blur = to_d(x_2_blur, sigmas[i + 1], denoised_2_blur)
            d_prime_sharp = (d_sharp + d_2_sharp) / 2
            d_prime_blur = (d_blur + d_2_blur) / 2
            x_sharp = x_sharp + d_prime_sharp * dt
            x_blur = x_blur + d_prime_blur * dt


        if sigmas[f_i - 1] == 0:
            x_sharp_f = x_sharp_f - d_sharp_f * dt_f
            x_blur_f = x_blur_f - d_blur_f * dt_f
        else:
            x_2_sharp_f = x_sharp_f - d_sharp_f * dt_f
            x_2_blur_f = x_blur_f - d_blur_f * dt_f
            denoised_sharp_f = denoiser(x_2_sharp_f, sigmas[f_i-1] * s_in)
            denoised_blur_f = denoiser(x_2_blur_f, sigmas[f_i-1] * s_in)
            d_2_sharp_f = to_d(x_2_sharp_f, sigmas[f_i-1], denoised_sharp_f)
            d_2_blur_f = to_d(x_2_blur_f, sigmas[f_i-1], denoised_blur_f)
            d_prime_sharp_f = (d_sharp_f + d_2_sharp_f) / 2
            d_prime_blur_f = (d_blur_f + d_2_blur_f) / 2
            x_sharp_f = x_sharp_f - d_prime_sharp_f * dt_f
            x_blur_f = x_blur_f - d_prime_blur_f * dt_f
            

        loss_sharp = lpips_loss((x_sharp + 1) / 2.0, (x_sharp_ori + 1) / 2.0,)
        loss_blur = lpips_loss((x_blur + 1) / 2.0, (x_blur_ori + 1) / 2.0,)

        diff_sharp = th.abs(denoised_sharp - x_sharp).mean()
        diff_blur = th.abs(denoised_blur - x_blur).mean()
    
    
    
        if use_wandb:
            wandb.log(dict(
                        diff_sharp=diff_sharp, diff_blur=diff_blur,\
                        # diff_sharp_sq=diff_sharp_sq, diffdiff_blur_sq_blur=diff_blur_sq,\
                        # diff_x_0_sharp=diff_sharp_ori, diff_x_0_blur=diff_blur_ori, \
                        ))
            
        # logger.logkv("diff_sharp", diff_sharp)
        # logger.logkv("diff_blur", diff_blur)

        vtils.save_image(x_sharp, f'{save_dir}/{sample_idx}_sharp_step{i}.png', normalize=True)
        vtils.save_image(x_blur, f'{save_dir}/{sample_idx}_blur_step{i}.png', normalize=True)


        # vtils.save_image(x_sharp_f, f'{save_dir}/{sample_idx}_sharp_forward_step{f_i}.png', normalize=True)
        # vtils.save_image(x_blur_f, f'{save_dir}/{sample_idx}_blur_forward_step{f_i}.png', normalize=True)

    # for i in indices:
    #     gamma = (
    #         min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
    #         if s_tmin <= sigmas[len(sigmas)-1-i ] <= s_tmax
    #         else 0.0
    #     )

    #     eps = generator.randn_like(x_sharp) * s_noise
    #     sigma_hat = sigmas[len(sigmas)-1-i ] * (gamma + 1)
    #     if gamma > 0:
    #         x_sharp = x_sharp + eps * (sigma_hat**2 - sigmas[len(sigmas)-1-i ] ** 2) ** 0.5
    #         x_blur = x_blur + eps * (sigma_hat**2 - sigmas[len(sigmas)-1-i ] ** 2) ** 0.5
            
    #     denoised_sharp = denoiser(x_sharp, sigma_hat * s_in)
    #     denoised_blur = denoiser(x_blur, sigma_hat * s_in)
    #     d_sharp = to_d(x_sharp, sigma_hat, denoised_sharp)
    #     d_blur = to_d(x_blur, sigma_hat, denoised_blur)

    #     dt = sigmas[len(sigmas)-1-i ] - sigma_hat
    #     if sigmas[len(sigmas)-1-i ] == 0:
    #         # Euler method
    #         x_sharp = x_sharp + d_sharp * dt
    #         x_blur = x_blur + d_blur * dt
    #     else:
    #         # Heun's method
    #         x_2_sharp = x_sharp + d_sharp * dt
    #         x_2_blur = x_blur + d_blur * dt
            
    #         denoised_2_sharp = denoiser(x_2_sharp, sigmas[len(sigmas)-1-i ] * s_in)
    #         denoised_2_blur = denoiser(x_2_blur, sigmas[len(sigmas)-1-i ] * s_in)
    #         d_2_sharp = to_d(x_2_sharp, sigmas[len(sigmas)-1-i ], denoised_2_sharp)
    #         d_2_blur = to_d(x_2_blur, sigmas[len(sigmas)-1-i ], denoised_2_blur)
    #         d_prime_sharp = (d_sharp + d_2_sharp) / 2
    #         d_prime_blur = (d_blur + d_2_blur) / 2
    #         x_sharp = x_sharp + d_prime_sharp * dt
    #         x_blur = x_blur + d_prime_blur * dt
            
    #     vtils.save_image(x_sharp, f'{save_dir}/{sample_idx}_fianl_sharp_step{i}.png', normalize=True)
    #     vtils.save_image(x_blur, f'{save_dir}/{sample_idx}_final_blur_step{i}.png', normalize=True)

                    
    return x


# @th.no_grad()
# def sample_heun_forward_backward(
#     denoiser,
#     x,
#     sigmas,
#     generator,
#     batch, 
#     cond, 
#     batch_size,
#     progress=False,
#     callback=None,
#     s_churn=0.0,
#     s_tmin=0.0,
#     s_tmax=float("inf"),
#     s_noise=1.0,
#     sample_idx=0,
#     save_dir='./',
#     device=None,
# ):
    
#     # @th.no_grad()
#     # def euler_solver(samples, t, next_t, denoiser):
#     #     x = samples
#     #     dims = samples.ndim
#     #     denoised = denoiser(x, t)
#     #     d = (x - denoised) / append_dims(t, dims)
#     #     samples = x + d * append_dims(next_t - t, dims)

#     #     return samples

#     """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
#     s_in = x.new_ones([x.shape[0]])
#     indices = range(len(sigmas) - 1)
#     if progress:
#         from tqdm.auto import tqdm

#         indices = tqdm(indices)
    
#     x_sharp = batch[:][0].to(device)
#     x_blur = batch[:][1].to(device)
#     x_sharp2 = x_sharp.clone()
#     x_blur2 = x_blur.clone()
#     x_sharp_ori = x_sharp.clone()
#     x_blur_ori = x_blur.clone()
#     vtils.save_image(x_sharp, f'{save_dir}/{sample_idx}_original_sharp.png', normalize=True)
#     vtils.save_image(x_blur, f'{save_dir}/{sample_idx}_original_blur.png', normalize=True)

#     # x_sharp_f = x_sharp.clone()
#     # x_blur_f = x_blur.clone()
#     # import pdb; pdb.set_trace()
#     logger.log(f"Forward/Backward process")
#     # Compare one step forward / backward 
#     for i in indices:
#         gamma = (
#             min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
#             if s_tmin <= sigmas[i] <= s_tmax
#             else 0.0
#         )

#         # f_i = len(sigmas)-1-i
#         # gamma_f = (
#         #     min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
#         #     if s_tmin <= sigmas[f_i] <= s_tmax
#         #     else 0.0
#         # )

#         eps = generator.randn_like(x) * s_noise
#         sigma_hat = sigmas[i] * (gamma + 1)
#         # sigma_hat_f = sigmas[f_i] * (gamma_f + 1)
#         if gamma > 0:
#             x_sharp = x_sharp + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
#             x_blur = x_blur + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
#             x_sharp2 = x_sharp2 + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
#             x_blur2 = x_blur2 + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

#         # if gamma_f > 0:
#             # x_sharp_f = x_sharp_f + eps * (sigma_hat_f**2 - sigmas[f_i] ** 2) ** 0.5
#             # x_blur_f = x_blur_f + eps * (sigma_hat_f**2 - sigmas[f_i] ** 2) ** 0.5
            
#         denoised_sharp = denoiser(x_sharp, sigma_hat * s_in)
#         denoised_blur = denoiser(x_blur, sigma_hat * s_in)

#         denoised_sharp2 = denoiser(x_sharp2, sigma_hat * s_in)
#         denoised_blur2 = denoiser(x_blur2, sigma_hat * s_in)
        
#         # denoised_sharp_f = denoiser(x_sharp_f, sigma_hat_f * s_in)
#         # denoised_blur_f = denoiser(x_blur_f, sigma_hat_f * s_in)
        
#         d_sharp = to_d(x_sharp, sigma_hat, denoised_sharp)
#         d_blur = to_d(x_blur, sigma_hat, denoised_blur)

#         d_sharp2 = to_d(x_sharp2, sigma_hat, denoised_sharp2)
#         d_blur2 = to_d(x_blur2, sigma_hat, denoised_blur2)

#         # d_sharp_f = to_d(x_sharp_f, sigma_hat_f, denoised_sharp_f)
#         # d_blur_f = to_d(x_blur_f, sigma_hat_f, denoised_blur_f)
        
#         dt = sigmas[i + 1] - sigma_hat
#         # dt_f = sigmas[f_i-1] - sigma_hat_f
#         if sigmas[i + 1] == 0:
#             # Euler method
#             # x = x + d * dt
#             x_sharp = x_sharp + d_sharp * dt
#             x_blur = x_blur + d_sharp * dt

#             x_sharp2 = x_sharp2 - d_sharp2 * dt
#             x_blur2 = x_blur2 - d_sharp2 * dt
            
#             # x_sharp_f = x_sharp_f - d_sharp_f * dt_f
#             # x_blur_f = x_blur_f - d_blur_f * dt_f
#         else:
#             # Heun's method
#             x_2_sharp = x_sharp + d_sharp * dt
#             x_2_blur = x_blur + d_blur * dt

#             x_2_sharp2 = x_sharp2 - d_sharp2 * dt
#             x_2_blur2 = x_blur2 - d_blur2 * dt
            
#             # x_2_sharp_f = x_sharp_f - d_sharp_f * dt_f
#             # x_2_blur_f = x_blur_f - d_blur_f * dt_f

#             denoised_sharp = denoiser(x_2_sharp, sigmas[i + 1] * s_in)
#             denoised_blur = denoiser(x_2_blur, sigmas[i + 1] * s_in)

#             denoised_sharp2 = denoiser(x_2_sharp2, sigmas[i + 1] * s_in)
#             denoised_blur2 = denoiser(x_2_blur2, sigmas[i + 1] * s_in)
            
#             # denoised_sharp_f = denoiser(x_2_sharp_f, sigmas[f_i-1] * s_in)
#             # denoised_blur_f = denoiser(x_2_blur_f, sigmas[f_i-1] * s_in)
            
#             d_2_sharp = to_d(x_2_sharp, sigmas[i + 1], denoised_sharp)
#             d_2_blur = to_d(x_2_blur, sigmas[i + 1], denoised_blur)
            
#             d_2_sharp2 = to_d(x_2_sharp2, sigmas[i + 1], denoised_sharp2)
#             d_2_blur2 = to_d(x_2_blur2, sigmas[i + 1], denoised_blur2)
            
#             # d_2_sharp_f = to_d(x_2_sharp_f, sigmas[f_i-1], denoised_sharp_f)
#             # d_2_blur_f = to_d(x_2_blur_f, sigmas[f_i-1], denoised_blur_f)

#             d_prime_sharp = (d_sharp + d_2_sharp) / 2
#             d_prime_blur = (d_blur + d_2_blur) / 2
            
#             d_prime_sharp2 = (d_sharp2 + d_2_sharp2) / 2
#             d_prime_blur2 = (d_blur2 + d_2_blur2) / 2

#             # d_prime_sharp_f = (d_sharp_f + d_2_sharp_f) / 2
#             # d_prime_blur_f = (d_blur_f + d_2_blur_f) / 2
            
#             x_sharp = x_sharp + d_prime_sharp * dt
#             x_blur = x_blur + d_prime_blur * dt
            
#             x_sharp2 = x_sharp2 - d_prime_sharp2 * dt
#             x_blur2 = x_blur2 - d_prime_blur2 * dt
            
#             # x_sharp_f = x_sharp_f - d_prime_sharp_f * dt_f
#             # x_blur_f = x_blur_f - d_prime_blur_f * dt_f
            
#         diff_sharp = th.abs(denoised_sharp - x_sharp).mean()
#         diff_blur = th.abs(denoised_blur - x_blur).mean()
    
#         diff_sharp_ori = th.abs(denoised_sharp - x_sharp_ori).mean()
#         diff_blur_ori = th.abs(denoised_blur - x_blur_ori).mean()
    
    
#         diff_sharp_sq = th.square(diff_sharp)
#         diff_blur_sq = th.square(diff_blur)
#         wandb.log(dict(
#                     diff_sharp=diff_sharp, diff_blur=diff_blur,\
#                     diff_sharp_sq=diff_sharp_sq, diffdiff_blur_sq_blur=diff_blur_sq,\
#                     diff_x_0_sharp=diff_sharp_ori, diff_x_0_blur=diff_blur_ori, \
#                     ))
        
#         # logger.logkv("diff_sharp", diff_sharp)
#         # logger.logkv("diff_blur", diff_blur)

#         vtils.save_image(x_sharp, f'{save_dir}/{sample_idx}_sharp_step{i}.png', normalize=True)
#         vtils.save_image(x_blur, f'{save_dir}/{sample_idx}_blur_step{i}.png', normalize=True)

#         vtils.save_image(x_sharp2, f'{save_dir}/{sample_idx}_sharp_forward_step{i}.png', normalize=True)
#         vtils.save_image(x_blur2, f'{save_dir}/{sample_idx}_blur_forward_step{i}.png', normalize=True)


#         # vtils.save_image(x_sharp_f, f'{save_dir}/{sample_idx}_sharp_forward_step{f_i}.png', normalize=True)
#         # vtils.save_image(x_blur_f, f'{save_dir}/{sample_idx}_blur_forward_step{f_i}.png', normalize=True)

#     # for i in indices:
#     #     gamma = (
#     #         min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
#     #         if s_tmin <= sigmas[len(sigmas)-1-i ] <= s_tmax
#     #         else 0.0
#     #     )

#     #     eps = generator.randn_like(x_sharp) * s_noise
#     #     sigma_hat = sigmas[len(sigmas)-1-i ] * (gamma + 1)
#     #     if gamma > 0:
#     #         x_sharp = x_sharp + eps * (sigma_hat**2 - sigmas[len(sigmas)-1-i ] ** 2) ** 0.5
#     #         x_blur = x_blur + eps * (sigma_hat**2 - sigmas[len(sigmas)-1-i ] ** 2) ** 0.5
            
#     #     denoised_sharp = denoiser(x_sharp, sigma_hat * s_in)
#     #     denoised_blur = denoiser(x_blur, sigma_hat * s_in)
#     #     d_sharp = to_d(x_sharp, sigma_hat, denoised_sharp)
#     #     d_blur = to_d(x_blur, sigma_hat, denoised_blur)

#     #     dt = sigmas[len(sigmas)-1-i ] - sigma_hat
#     #     if sigmas[len(sigmas)-1-i ] == 0:
#     #         # Euler method
#     #         x_sharp = x_sharp + d_sharp * dt
#     #         x_blur = x_blur + d_blur * dt
#     #     else:
#     #         # Heun's method
#     #         x_2_sharp = x_sharp + d_sharp * dt
#     #         x_2_blur = x_blur + d_blur * dt
            
#     #         denoised_2_sharp = denoiser(x_2_sharp, sigmas[len(sigmas)-1-i ] * s_in)
#     #         denoised_2_blur = denoiser(x_2_blur, sigmas[len(sigmas)-1-i ] * s_in)
#     #         d_2_sharp = to_d(x_2_sharp, sigmas[len(sigmas)-1-i ], denoised_2_sharp)
#     #         d_2_blur = to_d(x_2_blur, sigmas[len(sigmas)-1-i ], denoised_2_blur)
#     #         d_prime_sharp = (d_sharp + d_2_sharp) / 2
#     #         d_prime_blur = (d_blur + d_2_blur) / 2
#     #         x_sharp = x_sharp + d_prime_sharp * dt
#     #         x_blur = x_blur + d_prime_blur * dt
            
#     #     vtils.save_image(x_sharp, f'{save_dir}/{sample_idx}_fianl_sharp_step{i}.png', normalize=True)
#     #     vtils.save_image(x_blur, f'{save_dir}/{sample_idx}_final_blur_step{i}.png', normalize=True)

                    
#     return x

# @th.no_grad()
# def sample_heun_forward_backward(
#     denoiser,
#     x,
#     sigmas,
#     generator,
#     batch, 
#     cond, 
#     batch_size,
#     progress=False,
#     callback=None,
#     s_churn=0.0,
#     s_tmin=0.0,
#     s_tmax=float("inf"),
#     s_noise=1.0,
#     sample_idx=0,
#     save_dir='./',
#     device=None,
# ):
    
#     # @th.no_grad()
#     # def euler_solver(samples, t, next_t, denoiser):
#     #     x = samples
#     #     dims = samples.ndim
#     #     denoised = denoiser(x, t)
#     #     d = (x - denoised) / append_dims(t, dims)
#     #     samples = x + d * append_dims(next_t - t, dims)

#     #     return samples

#     """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
#     s_in = x.new_ones([x.shape[0]])
#     indices = range(len(sigmas) - 1)
#     if progress:
#         from tqdm.auto import tqdm

#         indices = tqdm(indices)
    
#     x_sharp = batch[:][0].to(device)
#     x_blur = batch[:][1].to(device)
#     x_sharp2 = x_sharp.clone()
#     x_blur2 = x_blur.clone()
#     # x_sharp_f = x_sharp.clone()
#     # x_blur_f = x_blur.clone()
#     # import pdb; pdb.set_trace()
#     logger.log(f"Forward/Backward process")
#     # Compare one step forward / backward 
#     for i in indices:
#         gamma = (
#             min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
#             if s_tmin <= sigmas[len(sigmas)-1-i] <= s_tmax
#             else 0.0
#         )

#         # f_i = len(sigmas)-1-i
#         # gamma_f = (
#         #     min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
#         #     if s_tmin <= sigmas[f_i] <= s_tmax
#         #     else 0.0
#         # )

#         eps = generator.randn_like(x) * s_noise
#         sigma_hat = sigmas[len(sigmas)-1-i] * (gamma + 1)
#         # sigma_hat_f = sigmas[f_i] * (gamma_f + 1)
#         if gamma > 0:
#             x_sharp = x_sharp + eps * (sigma_hat**2 - sigmas[len(sigmas)-1-i] ** 2) ** 0.5
#             x_blur = x_blur + eps * (sigma_hat**2 - sigmas[len(sigmas)-1-i] ** 2) ** 0.5
#             x_sharp2 = x_sharp2 + eps * (sigma_hat**2 - sigmas[len(sigmas)-1-i] ** 2) ** 0.5
#             x_blur2 = x_blur2 + eps * (sigma_hat**2 - sigmas[len(sigmas)-1-i] ** 2) ** 0.5

#         # if gamma_f > 0:
#             # x_sharp_f = x_sharp_f + eps * (sigma_hat_f**2 - sigmas[f_i] ** 2) ** 0.5
#             # x_blur_f = x_blur_f + eps * (sigma_hat_f**2 - sigmas[f_i] ** 2) ** 0.5
            
#         denoised_sharp = denoiser(x_sharp, sigma_hat * s_in)
#         denoised_blur = denoiser(x_blur, sigma_hat * s_in)

#         denoised_sharp2 = denoiser(x_sharp2, sigma_hat * s_in)
#         denoised_blur2 = denoiser(x_blur2, sigma_hat * s_in)
        
#         # denoised_sharp_f = denoiser(x_sharp_f, sigma_hat_f * s_in)
#         # denoised_blur_f = denoiser(x_blur_f, sigma_hat_f * s_in)
        
#         d_sharp = to_d(x_sharp, sigma_hat, denoised_sharp)
#         d_blur = to_d(x_blur, sigma_hat, denoised_blur)

#         d_sharp2 = to_d(x_sharp2, sigma_hat, denoised_sharp2)
#         d_blur2 = to_d(x_blur2, sigma_hat, denoised_blur2)

#         # d_sharp_f = to_d(x_sharp_f, sigma_hat_f, denoised_sharp_f)
#         # d_blur_f = to_d(x_blur_f, sigma_hat_f, denoised_blur_f)
        
#         dt = sigmas[len(sigmas)-1-i - 1] - sigma_hat
#         # dt_f = sigmas[f_i-1] - sigma_hat_f
#         if sigmas[len(sigmas)-1-i - 1] == 0:
#             # Euler method
#             # x = x + d * dt
#             x_sharp = x_sharp + d_sharp * dt
#             x_blur = x_blur + d_sharp * dt

#             x_sharp2 = x_sharp2 - d_sharp2 * dt
#             x_blur2 = x_blur2 - d_sharp2 * dt
            
#             # x_sharp_f = x_sharp_f - d_sharp_f * dt_f
#             # x_blur_f = x_blur_f - d_blur_f * dt_f
#         else:
#             # Heun's method
#             x_2_sharp = x_sharp + d_sharp * dt
#             x_2_blur = x_blur + d_blur * dt

#             x_2_sharp2 = x_sharp2 - d_sharp2 * dt
#             x_2_blur2 = x_blur2 - d_blur2 * dt
            
#             # x_2_sharp_f = x_sharp_f - d_sharp_f * dt_f
#             # x_2_blur_f = x_blur_f - d_blur_f * dt_f

#             denoised_sharp = denoiser(x_2_sharp, sigmas[len(sigmas)-1-i - 1] * s_in)
#             denoised_blur = denoiser(x_2_blur, sigmas[len(sigmas)-1-i - 1] * s_in)

#             denoised_sharp2 = denoiser(x_2_sharp2, sigmas[len(sigmas)-1-i - 1] * s_in)
#             denoised_blur2 = denoiser(x_2_blur2, sigmas[len(sigmas)-1-i - 1] * s_in)
            
#             # denoised_sharp_f = denoiser(x_2_sharp_f, sigmas[f_i-1] * s_in)
#             # denoised_blur_f = denoiser(x_2_blur_f, sigmas[f_i-1] * s_in)
            
#             d_2_sharp = to_d(x_2_sharp, sigmas[len(sigmas)-1-i - 1], denoised_sharp)
#             d_2_blur = to_d(x_2_blur, sigmas[len(sigmas)-1-i - 1], denoised_blur)
            
#             d_2_sharp2 = to_d(x_2_sharp2, sigmas[len(sigmas)-1-i - 1], denoised_sharp2)
#             d_2_blur2 = to_d(x_2_blur2, sigmas[len(sigmas)-1-i - 1], denoised_blur2)
            
#             # d_2_sharp_f = to_d(x_2_sharp_f, sigmas[f_i-1], denoised_sharp_f)
#             # d_2_blur_f = to_d(x_2_blur_f, sigmas[f_i-1], denoised_blur_f)

#             d_prime_sharp = (d_sharp + d_2_sharp) / 2
#             d_prime_blur = (d_blur + d_2_blur) / 2
            
#             d_prime_sharp2 = (d_sharp2 + d_2_sharp2) / 2
#             d_prime_blur2 = (d_blur2 + d_2_blur2) / 2

#             # d_prime_sharp_f = (d_sharp_f + d_2_sharp_f) / 2
#             # d_prime_blur_f = (d_blur_f + d_2_blur_f) / 2
            
#             x_sharp = x_sharp + d_prime_sharp * dt
#             x_blur = x_blur + d_prime_blur * dt
            
#             x_sharp2 = x_sharp2 - d_prime_sharp2 * dt
#             x_blur2 = x_blur2 - d_prime_blur2 * dt
            
#             # x_sharp_f = x_sharp_f - d_prime_sharp_f * dt_f
#             # x_blur_f = x_blur_f - d_prime_blur_f * dt_f
            
#         vtils.save_image(x_sharp, f'{save_dir}/{sample_idx}_sharp_step{i}.png', normalize=True)
#         vtils.save_image(x_blur, f'{save_dir}/{sample_idx}_blur_step{i}.png', normalize=True)

#         vtils.save_image(x_sharp2, f'{save_dir}/{sample_idx}_sharp_forward_step{i}.png', normalize=True)
#         vtils.save_image(x_blur2, f'{save_dir}/{sample_idx}_blur_forward_step{i}.png', normalize=True)


#         # vtils.save_image(x_sharp_f, f'{save_dir}/{sample_idx}_sharp_forward_step{f_i}.png', normalize=True)
#         # vtils.save_image(x_blur_f, f'{save_dir}/{sample_idx}_blur_forward_step{f_i}.png', normalize=True)

#     for i in indices:
#         gamma = (
#             min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
#             if s_tmin <= sigmas[i] <= s_tmax
#             else 0.0
#         )

#         eps = generator.randn_like(x_sharp) * s_noise
#         sigma_hat = sigmas[i] * (gamma + 1)
#         if gamma > 0:
#             x_sharp = x_sharp + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
#             x_blur = x_blur + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
            
#         denoised_sharp = denoiser(x_sharp, sigma_hat * s_in)
#         denoised_blur = denoiser(x_blur, sigma_hat * s_in)
#         d_sharp = to_d(x_sharp, sigma_hat, denoised_sharp)
#         d_blur = to_d(x_blur, sigma_hat, denoised_blur)

#         dt = sigmas[i + 1] - sigma_hat
#         if sigmas[i + 1] == 0:
#             # Euler method
#             x_sharp = x_sharp + d_sharp * dt
#             x_blur = x_blur + d_blur * dt
#         else:
#             # Heun's method
#             x_2_sharp = x_sharp + d_sharp * dt
#             x_2_blur = x_blur + d_blur * dt
            
#             denoised_2_sharp = denoiser(x_2_sharp, sigmas[i + 1] * s_in)
#             denoised_2_blur = denoiser(x_2_blur, sigmas[i + 1] * s_in)
#             d_2_sharp = to_d(x_2_sharp, sigmas[i + 1], denoised_2_sharp)
#             d_2_blur = to_d(x_2_blur, sigmas[i + 1], denoised_2_blur)
#             d_prime_sharp = (d_sharp + d_2_sharp) / 2
#             d_prime_blur = (d_blur + d_2_blur) / 2
#             x_sharp = x_sharp + d_prime_sharp * dt
#             x_blur = x_blur + d_prime_blur * dt
            
#         vtils.save_image(x_sharp, f'{save_dir}/{sample_idx}_fianl_sharp_step{i}.png', normalize=True)
#         vtils.save_image(x_blur, f'{save_dir}/{sample_idx}_final_blur_step{i}.png', normalize=True)

                    
#     return x


def get_ancestral_step(sigma_from, sigma_to):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = (
        sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2
    ) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


@th.no_grad()
def sample_euler_ancestral(model, x, sigmas, generator, progress=False, callback=None):
    """Ancestral sampling with Euler method steps."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        denoised = model(x, sigmas[i] * s_in)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        x = x + generator.randn_like(x) * sigma_up
    return x


@th.no_grad()
def sample_midpoint_ancestral(model, x, ts, generator, progress=False, callback=None):
    """Ancestral sampling with midpoint method steps."""
    s_in = x.new_ones([x.shape[0]])
    step_size = 1 / len(ts)
    if progress:
        from tqdm.auto import tqdm

        ts = tqdm(ts)

    for tn in ts:
        dn = model(x, tn * s_in)
        dn_2 = model(x + (step_size / 2) * dn, (tn + step_size / 2) * s_in)
        x = x + step_size * dn_2
        if callback is not None:
            callback({"x": x, "tn": tn, "dn": dn, "dn_2": dn_2})
    return x


@th.no_grad()
def sample_heun(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    sample_idx=0,
    save_dir='./'
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)
        
    logger.log(f"sample_idx: {sample_idx}")
    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
            
        # Save Intermediate noisy samples
        sample = x.clone()
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        sample = sample.cpu().numpy()

        for idx in range(sample.shape[0]):
            img = Image.fromarray(sample[idx])
            img_path = f'{save_dir}/{sample_idx + idx}_step{i}.png'
            img.save(img_path)
            # logger.log(f"img_path: {img_path}")
            
                    
    return x


@th.no_grad()
def sample_euler(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma
        x = x + d * dt
    return x


@th.no_grad()
def sample_dpm(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
        sigma_mid = ((sigma_hat ** (1 / 3) + sigmas[i + 1] ** (1 / 3)) / 2) ** 3
        dt_1 = sigma_mid - sigma_hat
        dt_2 = sigmas[i + 1] - sigma_hat
        x_2 = x + d * dt_1
        denoised_2 = denoiser(x_2, sigma_mid * s_in)
        d_2 = to_d(x_2, sigma_mid, denoised_2)
        x = x + d_2 * dt_2
    return x


@th.no_grad()
def sample_onestep(
    distiller,
    x,
    sigmas,
    generator=None,
    progress=False,
    callback=None,
):
    """Single-step generation from a distilled model."""
    s_in = x.new_ones([x.shape[0]])
    return distiller(x, sigmas[0] * s_in)


@th.no_grad()
def stochastic_iterative_sampler(
    distiller,
    x,
    sigmas,
    generator,
    ts,
    progress=False,
    callback=None,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    sample_idx=0,
    save_dir='./'
):
    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    
    logger.log(f"stochastic_iterative_sampler sample_idx: {sample_idx}")
    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

        # Save Intermediate noisy samples
        sample = x.clone()
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        sample = sample.cpu().numpy()

        for idx in range(sample.shape[0]):
            img = Image.fromarray(sample[idx])
            img_path = f'{save_dir}/{sample_idx + idx}_step{i}.png'
            img.save(img_path)
            logger.log(f"intermediate_img_path: {img_path}")
            
    return x


@th.no_grad()
def sample_progdist(
    denoiser,
    x,
    sigmas,
    generator=None,
    progress=False,
    callback=None,
):
    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas[:-1]  # skip the zero sigma

    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigma,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma
        x = x + d * dt

    return x


@th.no_grad()
def iterative_colorization(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
):
    def obtain_orthogonal_matrix():
        vector = np.asarray([0.2989, 0.5870, 0.1140])
        vector = vector / np.linalg.norm(vector)
        matrix = np.eye(3)
        matrix[:, 0] = vector
        matrix = np.linalg.qr(matrix)[0]
        if np.sum(matrix[:, 0]) < 0:
            matrix = -matrix
        return matrix

    Q = th.from_numpy(obtain_orthogonal_matrix()).to(dist_util.dev()).to(th.float32)
    mask = th.zeros(*x.shape[1:], device=dist_util.dev())
    mask[0, ...] = 1.0

    def replacement(x0, x1):
        x0 = th.einsum("bchw,cd->bdhw", x0, Q)
        x1 = th.einsum("bchw,cd->bdhw", x1, Q)

        x_mix = x0 * mask + x1 * (1.0 - mask)
        x_mix = th.einsum("bdhw,cd->bchw", x_mix, Q)
        return x_mix

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    images = replacement(images, th.zeros_like(images))

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        x0 = th.clamp(x0, -1.0, 1.0)
        x0 = replacement(images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x, images


@th.no_grad()
def iterative_inpainting(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
):
    from PIL import Image, ImageDraw, ImageFont

    image_size = x.shape[-1]

    # create a blank image with a white background
    img = Image.new("RGB", (image_size, image_size), color="white")

    # get a drawing context for the image
    draw = ImageDraw.Draw(img)

    # load a font
    font = ImageFont.truetype("arial.ttf", 250)

    # draw the letter "C" in black
    draw.text((50, 0), "S", font=font, fill=(0, 0, 0))

    # convert the image to a numpy array
    img_np = np.array(img)
    img_np = img_np.transpose(2, 0, 1)
    img_th = th.from_numpy(img_np).to(dist_util.dev())

    mask = th.zeros(*x.shape, device=dist_util.dev())
    mask = mask.reshape(-1, 7, 3, image_size, image_size)

    mask[::2, :, img_th > 0.5] = 1.0
    mask[1::2, :, img_th < 0.5] = 1.0
    mask = mask.reshape(-1, 3, image_size, image_size)

    def replacement(x0, x1):
        x_mix = x0 * mask + x1 * (1 - mask)
        return x_mix

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    images = replacement(images, -th.ones_like(images))

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        x0 = th.clamp(x0, -1.0, 1.0)
        x0 = replacement(images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x, images


@th.no_grad()
def iterative_superres(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
):
    patch_size = 8

    def obtain_orthogonal_matrix():
        vector = np.asarray([1] * patch_size**2)
        vector = vector / np.linalg.norm(vector)
        matrix = np.eye(patch_size**2)
        matrix[:, 0] = vector
        matrix = np.linalg.qr(matrix)[0]
        if np.sum(matrix[:, 0]) < 0:
            matrix = -matrix
        return matrix

    Q = th.from_numpy(obtain_orthogonal_matrix()).to(dist_util.dev()).to(th.float32)

    image_size = x.shape[-1]

    def replacement(x0, x1):
        x0_flatten = (
            x0.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x1_flatten = (
            x1.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x0 = th.einsum("bcnd,de->bcne", x0_flatten, Q)
        x1 = th.einsum("bcnd,de->bcne", x1_flatten, Q)
        x_mix = x0.new_zeros(x0.shape)
        x_mix[..., 0] = x0[..., 0]
        x_mix[..., 1:] = x1[..., 1:]
        x_mix = th.einsum("bcne,de->bcnd", x_mix, Q)
        x_mix = (
            x_mix.reshape(
                -1,
                3,
                image_size // patch_size,
                image_size // patch_size,
                patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size, image_size)
        )
        return x_mix

    def average_image_patches(x):
        x_flatten = (
            x.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x_flatten[..., :] = x_flatten.mean(dim=-1, keepdim=True)
        return (
            x_flatten.reshape(
                -1,
                3,
                image_size // patch_size,
                image_size // patch_size,
                patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size, image_size)
        )

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    images = average_image_patches(images)

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        x0 = th.clamp(x0, -1.0, 1.0)
        x0 = replacement(images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x, images
