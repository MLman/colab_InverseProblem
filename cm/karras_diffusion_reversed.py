"""
Based on: https://github.com/crowsonkb/k-diffusion
"""
import random

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vtils
from cm import logger

from piq import LPIPS
from torchvision.transforms import RandomCrop
from . import dist_util
import wandb

from .nn import mean_flat, append_dims, append_zero
from .random_util import get_generator

lpips = LPIPS(replace_pooling=True, reduction="none")
def get_loss(input, target):
    
    lpips_loss = lpips((input + 1) / 2.0, (target + 1) / 2.0,).mean()
    l2_loss = ((input - target) ** 2).mean()

    results = dict(lpips=lpips_loss, l2=l2_loss)
    return results

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
    images,
    original_image,
    enc_noise=None,
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
    use_wandb=False,
    directory=None,
):
    if generator is None:
        generator = get_generator("dummy")

    if sampler == "progdist":
        sigmas = get_sigmas_karras(steps + 1, sigma_min, sigma_max, rho, device=device)
    else:
        sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)

    # x_T = generator.randn(*shape, device=device) * sigma_max
    # if enc_noise is None:
    #     x_T_sharp = generator.randn(*shape, device=device) * sigma_max
    #     x_T_blur = generator.randn(*shape, device=device) * sigma_max
    # else:
    #     x_T_sharp, x_T_blur = enc_noise[0], enc_noise[1]
    x_T_sharp, x_T_blur = images[0], images[1]

    sample_fn = {
        "heun": sample_heun,
        "dpm": sample_dpm,
        "ancestral": sample_euler_ancestral,
        "onestep": sample_onestep,
        "progdist": sample_progdist,
        "euler": sample_euler,
        "multistep": stochastic_iterative_sampler,
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

    args_dict = dict(original_image=original_image, use_wandb=use_wandb, directory=directory)
    sampler_args.update(args_dict)

    def denoiser(x_t, sigma):
        _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    x_0_sharp, x_0_blur = sample_fn(
        denoiser,
        [x_T_sharp, x_T_blur],
        sigmas,
        generator,
        progress=progress,
        callback=callback,
        **sampler_args,
    )
    return x_0_sharp.clamp(-1, 1), x_0_blur.clamp(-1, 1)


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


def get_ancestral_step(sigma_from, sigma_to):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = (
        sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2
    ) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


@th.no_grad()
def sample_euler_ancestral(model, x_images, sigmas, generator, progress=False, callback=None, original_image=None, use_wandb=False, directory=None):
    """Ancestral sampling with Euler method steps."""
    x_sharp, x_blur = x_images[0], x_images[1]
    ori_sharp, ori_blur = original_image[0], original_image[1]

    s_in = x_sharp.new_ones([x_sharp.shape[0]])
    indices = range(len(sigmas) - 1)
    indices2 = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)
        indices2 = tqdm(indices2)

    for i in indices:
        j = len(sigmas) - 1 - i

        sigma_down, sigma_up = get_ancestral_step(sigmas[j-1], sigmas[j])

        denoised_sharp = model(x_sharp, sigmas[j] * s_in)
        denoised_blur = model(x_blur, sigmas[j] * s_in)

        if callback is not None:
            callback(
                {
                    "x_sharp": x_sharp,
                    "x_blur": x_blur,
                    "j": j,
                    "sigma": sigmas[j],
                    "sigma_hat": sigmas[j],
                    "denoised_sharp": denoised_sharp,
                    "denoised_blur": denoised_blur,
                }
            )
        d_sharp = to_d(x_sharp, sigmas[j-1], denoised_sharp)
        d_blur = to_d(x_blur, sigmas[j-1], denoised_blur)

        dt = sigma_up - sigmas[j]
        x_sharp = x_sharp + d_sharp * dt
        x_blur = x_blur + d_blur * dt
        x_sharp = x_sharp + generator.randn_like(x_sharp) * sigma_up
        x_blur = x_blur + generator.randn_like(x_blur) * sigma_up
        
        loss_sharp = get_loss(ori_sharp, denoised_sharp)
        loss_blur = get_loss(ori_blur, denoised_blur)
        if use_wandb:
            wandb_log = {
                'enc_sharp_LPIPS': loss_sharp['lpips'], 'enc_blur_LPIPS': loss_blur['lpips'],
                'enc_sharp_L2': loss_sharp['l2'], 'enc_blur_L2': loss_blur['l2'],
                }
            wandb.log(wandb_log)
        vtils.save_image(denoised_sharp, f'{directory}reversed_denoisedSharp_step{j}.png', range=(-1,1), normalize=True)
        vtils.save_image(denoised_blur, f'{directory}reversed_denoisedBlur_step{j}.png', range=(-1,1), normalize=True)
        vtils.save_image(x_sharp, f'{directory}reversed_sharp_step{j}.png', range=(-1,1), normalize=True)
        vtils.save_image(x_blur, f'{directory}reversed_blur_step{j}.png', range=(-1,1), normalize=True)

    dt_list = []
    sigma_list = []
    for i in indices2:
        denoised_sharp = model(x_sharp, sigmas[i] * s_in)
        denoised_blur = model(x_blur, sigmas[i] * s_in)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])


        if callback is not None:
            callback(
                {
                    "x_sharp": x_sharp,
                    "x_blur": x_blur,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised_sharp": denoised_sharp,
                    "denoised_blur": denoised_blur,
                }
            )
        d_sharp = to_d(x_sharp, sigmas[i], denoised_sharp)
        d_blur = to_d(x_blur, sigmas[i], denoised_blur)


        # Euler method
        dt = sigma_down - sigmas[i]
        x_sharp = x_sharp + d_sharp * dt
        x_blur = x_blur + d_blur * dt
        x_sharp = x_sharp + generator.randn_like(x_sharp) * sigma_up
        x_blur = x_blur + generator.randn_like(x_blur) * sigma_up

 
        dt_list.append(dt.item())
        sigma_list.append(dict(sigma_down=sigma_down.item(), sigma_up=sigma_up.item()))
        logger.log(f'mean of dt {dt.mean()}')

        loss_sharp = get_loss(ori_sharp, denoised_sharp)
        loss_blur = get_loss(ori_blur, denoised_blur)
        if use_wandb:
            wandb_log = {
                    'dec_sharp_LPIPS': loss_sharp['lpips'], 'dec_blur_LPIPS': loss_blur['lpips'],
                    'dec_sharp_L2': loss_sharp['l2'], 'dec_blur_L2': loss_blur['l2'],
                    }
            wandb.log(wandb_log)
        vtils.save_image(denoised_sharp, f'{directory}recon_denoisedSharp_step{j}.png', range=(-1,1), normalize=True)
        vtils.save_image(denoised_blur, f'{directory}recon_denoisedBlur_step{j}.png', range=(-1,1), normalize=True)
        vtils.save_image(x_sharp, f'{directory}recon_sharp_step{i}.png', range=(-1,1), normalize=True)
        vtils.save_image(x_blur, f'{directory}recon_blur_step{i}.png', range=(-1,1), normalize=True)

    return x_sharp, x_blur

# @th.no_grad()
# def sample_midpoint_ancestral(model, x, ts, generator, progress=False, callback=None, use_wandb=False, directory=None):
#     """Ancestral sampling with midpoint method steps."""
#     s_in = x.new_ones([x.shape[0]])
#     step_size = 1 / len(ts)
#     if progress:
#         from tqdm.auto import tqdm

#         ts = tqdm(ts)

#     for tn in ts:
#         dn = model(x, tn * s_in)
#         dn_2 = model(x + (step_size / 2) * dn, (tn + step_size / 2) * s_in)
#         x = x + step_size * dn_2
#         if callback is not None:
#             callback({"x": x, "tn": tn, "dn": dn, "dn_2": dn_2})
#     return x


@th.no_grad()
def sample_heun(
    denoiser,
    x_images,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    original_image=None,
    use_wandb=False, 
    directory=None,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    x_sharp, x_blur = x_images[0], x_images[1]
    ori_sharp, ori_blur = original_image[0], original_image[1]

    s_in = x_sharp.new_ones([x_sharp.shape[0]])
    indices = range(len(sigmas) - 1)
    indices2 = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)
        indices2 = tqdm(indices2)

    # Forward
    for i in indices:
        j = len(sigmas) - 1 - i
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[j-1] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x_sharp) * s_noise
        sigma_hat = sigmas[j-1] * (gamma + 1)
        if gamma > 0:
            x_sharp = x_sharp + eps * (sigma_hat**2 - sigmas[j] ** 2) ** 0.5
            x_blur = x_blur + eps * (sigma_hat**2 - sigmas[j] ** 2) ** 0.5
        denoised_sharp = denoiser(x_sharp, sigma_hat * s_in)
        denoised_blur = denoiser(x_blur, sigma_hat * s_in)
        d_sharp = to_d(x_sharp, sigma_hat, denoised_sharp)
        d_blur = to_d(x_blur, sigma_hat, denoised_blur)
        if callback is not None:
            callback(
                {
                    "x_sharp": x_sharp,
                    "x_blur": x_blur,
                    "j": j,
                    "sigma": sigmas[j],
                    "sigma_hat": sigma_hat,
                    "denoised_sharp": denoised_sharp,
                    "denoised_blur": denoised_blur,
                }
            )
        dt = sigmas[j] - sigma_hat
        if sigmas[j] == 0:
            # Euler method
            x_sharp = x_sharp + d_sharp * dt
            x_blur = x_blur + d_blur * dt
        else:
            # Heun's method
            x_2_sharp = x_sharp - d_sharp * dt
            x_2_blur = x_blur - d_blur * dt
            denoised_2_sharp = denoiser(x_2_sharp, sigmas[j] * s_in)
            denoised_2_blur= denoiser(x_2_blur, sigmas[j] * s_in)
            d_2_sharp = to_d(x_2_sharp, sigmas[j], denoised_2_sharp)
            d_2_blur = to_d(x_2_blur, sigmas[j], denoised_2_blur)
            d_prime_sharp = (d_sharp + d_2_sharp) / 2
            d_prime_blur = (d_blur + d_2_blur) / 2
            x_sharp = x_sharp - d_prime_sharp * dt
            x_blur = x_blur - d_prime_blur * dt
            
            vtils.save_image(denoised_2_sharp, f'{directory}reversed_denoisedSharp_step{j}.png', range=(-1,1), normalize=True)
            vtils.save_image(denoised_2_blur, f'{directory}reversed_denoisedBlur_step{j}.png', range=(-1,1), normalize=True)
            
            loss_sharp = get_loss(ori_sharp, denoised_2_sharp)
            loss_blur = get_loss(ori_blur, denoised_2_blur)
            if use_wandb:
                wandb_log = {
                    'enc_sharp_LPIPS': loss_sharp['lpips'], 'enc_blur_LPIPS': loss_blur['lpips'],
                    'enc_sharp_L2': loss_sharp['l2'], 'enc_blur_L2': loss_blur['l2'],
                    }
                wandb.log(wandb_log)

        vtils.save_image(x_sharp, f'{directory}reversed_sharp_step{j}.png', range=(-1,1), normalize=True)
        vtils.save_image(x_blur, f'{directory}reversed_blur_step{j}.png', range=(-1,1), normalize=True)


    # Denoising
    for i in indices2:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x_sharp) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x_sharp = x_sharp + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
            x_blur = x_blur + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised_sharp = denoiser(x_sharp, sigma_hat * s_in)
        denoised_blur = denoiser(x_blur, sigma_hat * s_in)
        d_sharp = to_d(x_sharp, sigma_hat, denoised_sharp)
        d_blur = to_d(x_blur, sigma_hat, denoised_blur)
        if callback is not None:
            callback(
                {
                    "x_sharp": x_sharp,
                    "x_blur": x_blur,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised_sharp": denoised_sharp,
                    "denoised_blur": denoised_blur,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x_sharp = x_sharp + d_sharp * dt
            x_blur = x_blur + d_blur * dt
        else:
            # Heun's method
            x_2_sharp = x_sharp + d_sharp * dt
            x_2_blur = x_blur + d_blur * dt
            denoised_2_sharp = denoiser(x_2_sharp, sigmas[i + 1] * s_in)
            denoised_2_blur= denoiser(x_2_blur, sigmas[i + 1] * s_in)
            d_2_sharp = to_d(x_2_sharp, sigmas[i + 1], denoised_2_sharp)
            d_2_blur = to_d(x_2_blur, sigmas[i + 1], denoised_2_blur)
            d_prime_sharp = (d_sharp + d_2_sharp) / 2
            d_prime_blur = (d_blur + d_2_blur) / 2
            x_sharp = x_sharp + d_prime_sharp * dt
            x_blur = x_blur + d_prime_blur * dt
            
            loss_sharp = get_loss(ori_sharp, denoised_2_sharp)
            loss_blur = get_loss(ori_blur, denoised_2_blur)
            if use_wandb:
                wandb_log = {
                    'dec_sharp_LPIPS': loss_sharp['lpips'], 'dec_blur_LPIPS': loss_blur['lpips'],
                    'dec_sharp_L2': loss_sharp['l2'], 'dec_blur_L2': loss_blur['l2'],
                    }
                wandb.log(wandb_log)
            vtils.save_image(denoised_2_sharp, f'{directory}recon_denoisedSharp_step{i}.png', range=(-1,1), normalize=True)
            vtils.save_image(denoised_2_blur, f'{directory}recon_denoisedBlur_step{i}.png', range=(-1,1), normalize=True)
            vtils.save_image(x_sharp, f'{directory}recon_sharp_step{i}.png', range=(-1,1), normalize=True)
            vtils.save_image(x_blur, f'{directory}recon_blur_step{i}.png', range=(-1,1), normalize=True)

    return x_sharp, x_blur


@th.no_grad()
def sample_euler(
    denoiser,
    x_images,
    sigmas,
    generator,
    progress=False,
    callback=None,
    original_image=None,
    use_wandb=False,
    directory=None,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    x_sharp, x_blur = x_images[0], x_images[1]
    ori_sharp, ori_blur = original_image[0], original_image[1]

    s_in = x_sharp.new_ones([x_sharp.shape[0]])
    indices = range(len(sigmas) - 1)
    indices2 = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)
        indices2 = tqdm(indices2)

    for i in indices:
        j = len(sigmas) - 1 - i
        sigma = sigmas[j]
        denoised_sharp = denoiser(x_sharp, sigma * s_in)
        denoised_blur = denoiser(x_blur, sigma * s_in)
        d_sharp = to_d(x_sharp, sigmas[j-1], denoised_sharp)
        d_blur = to_d(x_blur, sigmas[j-1], denoised_blur)
        # d_sharp = to_d(x_sharp, sigmas[j], denoised_sharp)
        # d_blur = to_d(x_blur, sigmas[j], denoised_blur)

        if callback is not None:
            callback(
                {
                    "x_sharp": x_sharp,
                    "x_blur": x_blur,
                    "j": j,
                    "sigma": sigmas[j],
                    "denoised_sharp": denoised_sharp,
                    "denoised_blur": denoised_blur,
                }
            )
        dt = sigmas[j-1] - sigma
        x_sharp = x_sharp + d_sharp * dt
        x_blur = x_blur + d_blur * dt

        loss_sharp = get_loss(ori_sharp, denoised_sharp)
        loss_blur = get_loss(ori_blur, denoised_blur)
        if use_wandb:
            wandb_log = {
                'enc_sharp_LPIPS': loss_sharp['lpips'], 'enc_blur_LPIPS': loss_blur['lpips'],
                'enc_sharp_L2': loss_sharp['l2'], 'enc_blur_L2': loss_blur['l2'],
                }
            wandb.log(wandb_log)
        vtils.save_image(denoised_sharp, f'{directory}reversed_denoisedSharp_step{j}.png', range=(-1,1), normalize=True)
        vtils.save_image(denoised_blur, f'{directory}reversed_denoisedBlur_step{j}.png', range=(-1,1), normalize=True)
        vtils.save_image(x_sharp, f'{directory}reversed_sharp_step{j}.png', range=(-1,1), normalize=True)
        vtils.save_image(x_blur, f'{directory}reversed_blur_step{j}.png', range=(-1,1), normalize=True)


    for i in indices2:
        sigma = sigmas[i]
        denoised_sharp = denoiser(x_sharp, sigma * s_in)
        denoised_blur = denoiser(x_blur, sigma * s_in)
        d_sharp = to_d(x_sharp, sigma, denoised_sharp)
        d_blur = to_d(x_blur, sigma, denoised_blur)
        if callback is not None:
            callback(
                {
                    "x_sharp": x_sharp,
                    "x_blur": x_blur,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised_sharp": denoised_sharp,
                    "denoised_blur": denoised_blur,
                }
            )
        dt = sigmas[i + 1] - sigma
        x_sharp = x_sharp + d_sharp * dt
        x_blur = x_blur + d_blur * dt

        loss_sharp = get_loss(ori_sharp, denoised_sharp)
        loss_blur = get_loss(ori_blur, denoised_blur)
        if use_wandb:
            wandb_log = {
                    'dec_sharp_LPIPS': loss_sharp['lpips'], 'dec_blur_LPIPS': loss_blur['lpips'],
                    'dec_sharp_L2': loss_sharp['l2'], 'dec_blur_L2': loss_blur['l2'],
                    }
            wandb.log(wandb_log)
        vtils.save_image(denoised_sharp, f'{directory}recon_denoisedSharp_step{i}.png', range=(-1,1), normalize=True)
        vtils.save_image(denoised_blur, f'{directory}recon_denoisedBlur_step{i}.png', range=(-1,1), normalize=True)
        vtils.save_image(x_sharp, f'{directory}recon_sharp_step{i}.png', range=(-1,1), normalize=True)
        vtils.save_image(x_blur, f'{directory}recon_blur_step{i}.png', range=(-1,1), normalize=True)

    return x_sharp, x_blur


@th.no_grad()
def sample_dpm(
    denoiser,
    x_images,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    original_image=None,
    use_wandb=False,
    directory=None,
):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    x_sharp, x_blur = x_images[0], x_images[1]
    ori_sharp, ori_blur = original_image[0], original_image[1]

    s_in = x_sharp.new_ones([x_sharp.shape[0]])
    indices = range(len(sigmas) - 1)
    indices2 = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)
        indices2 = tqdm(indices2)

    for i in indices:
        j = len(sigmas) - 1 - i
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[j-1] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x_sharp) * s_noise
        sigma_hat = sigmas[j-1] * (gamma + 1)
        if gamma > 0:
            x_sharp = x_sharp + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
            x_blur = x_blur + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised_sharp = denoiser(x_sharp, sigma_hat * s_in)
        denoised_blur = denoiser(x_blur, sigma_hat * s_in)
        d_sharp = to_d(x_sharp, sigma_hat, denoised_sharp)
        d_blur = to_d(x_blur, sigma_hat, denoised_blur)
        if callback is not None:
            callback(
                {
                    "x_sharp": x_sharp,
                    "x_blur": x_blur,
                    "j": j,
                    "sigma": sigmas[j],
                    "sigma_hat": sigma_hat,
                    "denoised_sharp": denoised_sharp,
                    "denoised_blur": denoised_blur,
                }
            )
        # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
        sigma_mid = ((sigma_hat ** (1 / 3) + sigmas[j] ** (1 / 3)) / 2) ** 3
        dt_1 = sigma_mid - sigma_hat
        dt_2 = sigmas[j] - sigma_hat
        x_2_sharp = x_sharp - d_sharp * dt_1
        x_2_blur = x_blur - d_blur * dt_1
        denoised_2_sharp = denoiser(x_2_sharp, sigma_mid * s_in)
        denoised_2_blur = denoiser(x_2_blur, sigma_mid * s_in)
        d_2_sharp = to_d(x_2_sharp, sigma_mid, denoised_2_sharp)
        d_2_blur = to_d(x_2_blur, sigma_mid, denoised_2_blur)
        x_sharp = x_sharp - d_2_sharp * dt_2
        x_blur = x_blur - d_2_blur * dt_2

        loss_sharp = get_loss(ori_sharp, denoised_2_sharp)
        loss_blur = get_loss(ori_blur, denoised_2_blur)
        if use_wandb:
            wandb_log = {
                'enc_sharp_LPIPS': loss_sharp['lpips'], 'enc_blur_LPIPS': loss_blur['lpips'],
                'enc_sharp_L2': loss_sharp['l2'], 'enc_blur_L2': loss_blur['l2'],
                }
            wandb.log(wandb_log)
        vtils.save_image(denoised_2_sharp, f'{directory}reversed_denoisedSharp_step{j}.png', range=(-1,1), normalize=True)
        vtils.save_image(denoised_2_blur, f'{directory}reversed_denoisedBlur_step{j}.png', range=(-1,1), normalize=True)
        vtils.save_image(x_sharp, f'{directory}reversed_sharp_step{j}.png', range=(-1,1), normalize=True)
        vtils.save_image(x_blur, f'{directory}reversed_blur_step{j}.png', range=(-1,1), normalize=True)


    for i in indices2:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x_sharp) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x_sharp = x_sharp + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
            x_blur = x_blur + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised_sharp = denoiser(x_sharp, sigma_hat * s_in)
        denoised_blur = denoiser(x_blur, sigma_hat * s_in)
        d_sharp = to_d(x_sharp, sigma_hat, denoised_sharp)
        d_blur = to_d(x_blur, sigma_hat, denoised_blur)
        if callback is not None:
            callback(
                {
                    "x_sharp": x_sharp,
                    "x_blur": x_blur,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised_sharp": denoised_sharp,
                    "denoised_blur": denoised_blur,
                }
            )
        # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
        sigma_mid = ((sigma_hat ** (1 / 3) + sigmas[i + 1] ** (1 / 3)) / 2) ** 3
        dt_1 = sigma_mid - sigma_hat
        dt_2 = sigmas[i + 1] - sigma_hat
        x_2_sharp = x_sharp + d_sharp * dt_1
        x_2_blur = x_blur + d_blur * dt_1
        denoised_2_sharp = denoiser(x_2_sharp, sigma_mid * s_in)
        denoised_2_blur = denoiser(x_2_blur, sigma_mid * s_in)
        d_2_sharp = to_d(x_2_sharp, sigma_mid, denoised_2_sharp)
        d_2_blur = to_d(x_2_blur, sigma_mid, denoised_2_blur)
        x_sharp = x_sharp + d_2_sharp * dt_2
        x_blur = x_blur + d_2_blur * dt_2

        loss_sharp = get_loss(ori_sharp, denoised_2_sharp)
        loss_blur = get_loss(ori_blur, denoised_2_blur)
        if use_wandb:
            wandb_log = {
                    'dec_sharp_LPIPS': loss_sharp['lpips'], 'dec_blur_LPIPS': loss_blur['lpips'],
                    'dec_sharp_L2': loss_sharp['l2'], 'dec_blur_L2': loss_blur['l2'],
                    }
            wandb.log(wandb_log)

        vtils.save_image(denoised_2_sharp, f'{directory}recon_denoisedSharp_step{i}.png', range=(-1,1), normalize=True)
        vtils.save_image(denoised_2_blur, f'{directory}recon_denoisedBlur_step{i}.png', range=(-1,1), normalize=True)
        vtils.save_image(x_sharp, f'{directory}recon_sharp_step{i}.png', range=(-1,1), normalize=True)
        vtils.save_image(x_blur, f'{directory}recon_blur_step{i}.png', range=(-1,1), normalize=True)

    return x_sharp, x_blur


@th.no_grad()
def sample_onestep(
    distiller,
    x_images,
    sigmas,
    generator=None,
    progress=False,
    callback=None,
    original_image=None,
    use_wandb=False,
    directory=None,
):
    """Single-step generation from a distilled model."""
    x_sharp, x_blur = x_images[0], x_images[1]
    ori_sharp, ori_blur = original_image[0], original_image[1]

    s_in = x_sharp.new_ones([x_sharp.shape[0]])
    
    x_sharp = distiller(x_sharp, sigmas[-2] * s_in)
    x_blur = distiller(x_blur, sigmas[-2] * s_in)
    vtils.save_image(x_sharp, f'{directory}reversed_sharp_1step.png', range=(-1,1), normalize=True)
    vtils.save_image(x_blur, f'{directory}reversed_blur_1step.png', range=(-1,1), normalize=True)

    x_sharp = distiller(x_sharp, sigmas[0] * s_in)
    x_blur = distiller(x_blur, sigmas[0] * s_in)
    vtils.save_image(x_sharp, f'{directory}recon_sharp_1step.png', range=(-1,1), normalize=True)
    vtils.save_image(x_blur, f'{directory}recon_blur_1step.png', range=(-1,1), normalize=True)

    return x_sharp, x_blur


@th.no_grad()
def stochastic_iterative_sampler(
    distiller,
    x_images,
    sigmas,
    generator,
    ts,
    progress=False,
    callback=None,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    original_image=None,
    use_wandb=False,
    directory=None,
):
    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    x_sharp, x_blur = x_images[0], x_images[1]
    ori_sharp, ori_blur = original_image[0], original_image[1]

    s_in = x_sharp.new_ones([x_sharp.shape[0]])

    for i in range(len(ts) - 1):
        j = len(ts) - 1 - i
        t = (t_max_rho + ts[j] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0_sharp = distiller(x_sharp, t * s_in)
        x0_blur = distiller(x_blur, t * s_in)
        next_t = (t_max_rho + ts[j - 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x_sharp = x0_sharp - generator.randn_like(x_sharp) * np.sqrt(next_t**2 - t_min**2)
        x_blur = x0_sharp - generator.randn_like(x_blur) * np.sqrt(next_t**2 - t_min**2)

        loss_sharp = get_loss(ori_sharp, x0_sharp)
        loss_blur = get_loss(ori_blur, x0_blur)
        if use_wandb:
            wandb_log = {
                'enc_sharp_LPIPS': loss_sharp['lpips'], 'enc_blur_LPIPS': loss_blur['lpips'],
                'enc_sharp_L2': loss_sharp['l2'], 'enc_blur_L2': loss_blur['l2'],
                }
            wandb.log(wandb_log)

        vtils.save_image(x0_sharp, f'{directory}reversed_x_0sharp_step{j}.png', range=(-1,1), normalize=True)
        vtils.save_image(x0_blur, f'{directory}reversed_x_0blur_step{j}.png', range=(-1,1), normalize=True)
        vtils.save_image(x_sharp, f'{directory}reversed_sharp_step{j}.png', range=(-1,1), normalize=True)
        vtils.save_image(x_blur, f'{directory}reversed_blur_step{j}.png', range=(-1,1), normalize=True)


    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0_sharp = distiller(x_sharp, t * s_in)
        x0_blur = distiller(x_blur, t * s_in)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x_sharp = x0_sharp + generator.randn_like(x_sharp) * np.sqrt(next_t**2 - t_min**2)
        x_blur = x0_sharp + generator.randn_like(x_blur) * np.sqrt(next_t**2 - t_min**2)

        loss_sharp = get_loss(ori_sharp, x0_sharp)
        loss_blur = get_loss(ori_blur, x0_blur)
        if use_wandb:
            wandb_log = {
                    'dec_sharp_LPIPS': loss_sharp['lpips'], 'dec_blur_LPIPS': loss_blur['lpips'],
                    'dec_sharp_L2': loss_sharp['l2'], 'dec_blur_L2': loss_blur['l2'],
                    }
            wandb.log(wandb_log)
        vtils.save_image(x0_sharp, f'{directory}recon_x_0sharp_step{i}.png', range=(-1,1), normalize=True)
        vtils.save_image(x0_blur, f'{directory}recon_x_0blur_step{i}.png', range=(-1,1), normalize=True)
        vtils.save_image(x_sharp, f'{directory}recon_sharp_step{i}.png', range=(-1,1), normalize=True)
        vtils.save_image(x_blur, f'{directory}recon_blur_step{i}.png', range=(-1,1), normalize=True)

    return x_sharp, x_blur



@th.no_grad()
def sample_progdist(
    denoiser,
    x_images,
    sigmas,
    generator=None,
    progress=False,
    callback=None,
    original_image=None,
    use_wandb=False,
    directory=None,
):
    raise NotImplementedError
    x_sharp, x_blur = x_images[0], x_images[1]
    ori_sharp, ori_blur = original_image[0], original_image[1]

    s_in = x_sharp.new_ones([x_sharp.shape[0]])
    sigmas = sigmas[:-1]  # skip the zero sigma

    indices = range(len(sigmas) - 1)
    indices2 = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)
        indices2 = tqdm(indices2)

    for i in indices:
        j = len(sigmas) - 1 - i
        sigma = sigmas[j]
        denoised_sharp = denoiser(x_sharp, sigma * s_in)
        denoised_blur = denoiser(x_blur, sigma * s_in)
        d_sharp = to_d(x_sharp, sigmas[j-1], denoised_sharp)
        d_blur = to_d(x_blur, sigma[j-1], denoised_blur)
        if callback is not None:
            callback(
                {
                    "x_sharp": x_sharp,
                    "x_blur": x_blur,
                    "j": j,
                    "sigma": sigma,
                    "denoised_sharp": denoised_sharp,
                    "denoised_blur": denoised_blur,
                }
            )
        dt = sigmas[j - 1] - sigma
        x_sharp = x_sharp - d_sharp * dt
        x_blur = x_blur - d_blur * dt
        
        vtils.save_image(denoised_sharp, f'{directory}reversed_denoisedSharp_step{j}.png', range=(-1,1), normalize=True)
        vtils.save_image(denoised_blur, f'{directory}reversed_denoisedBlur_step{j}.png', range=(-1,1), normalize=True)
        vtils.save_image(x_sharp, f'{directory}reversed_sharp_step{j}.png', range=(-1,1), normalize=True)
        vtils.save_image(x_blur, f'{directory}reversed_blur_step{j}.png', range=(-1,1), normalize=True)


    for i in indices2:
        sigma = sigmas[i]
        denoised_sharp = denoiser(x_sharp, sigma * s_in)
        denoised_blur = denoiser(x_blur, sigma * s_in)
        d_sharp = to_d(x_sharp, sigma, denoised_sharp)
        d_blur = to_d(x_blur, sigma, denoised_blur)
        if callback is not None:
            callback(
                {
                    "x_sharp": x_sharp,
                    "x_blur": x_blur,
                    "i": i,
                    "sigma": sigma,
                    "denoised_sharp": denoised_sharp,
                    "denoised_blur": denoised_blur,
                }
            )
        dt = sigmas[i + 1] - sigma
        x_sharp = x_sharp + d_sharp * dt
        x_blur = x_blur + d_blur * dt

        vtils.save_image(x_sharp, f'{directory}recon_sharp_step{i}.png', range=(-1,1), normalize=True)
        vtils.save_image(x_blur, f'{directory}recon_blur_step{i}.png', range=(-1,1), normalize=True)

    return x_sharp, x_blur
