"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
# This code is used in toy230629 and toy230704

import enum
import math
import numpy as np
import torch as th
import os
from .losses import normal_kl, discretized_gaussian_log_likelihood
from .nn import mean_flat
from piq import LPIPS
import wandb
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.utils as vtils
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

lpips = LPIPS(replace_pooling=True, reduction="none")
def get_loss(input, target):
    
    lpips_loss = lpips((input + 1) / 2.0, (target + 1) / 2.0,).mean()
    l2_loss = ((input - target) ** 2).mean()

    results = dict(lpips=lpips_loss, l2=l2_loss)
    return results


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
            self,
            *,
            betas,
            model_mean_type,
            model_var_type,
            loss_type,
            rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
  
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
    
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
                _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
                - _extract_into_tensor(
            self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
        )
                * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
                       _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                       - pred_xstart
               ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
                p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            # Remove with th.no_grad():
            out = self.p_sample(
                model,
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
            )
            yield out
            img = out["sample"]

    def ddim_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
                eta
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    def ddim_sample_loop(
            self,
            model,
            shape,
            operator,            
            noise=None,
            original_image=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
            use_wandb=False,
            directory=None,
            debug_mode=False,
            toyver=None,
            norm=None,
            measurement_cond_fn=None,
            y0_measurement=None,
            gram_model=None,
            exp_name=None,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
                model,
                operator,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                eta=eta,
                original_image=original_image,
                use_wandb=use_wandb,
                directory=directory,
                debug_mode=debug_mode,
                norm=norm,
                measurement_cond_fn=measurement_cond_fn,
                y0_measurement=y0_measurement,
                gram_model=gram_model,
                exp_name=exp_name,
        ):
            final = sample

        final_blur = final["sample"]

        return final_blur

    def ddim_sample_loop_progressive(
            self,
            model,
            operator,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
            original_image=None,
            use_wandb=False,
            directory=None,
            debug_mode=False,
            norm=None,
            measurement_cond_fn=None,
            y0_measurement=None,
            gram_model=None,
            exp_name=None,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            y_prev = noise
            ori_cleanGT = original_image
        else:
            # img = th.randn(*shape, device=device)
            assert NotImplementedError
        
        b, c, h, w = shape[0], shape[1], shape[2], shape[3]

        ##### U-Net based Gram: Fail #####
        time_zero = th.tensor([0] * shape[0], device=device)
        y0_grammatrix = model(y0_measurement, self._scale_timesteps(time_zero))
        G_y0 = model.gram_matrices
        ##################################

        norm_loss = norm['loss']
        reg_scale = norm['reg_scale']
        gram_type = norm['gram_type']
        early_stop_step = norm['early_stop']
        reg_content = norm['reg_content']
        reg_style = norm['reg_style']

        if 'early_stop' in exp_name:
            indices = list(range(early_stop_step))[::-1]
        else:
            indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            y_prev = y_prev.requires_grad_()
     
            if (exp_name is not None) and ("time" in exp_name): 
                time_scale = th.from_numpy(self.sqrt_alphas_cumprod).to(device)[int(self._scale_timesteps(i))].float()
            elif (exp_name is not None) and ("reversed" in exp_name):
                time_idx = self.num_timesteps - i - 1
                time_scale = th.from_numpy(self.sqrt_alphas_cumprod).to(device)[int(self._scale_timesteps(time_idx))].float()
                time_scale *= reg_scale
            elif (exp_name is not None) and ("exp" in exp_name):
                if 'early_stop' in exp_name:
                    m1 = early_stop_step
                else:
                    m1 = self.num_timesteps // 4
                m2 = self.num_timesteps - m1
                
                if i < m1:
                    if 'oriexp' in exp_name:
                        time_scale = np.exp(-((i-m1)**2)/reg_scale)
                    else:
                        time_scale = np.exp((i-m1)*reg_scale)
                elif i < m2:
                    time_scale = 1.0
                else:
                    if 'oriexp' in exp_name:
                        time_scale = np.exp(-((i-m2)**2)/reg_scale)
                    else:
                        time_scale = np.exp((m2-i)*reg_scale)
            else:
                time_scale = 1.0            
            print(f"time_scale {time_scale}")
                
            reg_scale_cond = time_scale * norm_loss

            yi = self.ddim_sample(
                model,
                y_prev,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                eta=eta,
            )
            y_i_img_t = yi["sample"]
            y_i_img_0hat = yi["pred_xstart"]
    
            # ##### U-Net based Gram: Fail #####
            # if 'GramB' in exp_name and (i > 0):
            #     if gram_type == 'y0hat':
            #         y0hat_grammatrix = model(y_i_img_0hat, self._scale_timesteps(time_zero))
            #     elif gram_type == 'Ay0hat':
            #         Ay0hat = operator.A(y_i_img_0hat) # H * y_0_hat
            #         Ay0hat = Ay0hat.reshape((b, c, h, w))
            #         ypred_grammatrix = model(Ay0hat, self._scale_timesteps(time_zero))
                
            #     G_pred = model.gram_matrices

            #     normG = 0.0
            #     for k in range(len(G_pred)):
            #         difference = G_y0[k] - G_pred[k]
            #         normG += th.linalg.norm(difference)
            #     norm_grad_G = th.autograd.grad(outputs=normG, inputs=y_prev)[0]
            # ##################################

            if 'vggGramB' in exp_name and (i > 0):
                if 'y0hatGram' in exp_name:
                    Ay0hat = operator.A(y_i_img_0hat).reshape((b, c, h, w))
         
                    if 'content':
                        content_loss, _ = gram_model(y_i_img_0hat) 
                        _, style_loss = gram_model(Ay0hat) 
                    elif 'style':
                        _, style_loss = gram_model(y_i_img_0hat) 
                        content_loss, _ = gram_model(Ay0hat) 
                        
                    gram_loss = reg_content * content_loss + reg_style * style_loss
                    
                else: # 0811~0812 Exp
                    if 'Apinv_Ax' in exp_name: # range space
                        Ay0hat = operator.A(y_i_img_0hat)
                        Apinv_Ax = operator.A_pinv(Ay0hat).reshape((b, c, h, w))
                        op_result = Apinv_Ax

                    elif 'Apinv_Ashapedx' in exp_name: # range space
                        Ay0hat = operator.A(y_i_img_0hat.reshape(y_i_img_0hat.size(0), -1))
                        Apinv_Ax = operator.A_pinv(Ay0hat).reshape((b, c, h, w))
                        op_result = Apinv_Ax

                    elif 'At_Ax' in exp_name:
                        Ay0hat = operator.A(y_i_img_0hat)
                        At_Ax = operator.At(Ay0hat).reshape((b, c, h, w))
                        op_result = At_Ax

                    elif 'At_Ashapedx' in exp_name:
                        Ay0hat = operator.A(y_i_img_0hat.reshape(y_i_img_0hat.size(0), -1))
                        At_Ax = operator.At(Ay0hat).reshape((b, c, h, w))
                        op_result = At_Ax

                    elif 'default_reshaped_x' in exp_name:
                        Ay0hat = operator.A(y_i_img_0hat.reshape(y_i_img_0hat.size(0), -1)).reshape((b, c, h, w))
                        op_result = Ay0hat

                    else: # default Ay0hat setting
                        Ay0hat = operator.A(y_i_img_0hat).reshape((b, c, h, w))
                        op_result = Ay0hat

                    content_loss, style_loss = gram_model(op_result) 
                    gram_loss = reg_content * content_loss + reg_style * style_loss

                if 'norm' in exp_name:
                    norm = th.linalg.norm(gram_loss)
                    norm_grad_G = norm
                elif 'grad' in exp_name:
                    norm = th.linalg.norm(gram_loss)

                    if 'grad_y_prev' in exp_name:
                        norm_grad_G = th.autograd.grad(outputs=norm, inputs=y_prev)[0]
                    elif 'grad_y0hat' in exp_name:
                        norm_grad_G = th.autograd.grad(outputs=norm, inputs=y_i_img_0hat)[0]
                else:
                    norm_grad_G = gram_loss

                print(f"gram_score {norm_grad_G}")

            ########### [Gram Matrix] ##############
            if ('BefvggGramB' in exp_name) and (i > 0):
                y_i_img_t = y_i_img_t - norm_grad_G * reg_scale_cond

            ########### [Conditioning] ###########
            if 'condB' in exp_name:
                y_i_new_img_t, distance = measurement_cond_fn(x_t=y_i_img_t,
                                                              measurement=y0_measurement, # y0
                                                              noisy_measurement=None, # measurement y0에 forward
                                                              x_prev=y_prev,
                                                              x_0_hat=y_i_img_0hat,
                                                              reg_scale=reg_scale_cond)
            elif 'no_gradB' in exp_name:
                Ay0hat = operator.A(y_i_img_0hat) # H * y_0_hat    
                Ay0hat = Ay0hat.reshape((b, c, h, w))

                y_minus_Ay0hat = y0_measurement - Ay0hat # y - H*y_0_hat

                H_t_mul_diff = operator.At(y_minus_Ay0hat) # H^T * (y - H*y_0hat)
                H_t_mul_diff = H_t_mul_diff.reshape((b, c, h, w))
                
                d_scale = self.sqrt_alphas_cumprod[int(self._scale_timesteps(i))]
                
                if 'div' in exp_name:
                    y_i_new_img_t = y_i_img_t - 2 * H_t_mul_diff * reg_scale_cond * (1.0 / d_scale)
                else:
                    y_i_new_img_t = y_i_img_t - 2 * H_t_mul_diff * reg_scale_cond * d_scale
            
            else:
                y_i_new_img_t = y_i_img_t
            
            ########### [Gram Matrix] ##############
            if ('AftvggGramB' in exp_name) and (i > 0):
                y_i_new_img_t = y_i_new_img_t - norm_grad_G * reg_scale_cond

            y_prev = y_i_new_img_t.detach_()
            yi["sample"] = y_prev
            yield yi
            
            loss_blur = get_loss(ori_cleanGT, y_i_img_0hat)

            psnr, ssim = 0.0, 0.0
            for idx in range(ori_cleanGT.shape[0]):
                restored = th.clamp(y_i_img_0hat[idx], -1., 1.).cpu().detach().numpy()
                target = th.clamp(ori_cleanGT[idx], -1., 1.).cpu().detach().numpy()
                ps = psnr_loss(restored, target)
                ss = ssim_loss(restored, target, data_range=2.0, multichannel=True, channel_axis=0)
                psnr += ps
                ssim += ss
                print(f"[PSNR]: %.4f, [SSIM]: %.4f"% (ps, ss)+'\n')
            psnr /= ori_cleanGT.shape[0]
            ssim /=ori_cleanGT.shape[0]

            if use_wandb:
                wandb_log = {'dec_blur_LPIPS': loss_blur['lpips'], 'dec_blur_L2': loss_blur['l2'], \
                             'time_scale': time_scale, 'dec_psnr_x0hat': psnr, 'dec_ssim_x0hat': ssim, \
                            'norm_grad_G': norm_grad_G.mean()}
                wandb.log(wandb_log)
            if i % 100 == 0:
                if 'no_gradB' in exp_name:
                    vtils.save_image(Ay0hat, f'{directory}nogradB_Ay0hat{i}.png', range=(-1,1), normalize=True)
                    vtils.save_image(H_t_mul_diff, f'{directory}nogradB_H_t_mul_diff{i}.png', range=(-1,1), normalize=True)
                vtils.save_image(y_i_img_0hat, f'{directory}_x_0_hat{i}.png', range=(-1,1), normalize=True)
                vtils.save_image(y_i_new_img_t, f'{directory}_x_t{i}.png', range=(-1,1), normalize=True)
            
            if debug_mode:
                break

    def ddim_reverse_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        XS:
            Modified this function to include classifier guidance (i.e. condition_score).
            Note that the (source) label information is included in model_kwargs.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        # XS: the following is the same as _predict_eps_from_xstart
        eps = (
                      _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                      - out["pred_xstart"]
              ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_next)
                + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}


    def ddim_reverse_sample_loop(
            self,
            model,
            image,
            operator,
            original_image,
            toyver,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
            use_wandb=False,
            directory=None,
            debug_mode=False,
            norm=None,
            measurement_cond_fn=None,
            gram_model=None,
            exp_name=None,
            fea_storer=None,
            feature_layers=None,
    ):
        """
        XS: Encode image into latent using DDIM ODE.
        """
        final = None

        if toyver == 1:
            for sample in self.ddim_reverse_sample_loop_progressive_ver1(
                    model,
                    image,
                    operator=operator,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    device=device,
                    progress=progress,
                    eta=eta,
                    original_image=original_image,
                    use_wandb=use_wandb,
                    directory=directory,
                    debug_mode=debug_mode,
                    norm=norm,
                    measurement_cond_fn=measurement_cond_fn,
                    gram_model=gram_model,
                    exp_name=exp_name,
            ):
                final = sample
            final_blur = final["sample"]


        return final_blur

    def ddim_reverse_sample_loop_progressive_ver1( # ver 1
            self,
            model,
            image,
            operator,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
            original_image=None,
            use_wandb=False,
            directory=None,
            debug_mode=False,
            norm=None,
            measurement_cond_fn=None,
            gram_model=None,
            exp_name=None,
    ):
        """
        XS: Use DDIM to perform encoding / inference, until isotropic Gaussian.
        """
        if device is None:
            device = next(model.parameters()).device
        
        y_prev = image # y^(0)
        y0_measurement = image.clone().detach() # y_0

        ori_sharp = original_image # for PSNR, SSIM
        shape = image.shape
        b, c, h, w = shape[0], shape[1], shape[2], shape[3]

        ##### U-Net based Gram: Fail #####
        time_zero = th.tensor([0] * shape[0], device=device)
        y0_grammatrix = model(y0_measurement, self._scale_timesteps(time_zero))
        G_y0 = model.gram_matrices
        ##################################

        norm_loss = norm['loss']
        reg_scale = norm['reg_scale']
        gram_type = norm['gram_type']
        early_stop_step = norm['early_stop']
        reg_content = norm['reg_content']
        reg_style = norm['reg_style']

        if 'early_stop' in exp_name:
            indices = list(range(early_stop_step))
        else:
            indices = list(range(self.num_timesteps))

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            y_prev = y_prev.requires_grad_()

            if (exp_name is not None) and ("time" in exp_name): 
                time_scale = th.from_numpy(self.sqrt_alphas_cumprod).to(device)[int(self._scale_timesteps(i))].float()
            elif (exp_name is not None) and ("reversed" in exp_name):
                time_idx = self.num_timesteps - i - 1
                time_scale = th.from_numpy(self.sqrt_alphas_cumprod).to(device)[int(self._scale_timesteps(time_idx))].float()
                time_scale *= reg_scale
            elif (exp_name is not None) and ("exp" in exp_name):
                if 'early_stop' in exp_name:
                    m1 = early_stop_step
                else:
                    m1 = self.num_timesteps // 4
                m2 = self.num_timesteps - m1

                if i < m1:
                    if 'oriexp' in exp_name:
                        time_scale = np.exp(-((i-m1)**2)/reg_scale)
                    else:
                        time_scale = np.exp((i-m1)*reg_scale)
                elif i < m2:
                    time_scale = 1.0
                else:
                    if 'oriexp' in exp_name:
                        time_scale = np.exp(-((i-m2)**2)/reg_scale)
                    else:
                        time_scale = np.exp((m2-i)*reg_scale)
            else:
                time_scale = 1.0
            print(f"time_scale {time_scale}")
            
            reg_scale_cond = time_scale * norm_loss

            yi = self.ddim_reverse_sample(
                model,
                y_prev,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                eta=eta,
            )
            y_i_img_t = yi["sample"] 
            y_i_img_0hat = yi["pred_xstart"] 

            # ##### U-Net based Gram: Fail #####
            # if 'GramF' in exp_name and (i > 0):
            #     if gram_type == 'y0hat':
            #         y0hat_grammatrix = model(y_i_img_0hat, self._scale_timesteps(time_zero))
            #     elif gram_type == 'Ay0hat':
            #         Ay0hat = operator.A(y_i_img_0hat) # H * y_0_hat
            #         Ay0hat = Ay0hat.reshape((b, c, h, w))
            #         ypred_grammatrix = model(Ay0hat, self._scale_timesteps(time_zero))

            #     G_pred = model.gram_matrices

            #     normG = 0.0
            #     for k in range(len(G_pred)):
            #         difference = G_y0[k] - G_pred[k]
            #         normG += th.linalg.norm(difference)
            #     norm_grad_G = th.autograd.grad(outputs=normG, inputs=y_prev)[0]
            # ##################################

            if 'vggGramF' in exp_name and (i > 0):

                if 'Apinv_Ax' in exp_name: # range space
                    raise NotImplementedError
                    Ay0hat = operator.A(y_i_img_0hat)


                    Apinv_Ax = operator.A_pinv(Ay0hat)
                    # H_funcs.H_pinv(H_funcs.H(torch.ones_like(pinv_y_0)))


                # elif 'At_Ax' in exp_name:

   

                else: # default Ay0hat setting
                    Ay0hat = operator.A(y_i_img_0hat) # Ax
                    Ay0hat = Ay0hat.reshape((b, c, h, w))
                    op_result = Ay0hat


                content_loss, style_loss = gram_model(op_result) 
                gram_loss = reg_content * content_loss + reg_style * style_loss

                if 'norm' in exp_name:
                    norm = th.linalg.norm(gram_loss)
                    norm_grad_G = norm
                elif 'grad' in exp_name:
                    norm = th.linalg.norm(gram_loss)

                    if 'grad_y_prev' in exp_name:
                        norm_grad_G = th.autograd.grad(outputs=norm, inputs=y_prev)[0]
                    elif 'grad_y0hat' in exp_name:
                        norm_grad_G = th.autograd.grad(outputs=norm, inputs=y_i_img_0hat)[0]
                
                elif 'raw' in exp_name:
                    norm_grad_G = gram_loss

                print(f"gram_score {norm_grad_G}")

            ########### [Gram Matrix] ##############
            if ('BefvggGramF' in exp_name) and (i > 0):
                y_i_img_t = y_i_img_t - norm_grad_G * reg_scale_cond

            ########### [Conditioning] ###########
            if 'condF' in exp_name:
                y_i_new_img_t, distance = measurement_cond_fn(x_t=y_i_img_t,
                                                              measurement=y0_measurement, # y0
                                                              noisy_measurement=None, # measurement y0에 forward
                                                              x_prev=y_prev,
                                                              x_0_hat=y_i_img_0hat,
                                                              reg_scale=reg_scale_cond)
            elif 'no_gradF' in exp_name:
                Ay0hat = operator.A(y_i_img_0hat) # H * y_0_hat
                Ay0hat = Ay0hat.reshape((b, c, h, w))

                y_minus_Ay0hat = y0_measurement - Ay0hat # y - H*y_0_hat (1, 3, 256, 256)
                
                H_t_mul_diff = operator.At(y_minus_Ay0hat) # H^T * (y - H*y_0hat)
                H_t_mul_diff = H_t_mul_diff.reshape((b, c, h, w))

                d_scale = self.sqrt_alphas_cumprod[int(self._scale_timesteps(i))]
            
                if 'div' in exp_name:
                    y_i_new_img_t = y_i_img_t - 2 * H_t_mul_diff * reg_scale_cond * (1 / d_scale)
                else:
                    y_i_new_img_t = y_i_img_t - 2 * H_t_mul_diff * reg_scale_cond * d_scale
            
            else:
                y_i_new_img_t = y_i_img_t

            ########### [Gram Matrix] ##############
            if ('AftvggGramF' in exp_name) and (i > 0):
                y_i_new_img_t = y_i_new_img_t - norm_grad_G * reg_scale_cond

            y_prev = y_i_new_img_t.detach_()
            yi["sample"] = y_prev
            yield yi
    
            loss_before = get_loss(y_i_img_0hat, y0_measurement)
            loss_after = get_loss(y_i_new_img_t, y0_measurement)

            psnr, ssim = 0.0, 0.0
            for idx in range(ori_sharp.shape[0]):
                restored = th.clamp(y_i_img_0hat[idx], -1., 1.).cpu().detach().numpy()
                target = th.clamp(ori_sharp[idx], -1., 1.).cpu().detach().numpy()
                ps = psnr_loss(restored, target)
                ss = ssim_loss(restored, target, data_range=2.0, multichannel=True, channel_axis=0)
                psnr += ps
                ssim += ss
                print(f"[PSNR]: %.4f, [SSIM]: %.4f"% (ps, ss)+'\n')
            psnr /= ori_sharp.shape[0]
            ssim /=ori_sharp.shape[0]

            if use_wandb:
                wandb_log = {'enc_LPIPS_bef': loss_before['lpips'], 'enc_L2_bef': loss_before['l2'], \
                             'enc_LPIPS_aft': loss_after['lpips'], 'enc_L2_aft': loss_after['l2'], \
                             'time_scale': time_scale, 'enc_psnr_x0hat': psnr, 'enc_ssim_x0hat': ssim, \
                             'norm_grad_G': norm_grad_G.mean()}
                wandb.log(wandb_log)
 
            if i % 100 == 0:
                if 'no_gradF' in exp_name:
                    vtils.save_image(Ay0hat, f'{directory}nogradF_Ay0hat{i}.png', range=(-1,1), normalize=True)
                    vtils.save_image(H_t_mul_diff, f'{directory}nogradF_H_t_mul_diff{i}.png', range=(-1,1), normalize=True)
                vtils.save_image(y_i_img_0hat, f'{directory}_updated_x_0_hat{i}.png', range=(-1,1), normalize=True)
                vtils.save_image(y_i_new_img_t, f'{directory}_x_t{i}.png', range=(-1,1), normalize=True)

            if debug_mode:
                break


    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            # Remove with th.no_grad():
            out = self._vb_terms_bpd(
                model,
                x_start=x_start,
                x_t=x_t,
                t=t_batch,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
            )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }
    

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)