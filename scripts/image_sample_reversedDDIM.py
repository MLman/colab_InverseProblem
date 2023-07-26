"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import random

import numpy as np
import torch as th
import torch.distributed as dist

from cm import dist_util, logger
from cm.image_datasets_pairs import load_data_pairs
import torchvision.utils as vtils

from guided_diffusion.script_util import(
    NUM_CLASSES,
    model_and_diffusion_defaults, # AFHQ
    ffhq_model_and_diffusion_defaults, # FFHQ
    imagenet_model_and_diffusion_defaults, # ImageNet
    gopro_model_and_diffusion_defaults, # GoPro
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def main():
    args = create_argparser().parse_args()
    
    # set save directory
    args.log_suffix = f'{args.toy_exp}_{args.log_suffix}'
    args.log_dir = os.path.join(args.log_dir, args.log_suffix)
    args.log_suffix = '_' + args.log_suffix
    save_dir = args.log_dir
    mkdir(save_dir)
    
    dist_util.setup_dist(args.gpu)
    logger.configure(dir=save_dir, format_strs=['stdout','log','csv','tensorboard'], log_suffix=args.log_suffix)

    logger.log("creating model and diffusion...")
    if 'afhq' in args.model_path:
        image_dict = args_to_dict(args, model_and_diffusion_defaults().keys())
        image_dict.update(model_and_diffusion_defaults())
        args.wandb_table = 'Model:AFHQ'
    elif 'ffhq' in args.model_path:
        image_dict = args_to_dict(args, ffhq_model_and_diffusion_defaults().keys())
        image_dict.update(ffhq_model_and_diffusion_defaults())
        args.wandb_table = 'Model:FFHQ'
    elif 'imagenet' in args.model_path:
        image_dict = args_to_dict(args, imagenet_model_and_diffusion_defaults().keys())
        image_dict.update(imagenet_model_and_diffusion_defaults())
        args.wandb_table = 'Model:ImageNet'
    elif 'GoPro' in args.model_path:
        image_dict = args_to_dict(args, gopro_model_and_diffusion_defaults().keys())
        image_dict.update(gopro_model_and_diffusion_defaults())
        args.wandb_table = 'Model:GoPro'
    else:
        NotImplementedError
        
    if args.use_wandb:
        import wandb
        table_name = f"{args.wandb_table}_steps{args.steps}"
        wandb.init(project="toy", name=table_name)
        wandb.config.update(args)
    
    model, diffusion = create_model_and_diffusion(
        **image_dict,
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    # Set dataloader for toy experiments
    dataloader = load_data_pairs(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        is_toy=True,
    )

    set_random_seed(args.seed)
    for i, (batch, extra) in enumerate(dataloader):

        x_sharp = batch[:][0].to(dist_util.dev())
        x_blur = batch[:][1].to(dist_util.dev())

        forward_dir = f'{save_dir}/{i}_forward_'
        backward_dir = f'{save_dir}/{i}_backward_'

        vtils.save_image(x_sharp, f'{save_dir}/Ori_sharp{i}.png', range=(-1,1), normalize=True)
        vtils.save_image(x_blur, f'{save_dir}/Ori_blur{i}.png', range=(-1,1), normalize=True)

        ori_sharp = x_sharp.clone()
        ori_blur = x_blur.clone()

        # First, use DDIM to encode to latents.
        logger.log("encoding the source images.")
        noise_sharp, noise_blur = diffusion.ddim_reverse_sample_loop(
            model,
            image=[x_sharp, x_blur],
            original_image=[ori_sharp, ori_blur],
            clip_denoised=True,
            device=dist_util.dev(),
            progress=True,
            use_wandb=args.use_wandb,
            directory=forward_dir,
        )
        if i == 0:
            vtils.save_image(noise_sharp, f'{save_dir}/Ori_sharp_ddim_noise.png', range=(-1,1), normalize=True)
            vtils.save_image(noise_blur, f'{save_dir}/Ori_blur_ddim_noise.png', range=(-1,1), normalize=True)
        logger.log(f"obtained latent representation for sharp/blur images...")

        # Next, decode the latents to the target class.
        sample_sharp, sample_blur = diffusion.ddim_sample_loop(
            model,
            (args.batch_size, 3, 256, 256),
            noise=[noise_sharp, noise_blur],
            original_image=[ori_sharp, ori_blur],
            clip_denoised=True,
            device=dist_util.dev(),
            progress=True,
            use_wandb=args.use_wandb,
            directory=backward_dir,
        )
        logger.log(f"obtained reconstructed sharp/blur images...")
        vtils.save_image(sample_sharp, f'{save_dir}/Recon_sharp{i}.png', range=(-1,1), normalize=True)
        vtils.save_image(sample_blur, f'{save_dir}/Recon_blur{i}.png', range=(-1,1), normalize=True)

        

    logger.log("Completed")


def create_argparser():
    defaults = dict(
        # edm_afhq_cat
        training_mode="edm",
        generator="determ",
        schedule_sampler="lognormal", 
        clip_denoised=True,
        num_samples=1000,
        batch_size=1,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        # steps=40,
        steps=1000,

        # model_path='./models/imagenet/256x256_diffusion_uncond.pt',

        # data_dir="/hub_data1/sojin/afhq_blur/afhq_cat_motionblur",
        # model_path="./models/afhqCat/guided_diffusion_afhqcat_ema_0.9999_625000.pt",
        # seed=42,
        # seed=1234,

        data_dir="/hub_data1/sojin/ffhq_1K_motionblur",
        model_path="./models/ffhq_1k/ffhq_10m.pt",
        seed=42,
        # seed=1234,

        # data_dir="/hub_data1/sojin/Restormer_GoPro/train",
        # model_path="./models/GoPro/clean/ema_0.9999_187500.pt",

    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu', type=str, default='0')
    # parser.add_argument('--log_dir', type=str, default='/hub_data1/sojin/sampling_results/')
    # parser.add_argument('--log_dir', type=str, default='./toy230629_ImageNet/afhqcat_')
    parser.add_argument('--log_dir', type=str, default='./results_toy/toy230724_FFHQ_oriDDIM')
    # parser.add_argument('--log_dir', type=str, default='./toy230629/AFHQ_Cat')
    # parser.add_argument('--log_dir', type=str, default='./toy230629/GoPro')
    # parser.add_argument('--log_dir', type=str, default='./toy230629/FFHQ_1K')
    parser.add_argument('--toy_exp', type=str, default='ddim_reverse')
    # parser.add_argument('--toy_exp', type=str, default='None')
    parser.add_argument('-log','--log_suffix', type=str, required=True) # Experiment name, starts with tb(tesorboard) ->  tb_exp1
    parser.add_argument('--use_wandb', type=bool, default=False) # Experiment name, starts with tb(tesorboard) ->  tb_exp1

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

# 