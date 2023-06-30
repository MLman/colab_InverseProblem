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

from cm.script_util_ori import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample

from PIL import Image

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
    logger.configure(dir=args.log_dir, format_strs=['stdout','log','csv','tensorboard'], log_suffix=args.log_suffix)
    
    # if args.use_wandb:
        # import wandb
        # table_name = f"steps_{args.steps}_batch_{args.batch_size}"
        # wandb.init(project="toy", name=table_name)
    
    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    # Set dataloader for toy experiments
    if args.toy_exp is not None:
        dataloader = load_data_pairs(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
        )
    else:
        dataloader = None   


    all_images = []
    all_labels = []
    generator = get_generator(args.generator, args.num_samples, args.seed)

    logger.log(f"Consistency Models official pretrained model")
    set_random_seed(args.seed)
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        
        cur_img_cnt = len(all_images) * args.batch_size
        
        sample = karras_sample(
            diffusion,
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            steps=args.steps,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            clip_denoised=args.clip_denoised,
            sampler=args.sampler,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            s_churn=args.s_churn,
            s_tmin=args.s_tmin,
            s_tmax=args.s_tmax,
            s_noise=args.s_noise,
            generator=generator,
            ts=ts,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        # all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        all_images.extend([sample.cpu().numpy()])
        if args.class_cond:
            # gathered_labels = [
            #     th.zeros_like(classes) for _ in range(dist.get_world_size())
            # ]
            # dist.all_gather(gathered_labels, classes)
            # all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            all_labels.extend([th.zeros_like(classes).cpu().numpy()])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

        # Save Image as png format
        new_img_arr = all_images[len(all_images)-1]
        # logger.log(f"len(all_images): {len(all_images)}")
        
        for idx in range(args.batch_size):
            img = Image.fromarray(new_img_arr[idx])
            img_path = f'{save_dir}/{cur_img_cnt + idx}.png'
            img.save(img_path)
            logger.log(f"img_path: {img_path}")

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        # edm_afhq_cat
        data_dir="/hub_data1/sojin/afhq_blur/afhq_cat_motionblur",
        training_mode="edm",
        generator="determ",
        schedule_sampler="lognormal", 
        clip_denoised=True,
        num_samples=1000,
        # batch_size=2, # for debugging
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        # steps=40,
        steps=20,
        ts="",
        weight_schedule="karras",
        seed=42,
        # sampler="multistep", # heun, dpm, ancestral, onestep, progdist, euler, multistep
        model_path="/hub_data1/sojin/cm_pretrained_weight/cd_bedroom256_lpips.pt",
        # model_path="/hub_data1/sojin/cm_pretrained_weight/ct_bedroom256.pt",
        # model_path="/hub_data1/sojin/cm_pretrained_weight/cd_cat256_lpips.pt",
        # model_path="/hub_data1/sojin/cm_pretrained_weight/ct_cat256.pt",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu', type=str, default='0')
    # parser.add_argument('--log_dir', type=str, default='/hub_data1/sojin/sampling_results/')
    parser.add_argument('--log_dir', type=str, default='./cm_official_pretrained_model/')
    parser.add_argument('--toy_exp', type=str, default='0630')
    # parser.add_argument('--toy_exp', type=str, default='None')
    parser.add_argument('-log','--log_suffix', type=str, required=True) # Experiment name, starts with tb(tesorboard) ->  tb_exp1
    parser.add_argument('--use_wandb', type=bool, default=False) # Experiment name, starts with tb(tesorboard) ->  tb_exp1

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

# 