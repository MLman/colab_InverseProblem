"""
Train a diffusion model on images.
"""

import argparse
import os

from cm import dist_util, logger
from cm.image_datasets_pairs import load_data_pairs
from cm.resample import create_named_schedule_sampler
from cm.script_util_encoding import (
    model_and_diffusion_defaults,
    create_two_model_and_diffusion_sampler, # karras_diffusion_encoding.py
    args_to_dict,
    add_dict_to_argparser,
)
from cm.train_util_pairs_old import TrainLoop
import torch.distributed as dist

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    args = create_argparser().parse_args()
    args.log_dir = os.path.join(args.log_dir, args.log_suffix)
    args.log_suffix = '_' + args.log_suffix

    mkdir(args.log_dir)

    dist_util.setup_dist(args.gpu_num)
    logger.configure()

    logger.log("creating model and diffusion...")
    model_sharp, model_blur, diffusion, sampler = create_two_model_and_diffusion_sampler(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model_sharp.to(dist_util.dev())
    model_blur.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size

    data = load_data_pairs(
        data_dir=args.data_dir,
        batch_size=batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("creating data loader...")

    logger.log("training...")
    TrainLoop(
        model_sharp=model_sharp,
        model_blur=model_blur,
        sharp_ckpt=args.sharp_ckpt,
        diffusion=diffusion,
        sampler=sampler,
        data=data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        test_interval=args.test_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        save_dir=args.log_dir,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="/hub_data2/sojin/Restormer_GoPro/train",
        sharp_ckpt="/home/sojin/diffusion/ckpt-60000-0.9999pt",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=2500,
        test_interval=1000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_num', type=str, default=None)

    parser.add_argument('--log_dir', type=str, default='/hub_data2/sojin/debugging')
    parser.add_argument('-log','--log_suffix', type=str, required=True) # Experiment name, starts with tb(tesorboard) ->  tb_exp1


    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
