"""
Train a diffusion model on images.
"""

import argparse
import os

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from cm import dist_util, logger
from cm.image_datasets import load_data
from cm.resample import create_named_schedule_sampler
from cm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion_sampler,
    args_to_dict,
    add_dict_to_argparser,
)
# from cm.train_util import TrainLoop
from cm.train_util_accelerate import TrainLoop
from cm.augment import AugmentPipe
import torch.distributed as dist

def cycle(dl):
    while True:
        for data in dl:
            yield data

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    args = create_argparser().parse_args()
    args.log_dir = os.path.join(args.log_dir, args.log_suffix)
    args.log_suffix = '_' + args.log_suffix

    mkdir(args.log_dir)
    
    # dist_util.setup_dist()
    logger.configure(dir=args.log_dir, format_strs=['stdout','log','csv','tensorboard'], log_suffix=args.log_suffix)

    # Check for configuration
    if not args.augment:
        assert args.augment_dim == 0
        augmentpipe = None
    else:
        logger.log('Non-leaky Augmentation activated')
        augmentpipe = AugmentPipe(p = 0.2, xflip = 1e8, yflip=1, scale=1, rotate_frac=1, aniso = 1, translate_frac=1)

    logger.log("creating model and diffusion...")
    model, diffusion, sampler = create_model_and_diffusion_sampler(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    # if args.batch_size == -1:
    #     batch_size = args.global_batch_size // dist.get_world_size()
    #     if args.global_batch_size % dist.get_world_size() != 0:
    #         logger.log(
    #             f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
    #         )
    # else:
    #     batch_size = args.batch_size
    batch_size = args.global_batch_size

    if args.user_data:
        if args.dataset == 'cifar10':
            transform_list = transforms.Compose([
                #transforms.Resize((args.resolution, args.resolution)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            dataset = dsets.CIFAR10(args.data_dir, transform = transform_list, download = True)
        elif args.dataset == 'gopro':
            transform_list = transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                #transforms.RandomHorizontalFlip(0.3),
                #transforms.RandomVerticalFlip(0.3),
                #transforms.RandomRotation(degrees = (0, 180)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            dataset = dsets.ImageFolder(args.data_dir, transform = transform_list)
        else:
            raise ValueError('Invalid Dataset')

        data = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True,
                                    pin_memory = True, drop_last = True)
        data = cycle(data)
        
        '''
        if args.blur_data:
            transform_list = transforms.Compose([
            #transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            blur_dataset = dsets.ImageFolder(args.blurdata_dir, transform = transform_list)

            blurdata = torch.utils.data.DataLoader(blur_dataset, batch_size = batch_size, shuffle = True,
                                        pin_memory = True, drop_last = True)
            data = cycle(data)
        '''
    else:
        data = load_data(
            data_dir=args.data_dir,
            batch_size=batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
        )

    logger.log("creating data loader...")

    logger.log("training...")
    TrainLoop(
        model=model,
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
        log_dir=args.log_dir,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="/hub_data2/sojin/Restormer_GoPro/train/input",
        augment_dim = 0,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        test_interval=1000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default='/hub_data2/sojin/tmp')
    parser.add_argument('-log','--log_suffix', type=str, required=True) # Experiment name, starts with tb(tesorboard) ->  tb_exp1

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
