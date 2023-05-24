"""
Train a diffusion model on images.
"""

import argparse
import os

from cm import dist_util, logger
from cm.image_datasets_pairs import load_data_pairs
from cm.resample import create_named_schedule_sampler
from cm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion_encoding_sampler,
    cm_train_defaults,
    args_to_dict,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
)
from cm.train_util_pairs import CMTrainLoop
from cm.augment import AugmentPipe
import torch.distributed as dist
import copy

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

    dist_util.setup_dist(args.gpu_num)
    # logger.configure(dir=args.log_dir, format_strs=['stdout','log','csv','tensorboard'], log_suffix=args.log_suffix)
    logger.configure(dir=args.log_dir, format_strs=['stdout','log','csv'], log_suffix=args.log_suffix)

    # Check for configuration
    if not args.augment:
        assert args.augment_dim == 0
        augmentpipe = None
    else:
        logger.log('Non-leaky Augmentation activated')
        augmentpipe = AugmentPipe(p = 0.2, xflip = 1e8, yflip=1, scale=1, rotate_frac=1, aniso = 1, translate_frac=1)

    logger.log("creating model and diffusion...")
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=args.target_ema_mode,
        start_ema=args.start_ema,
        scale_mode=args.scale_mode,
        start_scales=args.start_scales,
        end_scales=args.end_scales,
        total_steps=args.total_training_steps,
        distill_steps_per_iter=args.distill_steps_per_iter,
    )
    if args.training_mode == "progdist":
        distillation = False
    elif "consistency" in args.training_mode:
        distillation = True
    else:
        raise ValueError(f"unknown training mode {args.training_mode}")

    if args.sharp_target_model_path:
        logger.log(f"loading the shart target model from {args.sharp_target_model_path}")
        model_and_diffusion_kwargs = args_to_dict(args, model_and_diffusion_defaults().keys())
        model_and_diffusion_kwargs["distillation"] = distillation
        args_dict = {   "loss_norm": args.loss_norm, "ode_solver": args.ode_solver,
                        "loss_enc_weight": args.loss_enc_weight, "loss_dec_weight": args.loss_dec_weight,
                    }
        model_and_diffusion_kwargs.update(args_dict)
        model, diffusion, sampler = create_model_and_diffusion_encoding_sampler(**model_and_diffusion_kwargs)
        
        model.load_state_dict(
            dist_util.load_state_dict(args.sharp_target_model_path, map_location="cpu"),
        )

        model.to(dist_util.dev())
        model.train()
        if args.use_fp16:
            model.convert_to_fp16()

        ### Check loaded parameters ###
        # for param in model.parameters(): 
            # print(param)
    else:
        raise NotImplementedError("There isn't any Pretrained Target model to load statedict")

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

    # if args.user_data:
    #     if args.dataset == 'cifar10':
    #         transform_list = transforms.Compose([
    #             #transforms.Resize((args.resolution, args.resolution)),
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #         ])
    #         dataset = dsets.CIFAR10(args.data_dir, transform = transform_list, download = True)
    #     elif args.dataset == 'gopro':
    #         transform_list = transforms.Compose([
    #             transforms.Resize((args.image_size, args.image_size)),
    #             #transforms.RandomHorizontalFlip(0.3),
    #             #transforms.RandomVerticalFlip(0.3),
    #             #transforms.RandomRotation(degrees = (0, 180)),
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #         ])
    #         dataset = dsets.ImageFolder(args.data_dir, transform = transform_list)
    #     else:
    #         raise ValueError('Invalid Dataset')

    #     data = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True,
    #                                 pin_memory = True, drop_last = True)
    #     data = cycle(data)
        
    #     '''
    #     if args.blur_data:
    #         transform_list = transforms.Compose([
    #         #transforms.Resize((args.resolution, args.resolution)),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #         ])
    #         blur_dataset = dsets.ImageFolder(args.blurdata_dir, transform = transform_list)

    #         blurdata = torch.utils.data.DataLoader(blur_dataset, batch_size = batch_size, shuffle = True,
    #                                     pin_memory = True, drop_last = True)
    #         data = cycle(data)
    #     '''

    # else:
    data = load_data_pairs(
        data_dir=args.data_dir,
        batch_size=batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )



    # load the target model for distillation, if path specified.

    logger.log("creating the target model")
    target_model_and_diffusion_kwargs = copy.deepcopy(model_and_diffusion_kwargs)
    target_model, target_diffusion, target_sampler = create_model_and_diffusion_encoding_sampler(
        **target_model_and_diffusion_kwargs,
    )

    target_model.to(dist_util.dev())
    target_model.eval()

    dist_util.sync_params(target_model.parameters())
    dist_util.sync_params(target_model.buffers())

    for dst, src in zip(target_model.parameters(), model.parameters()):
        dst.data.copy_(src.data)

    if args.use_fp16:
        target_model.convert_to_fp16()

    ### Check loaded parameters ###
    # for param in target_model.parameters(): 
        # print(param)

    logger.log("training...")
    logger.log(f"with the training mode >>> {args.training_mode}")
    CMTrainLoop(
        model=model, # Blur model
        sampler = sampler,
        target_model=target_model, # Sharp model
        target_diffusion=target_diffusion, # Sharp model
        training_mode=args.training_mode,
        ema_scale_fn=ema_scale_fn,
        total_training_steps=args.total_training_steps,
        diffusion=diffusion,
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
        augmentpipe=augmentpipe,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="/hub_data2/sojin/Restormer_GoPro/train",
        sharp_target_model_path="/home/sojin/diffusion/ckpt-53000-0.9999.pt",
        augment=False,
        augment_dim=0,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10,
        test_interval=10,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cm_train_defaults())
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_num', type=str, default=None)

    # Set this parameters
    parser.add_argument('--ode_solver', type=str, default="euler") # euler, heun
    parser.add_argument('--loss_norm', type=str, default="l2")
    parser.add_argument('--loss_enc_weight', type=str, default="start")
    parser.add_argument('--loss_dec_weight', type=str, default="end")

    parser.add_argument('--log_dir', type=str, default='/hub_data2/sojin/0509debugging')
    parser.add_argument('-log','--log_suffix', type=str, required=True) # Experiment name, starts with tb(tesorboard) ->  tb_exp1

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()



# python scripts/cm_train_encoding.py --loss_norm lpips --attention_resolutions 16,8 --class_cond False --dropout 0.0 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 2 --image_size 256 --lr 0.00005 --num_channels 128 --num_res_blocks 2 --resblock_updown True --schedule_sampler uniform --use_fp16 True --use_scale_shift_norm True --weight_decay 0.0 --weight_schedule uniform --data_dir /hub_data2/sojin/Restormer_GoPro/train -log gopro_clean_train --gpu_num 3 --training_mode deblur_consistency_training_case1 --target_ema_mode adaptive --start_ema 0.95 --scale_mode progressive --start_scales 2 --end_scales 150

# CUDA_VISIBLE_DEVICES=3 python scripts/cm_train_encoding.py --attention_resolutions 16,8 --class_cond False --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 2 --image_size 256 --lr 0.0001 --num_channels 128 --num_res_blocks 2 --resblock_updown True --schedule_sampler lognormal --use_fp16 True --use_scale_shift_norm True --weight_decay 0.0 --weight_schedule karras --data_dir /hub_data2/sojin/Restormer_GoPro/train --gpu_num 2 -log tmp2