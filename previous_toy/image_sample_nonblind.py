"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from functools import partial
import argparse
import os, sys
import yaml
import random
from tqdm import tqdm
from piq import LPIPS
import numpy as np
import torch as th
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from util.img_utils import clear_color, mask_generator
from data.dataloader import get_dataset, get_dataloader

from cm import dist_util, logger
from cm.nn import mean_flat
from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.svd_operators import Deblurring

import torchvision.utils as vtils

from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

# from guided_diffusion.script_util_gradient import(
# from guided_diffusion.script_util import(
from guided_diffusion.script_util_nonblind import(
    NUM_CLASSES,
    model_and_diffusion_defaults, # AFHQ
    ffhq_model_and_diffusion_defaults, # FFHQ
    imagenet_model_and_diffusion_defaults, # ImageNet
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

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
    set_random_seed(args.seed)

    task_config = load_yaml(args.task_config)
    measure_config = task_config['measurement']
    data_name = task_config['data']['name'].upper()
    task_name = measure_config['operator']['name']

    norm_dict = {"loss":args.norm_loss, "img":args.norm_img, "reg_scale":args.reg_scale, "early_stop":args.early_stop, \
                "forward_free":args.forward_free, "forward_free_type":args.forward_free_type}

    # set save directory
    args.ori_logsuffix = args.log_suffix
    if args.run:
        args.log_suffix = args.log_suffix
    else:
        if args.toyver == 1:
            args.log_suffix = f'{args.log_suffix}/{data_name}_toyver{args.toyver}{args.exp_name}{task_name}/time{args.diffusion_steps}normL{args.norm_loss}_regscale{args.reg_scale}'
        elif args.toyver == 2:
            args.log_suffix = f'{args.log_suffix}/{data_name}_toyver{args.toyver}_{task_name}/time{args.diffusion_steps}normimg{args.norm_img}_forfree{args.forward_free_type}{args.forward_free}'
        # else:
            # args.log_suffix = f'{args.log_suffix}/{data_name}_toyver{args.toyver}_{task_name}/normL{args.norm_loss}'

    if 'early_stop' in args.exp_name:
        replace_name = f'early_stop{args.early_stop}'
        args.log_suffix = args.log_suffix.replace('early_stop', replace_name)

    args.log_dir = os.path.join(args.log_dir, args.log_suffix)
    args.save_dir = args.log_dir

    if args.kakao:
        model_path = args.model_path.split('/')[-1]
        args.model_path = os.path.join('/app/input/dataset/dps-ckpt', model_path)

    mkdir(args.save_dir)
    
    dist_util.setup_dist(args.gpu)
    # logger.configure(dir=args.save_dir, format_strs=['stdout','log','csv','tensorboard'], log_suffix=args.ori_logsuffix)
    logger.configure(dir=args.save_dir)

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
    else:
        NotImplementedError()

    diffusion_dict = {'diffusion_steps':args.diffusion_steps}
    image_dict.update(diffusion_dict)

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

    # Prepare Operator and noise
    if measure_config['operator']['name'] == 'gaussian_blur':
        sigma = measure_config['operator']['intensity']
        kernel_size = measure_config['operator']['kernel_size']

        def pdf(x, sigma=sigma):
            return th.exp(th.Tensor([-0.5 * (x / sigma) ** 2]))
        
        kernel = th.Tensor([pdf(i) for i in range(-int(kernel_size//2), int(kernel_size//2)+1)])
        operator = Deblurring(kernel / kernel.sum(), 3, 256, dist_util.dev())
    else:
        raise NotImplementedError
    # operator_default = get_operator(device=dist_util.dev(), **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")
    
    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")


    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    # for img_dir in ['input', 'recon', 'progress', 'label', 'ddim_noise']:
    #     os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']

    if args.kakao:
        data_config['root'] = '/app/input/dataset/ffhq1k'
    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )

    if args.use_wandb:
        import wandb
        table_name = f"{args.wandb_table}_{args.toy_exp}"
        wandb.init(project="toy", name=table_name)
        wandb.config.update(args)

    lpips = LPIPS(replace_pooling=True, reduction="none")
    
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(dist_util.dev())
        ref_img = ref_img * 2 - 1

        # Exception) In case of inpainging,
        if measure_config['operator'] ['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)

            # Forward measurement model (Ax + n)
            y = operator.A(ref_img, mask=mask)
        else: 
            # Forward measurement model (Ax + n)
            y = operator.A(ref_img)

        b, hwc = y.size()
        hw = hwc / 3
        h = w = int(hw ** 0.5)
        y = y.reshape((b, 3, h, w))

        y_n = noiser(y)
        y_measurement = y_n.clone()
        plt.imsave(os.path.join(out_path, f'y{fname}'), clear_color(y))
        plt.imsave(os.path.join(out_path, f'y_n{fname}'), clear_color(y_n))

        forward_dir = f'{args.save_dir}/{i}_for_'
        backward_dir = f'{args.save_dir}/{i}_back_'

        x_start = y_n.requires_grad_()
        
        logger.log("encoding the source images.")
        noise_restored = diffusion.ddim_reverse_sample_loop(
            model,
            image=x_start,
            clip_denoised=True,
            original_image=ref_img, # for PSNR, SSIM
            device=dist_util.dev(),
            progress=True,
            use_wandb=args.use_wandb,
            directory=forward_dir,
            debug_mode=args.debug_mode,
            norm=norm_dict,
            toyver=args.toyver,
            measurement_cond_fn=measurement_cond_fn,
            exp_name=args.exp_name
        )
        # plt.imsave(os.path.join(out_path, 'ddim_noise', fname), clear_color(noise_restored))
        plt.imsave(os.path.join(out_path, f'ddim_noise{fname}'), clear_color(noise_restored))

        logger.log(f"obtained latent representation for restored images...")

        sample_restored = diffusion.ddim_sample_loop(
            model,
            (args.batch_size, 3, 256, 256),
            noise=noise_restored,
            clip_denoised=True,
            device=dist_util.dev(),
            progress=True,
            use_wandb=args.use_wandb,
            directory=backward_dir,
            original_image=ref_img,
            debug_mode=args.debug_mode,
            norm=norm_dict,
            toyver=args.toyver,
            measurement_cond_fn=measurement_cond_fn,
            y0_measurement=y_measurement,
            exp_name=args.exp_name
        )
        logger.log(f"obtained reconstructed restored images...")
        # plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample_restored))
        plt.imsave(os.path.join(out_path, f'Recon{fname}'), clear_color(sample_restored))

        l2_loss = (ref_img - sample_restored) ** 2
        l2_loss = mean_flat(l2_loss) # * weights
        l2_loss = l2_loss.mean()

        lpips_loss = lpips((sample_restored + 1) / 2.0, (ref_img + 1) / 2.0) # * weights
        lpips_loss = mean_flat(lpips_loss) # * weights

        psnr, ssim = 0.0, 0.0
        for idx in range(ref_img.shape[0]):
            restored = th.clamp(sample_restored[idx], -1., 1.).cpu().detach().numpy()
            target = th.clamp(ref_img[idx], -1., 1.).cpu().detach().numpy()
            ps = psnr_loss(restored, target)
            ss = ssim_loss(restored, target, data_range=2.0, multichannel=True, channel_axis=0)
            psnr += ps
            ssim += ss
            print(f"[PSNR]: %.4f, [SSIM]: %.4f"% (ps, ss)+'\n')
            
        psnr /= args.batch_size
        ssim /= args.batch_size
        
        loss_dict = {"l2_loss": l2_loss, "lpips_loss": lpips_loss}
        if args.use_wandb:
            wandb.log(loss_dict)

        with open(os.path.join(args.save_dir,'results.txt'),'a') as f:
            results = f'{i}th iter --->' + "[PSNR]: %.4f, [SSIM]: %.4f, [L2 loss]: %.4f, [LPIPS loss]: %.4f"% (psnr, ssim, l2_loss, lpips_loss) + '\n'
            print(results)
            f.write(results)

        if args.debug_mode and i ==0: return

        return


    logger.log("Completed")


def create_argparser():
    defaults = dict(
        num_samples=1000,
        batch_size=1,
        steps=1000,
        seed=42,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu', type=str, default='6')
    # parser.add_argument('--model_config', type=str, default='configs/ffhq_model_config.yaml')
    parser.add_argument('--task_config', type=str, default='configs/noise_0.05/gaussian_deblur_config.yaml')

    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='./results_toy/0731_nonBlinddebug')
    parser.add_argument('-log','--log_suffix', type=str, required=True) # Experiment name, starts with tb(tesorboard) ->  tb_exp1
    # parser.add_argument('--model_path', type=str, default='./models/afhqCat/guided_diffusion_afhqcat_ema_0.9999_625000.pt')
    parser.add_argument('--model_path', type=str, default='./models/ffhq_1k/ffhq_10m.pt')

    parser.add_argument('--kakao', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--run', action='store_true', default=False)
    parser.add_argument('--debug_mode', action='store_true', default=False)
    parser.add_argument('--diffusion_steps', type=int, default=500)
    parser.add_argument('--toyver', type=int, default=1)

    parser.add_argument('--norm_img', type=float, default=0.01) 
    parser.add_argument('--norm_loss', type=float, default=0.01) 
    parser.add_argument('--reg_scale', type=float, default=0.01) 
    parser.add_argument('--early_stop', type=int, default=300)
    parser.add_argument('--forward_free', type=float, default=-0.1)
    parser.add_argument('--forward_free_type', type=str, default="linear_increase") # linear_increase, time_scale

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

