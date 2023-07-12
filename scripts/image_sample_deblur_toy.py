"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, sys, math
import random
from tqdm import tqdm
from piq import LPIPS

import numpy as np
import torch as th
import torch.distributed as dist
import torch.optim as optim

from cm import dist_util, logger
from cm.image_datasets_pairs import load_data_pairs
from cm.nn import mean_flat
import torchvision.utils as vtils
from torch.utils.data import DataLoader
from timm.utils import NativeScaler

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))

from deblur_method.Uformer.config import model_Uformer_GoPro_defaults, model_Uformer_SIDD_defaults, vit_defaults
# from deblur_method.Uformer.utils import *
import deblur_method.Uformer.utils as utils_uformer
from deblur_method.Uformer.dataset.dataset_motiondeblur import *
from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

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

def expand2square(timg,factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)

    img = torch.zeros(1,3,X,X).type_as(timg) # 3, h,w
    mask = torch.zeros(1,1,X,X).type_as(timg)

    img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg[:,:3]
    mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)
    
    return img, mask

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

    # Load Deblur model
    if "Uformer_B" in args.deblur_model_path:
        args.deblur_model_name = 'Uformer_B'
        if "GoPro" in args.deblur_model_path:
            deblur_dict = model_Uformer_GoPro_defaults()
            args.deblur_model_name += "_GoPro"
        elif "SIDD" in args.deblur_model_path:
            deblur_dict = model_Uformer_SIDD_defaults()  
            args.deblur_model_name += "_SIDD"
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    args.wandb_table += '_' + args.deblur_model_name

    test_dataset = get_validation_deblur_data(args.data_dir)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

    model_restoration= utils_uformer.get_arch(**deblur_dict)
    logger.log(f"Load Pretrained Deblur Model: {args.deblur_model_name}")
    utils_uformer.load_checkpoint(model_restoration, args.deblur_model_path)    
    # model_restoration.load_state_dict(
        # dist_util.load_state_dict(args.deblur_model_path, map_location="cpu")
    # )

    model_restoration.cuda()
    model_restoration.train()
    optimizer = optim.AdamW(model_restoration.parameters(), lr=args.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=args.weight_decay)

    # for name, param in model_restoration.named_parameters():
        # print(f'{name}: {param[:5]}')

    if args.use_wandb:
        import wandb
        table_name = f"{args.wandb_table}_loss:{args.loss}_steps:{args.steps}"
        wandb.init(project="toy", name=table_name)
        wandb.config.update(args)
    
    set_random_seed(args.seed)

    lpips = LPIPS(replace_pooling=True, reduction="none")
    loss_scaler = NativeScaler()
    dataloader = load_data_pairs(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    best_psnr = 0.0
    best_ssim = 0.0
    for epoch in range(0, args.epoch + 1):
        epoch_loss_dict = dict(epoch_loss=0, lpips_loss=0, l2_loss=0)
        psnr_rgb, ssim_rgb = [], []

        for i, (batch, extra) in enumerate(dataloader):

            x_sharp = batch[:][0].to(dist_util.dev())
            x_blur = batch[:][1].to(dist_util.dev())

            forward_dir = f'{save_dir}/{i}_forward_'
            backward_dir = f'{save_dir}/{i}_backward_'

            if epoch % 10 == 0:
                vtils.save_image(x_sharp, f'{save_dir}/Ori_sharp_{epoch}_{i}.png', range=(-1,1), normalize=True)
                vtils.save_image(x_blur, f'{save_dir}/Ori_blur_{epoch}_{i}.png', range=(-1,1), normalize=True)

            ori_sharp = x_sharp.clone()
            ori_blur = x_blur.detach().clone()

            # 1. Get Deblurred image y' from f_\psi (model_restoration)
            optimizer.zero_grad()
            x_blur = x_blur.requires_grad_(True)
            x_restored = model_restoration(x_blur) # y'=x_restored
            x_restored = torch.clamp(x_restored, 0, 1)
            # vtils.save_image(x_restored, f'x_restored_images.png', range=(-1,1), normalize=True)

            # 2. Encoding from y' to y_t
            logger.log("encoding the source images.")
            noise_sharp, noise_restored = diffusion.ddim_reverse_sample_loop(
                model,
                image=[x_sharp, x_restored],
                original_image=[ori_sharp, ori_blur],
                clip_denoised=True,
                device=dist_util.dev(),
                progress=True,
                use_wandb=args.use_wandb,
                directory=forward_dir,
                debug_mode=args.debug_mode
            )
            if epoch == 0 and i == 0:
                vtils.save_image(noise_sharp, f'{save_dir}/Ori_sharp_ddim_noise.png', range=(-1,1), normalize=True)
                vtils.save_image(noise_restored, f'{save_dir}/Ori_restored_ddim_noise.png', range=(-1,1), normalize=True)
            logger.log(f"obtained latent representation for sharp/restored images...")

            # 3. Decoding from y_t to y'_0,t
            sample_sharp, sample_restored = diffusion.ddim_sample_loop(
                model,
                (args.batch_size, 3, 256, 256),
                noise=[noise_sharp, noise_restored],
                original_image=[ori_sharp, ori_blur],
                clip_denoised=True,
                device=dist_util.dev(),
                progress=True,
                use_wandb=args.use_wandb,
                directory=backward_dir,
                debug_mode=args.debug_mode
            )
            logger.log(f"obtained reconstructed sharp/restored images...")
            vtils.save_image(sample_sharp, f'{save_dir}/Recon_epoch{epoch}_sharp{i}.png', range=(-1,1), normalize=True)
            vtils.save_image(sample_restored, f'{save_dir}/Recon_epoch{epoch}_restored{i}.png', range=(-1,1), normalize=True)

            # y^prime = x_restored
            # y^prime_0,t = sample_restored
            l2_loss = (x_restored - sample_restored) ** 2
            l2_loss = mean_flat(l2_loss) # * weights
            l2_loss = l2_loss.mean()

            lpips_loss = lpips((sample_restored + 1) / 2.0, (x_restored + 1) / 2.0) # * weights
            lpips_loss = mean_flat(lpips_loss) # * weights

            if args.loss == 'l2':
                loss = l2_loss
            elif args.loss == 'lpips':
                loss = lpips_loss
            else:
                raise NotImplementedError
            
            loss_scaler(loss, optimizer, parameters=model_restoration.parameters())

            epoch_loss_dict['epoch_loss'] += loss.item()
            epoch_loss_dict['lpips_loss'] += lpips_loss.item()
            epoch_loss_dict['l2_loss'] += l2_loss.item()

            psnr, ssim = 0.0, 0.0
            for idx in range(ori_sharp.shape[0]):
                restored = torch.clamp(sample_restored[idx], -1., 1.).cpu().detach().numpy()
                target = torch.clamp(ori_sharp[idx], -1., 1.).cpu().detach().numpy()
                ps = psnr_loss(restored, target)
                ss = ssim_loss(restored, target, data_range=2.0, multichannel=True, channel_axis=0)
                psnr += ps
                ssim += ss
                print(f"[PSNR]: %.4f, [SSIM]: %.4f"% (ps, ss)+'\n')
                
            psnr /= args.batch_size
            ssim /= args.batch_size
            psnr_rgb.append(psnr)
            ssim_rgb.append(ssim)
            
            loss_dict = {"l2_loss": l2_loss, "lpips_loss": lpips_loss, "psnr": psnr_rgb[i], "ssim": ssim_rgb[i]}
            if args.use_wandb:
                wandb.log(loss_dict)

            with open(os.path.join(save_dir,'results.txt'),'a') as f:
                results = f'Epoch{epoch}: {i}th iter --->' + "[PSNR]: %.4f, [SSIM]: %.4f, [L2 loss]: %.4f, [LPIPS loss]: %.4f"% (psnr, ssim, l2_loss, lpips_loss) + '\n'
                print(results)
                f.write(results)

            if args.debug_mode and i ==1: break

        epoch_psnr = sum(psnr_rgb)/len(psnr_rgb)
        epoch_ssim = sum(ssim_rgb)/len(ssim_rgb)

        if best_psnr < epoch_psnr:
            best_psnr = epoch_psnr
            torch.save({'epoch': epoch,'state_dict': model_restoration.state_dict(),}, os.path.join(save_dir, "model_best.pth"))

        if best_ssim < epoch_ssim:
            best_ssim = epoch_ssim

        loss_dict = {"epoch_loss": epoch_loss_dict['epoch_loss']/(i+1), \
                     "lpips_loss": epoch_loss_dict['lpips_loss']/(i+1), \
                     "l2_loss": epoch_loss_dict['l2_loss']/(i+1),\
                     "epoch_psnr": epoch_psnr, "epoch_ssim": epoch_ssim, \
                     "best_psnr": best_psnr, "best_ssim": best_ssim}
        if args.use_wandb:
            wandb.log(loss_dict)

        with open(os.path.join(save_dir,'results.txt'),'a') as f:
            results1 = f'Epoch{epoch}: ' + "loss]: %.4f, [SSIM]: %.4f"% (psnr, ssim)+'\n'
            
            print(loss_dict)
            f.write(str(loss_dict))


        if args.debug_mode and epoch==1: break

        torch.save({'epoch': epoch,'state_dict': model_restoration.state_dict(),}, os.path.join(save_dir, "model_last.pth"))

    logger.log("Completed")


def create_argparser():
    defaults = dict(
        # edm_afhq_cat
        training_mode="edm",
        generator="determ",
        schedule_sampler="lognormal", 
        clip_denoised=True,
        num_samples=1000,
        batch_size=4,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=1000,

        deblur_model_path="./models/Uformer_B/Uformer_B_pretrained_GoPro.pth",
        # deblur_model_path="./models/Uformer_B/Uformer_B_pretrained_SIDD.pth"
        optimizer="adamw",
        lr_initial=2e-03,
        step_lr=50,
        weight_decay=2e-02,
        # loss='lpips',
        # loss='l2',
        epoch=100,

        # model_path='./models/imagenet/256x256_diffusion_uncond.pt',

        data_dir="/hub_data2/sojin/afhq256_motionblur/afhqcat256_motionblur",
        model_path="./models/afhqCat/guided_diffusion_afhqcat_ema_0.9999_625000.pt",
        seed=42,
        # seed=1234,

        # data_dir="/hub_data1/sojin/ffhq_1K_motionblur",
        # model_path="./models/ffhq_1k/ffhq_10m.pt",
        # seed=42,
        # seed=1234,

        # data_dir="/hub_data1/sojin/Restormer_GoPro/train",
        # model_path="./models/GoPro/clean/ema_0.9999_187500.pt",

    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu', type=str, default='6')
    # parser.add_argument('--log_dir', type=str, default='/hub_data1/sojin/sampling_results/')
    parser.add_argument('--log_dir', type=str, default='./results_toy/toy230712')
    # parser.add_argument('--log_dir', type=str, default='./toy230711/DeblurToy_AFHQ_Cat')
    # parser.add_argument('--log_dir', type=str, default='./toy230711/DeblurToy_GoPro')
    # parser.add_argument('--log_dir', type=str, default='./toy230711/DeblurToy_FFHQ_1K')
    parser.add_argument('--loss', type=str, default='lpips') # lpips, l2
    parser.add_argument('--toy_exp', type=str, default='deblur_Uformer')
    # parser.add_argument('--toy_exp', type=str, default='None')
    parser.add_argument('-log','--log_suffix', type=str, required=True) # Experiment name, starts with tb(tesorboard) ->  tb_exp1
    parser.add_argument('--use_wandb', type=bool, default=False)
    parser.add_argument('--debug_mode', action='store_true', default=False)

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

# 