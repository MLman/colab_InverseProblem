"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, sys, math
import random
from tqdm import tqdm
from piq import LPIPS
from einops import repeat
import numpy as np
import torch as th
import torch.distributed as dist
import torch.optim as optim

from cm import dist_util, logger
from cm.image_datasets_pairs import load_data_pairs
from cm.nn import mean_flat
from kernel_utils.deblurfunc import MotionBlurOperator, GaussianBlurOperator, GaussianDeblurOperator, DeblurOperator

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
from motionblur.motionblur import Kernel
from kernel_utils.deblurfunc import BlindBlurOperator, ToyDeblurFunc

from guided_diffusion.script_util_gradient import(
# from guided_diffusion.script_util import(
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

    norm_dict = {"loss":args.norm_loss, "img":args.norm_img, "reg_scale":args.reg_scale}

    # set save directory
    args.ori_logsuffix = args.log_suffix
    args.log_suffix = f'toyver{args.toyver}_{args.toy_exp}_GT{args.gt}_{args.log_suffix}'
    data_name = args.data_dir.split('/')[-2]
    args.log_suffix = f'{args.log_suffix}/{data_name}_normL{args.norm_loss}_img{args.norm_img}_reg{args.reg_scale}'
    # args.log_suffix = f'{args.log_suffix}/{data_name}_normL{args.norm_loss}_reg{args.reg_scale}'

    args.log_dir = os.path.join(args.log_dir, args.log_suffix)
    args.log_suffix = '_' + args.log_suffix
    save_dir = args.log_dir
    mkdir(save_dir)
    
    dist_util.setup_dist(args.gpu)
    logger.configure(dir=save_dir, format_strs=['stdout','log','csv','tensorboard'], log_suffix=args.ori_logsuffix)

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

    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    # Load Simple Deblur kernel
    deblur_model = ToyDeblurFunc(args)
    # 230721 Meeting Related Model ckpt
    # if args.n_feats == 256:
    #     args.deblur_model_path = './results_toy/toy230718_train_deblurfunc/8img_7conv_lr2e-03_lpips/gaussiankernel16_intensity0.1/model_ep0_iter4000.pth'
    # elif args.n_feats == 512:
    #     args.deblur_model_path = './results_toy/toy230718_train_deblurfunc/8img_7conv_512feat_lr2e-03_lpips_gaussianker16/gaussiankernel8_intensity0.1/model_ep0_iter1000.pth'
    # elif args.n_feats == 768:
    #     args.deblur_model_path = 'results_toy/toy230718_train_deblurfunc/8img_7conv_768feat_lr2e-03_lpips_gaussianker16/gaussiankernel8_intensity0.1/model_ep0_iter1000.pth'
    # elif args.n_feats == 1024:
    #     args.deblur_model_path = './results_toy/toy230718_train_deblurfunc/8img_7conv_1024feat_lr2e-03_lpips_gaussianker16/gaussiankernel8_intensity0.1/model_ep0_iter1000.pth'

    deblur_model.load_state_dict(
            dist_util.load_state_dict(args.deblur_model_path, map_location="cpu")['state_dict']
        )
    deblur_model.to(dist_util.dev())
    deblur_model.eval()

    if args.use_wandb:
        import wandb
        # table_name = f"{args.wandb_table}_loss:{args.loss}_lr:{args.lr_initial}"
        table_name = f"{args.wandb_table}_{args.toy_exp}"
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
        is_toy=True
    )

    best_psnr = 0.0
    best_ssim = 0.0
    for epoch in range(0, args.epoch + 1):
        epoch_loss_dict = dict(epoch_loss=0, lpips_loss=0, l2_loss=0)
        psnr_rgb, ssim_rgb = [], []

        for i, (batch, extra) in enumerate(dataloader):

            ori_sharp = batch[:][0].to(dist_util.dev())
            x_blur = batch[:][1].to(dist_util.dev())

            forward_dir = f'{save_dir}/{i}_forward_'
            backward_dir = f'{save_dir}/{i}_backward_'

            if epoch == 0:
                vtils.save_image(ori_sharp, f'{save_dir}/Ori_sharp_{epoch}_{i}.png', range=(-1,1), normalize=True)
                vtils.save_image(x_blur, f'{save_dir}/Ori_blur_{epoch}_{i}.png', range=(-1,1), normalize=True)

            if args.gt == 'deblur':
                x_restored = deblur_model(x_blur)
            elif args.gt == 'cleanGT':
                x_restored = ori_sharp # init
            elif args.gt == 'blurGT':
                x_restored = x_blur # init

            vtils.save_image(x_restored, f'{save_dir}/x_restored_images{i}.png', range=(-1,1), normalize=True)
            
            if args.toyver == 3:
                x_restored = [ori_sharp, x_restored]

            # 2. Encoding from y' to y_t
            logger.log("encoding the source images.")
            noise_restored = diffusion.ddim_reverse_sample_loop(
                model,
                image=x_restored,
                original_image=ori_sharp,
                clip_denoised=True,
                device=dist_util.dev(),
                progress=True,
                use_wandb=args.use_wandb,
                directory=forward_dir,
                debug_mode=args.debug_mode,
                norm=norm_dict,
                toyver=args.toyver,
                exp_name=args.exp_name
            )
            if epoch == 0 and i == 0:
                vtils.save_image(noise_restored, f'{save_dir}/Ori_restored_ddim_noise.png', range=(-1,1), normalize=True)
            logger.log(f"obtained latent representation for restored images...")

            # 3. Decoding from y_t to y'_0,t
            sample_restored = diffusion.ddim_sample_loop(
                model,
                (args.batch_size, 3, 256, 256),
                noise=noise_restored,
                original_image=ori_sharp,
                image_deblur=x_restored,
                clip_denoised=True,
                device=dist_util.dev(),
                progress=True,
                use_wandb=args.use_wandb,
                directory=backward_dir,
                debug_mode=args.debug_mode
            )
            logger.log(f"obtained reconstructed restored images...")
            vtils.save_image(sample_restored, f'{save_dir}/Recon_epoch{epoch}_restored{i}.png', range=(-1,1), normalize=True)

            # y^prime = x_restored
            # y^prime_0,t = sample_restored
            if args.toyver == 3:
                if i == 7: return
                else:
                    continue
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
                raise NotImplementedError()

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

            if args.debug_mode and i ==0: return

            if i == 7: return
        epoch_psnr = sum(psnr_rgb)/len(psnr_rgb)
        epoch_ssim = sum(ssim_rgb)/len(ssim_rgb)

        if best_psnr < epoch_psnr:
            best_psnr = epoch_psnr
            # torch.save({'epoch': epoch,'state_dict': kernel.state_dict(),}, os.path.join(save_dir, "model_best.pth"))

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

        # torch.save({'epoch': epoch,'state_dict': kernel.state_dict(),}, os.path.join(save_dir, "model_last.pth"))

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
        steps=1000,

        lr_initial=2e-03, 
        
        # python scripts/image_sample_deblur_toy.py --gpu 6 --use_wandb True -log lpips_loss_lr
        step_lr=50,
        weight_decay=2e-02,
        epoch=100,

        # model_path='./models/imagenet/256x256_diffusion_uncond.pt',
        kernel='blind_blur',
        # kernel='gaussian_deblur',
        kernel_size=16,
        intensity=0.1,
        # data_dir="./easy_blur/gaussiankernel16_intensity0.1/blind_blur",
        # model_path="./models/afhqCat/guided_diffusion_afhqcat_ema_0.9999_625000.pt",
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
    parser.add_argument('--data_dir', type=str, default='./easy_blur/gaussiankernel16_intensity0.1/blind_blur')

    parser.add_argument('--log_dir', type=str, default='./results_toy/deblurfunc_debug')
    # parser.add_argument('--log_dir', type=str, default='./results_toy/toy230720_toyver1/DeblurToy_AFHQ_Cat')
    # parser.add_argument('--log_dir', type=str, default='./toy230718/DeblurToy_GoPro')
    # parser.add_argument('--log_dir', type=str, default='./results_toy/toy230718/DeblurToy_FFHQ_1K')
    parser.add_argument('--loss', type=str, default='lpips') # lpips, l2
    # parser.add_argument('--toy_exp', type=str, default='toy230720_ApplyDeblurKernel')
    parser.add_argument('--toy_exp', type=str, default='toy230723')
    parser.add_argument('-log','--log_suffix', type=str, required=True) # Experiment name, starts with tb(tesorboard) ->  tb_exp1
    parser.add_argument('--deblur_model_path', type=str, default='./results_toy/toy230722_train_deblurfunc/AFHQ/8img_7conv_256feat_lr2e-03_lpips_gaussiankernel16_intensity0.1/gaussiankernel16_intensity0.1/model_ep0_iter3000.pth')
    parser.add_argument('--model_path', type=str, default='./models/afhqCat/guided_diffusion_afhqcat_ema_0.9999_625000.pt')

    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--debug_mode', action='store_true', default=False)
    parser.add_argument('--n_feats', type=int, default=256)
    parser.add_argument('--toyver', type=int, default=2)
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    
    # parser.add_argument('--exp_name', type=str, default='A1')
    # parser.add_argument('--exp_name', type=str, default='A2')
    # parser.add_argument('--exp_name', type=str, default='B1')
    parser.add_argument('--exp_name', type=str, default='B2')

    parser.add_argument('--norm_img', type=float, default=0.01) 
    parser.add_argument('--norm_loss', type=float, default=0.01) 
    parser.add_argument('--reg_scale', type=float, default=0.01) 
    parser.add_argument('--gt', type=str, default='deblur') 


    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

