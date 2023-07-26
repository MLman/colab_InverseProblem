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
from kernel_utils.deblurfunc import ToyDeblurFunc

import torchvision.utils as vtils
from torch.utils.data import DataLoader
from timm.utils import NativeScaler

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

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
    args.ori_logsuffix = args.log_suffix
    data_name = args.data_dir.split('/')[-2]
    args.log_suffix = f'{args.log_suffix}/{data_name}'

    args.log_dir = os.path.join(args.log_dir, args.log_suffix)
    args.log_suffix = '_' + args.log_suffix
    save_dir = args.log_dir
    mkdir(save_dir)
    
    dist_util.setup_dist(args.gpu)
    logger.configure(dir=save_dir, format_strs=['stdout','log','csv','tensorboard'], log_suffix=args.ori_logsuffix)

    logger.log("creating deblur model...")
    model = ToyDeblurFunc(args)
    if args.resume:
        model.load_state_dict(
            dist_util.load_state_dict(args.modelresume_path_path, map_location="cpu")
        )
    model.to(dist_util.dev())
    model.train()

    if args.use_wandb:
        import wandb
        table_name = f"{args.wandb_table}_{args.toy_exp}"
        wandb.init(project="toy", name=table_name)
        wandb.config.update(args)
    
    set_random_seed(args.seed)

    optimizer = optim.Adam(model.parameters(), lr=args.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=args.weight_decay)
    lpips = LPIPS(replace_pooling=True, reduction="none")
    loss_scaler = NativeScaler()
    dataloader = load_data_pairs(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=False,
        is_toy=True,
    )

    best_psnr = 0.0
    best_ssim = 0.0
    for epoch in range(0, args.epoch + 1):
        epoch_loss_dict = dict(epoch_loss=0, lpips_loss=0, l2_loss=0)
        psnr_rgb, ssim_rgb = [], []

        for i, (batch, extra) in enumerate(dataloader):
            ori_sharp = batch[0].to(dist_util.dev())
            x_blur = batch[1].to(dist_util.dev())

            x_restored = model(x_blur)
            if i % 100 == 0:
                vtils.save_image(ori_sharp, f'{save_dir}/orisharp_ep{epoch}_iter{i}.png', range=(-1,1), normalize=True)
                vtils.save_image(x_blur, f'{save_dir}/oriblur_ep{epoch}_iter{i}.png', range=(-1,1), normalize=True)
                vtils.save_image(x_restored, f'{save_dir}/x_restored_ep{epoch}_iter{i}.png', range=(-1,1), normalize=True)

            l2_loss = (ori_sharp - x_restored) ** 2
            l2_loss = mean_flat(l2_loss) # * weights
            l2_loss = l2_loss.mean()

            lpips_loss = lpips((x_restored + 1) / 2.0, (ori_sharp + 1) / 2.0) # * weights
            lpips_loss = mean_flat(lpips_loss) # * weights

            if args.loss == 'l2':
                loss = l2_loss
            elif args.loss == 'lpips':
                loss = lpips_loss
            else:
                raise NotImplementedError()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_dict['epoch_loss'] += loss.item()
            epoch_loss_dict['lpips_loss'] += lpips_loss.item()
            epoch_loss_dict['l2_loss'] += l2_loss.item()

            psnr, ssim = 0.0, 0.0
            for idx in range(ori_sharp.shape[0]):
                restored = th.clamp(x_restored[idx], -1., 1.).cpu().detach().numpy()
                target = th.clamp(ori_sharp[idx], -1., 1.).cpu().detach().numpy()
                ps = psnr_loss(restored, target)
                ss = ssim_loss(restored, target, data_range=2.0, multichannel=True, channel_axis=0)
                psnr += ps
                ssim += ss
                print(f"[PSNR]: %.4f, [SSIM]: %.4f"% (ps, ss)+'\n')
                
            psnr /= args.batch_size
            ssim /= args.batch_size
            psnr_rgb.append(psnr)
            ssim_rgb.append(ssim)
            
            diff = (x_blur - x_restored) ** 2
            diff = diff.mean()
            loss_dict = {"l2_loss": l2_loss, "lpips_loss": lpips_loss, "psnr": psnr_rgb[i], "ssim": ssim_rgb[i]}
            if args.use_wandb:
                wandb.log(loss_dict)

            with open(os.path.join(save_dir,'results.txt'),'a') as f:
                results = f'Epoch{epoch}: {i}th iter --->' + "diff: %.4f   [PSNR]: %.4f, [SSIM]: %.4f, [L2 loss]: %.4f, [LPIPS loss]: %.4f"% (diff, psnr, ssim, l2_loss, lpips_loss) + '\n'
                print(results)
                f.write(results)

            if i % 500 == 0:
                th.save({'epoch': epoch,'state_dict': model.state_dict(),}, os.path.join(save_dir, f"model_ep{epoch}_iter{i}.pth"))
            if i == 3000:
                return
            
        epoch_psnr = sum(psnr_rgb)/len(psnr_rgb)
        epoch_ssim = sum(ssim_rgb)/len(ssim_rgb)

        if best_psnr < epoch_psnr:
            best_psnr = epoch_psnr
            th.save({'epoch': epoch,'state_dict': model.state_dict(),}, os.path.join(save_dir, "model_best.pth"))

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


        th.save({'epoch': epoch,'state_dict': model.state_dict(),}, os.path.join(save_dir, "model_last.pth"))

    logger.log("Completed")

# python scripts/image_sample_deblur_toy.py --gpu 0 -log train
def create_argparser():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lr_initial', type=float, default=2e-03)
    parser.add_argument('--weight_decay', type=float, default=2e-02)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_feats', type=int, default=256)
    parser.add_argument('--n_resblocks', type=int, default=4)
    parser.add_argument('--rgb_range', type=int, default=255)
    parser.add_argument('--kernel_size', type=int, default=3)
    # parser.add_argument('--data_dir', type=str, default='./easy_blur/gaussiankernel16_intensity0.1/blind_blur')
    parser.add_argument('--data_dir', type=str, default='./easy_blur_ffhq/motionkernel8_intensity0.1/blind_blur')
    parser.add_argument('--resume_path', type=str, default='')
    parser.add_argument('--loss', type=str, default='lpips') # lpips, l2


    parser.add_argument('--gpu', type=str, default='6')
    parser.add_argument('--log_dir', type=str, default='./results_toy/debug')
    # parser.add_argument('--log_dir', type=str, default='./results_toy/toy230718_train_deblurfunc')
    parser.add_argument('--toy_exp', type=str, default='toy230719_TrainDeblurFunc')
    parser.add_argument('-log','--log_suffix', type=str, required=True) 

    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)

    return parser


if __name__ == "__main__":
    main()

# Done
# python scripts/train_simple_deblurfunc.py -log 8img_7conv_768feat_lr2e-03_lpips_gaussianker16 --lr_initial 2e-03 --n_feats 768 --gpu 2

