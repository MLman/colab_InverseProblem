"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from functools import partial
import argparse
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import yaml
import random
from tqdm import tqdm
from piq import LPIPS
import numpy as np
import torch as th
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib.pyplot as plt

from util.img_utils import clear_color, mask_generator, add_caption_to_image
from data.dataloader import get_dataset, get_dataloader

from cm import dist_util, logger
from cm.nn import mean_flat
from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.svd_operators import Deblurring, SRConv
from guided_diffusion.gram_util import GramModel
from torchinfo import summary
import torchvision.utils as vtils

from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

from guided_diffusion.script_util_nonblind_grammatrix import(
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

def list_of_strings(arg):
    return arg.split(',')

def main():
    args = create_argparser().parse_args()
    set_random_seed(args.seed)

    task_config = load_yaml(args.task_config)
    measure_config = task_config['measurement']
    task_name = measure_config['operator']['name']

    norm_dict = {"loss":args.norm_loss, "reg_dps":args.reg_dps, "reg_style":args.reg_style, "reg_content":args.reg_content, \
                 }
        
    if args.log_suffix is not None:
        args.log_suffix = f'{args.log_suffix}_{args.exp_name}'
    
    args.sub_directory = f'{task_name}_toyver{args.toyver}/{args.exp_name}/time{args.diffusion_steps}normL{args.norm_loss}'
    if "Gram" in args.exp_name:
        lay_sty = '_'.join(args.layer_style)
        lay_con = '_'.join(args.layer_content)
        args.log_suffix = f'{args.sub_directory}/laySty{lay_sty}_layCon{lay_con}/regDPS{args.reg_dps}regSty{args.reg_style}_regCon{args.reg_content}'
    else:
        args.log_suffix = f'{args.sub_directory}/regDPS{args.reg_dps}'

    # set save directory
    if args.run:
        args.global_result_path = os.path.join(args.log_dir, 'total_results')
        mkdir(args.global_result_path)
        
        # args.sub_result_path = os.path.join(args.log_dir, args.sub_directory, 'sub_results')
        # mkdir(args.sub_result_path)

    args.log_dir = os.path.join(args.log_dir, args.log_suffix)
    args.save_dir = args.log_dir
    mkdir(args.save_dir)
    
    dist_util.setup_dist(args.gpu)
    logger.configure(dir=args.save_dir)

    if args.kakao:
        model_path = args.model_path.split('/')[-1]
        args.model_path = os.path.join('/app/input/dataset/dps/dps-checkpoint', model_path)

    logger.log("creating model and diffusion...")
    if 'ffhq' in args.model_path:
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

    th.set_default_device(dist_util.dev())
    # Prepare VGG Network for Gram Matrix
    # vgg_cnn = models.vgg19(pretrained=True).to(dist_util.dev())
    # vgg_cnn = vgg_cnn.features.eval()
    vgg_cnn = models.vgg19(pretrained=True).features.eval()

    # Prepare Operator and noise
    if measure_config['operator']['name'] == 'gaussian_blur':
        sigma = measure_config['operator']['intensity']
        kernel_size = measure_config['operator']['kernel_size']

        def pdf(x, sigma=sigma):
            return th.exp(th.Tensor([-0.5 * (x / sigma) ** 2]))
        
        kernel = th.Tensor([pdf(i) for i in range(-int(kernel_size//2), int(kernel_size//2)+1)])
        operator = Deblurring(kernel / kernel.sum(), 3, 256, dist_util.dev())
    
    elif measure_config['operator']['name'] == 'super_resolution': # bicubic
        factor = measure_config['operator']['scale_factor']

        def bicubic_kernel(x, a=-0.5):
            if abs(x) <= 1:
                return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
            elif 1 < abs(x) and abs(x) < 2:
                return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
            else:
                return 0
        k = np.zeros((factor * 4))
        for i in range(factor * 4):
            x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
            k[i] = bicubic_kernel(x)
        k = k / np.sum(k)
        kernel = th.from_numpy(k).float().to(dist_util.dev())
        operator = SRConv(kernel / kernel.sum(), 3, 256, dist_util.dev(), stride=factor)
    
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

    # Prepare dataloader
    data_config = task_config['data']
    if args.kakao:
        data_config['root'] = '/app/input/dataset/dps/ffhq_1K'
    
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
        table_name = f"{args.wandb_table}_{args.exp_name}"
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
        plt.imsave(os.path.join(out_path, f'label{fname}'), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, f'y{fname}'), clear_color(y))
        plt.imsave(os.path.join(out_path, f'y_n{fname}'), clear_color(y_n))

        forward_dir = f'{args.save_dir}/{i}_for_'
        backward_dir = f'{args.save_dir}/{i}_back_'

        x_start = y_n.requires_grad_()
        
        if 'Gram' in args.exp_name:
            if 'cleanGT' in args.exp_name:
                logger.log(f'! ! ! Clean GT TEST ! ! !')
                gram_model = GramModel(cnn=vgg_cnn, style_img=ref_img, content_img=ref_img, content_layers=args.layer_content, style_layers=args.layer_style)
            else:
                gram_model = GramModel(cnn=vgg_cnn, style_img=y_n, content_img=y_n, content_layers=args.layer_content, style_layers=args.layer_style)
            gram_model = gram_model.to(dist_util.dev())
            gram_model.eval()
            gram_model.requires_grad_(False)
        else:
            gram_model = None
            
        if args.ddpm:
            logger.log(f"############ DDPM ############")
            
            noise = th.randn(1, 3, 256, 256, device=dist_util.dev())
            sample_restored = diffusion.p_sample_loop(
                model,
                (args.batch_size, 3, 256, 256),
                noise=noise,
                operator=operator,
                clip_denoised=True,
                device=dist_util.dev(),
                progress=True,
                use_wandb=args.use_wandb,
                directory=backward_dir,
                original_image=ref_img,
                debug_mode=args.debug_mode,
                toyver=args.toyver,
                norm=norm_dict,
                measurement_cond_fn=measurement_cond_fn,
                y0_measurement=y_measurement,
                gram_model=gram_model,
                exp_name=args.exp_name
            )
        else:
            logger.log(f"!!!!!!!!!!! DDIM !!!!!!!!!!!")
            
            if args.no_encoding:
                logger.log(f"Random Noise...")
                noise_restored = th.randn(1, 3, 256, 256, device=dist_util.dev())
            else:
                logger.log("Reverse DDIM: encoding the source images.")
                noise_restored = diffusion.ddim_reverse_sample_loop(
                    model,
                    image=x_start,
                    operator=operator,
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
                    y0_measurement=y_measurement,
                    gram_model=gram_model,
                    exp_name=args.exp_name,
                )
                logger.log(f"obtained latent representation for restored images...")
            plt.imsave(os.path.join(out_path, f'ddim_noise{fname}'), clear_color(noise_restored))
            
            sample_restored = diffusion.ddim_sample_loop(
                model,
                (args.batch_size, 3, 256, 256),
                noise=noise_restored,
                operator=operator,
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
                gram_model=gram_model,
                exp_name=args.exp_name
            )
            logger.log(f"obtained reconstructed restored images...")
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
            # ss = ssim_loss(restored, target, data_range=2.0, multichannel=True, channel_axis=0)
            psnr += ps
            ss = 0
            ssim += ss
            # result = f"[PSNR]: %.4f, [SSIM]: %.4f"% (ps, ss)+'\n'
            result = f"[PSNR]: %.4f"% (ps)+'\n'
            print(result)
            logger.log(result)            
        psnr /= args.batch_size
        ssim /= args.batch_size
        
        loss_dict = {"l2_loss": l2_loss, "lpips_loss": lpips_loss}
        if args.use_wandb:
            wandb.log(loss_dict)

        # results = f'{i}th iter --->' + "[PSNR]: %.4f, [SSIM]: %.4f, [L2 loss]: %.4f, [LPIPS loss]: %.4f"% (psnr, ssim, l2_loss, lpips_loss) + '\n'
        results = f'{i}th iter --->' + "[PSNR]: %.4f,  [L2 loss]: %.4f, [LPIPS loss]: %.4f"% (psnr, l2_loss, lpips_loss) + '\n'
        logger.log(results)
        
        if args.run:
            dir_list = args.log_suffix.split('/')
            total_path = os.path.join(args.global_result_path,'_'.join(dir_list[:2]))
            mkdir(total_path)
            recon_name = '_'.join(dir_list[2:])
            img_path = os.path.join(total_path, f'{recon_name}.png')
            plt.imsave(img_path, clear_color(sample_restored))
            
            caption1 = 'psnr %.4f'% (psnr)
            # caption2 = 'ssim %.4f'% (ssim)
            caption2 = ''
            add_caption_to_image(img_path, caption1, caption2, font_path='util/NanumSquareB.ttf')

            with open(os.path.join(args.global_result_path,'total_results.txt'),'a') as f:
                save_name = '_'.join(dir_list[1:])
                # results = f'{save_name}\n' + "[PSNR]: %.4f, [SSIM]: %.4f, [L2 loss]: %.4f, [LPIPS loss]: %.4f"% (psnr, ssim, l2_loss, lpips_loss) + '\n\n'
                results = f'{save_name}\n' + "[PSNR]: %.4f, [L2 loss]: %.4f, [LPIPS loss]: %.4f"% (psnr, l2_loss, lpips_loss) + '\n\n'
                f.write(results)
        else:
            caption1 = 'psnr %.4f'% (psnr)
            # caption2 = 'ssim %.4f'% (ssim)
            caption2 = ''
            
            img_path = os.path.join(out_path, f'Recon{fname}')
            add_caption_to_image(img_path, caption1, caption2, font_path='util/NanumSquareB.ttf')

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
    parser.add_argument('--task_config', type=str, default='configs/noise_0.05/gaussian_deblur_config.yaml')

    parser.add_argument('--exp_name', type=str, default='GramB')
    parser.add_argument('--log_dir', type=str, default='./results_toy/0821_GramDebug')
    parser.add_argument('-log','--log_suffix', type=str) 
    parser.add_argument('--model_path', type=str, default='./models/ffhq_1k/ffhq_10m.pt')
    
    parser.add_argument('--kakao', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--no_encoding', action='store_true', default=False)
    parser.add_argument('--ddpm', action='store_true', default=False)
    parser.add_argument('--run', action='store_true', default=False)
    parser.add_argument('--debug_mode', action='store_true', default=False)
    
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--toyver', type=int, default=1)

    parser.add_argument('--norm_loss', type=float, default=0.1) 
    parser.add_argument('--reg_dps', type=float, default=1) 
    parser.add_argument('--reg_style', type=float, default=1000) 
    parser.add_argument('--reg_content', type=float, default=100) 
    
    parser.add_argument('--layer_style', type=str, nargs='+', default=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']) 
    parser.add_argument('--layer_content', type=str, nargs='+', default=['conv_4']) 

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

