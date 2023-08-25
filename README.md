# main.ipynb

scripts/image_sample_nonblind_grammatrix.py
guided_diffusion/gaussian_diffusion_nonblind_grammatrix.py


layer_style: conv_5
layer_content: conv_2

regDPS0.0regSty-1000.0_regCon50.0
[PSNR]: 25.7867, [SSIM]: 0.7303

python scripts/image_sample_nonblind_grammatrix.py --gpu 0 --ddpm --norm_loss 1 --reg_style -1000 --reg_content 50
