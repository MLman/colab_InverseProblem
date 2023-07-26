import argparse
import ast
import subprocess
import time
import pynvml
import os
from datetime import datetime

def waitGPU(gpus = ["0"], waitTime=60):
    avail_gpus = []
    pynvml.nvmlInit()
    while True:
        for gpu in gpus:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
            if len(pynvml.nvmlDeviceGetComputeRunningProcesses(handle)) == 0:
                avail_gpus.append(gpu)
                # return gpu

        # for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
        #     result[gpu] = [proc.pid, proc.usedGpuMemory]

        if len(avail_gpus) == 0:
            print("Wait for finish")
            time.sleep(waitTime)
        else:
            return avail_gpus

def on_terminate(proc):
    print("process {} terminated".format(proc))

def str2list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

parser = argparse.ArgumentParser(description="Baseline Reproduce")
parser.add_argument('--gpus', default=[0, 1, 2, 3, 4, 5, 6, 7], type=str2list)


parser.add_argument('--gpu', type=int, default=3)
# parser.add_argument('--log_dir', type=str, default='./results_toy/toy230720_toyver1/DeblurToy_AFHQ_Cat')
# parser.add_argument('--log_dir', type=str, default='./results_toy/toy230720_toyver2/DeblurToy_AFHQ_Cat')
# parser.add_argument('--log_dir', type=str, default='./results_toy/toy230720_toyver3/DeblurToy_AFHQ_Cat')
parser.add_argument('--log_dir', type=str, default='./results_toy/toy230723_respacing')
parser.add_argument('--data_type', type=str, default='FFHQ')
parser.add_argument('--diffusion_steps', type=int, default=['100','250','500','2000'])

parser.add_argument('--exp_name', type=str2list, default=['A1','A2','B1','B2'])
# parser.add_argument('--exp_name', type=str2list, default=['None'])
parser.add_argument('--n_feats', type=int, default=256)

parser.add_argument('--norm_img', type=str2list, default=["0.01", "1.0", "100"]) # ver2
parser.add_argument('--norm_loss', type=str2list, default=["0.01", "1.0", "100"]) # ver1,3

# parser.add_argument('--gt', type=str2list, default=["deblur", "cleanGT", "blurGT"]) 
# parser.add_argument('--gt', type=str2list, default=["deblur"]) 
parser.add_argument('--gt', type=str2list, default=["cleanGT"]) 
# parser.add_argument('--gt', type=str2list, default=["blurGT"]) 

args = parser.parse_args()

gpus = waitGPU(args.gpus, 120)
print("Activate GPUS : ", gpus)

args.log_dir = os.path.join(args.log_dir, args.data_type)
sub_process_log = f'{args.log_dir}/run_command.txt'
os.makedirs(args.log_dir, exist_ok=True)

if args.data_type == 'AFHQ':
    data_dir_list = [
            './easy_blur/gaussiankernel16_intensity0.1/blind_blur',
            './easy_blur/gaussiankernel16_intensity0.3/blind_blur',
            './easy_blur/motionkernel16_intensity0.1/blind_blur',
            './easy_blur/motionkernel16_intensity0.3/blind_blur',
        ]
    model_path="./models/afhqCat/guided_diffusion_afhqcat_ema_0.9999_625000.pt" 
elif args.data_type == 'FFHQ':
    data_dir_list = [
            './easy_blur_ffhq/gaussiankernel16_intensity0.1/blind_blur',
            './easy_blur_ffhq/gaussiankernel16_intensity0.3/blind_blur',
            './easy_blur_ffhq/motionkernel16_intensity0.1/blind_blur',
            './easy_blur_ffhq/motionkernel16_intensity0.3/blind_blur',
        ]
    model_path="./models/ffhq_1k/ffhq_10m.pt"

for diffusion_steps in args.diffusion_steps:
    for data_dir in data_dir_list:
        data_name = data_dir.split('/')[-2]
        deblur_model_path = f'./results_toy/toy230722_train_deblurfunc/{args.data_type}/8img_7conv_{args.n_feats}feat_lr2e-03_lpips_{data_name}/{data_name}/model_ep0_iter3000.pth'

        # for norm_loss in args.norm_loss: # ver 1
        #     for gt in args.gt:
        #         toyver = 1
        #         if 'advanced' in args.exp_name:
        #             log_name = f'timestep{diffusion_steps}Advanced_timescale_feat{args.n_feats}_{data_name}'
        #         else:
        #             log_name = f'timestep{diffusion_steps}timescale_feat{args.n_feats}_{data_name}'

        #         gpus = waitGPU(args.gpus, 120)
        #         print("Activate GPUS : ", gpus)

        #         script = f'python scripts/image_sample_simple_deblurfunc.py --diffusion_steps {diffusion_steps} --model_path {model_path} --data_dir {data_dir} --deblur_model_path {deblur_model_path} --toyver {toyver} --n_feats {args.n_feats} --gt {gt} --norm_loss {norm_loss} --log_dir {args.log_dir} -log {log_name} --gpu {gpus[0]}'

        #         print(script)
        #         f = open(sub_process_log, 'a')
        #         f.write(script)
        #         f.write('\n')
        #         f.close()
        #         subprocess.call(script, shell=True) 
            
        for norm_img in args.norm_img: # ver 2
            for exp_name in args.exp_name:
                for gt in args.gt:
                    toyver = 2
                    log_name = f'timestep{diffusion_steps}exp{exp_name}timescale_feat{args.n_feats}_{data_name}'

                    gpus = waitGPU(args.gpus, 120)
                    print("Activate GPUS : ", gpus)

                    script = f'python scripts/image_sample_simple_deblurfunc.py --diffusion_steps {diffusion_steps} --model_path {model_path} --data_dir {data_dir} --deblur_model_path {deblur_model_path} --toyver {toyver} --exp_name {exp_name} --n_feats {args.n_feats} --gt {gt} --norm_img {norm_img} --log_dir {args.log_dir} -log {log_name} --gpu {gpus[0]}'

                    print(script)
                    f = open(sub_process_log, 'a')
                    f.write(script)
                    f.write('\n')
                    f.close()
                    subprocess.call(script, shell=True)

        # for norm_loss in args.norm_loss: # ver 3
        #     for exp_name in args.exp_name:
        #         toyver = 3
        #         log_name = f'timestep{diffusion_steps}exp{exp_name}_timescale_feat{args.n_feats}_{data_name}'

        #         gpus = waitGPU(args.gpus, 120)
        #         print("Activate GPUS : ", gpus)

        #         script = f'python scripts/image_sample_simple_deblurfunc.py --diffusion_steps {diffusion_steps} --exp_name {exp_name} --model_path {model_path} --data_dir {data_dir} --deblur_model_path {deblur_model_path} --toyver {toyver} --n_feats {args.n_feats} --norm_loss {norm_loss} --log_dir {args.log_dir} -log {log_name} --gpu {gpus[0]}'
                
        #         print(script)
        #         f = open(sub_process_log, 'a')
        #         f.write(script)
        #         f.write('\n')
        #         f.close()
        #         subprocess.call(script, shell=True) 
            