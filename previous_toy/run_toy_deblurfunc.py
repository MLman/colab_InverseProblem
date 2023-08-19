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
parser.add_argument('--gpus', default=[0], type=str2list)
# parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--n_feats', type=str2list, default=['256', '512', '1024'])

args = parser.parse_args()

gpus = waitGPU(args.gpus, 120)
# gpus = [args.gpu]
print("Activate GPUS : ", gpus)
gpus = gpus[0]

# data_dir_list = [
#     './easy_blur/gaussiankernel16_intensity0.1/blind_blur',
#     './easy_blur/gaussiankernel16_intensity0.3/blind_blur',
#     './easy_blur/motionkernel16_intensity0.1/blind_blur',
#     './easy_blur/motionkernel16_intensity0.3/blind_blur',
# ]

script_list = []
for n_feat in args.n_feats: 
    
    # FFHQ
    log_dir = './results_toy/toy230722_train_deblurfunc/FFHQ'
    sub_process_log = f'{log_dir}/run_command.txt'
    os.makedirs(log_dir, exist_ok=True)
    
    data_dir = './easy_blur_ffhq/gaussiankernel16_intensity0.1/blind_blur'
    log_name = f'8img_7conv_{n_feat}feat_lr2e-03_lpips_gaussianker16_intensity0.1'
    script = f'python scripts/train_simple_deblurfunc.py -log {log_name} --log_dir {log_dir} --data_dir {data_dir} --lr_initial 2e-03 --n_feats {n_feat} --gpu {gpus}'
    script_list.append(script)

    data_dir = './easy_blur_ffhq/gaussiankernel16_intensity0.3/blind_blur'
    log_name = f'8img_7conv_{n_feat}feat_lr2e-03_lpips_gaussianker16_intensity0.3'
    script = f'python scripts/train_simple_deblurfunc.py -log {log_name} --log_dir {log_dir} --data_dir {data_dir} --lr_initial 2e-03 --n_feats {n_feat} --gpu {gpus}'
    script_list.append(script)
    
    data_dir = './easy_blur_ffhq/motionkernel16_intensity0.1/blind_blur'
    log_name = f'8img_7conv_{n_feat}feat_lr2e-03_lpips_motionkernel16_intensity0.1'
    script = f'python scripts/train_simple_deblurfunc.py -log {log_name} --log_dir {log_dir} --data_dir {data_dir} --lr_initial 2e-03 --n_feats {n_feat} --gpu {gpus}'
    script_list.append(script)
               
    data_dir = './easy_blur_ffhq/motionkernel16_intensity0.3/blind_blur'
    log_name = f'8img_7conv_{n_feat}feat_lr2e-03_lpips_motionkernel16_intensity0.3'
    script = f'python scripts/train_simple_deblurfunc.py -log {log_name} --log_dir {log_dir} --data_dir {data_dir} --lr_initial 2e-03 --n_feats {n_feat} --gpu {gpus}'
    script_list.append(script)
    
    
    # AFHQ
    log_dir = './results_toy/toy230722_train_deblurfunc/AFHQ'
    sub_process_log = f'{log_dir}/run_command.txt'
    os.makedirs(log_dir, exist_ok=True)
    
    data_dir = './easy_blur/gaussiankernel16_intensity0.1/blind_blur'
    log_name = f'8img_7conv_{n_feat}feat_lr2e-03_lpips_gaussianker16_intensity0.1'
    script = f'python scripts/train_simple_deblurfunc.py -log {log_name} --log_dir {log_dir} --data_dir {data_dir} --lr_initial 2e-03 --n_feats {n_feat} --gpu {gpus}'
    script_list.append(script)

    data_dir = './easy_blur/gaussiankernel16_intensity0.3/blind_blur'
    log_name = f'8img_7conv_{n_feat}feat_lr2e-03_lpips_gaussianker16_intensity0.3'
    script = f'python scripts/train_simple_deblurfunc.py -log {log_name} --log_dir {log_dir} --data_dir {data_dir} --lr_initial 2e-03 --n_feats {n_feat} --gpu {gpus}'
    script_list.append(script)
    
    data_dir = './easy_blur/motionkernel16_intensity0.1/blind_blur'
    log_name = f'8img_7conv_{n_feat}feat_lr2e-03_lpips_motionkernel16_intensity0.1'
    script = f'python scripts/train_simple_deblurfunc.py -log {log_name} --log_dir {log_dir} --data_dir {data_dir} --lr_initial 2e-03 --n_feats {n_feat} --gpu {gpus}'
    script_list.append(script)
              
    data_dir = './easy_blur/motionkernel16_intensity0.3/blind_blur'
    log_name = f'8img_7conv_{n_feat}feat_lr2e-03_lpips_motionkernel16_intensity0.3'
    script = f'python scripts/train_simple_deblurfunc.py -log {log_name} --log_dir {log_dir} --data_dir {data_dir} --lr_initial 2e-03 --n_feats {n_feat} --gpu {gpus}'
    script_list.append(script)
    

for script in script_list:
    print(script)
    f = open(sub_process_log, 'a')
    f.write(script)
    f.write('\n')
    f.close()
    subprocess.call(script, shell=True)
