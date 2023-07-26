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
parser.add_argument('--log_dir', type=str, default='./results_toy/0725_nonBlind_scale')

# parser.add_argument('--exp_name', type=str, default='time')
parser.add_argument('--exp_name', type=str2list, default=['reversed', 'exp'])

parser.add_argument('--diffusion_steps', type=int, default=['100','250','500','1000','2000'])
parser.add_argument('--norm_loss', type=str2list, default=["1e-3", "1e-1", "1", "5", "20","1000"]) 
parser.add_argument('--reg_scale', type=str2list, default=["1e-3", "1e-1", "1", "5", "20","1000"]) 


args = parser.parse_args()

sub_process_log = f'{args.log_dir}/run_command.txt'
os.makedirs(args.log_dir, exist_ok=True)

task_config_list = [
    'configs/noise_0.05/gaussian_deblur_config.yaml', # OK
    # 'configs/noise_0.05/inpainting_box_config.yaml', # OK
    # 'configs/noise_0.05/inpainting_config.yaml',
    'configs/noise_0.05/motion_deblur_config.yaml', # OK
    # 'configs/noise_0.05/nonlinear_deblur_config.yaml',
    # 'configs/noise_0.05/phase_retrieval_config.yaml',
    # 'configs/noise_0.05/super_resolution_config.yaml', # OK
]

for task_config in task_config_list:
    task_name = task_config.split('/')[-1].split('_config')[0]
    
    for norm_loss in args.norm_loss: # ver 1
        for reg_scale in args.reg_scale: # ver 1
            for diffusion_steps in args.diffusion_steps:
                for exp_name in args.exp_name:
                    toyver = 1
                    log_name = f'timescale_{exp_name}_toyver{toyver}_{task_name}/time{diffusion_steps}normL{norm_loss}_reg{reg_scale}'

                    gpus = waitGPU(args.gpus, 120)
                    print("Activate GPUS : ", gpus)

                    script = f'python scripts/image_sample_nonblind.py --run --diffusion_steps {diffusion_steps} --toyver {toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir {args.log_dir} -log {log_name} --gpu {gpus[0]}'

                    print(script)
                    f = open(sub_process_log, 'a')
                    f.write(script)
                    f.write('\n')
                    f.close()
                    subprocess.call(script, shell=True) 
