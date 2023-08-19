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
# parser.add_argument('--gpus', default=[0, 1, 2, 3, 4, 5, 6, 7], type=str2list)
parser.add_argument('--gpus', default=[5], type=str2list)
# parser.add_argument('--log_dir', type=str, default='./results_toy/0727_nonBlind_advanced_goodresults')
# parser.add_argument('--log_dir', type=str, default='./results_toy/0727_nonBlind_cond_nograd')
parser.add_argument('--log_dir', type=str, default='./results_toy/0727_nonBlind_aftermeeting_timescheduling')
# parser.add_argument('--log_dir', type=str, default='./results_toy/0802_nonBlind')

parser.add_argument('--toyver', type=int, default=2)
parser.add_argument('--diffusion_steps', type=int, default=['250', '500','1000'])
parser.add_argument('--early_stop', type=str2list, default=["50", "100", "200"])

parser.add_argument('--norm_loss', type=str2list, default=["1e-3", "1", "5"]) 

parser.add_argument('--reg_scale', type=str2list, default=["1e-3", "1", "100"]) 

parser.add_argument('--forward_free_type', type=str2list, default=["linear_increase", "time_scale"]) # toy2
parser.add_argument('--forward_free', type=str2list, default=["1e-5", "1e-3", "5e-1", "1e-1", "-0.001", "-0.05", "-0.5"]) # toy2

args = parser.parse_args()

# measurement based condition list:
condF_list = ['condF', 'condF_no_gradF', 'None'] # 3
condB_list = ['condB', 'condB_no_gradB', 'None'] # 3

# time scaling list:
time_list = ['time', 'reversed', 'exp', 'None'] # 4

# early_stop list:
early_stop_list = ['early_stop', 'None'] # 2

ddpmF_list = ['ddpmF', 'None'] # 2 for only toy2

all_exp_cond_list = []
for condF in condF_list:
    for condB in condB_list:
        for times in time_list:
            for es in early_stop_list:
                
                if args.toyver == 1:
                    all_exp_cond_list.append(f'{condF}_{condB}_{times}_{es}')
                
                elif args.toyver == 2:
                    for ddpmF in ddpmF_list:
                        all_exp_cond_list.append(f'{condF}_{condB}_{times}_{es}_{ddpmF}')

print(all_exp_cond_list)
print(len(all_exp_cond_list)) # toy1: 72, toy2: 144



sub_process_log = f'{args.log_dir}/run_command.txt'
os.makedirs(args.log_dir, exist_ok=True)

task_config_list = [
    'configs/noise_0.05/gaussian_deblur_config.yaml', # OK
    # 'configs/noise_0.05/inpainting_box_config.yaml', # OK
    # 'configs/noise_0.05/inpainting_config.yaml',
    # 'configs/noise_0.05/motion_deblur_config.yaml', # OK
    # 'configs/noise_0.05/nonlinear_deblur_config.yaml',
    # 'configs/noise_0.05/phase_retrieval_config.yaml',
    # 'configs/noise_0.05/super_resolution_config.yaml', # OK
]

for task_config in task_config_list:
    task_name = task_config.split('/')[-1].split('_config')[0]
    
    for reg_scale in args.reg_scale:
        for norm_loss in args.norm_loss:
            for diffusion_steps in args.diffusion_steps:
                for exp_name in all_exp_cond_list:

                    gpus = waitGPU(args.gpus, 120)
                    print("Activate GPUS : ", gpus)

                    if args.toyver == 1:
                        log_dir = os.path.join(args.log_dir, 'toyver1')
                        log_name = f'{exp_name}_toyver{args.toyver}_{task_name}/time{diffusion_steps}normL{norm_loss}_reg{reg_scale}'
                        
                        if 'early_stop' in exp_name:
                            for early_stop in args.early_stop:
                                script = f'python scripts/image_sample_nonblind.py --run --early_stop {early_stop} --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir {log_dir} -log {log_name} --gpu {gpus[0]}'
                                subprocess.call(script, shell=True) 
                        else:
                            script = f'python scripts/image_sample_nonblind.py --run --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir {log_dir} -log {log_name} --gpu {gpus[0]}'
                            subprocess.call(script, shell=True) 


                    elif args.toyver == 2:
                        log_dir = os.path.join(args.log_dir, 'toyver2')
                        
                        for forward_free_type in args.forward_free_type:
                            for forward_free in args.forward_free:
                                log_name = f'{exp_name}_toyver{args.toyver}_forfree{forward_free_type}{forward_free}_{task_name}/time{diffusion_steps}normL{norm_loss}_reg{reg_scale}'

                                if 'early_stop' in exp_name:
                                    for early_stop in args.early_stop:
                                        script = f'python scripts/image_sample_nonblind.py --run --forward_free_type {forward_free_type} --forward_free {forward_free} --early_stop {early_stop} --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir {log_dir} -log {log_name} --gpu {gpus[0]}'
                                        subprocess.call(script, shell=True) 
                                else:
                                    script = f'python scripts/image_sample_nonblind.py --run --forward_free_type {forward_free_type} --forward_free {forward_free} --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir {log_dir} -log {log_name} --gpu {gpus[0]}'
                                    subprocess.call(script, shell=True) 

                                # debug
                                # script = f'python scripts/image_sample_nonblind_debug.py --forward_free {forward_free} --run --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir {log_dir} -log {log_name} --gpu {gpus[0]}'
                                # script = f'python scripts/image_sample_nonblind_debug.py --debug_mode --run --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir {log_dir} -log {log_name} --gpu {gpus[0]}'

                            # print(script)
                            # f = open(sub_process_log, 'a')
                            # f.write(script)
                            # f.write('\n')
                            # f.close()
                            # subprocess.call(script, shell=True) 
