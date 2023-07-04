import argparse
import ast
import subprocess
import time
# import pynvml
import os
from datetime import datetime

# def waitGPU(gpus = ["0"], waitTime=60):
#     avail_gpus = []
#     pynvml.nvmlInit()
#     while True:
#         for gpu in gpus:
#             handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
#             if len(pynvml.nvmlDeviceGetComputeRunningProcesses(handle)) == 0:
#                 avail_gpus.append(gpu)
#                 # return gpu

#         # for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
#         #     result[gpu] = [proc.pid, proc.usedGpuMemory]

#         if len(avail_gpus) == 0:
#             print("Wait for finish")
#             time.sleep(waitTime)
#         else:
#             return avail_gpus

def on_terminate(proc):
    print("process {} terminated".format(proc))

def str2list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

parser = argparse.ArgumentParser(description="Baseline Reproduce")
# parser.add_argument('--gpus', default=[0, 1, 2, 3, 4, 5, 6, 7], type=str2list)

# parser.add_argument("--sampler", type=str2list, default=['heun', 'dpm', 'ancestral', 'onestep', 'euler', 'multistep'])
# parser.add_argument("--sampler", type=str2list, default=['heun', 'dpm', 'ancestral']) # gpu 0
parser.add_argument("--sampler", type=str2list, default=['onestep', 'euler', 'multistep']) # gpu 1

parser.add_argument('--gpu', type=int, default=3)
parser.add_argument('--log_dir', type=str, default='./toy230704')

args = parser.parse_args()

# gpus = waitGPU(args.gpus, 120)
# print("Activate GPUS : ", gpus)


sub_process_log = f'{args.log_dir}/run_command.txt'
os.makedirs(args.log_dir, exist_ok=True)

for sampler in args.sampler:

    script_list = []

    log_name = f'EDM:bedroom_sampler:{sampler}'
    script_list.append(f'python scripts/image_sample_cm_pretrained.py --training_mode edm --generator determ-indiv --batch_size 16 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler {sampler} --model_path /hub_data1/sojin/cm_pretrained_weight/edm_bedroom256_ema.pt --attention_resolutions 32,16,8 --class_cond False --dropout 0.1 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples 500 --resblock_updown True --use_fp16 True --use_scale_shift_norm False --weight_schedule karras -log {log_name} --gpu {args.gpu} --use_wandb True')
    
    log_name = f'EDM:cat_sampler:{sampler}'
    script_list.append(f'python scripts/image_sample_cm_pretrained.py --training_mode edm --generator determ-indiv --batch_size 16 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler {sampler} --model_path /hub_data1/sojin/cm_pretrained_weight/edm_cat256_ema.pt --attention_resolutions 32,16,8 --class_cond False --dropout 0.1 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples 500 --resblock_updown True --use_fp16 True --use_scale_shift_norm False --weight_schedule karras -log {log_name} --gpu {args.gpu} --use_wandb True')
    
    log_name = f'CD:bedroom_sampler:{sampler}'
    script_list.append(f'python scripts/image_sample_cm_pretrained.py --batch_size 16 --generator determ-indiv --training_mode consistency_distillation --sampler {sampler} --model_path /hub_data1/sojin/cm_pretrained_weight/ct_bedroom256.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples 100 --resblock_updown True --use_fp16 True --weight_schedule uniform -log {log_name} --gpu {args.gpu} --use_wandb True')
            
    log_name = f'CD:bedroom_sampler:{sampler}'
    script_list.append(f'python scripts/image_sample_cm_pretrained.py --batch_size 16 --training_mode consistency_distillation --sampler {sampler} --ts 0,17,39 --steps 40 --model_path /hub_data1/sojin/cm_pretrained_weight/cd_bedroom256_lpips.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples 500 --resblock_updown True --use_fp16 True --weight_schedule uniform -log {log_name} --gpu {args.gpu} --use_wandb True')

    log_name = f'CT:bedroom_sampler:{sampler}'
    script_list.append(f'python scripts/image_sample_cm_pretrained.py --batch_size 16 --training_mode consistency_distillation --sampler {sampler} --ts 0,67,150 --steps 151 --model_path /hub_data1/sojin/cm_pretrained_weight/ct_bedroom256.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples 500 --resblock_updown True --use_fp16 True --weight_schedule uniform -log {log_name} --gpu {args.gpu} --use_wandb True')

    log_name = f'CT:cat_sampler:{sampler}'
    script_list.append(f'python scripts/image_sample_cm_pretrained.py --batch_size 16 --training_mode consistency_distillation --sampler {sampler} --ts 0,62,150 --steps 151 --model_path /hub_data1/sojin/cm_pretrained_weight/ct_cat256.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples 500 --resblock_updown True --use_fp16 True --weight_schedule uniform -log {log_name} --gpu {args.gpu} --use_wandb True')
            
    for i in range(len(script_list)):
        print(script_list[i])
            
        f = open(sub_process_log, 'a')
        f.write(script_list[i])
        f.write('\n')
        f.close()
        subprocess.call(script_list[i], shell=True)