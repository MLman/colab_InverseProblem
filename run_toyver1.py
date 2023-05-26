import argparse
import ast
import subprocess
import time
import pynvml
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

parser = argparse.ArgumentParser(description="Toy Experiments")
parser.add_argument('--data_dir', type=str, default='/hub_data2/sojin/afhq_blur/afhq_cat_motionblur')
parser.add_argument('--sharp_target_model_path', type=str, default='/home/sojin/diffusion/afhq_cat_clean_edm_ckpt-300000-0.9999.pt')
parser.add_argument('--total_training_steps', type=int, default=60000)

parser.add_argument('--ode_solver', type=str2list, default=["euler", "heun"]) # euler, heun
parser.add_argument('--loss_enc_weight', type=str2list, default=["start", "end"])
parser.add_argument('--loss_dec_weight', type=str2list, default=["start", "end"])
parser.add_argument('--log_directory', type=str, default='/hub_data2/sojin/toy_ver1_afhqCat')

parser.add_argument('--gpu', type=int, default=3)

args = parser.parse_args()

# gpus = waitGPU(args.gpus, 120)
# print("Activate GPUS : ", gpus)

training_mode_list = ['case1_typeA', 'case1_typeB', 'case2_typeA', 'case2_typeB']
train_mode_type = training_mode_list[args.gpu]
args.training_mode = f'deblur_consistency_training_{train_mode_type}'
args.log = train_mode_type

for ode_solver in args.ode_solver:
    for loss_enc_weight in args.loss_enc_weight:
        for loss_dec_weight in args.loss_dec_weight:
            
            args.log_dir = f'{args.log_directory}weight_enc:{loss_enc_weight}_dec:{loss_dec_weight}'
            sub_process_log = f'{args.log_dir}/run_command.txt'
            os.makedirs(args.log_dir, exist_ok=True)

            script = f"python scripts/cm_train_encoding.py --total_training_steps {args.total_training_steps} --loss_norm l2 --attention_resolutions 16,8 --class_cond False --dropout 0.0 --ema_rate 0.9999 --global_batch_size 8 --image_size 256 --lr 0.00005 --num_channels 128 --num_res_blocks 2 --resblock_updown True --schedule_sampler uniform --use_fp16 True --use_scale_shift_norm True --weight_decay 0.0 --weight_schedule uniform --data_dir {args.data_dir} --log_dir {args.log_dir} -log {args.log} --gpu_num {args.gpu} --training_mode {args.training_mode} --target_ema_mode adaptive --start_ema 0.95 --scale_mode progressive --start_scales 2 --end_scales 150 --loss_enc_weight {loss_enc_weight} --loss_dec_weight {loss_dec_weight} --ode_solver {ode_solver}"
            print(script)
            
            f = open(sub_process_log, 'a')
            f.write(script)
            f.write('\n')
            f.close()
            subprocess.call(script, shell=True)