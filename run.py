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
parser.add_argument("--num_samples", type=int, help="Sampling Number of Images", default=100)

parser.add_argument("--steps", type=str2list, help="Sampling Steps", default=['40','80','100','120','240','500','1000','2000'])
parser.add_argument("--sigma_max", type=str2list, help="Intensity of Noise", default=['40','80','100','120','240','500','1000','2000'])
parser.add_argument("--sigma_min", type=str2list, help="Intensity of Noise", default=['0.002'])

parser.add_argument('--gpu', type=int, default=3)
parser.add_argument('--log_dir', type=str, default='/hub_data2/sojin/sampling_results/gopro_clean_edm_230508_ckpt-153000')

args = parser.parse_args()

# gpus = waitGPU(args.gpus, 120)
# print("Activate GPUS : ", gpus)


for steps in args.steps:
    for sigma_max in args.sigma_max:
        for sigma_min in args.sigma_min:
            # if len(gpus) == 0:
            #     time.sleep(30)
            #     gpus = waitGPU(args.gpus, 120)
            #     print("Activate GPUS : ", gpus)
            # gpu = gpus.pop()
            
            log_name = f'steps{steps}_sigmamax{sigma_max}_min{sigma_min}'

            sub_process_log = f'{args.log_dir}/run_command.txt'
            os.makedirs(args.log_dir, exist_ok=True)

            script = f"python -u scripts/image_sample_edm.py --training_mode edm --generator determ-indiv --attention_resolutions 16,8 --class_cond False --dropout 0.1 --image_size 256 --num_channels 128 --num_res_blocks 2 --resblock_updown True --weight_schedule karras  --s_churn 0 --steps {steps} --sigma_max {sigma_max} --sigma_min {sigma_min} --gpu_num {args.gpu} --log_dir {args.log_dir} -log {log_name} --num_samples {args.num_samples}"
            print(script)
            
            f = open(sub_process_log, 'a')
            f.write(script)
            f.write('\n')
            f.close()
            subprocess.call(script, shell=True)