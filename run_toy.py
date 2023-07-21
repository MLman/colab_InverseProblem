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
parser.add_argument('--log_dir', type=str, default='./results_toy/toy230720_toyver3/DeblurToy_AFHQ_Cat')
parser.add_argument('--n_feats', type=int, default=512)
parser.add_argument('--toyver', type=int, default=3)

# parser.add_argument('--norm_img', type=str2list, default=["0.01", "0.05", "0.1"]) # ver2
# parser.add_argument('--norm_img', type=str2list, default=["0.01"]) 
# parser.add_argument('--norm_img', type=str2list, default=["0.05"]) 
# parser.add_argument('--norm_img', type=str2list, default=["0.01"]) 

# parser.add_argument('--norm_loss', type=str2list, default=["0.1", "0.05", "0.01"])  # ver1,3
# parser.add_argument('--norm_loss', type=str2list, default=["0.1"])  # ver1,3
parser.add_argument('--norm_loss', type=str2list, default=["0.05"])  # ver1,3
# parser.add_argument('--norm_loss', type=str2list, default=["0.01"])  # ver1,3


# parser.add_argument('--reg_scale', type=str2list, default=["0.1"]) 
# parser.add_argument('--reg_scale', type=str2list, default=["0.05"]) 
# parser.add_argument('--reg_scale', type=str2list, default=["0.01"]) 
# parser.add_argument('--gt', type=str2list, default=["deblur", "cleanGT", "blurGT"]) 
parser.add_argument('--gt', type=str2list, default=["blurGT"]) # toy 3 -> only blurGT

args = parser.parse_args()

gpus = waitGPU(args.gpus, 120)
print("Activate GPUS : ", gpus)


sub_process_log = f'{args.log_dir}/run_command.txt'
os.makedirs(args.log_dir, exist_ok=True)


for norm_loss in args.norm_loss: # ver 1,3
# for norm_img in args.norm_img: # ver 2
    for gt in args.gt:
        log_name = f'timescale_deblurFunc{args.n_feats}'

        # ver 1,3
        script = f'python scripts/image_sample_simple_deblurfunc.py --n_feats {args.n_feats} --gt {gt} --norm_loss {norm_loss} --log_dir {args.log_dir} -log {log_name} --gpu {gpus[0]}'
        
        # ver 2
        # script = f'python scripts/image_sample_simple_deblurfunc.py --n_feats {args.n_feats} --gt {gt} --norm_img {norm_img} --log_dir {args.log_dir} -log {log_name} --gpu {gpus[0]}'

        print(script)
        f = open(sub_process_log, 'a')
        f.write(script)
        f.write('\n')
        f.close()
        subprocess.call(script, shell=True)
