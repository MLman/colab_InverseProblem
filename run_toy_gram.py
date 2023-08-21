import argparse
import ast
import subprocess
import time
import pynvml
import os
from datetime import datetime
from itertools import combinations

# kiml run submit --dataset dps --experiment toy-inverseproblem --image dps --instance-type 0.14A100-2-MO --num-replica 1 "python run_toy_gram.py --kakao"
# kiml run submit --dataset dps --experiment toy-inverseproblem --image dps --instance-type 1A100-16-MO --num-replica 1 "python run_toy_gram.py --kakao"

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

# Define a custom argument type for a list of integers
def list_of_strings(arg):
    return arg.split(',')
 
def make_layer_group(layer_list, layer_num):
    layers = []
    for i in combinations(layer_list, layer_num):
        layers.append(list(i))
    
    return layers

parser = argparse.ArgumentParser(description="Toy Experiments")
# parser.add_argument('--gpus', default=[0, 1, 2, 3, 4, 5, 6, 7], type=str2list)
parser.add_argument('--gpus', default=[3], type=str2list)

parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--nowait', action='store_true', default=False)
parser.add_argument('--kakao', action='store_true', default=False)

# Run option
parser.add_argument('--no_encoding', action='store_true', default=False)
parser.add_argument('--ddpm', action='store_true', default=False)

parser.add_argument('--toyver', type=int, default=1)

parser.add_argument('--diffusion_steps', type=str2list, default=['1000'])

# parser.add_argument('--norm_loss', type=str2list, default=["0.1", "1", "10"]) 
parser.add_argument('--norm_loss', type=str2list, default=["1"]) 

# parser.add_argument('--reg_dps', type=str2list, default=["0.001", "0.01", "0.1", "1", "10"]) 
parser.add_argument('--reg_dps', type=str2list, default=["0"]) 

# parser.add_argument('--reg_style', type=str2list, default=["1000000", "100000", "10000", "1000", "100", "1", "0.1", "-1000000", "-100000", "-10000", "-1000", "-100", "-1", "-0.1"])
# parser.add_argument('--reg_style', type=str2list, default=["1000000", "100000", "10000", "1000", "100"])
# parser.add_argument('--reg_style', type=str2list, default=["1", "0.1", "-1000000", "-100000", "-10000"])
parser.add_argument('--reg_style', type=str2list, default=["-1000", "-100", "-1", "-0.1"])

# parser.add_argument('--reg_content', type=str2list, default=["10","100","1000","-10","-100","-1000"]) 
parser.add_argument('--reg_content', type=str2list, default=["0"]) 

parser.add_argument('--layer_num_style', type=int, default=2) 
parser.add_argument('--layer_list_style', type=list, default=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8', 'conv_9', 'conv_10', 'conv_11', 'conv_12', 'conv_13', 'conv_14', 'conv_15', 'conv_16']) 

parser.add_argument('--layer_num_content', type=int, default=1)
parser.add_argument('--layer_list_content', type=list, default=['conv_4'])

args = parser.parse_args()

run_option = f'--run'
if args.no_encoding:
    run_option = f'{run_option} --no_encoding'
if args.ddpm:
    run_option = f'{run_option} --ddpm'

condF_list = ['None']

# args.log_dir = f'./results_toy/0821_onlyGramStyle_withCleanGT_layer1/ddpm/'
args.log_dir = f'./results_toy/0821_onlyGramStyle_withCleanGT_layer2/ddpm/'



# condB_list = ['condB']
# condB_list = ['condB_GramB']
# condB_list = ['GramB', 'condB', 'condB_GramB']
# time_list = ['time', 'None', 'reversed']

condB_list = ['GramB_cleanGT']
time_list = ['time']


is_Gram = False
all_exp_cond_list = []
for condF in condF_list:
    for condB in condB_list:

        if (condF == 'None') and (condB == 'None'):
            continue
        for times in time_list:
            exp = f'{condF}_{condB}_{times}'
            all_exp_cond_list.append(exp)
            if 'Gram' in exp:
                is_Gram = True

all_exp_cnt = len(all_exp_cond_list) * len(args.norm_loss) * len(args.reg_dps)

if is_Gram:
    layer_style = make_layer_group(args.layer_list_style, args.layer_num_style)
    layer_content = make_layer_group(args.layer_list_content, args.layer_num_content)
    print(f'layer style:{len(layer_style)}  layer content:{len(layer_content)}')
    
    all_exp_cnt = all_exp_cnt * len(args.reg_style) * len(args.reg_content) * len(layer_style) * len(layer_content)

print(all_exp_cond_list)
print(f'Total number of Exp {all_exp_cnt}')


task_config_list = [
    'configs/noise_0.05/gaussian_deblur_config.yaml', # OK
    # 'configs/noise_0.05/inpainting_box_config.yaml',
    # 'configs/noise_0.05/inpainting_config.yaml',
    # 'configs/noise_0.05/motion_deblur_config.yaml',
    # 'configs/noise_0.05/nonlinear_deblur_config.yaml',
    # 'configs/noise_0.05/phase_retrieval_config.yaml',
    # 'configs/noise_0.05/super_resolution_config.yaml', # OK
]

if args.debug or args.nowait:
    gpus = args.gpus
else:
    gpus = waitGPU(args.gpus, 120)
    print("Activate GPUS : ", gpus)

cur_cnt = 0

for task_config in task_config_list:
    task_name = task_config.split('/')[-1].split('_config')[0]
    
    for exp_name in all_exp_cond_list:
        for diffusion_steps in args.diffusion_steps:
            for norm_loss in args.norm_loss:
                for reg_dps in args.reg_dps:

                    if 'Gram' in exp_name:
                        for reg_style in args.reg_style:
                            for layer_s in layer_style:
                                layers_sty = ' '.join(layer_s)
                                for reg_content in args.reg_content:
                                    for layer_c in layer_content:
                                        layers_con = ' '.join(layer_c)

                                        if args.kakao:
                                            script = f'python scripts/image_sample_nonblind_grammatrix.py --kakao {run_option} --reg_dps {reg_dps} --reg_style {reg_style} --reg_content {reg_content} --layer_style {layers_sty} --layer_content {layers_con} --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --norm_loss {norm_loss} --log_dir /app/outputs --gpu 0'
                                        else:
                                            script = f'python scripts/image_sample_nonblind_grammatrix.py {run_option} --reg_dps {reg_dps} --reg_style {reg_style} --reg_content {reg_content} --layer_style {layers_sty} --layer_content {layers_con} --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --norm_loss {norm_loss} --log_dir {args.log_dir} --gpu {gpus[0]}'

                                        print(f"Start {cur_cnt}/{all_exp_cnt}")
                                        subprocess.call(script, shell=True)
                                        cur_cnt += 1
                    else:
                        if args.kakao:
                            script = f'python scripts/image_sample_nonblind_grammatrix.py --kakao {run_option} --reg_dps {reg_dps} --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --norm_loss {norm_loss} --log_dir /app/outputs --gpu 0'
                        else:
                            script = f'python scripts/image_sample_nonblind_grammatrix.py {run_option} --reg_dps {reg_dps} --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --norm_loss {norm_loss} --log_dir {args.log_dir} --gpu {gpus[0]}'
                                
                        print(f"Start {cur_cnt}/{all_exp_cnt}")
                        subprocess.call(script, shell=True) 
                        cur_cnt += 1