import argparse
import ast
import subprocess
import time
import pynvml
import os
from datetime import datetime
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

parser = argparse.ArgumentParser(description="Baseline Reproduce")
# parser.add_argument('--gpus', default=[0, 1, 2, 3, 4, 5, 6, 7], type=str2list)
parser.add_argument('--gpus', default=[3], type=str2list)
# parser.add_argument('--log_dir', type=str, default='./results_toy/0804_aftermeeting/noGrad_edit_scalingEq12')
# parser.add_argument('--log_dir', type=str, default='./results_toy/0806_hypTune_For_Gram_and_nograd')
# parser.add_argument('--log_dir', type=str, default='./results_toy/0807_hypTune_For_nograd') # 136 & kakao
# parser.add_argument('--log_dir', type=str, default='./results_toy/0807_hypTune_For_Gram_Hy0hat') # 136 done

# parser.add_argument('--log_dir', type=str, default='./results_toy/0807_hypTune_Noencoding_For_nograd') # kakao

# parser.add_argument('--log_dir', type=str, default='./results_toy/0809_VGG_GramHx_grad_y_prev') # norm_loss 바꿔서 다시
# parser.add_argument('--log_dir', type=str, default='./results_toy/0809_VGG_GramHx_grad_y0hat') # norm_loss 바꿔서 다시
# parser.add_argument('--log_dir', type=str, default='./results_toy/0809_VGG_GramHx_norm') # not yet

# TODO
# parser.add_argument('--log_dir', type=str, default='./results_toy/0811_VGG_GramHx_Noencoding_norm/Time') 
parser.add_argument('--log_dir', type=str, default='./results_toy/0811_VGG_GramHx_Noencoding_norm/Exp') 


# parser.add_argument('--log_dir', type=str, default='./results_toy/0809_VGG_GramHx_debug')

# parser.add_argument('--log_dir', type=str, default='./results_toy/0807_hypTune_Noencoding_For_GramPlusNo_grad') # 136
# parser.add_argument('--log_dir', type=str, default='./results_toy/0808_hypTune_Noencoding_For_condB') # 136
# parser.add_argument('--log_dir', type=str, default='./results_toy/0808_hypTune_cond_comparison_For_nograd') # kakao, 136

# parser.add_argument('--log_dir', type=str, default='./results_toy/0807_hypTune_For_GramPlusNo_grad') # kakao & 136

parser.add_argument('--kakao', action='store_true', default=False)

parser.add_argument('--toyver', type=int, default=1)
# parser.add_argument('--diffusion_steps', type=int, default=['250', '500','1000'])
# parser.add_argument('--diffusion_steps', type=int, default=['250'])
parser.add_argument('--diffusion_steps', type=int, default=['1000'])
# parser.add_argument('--diffusion_steps', type=int, default=['250', '500'])

parser.add_argument('--early_stop', type=str2list, default=["50", "100", "200"])
# parser.add_argument('--early_stop', type=str2list, default=["50"])
# parser.add_argument('--early_stop', type=str2list, default=["100"])
# parser.add_argument('--early_stop', type=str2list, default=["200"])

# parser.add_argument('--norm_loss', type=str2list, default=["1e-3", "1", "5"]) 
# parser.add_argument('--norm_loss', type=str2list, default=["-0.1"]) 
# parser.add_argument('--norm_loss', type=str2list, default=["1"]) 

# kakao
# parser.add_argument('--norm_loss', type=str2list, default=["-1000", "-100", "1000", "100"]) # 0.14A100
# parser.add_argument('--norm_loss', type=str2list, default=["-10", "-5", "10", "5", "-0.00001", "-0.000000001"])
# parser.add_argument('--norm_loss', type=str2list, default=["-1", "-0.1", "-0.01", "1", "0.1", "0.01", "0.00001", "0.000000001"]) 

# parser.add_argument('--norm_loss', type=str2list, default=["-1000", "-100", "-10", "-5", "-1", "-0.1", "-0.01", "-0.00001", "-0.000000001", "1000", "100", "10", "5", "1", "0.1", "0.01", "0.00001", "0.000000001"]) 
# parser.add_argument('--norm_loss', type=str2list, default=["-1000", "-10", "-1", "-0.1", "-0.001", "0.001", "0.1", "1", "10", "1000"]) 
# parser.add_argument('--norm_loss', type=str2list, default=["-1000", "-10"]) # 0809 수: Gram matrix -> fail 
# parser.add_argument('--norm_loss', type=str2list, default=["-1", "-0.1", "-0.001", "0.001", "0.1", "1", "10", "1000"]) 
parser.add_argument('--norm_loss', type=str2list, default=["1", "10", "-1", "-0.1", "-0.001", "0.001", "0.1", "1000"]) 

# parser.add_argument('--reg_style', type=str2list, default=["1000000", "500000", "100000", "10000", "1000", "100"])
# parser.add_argument('--reg_style', type=str2list, default=["1000000", "500000", "100000"])
parser.add_argument('--reg_style', type=str2list, default=["10000", "1000", "100", "1", "0.1"])

parser.add_argument('--reg_content', type=str2list, default=["0", "1", "100"])

# parser.add_argument('--reg_scale', type=str2list, default=["1e-3", "1", "100"]) 
# parser.add_argument('--reg_scale', type=str2list, default=["1e-3"]) 
parser.add_argument('--reg_scale', type=str2list, default=["1"])


# parser.add_argument('--feature_type', type=str2list, default=["in", "mid", "out", "in_mid", "in_out", "mid_out", "in_mid_out"]) 
parser.add_argument('--feature_type', type=str2list, default=["in"])

parser.add_argument('--gram_type', type=str2list, default=["Hy0hat"]) # Hy0hat, y0hat, yi


args = parser.parse_args()

# measurement based condition list:
# condF_list = ['no_gradF', 'condF', 'None', 'BefGramF', 'no_gradF_BefGramF', 'AftGramF', 'no_gradF_AftGramF'] 
# condB_list = ['no_gradB', 'condB', 'None', 'BefGramB', 'no_gradB_BefGramB', 'AftGramB', 'no_gradB_AftGramFB']


# 230807: Check no_grad Effect
# condF_list = ['no_gradF', 'None']
# condB_list = ['no_gradB', 'None']
# time_list = ['time', 'None', 'exp', 'reversed'] # 1A100
# time_list = ['time_div', 'None_div'] # 0.14A100

# 230807: Check Gram Effect
# condF_list = ['BefGramF', 'None']
# condB_list = ['BefGramB', 'None']
# time_list = ['time', 'None', 'exp', 'reversed'] 

# 230811: Noencoding VGG Gram
# condF_list = ['None']
# condB_list = ['grad_y_prev_BefvggGramB'] # done
# condB_list = ['grad_y0hat_BefvggGramB'] # done
# condB_list = ['norm_BefvggGramB'] # done
# time_list = ['time', 'exp'] 


# 2308??: VGG Gram
# condB_list = ['grad_y_prev_BefvggGramB', 'grad_y0hat_BefvggGramB', 'norm_BefvggGramB']

# condF_list = ['grad_y_prev_BefvggGramF', 'None']
# condB_list = ['grad_y_prev_BefvggGramB', 'None']

# condF_list = ['grad_y0hat_BefvggGramF', 'None']
# condB_list = ['grad_y0hat_BefvggGramB', 'None']

# condF_list = ['norm_BefvggGramF', 'None']
# condB_list = ['norm_BefvggGramB', 'None']

# time_list = ['time'] 
# time_list = ['exp']
# time_list = ['time', 'None', 'exp', 'reversed'] 


# 230807: Check no_grad Effect when No Encoding - kakao
# condF_list = ['None']
# condB_list = ['no_gradB']
# time_list = ['time', 'None', 'exp'] 
# time_list = ['reversed'] 
# time_list = ['time_div', 'None_div', 'None'] # kakao DONE

# 230808: Comparison with no_grad / Gram when No Encoding
# condF_list = ['None']
# condB_list = ['condB', 'None']
# time_list = ['time', 'None', 'exp', 'reversed'] 

# 230808: Comparison with no_grad / Gram
# condF_list = ['condF', 'None']
# condB_list = ['condB', 'None']
# time_list = ['time', 'None', 'exp', 'reversed'] # kakao & 136


# 230807: Check Gram Effect when No Encoding 돌려놓음
# condF_list = ['None']
# condB_list = ['BefGramB']
# time_list = ['time', 'None', 'exp', 'reversed'] # gpu0


# 230807: Check Gram Effect when No Encoding
# condF_list = ['None']
# condB_list = ['BefGramB_no_gradB', 'AftGramB_no_gradB']
# time_list = ['time_div', 'None_div', 'time', 'None', 'exp', 'reversed'] # gpu1

# 230807: Check Gram Plus No_grad Effect: kakao
# Forward시 효과 
# condF_list = ['no_gradF_BefGramF', 'no_gradF_AftGramF', 'condF_BefGramF', 'condF_AftGramF'] 
# condB_list = ['None']
# time_list = ['time_div', 'None_div', 'time', 'None', 'exp', 'reversed'] 

# Backward시 효과
# condF_list = ['None']
# condB_list = ['no_gradB_BefGramB', 'no_gradB_AftGramB', 'condB_BefGramB', 'condB_AftGramB']
# time_list = ['time_div', 'None_div', 'time', 'None', 'exp', 'reversed'] 


# time scaling list:
# time_list = ['time_div', 'None_div', 'time', 'None', 'exp', 'reversed']
# time_list = ['time', 'None', 'exp', 'reversed'] 


# early_stop list:
# early_stop_list = ['early_stop', 'None'] # 2
early_stop_list = ['None'] # 2
# early_stop_list = ['early_stop'] # 2

ddpmF_list = ['ddpmF', 'None'] # 2 for only toy2

all_exp_cond_list = []
for condF in condF_list:
    for condB in condB_list:

        if (condF == 'None') and (condB == 'None'):
            continue
        
        for times in time_list:
            for es in early_stop_list:
                
                if args.toyver == 1:
                    all_exp_cond_list.append(f'{condF}_{condB}_{times}_{es}')
                
                elif args.toyver == 2:
                    for ddpmF in ddpmF_list:
                        all_exp_cond_list.append(f'{condF}_{condB}_{times}_{es}_{ddpmF}')

print(all_exp_cond_list)
print(len(all_exp_cond_list)) # toy1: 72


# gpus = args.gpus

# sub_process_log = f'{args.log_dir}/run_command.txt'
# os.makedirs(args.log_dir, exist_ok=True)


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

                    if 'Gram' in exp_name:
                        for feature_type in args.feature_type:
                            for gram_type in args.gram_type:
                                for reg_style in args.reg_style:
                                    for reg_content in args.reg_content:
                                        if args.toyver == 1:
                                            log_dir = os.path.join(args.log_dir, 'toyver1')
                                            log_name = f'{exp_name}_toyver{args.toyver}_{task_name}/time{diffusion_steps}gram{gram_type}{feature_type}_normL{norm_loss}_reg{reg_scale}_style{reg_style}_content{reg_content}'
                                            
                                            if 'early_stop' in exp_name:
                                                for early_stop in args.early_stop:

                                                    if args.kakao:
                                                        script = f'python scripts/image_sample_nonblind_grammatrix.py --kakao --run --reg_content {reg_content} --reg_style {reg_style} --gram_type {gram_type} --feature_type {feature_type} --early_stop {early_stop} --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir /app/outputs -log {log_name} --gpu 0'
                                                    else:
                                                        gpus = waitGPU(args.gpus, 120)
                                                        print("Activate GPUS : ", gpus)
                                                        script = f'python scripts/image_sample_nonblind_grammatrix.py --run --reg_content {reg_content} --reg_style {reg_style} --gram_type {gram_type} --feature_type {feature_type} --early_stop {early_stop} --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir {log_dir} -log {log_name} --gpu {gpus[0]}'
                                                    subprocess.call(script, shell=True) 
                                            else:
                                                if args.kakao:
                                                    script = f'python scripts/image_sample_nonblind_grammatrix.py --kakao --run --reg_content {reg_content} --reg_style {reg_style} --gram_type {gram_type} --feature_type {feature_type} --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir /app/outputs -log {log_name} --gpu 0'
                                                else:
                                                    gpus = waitGPU(args.gpus, 120)
                                                    print("Activate GPUS : ", gpus)
                                                    script = f'python scripts/image_sample_nonblind_grammatrix.py --run --reg_content {reg_content} --reg_style {reg_style} --gram_type {gram_type} --feature_type {feature_type} --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir {log_dir} -log {log_name} --gpu {gpus[0]}'
                                                subprocess.call(script, shell=True) 
                    else:
              
                        if args.toyver == 1:
                            log_dir = os.path.join(args.log_dir, 'toyver1')
                            log_name = f'{exp_name}_toyver{args.toyver}_{task_name}/time{diffusion_steps}_normL{norm_loss}_reg{reg_scale}'
                            
                            if 'early_stop' in exp_name:
                                for early_stop in args.early_stop:

                                    if args.kakao:
                                        script = f'python scripts/image_sample_nonblind_grammatrix.py --kakao --run --early_stop {early_stop} --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir /app/outputs -log {log_name} --gpu 0'
                                    else:
                                        gpus = waitGPU(args.gpus, 120)
                                        print("Activate GPUS : ", gpus)
                                        script = f'python scripts/image_sample_nonblind_grammatrix.py --run --early_stop {early_stop} --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir {log_dir} -log {log_name} --gpu {gpus[0]}'
                                    subprocess.call(script, shell=True) 
                            else:
                                if args.kakao:
                                    script = f'python scripts/image_sample_nonblind_grammatrix.py --kakao --run --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir /app/outputs -log {log_name} --gpu 0'
                                else:
                                    gpus = waitGPU(args.gpus, 120)
                                    print("Activate GPUS : ", gpus)
                                    script = f'python scripts/image_sample_nonblind_grammatrix.py --run --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir {log_dir} -log {log_name} --gpu {gpus[0]}'
                                subprocess.call(script, shell=True) 