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


# parser.add_argument('--log_dir', type=str, default='./results_toy/0814_VGG_GramHx_debug_ddpm')

parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--nowait', action='store_true', default=False)
parser.add_argument('--kakao', action='store_true', default=False)
parser.add_argument('--ddpm', action='store_true', default=False)

parser.add_argument('--toyver', type=int, default=2)
# parser.add_argument('--diffusion_steps', type=int, default=['250', '500','1000'])
parser.add_argument('--diffusion_steps', type=int, default=['1000'])

parser.add_argument('--early_stop', type=str2list, default=["50", "100", "200"])


parser.add_argument('--reg_style', type=str2list, default=["10000", "1000", "100", "1", "0.1","-10000", "-1000", "-100", "-1", "-0.1"])
# parser.add_argument('--reg_style', type=str2list, default=["100000", "10000", "1000", "100", "1", "0.1"]) 
# parser.add_argument('--reg_style', type=str2list, default=["-100000", "-10000", "-1000", "-100", "-1", "-0.1"]) 


parser.add_argument('--reg_content', type=str2list, default=["10","100","1000","-10","-100","-1000"]) 
# parser.add_argument('--reg_content', type=str2list, default=["10","100","1000"])
# parser.add_argument('--reg_content', type=str2list, default=["-10","-100","-1000"]) 

parser.add_argument('--norm_loss', type=str2list, default=["0.1", "1", "10"]) 


# parser.add_argument('--reg_style', type=str2list, default=[ "100"])
# parser.add_argument('--reg_content', type=str2list, default=["1",]) 

# parser.add_argument('--reg_scale', type=str2list, default=["1e-3", "1", "100"]) 
parser.add_argument('--reg_scale', type=str2list, default=["1"])
parser.add_argument('--gram_type', type=str2list, default=["Hy0hat"]) # Hy0hat, y0hat, yi


# args = parser.parse_args()


# 230814 two Gram guidance
# condF_list = ['None']
# condB_list = ['y0hatGram_content_BefvggGramB', 'y0hatGram_style_BefvggGramB'] 
####
# condB_list = ['y0hatGram_content_grad_y_prev_BefvggGramB', 'y0hatGram_style_grad_y_prev_BefvggGramB']
# To compare y0hatGram_style_grad_y_prev_BefvggGramB => grad_y_prev_BefvggGramB
# condB_list = ['grad_y_prev_BefvggGramB']
####
# condB_list = ['y0hatGram_content_grad_y0hat_BefvggGramB', 'y0hatGram_style_grad_y0hat_BefvggGramB']
# To compare y0hatGram_style_grad_y0hat_BefvggGramB => grad_y0hat_BefvggGramB
# condB_list = ['grad_y0hat_BefvggGramB']
###################

# 230816: Toyver 2 -> python run_toy_gram.py --toyver 2

# 1 vggGramB
# condB_list = ['grad_y_prev_vggGramB', 'grad_y0hat_vggGramB', 'norm_vggGramB', 'raw_vggGramB']

# 2 y0hatGram_content
# 3 y0hatGram_style
# 4 y0hatGram_content_separate1,2
# 5 y0hatGram_style_separate1,2
condF_list = ['None']

##### 1 DONE
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale/content:Time_style:Reversed')
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale/content:Time_style:Exp')
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale/content:Exp_style:Time')
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale/content:Reversed_style:Time')

condB_list = ['grad_y_prev_vggGramB'] # done
# condB_list = ['grad_y0hat_vggGramB'] # done
#######################################

# ##### 2 DONE
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram/content/content:Time_style:Reversed/y_prev')
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram/content/content:Time_style:Reversed/y0hat')
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram/content/content:Reversed_style:Time/y_prev')
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram/content/content:Reversed_style:Time/y0hat')
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram/content/content:Time_style:Exp/y_prev')
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram/content/content:Exp_style:Time/y_prev')

# condB_list = ['y0hatGram_content_grad_y_prev_vggGramB']
# condB_list = ['y0hatGram_content_grad_y0hat_vggGramB']

# # # # # # # # # # # # # # # # # # # # # # # # # 
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram/style/content:Time_style:Reversed/y_prev')
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram/style/content:Time_style:Reversed/y0hat') 
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram/style/content:Reversed_style:Time/y_prev')
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram/style/content:Reversed_style:Time/y0hat')
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram/style/content:Time_style:Exp/y_prev')
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram/style/content:Exp_style:Time/y_prev')

condB_list = ['y0hatGram_style_grad_y_prev_vggGramB'] 
# condB_list = ['y0hatGram_style_grad_y0hat_vggGramB'] 
# #######################################

# ##### 3
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram_separate_case1/content/content:Time_style:Reversed/y_prev')
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram_separate_case1/content/content:Time_style:Reversed/y0hat')
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram_separate_case1/content/content:Reversed_style:Time/y_prev')
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram_separate_case1/content/content:Reversed_style:Time/y0hat')
parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram_separate_case1/content/content:Time_style:Exp/y_prev')
parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram_separate_case1/content/content:Exp_style:Time/y_prev')

condB_list = ['y0hatGram_content_grad_y_prev_vggGramB_separate_case1']
# condB_list = ['y0hatGram_content_grad_y0hat_vggGramB_separate_case1']

time_list = ['contentT_styleE']
time_list = ['contentE_styleT']
# # # # # # # # # # # # # # # # # # # # # # # # # 

# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram_separate_case1/style/y_prev')
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram_separate_case1/style/y0hat')
parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram_separate_case1/style/content:Time_style:Exp/y_prev')
parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram_separate_case1/style/content:Exp_style:Time/y_prev')

# condB_list = ['y0hatGram_style_grad_y_prev_vggGramB_separate_case1']
# condB_list = ['y0hatGram_style_grad_y0hat_vggGramB_separate_case1']

# # # # # # # # # # # # # # # # # # # # # # # # # 

# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram_separate_case2/content/y_prev')
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram_separate_case2/content/y0hat')
parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram_separate_case2/content/content:Time_style:Exp/y_prev')
parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram_separate_case2/content/content:Exp_style:Time/y_prev')


# condB_list = ['y0hatGram_content_grad_y_prev_vggGramB_separate_case2']
# condB_list = ['y0hatGram_content_grad_y0hat_vggGramB_separate_case2']
# # # # # # # # # # # # # # # # # # # # # # # # # 

# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram_separate_case2/style/y_prev')
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram_separate_case2/style/y0hat')

parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram_separate_case2/style/content:Time_style:Exp/y_prev')
parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_GramHx_toy2_NoEnc_DiffScale_y0hatGram_separate_case2/style/content:Exp_style:Time/y_prev')

# condB_list = ['y0hatGram_style_grad_y_prev_vggGramB_separate_case2']
# condB_list = ['y0hatGram_style_grad_y0hat_vggGramB_separate_case2']

# time_list = ['contentT_styleR', 'contentR_styleT'] 

## 여기까지

# time_list = ['contentT_styleR']
# # time_list = ['contentR_styleT']
# time_list = ['contentT_styleR', 'contentR_styleT']

# For DEBUG SR
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_SuperRes_no_gradB') # Not Implemented for SR
# condB_list = ['no_gradB'] # Not Implemented for SR
# parser.add_argument('--log_dir', type=str, default='./results_toy/0816_VGG_SuperRes_BefvggGram_y_prev')
# condB_list = ['grad_y_prev_BefvggGramB'] # For toy 1 -> Bef/ Aft
# time_list = ['time']

args = parser.parse_args()



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
                
                if args.toyver == 1 or args.toyver == 2:
                    all_exp_cond_list.append(f'{condF}_{condB}_{times}_{es}')
                

print(all_exp_cond_list)

# if ('vgg' in condF_list) or ('vgg' in condB_list):
all_exp_cnt = len(args.norm_loss) * len(all_exp_cond_list) * len(args.reg_style) * len(args.reg_content)
# else:
# all_exp_cnt = len(args.norm_loss) * len(all_exp_cond_list)

print(all_exp_cnt)


task_config_list = [
    'configs/noise_0.05/gaussian_deblur_config.yaml', # OK
    # 'configs/noise_0.05/inpainting_box_config.yaml', # OK
    # 'configs/noise_0.05/inpainting_config.yaml',
    # 'configs/noise_0.05/motion_deblur_config.yaml', # OK
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
log_dir = f'{args.log_dir}/toyver{args.toyver}'

for task_config in task_config_list:
    task_name = task_config.split('/')[-1].split('_config')[0]
    
    for exp_name in all_exp_cond_list:
    
        for reg_scale in args.reg_scale:
            for norm_loss in args.norm_loss:
                for diffusion_steps in args.diffusion_steps:

                    if 'Gram' in exp_name:
                        for gram_type in args.gram_type:
                            for reg_style in args.reg_style:
                                for reg_content in args.reg_content:

                                    log_name = f'{exp_name}_toyver{args.toyver}_{task_name}/time{diffusion_steps}gram{gram_type}_normL{norm_loss}_reg{reg_scale}_style{reg_style}_content{reg_content}'

                                    if 'early_stop' in exp_name: ## ddpm is not implemented earlyStopping
                                        for early_stop in args.early_stop:

                                            if args.kakao:
                                                script = f'python scripts/image_sample_nonblind_grammatrix.py --kakao --run --reg_content {reg_content} --reg_style {reg_style} --gram_type {gram_type}--early_stop {early_stop} --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir /app/outputs -log {log_name} --gpu 0'
                                            elif args.debug:
                                                script = f'python scripts/image_sample_nonblind_grammatrix_debug.py --run --reg_content {reg_content} --reg_style {reg_style} --gram_type {gram_type} --early_stop {early_stop} --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir {log_dir} -log {log_name} --gpu {gpus[0]}'
                                            else:
                                                script = f'python scripts/image_sample_nonblind_grammatrix.py --run --reg_content {reg_content} --reg_style {reg_style} --gram_type {gram_type} --early_stop {early_stop} --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir {log_dir} -log {log_name} --gpu {gpus[0]}'
                                            
                                            print(f"Start {cur_cnt}/{all_exp_cnt}")
                                            subprocess.call(script, shell=True) 
                                            cur_cnt += 1
                                    else:
                                        if args.kakao:
                                            script = f'python scripts/image_sample_nonblind_grammatrix.py --kakao --run --reg_content {reg_content} --reg_style {reg_style} --gram_type {gram_type} --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir /app/outputs -log {log_name} --gpu 0'
                                        else:
                                            if args.ddpm:
                                                print(f"!!! DDPM !!!")
                                                script = f'python scripts/image_sample_nonblind_grammatrix_ddpm.py  --run --reg_content {reg_content} --reg_style {reg_style} --gram_type {gram_type} --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir {log_dir} -log {log_name} --gpu {gpus[0]}'
                                            elif args.debug:
                                                script = f'python scripts/image_sample_nonblind_grammatrix_debug.py --no_encoding --run --reg_content {reg_content} --reg_style {reg_style} --gram_type {gram_type} --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir {log_dir} -log {log_name} --gpu {gpus[0]}'
                                            else:
                                                print(f"!!! DDIM DDIM NO Encoding!!!")
                                                script = f'python scripts/image_sample_nonblind_grammatrix.py --no_encoding --run --reg_content {reg_content} --reg_style {reg_style} --gram_type {gram_type} --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir {log_dir} -log {log_name} --gpu {gpus[0]}'

                                        print(f"Start {cur_cnt}/{all_exp_cnt}")
                                        subprocess.call(script, shell=True)
                                        cur_cnt += 1

                    else:
                        log_name = f'{exp_name}_toyver{args.toyver}_{task_name}/time{diffusion_steps}_normL{norm_loss}_reg{reg_scale}'
                        
                        if 'early_stop' in exp_name: ## ddpm is not implemented earlyStopping
                            for early_stop in args.early_stop:

                                if args.kakao:
                                    script = f'python scripts/image_sample_nonblind_grammatrix.py --kakao --run --early_stop {early_stop} --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir /app/outputs -log {log_name} --gpu 0'
                                elif args.debug:
                                    script = f'python scripts/image_sample_nonblind_grammatrix_debug.py --run --early_stop {early_stop} --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir {log_dir} -log {log_name} --gpu {gpus[0]}'
                                else:
                                    script = f'python scripts/image_sample_nonblind_grammatrix.py --run --early_stop {early_stop} --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir {log_dir} -log {log_name} --gpu {gpus[0]}'
                                
                                print(f"Start {cur_cnt}/{all_exp_cnt}")
                                subprocess.call(script, shell=True)
                                cur_cnt += 1
                        else:
                            if args.kakao:
                                script = f'python scripts/image_sample_nonblind_grammatrix.py --kakao --run --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir /app/outputs -log {log_name} --gpu 0'
                            else:
                                if args.ddpm:
                                    print(f"!!! DDPM !!!")
                                    script = f'python scripts/image_sample_nonblind_grammatrix_ddpm.py --run --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir {log_dir} -log {log_name} --gpu {gpus[0]}'
                                elif args.debug:
                                    script = f'python scripts/image_sample_nonblind_grammatrix_debug.py --no_encoding --run --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir {log_dir} -log {log_name} --gpu {gpus[0]}'
                                else:
                                    # print(f"!!! DDIM DDIM!!!")
                                    print(f"!!! DDIM DDIM NO Encoding!!!")
                                    script = f'python scripts/image_sample_nonblind_grammatrix.py --no_encoding --run --diffusion_steps {diffusion_steps} --toyver {args.toyver} --task_config {task_config} --exp_name {exp_name} --reg_scale {reg_scale} --norm_loss {norm_loss} --log_dir {log_dir} -log {log_name} --gpu {gpus[0]}'
                                    
                            
                            print(f"Start {cur_cnt}/{all_exp_cnt}")
                            subprocess.call(script, shell=True) 
                            cur_cnt += 1

