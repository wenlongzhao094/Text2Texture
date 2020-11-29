import numpy as np
import os
import pdb
import sys

exp_name = '06_train_stylegan5'
root = '/home/zezhoucheng/NLP/DF-GAN/code/experiments'

script_path = os.path.join(root, exp_name, 'scripts')
out_dir = os.path.join(root, exp_name, 'outs')
slurm_dir = os.path.join(root, exp_name, 'slurms')
model_path = os.path.join(root, exp_name, 'models')

if not os.path.exists(script_path):
    os.makedirs(script_path)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(slurm_dir):
    os.makedirs(slurm_dir)
if not os.path.exists(model_path):
    os.makedirs(model_path)

launch_eval_script = os.path.join(script_path, 'launch_train.sh')
f_eval_launch = open(launch_eval_script, 'w')
f_eval_launch.write('#!/bin/bash\n')



exp_name = '0-stylegan-DG-text'
out_path = os.path.join(out_dir, exp_name)
if not os.path.exists(out_path):
    os.makedirs(out_path)
sh_filename = os.path.join(script_path, 'train_%s.sh' % exp_name)
slurm_name = sh_filename.split('/')[-1][:-3]
cmd_str = 'CUDA_VISIBLE_DEVICE=0 python main_stylegan5.py --mixing --out_path %s --model_path %s' % (out_path, model_path)
with open(sh_filename, 'w') as f:
    f.write('#!/bin/bash\n')
    f.write(cmd_str)
f_eval_launch.write('sbatch -p 1080ti-long -o %s --gres=gpu:1 --mem=100000 %s' % ( 
                    slurm_dir+'/'+slurm_name+'_%J.out', sh_filename + '\n'))


