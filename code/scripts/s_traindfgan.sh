#!/bin/bash
#
#SBATCH --job-name=DFGAN_Text550
#SBATCH --partition=2080ti-long
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4

python main.py --cfg cfg/texture.yml --gpu 0

#python main.py --cfg cfg/bird.yml --gpu 0

#python main.py --cfg cfg/coco..yml --gpu 0


