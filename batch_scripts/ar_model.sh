#!/bin/bash

#SBATCH --job-name=llada-inference
#SBATCH --account=eecs595f25_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=eval.out

CUDA_VISIBLE_DEVICES=0 && python3 eval_ar_model.py --model-name "Qwen/Qwen2.5-7B-Instruct" --temperature 0.3 --top-p 0.8 --top-k 20 --max-tokens 512 --chunk-size 1 --tensor-parallel-size 1