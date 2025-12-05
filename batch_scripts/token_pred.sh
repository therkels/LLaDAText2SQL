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

python run_595_eval.py --remask_strategy Text2SQL --use_dynamic_context --save_path /scratch/eecs595f25_class_root/eecs595f25_class/llada_data/masking_and_gen_pred_res.csv