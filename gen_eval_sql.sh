#!/bin/bash
#SBATCH --job-name=llada          # job name
#SBATCH --output=job_logs/llada_%j.log  # logs
#SBATCH --nodes=1                     # nodes applying
#SBATCH --partition=spgpu           # partition
#SBATCH --ntasks=1                    # job number
#SBATCH --cpus-per-task=1             # CPU cores per task
#SBATCH --time=4:00:00               # time limit
#SBATCH --mem=32G                     # Memory
#SBATCH --gres=gpu:1                   # GPU number apply
#SBATCH --account=eecs595f25_class

export HF_HOME=/gpfs/accounts/eecs595f25_class_root/eecs595f25_class/koussa/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

source ~/.bashrc
conda activate llada
cd /home/koussa/cse595/LLaDAText2SQL
python generate_with_structure_enforcing.py