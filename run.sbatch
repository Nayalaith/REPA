#!/bin/bash
#SBATCH --job-name=repa
#SBATCH --output=runs/%j.log
#SBATCH --error=runs/%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --constraint="type_a"   
#SBATCH -A proj_1637
module load Python/Anaconda_v03.2023

source /opt/software/python/anaconda/2023_03/etc/profile.d/conda.sh

conda activate repa



torchrun --nnodes=1 --nproc_per_node=1 generate.py \
  --model SiT-XL/2 \
  --num-fid-samples=50000 \
  --encoder-depth=8 \
  --projector-embed-dims=768 \
  --per-proc-batch-size=8 \
  --mode=sde \
  --num-steps=250 \
  --cfg-scale=1.8 \
  --guidance-high=0.7 \
