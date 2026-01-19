#!/bin/bash
#SBATCH --job-name=repa
#SBATCH --output=runs/%j.log
#SBATCH --error=runs/%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --constraint=type_a|type_b|type_c|type_e
#SBATCH -A proj_1637

############################
# Load cluster modules
############################
module load Python/Anaconda_v03.2023
module load CUDA/12.2

############################
# Proper conda initialization
############################
source /opt/software/python/anaconda/2023_03/etc/profile.d/conda.sh

############################
# Prevent pollution from ~/.local
############################
export PYTHONNOUSERSITE=1
unset PYTHONPATH

############################
# Activate environment
############################
conda activate repa

############################
# Debug information
############################
echo "Python binary: $(which python)"
python - << 'EOF'
import sys, torch, torchvision
print("sys.path:", sys.path)
print("Torch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
EOF

############################
# Launch job
############################
torchrun --nnodes=1 --nproc_per_node=1 --standalone generate.py \
  --model SiT-XL/2 \
  --num-fid-samples=50000 \
  --encoder-depth=8 \
  --projector-embed-dims=768 \
  --per-proc-batch-size=8 \
  --mode=sde \
  --num-steps=250 \
  --cfg-scale=1.8 \
  --guidance-high=0.7
