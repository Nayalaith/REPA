#!/bin/bash
#SBATCH --job-name=repa
#SBATCH --output=runs/%j.log
#SBATCH --error=runs/%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --constraint="[type_a|type_b|type_c]"
#SBATCH -A proj_1637

# Load Anaconda module
module load Python/Anaconda_v03.2023
module load CUDA/12.2

# Initialize conda for non-interactive shells
source /opt/software/python/anaconda/2023_03/etc/profile.d/conda.sh

unset PYTHONPATH
export PYTHONNOUSERSITE=1
# Activate your environment
conda activate repa
which python
python -c "import sys; print(sys.path)"
python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Run generate.py with torchrun using standalone mode
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
