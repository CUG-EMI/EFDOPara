#!/bin/bash
#SBATCH -J GPU_Test
#SBATCH --account=luosheng_w
#SBATCH --cluster=priv
#SBATCH --partition=priv_a100x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

echo "--- GPU Configuration of the Node ---"
module load nvidia/cuda/12.2
nvidia-smi

echo ""  
echo "################################################################"  
echo "@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@@"  
date  
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"  
echo "################################################################"