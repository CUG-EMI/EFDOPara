#!/bin/bash  
#SBATCH -J EFDO_4GPU  
#SBATCH --account=luosheng_w  
#SBATCH --cluster=priv  
#SBATCH --partition=priv_a100x4  
#SBATCH --nodes=4  
#SBATCH --ntasks-per-node=1  
#SBATCH --gres=gpu:1  
#SBATCH --output=%x_%j.out  
#SBATCH --error=%x_%j.err  



export MASTER_PORT=12340
export WORLD_SIZE=4

NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)  
echo "About to launch distributed"  
echo "Allocated nodes: $NODELIST"  
NODE_ARR=($NODELIST)  
MASTER_ADDR=${NODE_ARR[0]}  
MASTER_PORT=$((12000 + RANDOM % 20000))  

# Export MASTER_PORT  
export MASTER_PORT=$MASTER_PORT  
export SLURM_JOB_NODELIST=$MASTER_ADDR  

source /project/luosheng_w/mambaforge/etc/profile.d/conda.sh  
conda activate torch2  

NNODES=${SLURM_NNODES}  
echo "MASTER_ADDR $MASTER_ADDR, MASTER_PORT $MASTER_PORT, NNODES $NNODES"  

srun python EFDO_main_parallel1.py > EFDO_config_20250418_4GPU_running_info_multi.txt

echo "Launched distributed"