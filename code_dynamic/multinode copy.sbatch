#!/bin/bash  
#SBATCH -J EFDO_4GPU  
#SBATCH --account=luosheng_w  
#SBATCH --cluster=priv  
#SBATCH --partition=priv_a100x4  
#SBATCH --nodes=2  
#SBATCH --ntasks-per-node=1  
#SBATCH --gres=gpu:2  
#SBATCH --output=%x_%j.out  
#SBATCH --error=%x_%j.err  

source /project/luosheng_w/mambaforge/etc/profile.d/conda.sh  
conda activate torch2  

NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)  
echo "About to launch distributed"  
echo "Allocated nodes: $NODELIST"  
NODE_ARR=($NODELIST)  
MASTER_ADDR=${NODE_ARR[0]}  
MASTER_PORT=$((12000 + RANDOM % 20000))  

# Use SLURM_NNODES instead of SLURM_JOB_NUM_NODES  
NNODES=${SLURM_NNODES}  
echo "MASTER_ADDR $MASTER_ADDR, MASTER_PORT $MASTER_PORT, NNODES $NNODES"  

srun --export=ALL python -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes=$NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    EFDO_main_parallel1.py --item EFDO_config_20250418_4GPU  

echo "Launched distributed"