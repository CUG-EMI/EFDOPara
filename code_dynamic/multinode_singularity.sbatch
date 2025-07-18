#!/bin/bash

#SBATCH --job-name=COOL_JOB_NAME    # create a short name for your job
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2      # total number of tasks per node
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
##SBATCH --mem=240GB               # total memory per node
##SBATCH --time=23:59:00          # total run time limit (HH:MM:SS)




##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

# If you want to load things from your .bashrc profile, e.g. cuda drivers, singularity etc
#source ~/.bashrc

module load singularity
module load nvidia/cuda/12.2

# ******************* These are read internally it seems ***********************************
# ******** Master port, address and world size MUST be passed as variables for DDP to work 
# Add these before srun command  
export LOCAL_RANK=$SLURM_LOCALID  
export RANK=$SLURM_PROCID

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
# ******************************************************************************************

# zoom zoom - recommended from lightning
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32


echo "Run started at:- "
date

# Actual run of script
#srun python main.py # Use this if you have python in your environment
#srun singularity exec --nv /Location/Of/Your/Containers/pytorch_22.05-py3.sif python main.py
srun singularity exec --nv -B /project/luosheng_w/Test/EFDOPara-main:/workspace \
     /project/luosheng_w/software/pytorch2_torchinfo.sif \
     python \ 
     --node_rank=$SLURM_NODEID \
     --master_addr=$MASTER_ADDR \
     --master_port=$MASTER_PORT \
     /workspace/code/EFDO_main_parallel.py --item EFDO_config_20250418_8GPU
echo ""
echo "################################################################"
echo "@@@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@"
date
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "################################################################"

