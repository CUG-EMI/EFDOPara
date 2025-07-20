#!/bin/bash  
#SBATCH -J EFDO_P2G_F 
#SBATCH --account=luosheng_w  
#SBATCH --cluster=priv  
#SBATCH --partition=priv_a100x4  
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1  
#SBATCH --gres=gpu:2 
#SBATCH --time=2400:00:00  
#SBATCH --output=%x_%j.out  
#SBATCH --error=%x_%j.err   

##### Number of total processes  
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "  
echo "Nodelist:= " $SLURM_JOB_NODELIST  
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES  
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE  
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "  

# Load necessary modules
module load nvidia/cuda/12.2
module load openmpi/5.0_cuda12.2_gcc11.2  


# activate conda environment  
source /project/luosheng_w/mambaforge/etc/profile.d/conda.sh  
conda activate torch2 

# set the script path 
SCRIPT_PATH="/project/luosheng_w/Test/EFDOPara-main/code_floatbs/EFDO_main_parallel.py"  
chmod +x "$SCRIPT_PATH"

echo "Run started at:- "  
date  

# use a random port for distributed training
RANDOM_PORT=$((30000 + RANDOM % 10000))
echo "Using port: $RANDOM_PORT for distributed training"

# set environment variables for distributed training
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export MASTER_PORT=$RANDOM_PORT
export OMP_NUM_THREADS=8

cd /project/luosheng_w/Test/EFDOPara-main/code_floatbs

python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=$RANDOM_PORT \
    "$SCRIPT_PATH" --item EFDO_config_20250505_MPI_2GPU_BS25 \
    > EFDO_Para_2GPU_BS25_20250528.txt

echo ""  
echo "################################################################"  
echo "@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@@"  
date  
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"  
echo "################################################################"