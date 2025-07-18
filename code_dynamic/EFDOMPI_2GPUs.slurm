#!/bin/bash  
#SBATCH -J EFDO_M2GPU  
#SBATCH --account=luosheng_w  
#SBATCH --cluster=priv  
#SBATCH --partition=priv_a100x4  
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1  
#SBATCH --gres=gpu:2 
##SBATCH --nodelist=g0152
#SBATCH --time=2400:00:00    
#SBATCH --output=%x_%j.out  
#SBATCH --error=%x_%j.err   

##### Number of total processes  
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "  
echo "Nodelist:= " $SLURM_JOB_NODELIST  
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES  
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE  
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "  

# module load singularity  
# module load nvidia/cuda/12.2 
# module load openmpi/5.0_cuda12.2_gcc11.2 
# 激活Conda环境  
source /project/luosheng_w/mambaforge/etc/profile.d/conda.sh  
conda activate torch2 

# 设置完整路径  
SCRIPT_PATH="/project/luosheng_w/Test/EFDOPara-main/code_prepare/EFDO_main_parallel_MPI.py"  
chmod +x "$SCRIPT_PATH"

# 设置PyTorch分布式训练的环境变量
export MASTER_ADDR="localhost"
export MASTER_PORT="${MASTER_PORT}"

# 生成随机端口号(10000-65535范围)
MASTER_PORT=$(shuf -i 10000-65535 -n 1)
echo "Using random port: $MASTER_PORT"

echo "Run started at:- "  
date  

# # 修改2: 使用torchrun替代torch.distributed.launch
mpiexec -np 2 python "$SCRIPT_PATH" --item EFDO_config_20250505_MPI_2GPU \
    > EFDO_MPI_2GPU_202505010.txt

echo ""  
echo "################################################################"  
echo "@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@@"  
date  
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"  
echo "################################################################"