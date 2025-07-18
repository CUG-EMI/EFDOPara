#!/bin/bash  
#SBATCH -J EFDO_M1GPU  
#SBATCH --account=luosheng_w  
#SBATCH --cluster=priv  
#SBATCH --partition=priv_a100x4  
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1  
#SBATCH --gres=gpu:1 
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
# activate Conda environment  
source /project/luosheng_w/mambaforge/etc/profile.d/conda.sh  
conda activate torch2 

# set the path to your script  
SCRIPT_PATH="/project/luosheng_w/Test/EFDOPara-main/code_prepare/EFDO_main_parallel_MPI.py"  
chmod +x "$SCRIPT_PATH"

echo "Run started at:- "  
date  

# 
mpiexec -np 1 python "$SCRIPT_PATH" --item EFDO_config_20250505_MPI_1GPU \
    > EFDO_MPI_1GPU_202505014.txt

echo ""  
echo "################################################################"  
echo "@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@@"  
date  
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"  
echo "################################################################"