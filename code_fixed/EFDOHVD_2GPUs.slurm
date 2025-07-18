#!/bin/bash  
#SBATCH -J EFDO_H2GPU  
#SBATCH --account=luosheng_w  
#SBATCH --cluster=priv  
#SBATCH --partition=priv_a100x4  
#SBATCH --nodes=1  
#SBATCH --ntasks=1       
#SBATCH --gres=gpu:2 
#SBATCH --time=24:00:00    
#SBATCH --output=%x_%j.out  
#SBATCH --error=%x_%j.err   

##### Number of total processes  
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "  
echo "Nodelist:= " $SLURM_JOB_NODELIST  
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES  
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE  
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "  

module load singularity  
module load nvidia/cuda/12.2  

echo "Run started at:- "  
date  

# HVD-singularity-2GPU
singularity exec --nv \
     -B /project/luosheng_w/Test/EFDOPara-main:/workspace \
     /project/luosheng_w/singularity_sif/horovodv2.sif \
     bash -c "cd /workspace/code_prepare && horovodrun --gloo -np 2 -H localhost:2 python \
     EFDO_main_parallel_HVD.py --item EFDO_config_20250505_HVD_2GPU" \
     > EFDO_HVD_2GPU_202505010.txt

echo ""  
echo "################################################################"  
echo "@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@@"  
date  
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"  
echo "################################################################"