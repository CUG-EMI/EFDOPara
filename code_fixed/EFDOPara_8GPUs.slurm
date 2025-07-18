#!/bin/bash  
#SBATCH -J EFDO_P8GPU  
#SBATCH --account=luosheng_w  
#SBATCH --cluster=priv  
#SBATCH --partition=priv_a100x8  
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1  
#SBATCH --gres=gpu:8
#SBATCH --time=2400:00:00   
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

# 
srun singularity exec --nv -B /project/luosheng_w/Test/EFDOPara-main:/workspace \
     /project/luosheng_w/singularity_sif/pytorchv1_2.3.0.sif \
     bash -c "cd /workspace/code_prepare && python -m torch.distributed.run \
     --nproc_per_node=8 \
     EFDO_main_parallel.py --item EFDO_config_20250505_MPI_8GPU" \
     > EFDO_Para_8GPU_20250613.txt

echo ""  
echo "################################################################"  
echo "@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@@"  
date  
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"  
echo "################################################################"