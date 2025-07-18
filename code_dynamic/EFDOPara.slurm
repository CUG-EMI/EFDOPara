#!/bin/bash
#SBATCH -J EFDO
#SBATCH -p debug

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2400:2400:00

python -m torch.distributed.run --nproc_per_node=2 EFDO_main_parallel.py > EFDO_parallel.txt

