#!/bin/bash
#!/bin/bash  
#SBATCH -J EFDO  
#SBATCH --account=luosheng_w
#SBATCH --cluster=priv
#SBATCH --partition=priv_a100x4 

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2400:2400:00

python EFDO_main.py EFDO_config > EFDO_running_info.txt

