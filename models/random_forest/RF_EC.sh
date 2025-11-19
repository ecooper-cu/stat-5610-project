#!/bin/bash

#SBATCH --account=ucb678_asc1
#SBATCH --partition=amilan
#SBATCH --job-name=RF
#SBATCH --output=out/stats/%j.out
#SBATCH --time=8:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emco4286@colorado.edu

module purge
module load python/3.10.2

source /projects/emco4286/environments/stats/bin/activate
python /home/emco4286/stat-5610-project/models/random_forest/RF_EC.py