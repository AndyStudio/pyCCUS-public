#!/bin/bash
#SBATCH --job-name=test
#SBTACH --partition=serc
#SBATCH --time=1-2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

module use /home/groups/sh_s-dss/share/sdss/modules/modulefiles
module --ignore-cache load CMG

python pycontrol.py
