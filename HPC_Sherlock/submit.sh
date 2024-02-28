#!/bin/bash
#SBATCH --job-name=test
#SBTACH --partition=serc
#SBATCH --time=1-22:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

module use /home/groups/sh_s-dss/share/sdss/modules/modulefiles
module --ignore-cache load CMG

#module python/3.9

python ../pyCTRLfiles/pycontrol_1.py
