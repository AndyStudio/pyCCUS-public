#!/bin/bash 
module use /home/groups/sh_s-dss/share/sdss/modules/modulefiles
module --ignore-cache load CMG

# module use /home/groups/s-ees/share/cees/modules/modulefiles 
# module load CMG-cees/ 

for i in {1..10}; do
        script_path="pycontrol_exe.py -runname case${i}"
        job_name="job_${i}"
        output_folder="../slurm"
        output_file="${output_folder}/${job_name}.out"
        error_file="${output_folder}/${job_name}.err"
        sbatch --job-name="wopt${i}" --partition=serc --time=1-22:00:00 --ntasks=1 --cpus-per-task=8 --output="${output_file}" --wrap="python ${script_path}"
done
