#!/bin/bash

# Submit this job by running sbatch job_example.sh

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling.
# Add an extra hash # symbol to comment out SBATCH settings.

#SBATCH --job-name=clip_score               # sets the job name
##SBATCH --output=slurm_outputs/%j.out
##SBATCH --output=sample.out             # indicates a file to redirect STDOUT to; %j is the jobid. Must be set to a file instead of a directory or else submission will fail.
##SBATCH --error=sample.out              # indicates a file to redirect STDERR to; %j is the jobid. Must be set to a file instead of a directory or else submission will fail.
##SBATCH -output=myfile.out
##SBATCH -error=myfile.err

#SBATCH --time=47:59:59                # how long you think your job will take to complete; format=dd-hh:mm:ss
##SBATCH --qos=high                     # set QOS, this will determine what resources can be requested. show-qos for more info
#SBATCH --mem=128gb                    # memory required by job; if unit is not specified MB will be assumed
#SBATCH --account=scavenger 
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodes=1
##SBATCH --ntasks=8
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=1

# GAMMA Commands
##SBATCH --time=48:00:00
##SBATCH --gres=gpu:1
##SBATCH --mem=64gb 
##SBATCH --account=gamma
##SBATCH --partition=gamma
##SBATCH --nodelist=gammagpu[03]        # You can specify the example GPU node if you want

# Full Python Path Here
# PYTHON_EXE="/fs/nexus-scratch/joesmith/miniconda3/envs/my_env/bin/python"
# module load anaconda/3
set -ex
source /fs/nexus-scratch/bbp13/inforatio/inforatio/info_env/bin/activate
#module load Python3/3.9.5
CUDA_VISIBLE_DEVICES=0,1,2,3
OMP_NUM_THREADS=8
#pip3 install matplotlib pandas 
#pip3 install scikit-learn gymnasium gym torch numpy

start=`date +%s`
#srun python3 run_episode.py -seed $1 -p $2 -v $2 -samples $3 --ac $4 -e $5 -i $6 -env $7 -grid $8 -mp $9 -gr ${10} -df ${11}

srun python3 clip_score_calculate.py
end=`date +%s`
runtime=$((end-start))

# ${PYTHON_EXE} my_script.py

