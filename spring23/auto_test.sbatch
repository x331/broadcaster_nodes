#!/bin/bash
# #SBATCH --job-name=bc_${5}_${1}_${3}_${2}_${6}
#SBATCH -N1                          # Ensure that all cores are on one machine
#SBATCH --partition=gpu              # Partition to submit to (serial_requeue)
#SBATCH --exclude gpu[001,002,003,004]

#SBATCH --mem=16384              # Memory pool for all cores (see also --mem-per-cpu) 16384 
# #SBATCH --output=logs/br2_test_set_${5}_${1}_testset${3}_iter${2}_${7}_%j.out            # File to which STDOUT will be written
# #SBATCH --error=logs/br2_test_set_${5}_${1}_testset${3}_iter${2}_${7}_%j.err            # File to which STDERR will be written
#SBATCH --gres=gpu:1
####efefSBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ssahibul@umass.edu

echo `pwd`
# echo "SLURM task ID: "$SLURM_ARRAY_TASK_ID
#module unload cudnn/4.0
#module unload cudnn/5.1
#module unload nvidia_drm
#module unload nvidia_modeset
#module unload nvidia_uvm

##### Experiment settings #####
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/modules/apps/miniconda/4.8.3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
	eval "$__conda_setup"
else
	if [ -f "/modules/apps/miniconda/4.8.3/etc/profile.d/conda.sh" ]; then
		. "/modules/apps/miniconda/4.8.3/etc/profile.d/conda.sh"
	else
		export PATH="/modules/apps/miniconda/4.8.3/bin:$PATH"
	fi  
fi


unset __conda_setup
# <<< conda initialize <<
# conda init bash
conda config --append envs_dirs /work/ssahibul_umass_edu/.conda/envs
conda activate bc38
sleep 1


#(slurmID, in_dataset, iterations, in_nets, convType,kmeans,clus_wait,lr, ldecay)
python testing2layer.py ${9}br2_${4}_${1}_test${3}_iter${2}_clus${5}_wait${6}_lr${7}_lrdecay${8}_$SLURM_JOB_ID $1 $2 $3 $4 $5 $6 $7 $8

sleep 1

hostname

sleep 1
exit
