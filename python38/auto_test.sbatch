#!/bin/bash
# #SBATCH --job-name=bc_$3_$1_$2_$5
#SBATCH -N1                          # Ensure that all cores are on one machine
#SBATCH --partition=gpu             # Partition to submit to (serial_requeue)
#SBATCH --mem=8192               # Memory pool for all cores (see also --mem-per-cpu)
# #SBATCH --output=logs/br2_test_set_$3_$1_$2_$5_%j.out            # File to which STDOUT will be written
# #SBATCH --error=logs/br2_test_set_$3_$1_$2_$5_%j.err            # File to which STDERR will be written
#SBATCH --gres=gpu:1
####efefSBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ssahibul@cs.umass.edu

echo `pwd`
# echo "SLURM task ID: "$SLURM_ARRAY_TASK_ID
#module unload cudnn/4.0
#module unload cudnn/5.1
set -x -e
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
conda init bash
conda info --envs #get rid of
conda config --append envs_dirs /work/ssahibul_umass_edu/.conda/envs
conda info --envs #get rid of
conda activate broadcaster38
sleep 1

unset __conda_setup
# <<< conda initialize <<
conda init bash
#netTest2layer(slurmID, in_dataset, iterations, in_nets, convType)
python testing2layer.py br2_test_set_$3_$1_$2_$5_$SLURM_JOB_ID $1 $2 $3 $4 $5
sleep 1

hostname

sleep 1
exit
