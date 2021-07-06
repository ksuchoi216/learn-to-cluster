#!/bin/bash
#
#SBATCH --job-name=face_clustering # Job name for tracking
#SBATCH --partition=gpu-batch  # Partition you wish to use (see above for list)
#SBATCH --cpus-per-task=1      # Number of CPU cores used by your job
##SBATCH --mem=1000             # RAM in MB needed in total (also available --mem-per-cpu)
#SBATCH --time=00:30:00        # Job limited to 12 hours (see sinfo for maximums)

module purge
module load cuda

srun sh './scripts/gae/train_gae_cut.sh'