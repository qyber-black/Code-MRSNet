#!/bin/bash --login
#
# scw-template.sh - template for SCW jobs
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2022 Frank C Langbein <frank@langbein.org>, Cardiff University
#
#SBATCH --job-name=@NAME@
#SBATCH --output=@NAME@.log
#SBATCH --error=@NAME@.err
#SBATCH -p gpu_v100
#SBARCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00

module purge
module load system/auto
module load hpcw
module load python/3.9.2
module load CUDA/11.5

export LD_LIBRARY_PATH=~/.local/cuda/lib:~/.local/lib:$LD_LIBRARY_PATH

cd ~/code-mrsnet

echo "Job @NAME@:"
echo "  $SLURM_JOB_NAME/$SLURM_JOB_ID on $SLURM_JOB_NODELIST"

/usr/bin/env python3 ./mrsnet.py @ARGS@

