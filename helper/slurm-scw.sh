#!/usr/bin/env bash
#
# slurm_scw - MRSNet - execute as slurm job on SCW
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2021, Frank C Langbein <frank@langbein.org>, Cardiff University
#

name="MRSNet-`date '+%Y%m%d%H%M%S-%N'`"

cmd="$*"

mkdir -p data/jobs-scw

cat >jobs/${name}.sh <<EOF
#!/bin/bash --login
###
#SBATCH --job-name=${name}
#SBATCH --output=data/jobs-scw/${name}.out
#SBATCH --error=data/jobs-scw/${name}.err
#SBATCH --time=1-00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32768
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1
###

module load python/3.9.2
module load CUDA/11.2

export PATH=~/.local/bin:$PATH
export PYTHONPATH=~/.local/lib/python3.9/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=~/.local/lib64:$LD_LIBRARY_PATH

cd ~/code-mrsnet

echo "Running on $SLURM_JOB_NODELIST"
/usr/bin/env python3 ${cmd}
EOF

sbatch -A `grep '^MRSNet ' ~/projects  | cut -d\  -f2` data/jobs-scw/${name}.sh
