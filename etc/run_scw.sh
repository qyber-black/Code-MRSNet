#!/usr/bin/env bash
#
# run_scw.sh - MRSNet - run jobs on SCW
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2021-2022 Frank C Langbein <frank@langbein.org>, Cardiff University

default_host="hawklogin.cf.ac.uk"

cmd=""
user=""
host=$default_host

if [ ! -x ./mrsnet.py ]; then
  echo "$0: must be run from root of mrsnet source directory" >&2
  exit 1
fi

while [ -n "$1" ]; do
  if [ "$1" = "--host" ]; then
    shift
    host=$1
    shift
  elif [ "$1" = "-h" -o "$1" = "--help" ]; then
    cat <<EOD
$0 [-h|--help] [--host HOST] USER [sync DATASET_PATH |
                                   run MODEL_PATH ARGS |
                                   check MODEL_PATH]

Schedule MRSNet job on SCW.

  COMMAND:
    sync            synchronise MRSNet code and given dataset
    run             start MRSNet job
    check           check if MRSNet job is done

  USER              SCW user id
  HOST              SCW scheduling host
  DATASET_PATH      Dataset path, relative to current directory
  MODEL_PATH        Model path, relative to current directory
  ARGS              MRSNet arguments

  OPTIONS:
    -h, --help      This help text
    --host          SCW login host [$default_host]
EOD
    exit 0
  else
    if [ -z "$user" ]; then
      user="$1"
      shift
    elif [ -z "$cmd" ]; then
      cmd="$1"
      shift
      if [ "$cmd" = run -o "$cmd" = check ]; then
        MODEL_PATH="$1"
        shift
        ARGS="$*"
        break
      elif [ "$cmd" = "sync" ]; then
        DATASET_PATH="$1"
        break
      else
        echo "$0: unknown command $cmd" >&2
        exit 1
      fi
    else
      echo "$0: extra argument: $1" >&2
      exit 1
    fi
  fi
done

if [ -z "$user" ]; then
  echo "$0: no user specified" >&2
  exit 1
fi
if [ -z "$cmd" ]; then
  echo "$0: no command specified" >&2
  exit 1
fi

if [ "$cmd" == sync ]; then

  echo "# Sync'ing code from `pwd`"
  rsync -a --delete --append-verify --progress --exclude='.git' --exclude='__pycache__' --exclude 'data' --exclude 'mrsnet/simulators' . ${user}@${host}:code-mrsnet
  ssh ${user}@${host} 'find code-mrsnet/data/jobs-scw -type d -empty -delete 2>/dev/null 1>&2'

  echo "# Sync'ing dataset $DATASET_PATH"
  ssh ${user}@${host} 'mkdir -p 'code-mrsnet/$DATASET_PATH
  rsync -a --delete --append-verify --progress $DATASET_PATH/ ${user}@${host}:code-mrsnet/$DATASET_PATH

elif [ "$cmd" == run ]; then

  echo "# Schedule job for $MODEL_PATH"
  folder="code-mrsnet/data/jobs-scw/$MODEL_PATH"

  cat <<EOF | ssh ${user}@${host} 'mkdir -p "'$folder'" && cat - >"'$folder'/job.sh"'
#!/bin/bash --login
#SBATCH --job-name=${MODEL_PATH}
#SBATCH --output=${folder}/out.log
#SBATCH --error=${folder}/err.log
#SBATCH -p gpu_v100,gpu
#SBARCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00

module purge
module load system/auto
module load hpcw
module load python/3.9.2
module load CUDA/11.5

export LD_LIBRARY_PATH=~/.local/cuda/lib:~/.local/lib:\$LD_LIBRARY_PATH

cd ~/code-mrsnet

echo "Job ${MODEL_PATH}:"
echo "  \$SLURM_JOB_NAME/\$SLURM_JOB_ID on \$SLURM_JOB_NODELIST"

/usr/bin/env python3 ./mrsnet.py train --verbose $ARGS

test "\$?" = 0 && date >~/${folder}/done
EOF

  ssh ${user}@${host} 'sbatch -A `grep '^MRSNet ' ~/projects  | cut -d\  -f2` '$folder/job.sh

else

  echo "# Check $MODEL_PATH"
  folder="code-mrsnet/data/jobs-scw/$MODEL_PATH"

  ssh ${user}@${host} 'ls '$folder 2>/dev/null 1>&2
  if [ "$?" != 0 ]; then
    echo "OFFLINE"
  else
    ssh ${user}@${host} 'ls '$folder/done 2>/dev/null 1>&2
    if [ "$?" != 0 ]; then
      n="`ssh ${user}@${host} squeue -n $MODEL_PATH 2>/dev/null | sed -n '1!p'`"
      if [ -z "$n" ]; then
        # Delete job folder and model
        ssh ${user}@${host} rm -rf $folder code-mrsnet/$MODEL_PATH
        echo "FAILED"
      else
        ssh ${user}@${host} tail -2 $folder/out.log
        echo "WAIT"
      fi
    else
      # Copy results
      echo "Done: $MODEL_PATH"
      mkdir -p $MODEL_PATH
      src="`echo $MODEL_PATH | sed -e 's,^.*/data/model/,,'`"
      rsync -a --delete --append-verify --progress ${user}@${host}:code-mrsnet/data/model/$src/ $MODEL_PATH
      # Delete job folder and model
      ssh ${user}@${host} rm -rf $folder code-mrsnet/$MODEL_PATH
      echo "DONE"
    fi
  fi

fi
