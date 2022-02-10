#!/usr/bin/env bash
#
# run_scw.sh - MRSNet - run jobs on SCW
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2021 Frank C Langbein <frank@langbein.org>, Cardiff University

default_host="hawklogin.cf.ac.uk"

cmd=""
user=""
host=$default_host

while [ -n "$1" ]; do
  if [ "$1" = "--host" ]; then
    shift
    host=$1
    shift
  elif [ "$1" = "-h" -o "$1" = "--help" ]; then
    cat <<EOD
$0 [OPTIONS] USER [sync | run ARGS| check ARGS]
Schedule MRSNet job on SCW.

  COMMAND:
    sync            synchronise MRSNet code
    run             start MRSNet job
    check           check if MRSNet job is done

  USER              SCW user id
  ARGS              MRSNet job arguments:
                      DATASET        - path
                      METABOLITES    - - separated strings
                      PULSE_SEQUENCE - string
                      EPOCHS         - number
                      VALIDATE       - number
                      NORM           - string
                      ACQUISITIONS   - - separated strings
                      DATATYPE       - - separated strings
                      MODEL          - string
                      BATCH_SIZE     - number

  OPTIONS:
    -h, --help        this help text
    --host        SCW login host [$default_host]
EOD
    exit 0
  else
    if [ -z "$user" ]; then
      user="$1"
    elif [ -z "$cmd" ]; then
      cmd="$1"
      if [ "$cmd" = run -o "$cmd" = check ]; then
        for var in DATASET METABOLITES PULSE_SEQUENCE EPOCHS VALIDATE NORM ACQUISITIONS DATATYPE MODEL BATCH_SIZE; do
          shift
          if [ -z "$1" ]; then
            echo "$0: no $var value" >&2
            exit 1
          fi
          eval $var=\"$1\"
        done
      elif [ "$cmd" != "sync" ]; then
        echo "$0: unknown command $cmd" >&2
        exit 1
      fi
    else
      echo "$0: extra argument: $1" >&2
      exit 1
    fi
    shift
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
  rsync -a --progress --exclude='.git' --exclude='__pycache__' --exclude 'data' . ${user}@${host}:code-mrsnet
  ssh ${user}@${host} 'find code-mrsnet/data/{jobs-scw,model,sim-spectra} -type d -empty -delete'
elif [ "$cmd" == run ]; then
  DATASET="`echo $DATASET | sed -e 's,^.*/sim-spectra/,,'`"
  echo "# Sync'ing dataset $DATASET"
  ssh ${user}@${host} 'mkdir -p 'code-mrsnet/data/sim-spectra/$DATASET
  rsync -a --progress data/sim-spectra/$DATASET/*.joblib ${user}@${host}:code-mrsnet/data/sim-spectra/$DATASET

  echo "# Schedule job"
  DATASET_ID="`echo $DATASET | sed -e s,/,_,g`"
  id="$DATASET_ID/$METABOLITES/$PULSE_SEQUENCE/$EPOCHS/$VALIDATE/$NORM/$ACQUISITIONS/$DATATYPE/$MODEL/$BATCH_SIZE"
  folder="code-mrsnet/data/jobs-scw/$id"

  METABOLITES="`echo $METABOLITES | sed -e 's,-, ,g'`"
  DATATYPE="`echo $DATATYPE | sed -e 's,-, ,g'`"
  ACQUISITIONS="`echo $ACQUISITIONS | sed -e 's,-, ,g'`"

  cat <<EOF | ssh ${user}@${host} 'mkdir -p "'$folder'" && cat - >"'$folder'/job.sh"'
#!/bin/bash --login
#SBATCH --job-name=${id}
#SBATCH --output=${folder}/out
#SBATCH --error=${folder}/err
#SBATCH -p gpu_v100,gpu
#SBARCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32768
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00

module load python/3.9.2
module load CUDA/11.5

export PATH=~/.local/bin:\$PATH
export PYTHONPATH=~/.local/lib/python3.9/site-packages:\$PYTHONPATH
export LD_LIBRARY_PATH=~/.local/cuda/lib:~/.local/lib:\$LD_LIBRARY_PATH

cd ~/code-mrsnet

echo "Job ${id} :"
echo "  \$SLURM_JOB_NAME/\$SLURM_JOB_ID on \$SLURM_JOB_NODELIST"
/usr/bin/env python3 ./mrsnet.py train --no-show --verbose \\
  --metabolites ${METABOLITES} \\
  --dataset data/sim-spectra/${DATASET} \\
  --epochs ${EPOCHS} \\
  --validate ${VALIDATE} \\
  --norm ${NORM} \\
  --acquisitions ${ACQUISITIONS} \\
  --datatype ${DATATYPE} \\
  --model ${MODEL} \\
  --batch-size ${BATCH_SIZE}
test "\$?" = 0 && date >~/${folder}/done
EOF

  ssh ${user}@${host} 'sbatch -A `grep '^MRSNet ' ~/projects  | cut -d\  -f2` '$folder/job.sh

else
  DATASET="`echo $DATASET | sed -e 's,^.*/sim-spectra/,,'`"
  echo "# Check if done"
  DATASET_ID="`echo $DATASET | sed -e s,/,_,g`"
  id="$DATASET_ID/$METABOLITES/$PULSE_SEQUENCE/$EPOCHS/$VALIDATE/$NORM/$ACQUISITIONS/$DATATYPE/$MODEL/$BATCH_SIZE"
  folder="code-mrsnet/data/jobs-scw/$id"
  if [ "$VALIDATE" = 0.0 -o "$VALIDATE" = 0 ]; then
    VALIDATOR="NoValidation"
  elif [ "`echo $VALIDATE | cut -c1`" = - ]; then
    if [ "`echo $VALIDATE | cut -c2`" = 0 ]; then
      VALIDATOR="DuplexSplit_${VALIDATE}"
    else
      VALIDATOR="DuplexKFold_`echo ${VALIDATE} | cut -d. -f1`"
    fi
  else
    if [ "`echo $VALIDATE | cut -c1`" = 0 ]; then
      VALIDATOR="Split_${VALIDATE}"
    else
      VALIDATOR="KFold_`echo ${VALIDATE} | cut -d. -f1`"
    fi
  fi
  # Remotely we presumably only have one run, so only -1
  model_folder="code-mrsnet/data/model/$MODEL/$METABOLITES/$PULSE_SEQUENCE/$ACQUISITIONS/$DATATYPE/$NORM/$BATCH_SIZE/$EPOCHS/$DATASET_ID/$VALIDATOR-1"
  # We'd only call remotely if there is no local data, so also only -1
  local_model_folder="data/model/$MODEL/$METABOLITES/$PULSE_SEQUENCE/$ACQUISITIONS/$DATATYPE/$NORM/$BATCH_SIZE/$EPOCHS/$DATASET_ID/$VALIDATOR-1"
  ssh ${user}@${host} 'ls '$folder 2>/dev/null 1>&2
  if [ "$?" != 0 ]; then
    echo "OFFLINE"
  else
    ssh ${user}@${host} 'ls '$folder/done 2>/dev/null 1>&2
    if [ "$?" != 0 ]; then
      n="`ssh ${user}@${host} squeue -n $id 2>/dev/null | sed -n '1!p'`"
      echo $n
      if [ -z "$n" ]; then
        # Delete job folder and model
        ssh ${user}@${host} rm -rf $folder $model_folder
        echo "FAILED"
      else
        ssh ${user}@${host} tail -2 $folder/out
        echo "WAIT"
      fi
    else
      # Copy results
      echo "Done: $model_folder -> $local_model_folder"
      mkdir -p $local_model_folder
      rsync -a --progress ${user}@${host}:$model_folder/ $local_model_folder/
      # Delete job folder and model
      ssh ${user}@${host} rm -rf $folder $model_folder
      echo "DONE"
    fi
  fi
fi
