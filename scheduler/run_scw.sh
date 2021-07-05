#!/usr/bin/env bash
#
# run_scw.sh - MRSNet - scheduled jobs on SCW
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2021, Frank C Langbein <frank@langbein.org>, Cardiff University
#

default_host="hawklogin.cf.ac.uk"
default_jobs=10
default_repeats=1
default_wait=1h

if [ ! -r helper/slurm-scw.sh ]; then
  echo "$0: must be run in code-mrsnet folder" >&2
  exit 1
fi

user=""
test=""
r=0
max_jobs=$default_jobs
host=$default_host
repeats=$default_repeats
wait=$default_wait

while [ -n "$1" ]; do
  if [ "$1" = "-c" -o "$1" = "--continuous" ]; then
    r=1
    shift
  elif [ "$1" = "--sleep" ]; then
    shift
    wait=$1
    shift
  elif [ "$1" = "--host" ]; then
    shift
    host=$1
    shift
  elif [ "$1" = "-j" -o "$1" = "--jobs" ]; then
    shift
    case $1 in
      ''|*[!0-9]*) echo "$0: argument of -j|--jobs must be a number"; exit 1 ;;
    esac
    max_jobs=$1
    shift
  elif [ "$1" = "-r" -o "$1" = "--repeats" ]; then
    shift
    case $1 in
      ''|*[!0-9]*) echo "$0: argument of -r|--repeats must be a number"; exit 1 ;;
    esac
    repeats=$1
    shift
  elif [ "$1" = "-h" -o "$1" = "--help" ]; then
    cat <<EOD
$0 [OPTIONS] USER TEST-DATASET...
Schedule test dataset trainig on SCW.

  USER              SCW user id
  TEST-DATASET      run-test.py test dataset ids

  OPTIONS:
    -r, --repeats     number of repeats for dataset [$default_repeats]
    -j, --jobs        maximum jobs to schedule in parallel [$default_jobs]
    -c, --continuous  schedule continously (CTRL-C to quit)
        --sleep       sleep time between schedule attempts [$default_wait]
        --host        SCW login host [$default_host]
    -h, --help        this help text
EOD
    exit 0
  else
    if [ -z "$user" ]; then
      user="$1"
    else
      test="$test $1"
    fi
    shift
  fi
done

if [ -z "$user" ]; then
  echo "$0: no user specified" >&2
  exit 1
fi
if [ -z "$test" ]; then
  echo "$0: no dataset specified" >&2
  exit 1
fi

while true; do

  date

  n="`ssh ${user}@${host} squeue | grep $user | wc -l`"
  d=`expr $max_jobs - $n`

  if [ $d -gt 0 ]; then
    rsync -av --exclude='data/model' --exclude='data/benchmark' --exclude='data/jobs*' --exclude='.git' --exclude='__pycache__' . ${user}@${host}:code-mrsnet
    all_done=1
    for t in $test; do
      ssh ${user}@${host} 'cd ~/code-mrsnet && ./run_test.py '${t}' '${repeats}' -e ./helper/slurm-scw.sh -m '${d}' --no_benchmark'
      if [ "$?" != 0 ]; then
        all_done=0
      fi
    done
    if [ "$all_done" == 1 ]; then
      echo "All tests scheduled"
      exit 0
    fi
  fi

  if [ "$r" = 0 ]; then
    exit 0
  fi

  echo "  Waiting $wait..."
  sleep $wait

done
