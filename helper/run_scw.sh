#!/usr/bin/env bash
#
# run_scw.sh - MRSNet - scheduled jobs on SCW
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2021, Frank C Langbein <frank@langbein.org>, Cardiff University
#

user="$1"
if [ -z "$user" ]; then
  echo "Error: first argument should be user" >&2
  exit 1
fi

test="$2"
if [ -z "$test" ]; then
  echo "Error: second argument should be test dataset to run" >&2
  exit 1
fi

if [ -z "`ssh ${user}@hawklogin.cf.ac.uk squeue | grep $user`" ]; then
  rsync -av --exclude='data/model/*' --exclude='data/benchmark/*' --exclude='*/__pycache__/*' --exclude='.git/*'. ${user}@hawklogin.cf.ac.uk:code-mrsnet
  ssh ${user}@hawklogin.cf.ac.uk 'cd ~/code-mrsnet && ./run_test.py '${test}' 1 -e ./helper/slurm-scw.sh -m 10 --no_benchmark'
fi
