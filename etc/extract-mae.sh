#!/usr/bin/env bash
#
# extract-mae.sh - MRSNet - helper script to quick extract MAE errors from json files
# (do not rely on this; just for quick checks)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2022 Frank C Langbein <frank@langbein.org>, Cardiff University

for l in `ls *.json | sort`; do
  mean="`cat $l | sed -n '/"total":/,$p' | grep mean | head -1 | cut -d: -f2- | cut -d, -f1 | sed -e's, ,,g'`"
  std="`cat $l | sed -n '/"total":/,$p' | grep std | head -1 | cut -d: -f2- | cut -d, -f1 | sed -e's, ,,g'`"
  #echo `echo $l | cut -d_ -f1-2` $mean@$std
  echo -n "$mean@$std | "
done
echo
