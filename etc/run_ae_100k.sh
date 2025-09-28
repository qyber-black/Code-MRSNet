#!/usr/bin/env bash
#
# Train the 100k autoencoders required by AEQ model selection.
#
# SPDX-FileCopyrightText: Copyright (C) 2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Usage:
#   bash etc/run_ae_100k.sh

set -euo pipefail

ROOT="/srv/data/Prj/MRS/code-mrsnet"

# Dataset and common settings
METABS="Cr-GABA-Gln-Glu-NAA"
PULSE="megapress"
DATASET="$ROOT/data/sim-spectra-megapress/fid-a-2d_2000_4096/siemens/123.23/2.0/$METABS/$PULSE/sobol/1.0-adc_normal-0.0-0.03/100000-1"
EPOCHS=200
VALID=5              # KFold 5 for robust encoders
BATCH=16
NORM=sum

# Check if a model (for a specific context) already exists in common model roots
exists_model_ctx() {
  local MODEL="$1"     # ae_fc_...
  local ACQ_DIR="$2"   # e.g., difference-edit_on
  local DT_DIR="$3"    # e.g., real or imaginary-real
  local DS="`echo $DATASET | cut -d / -f 6- | sed 's|/|_|g'`"
  local TRAINER="KFold_${VALID}-1"
  for BASE in "$ROOT/data/model-ae" "$ROOT/data/model" "$ROOT/data/model-dist"; do
    local TARGET="$BASE/$MODEL/$METABS/$PULSE/$ACQ_DIR/$DT_DIR/$NORM/$BATCH/$EPOCHS/$DS/$TRAINER"
    if [ -d "$TARGET" ]; then
      echo "[FOUND] $TARGET"
      return 0
    fi
  done
  return 1
}

# Train one AE if not already present
# Args: MODEL ACQ_DIR DT_DIR
run() {
  local MODEL="$1"; shift
  local ACQ_DIR="$1"; shift
  local DT_DIR="$1"; shift
  local LABEL="$MODEL -- $ACQ_DIR -- $DT_DIR"

  echo "[CHECK] $LABEL"
  if exists_model_ctx "$MODEL" "$ACQ_DIR" "$DT_DIR"; then
    echo "[SKIP] $LABEL (exists)"
    return 0
  fi

  IFS='-' read -r -a ACQS <<< "$ACQ_DIR"
  IFS='-' read -r -a DTS <<< "$DT_DIR"

  echo "[TRAIN] $LABEL"
  "$ROOT/mrsnet.py" train \
    -d "$DATASET" \
    -e "$EPOCHS" -k "$VALID" \
    --norm "$NORM" \
    --acquisitions "${ACQS[@]}" \
    --datatype "${DTS[@]}" \
    -m "$MODEL" \
    -b "$BATCH" -v
}

# ==============================
# Required encoders per context
# ==============================

# Real
run ae_fc_5_7_tanh_tanh_0.3 difference-edit_on real
run ae_fc_5_7_relu_tanh_0.2 difference-edit_on real

run ae_fc_5_5_tanh_tanh_0.3 edit_off-edit_on real
run ae_fc_5_8_relu_linear_0.2 edit_off-edit_on real

run ae_fc_5_7_tanh_tanh_0.3 difference-edit_off real
run ae_fc_6_8_relu_linear_0.3 difference-edit_off real

run ae_fc_5_7_tanh_tanh_0.3 difference-edit_off-edit_on real
run ae_fc_5_7_relu_linear_0.2 difference-edit_off-edit_on real

# Imaginary + Real
run ae_fc_5_7_relu_linear_0.3 edit_off-edit_on imaginary-real
run ae_fc_6_6_relu_linear_0.3 difference-edit_off-edit_on imaginary-real
run ae_fc_5_6_relu_linear_0.2 difference-edit_on imaginary-real
run ae_fc_6_6_relu_tanh_0.3 difference-edit_on imaginary-real

echo "All required AE (100k) training jobs submitted."
