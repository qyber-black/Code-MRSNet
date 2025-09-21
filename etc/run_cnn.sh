#!/usr/bin/env bash
#
# Some CNN runs without a full grid
#
# Usage:
#   bash etc/run_cnn.sh

set -euo pipefail

# Below has been setup for the best CNN parameterised models on the 100K dataset

# Dataset and training, adjust as needed.
METABS="Cr-GABA-Gln-Glu-NAA"
DATASET=./data/sim-spectra-megapress/fid-a-2d_2000_4096/siemens/123.23/2.0/$METABS/megapress/sobol/1.0-adc_normal-0.0-0.03/100000-1
EPOCHS=1000
SPLIT=5

BATCH=16
NORM=sum
PULSE=megapress

# Check if a model already exists in data/model or data/model-cnn
exists_model() {
  local MODEL="$1"     # cnn_...
  local ACQ_DIR="$2"   # e.g., edit_off-edit_on
  local DT_DIR="$3"    # e.g., imaginary-real
  for BASE in ./data/model-cnn; do
    local TARGET="$BASE/$MODEL/$METABS/$PULSE/$ACQ_DIR/$DT_DIR/$NORM/$BATCH/$EPOCHS/fid-a-2d_2000_4096_siemens_123.23_2.0_Cr-GABA-Gln-Glu-NAA_megapress_sobol_1.0-adc_normal-0.0-0.03_10000-1"
    if [ -d "$TARGET/KFold_5-1" ]; then
      echo "[FOUND] $TARGET"
      return 0
    fi
  done
  return 1
}

# Run one job if not already existing
# Args: LABEL MODEL_STRING ACQ_DIR DT_DIR
run() {
  local MODEL="$1"; shift
  local ACQ_DIR="$1"; shift
  local DT_DIR="$1"; shift
  local LABEL="$MODEL--$ACQ_DIR--$DT_DIR"

  echo "[CHECK] $MODEL"
  if exists_model "$MODEL" "$ACQ_DIR" "$DT_DIR"; then
    echo "[SKIP] $LABEL (exists)"
    return 0
  fi

  # Build CLI arrays from dirs
  IFS='-' read -r -a ACQS <<< "$ACQ_DIR"
  IFS='-' read -r -a DTS <<< "$DT_DIR"

  echo "[RUN] $LABEL"
  ./mrsnet.py train \
    -d "$DATASET" \
    -e "$EPOCHS" -k "$SPLIT" \
    --norm "$NORM" \
    --acquisitions "${ACQS[@]}" \
    --datatype "${DTS[@]}" \
    -m "$MODEL" \
    -b "$BATCH" -v
}

run cnn_-1_-3_9_9_5_3_0.0_0.3_256_512_2048_sigmoid edit_off-edit_on imaginary-real
run cnn_-1_-3_9_7_3_3_0.0_0.3_512_512_2048_sigmoid edit_off-edit_on real
run cnn_-1_-3_7_9_5_3_0.0_0.3_512_512_2048_sigmoid edit_off-edit_on imaginary-real

run cnn_medium_sigmoid_pool edit_off-edit_on real
