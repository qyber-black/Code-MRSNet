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
SPLIT=0
FOLD=NoValidation-1

BATCH=16
NORM=sum
PULSE=megapress

# Check if a model already exists in data/model or data/model-cnn
exists_model() {
  local MODEL="$1"     # cnn_...
  local ACQ_DIR="$2"   # e.g., edit_off-edit_on
  local DT_DIR="$3"    # e.g., imaginary-real
  local DS="`echo $DATASET | cut -d / -f 4- | sed 's|/|_|g'`"
  for BASE in ./data/model ./data/model-dist ./data/model-cnn data/model-ae; do
    local TARGET="$BASE/$MODEL/$METABS/$PULSE/$ACQ_DIR/$DT_DIR/$NORM/$BATCH/$EPOCHS/$DS"
    if [ -d "$TARGET/$FOLD" ]; then
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

run cnn_medium_sigmoid_pool edit_off-edit_on real
