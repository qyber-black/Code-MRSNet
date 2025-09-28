#!/usr/bin/env bash
#
# Run AEQ (encoder-quantifier) model selections on the 100k dataset.
#
# SPDX-FileCopyrightText: Copyright (C) 2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Usage:
#   bash etc/run_aeq.sh

set -euo pipefail

ROOT="/srv/data/Prj/MRS/code-mrsnet"

DATASET="$ROOT/data/sim-spectra-megapress/fid-a-2d_2000_4096/siemens/123.23/2.0/Cr-GABA-Gln-Glu-NAA/megapress/sobol/1.0-adc_normal-0.0-0.03/100000-1"
EPOCHS=200
VALID=0.8   # treat Split and KFold equally; adjust to 5 for KFold

SPEC_DIR="$ROOT/data/model-ae/selection-spec"

run_sel() {
  local SPEC="$1"; shift
  local REPEATS="$1"; shift
  echo "[RUN] $SPEC (repeats=$REPEATS)"
  "$ROOT/mrsnet.py" select \
    -d "$DATASET" -e "$EPOCHS" -k "$VALID" \
    --method gpo "$SPEC" -v -r "$REPEATS"
}

# Real contexts
run_sel "$SPEC_DIR/aeq-para-diffon-real.json" 160
run_sel "$SPEC_DIR/aeq-para-offon-real.json" 160
run_sel "$SPEC_DIR/aeq-para-diffoff-real.json" 120
run_sel "$SPEC_DIR/aeq-para-diffoffon-real.json" 160

# Imaginary+Real contexts
run_sel "$SPEC_DIR/aeq-para-offon-imagreal.json" 160
run_sel "$SPEC_DIR/aeq-para-diffoffon-imagreal.json" 140
run_sel "$SPEC_DIR/aeq-para-diffon-imagreal.json" 140

echo "All AEQ selections submitted."


