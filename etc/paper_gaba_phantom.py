#!/usr/bin/env python3
#
# paper_gaba_phantom.py - MRSNet - result data of the GABA phantom paper
#
# SPDX-FileCopyrightText: Copyright (C) 2026 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

r"""Generate the result data of the GABA phantom paper from the released data repositories.

Z Ma, SM Shermer, O Karakus, FC Langbein (MELBA submission, 2026).

The paper repository renders its tables, LaTeX macros and figures from these
files (its scripts/model_mae_analysis/*.py, driven by its Makefile). This script
only produces data: it renders no table, macro or figure, loads no model and runs
no inference in its default path. It does not import the paper repository and
needs only NumPy (it does not import the mrsnet library). The composition-matched
control, which needs inference, is the separate script etc/composition_matched.py;
its finished result folder is packaged by the stage composition below. Its
linewidth/noise decomposition, likewise inference, is the separate script
etc/composition_matched_decomposition.py; its result folder is verified and
packaged by the stage decomposition below. The per-metabolite signal-gain fit
is the separate script etc/metabolite_gain_fit.py (mrsnet library, SciPy, no
TensorFlow); the stage gains below reruns it and packages its result folder.

Inputs (all released; paths below the MRSNet checkout --code-root, default: the
checkout containing this script):

* data/benchmark            data-megapress-spectra (info.json metadata, export folders)
* data/model-dist           results-mrsnet-models: model registry and the thirteen
                            adopted checkpoints with their phantom and simulated raw
                            predictions and ground truth
* data/model-yae            results-mrsnet-models-yae: caeq_noval five-seed candidate
                            runs, caeq/aeq/ae_kfold selection folds, and the superseded
                            OFF/ON YAE pair at the repository root
* data/model-cnn            results-mrsnet-models-cnn: CNN candidate seeds and folds
* data/model-extra          results-mrsnet-models-extra (searched for folds)
* MANIFEST_CURATED below    curated identity and dependency policy (stage manifest)
* optionally a finished ``etc/composition_matched.py --outdir DIR`` result folder
* a finished ``etc/composition_matched_decomposition.py`` result folder (default:
  the released data/sim2real/paper-gaba-phantom/composition-decomposition)
* the released data/sim2real/paper-gaba-phantom/metabolite-gains folder, if present,
  and an interpreter with the mrsnet dependencies for etc/metabolite_gain_fit.py
  (--gains-python, --gains-python-path)

Outputs (below --outdir, default data/sim2real/paper-gaba-phantom, i.e. the
results-mrsnet-sim2real repository). The paper consumer of each file is named in
the docstring of the stage that writes it:

  benchmark_manifest.json                                        stage manifest
  selected/{phantom_metrics,simulated_metrics,phantom_errors}.csv,
  selected/simulated_rows.tex, selected/source_catalog.json      stage selected
  pipe-b-audit/{inventory.csv,topk_sources.json}                 stage audit
  pipe-b-descriptive/{descriptive_rank_inputs.csv,
                      descriptive_input_sha256.json}             stage descriptive
  cnn-selection/simple_grid_ranked.csv                           stage cnn
  composition-matched/*                                          stage composition
  composition-decomposition/*                                    stage decomposition
  metabolite-gains/*                                             stage gains
  provenance.json        script and input hashes of the last run of each stage

The spectral measurements (data/sim2real/magnitude-residuals, linewidth) and the
YAE selection aggregates (data/model-yae/*_kfold/aggregate/all_results.csv) are
released already; the paper reads them directly.

Usage, from the repository root::

  etc/paper_gaba_phantom.py                            # all non-inference stages
  etc/paper_gaba_phantom.py --stage selected           # one stage (repeatable)
  etc/paper_gaba_phantom.py --composition-matched DIR  # also package DIR
  etc/paper_gaba_phantom.py --decomposition DIR        # verify and package another decomposition
  etc/paper_gaba_phantom.py --gains-python PY --gains-python-path DIRS  # interpreter for stage gains

Every stage checks its inputs (registry checkpoint and result pins, saved summaries
against raw arrays, benchmark metadata against saved truth) and fails rather than
writing a partial or inconsistent output. Floats are written in Python's shortest
round-trip form, so a rerun on the same inputs reproduces the files byte for byte.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = "etc/paper_gaba_phantom.py"
METABOLITES = ["Cr", "GABA", "Gln", "Glu", "NAA"]
OUTCOMES = [*METABOLITES, "Overall"]
# Metadata concentration keys of data-megapress-spectra, in metabolite order.
NAMES = dict(zip(["creatine", "gaba", "glutamine", "glutamate", "n-acetylaspartate"], METABOLITES, strict=True))
# The fourteen acquisition series of the 144-spectrum cohort and their sizes.
SIZES = dict(
  zip(
    ["E1", "E2", "E3", "E4a", "E4b", "E4c", "E4d", "E6", "E7", "E8", "E9a", "E9b", "E11", "E14"],
    [13, 16, 15, 8, 8, 8, 8, 10, 16, 8, 8, 7, 8, 11],
    strict=True,
  )
)
BENCHMARK_DOCS = ("README.md", "readme-phantoms.md", "readme-experiment.md", "readme-old_names.md")
REGISTRY = "data/model-dist/paper-model-registry/model_registry.json"
YAE = "caeq_fc_5_6_tanh_tanh_0.2_384_2_sigmoid_sigmoid_-1.0"
AUGMENTED = "1.0:0.2:10.0"
# The eleven instances of the paper's selection (all eight fixed models and the
# YAE, CNN and FCNN augmented models). Rows of the selected/*.csv files follow
# this order, the order in which the paper's selection was assembled.
SELECTION_ORDER = (
  ("cnn_-1_-3_9_9_5_3_0.0_0.3_512_512_2048_sigmoid", "fixed"),
  ("cnn_medium_sigmoid_pool", "augmented"),
  ("cnn_medium_sigmoid_pool", "fixed"),
  ("encdec_default", "fixed"),
  ("fcnn_default", "augmented"),
  ("fcnn_default", "fixed"),
  ("qmrs_default", "fixed"),
  ("qnet_default", "fixed"),
  (YAE, "augmented"),
  (YAE, "fixed"),
  ("qnet_basis_default", "fixed"),
)
# Labels and row order of the legacy simulated-summary fragment selected/simulated_rows.tex
# (the paper's current table uses its own labels and order; see stage selected).
LEGACY_LABELS = {
  YAE: "MRSNet-YAE",
  "cnn_-1_-3_9_9_5_3_0.0_0.3_512_512_2048_sigmoid": "CNN-param",
  "cnn_medium_sigmoid_pool": "MRSNet-CNN",
  "encdec_default": "EncDec",
  "fcnn_default": "FCNN",
  "qmrs_default": "QMRS",
  "qnet_basis_default": "QNet-basis",
  "qnet_default": "QNet",
}
# Five-candidate selection families of the paper's candidate study (paper scripts/sim2real/
# rank_comparison_topk.py FAMILIES): the five best configurations of each selection search.
FAMILIES = {
  "CNN": [
    ("cnn_-1_-3_9_9_5_3_0.0_0.3_512_512_2048_sigmoid", "edit_off-edit_on", "imaginary-real"),
    ("cnn_-2_-3_9_9_5_3_0.0_0.3_256_512_2048_sigmoid", "edit_off-edit_on", "imaginary-real"),
    ("cnn_-1_-3_9_9_5_3_0.0_0.3_256_512_1024_sigmoid", "edit_off-edit_on", "imaginary-real"),
    ("cnn_-1_-3_7_5_7_1_0.0_0.3_512_512_2048_sigmoid", "edit_off-edit_on", "imaginary-real"),
    ("cnn_-2_-3_7_9_5_3_0.0_0.3_512_512_2048_sigmoid", "edit_off-edit_on", "imaginary-real"),
  ],
  "CAEQ": [
    ("caeq_fc_5_7_tanh_tanh_0.2_448_4_sigmoid_sigmoid_-1.0", "edit_off-edit_on", "imaginary-real"),
    ("caeq_fc_5_6_tanh_tanh_0.2_384_2_sigmoid_softmax_-1.0", "edit_off-edit_on", "imaginary-real"),
    ("caeq_fc_5_7_tanh_tanh_0.2_512_2_sigmoid_softmax_-1.0", "edit_off-edit_on", "imaginary-real"),
    ("caeq_fc_5_7_tanh_linear_0.2_512_4_sigmoid_sigmoid_-1.0", "edit_off-edit_on", "imaginary-real"),
    ("caeq_fc_5_7_tanh_linear_0.2_512_2_sigmoid_softmax_-1.0", "edit_off-edit_on", "imaginary-real"),
  ],
}
# Released roots searched for the five no-validation seeds of each candidate ...
SEED_ROOTS = ("data/model-cnn", "data/model-yae/caeq_noval", "data/model-dist")
# ... and for the five KFold selection folds of each candidate.
FOLD_ROOTS = ("model-cnn", "model-extra", "model-yae/ae_kfold", "model-yae/aeq_kfold", "model-yae/caeq_kfold")
# Curated fields of benchmark_manifest.json. These are author or experimenter judgement
# and cannot be derived from the released benchmark metadata; everything else in the
# manifest (records, concentrations, counts, source hashes) is derived (stage manifest).
MANIFEST_CURATED = {
  "schema_version": 1,
  "status": "verified within-experiment identities; experimenter confirms distinct objects across numerical "
  "experiments; physical batch IDs unrecovered",
  "experimenter_confirmation": {
    "response": "Different number has different object scanned.",
    "interpretation": "Distinct numerical experiments scan distinct objects; lettered acquisitions repeat objects "
    "within a numerical experiment.",
    "scope_limit": "This confirms object non-reuse, not recovered physical preparation-batch IDs or absence of every "
    "shared experimental influence.",
  },
  "identity_policy": "Use exact saved spectrum IDs; E14 metadata uses its verified six-character sample prefix. E9 "
  "GEL05_TX_ADJ is a repeat of GEL05. Acquisition round and repeat IDs are categorical, not chronological indices.",
  "dependency_policy": "Keep the complete numerical experiment together. Preparation and titration dependency groups "
  "are conservative analysis groups, not recovered physical batch or chain identifiers. Unknown physical identifiers "
  "and dates are null. Do not infer independence between concentration states.",
  # Numerical experiments prepared as liquid titration series (the others are gels).
  "liquid_experiments": ["E1", "E2", "E3", "E6", "E7"],
  # Saved E14 spectrum names carry a scan suffix; the metadata sample is the prefix.
  "sample_prefix_length": {"E14": 6},
  # E9a GEL05_TX_ADJ is a transmitter-adjusted repeat scan of the object GEL05.
  "repeat_aliases": {"GEL05_TX_ADJ": {"object": "GEL05", "repeat": "TX_ADJ"}},
  # The only acquisition dates recovered from the experiment records.
  "acquisition_dates": {"E4a": "2018-10-17", "E4b": "2018-10-17", "E4c": "2018-10-11", "E4d": "2018-10-11"},
}
# Files of an etc/composition_matched.py result folder that the paper renders from.
MATCHED_MODELS = ("YAE", "CNN", "FCNN", "QNet", "QNet-basis")
MATCHED_FILES = (
  "protocol.json",
  "metrics.csv",
  *(m + ".npz" for m in MATCHED_MODELS),
  *(m + "-phantom-runtime-control.npz" for m in MATCHED_MODELS),
)
# Files of an etc/composition_matched_decomposition.py result folder that the paper renders from.
DECOMPOSITION = "data/sim2real/paper-gaba-phantom/composition-decomposition"
DECOMPOSITION_VARIANTS = ("A-baseline", "B-linewidth", "C-noise", "D-linewidth-noise")
DECOMPOSITION_FILES = ("protocol.json", "metrics.csv", *(v + ".npz" for v in DECOMPOSITION_VARIANTS))
# Inputs whose SHA-256 a decomposition protocol must record (besides the registry
# checkpoints and metadata, the YAE phantom inputs and the basis files of every
# linewidth used, which are derived below), whatever its own list says.
DECOMPOSITION_REFERENCE = "data/sim2real/paper-gaba-phantom/composition-matched"
DECOMPOSITION_LINEWIDTHS = "data/sim2real/linewidth/basis-mixture-off/linewidth_per_spectrum.csv"
DECOMPOSITION_SOURCES = (
  "etc/composition_matched_decomposition.py",
  "etc/composition_matched.py",
  REGISTRY,
  DECOMPOSITION_LINEWIDTHS,
  *(DECOMPOSITION_REFERENCE + "/" + n for n in ("metrics.csv", *(m + ".npz" for m in MATCHED_MODELS))),
  *(
    "mrsnet/" + n
    for n in (
      "basis.py",
      "dataset.py",
      "spectrum.py",
      "molecules.py",
      "benchmark_data.py",
      "normalisation.py",
      "cfg.py",
      "getfolder.py",
      "cnn.py",
      "ae_quantifier.py",
      "fcnn.py",
      "qnet.py",
      "basis_lls.py",
      "train.py",
    )
  ),
)
BASIS_DIR = "data/basis-dist/fid-a-2d/basis_files/"
BASIS_FILE = re.compile(r"FIDA2D_([A-Za-z0-9]+)_MEGAPRESS_EDIT(ON|OFF)_(\d+\.\d\d)_2000_4096_123\.23\.mat")
# Thread settings under which the decomposition is run and recorded.
DECOMPOSITION_THREADS = {"TF_NUM_INTRAOP_THREADS": "4", "TF_NUM_INTEROP_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
# Result folder of etc/metabolite_gain_fit.py, its files, and the files a rerun must reproduce byte for byte.
GAINS = "data/sim2real/paper-gaba-phantom/metabolite-gains"
GAINS_FILES = ("series_gains.csv", "per_spectrum.csv", "recovery.csv", "gate1.json", "README.md", "protocol.json")
GAINS_REPRODUCED = GAINS_FILES[:-1]
# Inputs whose SHA-256 a gains protocol must record (besides the ten 1.0 Hz basis files).
GAINS_SOURCES = (
  "etc/metabolite_gain_fit.py",
  "data/sim2real/paper-gaba-phantom/benchmark_manifest.json",
  "data/benchmark/benchmark_sequences.json",
  DECOMPOSITION_LINEWIDTHS,
  DECOMPOSITION + "/protocol.json",
  *(
    "data/model-dist/qnet_default/Cr-GABA-Gln-Glu-NAA/megapress/edit_off-edit_on/real/sum/16/2000/"
    "fid-a-2d_2000_4096_siemens_123.23_2.0_Cr-GABA-Gln-Glu-NAA_megapress_sobol_1.0-adc_normal-0.0-0.03_100000-1/"
    "NoValidation-1/evaluation/phantom/" + n
    for n in ("report.json", "inputs.npz")
  ),
  *(
    "mrsnet/" + n
    for n in (
      "basis.py",
      "benchmark_data.py",
      "cfg.py",
      "dataset.py",
      "linewidth.py",
      "molecules.py",
      "normalisation.py",
      "robust_linewidth.py",
      "spectrum.py",
    )
  ),
)
# Thread settings under which the gain fit is run and recorded.
GAINS_THREADS = {"OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"}
# Stages in the order they run.
STAGES = ("manifest", "selected", "audit", "descriptive", "cnn", "composition", "decomposition", "gains")


# ------------------------------------------------------------------ helpers
def sha256(path):
  """Hash a file in blocks (checkpoints are large)."""
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1048576), b""):
      digest.update(block)
  return digest.hexdigest()


def child(root, name):
  """Resolve a catalogued relative path, rejecting paths that escape their repository."""
  path = Path(name)
  if path.is_absolute() or ".." in path.parts or not path.parts:
    raise ValueError("unsafe source path: " + name)
  result = root / path
  if not result.resolve().is_relative_to(root.resolve()):
    raise ValueError("source escapes its repository: " + name)
  return result


def write_csv(path, rows):
  """Write a deterministic CSV with the column order of the first row."""
  if not rows:
    raise ValueError("empty output: " + str(path))
  with path.open("w", newline="", encoding="ascii") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def write_json(path, value):
  """Write sorted, indented ASCII JSON."""
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")


def normalise(values, role):
  """Validate and independently max-normalise rows, retaining negative predictions."""
  arr = np.asarray(values, dtype=float)
  if arr.ndim != 2 or arr.shape[1] != 5 or not len(arr):
    raise ValueError("expected nonempty five-metabolite rows")
  if not np.isfinite(arr).all() or (role == "truth" and (arr < 0).any()):
    raise ValueError(f"invalid {role} values")
  factors = arr.max(axis=1)
  if (factors <= 0).any():
    raise ValueError(f"non-positive {role} maximum")
  return arr / factors[:, None], factors


def metrics(prediction, truth):
  """Absolute errors, five per-metabolite MAEs and Overall (equal spectrum and metabolite weights)."""
  prediction, _ = normalise(prediction, "prediction")
  truth, _ = normalise(truth, "truth")
  errors = np.abs(prediction - truth)
  means = errors.mean(axis=0)
  overall = errors.mean(axis=1).mean()
  if not np.isclose(overall, means.mean(), atol=1e-12, rtol=1e-12):
    raise ValueError("Overall identity failed")
  return errors, [*means, overall]


def metric_rows(entry, prediction, truth):
  """Per-metabolite and Overall MAE with descriptive SD (ddof 1) of one prediction array."""
  errors, means = metrics(prediction, truth)
  sd = [*errors.std(axis=0, ddof=1), errors.mean(axis=1).std(ddof=1)]
  return [
    {
      "model": entry["model"],
      "regime": entry["regime"],
      "metabolite": m,
      "n": len(truth),
      "mae": float(mean),
      "sd": float(std),
      "negative_predictions": int(np.count_nonzero(np.asarray(prediction) < 0)),
    }
    for m, mean, std in zip(OUTCOMES, means, sd, strict=True)
  ]


def detailed_records(run, ids, prediction, truth):
  """Per-spectrum records: saved values, evaluation factors, normalised values and errors."""
  pred, pf = normalise(prediction, "prediction")
  true, tf = normalise(truth, "truth")
  records = []
  for i, sample_id in enumerate(ids):
    for j, metabolite in enumerate(METABOLITES):
      records.append(
        {
          "run": run,
          "spectrum": sample_id,
          "metabolite": metabolite,
          "saved_prediction": prediction[i][j],
          "saved_truth": truth[i][j],
          "prediction_evaluation_factor": pf[i],
          "truth_evaluation_factor": tf[i],
          "evaluation_prediction": pred[i, j],
          "evaluation_truth": true[i, j],
          "signed_error": pred[i, j] - true[i, j],
          "absolute_error": abs(pred[i, j] - true[i, j]),
        }
      )
  return records


def validate_ids(ids, expected=None):
  """Require complete unique IDs, optionally in a given saved order."""
  if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids):
    raise ValueError("empty or duplicate sample IDs")
  if expected is not None and not np.array_equal(ids, expected):
    raise ValueError("prediction/truth sample IDs disagree")


def quantify(path):
  """Read a MRSNet *_quantify.csv: spectrum IDs, predictions and truth in metabolite order."""
  ids, rows, current = [], [], {}
  order = None
  with path.open(newline="") as stream:
    for row in csv.reader(stream):
      if not row:
        continue
      if row[0] == "Metabolites":
        order = row[1:]
      elif row[0].startswith("Spectrum:"):
        if ids:
          rows.append(current)
        ids.append(row[0].split(":", 1)[1].strip())
        current = {}
      elif row[0] in METABOLITES:
        if not ids or row[0] in current or len(row) != 4:
          raise ValueError(f"invalid metabolite record: {path}")
        current[row[0]] = [float(x) for x in row[1:]]
  if ids:
    rows.append(current)
  if order != METABOLITES or len(set(ids)) != len(ids):
    raise ValueError(f"order or duplicate spectrum IDs: {path}")
  if not rows or any(set(row) != set(METABOLITES) for row in rows):
    raise ValueError(f"missing metabolite records: {path}")
  values = np.asarray([[row[m] for m in METABOLITES] for row in rows])
  if not np.isfinite(values).all():
    raise ValueError(f"non-finite CSV values: {path}")
  if not np.allclose(values[:, :, 2], values[:, :, 0] - values[:, :, 1], atol=1e-10, rtol=1e-8):
    raise ValueError(f"CSV signed-error identity failed: {path}")
  return ids, values[:, :, 0], values[:, :, 1]


def phantom_arrays(folder):
  """Read IDs, raw predictions and truth of a released phantom evaluation folder.

  Folders written by the paper evaluation hold ground_truth.npz/raw_predictions.npz;
  older canonical phantom folders hold inputs.npz/predictions.npz ("canonical").
  """
  if (folder / "ground_truth.npz").is_file():
    with np.load(folder / "ground_truth.npz", allow_pickle=False) as saved:
      ids, truth = saved["id"], saved["truth_raw"]
    with np.load(folder / "raw_predictions.npz", allow_pickle=False) as saved:
      validate_ids(saved["id"], ids)
      prediction = saved["prediction_raw"]
  else:
    with np.load(folder / "inputs.npz", allow_pickle=False) as saved:
      ids, truth = saved["id"], saved["truth"]
    with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
      validate_ids(saved["id"], ids)
      prediction = saved["canonical"]
  validate_ids(ids)
  return ids, prediction, truth


def benchmark_truth(code):
  """Metadata concentrations (mM, metabolite order) of every benchmark sample, by series."""
  expected = {}
  for series in SIZES:
    info = json.loads((code / "data/benchmark" / series / "info.json").read_text())
    expected[series] = {
      sample: [float({NAMES.get(k, k): v for k, v in conc.items()}.get(m, 0)) for m in METABOLITES]
      for sample, conc in info["concentrations"].items()
    }
  return expected


def sample_key(sample_id):
  """Series and metadata sample name of a saved spectrum ID (E14 uses a six-character prefix)."""
  parts = sample_id.rstrip("/").split("/")
  return parts[0], (parts[-1][:6] if parts[0] == "E14" else parts[-1])


def load_manifest(out):
  """Read the generated manifest (stage manifest must run first)."""
  path = out / "benchmark_manifest.json"
  if not path.is_file():
    raise ValueError("run stage manifest first: " + str(path))
  return json.loads(path.read_text())


# ------------------------------------------------------------------ registry
def deployed_instances(code):
  """Check the thirteen registry instances: checkpoint, metadata and result pins.

  Mirrors the paper's canonical_prediction_sources.deployed_instances, so the paper
  renderers accept the sources recorded here.
  """
  record = json.loads(child(code, REGISTRY).read_text())
  entries = record["instances"]
  fixed = {"YAE", "CNN", "CNN-param", "FCNN", "EncDec", "QMRS", "QNet", "QNet-basis"}
  expected = {(name, "fixed") for name in fixed}
  expected |= {(name, "augmented") for name in ("YAE", "CNN", "FCNN", "QNet", "QNet-basis")}
  if len(entries) != 13 or {(r["paper_name"], r["regime"]) for r in entries} != expected:
    raise ValueError("incomplete current deployment registry")
  by_key = {}
  for entry in entries:
    checkpoint = child(code, entry["checkpoint_path"])
    if not str(checkpoint.parent.relative_to(code)).startswith("data/model-dist/"):
      raise ValueError("non-distributed deployment")
    for kind in ("checkpoint", "metadata"):
      if sha256(child(code, entry[kind + "_path"])) != entry[kind + "_sha256"]:
        raise ValueError("deployment identity changed: " + entry[kind + "_path"])
    ph = child(code, entry["phantom_result"].removeprefix("code/"))
    si = child(code, entry["simulated_result"].removeprefix("code/"))
    if ph.parent != si.parent or not ph.is_relative_to(checkpoint.parent):
      raise ValueError("evaluation outside canonical model run")
    if not entry["result_sha256"]:
      raise ValueError("empty deployment result pins")
    for name, digest in entry["result_sha256"].items():
      if sha256(child(si.parent, name)) != digest:
        raise ValueError("saved result changed: " + str(si.parent / name))
    key = entry["metadata"]["model"], entry["regime"]
    if key in by_key:
      raise ValueError("duplicate deployment")
    by_key[key] = entry
  return by_key


def domain_arrays(code, entry, domain):
  """Read one domain's raw arrays and check them against the saved metrics.json."""
  folder = child(code, entry[domain + "_result"].removeprefix("code/"))
  ids, prediction, truth = phantom_arrays(folder)
  count = 144 if domain == "phantom" else 100000
  if len(ids) != count or prediction.shape != (count, 5) or truth.shape != prediction.shape:
    raise ValueError("canonical prediction cohort differs")
  errors = np.abs(normalise(prediction, "prediction")[0] - normalise(truth, "truth")[0])
  summary = json.loads((folder / "metrics.json").read_text())
  if (
    summary["norm"] != "max"
    or summary["metabolites"] != METABOLITES
    or summary["n"] != count
    or summary["checkpoint_sha256"] != entry["checkpoint_sha256"]
    or summary["negative_predictions"] != int((prediction < 0).sum())
    or not np.allclose(
      [summary["per_metabolite_mae"][m] for m in METABOLITES] + [summary["overall_mae"]],
      [*errors.mean(axis=0), errors.mean()],
      rtol=0,
      atol=1e-12,
    )
  ):
    raise ValueError("canonical summary differs from raw predictions: " + str(folder))
  return ids, prediction, truth


def run_row(code, run, model, regime):
  """One inventory row of a released training run, read from its mrsnet.json."""
  meta = json.loads((run / "mrsnet.json").read_text())
  if meta["model"] != model or meta["metabolites"] != METABOLITES:
    raise ValueError("run metadata differs: " + str(run))
  return {
    "run": str(run.relative_to(code)),
    "model": model,
    "regime": regime,
    "training_target": meta["norm"],
    "evaluation": "max",
    "seed": str(meta.get("seed", "unknown")),
    "checkpoint": next((p.name for p in sorted(run.glob("*.keras"))), "not established"),
  }


# ------------------------------------------------------------------ stage: manifest
def stage_manifest(code, out):
  """Write benchmark_manifest.json: identities, groups and known compositions of the 144 spectra.

  Paper consumers: every renderer in scripts/model_mae_analysis (cluster structure for
  all phantom tables and intervals; constants/benchmark_counts.tex embeds this file's
  SHA-256). Formerly hand-curated as paper scripts/model_mae_analysis/benchmark_manifest.json.

  Derived from released data: spectrum IDs and their order (the saved phantom evaluation
  IDs of the registry instances in data/model-dist, identical for all thirteen), series
  and sample names (parsed from the IDs and checked against the data/benchmark export
  folders), concentrations (data/benchmark/<series>/info.json, key order kept), counts
  and the SHA-256 of the metadata and the four benchmark READMEs. Curated
  (MANIFEST_CURATED): status, experimenter confirmation, identity and
  dependency policy texts, which experiments are liquid series, the E14 sample prefix
  rule, the GEL05_TX_ADJ repeat alias and the four known acquisition dates.
  """
  curated = MANIFEST_CURATED
  entries = deployed_instances(code)
  order = None
  for entry in entries.values():
    folder = child(code, entry["phantom_result"].removeprefix("code/"))
    ids = phantom_arrays(folder)[0]
    if order is not None:
      validate_ids(ids, order)
    order = ids
  ids = [str(x) for x in order]
  metadata = {s: json.loads((code / f"data/benchmark/{s}/info.json").read_text()) for s in SIZES}
  sources = {f"data/benchmark/{s}/info.json": sha256(code / f"data/benchmark/{s}/info.json") for s in SIZES}
  sources.update({"data/benchmark/" + name: sha256(code / "data/benchmark" / name) for name in BENCHMARK_DOCS})
  liquid = set(curated["liquid_experiments"])
  aliases = curated["repeat_aliases"]
  records, objects = [], {}
  for sid in ids:
    parts = sid.rstrip("/").split("/")
    if len(parts) != 3 or any(p in ("", ".", "..") for p in parts):
      raise ValueError("unsafe spectrum identity: " + sid)
    series, export, sample = parts
    match = re.fullmatch(r"(E\d+)([a-z]?)", series)
    if series not in SIZES or match is None:
      raise ValueError("unexpected acquisition series: " + sid)
    experiment, letter = match.groups()
    folder = code / "data/benchmark" / series
    expected_export = "MEGA_RAW_Combi_WS_ON" if (folder / "MEGA_RAW_Combi_WS_ON").is_dir() else "MEGA_Combi_WS_ON"
    state = sample[: curated["sample_prefix_length"][series]] if series in curated["sample_prefix_length"] else sample
    if export != expected_export or not (folder / export / state).is_dir():
      raise ValueError("phantom export policy or folder mismatch: " + sid)
    alias = aliases.get(state)
    object_id = experiment + "/" + (alias["object"] if alias else state)
    material = "solution" if experiment in liquid else "gel"
    truth = metadata[series]["concentrations"][state]
    vector = np.array([truth.get(k, 0) for k in NAMES], dtype=float)
    if not np.isfinite(vector).all() or (vector < 0).any() or vector.max() <= 0:
      raise ValueError("invalid metadata concentrations: " + sid)
    if object_id in objects and objects[object_id] != truth:
      raise ValueError("repeated-object truth mismatch: " + object_id)
    objects[object_id] = truth
    records.append(
      {
        "experiment_id": experiment,
        "acquisition_series_id": series,
        "lettered_acquisition": letter or None,
        "acquisition_round_id": series,
        "acquisition_date": curated["acquisition_dates"].get(series),
        "material": material,
        "preparation_dependency_group": experiment + "/preparation",
        "verified_preparation_batch_id": None,
        "titration_dependency_group": experiment + "/liquid-series" if material == "solution" else None,
        "verified_titration_chain_id": None,
        "object_or_state_id": object_id,
        "gel_object_id": None if material == "solution" else object_id,
        "concentration_state_id": object_id,
        "metadata_sample_id": state,
        "repeat_id": series + ("/" + alias["repeat"] if alias else "/standard"),
        "spectrum_id": sid,
        "method_pairing_key": sid,
        "metadata_source": f"data/benchmark/{series}/info.json",
        # Integral concentrations are written as integers (15.0 in info.json -> 15).
        "concentrations_mM": {k: int(v) if float(v).is_integer() else v for k, v in truth.items()},
      }
    )
  if [sum(r["acquisition_series_id"] == s for r in records) for s in SIZES] != list(SIZES.values()):
    raise ValueError("series sizes differ from the 144-spectrum cohort")
  counts = {
    "experiments": len({r["experiment_id"] for r in records}),
    "acquisition_series": len({r["acquisition_series_id"] for r in records}),
    "objects_or_states": len(objects),
    "spectra": len(records),
    "liquid_states": len({r["object_or_state_id"] for r in records if r["material"] == "solution"}),
    "gel_objects": len({r["object_or_state_id"] for r in records if r["material"] == "gel"}),
  }
  experiment_counts = [
    {
      "experiment_id": e,
      "objects_or_states": len({r["object_or_state_id"] for r in records if r["experiment_id"] == e}),
      "spectra": sum(r["experiment_id"] == e for r in records),
    }
    for e in dict.fromkeys(r["experiment_id"] for r in records)
  ]
  manifest = {k: curated[k] for k in ("schema_version", "status", "experimenter_confirmation")}
  manifest["metabolite_order"] = METABOLITES
  manifest.update({k: curated[k] for k in ("identity_policy", "dependency_policy")})
  manifest.update(source_hashes_sha256=sources, counts=counts, experiment_counts=experiment_counts, records=records)
  # Layout: two-space indentation, the metabolite list on one line, one compact record per line.
  parts = []
  for key, value in manifest.items():
    if key == "metabolite_order":
      text = json.dumps(value)
    elif key == "records":
      text = "[\n" + ",\n".join("    " + json.dumps(r, separators=(",", ":")) for r in value) + "\n  ]"
    else:
      text = json.dumps(value, indent=2).replace("\n", "\n  ")
    parts.append("  " + json.dumps(key) + ": " + text)
  (out / "benchmark_manifest.json").write_text("{\n" + ",\n".join(parts) + "\n}\n", encoding="ascii")
  inputs = dict(sources)
  inputs[REGISTRY] = sha256(code / REGISTRY)
  return ["benchmark_manifest.json"], inputs


# ------------------------------------------------------------------ stage: selected
def legacy_simulated_rows(rows):
  """Render the legacy simulated MAE +/- SD rows, ordered by label.

  Kept so the file list hashed by the paper renderers stays complete; no paper table
  reads it (tables/fixed_simulated_rows.tex is rendered from simulated_metrics.csv).
  """
  grouped = {}
  for row in rows:
    entries = grouped.setdefault((row["model"], row["regime"]), {})
    if row["metabolite"] in entries:
      raise ValueError("duplicate table metric")
    entries[row["metabolite"]] = row
  lines = ["% Generated by audit_final_simulated.py; mean +/- descriptive SD."]
  labelled = []
  for (model, regime), entries in grouped.items():
    if set(entries) != set(OUTCOMES):
      raise ValueError("incomplete table metabolites")
    label = LEGACY_LABELS[model] + (" (aug.)" if regime == "augmented" else "")
    cells = [f"${entries[m]['mae']:.4f} \\pm {entries[m]['sd']:.4f}$" for m in OUTCOMES]
    labelled.append((label, label + " & " + " & ".join(cells) + r" \\"))
  lines.extend(line for _, line in sorted(labelled))
  return "\n".join(lines) + "\n"


def stage_selected(code, out):
  """Write selected/*: metrics and per-spectrum errors of the eleven selected instances.

  Paper consumers: render_selected_fixed.py (tables/fixed_main_table.tex,
  fixed_simulated_rows.tex, constants/values.tex, the three fixed_* figures),
  compare_selected_clusters.py (primary/secondary_comparisons.tex),
  summarise_signed_gaba.py (signed_gaba_*.tex) and render_selected_augmentation.py
  (augmentation table, macros and figure); all recompute these values from the same
  released arrays through canonical_prediction_sources.verified_selection and require
  agreement to 1e-12.

  Inputs: data/model-dist registry and, for each instance, evaluation/phantom and
  evaluation/simulated raw predictions and ground truth (pinned by the registry);
  benchmark_manifest.json (phantom truth check). Formerly produced by the paper's
  select_paper_predictions.py + select_adopted_yae_predictions.py +
  select_corrected_qnet_basis.py from working runs under data/model (now deleted) and
  paper-local canonical phantom folders; the current registry results are the same
  predictions.

  * phantom_metrics.csv    model, regime, metabolite, n=144, MAE, SD, negatives
  * simulated_metrics.csv  the same for the 100000-spectrum test set + checkpoint hash
  * phantom_errors.csv     per-spectrum, per-metabolite saved and evaluated values
  * simulated_rows.tex     legacy summary rows (see legacy_simulated_rows)
  * source_catalog.json    inventory and source folders of the eleven instances, with
                           the registry and result pins that verified_selection checks
  """
  manifest = {r["spectrum_id"]: r for r in load_manifest(out)["records"]}
  deployed = deployed_instances(code)
  folder = out / "selected"
  folder.mkdir(parents=True, exist_ok=True)
  phantom, simulated, detailed = [], [], []
  inventory, model_sources, phantom_sources = [], {}, {}
  inputs = {"code/" + REGISTRY: sha256(child(code, REGISTRY))}
  common = {}
  for model, regime in SELECTION_ORDER:
    entry = deployed[model, regime]
    run = child(code, entry["checkpoint_path"]).parent
    row = run_row(code, run, model, regime)
    if row["checkpoint"] != "model.keras" or row["training_target"] != "sum":
      raise ValueError("unexpected deployed checkpoint or target: " + row["run"])
    for domain in ("phantom", "simulated"):
      ids, prediction, truth = domain_arrays(code, entry, domain)
      if domain in common:
        validate_ids(ids, common[domain][0])
        a, b = truth, common[domain][1]
        if domain == "phantom":
          a, b = normalise(a, "truth")[0], normalise(b, "truth")[0]
        if not np.allclose(a, b, rtol=0, atol=1e-12):
          raise ValueError("deployment truth pairing differs")
      common[domain] = ids, truth
      records = metric_rows(row, prediction, truth)
      if domain == "phantom":
        known = [
          [
            {NAMES[k]: float(v) for k, v in manifest[str(s)]["concentrations_mM"].items()}.get(m, 0)
            for m in METABOLITES
          ]
          for s in ids
        ]
        if set(map(str, ids)) != set(manifest) or not np.allclose(
          normalise(truth, "truth")[0], normalise(known, "truth")[0], atol=1e-12, rtol=0
        ):
          raise ValueError("phantom truth differs from manifest")
        phantom.extend(records)
        detailed.extend(detailed_records(row["run"], [str(s) for s in ids], prediction, truth))
      else:
        simulated.extend(dict(r, checkpoint_sha256=entry["checkpoint_sha256"]) for r in records)
    inventory.append(row)
    model_sources[row["run"]] = entry["simulated_result"].removeprefix("code/")
    phantom_sources[row["run"]] = "code/" + entry["phantom_result"].removeprefix("code/")
    base = child(code, model_sources[row["run"]]).parent
    for name, digest in entry["result_sha256"].items():
      inputs["code/" + str(child(base, name).relative_to(code))] = digest
  write_csv(folder / "phantom_metrics.csv", phantom)
  write_csv(folder / "simulated_metrics.csv", simulated)
  write_csv(folder / "phantom_errors.csv", detailed)
  (folder / "simulated_rows.tex").write_text(legacy_simulated_rows(simulated), encoding="ascii")
  write_json(
    folder / "source_catalog.json",
    {
      "status": "eleven deployed instances read from the released model registry",
      "evaluation": "independent max; finite negatives retained",
      "deployment_scope": "model-dist",
      "inventory": sorted(inventory, key=lambda r: r["run"]),
      "model_sources": model_sources,
      "phantom_sources": phantom_sources,
      "input_sha256": inputs,
      "generator": GENERATOR + " (stage selected)",
    },
  )
  names = ["phantom_metrics.csv", "simulated_metrics.csv", "phantom_errors.csv", "simulated_rows.tex"]
  return ["selected/" + n for n in [*names, "source_catalog.json"]], {
    k.removeprefix("code/"): v for k, v in inputs.items()
  }


# ------------------------------------------------------------------ stage: audit
def stage_audit(code, out):
  """Write pipe-b-audit/inventory.csv and topk_sources.json.

  inventory.csv: the eleven instances as first audited, before the author adopted the
  DIFF/OFF/ON YAE pair. Nine rows are the registry runs in data/model-dist; the two YAE
  rows are the superseded OFF/ON YAE runs, released at the root of data/model-yae
  (their seeds are read from their mrsnet.json). Rows are sorted by the run path below
  the model repository root, as in the original audit. Paper consumers: every renderer
  that calls select_adopted_yae_predictions.selected_inventory, which requires the
  non-YAE rows of this file to equal the selected/source_catalog.json inventory and
  replaces only the two YAE identities; associate_magnitude_residuals.py and
  interval_selected_augmentation.py record its hash.

  topk_sources.json: benchmark MAEs of the five no-validation training seeds of each
  of the ten selection candidates (FAMILIES), recomputed from each seed's
  benchmark_all_max_quantify.csv in data/model-cnn, data/model-yae/caeq_noval and
  data/model-dist, with the saved truth checked against the benchmark metadata.
  Paper consumer: interval_selected_candidates.py (tables/topk_*_rows.tex), which
  recomputes every value and requires agreement to 1e-12.
  """
  folder = out / "pipe-b-audit"
  folder.mkdir(parents=True, exist_ok=True)
  deployed = deployed_instances(code)
  inputs = {REGISTRY: sha256(code / REGISTRY)}
  rows = []
  for model, regime in SELECTION_ORDER:
    if model == YAE:
      runs = [
        p.parent
        for p in sorted(
          code.glob(f"data/model-yae/{YAE}/Cr-GABA-Gln-Glu-NAA/megapress/*/*/sum/16/2000/*/NoValidation-1/mrsnet.json")
        )
        if (AUGMENTED in str(p)) == (regime == "augmented")
      ]
      if len(runs) != 1 or "/edit_off-edit_on/" not in str(runs[0]):
        raise ValueError("expected one superseded OFF/ON YAE run: " + regime)
      run = runs[0]
      for name in ("mrsnet.json", "model.keras"):
        inputs[str((run / name).relative_to(code))] = sha256(run / name)
    else:
      run = child(code, deployed[model, regime]["checkpoint_path"]).parent
      inputs[str((run / "mrsnet.json").relative_to(code))] = sha256(run / "mrsnet.json")
    rows.append(run_row(code, run, model, regime))
  rows.sort(key=lambda r: r["run"].split("/", 2)[2])
  write_csv(folder / "inventory.csv", rows)
  manifest_ids = {r["spectrum_id"] for r in load_manifest(out)["records"]}
  expected_truth = benchmark_truth(code)
  seed_sources = {}
  for family, candidates in FAMILIES.items():
    for model, acq, datatype in candidates:
      paths = []
      for root in SEED_ROOTS:
        paths.extend(
          (code / root).glob(
            f"{model}/Cr-GABA-Gln-Glu-NAA/megapress/{acq}/{datatype}/sum/16/2000/"
            "fid-a-2d_2000_4096_siemens_123.23_2.0_*100000-1/NoValidation-*/benchmark_all_max_quantify.csv"
          )
        )
      instances = set()
      for path in sorted(paths):
        instance = path.parent.name
        if instance in instances:
          raise ValueError(f"ambiguous seed instance: {model}/{instance}")
        instances.add(instance)
        ids, pred, true = quantify(path)
        if len(ids) != 144 or set(ids) != manifest_ids:
          raise ValueError(f"top-candidate sample set: {path}")
        for i, sample_id in enumerate(ids):
          series, sample = sample_key(sample_id)
          if not np.allclose(normalise([true[i]], "truth")[0], normalise([expected_truth[series][sample]], "truth")[0]):
            raise ValueError(f"top-candidate truth mismatch: {sample_id}")
        _, means = metrics(pred, true)
        seed_sources[str(path.relative_to(code))] = {
          "family": family,
          "model": model,
          "instance": instance,
          "mae": dict(zip(OUTCOMES, means, strict=True)),
        }
        inputs[str(path.relative_to(code))] = sha256(path)
      if len(instances) != 5:
        raise ValueError(f"expected 5 seed instances, found {len(instances)}: {model}")
  write_json(folder / "topk_sources.json", seed_sources)
  return ["pipe-b-audit/inventory.csv", "pipe-b-audit/topk_sources.json"], inputs


# ------------------------------------------------------------------ stage: descriptive
def selection_metrics(code, model, acq, datatype):
  """Mean over the five saved KFold selection folds, each verified against its saved summary."""
  pattern = (
    f"{model}/Cr-GABA-Gln-Glu-NAA/megapress/{acq}/{datatype}/sum/16/*/"
    "fid-a-2d_2000_4096_siemens_123.23_2.0_*_10000-1/KFold_*/fold-*/validation_concentration_errors.json"
  )
  files = sorted(p for folder in FOLD_ROOTS for p in (code / "data" / folder).glob(pattern))
  if len(files) != 5 or {p.parent.name for p in files} != {f"fold-{i}" for i in range(5)}:
    raise ValueError(f"expected exactly five unique selection folds: {model}/{acq}/{datatype}")
  values, sources = [], {}
  for path in files:
    data = json.loads(path.read_text())
    if data["norm"] != "max" or np.shape(data["true"]) != (2000, 5) or np.shape(data["predicted"]) != (2000, 5):
      raise ValueError(f"selection shape or saved evaluation mismatch: {path}")
    _, means = metrics(data["predicted"], data["true"])
    if not np.allclose(data["error"], np.asarray(data["predicted"]) - data["true"], atol=1e-10, rtol=1e-8):
      raise ValueError(f"selection signed-error mismatch: {path}")
    for j, m in enumerate(OUTCOMES):
      saved = data["total" if m == "Overall" else m]["abserror"]["mean"]
      if not np.isclose(means[j], saved, atol=1e-10, rtol=1e-8):
        raise ValueError(f"selection saved MAE mismatch: {path}/{m}")
    values.append(means)
    sources[str(path.relative_to(code))] = sha256(path)
  return dict(zip(OUTCOMES, np.mean(values, axis=0), strict=True)), sources


def verify_seed_pair(path):
  """Recompute a seed's MAEs and check its CSV against the companion JSON summary."""
  ids, pred, true = quantify(path)
  source = path.with_name("benchmark_all_max_concentration_errors.json")
  data = json.loads(source.read_text())
  if (
    len(ids) != 144
    or data["norm"] != "max"
    or not np.allclose(pred, data["predicted"])
    or not np.allclose(true, data["true"])
  ):
    raise ValueError(f"seed CSV/JSON source mismatch: {path}")
  _, means = metrics(pred, true)
  for j, m in enumerate(OUTCOMES):
    saved = data["total" if m == "Overall" else m]["abserror"]["mean"]
    if not np.isclose(means[j], saved, atol=1e-10, rtol=1e-8):
      raise ValueError(f"seed saved summary mismatch: {path}/{m}")
  return dict(zip(OUTCOMES, means, strict=True)), source


def stage_descriptive(code, out):
  """Write pipe-b-descriptive/descriptive_rank_inputs.csv and descriptive_input_sha256.json.

  descriptive_rank_inputs.csv: for each of the ten selection candidates and each
  outcome, the selection-stage simulation MAE (mean of five KFold validation folds,
  10000-spectrum datasets) and the phantom MAE (mean over the five seeds of
  topk_sources.json). descriptive_input_sha256.json: SHA-256 of every released seed
  CSV/JSON pair, seed mrsnet.json and selection fold read here.

  Paper consumer: interval_selected_candidates.py (tables/topk_*_rows.tex) checks the
  candidate point estimates against the CSV to 1e-12 and every source hash against
  the JSON (it pins the JSON's SHA-256). Formerly written by the paper's
  describe_pipe_b.py, which also wrote 48 "deployed" rows from the deleted
  data/model final-test runs and the historical phantom audit; no renderer reads
  those rows and they are not regenerated.
  """
  folder = out / "pipe-b-descriptive"
  folder.mkdir(parents=True, exist_ok=True)
  sources = json.loads((out / "pipe-b-audit/topk_sources.json").read_text())
  hashes, rows = {}, []
  for family, candidates in FAMILIES.items():
    for index, (model, acq, dtype) in enumerate(candidates, start=1):
      instances = {k: v for k, v in sources.items() if v["family"] == family and v["model"] == model}
      if len(instances) != 5:
        raise ValueError(f"incomplete five-seed candidate: {model}")
      seeds = set()
      for name, entry in instances.items():
        path = code / name
        values, companion = verify_seed_pair(path)
        if any(not np.isclose(values[m], entry["mae"][m], atol=1e-12, rtol=1e-12) for m in values):
          raise ValueError(f"stale audited seed metrics: {name}")
        meta_path = path.with_name("mrsnet.json")
        meta = json.loads(meta_path.read_text())
        if meta["model"] != model or meta["metabolites"] != METABOLITES or meta["norm"] != "sum":
          raise ValueError(f"candidate model metadata mismatch: {name}")
        if not isinstance(meta.get("seed"), int):
          raise ValueError(f"missing actual training seed: {name}")
        seeds.add(meta["seed"])
        for source in (path, companion, meta_path):
          hashes[str(source.relative_to(code))] = sha256(source)
      if len(seeds) != 5:
        raise ValueError(f"duplicate training seeds: {model}")
      sm, fold_hashes = selection_metrics(code, model, acq, dtype)
      hashes.update(fold_hashes)
      pm = {m: float(np.mean([v["mae"][m] for v in instances.values()])) for m in OUTCOMES}
      if not np.isclose(pm["Overall"], np.mean([pm[m] for m in METABOLITES]), atol=1e-12):
        raise ValueError(f"candidate Overall mismatch: {model}")
      for m in OUTCOMES:
        rows.append(
          {
            "family": family,
            "candidate": index,
            "model": model,
            "acquisition": acq,
            "datatype": dtype,
            "metabolite": m,
            "simulation_mae": sm[m],
            "phantom_mae": pm[m],
            "simulation_stage": "selection",
            "folds": 5,
            "seed_instances": 5,
          }
        )
  write_csv(folder / "descriptive_rank_inputs.csv", rows)
  write_json(
    folder / "descriptive_input_sha256.json",
    {
      "code_input_sha256": hashes,
      "inference": "none",
      "scope": "ten separate selection-stage candidates; deployed final-test rows not regenerated",
      "seed_sd_ddof": 1,
      "ties": "average ranks; Kendall tau-b",
      "weighting": "equal spectrum and equal metabolite; equal mean across training seeds",
    },
  )
  return [
    "pipe-b-descriptive/descriptive_rank_inputs.csv",
    "pipe-b-descriptive/descriptive_input_sha256.json",
  ], dict(hashes)


# ------------------------------------------------------------------ stage: composition
def stage_composition(code, out, source, run, python):
  """Package composition-matched/*: the composition-matched simulation control.

  Paper consumer: render_composition_matched.py (tables/composition_matched_rows.tex,
  constants/composition_matched_macros.tex), which recomputes all 90 MAEs from the raw
  arrays and checks them against metrics.csv to 1e-12.

  The control needs TensorFlow inference and is produced by the code-mrsnet script
  etc/composition_matched.py. This stage copies a finished result folder
  (--composition-matched DIR) after that script's own NumPy-only check
  (`etc/composition_matched.py --verify DIR`), or, with --run-composition-matched,
  first runs the script on the CPU (with --python) into a temporary folder
  (20-40 min). The files are copied byte for byte; nothing is recomputed here.
  """
  folder = out / "composition-matched"
  env = dict(os.environ)
  env["PYTHONPATH"] = os.pathsep.join([str(code), *filter(None, [env.get("PYTHONPATH")])])
  env.setdefault("CUDA_VISIBLE_DEVICES", "-1")
  env.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
  with tempfile.TemporaryDirectory(prefix="composition-matched-") as tmp:
    if run:
      source = Path(tmp) / "result"
      subprocess.run(  # noqa: S603
        [python, str(code / "etc/composition_matched.py"), "--code-root", str(code), "--outdir", str(source)],
        check=True,
        cwd=tmp,
        env=env,
      )
    source = Path(source).resolve()
    missing = [n for n in MATCHED_FILES if not (source / n).is_file()]
    if missing:
      raise ValueError("incomplete composition-matched result: " + ", ".join(missing))
    protocol = json.loads((source / "protocol.json").read_text())
    if protocol.get("status") != "complete" or protocol.get("models") != list(MATCHED_MODELS):
      raise ValueError("incomplete composition-matched protocol")
    subprocess.run(  # noqa: S603
      [python, str(code / "etc/composition_matched.py"), "--code-root", str(code), "--verify", str(source)],
      check=True,
      cwd=tmp,
      env=env,
      stdout=subprocess.DEVNULL,
    )
    folder.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in source.iterdir() if p.is_file())
    for name in names:
      shutil.copyfile(source / name, folder / name)
  return ["composition-matched/" + n for n in names], {
    "etc/composition_matched.py": sha256(code / "etc/composition_matched.py")
  }


# ------------------------------------------------------------------ stage: decomposition
def decomposition_linewidths(code, protocol):
  """Recompute the linewidth mapping of a decomposition protocol and return its basis linewidths.

  The complete basis linewidths (all ten ON/OFF files present) are listed from
  the checkout and must equal the protocol's grid; each spectrum's fitted
  linewidth in DECOMPOSITION_LINEWIDTHS is mapped to the nearest of them (ties
  to the smaller) and must give the protocol's mapping exactly.
  """
  record = protocol["linewidth"]
  need = {(m, a) for m in METABOLITES for a in ("ON", "OFF")}
  found = {}
  for path in child(code, BASIS_DIR).iterdir():
    match = BASIS_FILE.fullmatch(path.name)
    if match:
      found.setdefault(float(match[3]), set()).add((match[1], match[2]))
  grid = np.asarray(sorted(lw for lw, have in found.items() if need <= have))
  if grid.tolist() != record["available_complete_basis_linewidths_hz"]:
    raise ValueError("decomposition basis grid differs from the checkout")
  with child(code, DECOMPOSITION_LINEWIDTHS).open(newline="") as stream:
    rows = list(csv.DictReader(stream))
  fitted = np.asarray([float(r["linewidth_hz"]) for r in rows])
  if len(rows) != 144 or not np.isfinite(fitted).all():
    raise ValueError("unexpected fitted linewidth table")
  distance = np.abs(grid[None, :] - fitted[:, None])
  nearest = grid[np.argmax(distance <= distance.min(axis=1, keepdims=True) + 1e-9, axis=1)]
  expected = {
    r["spectrum_id"]: [f, float(b), float(b) - f, r["accepted"] == "True", r["rejection_reasons"]]
    for r, f, b in zip(rows, fitted.tolist(), nearest, strict=True)
  }
  saved = {
    r["spectrum_id"]: [
      r["fitted_linewidth_hz"],
      r["basis_linewidth_hz"],
      r["difference_hz"],
      r["fit_accepted"],
      r["fit_flags"],
    ]
    for r in record["mapping"]
  }
  if saved != expected or len(record["mapping"]) != 144:
    raise ValueError("decomposition linewidth mapping differs from its recomputation")
  if record["basis_linewidths_used"] != sorted(set(nearest.tolist())):
    raise ValueError("decomposition basis linewidths used differ from the mapping")
  return {sid: v[1] for sid, v in expected.items()}


def verify_decomposition(code, source):
  """Check a decomposition result folder against its pinned inputs and the control.

  Requires a complete protocol whose recorded SHA-256 keys are exactly the fixed
  set DECOMPOSITION_SOURCES plus the five registry checkpoints and metadata, the
  YAE phantom inputs and the ten basis files of every linewidth loaded (2 Hz and
  the recomputed mapping), each matching the checkout; the linewidth mapping
  recomputed from DECOMPOSITION_LINEWIDTHS and the checkout's basis grid
  (decomposition_linewidths); the A-baseline predictions and input hashes to
  equal the released control's; the fixed-sigma noise to have passed its mirror
  check against Dataset.add_noise over the calibrated sigma range; the
  per-spectrum basis linewidths and sigmas of the arrays to equal the protocol's,
  and the protocol's sigma summaries and counts to equal their recomputation;
  and every MAE, baseline MAE, phantom MAE and gap fraction of metrics.csv to
  equal the values recomputed from the raw arrays (1e-12). Returns the hashes of
  the control files that the check read.

  The B, C and D predictions themselves cannot be verified without inference:
  this check covers their inputs' provenance and the consistency of metrics.csv
  with the saved predictions, not that the models produce those predictions.
  Only a rerun (--run-decomposition) does that.
  """
  protocol = json.loads((source / "protocol.json").read_text())
  if (
    protocol.get("status") != "complete"
    or protocol.get("models") != list(MATCHED_MODELS)
    or list(protocol.get("variants", {})) != list(DECOMPOSITION_VARIANTS)
    or protocol.get("spectra") != 144
    or protocol.get("reference") != DECOMPOSITION_REFERENCE
  ):
    raise ValueError("incomplete composition-decomposition protocol")
  if protocol["runtime"]["threads"] != DECOMPOSITION_THREADS:
    raise ValueError("decomposition not run with the recorded thread settings")
  linewidth = decomposition_linewidths(code, protocol)
  entries = {
    e["paper_name"]: e
    for e in json.loads(child(code, REGISTRY).read_text())["instances"]
    if e["regime"] == "fixed" and e["paper_name"] in MATCHED_MODELS
  }
  if set(entries) != set(MATCHED_MODELS):
    raise ValueError("principal fixed models missing from the registry")
  required = set(DECOMPOSITION_SOURCES)
  for entry in entries.values():
    required |= {str(child(code, entry[k + "_path"]).relative_to(code)) for k in ("checkpoint", "metadata")}
  phantom = child(code, entries["YAE"]["phantom_result"].removeprefix("code/")) / "inputs.npz"
  required.add(str(phantom.relative_to(code)))
  loaded = sorted({2.0, *linewidth.values()})
  required |= {
    f"{BASIS_DIR}FIDA2D_{m}_MEGAPRESS_EDIT{a}_{lw:.2f}_2000_4096_123.23.mat"
    for lw in loaded
    for m in METABOLITES
    for a in ("OFF", "ON")
  }
  if set(protocol["source_sha256"]) != required:
    raise ValueError("decomposition source hashes are not the required set")
  for name, digest in protocol["source_sha256"].items():
    if sha256(child(code, name)) != digest:
      raise ValueError("decomposition input changed: " + name)
  gate = protocol["baseline_gate"]
  if set(gate) != set(MATCHED_MODELS) or not all(
    g["ids_equal"] and g["input_hashes_equal"] and g["max_abs_mae_difference"] <= 1e-12 for g in gate.values()
  ):
    raise ValueError("decomposition baseline gate failed")
  reference = child(code, DECOMPOSITION_REFERENCE)
  control = json.loads((reference / "protocol.json").read_text())
  seeds = protocol["noise_seeds"]
  if seeds != control["noise_seeds"] or protocol["noise_repeats"] != len(seeds):
    raise ValueError("decomposition noise seeds differ from the control")
  noise_record = protocol["noise"]
  mirror = {(c["basis"], c["sigma_draw"][1]): c for c in noise_record["fixed_noise_mirror_check"]}
  if set(mirror) != {(b, m) for b in ("2Hz", "matched") for m in (0.03, 0.2)} or not all(
    c["exports_identical"] is True and c["seeds"] == seeds and c["realisations"] == 144 * len(seeds)
    for c in mirror.values()
  ):
    raise ValueError("decomposition fixed-sigma noise not mirror-checked")
  sigmas = {}
  for variant, basis in (("C-noise", "2Hz"), ("D-linewidth-noise", "matched")):
    values = np.asarray([r[variant + "_sigma"] for r in noise_record["per_spectrum"]])
    record = noise_record[variant]
    summary = [values.min(), np.median(values), values.mean(), values.max()]
    if (
      len(values) != 144
      or not np.isfinite(values).all()
      or (values < 0).any()
      or not np.allclose(summary, [record["sigma"][k] for k in ("min", "median", "mean", "max")], rtol=0, atol=1e-15)
      or int(np.sum(values > 0.03)) != record["sigma_above_training_max_0.03"]
      or int(np.sum(values == 0)) != record["sigma_zero_noiseless_level_exceeds_phantom"]
      or values.max() > mirror[basis, 0.2]["drawn_sigma_max"]
    ):
      raise ValueError("decomposition sigma record inconsistent: " + variant)
    sigmas[variant] = dict(zip([r["spectrum_id"] for r in noise_record["per_spectrum"]], values.tolist(), strict=True))
  inputs = {}
  for name in ("protocol.json", "metrics.csv", *(m + ".npz" for m in MATCHED_MODELS)):
    inputs[str((reference / name).relative_to(code))] = sha256(reference / name)
  with (reference / "metrics.csv").open(newline="") as stream:
    released = {(r["model"], r["domain"], r["metabolite"]): float(r["mae"]) for r in csv.DictReader(stream)}
  arrays = {}
  for variant in DECOMPOSITION_VARIANTS:
    with np.load(source / (variant + ".npz"), allow_pickle=False) as saved:
      arrays[variant] = {k: saved[k] for k in saved.files}
  computed = {}
  for model in MATCHED_MODELS:
    with np.load(reference / (model + ".npz"), allow_pickle=False) as saved:
      ref = {k: saved[k] for k in ("id", "truth", "prediction_raw", "input_sha256")}
    ids = ref["id"].tolist()
    noisy_truth = np.tile(ref["truth"], (len(seeds), 1))
    for variant in DECOMPOSITION_VARIANTS:
      saved = arrays[variant]
      prediction = saved["prediction_raw_" + model]
      matched = variant in ("B-linewidth", "D-linewidth-noise")
      if (
        saved["id"].tolist() != ids
        or not np.array_equal(saved["truth"], ref["truth"])
        or saved["noise_seeds"].tolist() != seeds
        or prediction.shape != (len(seeds) + 1, 144, 5)
        or saved["basis_linewidth_hz"].tolist() != [linewidth[i] if matched else 2.0 for i in ids]
      ):
        raise ValueError(f"decomposition arrays differ from protocol or control: {variant}/{model}")
      if variant in ("C-noise", "D-linewidth-noise"):
        sigma = [sigmas[variant][i] for i in ids]
        if saved["noise_sigma"].tolist() != sigma or not np.isnan(prediction[0]).all():
          raise ValueError("decomposition noise record differs: " + variant)
        domains = {}
      else:
        domains = {"noiseless": metrics(prediction[0], ref["truth"])[1]}
      domains["matched-noisy"] = metrics(prediction[1:].reshape(-1, 5), noisy_truth)[1]
      if variant == "A-baseline" and not (
        np.array_equal(prediction, ref["prediction_raw"])
        and saved["input_sha256_" + model].tolist() == ref["input_sha256"].tolist()
      ):
        raise ValueError("A-baseline differs from the released control: " + model)
      for domain, values in domains.items():
        for m, value in zip(OUTCOMES, values, strict=True):
          computed[model, variant, domain, m] = value
          if variant == "A-baseline" and abs(value - released[model, domain, m]) > 1e-12:
            raise ValueError(f"A-baseline MAE differs from the control: {model}/{domain}/{m}")
  with (source / "metrics.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))
  keys = [(r["model"], r["variant"], r["domain"], r["metabolite"]) for r in rows]
  if len(keys) != len(computed) or set(keys) != set(computed):
    raise ValueError("decomposition metrics.csv rows differ from the raw arrays")
  for key, row in zip(keys, rows, strict=True):
    base = computed[key[0], "A-baseline", "matched-noisy", key[3]]
    phantom = released[key[0], "phantom-cpu", key[3]]
    expected = [computed[key], base, phantom]
    values = [float(row[k]) for k in ("mae", "mae_baseline_matched_noisy", "mae_phantom_cpu")]
    if key[2] == "matched-noisy":
      expected.append((computed[key] - base) / (phantom - base))
      values.append(float(row["gap_fraction_closed"]))
    elif row["gap_fraction_closed"] != "":
      raise ValueError("gap fraction given for a noiseless row")
    if not np.allclose(values, expected, rtol=0, atol=1e-12):
      raise ValueError("decomposition metric differs from the raw arrays: " + "/".join(key))
  return inputs


def compare_decomposition_metrics(released, rerun):
  """Print how a rerun's metrics.csv differs from the released one, per variant and row."""
  tables = []
  for path in (released, rerun):
    with path.open(newline="") as stream:
      tables.append({(r["model"], r["variant"], r["domain"], r["metabolite"]): r for r in csv.DictReader(stream)})
  old, new = tables
  if set(old) != set(new):
    print("decomposition rerun: metrics.csv rows differ from the released ones")
  columns = ("mae", "gap_fraction_closed")
  for variant in DECOMPOSITION_VARIANTS:
    keys = sorted(k for k in set(old) & set(new) if k[1] == variant)
    changed = [k for k in keys if any(old[k][c] != new[k][c] for c in columns)]
    largest = max(
      (abs(float(new[k][c]) - float(old[k][c])) for k in keys for c in columns if old[k][c] != ""), default=0.0
    )
    print(f"decomposition rerun {variant}: {len(changed)} of {len(keys)} rows differ, max abs difference {largest:.3g}")
    for k in changed:
      print("  " + "/".join(k), *(f"{c} {old[k][c]} -> {new[k][c]}" for c in columns if old[k][c] != new[k][c]))


def stage_decomposition(code, out, source, run, python):
  """Package composition-decomposition/*: linewidth and noise parts of the composition-matched gap.

  Paper consumer: render_composition_decomposition.py
  (tables/composition_decomposition_rows.tex,
  constants/composition_decomposition_macros.tex), which renders the gap fractions
  closed of metrics.csv and counts from protocol.json.

  The decomposition needs TensorFlow inference and is produced by the code-mrsnet
  script etc/composition_matched_decomposition.py. This stage verifies a finished
  result folder (--decomposition DIR; default: the released DECOMPOSITION folder of
  the checkout, so a rerun into the default --outdir leaves it unchanged) with
  verify_decomposition (NumPy only) and copies its files byte for byte, or, with
  --run-decomposition, first runs the script on the CPU (with --python and the
  thread settings DECOMPOSITION_THREADS) into a temporary folder (about 6 min on
  64 cores), verifies it, and prints how its metrics differ from the released
  DECOMPOSITION folder before the files replace those below --outdir. Nothing is
  recomputed into the output. The B, C and D predictions are not checked without
  such a rerun (see verify_decomposition).
  """
  folder = out / "composition-decomposition"
  env = dict(os.environ)
  env["PYTHONPATH"] = os.pathsep.join([str(code), *filter(None, [env.get("PYTHONPATH")])])
  env.update(DECOMPOSITION_THREADS)
  env.setdefault("CUDA_VISIBLE_DEVICES", "-1")
  env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
  env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
  env.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
  with tempfile.TemporaryDirectory(prefix="composition-decomposition-") as tmp:
    if run:
      source = Path(tmp) / "result"
      subprocess.run(  # noqa: S603
        [
          python,
          str(code / "etc/composition_matched_decomposition.py"),
          "--code-root",
          str(code),
          "--outdir",
          str(source),
        ],
        check=True,
        cwd=tmp,
        env=env,
      )
    source = Path(source or code / DECOMPOSITION).resolve()
    missing = [n for n in DECOMPOSITION_FILES if not (source / n).is_file()]
    if missing:
      raise ValueError("incomplete composition-decomposition result: " + ", ".join(missing))
    inputs = verify_decomposition(code, source)
    released = code / DECOMPOSITION / "metrics.csv"
    if run and released.is_file():
      compare_decomposition_metrics(released, source / "metrics.csv")
    folder.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in source.iterdir() if p.is_file())
    if source != folder.resolve():
      for name in names:
        shutil.copyfile(source / name, folder / name)
  inputs["etc/composition_matched_decomposition.py"] = sha256(code / "etc/composition_matched_decomposition.py")
  return ["composition-decomposition/" + n for n in names], inputs


# ------------------------------------------------------------------ stage: gains
def verify_gains(code, folder):
  """Check a metabolite-gains result folder against its protocol and the checkout.

  Requires all GAINS_FILES, a complete protocol for the 144 spectra whose gate 1
  passed and whose recorded thread settings are GAINS_THREADS, output hashes equal
  to the files, and recorded source hashes that are exactly GAINS_SOURCES plus the
  ten 1.0 Hz basis files, each equal to the checkout. Returns the source hashes.
  """
  missing = [n for n in GAINS_FILES if not (folder / n).is_file()]
  if missing:
    raise ValueError("incomplete metabolite-gains result: " + ", ".join(missing))
  protocol = json.loads((folder / "protocol.json").read_text())
  if protocol.get("status") != "complete" or protocol.get("spectra") != 144 or not protocol["gate1"]["passed"]:
    raise ValueError("incomplete metabolite-gains protocol")
  if protocol["runtime"]["threads"] != GAINS_THREADS:
    raise ValueError("gain fit not run with the recorded thread settings")
  if protocol["output_sha256"] != {n: sha256(folder / n) for n in GAINS_REPRODUCED}:
    raise ValueError("metabolite-gains files differ from their protocol")
  required = set(GAINS_SOURCES) | {
    f"{BASIS_DIR}FIDA2D_{m}_MEGAPRESS_EDIT{a}_1.00_2000_4096_123.23.mat" for m in METABOLITES for a in ("OFF", "ON")
  }
  if set(protocol["source_sha256"]) != required:
    raise ValueError("metabolite-gains source hashes are not the required set")
  for name, digest in protocol["source_sha256"].items():
    if sha256(child(code, name)) != digest:
      raise ValueError("metabolite-gains input changed: " + name)
  return dict(protocol["source_sha256"])


def stage_gains(code, out, python, python_path):
  """Rerun etc/metabolite_gain_fit.py and package metabolite-gains/*: per-metabolite signal gains.

  Paper consumer: render_metabolite_gains.py (tables/metabolite_gains_rows.tex,
  constants/metabolite_gains_macros.tex), which renders the OFF Cr, GABA and Glu gains
  that gate 2 recovers and checks that recovery.

  The fit needs the mrsnet library (DICOM loading, basis) and SciPy, not
  TensorFlow. This stage always reruns it into a temporary folder with
  --gains-python (default --python) and PYTHONPATH = the checkout plus
  --gains-python-path, under GAINS_THREADS, and checks the rerun with
  verify_gains. If the released GAINS folder exists it is verified the same way
  and every file but protocol.json (which records the runtime) must be byte
  identical to the rerun's; the released files are then copied unchanged below
  --outdir. Without a released folder the rerun's files are copied.
  """
  folder = out / "metabolite-gains"
  released = code / GAINS
  env = dict(os.environ)
  env["PYTHONPATH"] = os.pathsep.join([str(code), *filter(None, [python_path, env.get("PYTHONPATH")])])
  env.update(GAINS_THREADS)
  env.setdefault("CUDA_VISIBLE_DEVICES", "-1")
  env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
  with tempfile.TemporaryDirectory(prefix="metabolite-gains-") as tmp:
    rerun = Path(tmp) / "result"
    subprocess.run(  # noqa: S603
      [python, str(code / "etc/metabolite_gain_fit.py"), "--code-root", str(code), "--outdir", str(rerun)],
      check=True,
      cwd=tmp,
      env=env,
      stdout=subprocess.DEVNULL,
    )
    inputs = verify_gains(code, rerun)
    source = rerun
    if released.is_dir():
      if verify_gains(code, released) != inputs:
        raise ValueError("released metabolite-gains sources differ from the rerun's")
      differ = [n for n in GAINS_REPRODUCED if (released / n).read_bytes() != (rerun / n).read_bytes()]
      if differ:
        raise ValueError("metabolite-gains rerun differs from the released files: " + ", ".join(differ))
      source = released.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in source.iterdir() if p.is_file())
    if source != folder.resolve():
      for name in names:
        shutil.copyfile(source / name, folder / name)
  return ["metabolite-gains/" + n for n in names], inputs


# ------------------------------------------------------------------ stage: cnn
def stage_cnn(code, out):
  """Write cnn-selection/simple_grid_ranked.csv: the ranked simple-CNN grid search.

  Paper consumer: figures/results/cnn-simple-all-grid-100-5.0-1_top_25.pdf (drawn by the
  paper's scripts/cnn_model_selection/vis_selection.py from the grid run's
  model_performance.csv, which is not released). The released aggregate of the same 576
  five-fold runs, data/model-cnn/aggregate-simple/all_results.csv, reproduces all 25
  rows of that figure (configuration, validation and training MAE mean +/- SD to the
  printed five digits). This stage only ranks it; no selection run is repeated.

  Columns: rank by mean validation MAE (stable for ties), the configuration, the five
  validation and training fold MAEs, and their means and population SDs (ddof 0, as in
  the figure).
  """
  source = code / "data/model-cnn/aggregate-simple/all_results.csv"
  with source.open(newline="") as stream:
    rows = list(csv.DictReader(stream))
  config = ("model", "trainer", "dataset", "metabolites", "pulse", "acq", "datatype", "norm", "batch", "epochs")
  if len(rows) != 576 or {r["trainer"] for r in rows} != {"KFold_5-1"} or {r["epochs"] for r in rows} != {"100"}:
    raise ValueError("unexpected simple-CNN grid aggregate")
  ranked = []
  for row in rows:
    val = [float(row[f"val_mae_fold{i}"]) for i in range(5)]
    train = [float(row[f"train_mae_fold{i}"]) for i in range(5)]
    if not np.isclose(np.mean(val), float(row["val_mae_mean"]), atol=1e-12, rtol=0):
      raise ValueError("aggregate validation mean differs from its folds")
    ranked.append(
      {
        **{k: row[k] for k in config},
        **{f"val_mae_fold{i}": v for i, v in enumerate(val)},
        **{f"train_mae_fold{i}": v for i, v in enumerate(train)},
        "val_mae_mean": float(np.mean(val)),
        "val_mae_sd": float(np.std(val)),
        "train_mae_mean": float(np.mean(train)),
        "train_mae_sd": float(np.std(train)),
      }
    )
  ranked.sort(key=lambda r: r["val_mae_mean"])
  folder = out / "cnn-selection"
  folder.mkdir(parents=True, exist_ok=True)
  write_csv(folder / "simple_grid_ranked.csv", [{"rank": i + 1, **r} for i, r in enumerate(ranked)])
  return ["cnn-selection/simple_grid_ranked.csv"], {str(source.relative_to(code)): sha256(source)}


# ------------------------------------------------------------------ main
def revision(path):
  """Git revision of a released checkout, or None."""
  try:
    return subprocess.run(  # noqa: S603
      ["git", "-C", str(path), "rev-parse", "HEAD"],  # noqa: S607
      check=True,
      capture_output=True,
      text=True,
    ).stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    return None


def main():
  """Run the requested stages and record their provenance."""
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--code-root", type=Path, default=CODE_ROOT, help="MRSNet checkout with its data repositories")
  parser.add_argument(
    "--outdir", type=Path, default=None, help="output folder (default: CODE_ROOT/data/sim2real/paper-gaba-phantom)"
  )
  parser.add_argument(
    "--stage",
    action="append",
    choices=STAGES,
    help="stage to run (repeatable); default: all, with composition only if a source is given and "
    "decomposition only if its source exists (gains always runs: it reruns etc/metabolite_gain_fit.py)",
  )
  parser.add_argument("--composition-matched", type=Path, help="finished etc/composition_matched.py result folder")
  parser.add_argument(
    "--run-composition-matched", action="store_true", help="run etc/composition_matched.py inference (CPU, slow)"
  )
  parser.add_argument(
    "--decomposition",
    type=Path,
    help="finished etc/composition_matched_decomposition.py result folder (default: CODE_ROOT/" + DECOMPOSITION + ")",
  )
  parser.add_argument(
    "--run-decomposition",
    action="store_true",
    help="run etc/composition_matched_decomposition.py inference (CPU, slow)",
  )
  parser.add_argument(
    "--python", default=sys.executable, help="interpreter with TensorFlow for the composition and decomposition stages"
  )
  parser.add_argument(
    "--gains-python", help="interpreter with the mrsnet dependencies for stage gains (default --python)"
  )
  parser.add_argument(
    "--gains-python-path", help="extra PYTHONPATH entries (after the checkout) for --gains-python in stage gains"
  )
  args = parser.parse_args()
  code = args.code_root.resolve()
  out = (args.outdir or code / "data/sim2real/paper-gaba-phantom").resolve()
  if not (code / REGISTRY).is_file():
    raise SystemExit("not an MRSNet checkout with data/model-dist: " + str(code))
  matched = args.composition_matched is not None or args.run_composition_matched
  decomposed = args.run_decomposition or (args.decomposition or code / DECOMPOSITION).is_dir()
  stages = args.stage or [
    "manifest",
    "selected",
    "audit",
    "descriptive",
    "cnn",
    *(["composition"] if matched else []),
    *(["decomposition"] if decomposed else []),
    "gains",
  ]
  if "composition" in stages and not matched:
    raise SystemExit("stage composition needs --composition-matched DIR or --run-composition-matched")
  if "decomposition" in stages and not decomposed:
    raise SystemExit("stage decomposition needs a result folder (--decomposition DIR) or --run-decomposition")
  out.mkdir(parents=True, exist_ok=True)
  record_path = out / "provenance.json"
  record = json.loads(record_path.read_text()) if record_path.is_file() else {}
  record.update(
    generator=GENERATOR,
    generator_sha256=sha256(Path(__file__)),
    numpy=np.__version__,
  )
  record.setdefault("stages", {})
  runners = {
    "manifest": lambda: stage_manifest(code, out),
    "selected": lambda: stage_selected(code, out),
    "audit": lambda: stage_audit(code, out),
    "descriptive": lambda: stage_descriptive(code, out),
    "cnn": lambda: stage_cnn(code, out),
    "composition": lambda: stage_composition(
      code, out, args.composition_matched, args.run_composition_matched, args.python
    ),
    "decomposition": lambda: stage_decomposition(code, out, args.decomposition, args.run_decomposition, args.python),
    "gains": lambda: stage_gains(code, out, args.gains_python or args.python, args.gains_python_path),
  }
  repositories = ("benchmark", "model-dist", "model-yae", "model-cnn", "model-extra")
  for stage in STAGES:
    if stage not in stages:
      continue
    outputs, inputs = runners[stage]()
    record["stages"][stage] = {
      "generator_sha256": record["generator_sha256"],
      "revisions": {name: revision(code / "data" / name) for name in repositories} | {"code-mrsnet": revision(code)},
      "input_sha256": dict(sorted(inputs.items())),
      "output_sha256": {name: sha256(out / name) for name in outputs},
    }
    print(f"stage {stage}: wrote {len(outputs)} files")
  write_json(record_path, record)


if __name__ == "__main__":
  main()
