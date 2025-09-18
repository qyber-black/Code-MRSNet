#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (C) 2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Aggregate CNN model results across data/model-cnn into a single dataset.

- Recursively scans for trained CNN models under a root directory (default: ./data/model-cnn)
- Handles KFold, DuplexKFold, Split, DuplexSplit, and NoValidation trainers
- Deduplicates runs using realpath of trainer folders (to avoid symlink duplicates)
- Parses per-fold train/validation concentration error JSONs written by analyse.py
- Computes validation MAE summary, overfitting gap, generalization ratio, and stability metrics
- Extracts model hyperparameters from model name (including canonicalizing cnn_small/medium/large)
- Writes both CSV and JSON aggregate files for downstream analysis
- Optionally filters by dataset name substring and by linewidth substring
- Deduplicates repeated runs per signature by best Val_Mean
- Computes per-architecture win counts across contexts

Usage:
  python3 etc/aggregate_cnn_results.py \
    --root ./data/model-cnn \
    --out ./data/model-cnn/aggregate/fid-a-2d_2000_4096_2.0 \
    --dataset-filter fid-a-2d_2000_4096 \
    --linewidth-filter 2.0

Outputs:
  .../aggregate/fid-a-2d_2000_4096_2.0/cnn_models_aggregate.csv
  .../aggregate/fid-a-2d_2000_4096_2.0/cnn_models_aggregate.json
  .../aggregate/fid-a-2d_2000_4096_2.0/cnn_models_aggregate_dedup.csv
  .../aggregate/fid-a-2d_2000_4096_2.0/cnn_models_aggregate_dedup.json
  .../aggregate/fid-a-2d_2000_4096_2.0/cnn_best_per_context.csv
  .../aggregate/fid-a-2d_2000_4096_2.0/cnn_best_per_context.json
  .../aggregate/fid-a-2d_2000_4096_2.0/cnn_architecture_stats.json
  .../aggregate/fid-a-2d_2000_4096_2.0/cnn_contexts_overview.csv
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev

TRAINER_PREFIXES = (
  "KFold_",
  "DuplexKFold_",
  "Split_",
  "DuplexSplit_",
  "NoValidation",
)

FOLD_PREFIX = "fold-"

# Regex to capture trainer kind, parameter, and repeat id
# Examples: KFold_5-1, Split_0.8-3, DuplexKFold_5-2, NoValidation-1
TRAINER_RE = re.compile(
  r"^(?P<kind>KFold|DuplexKFold|Split|DuplexSplit|NoValidation)(?:_(?P<param>[^-]+))?(?:-(?P<repeat>\d+))?$"
)

# cnn_small/medium/large fixed mappings
CNN_CANONICAL_MAP = {
  "small": {
    "S1": 2,
    "S2": 3,
    "C1": 7,
    "C2": 5,
    "C3": 3,
    "C4": 3,
    "O1": 0.0,
    "O2": 0.3,
    "F1": 256,
    "F2": 512,
    "D": 1024,
  },
  "medium": {
    "S1": 2,
    "S2": 3,
    "C1": 9,
    "C2": 7,
    "C3": 5,
    "C4": 3,
    "O1": 0.0,
    "O2": 0.3,
    "F1": 256,
    "F2": 512,
    "D": 1024,
  },
  "large": {
    "S1": 2,
    "S2": 3,
    "C1": 11,
    "C2": 9,
    "C3": 7,
    "C4": 3,
    "O1": 0.0,
    "O2": 0.3,
    "F1": 256,
    "F2": 512,
    "D": 1024,
  },
}


@dataclass
class ModelMeta:
  model_name: str
  arch_key: str
  metabolites: list[str]
  pulse_sequence: str
  acquisitions: list[str]
  datatype: list[str]
  norm: str
  batchsize: int | None
  epochs: int | None
  dataset_name: str
  trainer_kind: str
  trainer_param: str | None
  trainer_repeat: int | None
  n_folds: int
  path: str
  realpath: str
  total_params: int | None
  trainable_params: int | None
  non_trainable_params: int | None
  flops: int | None


@dataclass
class Metrics:
  val_mean: float | None
  val_std: float | None
  val_min: float | None
  val_max: float | None
  val_median: float | None
  val_cv: float | None
  fold_range: float | None
  train_mean: float | None
  train_std: float | None
  overfitting_gap: float | None
  generalization_ratio: float | None
  val_folds: list[float]
  train_folds: list[float]


@dataclass
class AggregateRow:
  # Flattened fields for CSV
  model_name: str
  arch_key: str
  # Parsed numeric hyperparameters (if available)
  model_S1: float | None
  model_S2: float | None
  model_C1: int | None
  model_C2: int | None
  model_C3: int | None
  model_C4: int | None
  model_O1: float | None
  model_O2: float | None
  model_F1: int | None
  model_F2: int | None
  model_D: int | None
  model_ACTIVATION: str | None

  metabolites: str
  pulse_sequence: str
  acquisitions: str
  datatype: str
  norm: str
  batchsize: int | None
  epochs: int | None
  dataset_name: str
  trainer_kind: str
  trainer_param: str | None
  trainer_repeat: int | None
  n_folds: int
  path: str
  realpath: str
  total_params: int | None
  trainable_params: int | None
  non_trainable_params: int | None
  flops: int | None

  # Metrics
  val_mean: float | None
  val_std: float | None
  val_min: float | None
  val_max: float | None
  val_median: float | None
  val_cv: float | None
  fold_range: float | None
  train_mean: float | None
  train_std: float | None
  overfitting_gap: float | None
  generalization_ratio: float | None
  val_folds: str
  train_folds: str


def parse_trainer_info(trainer_dir: Path) -> tuple[str, str | None, int | None]:
  m = TRAINER_RE.match(trainer_dir.name)
  if not m:
    return trainer_dir.name, None, None
  kind = m.group("kind") or trainer_dir.name
  param = m.group("param")
  repeat = m.group("repeat")
  return kind, param, int(repeat) if repeat is not None else None


def parse_batch_epochs_dataset(trainer_dir: Path) -> tuple[int | None, int | None, str]:
  # Expected: .../<model path>/<batchsize>/<epochs>/<dataset>/<trainer>
  try:
    dataset = trainer_dir.parent.name
    epochs = int(trainer_dir.parent.parent.name)
    batchsize = int(trainer_dir.parent.parent.parent.name)
    return batchsize, epochs, dataset
  except Exception:
    return None, None, trainer_dir.parent.name


def load_json(path: Path) -> dict | None:
  try:
    with open(path) as f:
      return json.load(f)
  except Exception:
    return None


def parse_mrsnet_json(folder: Path) -> dict | None:
  return load_json(folder / "mrsnet.json")


def parse_error_json(folder: Path, prefix: str) -> dict | None:
  # prefix: 'train' or 'validation'
  return load_json(folder / f"{prefix}_concentration_errors.json")


def safe_stats(
  values: list[float],
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
  if not values:
    return None, None, None, None, None
  try:
    vmin = min(values)
    vmax = max(values)
    vmean = mean(values)
    vstd = pstdev(values) if len(values) > 1 else 0.0
    # median without numpy
    s = sorted(values)
    mid = len(s) // 2
    vmed = s[mid] if len(s) % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0
    return vmean, vstd, vmin, vmax, vmed
  except Exception:
    return None, None, None, None, None


def canonicalize_cnn_model(model_name: str) -> tuple[str, dict[str, float | None]]:
  """Return (arch_key, params) where arch_key is the numeric canonical name.

  For cnn_small/medium/large_[act][_pool], convert to numeric cnn_{S1}_{S2}_{C1}_{C2}_{C3}_{C4}_{O1}_{O2}_{F1}_{F2}_{D}_{ACT}
  (with S1/S2 negative if 'pool' is present). If already numeric, return as-is and parse values.
  """
  parts = model_name.split("_")
  params = {
    "S1": None,
    "S2": None,
    "C1": None,
    "C2": None,
    "C3": None,
    "C4": None,
    "O1": None,
    "O2": None,
    "F1": None,
    "F2": None,
    "D": None,
    "ACTIVATION": None,
  }
  if len(parts) >= 3 and parts[0] == "cnn" and parts[1] in ("small", "medium", "large"):
    size = parts[1]
    activation = parts[2]
    pool = len(parts) >= 4 and parts[3] == "pool"
    base = CNN_CANONICAL_MAP[size]
    s1 = -base["S1"] if pool else base["S1"]
    s2 = -base["S2"] if pool else base["S2"]
    arch_key = f"cnn_{s1}_{s2}_{base['C1']}_{base['C2']}_{base['C3']}_{base['C4']}_{base['O1']}_{base['O2']}_{base['F1']}_{base['F2']}_{base['D']}_{activation}"
    params.update(
      {
        "S1": float(s1),
        "S2": float(s2),
        "C1": base["C1"],
        "C2": base["C2"],
        "C3": base["C3"],
        "C4": base["C4"],
        "O1": float(base["O1"]),
        "O2": float(base["O2"]),
        "F1": base["F1"],
        "F2": base["F2"],
        "D": base["D"],
        "ACTIVATION": activation,
      }
    )
    return arch_key, params
  # Numeric form expected: cnn_S1_S2_C1_C2_C3_C4_O1_O2_F1_F2_D_ACT
  if parts[0] == "cnn" and len(parts) == 13:
    try:
      params.update(
        {
          "S1": float(parts[1]),
          "S2": float(parts[2]),
          "C1": int(parts[3]),
          "C2": int(parts[4]),
          "C3": int(parts[5]),
          "C4": int(parts[6]),
          "O1": float(parts[7]),
          "O2": float(parts[8]),
          "F1": int(parts[9]),
          "F2": int(parts[10]),
          "D": int(parts[11]),
          "ACTIVATION": parts[12],
        }
      )
      return model_name, params
    except Exception:
      return model_name, params
  # Unknown/other forms (keep as-is)
  return model_name, params


def collect_metrics_from_kfold(trainer_dir: Path) -> Metrics:
  val_folds: list[float] = []
  train_folds: list[float] = []
  # Read each fold's error files
  for fold_dir in sorted(trainer_dir.glob(f"{FOLD_PREFIX}*")):
    vj = parse_error_json(fold_dir, "validation")
    tj = parse_error_json(fold_dir, "train")
    if vj and "total" in vj and "abserror" in vj["total"] and "mean" in vj["total"]["abserror"]:
      val_folds.append(float(vj["total"]["abserror"]["mean"]))
    if tj and "total" in tj and "abserror" in tj["total"] and "mean" in tj["total"]["abserror"]:
      train_folds.append(float(tj["total"]["abserror"]["mean"]))
  # Compute summary
  vmean, vstd, vmin, vmax, vmed = safe_stats(val_folds)
  tmean, tstd, _, _, _ = safe_stats(train_folds)
  val_cv = (vstd / vmean) if (vstd is not None and vmean and vmean > 0) else None
  fold_range = (vmax - vmin) if (vmax is not None and vmin is not None) else None
  overfit = (tmean - vmean) if (tmean is not None and vmean is not None) else None
  gen_ratio = (vmean / tmean) if (tmean is not None and tmean > 0 and vmean is not None) else None
  return Metrics(
    val_mean=vmean,
    val_std=vstd,
    val_min=vmin,
    val_max=vmax,
    val_median=vmed,
    val_cv=val_cv,
    fold_range=fold_range,
    train_mean=tmean,
    train_std=tstd,
    overfitting_gap=overfit,
    generalization_ratio=gen_ratio,
    val_folds=val_folds,
    train_folds=train_folds,
  )


def collect_metrics_from_split(trainer_dir: Path) -> Metrics:
  # Single-fold like metrics
  vj = parse_error_json(trainer_dir, "validation")
  tj = parse_error_json(trainer_dir, "train")
  val_folds: list[float] = []
  train_folds: list[float] = []
  if vj and "total" in vj and "abserror" in vj["total"] and "mean" in vj["total"]["abserror"]:
    val_folds.append(float(vj["total"]["abserror"]["mean"]))
  if tj and "total" in tj and "abserror" in tj["total"] and "mean" in tj["total"]["abserror"]:
    train_folds.append(float(tj["total"]["abserror"]["mean"]))
  vmean, vstd, vmin, vmax, vmed = safe_stats(val_folds)
  tmean, tstd, _, _, _ = safe_stats(train_folds)
  val_cv = (vstd / vmean) if (vstd is not None and vmean and vmean > 0) else None
  fold_range = (vmax - vmin) if (vmax is not None and vmin is not None) else None
  overfit = (tmean - vmean) if (tmean is not None and vmean is not None) else None
  gen_ratio = (vmean / tmean) if (tmean is not None and tmean > 0 and vmean is not None) else None
  return Metrics(
    val_mean=vmean,
    val_std=vstd,
    val_min=vmin,
    val_max=vmax,
    val_median=vmed,
    val_cv=val_cv,
    fold_range=fold_range,
    train_mean=tmean,
    train_std=tstd,
    overfitting_gap=overfit,
    generalization_ratio=gen_ratio,
    val_folds=val_folds,
    train_folds=train_folds,
  )


def find_trainer_dirs(root: Path) -> list[Path]:
  trainer_dirs: list[Path] = []
  for p in root.rglob("*"):
    if not p.is_dir():
      continue
    name = p.name
    if name.startswith(TRAINER_PREFIXES):
      trainer_dirs.append(p)
  return trainer_dirs


def load_meta_from_trainer(trainer_dir: Path) -> ModelMeta | None:
  kind, param, repeat = parse_trainer_info(trainer_dir)
  batchsize, epochs, dataset = parse_batch_epochs_dataset(trainer_dir)

  # Find a representative folder that has mrsnet.json
  fold0 = None
  mrs = None
  if kind in ("KFold", "DuplexKFold"):
    # Prefer fold-0, else any fold
    fold0 = next((d for d in trainer_dir.glob("fold-*")), None)
    if fold0 is None:
      return None
    mrs = parse_mrsnet_json(fold0)
    # Fallback: search within fold0 or other folds if missing
    if mrs is None:
      for cand in fold0.rglob("mrsnet.json"):
        mrs = load_json(cand)
        if mrs:
          break
    if mrs is None:
      for fold in sorted(trainer_dir.glob("fold-*")):
        mrs = parse_mrsnet_json(fold)
        if mrs:
          break
        for cand in fold.rglob("mrsnet.json"):
          mrs = load_json(cand)
          if mrs:
            break
        if mrs:
          break
    if mrs is None:
      # As a last resort, search trainer_dir
      for cand in trainer_dir.rglob("mrsnet.json"):
        mrs = load_json(cand)
        if mrs:
          break
    if mrs is None:
      return None
  else:
    mrs = parse_mrsnet_json(trainer_dir)
    if mrs is None:
      # Some runs might still have mrsnet.json inside a nested folder; fallback: search
      for cand in trainer_dir.rglob("mrsnet.json"):
        mrs = load_json(cand)
        if mrs:
          break
    if mrs is None:
      return None

  # Basic fields
  model_name = mrs.get("model", "")
  arch_key, arch_params = canonicalize_cnn_model(model_name)
  metabolites = mrs.get("metabolites", [])
  pulse_sequence = mrs.get("pulse_sequence", "")
  acquisitions = mrs.get("acquisitions", [])
  datatype = mrs.get("datatype", [])
  norm = mrs.get("norm", "")
  dataset_name = mrs.get("train_dataset_name", dataset)

  meta = ModelMeta(
    model_name=model_name,
    arch_key=arch_key,
    metabolites=metabolites,
    pulse_sequence=pulse_sequence,
    acquisitions=acquisitions,
    datatype=datatype,
    norm=norm,
    batchsize=batchsize,
    epochs=epochs,
    dataset_name=dataset_name,
    trainer_kind=kind,
    trainer_param=param,
    trainer_repeat=repeat,
    n_folds=(len(list(trainer_dir.glob("fold-*"))) if kind in ("KFold", "DuplexKFold") else 1),
    path=str(trainer_dir),
    realpath=str(trainer_dir.resolve()),
    total_params=mrs.get("total_params"),
    trainable_params=mrs.get("trainable_params"),
    non_trainable_params=mrs.get("non_trainable_params"),
    flops=mrs.get("flops"),
  )
  return meta


def build_row(meta: ModelMeta, metrics: Metrics) -> AggregateRow:
  # Parse numeric params from arch_key
  arch_key, parsed = canonicalize_cnn_model(meta.arch_key)
  return AggregateRow(
    model_name=meta.model_name,
    arch_key=arch_key,
    model_S1=parsed["S1"],
    model_S2=parsed["S2"],
    model_C1=parsed["C1"],
    model_C2=parsed["C2"],
    model_C3=parsed["C3"],
    model_C4=parsed["C4"],
    model_O1=parsed["O1"],
    model_O2=parsed["O2"],
    model_F1=parsed["F1"],
    model_F2=parsed["F2"],
    model_D=parsed["D"],
    model_ACTIVATION=parsed["ACTIVATION"],
    metabolites="-".join(meta.metabolites),
    pulse_sequence=meta.pulse_sequence,
    acquisitions="-".join(meta.acquisitions),
    datatype="-".join(meta.datatype),
    norm=meta.norm,
    batchsize=meta.batchsize,
    epochs=meta.epochs,
    dataset_name=meta.dataset_name,
    trainer_kind=meta.trainer_kind,
    trainer_param=meta.trainer_param,
    trainer_repeat=meta.trainer_repeat,
    n_folds=meta.n_folds,
    path=meta.path,
    realpath=meta.realpath,
    total_params=meta.total_params,
    trainable_params=meta.trainable_params,
    non_trainable_params=meta.non_trainable_params,
    flops=meta.flops,
    val_mean=metrics.val_mean,
    val_std=metrics.val_std,
    val_min=metrics.val_min,
    val_max=metrics.val_max,
    val_median=metrics.val_median,
    val_cv=metrics.val_cv,
    fold_range=metrics.fold_range,
    train_mean=metrics.train_mean,
    train_std=metrics.train_std,
    overfitting_gap=metrics.overfitting_gap,
    generalization_ratio=metrics.generalization_ratio,
    val_folds=";".join([f"{v:.12g}" for v in metrics.val_folds]) if metrics.val_folds else "",
    train_folds=";".join([f"{v:.12g}" for v in metrics.train_folds]) if metrics.train_folds else "",
  )


def _rows_to_dicts(rows: list[AggregateRow]) -> list[dict]:
  return [asdict(r) for r in rows]


def _write_csv(path: Path, rows: list[AggregateRow]) -> None:
  with open(path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
    writer.writeheader()
    for r in rows:
      writer.writerow(asdict(r))


def write_outputs(rows: list[AggregateRow], outdir: Path) -> None:
  outdir.mkdir(parents=True, exist_ok=True)
  # CSV
  csv_path = outdir / "cnn_models_aggregate.csv"
  _write_csv(csv_path, rows)
  # JSON
  json_path = outdir / "cnn_models_aggregate.json"
  with open(json_path, "w") as f:
    json.dump(_rows_to_dicts(rows), f, indent=2)
  print(f"Wrote {csv_path}")
  print(f"Wrote {json_path}")


def signature_for_dedup(r: AggregateRow) -> tuple:
  # Unique signature for a model config ignoring trainer repeat and paths
  return (
    r.arch_key,
    r.metabolites,
    r.pulse_sequence,
    r.acquisitions,
    r.datatype,
    r.norm,
    r.batchsize,
    r.epochs,
    r.dataset_name,
    r.trainer_kind,
    r.trainer_param,
    r.n_folds,
  )


def context_key(r: AggregateRow) -> tuple:
  # Context excluding architecture; used to compare architectures within same setting
  return (
    r.metabolites,
    r.pulse_sequence,
    r.acquisitions,
    r.datatype,
    r.norm,
    r.batchsize,
    r.epochs,
    r.dataset_name,
    r.trainer_kind,
    r.trainer_param,
    r.n_folds,
  )


def deduplicate_best(rows: list[AggregateRow]) -> list[AggregateRow]:
  # Sort so that best Val_Mean appears first, then keep-first per signature
  def sk(r: AggregateRow):
    return (1, float("inf")) if r.val_mean is None else (0, r.val_mean)

  sorted_rows = sorted(rows, key=sk)
  seen = set()
  dedup: list[AggregateRow] = []
  for r in sorted_rows:
    sig = signature_for_dedup(r)
    if sig in seen:
      continue
    seen.add(sig)
    dedup.append(r)
  return dedup


def compute_stats(rows: list[AggregateRow], dedup_rows: list[AggregateRow], outdir: Path) -> None:
  # Best per context
  ctx_to_rows: dict[tuple, list[AggregateRow]] = defaultdict(list)
  for r in dedup_rows:
    if r.val_mean is None:
      continue
    ctx_to_rows[context_key(r)].append(r)
  best_per_context: list[AggregateRow] = []
  for ctx, rs in ctx_to_rows.items():
    best = min(rs, key=lambda x: x.val_mean)
    best_per_context.append(best)
  # Architecture wins and averages
  arch_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"wins": 0.0, "count": 0.0, "mean_val_mean": 0.0})
  # Count participations and mean
  arch_accum: dict[str, list[float]] = defaultdict(list)
  for r in dedup_rows:
    if r.val_mean is not None:
      arch_accum[r.arch_key].append(r.val_mean)
  for arch, vals in arch_accum.items():
    if len(vals) > 0:
      arch_stats[arch]["count"] = float(len(vals))
      arch_stats[arch]["mean_val_mean"] = float(sum(vals) / len(vals))
  # Wins
  for r in best_per_context:
    arch_stats[r.arch_key]["wins"] = arch_stats[r.arch_key].get("wins", 0.0) + 1.0
  # Write files
  outdir.mkdir(parents=True, exist_ok=True)
  # Dedup outputs
  _write_csv(outdir / "cnn_models_aggregate_dedup.csv", dedup_rows)
  with open(outdir / "cnn_models_aggregate_dedup.json", "w") as f:
    json.dump(_rows_to_dicts(dedup_rows), f, indent=2)
  # Best per context
  _write_csv(outdir / "cnn_best_per_context.csv", best_per_context)
  with open(outdir / "cnn_best_per_context.json", "w") as f:
    json.dump(_rows_to_dicts(best_per_context), f, indent=2)
  # Architecture stats
  with open(outdir / "cnn_architecture_stats.json", "w") as f:
    json.dump(arch_stats, f, indent=2)
  # Context overview (counts per context)
  ctx_rows = []
  for ctx, rs in ctx_to_rows.items():
    ctx_rows.append(
      {
        "metabolites": ctx[0],
        "pulse_sequence": ctx[1],
        "acquisitions": ctx[2],
        "datatype": ctx[3],
        "norm": ctx[4],
        "batchsize": ctx[5],
        "epochs": ctx[6],
        "dataset_name": ctx[7],
        "trainer_kind": ctx[8],
        "trainer_param": ctx[9],
        "n_folds": ctx[10],
        "num_models": len(rs),
      }
    )
  # Write as CSV
  ctx_csv = outdir / "cnn_contexts_overview.csv"
  with open(ctx_csv, "w", newline="") as f:
    if ctx_rows:
      w = csv.DictWriter(f, fieldnames=list(ctx_rows[0].keys()))
      w.writeheader()
      for row in ctx_rows:
        w.writerow(row)
    else:
      f.write("")


def main():
  parser = argparse.ArgumentParser(description="Aggregate CNN model results from data/model-cnn")
  parser.add_argument("--root", type=str, default="./data/model-cnn", help="Root folder to scan")
  parser.add_argument("--out", type=str, default="./out/cnn-aggregate", help="Output folder for aggregate files")
  parser.add_argument("--dataset-filter", type=str, default="", help="Substring to filter dataset_name/path")
  parser.add_argument("--linewidth-filter", type=str, default="", help="Substring to filter linewidth in dataset path")
  args = parser.parse_args()

  root = Path(args.root).resolve()
  outdir = Path(args.out)

  if not root.exists():
    raise SystemExit(f"Root not found: {root}")

  trainer_dirs = find_trainer_dirs(root)
  # Deduplicate by realpath to avoid symlink duplicates
  seen_realpaths = set()

  rows: list[AggregateRow] = []
  for trainer_dir in sorted(trainer_dirs):
    real = str(trainer_dir.resolve())
    if real in seen_realpaths:
      continue
    seen_realpaths.add(real)

    meta = load_meta_from_trainer(trainer_dir)
    if meta is None:
      continue

    # Only aggregate CNN models
    if not meta.model_name.startswith("cnn_"):
      continue

    # Collect metrics based on trainer kind
    kind = meta.trainer_kind
    if kind in ("KFold", "DuplexKFold"):
      metrics = collect_metrics_from_kfold(trainer_dir)
    elif kind in ("Split", "DuplexSplit"):
      metrics = collect_metrics_from_split(trainer_dir)
    elif kind == "NoValidation":
      # No validation metrics; record train metrics only
      m = collect_metrics_from_split(trainer_dir)
      metrics = Metrics(
        val_mean=None,
        val_std=None,
        val_min=None,
        val_max=None,
        val_median=None,
        val_cv=None,
        fold_range=None,
        train_mean=m.train_mean,
        train_std=m.train_std,
        overfitting_gap=None,
        generalization_ratio=None,
        val_folds=[],
        train_folds=m.train_folds,
      )
    else:
      # Unknown trainer naming; best-effort attempt
      metrics = collect_metrics_from_split(trainer_dir)

    row = build_row(meta, metrics)
    # Dataset filter
    ds_filter = args.dataset_filter.strip().lower()
    if ds_filter and (ds_filter not in row.dataset_name.lower() and ds_filter not in row.path.lower()):
      continue
    lw_filter = args.linewidth_filter.strip().lower()
    if lw_filter:
      # Match linewidth substring in dataset path/name (as segment)
      # Accept matches in dataset_name or model path
      if lw_filter not in row.dataset_name.lower() and lw_filter not in row.path.lower():
        continue
    rows.append(row)

  # Sort by validation mean ascending (None at end)
  def sort_key(r: AggregateRow):
    return (1, float("inf")) if r.val_mean is None else (0, r.val_mean)

  rows.sort(key=sort_key)
  write_outputs(rows, outdir)

  # Deduplicate repeats by signature and write stats
  dedup_rows = deduplicate_best(rows)
  compute_stats(rows, dedup_rows, outdir)


if __name__ == "__main__":
  main()
