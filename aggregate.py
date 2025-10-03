#!/usr/bin/env python3
#
# Aggregation of model training/validation and benchmark results
#
# SPDX-FileCopyrightText: Copyright (C) 2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Aggregate model training/validation and benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Trainer folder name prefixes to detect runs
TRAINER_PREFIXES = (
  "KFold_",
  "DuplexKFold_",
  "Split_",
  "DuplexSplit_",
  "NoValidation",
)

FOLD_PREFIX = "fold-"


@dataclass
class RunMeta:
  """Metadata describing a single discovered trainer run on disk."""

  model_name: str
  trainer: str
  dataset_name: str
  path: str

  # Optional enriched metadata (best-effort from mrsnet.json if available)
  metabolites: str | None = None
  pulse_sequence: str | None = None
  acquisitions: str | None = None
  datatype: str | None = None
  norm: str | None = None
  batchsize: str | None = None
  epochs: str | None = None


@dataclass
class ModelProps:
  """Model properties captured by MRSNet (mrsnet.json)."""

  total_params: int | None
  trainable_params: int | None
  non_trainable_params: int | None
  flops: float | int | None


@dataclass
class GroupedRun:
  """Represents a group of runs for the same model configuration across different trainer types."""

  # Model configuration (excluding trainer)
  model_name: str
  dataset_name: str
  metabolites: str | None
  pulse_sequence: str | None
  acquisitions: str | None
  datatype: str | None
  norm: str | None
  batchsize: str | None
  epochs: str | None

  # Combined trainer information
  trainer_types: list[str]  # List of trainer types found (e.g., ["KFold_", "Split_"])
  trainer_paths: list[str]  # List of paths to trainer directories

  # Combined model properties (should be identical across runs)
  model_props: ModelProps

  # Combined training/validation data
  all_val_maes: list[float | None]  # All validation MAEs from all trainer types
  all_train_maes: list[float | None]  # All training MAEs from all trainer types

  # Combined benchmark data
  combined_benchmarks: dict[str, dict[str, float | None]]


def _iter_trainer_dirs(root: Path) -> Iterable[Path]:
  for p in root.rglob("*"):
    if p.is_dir() and p.name.startswith(TRAINER_PREFIXES):
      # Skip symlinks to avoid duplicate processing
      if p.is_symlink():
        continue
      yield p


def _detect_family(trainer_dir: Path, root: Path) -> str:
  try:
    rel = trainer_dir.relative_to(root)
    parts = rel.parts
    return parts[0] if len(parts) > 0 else root.name
  except Exception:
    return trainer_dir.parents[0].name


def _load_json(path: Path) -> dict | None:
  try:
    with open(path) as f:
      return json.load(f)
  except Exception:
    return None


def _normalize_token(s: str | None) -> str | None:
  """Normalize tokens read from metadata to a path/CSV-safe form.

  For datasets coming from mrsnet.json the name may use '/'. Treat '/' as
  equivalent to '_' and return '_' form for CSV stability.
  """
  if s is None:
    return None
  t = str(s).strip()
  if not t:
    return None
  return t.replace('/', '_')


def _load_meta(trainer_dir: Path, root: Path) -> RunMeta | None:
  # First strategy (canonical): parse everything from path layout
  # <model>/<metabolites>/<pulse>/<acq>/<dtype>/<norm>/<batch>/<epochs>/<dataset>/<trainer>
  rel = trainer_dir.relative_to(root)
  parts = rel.parts

  if len(parts) < 10:
    # Not a canonical run path; fallback to minimal info
    model_name = trainer_dir.parents[1].name if len(trainer_dir.parents) >= 2 else trainer_dir.name
    dataset_name = trainer_dir.parent.name
    meta = RunMeta(
      model_name=model_name,
      trainer=trainer_dir.name,
      dataset_name=_normalize_token(dataset_name),
      path=str(trainer_dir),
    )
  else:
    p_model = parts[-10]
    p_metabolites = parts[-9]
    p_pulse = parts[-8]
    p_acq = parts[-7]
    p_dtype = parts[-6]
    p_norm = parts[-5]
    p_batch = parts[-4]
    p_epochs = parts[-3]
    p_dataset = parts[-2]
    meta = RunMeta(
      model_name=p_model,
      trainer=trainer_dir.name,
      dataset_name=_normalize_token(p_dataset),
      path=str(trainer_dir),
      metabolites=p_metabolites,
      pulse_sequence=p_pulse,
      acquisitions=p_acq,
      datatype=p_dtype,
      norm=p_norm,
      batchsize=p_batch,
      epochs=p_epochs,
    )

  # Second (validation) strategy: compare with mrsnet.json and warn on mismatches
  mrs = None
  fold0 = next((d for d in sorted(trainer_dir.glob(f"{FOLD_PREFIX}*"))), None)
  if fold0 is not None:
    mrs = _load_json(fold0 / "mrsnet.json") or _load_json(trainer_dir / "mrsnet.json")
  else:
    mrs = _load_json(trainer_dir / "mrsnet.json")
  if mrs:
    # Prepare comparable tokens
    m_model = str(mrs.get("model", "")) or None
    m_dataset = _normalize_token(str(mrs.get("train_dataset_name", "")))
    m_metabolites = "-".join(mrs.get("metabolites", [])) if isinstance(mrs.get("metabolites"), list) else (str(mrs.get("metabolites", "")) or None)
    m_pulse = str(mrs.get("pulse_sequence", "")) or None
    m_acq = "-".join(mrs.get("acquisitions", [])) if isinstance(mrs.get("acquisitions"), list) else (str(mrs.get("acquisitions", "")) or None)
    m_dtype = "-".join(mrs.get("datatype", [])) if isinstance(mrs.get("datatype"), list) else (str(mrs.get("datatype", "")) or None)
    m_norm = str(mrs.get("norm", "")) or None
    m_batch = str(mrs.get("batchsize", "")) or None
    m_epochs = str(mrs.get("epochs", "")) or None

    def warn_if_diff(name: str, a: str | None, b: str | None) -> None:
      if (a is not None and a != "") and (b is not None and b != "") and (str(a) != str(b)):
        print(f"# Warning: {name} mismatch (path='{a}' vs meta='{b}') for {trainer_dir}")

    warn_if_diff("model", meta.model_name, m_model)
    warn_if_diff("dataset", meta.dataset_name, m_dataset)
    warn_if_diff("metabolites", meta.metabolites, m_metabolites)
    warn_if_diff("pulse", meta.pulse_sequence, m_pulse)
    warn_if_diff("acquisitions", meta.acquisitions, m_acq)
    warn_if_diff("datatype", meta.datatype, m_dtype)
    warn_if_diff("norm", meta.norm, m_norm)
    warn_if_diff("batch", meta.batchsize, m_batch)
    warn_if_diff("epochs", meta.epochs, m_epochs)

  return meta


def _load_model_props(trainer_dir: Path) -> ModelProps:
  # Check fold-0 then trainer root
  mrs = None
  fold0 = next((d for d in sorted(trainer_dir.glob(f"{FOLD_PREFIX}*"))), None)
  if fold0 is not None:
    mrs = _load_json(fold0 / "mrsnet.json") or _load_json(trainer_dir / "mrsnet.json")
  else:
    mrs = _load_json(trainer_dir / "mrsnet.json")
  if not mrs:
    return ModelProps(None, None, None, None)
  return ModelProps(
    total_params=_safe_int(mrs.get("total_params")),
    trainable_params=_safe_int(mrs.get("trainable_params")),
    non_trainable_params=_safe_int(mrs.get("non_trainable_params")),
    flops=_safe_float(mrs.get("flops")),
  )


def _safe_int(v) -> int | None:
  try:
    return int(v) if v is not None else None
  except Exception:
    return None


def _safe_float(v) -> float | None:
  try:
    f = float(v)
    return f if not math.isnan(f) else None
  except Exception:
    return None


def _create_grouping_key(meta: RunMeta) -> tuple:
  """Create a grouping key for runs based on model configuration (excluding trainer type)."""
  return (
    meta.model_name,
    meta.dataset_name,
    meta.metabolites,
    meta.pulse_sequence,
    meta.acquisitions,
    meta.datatype,
    meta.norm,
    meta.batchsize,
    meta.epochs,
  )


def _extract_mae(j: dict | None) -> float | None:
  try:
    return float(j["total"]["abserror"]["mean"]) if j else None
  except Exception:
    return None


def _collect_train_val_maes(trainer_dir: Path) -> tuple[list[float | None], list[float | None]]:
  """Return lists of per-fold validation and training MAEs (empty lists if none)."""
  val_maes: list[float | None] = []
  train_maes: list[float | None] = []
  folds = sorted(trainer_dir.glob(f"{FOLD_PREFIX}*"))
  if folds:
    for f in folds:
      v = _extract_mae(_load_json(f / "validation_concentration_errors.json"))
      t = _extract_mae(_load_json(f / "train_concentration_errors.json"))
      val_maes.append(v)
      train_maes.append(t)
  else:
    v = _extract_mae(_load_json(trainer_dir / "validation_concentration_errors.json"))
    t = _extract_mae(_load_json(trainer_dir / "train_concentration_errors.json"))
    if v is not None:
      val_maes.append(v)
    if t is not None:
      train_maes.append(t)
  return val_maes, train_maes


def _normalize_benchmark_name(name: str) -> str:
  # Replace colons with underscores; strip trailing norm suffixes
  n = name.replace(":", "_")
  for suf in ("_max", "_sum"):
    if n.endswith(suf):
      n = n[: -len(suf)]
      break
  return n


def _collect_benchmarks(trainer_dir: Path) -> dict[str, dict[str, float | None]]:
  res: dict[str, dict[str, float | None]] = {}
  for file_path in sorted(trainer_dir.glob("*_concentration_errors.json")):
    fn = file_path.name
    if fn.startswith(("train_", "validation_")):
      continue
    j = _load_json(file_path)
    bench_name = _normalize_benchmark_name(fn.replace("_concentration_errors.json", ""))
    # Extract only abserror mean/std/min/max for MAE reporting
    metrics = {"mae": None, "mae_std": None, "mae_min": None, "mae_max": None}
    try:
      t = (j or {}).get("total", {})
      absj = t.get("abserror", {})
      metrics["mae"] = _safe_float(absj.get("mean"))
      metrics["mae_std"] = _safe_float(absj.get("std"))
      metrics["mae_min"] = _safe_float(absj.get("min"))
      metrics["mae_max"] = _safe_float(absj.get("max"))
    except Exception: # noqa: S110
      pass
    res[bench_name] = metrics
  return res


def _flatten_grouped_row(
  grouped_run: GroupedRun,
  bench_keys_all: list[str],
) -> dict:
  row: dict[str, object] = {}
  # Meta
  row.update(
    {
      "model": grouped_run.model_name,
      "trainer": "+".join(sorted(grouped_run.trainer_types)),  # Combined trainer types
      "dataset": grouped_run.dataset_name,
      "metabolites": grouped_run.metabolites,
      "pulse": grouped_run.pulse_sequence,
      "acq": grouped_run.acquisitions,
      "datatype": grouped_run.datatype,
      "norm": grouped_run.norm,
      "batch": grouped_run.batchsize,
      "epochs": grouped_run.epochs,
    }
  )

  # Trainer summary
  row["num_folds"] = len(grouped_run.all_val_maes) if grouped_run.all_val_maes else (len(grouped_run.all_train_maes) if grouped_run.all_train_maes else 0)
  row["num_trainer_types"] = len(grouped_run.trainer_types)

  # Per-fold columns: ordered and contiguous (all folds from all trainer types)
  for i, v in enumerate(grouped_run.all_val_maes):
    row[f"val_mae_fold{i}"] = None if v is None else float(v)
  for i, t in enumerate(grouped_run.all_train_maes):
    row[f"train_mae_fold{i}"] = None if t is None else float(t)

  # Aggregates (mean/std) across all folds from all trainer types
  def _safe_mean(xs: list[float | None]) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    return (sum(vals) / len(vals)) if vals else None

  def _safe_std(xs: list[float | None]) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    if len(vals) <= 1:
      return 0.0 if len(vals) == 1 else None
    m = sum(vals) / len(vals)
    return (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5

  row["val_mae_mean"] = _safe_mean(grouped_run.all_val_maes)
  row["val_mae_std"] = _safe_std(grouped_run.all_val_maes)
  row["train_mae_mean"] = _safe_mean(grouped_run.all_train_maes)
  row["train_mae_std"] = _safe_std(grouped_run.all_train_maes)

  # Model props
  row.update(
    {
      "total_params": grouped_run.model_props.total_params,
      "trainable_params": grouped_run.model_props.trainable_params,
      "non_trainable_params": grouped_run.model_props.non_trainable_params,
      "flops": grouped_run.model_props.flops,
    }
  )

  # Benchmarks: ensure stable set of columns across rows
  for bk in bench_keys_all:
    m = grouped_run.combined_benchmarks.get(bk, {})
    row[f"bench_{bk}_mae"] = m.get("mae")
    row[f"bench_{bk}_mae_std"] = m.get("mae_std")
    row[f"bench_{bk}_mae_min"] = m.get("mae_min")
    row[f"bench_{bk}_mae_max"] = m.get("mae_max")

  return row


def _write_csv(rows: list[dict], path: Path) -> None:
  os.makedirs(path.parent, exist_ok=True)
  # Union of all keys in stable order: meta -> trainer -> per-fold (sorted) -> aggregates -> props -> benchmarks
  # Start with keys from first row for deterministic ordering, then append unseen keys in alphabetical order
  fieldnames: list[str] = []
  seen: set[str] = set()
  if rows:
    for k in rows[0].keys():
      fieldnames.append(k)
      seen.add(k)
  for r in rows:
    for k in r.keys():
      if k not in seen:
        seen.add(k)
        fieldnames.append(k)
  with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
      w.writerow(r)


def main() -> None:
  """Main entry point.""" # noqa: D401
  parser = argparse.ArgumentParser(
    description="Consolidate model results: per-fold train/val MAEs, benchmarks, and model properties."
  )
  parser.add_argument("root", type=str, help="Root folder containing model families and runs")
  args = parser.parse_args()

  root = Path(args.root).resolve()
  if not root.exists():
    raise SystemExit(f"Root not found: {root}")

  # Discover all trainer dirs under single root
  trainer_dirs: list[Path] = sorted(_iter_trainer_dirs(root))

  # First pass: collect all benchmark keys for stable CSV columns
  bench_keys_all: set[str] = set()
  bench_cache: dict[str, dict[str, dict[str, float | None]]] = {}
  for td in trainer_dirs:
    bench = _collect_benchmarks(td)
    bench_cache[str(td)] = bench
    bench_keys_all.update(bench.keys())
  bench_keys_list = sorted(bench_keys_all)

  # Group runs by model configuration (excluding trainer type)
  grouped_runs: dict[tuple, GroupedRun] = {}

  for td in trainer_dirs:
    meta = _load_meta(td, root)
    if meta is None:
      continue

    grouping_key = _create_grouping_key(meta)

    if grouping_key not in grouped_runs:
      # Create new grouped run
      props = _load_model_props(td)
      val_maes, train_maes = _collect_train_val_maes(td)
      bench = bench_cache.get(str(td), {})

      grouped_runs[grouping_key] = GroupedRun(
        model_name=meta.model_name,
        dataset_name=meta.dataset_name,
        metabolites=meta.metabolites,
        pulse_sequence=meta.pulse_sequence,
        acquisitions=meta.acquisitions,
        datatype=meta.datatype,
        norm=meta.norm,
        batchsize=meta.batchsize,
        epochs=meta.epochs,
        trainer_types=[meta.trainer],
        trainer_paths=[meta.path],
        model_props=props,
        all_val_maes=val_maes.copy(),
        all_train_maes=train_maes.copy(),
        combined_benchmarks=bench.copy(),
      )
    else:
      # Add to existing grouped run
      existing = grouped_runs[grouping_key]
      existing.trainer_types.append(meta.trainer)
      existing.trainer_paths.append(meta.path)

      # Collect additional MAEs
      val_maes, train_maes = _collect_train_val_maes(td)
      existing.all_val_maes.extend(val_maes)
      existing.all_train_maes.extend(train_maes)

      # Combine benchmark data (average across trainer types)
      bench = bench_cache.get(str(td), {})
      for bench_name, bench_metrics in bench.items():
        if bench_name not in existing.combined_benchmarks:
          existing.combined_benchmarks[bench_name] = bench_metrics.copy()
        else:
          # Average benchmark metrics across trainer types
          existing_metrics = existing.combined_benchmarks[bench_name]
          for metric_name in ["mae", "mae_std", "mae_min", "mae_max"]:
            if existing_metrics.get(metric_name) is not None and bench_metrics.get(metric_name) is not None:
              # Simple average for now - could be improved with proper statistical aggregation
              existing_metrics[metric_name] = (existing_metrics[metric_name] + bench_metrics[metric_name]) / 2
            elif bench_metrics.get(metric_name) is not None:
              existing_metrics[metric_name] = bench_metrics[metric_name]

  # Convert grouped runs to rows
  rows: list[dict] = []
  for grouped_run in grouped_runs.values():
    row = _flatten_grouped_row(grouped_run, bench_keys_list)
    rows.append(row)

  # Write CSV under <root>/aggregate/
  out_dir = root / "aggregate"
  out_csv = out_dir / "all_results.csv"
  _write_csv(rows, out_csv)

  print(f"Aggregated {len(rows)} grouped model configurations from {len(trainer_dirs)} individual runs. Written CSV: {out_csv}")


if __name__ == "__main__":
  main()
