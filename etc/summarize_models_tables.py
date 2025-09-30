#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (C) 2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Summarise train and validation performance for all selected models under a root folder.

- Recursively scans trainer result folders under a root (default: ./data/model)
- Supports KFold/DuplexKFold, Split/DuplexSplit, and NoValidation trainers
- Extracts per-fold train/validation MAE from *concentration_errors.json files
- Aggregates folds to mean/std per run and renders tables including ALL models (no top-N)
- Produces one figure per top-level model family (first directory level under --root)
- Each figure is a table with a small horizontal bar plot (validation vs train) in the last column

Usage:
  python3 etc/summarize_models_tables.py \
    --root ./data/model \
    --out ./out/model-tables \
    --save_png
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

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
  """Metadata describing a single trainer run discovered on disk."""

  model_name: str
  family: str
  metabolites: str
  pulse_sequence: str
  acquisitions: str
  datatype: str
  norm: str
  batchsize: str
  epochs: str
  dataset_name: str
  trainer: str
  path: str


@dataclass
class RunMetrics:
  """Aggregated metrics (mean±std) computed from per-fold JSON files."""

  val_mean: float | None
  val_std: float | None
  train_mean: float | None
  train_std: float | None
  val_min: float | None
  val_max: float | None
  train_min: float | None
  train_max: float | None


def _safe_mean_std(values: list[float]) -> tuple[float, float]:
  if not values:
    return math.nan, math.nan
  if len(values) == 1:
    return float(values[0]), 0.0
  return float(mean(values)), float(pstdev(values))


def _load_json(path: Path) -> dict | None:
  try:
    with open(path) as f:
      return json.load(f)
  except Exception:
    return None


def _extract_mae(j: dict | None) -> float | None:
  try:
    return float(j["total"]["abserror"]["mean"]) if j else None
  except Exception:
    return None


def _iter_trainer_dirs(root: Path) -> Iterable[Path]:
  for p in root.rglob("*"):
    if p.is_dir() and p.name.startswith(TRAINER_PREFIXES):
      yield p


def _detect_family(trainer_dir: Path, root: Path) -> str:
  try:
    rel = trainer_dir.relative_to(root)
    parts = rel.parts
    return parts[0] if len(parts) > 0 else root.name
  except Exception:
    return trainer_dir.parents[0].name


def _parse_meta_from_path(trainer_dir: Path, root: Path) -> RunMeta | None:
  # Expect: .../<family>/<model>/<metabolites>/<pulse>/<acq>/<dtype>/<norm>/<batch>/<epochs>/<dataset>/<trainer>
  try:
    p = trainer_dir
    dataset_name = p.parent.name
    epochs = p.parent.parent.name
    batch = p.parent.parent.parent.name
    norm = p.parent.parent.parent.parent.name
    dtype = p.parent.parent.parent.parent.parent.name
    acq = p.parent.parent.parent.parent.parent.parent.name
    pulse = p.parent.parent.parent.parent.parent.parent.parent.name
    metabolites = p.parent.parent.parent.parent.parent.parent.parent.parent.name
    model = p.parent.parent.parent.parent.parent.parent.parent.parent.parent.name
    family = _detect_family(trainer_dir, root)
  except Exception:
    return None
  return RunMeta(
    model_name=model,
    family=family,
    metabolites=metabolites,
    pulse_sequence=pulse,
    acquisitions=acq,
    datatype=dtype,
    norm=norm,
    batchsize=batch,
    epochs=epochs,
    dataset_name=dataset_name,
    trainer=p.name,
    path=str(trainer_dir),
  )


def _load_meta(trainer_dir: Path, root: Path) -> RunMeta | None:
  # Prefer mrsnet.json if present (fold-0 or trainer root), else parse from path
  mrs = None
  fold0 = next((d for d in sorted(trainer_dir.glob(f"{FOLD_PREFIX}*"))), None)
  if fold0 is not None:
    mrs = _load_json(fold0 / "mrsnet.json")
    if mrs is None:
      for cand in fold0.rglob("mrsnet.json"):
        mrs = _load_json(cand)
        if mrs:
          break
  if mrs is None:
    mrs = _load_json(trainer_dir / "mrsnet.json")
  family = _detect_family(trainer_dir, root)
  if mrs:
    # Fallback for batchsize/epochs if missing in JSON: derive from path
    # Expected path: .../<norm>/<batch>/<epochs>/<dataset>/<trainer>
    try:
      p = trainer_dir
      epochs_from_path = p.parent.parent.name
      batch_from_path = p.parent.parent.parent.name
    except Exception:
      epochs_from_path = ""
      batch_from_path = ""
    batch_val = str(mrs.get("batchsize", "")) or batch_from_path
    epochs_val = str(mrs.get("epochs", "")) or epochs_from_path

    return RunMeta(
      model_name=str(mrs.get("model", "")),
      family=family,
      metabolites="-".join(mrs.get("metabolites", [])) if isinstance(mrs.get("metabolites"), list) else str(mrs.get("metabolites", "")),
      pulse_sequence=str(mrs.get("pulse_sequence", "")),
      acquisitions="-".join(mrs.get("acquisitions", [])) if isinstance(mrs.get("acquisitions"), list) else str(mrs.get("acquisitions", "")),
      datatype="-".join(mrs.get("datatype", [])) if isinstance(mrs.get("datatype"), list) else str(mrs.get("datatype", "")),
      norm=str(mrs.get("norm", "")),
      batchsize=batch_val,
      epochs=epochs_val,
      dataset_name=str(mrs.get("train_dataset_name", trainer_dir.parent.name)),
      trainer=trainer_dir.name,
      path=str(trainer_dir),
    )
  return _parse_meta_from_path(trainer_dir, root)


def _collect_metrics(trainer_dir: Path) -> RunMetrics:
  val_vals: list[float] = []
  train_vals: list[float] = []
  # Prefer fold-wise if present
  folds = sorted(trainer_dir.glob(f"{FOLD_PREFIX}*"))
  if folds:
    for f in folds:
      v = _extract_mae(_load_json(f / "validation_concentration_errors.json"))
      t = _extract_mae(_load_json(f / "train_concentration_errors.json"))
      if v is not None:
        val_vals.append(v)
      if t is not None:
        train_vals.append(t)
  else:
    v = _extract_mae(_load_json(trainer_dir / "validation_concentration_errors.json"))
    t = _extract_mae(_load_json(trainer_dir / "train_concentration_errors.json"))
    if v is not None:
      val_vals.append(v)
    if t is not None:
      train_vals.append(t)
  v_mean, v_std = _safe_mean_std(val_vals) if val_vals else (math.nan, math.nan)
  t_mean, t_std = _safe_mean_std(train_vals) if train_vals else (math.nan, math.nan)
  v_min = min(val_vals) if val_vals else math.nan
  v_max = max(val_vals) if val_vals else math.nan
  t_min = min(train_vals) if train_vals else math.nan
  t_max = max(train_vals) if train_vals else math.nan
  return RunMetrics(
    val_mean=(None if math.isnan(v_mean) else v_mean),
    val_std=(None if math.isnan(v_std) else v_std),
    train_mean=(None if math.isnan(t_mean) else t_mean),
    train_std=(None if math.isnan(t_std) else t_std),
    val_min=(None if math.isnan(v_min) else float(v_min)),
    val_max=(None if math.isnan(v_max) else float(v_max)),
    train_min=(None if math.isnan(t_min) else float(t_min)),
    train_max=(None if math.isnan(t_max) else float(t_max)),
  )


def _ensure_dir(path: str) -> None:
  os.makedirs(path, exist_ok=True)


def _shorten(text: str, max_len: int = 64) -> str:
  s = str(text)
  return s if len(s) <= max_len else (s[:max_len - 1] + "…")


def _make_combined_table_figure(rows_input: list[tuple[RunMeta, RunMetrics]], out_path_base: str, save_png: bool, *, bar_logscale: bool, fig_scale: float) -> None:
  if not rows_input:
    return
  # Sort all rows by validation mean (None at end)
  def row_sort_key(item: tuple[RunMeta, RunMetrics]):
    m = item[1]
    return (1, float('inf')) if m.val_mean is None else (0, m.val_mean)
  rows_sorted = sorted(rows_input, key=row_sort_key)

  # Build ordered info pairs per model: model → training → dataset
  def build_info_pairs(meta: RunMeta) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    pairs.append(("Model", meta.model_name))
    pairs.append(("Metabolites", meta.metabolites))
    pairs.append(("Pulse", meta.pulse_sequence))
    pairs.append(("Acq", meta.acquisitions))
    pairs.append(("Datatype", meta.datatype))
    pairs.append(("Norm", meta.norm))
    pairs.append(("Epochs", meta.epochs))
    pairs.append(("Batch", meta.batchsize))
    pairs.append(("Trainer", meta.trainer))

    # Split dataset name into components
    dataset_parts = meta.dataset_name.split('_')
    if len(dataset_parts) >= 2:
      pairs.append(("Dataset", dataset_parts[0]))  # e.g., "fid-a-2d"
      pairs.append(("Size", dataset_parts[-1]))    # e.g., "10000-1"
    else:
      pairs.append(("Dataset", meta.dataset_name))

    return pairs

  all_pairs = [build_info_pairs(meta) for meta, _ in rows_sorted]
  labels = [k for k, _ in all_pairs[0]] if all_pairs else []
  constant_labels: list[str] = []
  constants_text: list[str] = []
  for idx, label in enumerate(labels):
    vals = [pairs[idx][1] if idx < len(pairs) else "" for pairs in all_pairs]
    if len(vals) > 0 and all(v == vals[0] for v in vals):
      constant_labels.append(label)
      constants_text.append(f"{label}: {vals[0]}")

  # Compute all field names from non-constant pairs
  all_field_names: set[str] = set()
  for meta, _ in rows_sorted:
    pairs = [(k, v) for k, v in build_info_pairs(meta) if k not in constant_labels]
    all_field_names.update(k for k, v in pairs)

  # Create ordered field names and split into two header rows
  field_names_list = list(all_field_names)
  field_names_list.sort()  # Sort alphabetically for consistency

  # Split fields into two rows for header
  if len(field_names_list) <= 3:
    header_row1_fields = field_names_list
    header_row2_fields: list[str] = []
  else:
    mid = (len(field_names_list) + 1) // 2
    header_row1_fields = field_names_list[:mid]
    header_row2_fields = field_names_list[mid:]

  # Columns: max number of info columns across the two header rows | Val/Train | Plot
  info_cols = max(len(header_row1_fields), len(header_row2_fields))
  columns = [""] * info_cols + ["Validation/Training", "Plot"]
  col_widths = [1.6] * info_cols + [1.7, 4.0]

  # Determine x-limits
  x_vals_all: list[float] = []
  for _, m in rows_sorted:
    if m.val_mean is not None:
      x_vals_all.append(m.val_mean)
    if m.train_mean is not None:
      x_vals_all.append(m.train_mean)
  if x_vals_all:
    if bar_logscale:
      positives = [x for x in x_vals_all if x > 0]
      if positives:
        min_pos = min(positives)
        x_min = max(min_pos / 10.0, 1e-12)
        x_max = max(x_vals_all) * 1.5
      else:
        x_min, x_max = 1e-12, 1.0
    else:
      x_min = 0.0
      x_max = max(x_vals_all) * 1.15
  else:
    x_min, x_max = (1e-12, 1.0) if bar_logscale else (0.0, 1.0)

  # Total rows and height ratios: header (+separator) + [2 rows per model + thin separator] + optional spacer + constants
  header_rows = 2 if header_row2_fields else 1
  height_ratios: list[float] = []
  # Slightly tighter header rows
  height_ratios.extend([0.85] * header_rows)
  # Add a thin separator below header
  height_ratios.append(0.12)
  # Data rows and thin separators
  for i in range(len(rows_sorted)):
    height_ratios.extend([1.0, 0.95])
    if i != len(rows_sorted) - 1:
      height_ratios.append(0.12)
  # No separate x-axis row; enable x-axis only on last plot
  # Optional spacer + constants footer (slightly thin)
  if constants_text:
    # spacer between last plot x-axis and constants to avoid any overlap (increased)
    height_ratios.append(1.20)
    # constants row height
    height_ratios.append(0.70)
  total_rows = len(height_ratios)
  fig_w = sum(col_widths) * (1.12 * max(1.0, fig_scale))
  fig_h = 0.26 * total_rows + 0.30
  fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
  gs = gridspec.GridSpec(total_rows, len(columns), width_ratios=col_widths, height_ratios=height_ratios, hspace=0.01, wspace=0.035)
  # Ensure some bottom space for the last axis labels
  plt.subplots_adjust(bottom=0.06)

  # Header row 1
  current_row = 0
  for i in range(info_cols):
    ax = plt.subplot(gs[current_row, i])
    ax.axis('off')
    if i < len(header_row1_fields):
      ax.text(0.0, 0.5, header_row1_fields[i], fontsize=12, fontweight='bold', va='center', ha='left')

  # Validation header (row 1)
  ax_val = plt.subplot(gs[current_row, info_cols])
  ax_val.axis('off')
  ax_val.text(0.0, 0.5, "Validation", fontsize=12, fontweight='bold', va='center', ha='left')

  ax_plot = plt.subplot(gs[current_row, info_cols + 1])
  ax_plot.axis('off')
  ax_plot.text(0.0, 0.5, "Plot", fontsize=12, fontweight='bold', va='center', ha='left')

  current_row += 1

  # Header row 2 (only if we have second row fields)
  if header_row2_fields:
    for i in range(info_cols):
      ax = plt.subplot(gs[current_row, i])
      ax.axis('off')
      if i < len(header_row2_fields):
        ax.text(0.0, 0.5, header_row2_fields[i], fontsize=12, fontweight='bold', va='center', ha='left')

    # Training header (row 2)
    ax_train = plt.subplot(gs[current_row, info_cols])
    ax_train.axis('off')
    ax_train.text(0.0, 0.5, "Training", fontsize=12, fontweight='bold', va='center', ha='left')

    # Empty plot cell (spans from row 1)
    ax_plot = plt.subplot(gs[current_row, info_cols + 1])
    ax_plot.axis('off')

    current_row += 1

  # Header separator line row
  ax_hsep = plt.subplot(gs[current_row, :])
  ax_hsep.axis('off')
  ax_hsep.axhline(y=0.5, xmin=0, xmax=1, color='lightgray', linewidth=0.6)
  current_row += 1

  # Rows
  for i_row, (meta, m) in enumerate(rows_sorted):
    # Get all non-constant pairs for this model
    pairs = [(k, v) for k, v in build_info_pairs(meta) if k not in constant_labels]

    # Row 1: first half of fields + validation + plot
    for i in range(info_cols):
      ax_info = plt.subplot(gs[current_row, i])
      ax_info.axis('off')
      # Find value for this field in first half of pairs
      value = ""
      if i < len(header_row1_fields):
        field_to_find = header_row1_fields[i]
        for k, v in pairs:
          if k == field_to_find:
            value = v
            break
      if value:
        ax_info.text(0.0, 0.5, _shorten(value, 64), fontsize=10, va='center', ha='left')

    # Validation cell (row 1)
    ax_val = plt.subplot(gs[current_row, info_cols])
    ax_val.axis('off')
    ax_val.text(0.0, 0.5, ("n/a" if m.val_mean is None else f"{m.val_mean:.5f} ± {0.0 if m.val_std is None else m.val_std:.5f}"), fontsize=10, va='center', ha='left')

    # Plot cell (spans both rows)
    axp = plt.subplot(gs[current_row:current_row+2, info_cols + 1])
    if bar_logscale:
      axp.set_xscale('log')
    axp.set_xlim(x_min, x_max)
    axp.set_ylim(0.22, 0.78)
    axp.set_yticks([])
    # Hide ticks for non-last plots; last plot configured below
    axp.set_xticks([])
    axp.set_xticklabels([])
    axp.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axp.spines['top'].set_visible(False)
    axp.spines['right'].set_visible(False)
    axp.spines['left'].set_visible(False)
    axp.spines['bottom'].set_visible(False)

    # If this is the last plot, enable its x-axis with proper ticks and labels
    if i_row == len(rows_sorted) - 1:
      axp.spines['bottom'].set_visible(True)
      axp.set_zorder(2)
      axp.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, pad=4, labelsize=12, length=4, colors='black')
      axp.xaxis.label.set_color('black')
      if bar_logscale:
        # Use log locators/formatters for clear 10^k labels
        log_locator = mticker.LogLocator(base=10)
        minor_locator = mticker.LogLocator(base=10, subs=(2, 5))
        axp.xaxis.set_major_locator(log_locator)
        axp.xaxis.set_minor_locator(minor_locator)
        axp.xaxis.set_major_formatter(mticker.LogFormatterExponent(base=10))
        axp.xaxis.set_minor_formatter(mticker.NullFormatter())
        axp.set_xlabel('MAE (log10)', labelpad=6)
      else:
        axp.set_xticks(np.linspace(x_min, x_max, 6))
        axp.set_xlabel('MAE', labelpad=6)

    # Vertical guide lines aligned with x-axis ticks
    if bar_logscale:
      log_min = math.floor(math.log10(max(x_min, 1e-300)))
      log_max = math.ceil(math.log10(max(x_max, 1e-300)))
      for d in range(log_min, log_max + 1):
        t_major = 10 ** d
        if x_min <= t_major <= x_max:
          axp.axvline(t_major, color='lightgray', linewidth=0.3, alpha=0.6, zorder=0)
        for mult in (2, 5):
          t_minor = mult * (10 ** d)
          if x_min <= t_minor <= x_max:
            axp.axvline(t_minor, color='lightgray', linewidth=0.25, alpha=0.5, zorder=0)
    else:
      for tick in np.linspace(x_min, x_max, 6):
        axp.axvline(tick, color='lightgray', linewidth=0.3, alpha=0.6, zorder=0)

    if m.val_mean is not None or m.train_mean is not None:
      def _safe_x(x: float | None) -> float:
        if x is None:
          return x_min if bar_logscale else 0.0
        if bar_logscale:
          return max(x, x_min)
        return x
      xv = _safe_x(m.val_mean)
      xt = _safe_x(m.train_mean)
      # Asymmetric error bars from min/max across folds
      if m.val_min is not None and m.val_max is not None and xv is not None:
        ev_left = max(xv - _safe_x(m.val_min), 0.0)
        ev_right = max(_safe_x(m.val_max) - xv, 0.0)
      else:
        ev_left = ev_right = 0.0 if m.val_std is None else m.val_std
      if m.train_min is not None and m.train_max is not None and xt is not None:
        et_left = max(xt - _safe_x(m.train_min), 0.0)
        et_right = max(_safe_x(m.train_max) - xt, 0.0)
      else:
        et_left = et_right = 0.0 if m.train_std is None else m.train_std
      xerr = [[ev_left, et_left], [ev_right, et_right]]
      axp.barh([0.65, 0.35], [xv, xt], height=0.18, xerr=xerr, color=['steelblue', 'salmon'], capsize=2, error_kw={'elinewidth': 0.8, 'capthick': 0.8}, zorder=1)

    current_row += 1

    # Row 2: second half of fields + training (empty validation)
    for i in range(info_cols):
      ax_info = plt.subplot(gs[current_row, i])
      ax_info.axis('off')
      # Find value for this field in second half of pairs
      value = ""
      if i < len(header_row2_fields):
        field_to_find = header_row2_fields[i]
        for k, v in pairs:
          if k == field_to_find:
            value = v
            break
      if value:
        ax_info.text(0.0, 0.5, _shorten(value, 64), fontsize=10, va='center', ha='left')

    # Training cell (row 2)
    ax_train = plt.subplot(gs[current_row, info_cols])
    ax_train.axis('off')
    ax_train.text(0.0, 0.5, ("n/a" if m.train_mean is None else f"{m.train_mean:.5f} ± {0.0 if m.train_std is None else m.train_std:.5f}"), fontsize=10, va='center', ha='left')

    current_row += 1

    # Add horizontal separator line between models (except after the last model)
    if i_row != len(rows_sorted) - 1:  # Not the last model
      ax_sep = plt.subplot(gs[current_row, :])
      ax_sep.axis('off')
      ax_sep.axhline(y=0.5, xmin=0, xmax=1, color='lightgray', linewidth=0.6)
      current_row += 1

  # Constants footer on its own final row (with more spacing from x-axis)
  if constants_text:
    ax_const = plt.subplot(gs[current_row, :])
    ax_const.axis('off')
    # Ensure a solid background so x-axis tick labels from above cannot visually overlap
    try:
      from matplotlib.patches import Rectangle
      ax_const.add_patch(Rectangle((0, 0), 1, 1, transform=ax_const.transAxes, facecolor='white', edgecolor='none', zorder=0))
    except Exception: # noqa: S110
      pass
    ax_const.text(0.0, -1.0, "Constants: " + ";  ".join(constants_text), fontsize=10, va='center', ha='left', color='dimgray')


  pdf_path = f"{out_path_base}.pdf"
  _ensure_dir(os.path.dirname(pdf_path) or ".")
  fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.2)
  if save_png:
    png_path = f"{out_path_base}.png"
    fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
  plt.close(fig)


def main() -> None:
  """CLI entry point to generate the combined table figure.

  Scans --root recursively for trainer folders, aggregates metrics, groups by
  model family, and writes a single PDF/PNG table with per-family sections.
  """
  parser = argparse.ArgumentParser(description="Summarise all selected models under a root into tables with train/validation bars.")
  parser.add_argument('--root', type=str, default='./data/model', help='Root folder containing model families and runs')
  parser.add_argument('--out', type=str, default='./out/model-tables', help='Output folder for figures')
  parser.add_argument('--save_png', action='store_true', help='Also save PNG alongside PDF')
  parser.add_argument('--bar_logscale', action='store_true', help='Use logarithmic x-scale for the bar plots')
  parser.add_argument('--fig_scale', type=float, default=1.0, help='Figure width scale factor (>=1 widens)')
  args = parser.parse_args()

  root = Path(args.root).resolve()
  out_root = Path(args.out)
  if not root.exists():
    raise SystemExit(f"Root not found: {root}")

  # Scan all trainer dirs and assemble rows per family
  family_to_rows: dict[str, list[tuple[RunMeta, RunMetrics]]] = {}
  for trainer_dir in sorted(_iter_trainer_dirs(root)):
    meta = _load_meta(trainer_dir, root)
    if meta is None:
      continue
    metrics = _collect_metrics(trainer_dir)
    # Keep all runs; missing metrics will render as n/a and empty plot
    family_to_rows.setdefault(meta.family, []).append((meta, metrics))

  # Flatten all rows and render one combined (no grouping)
  all_rows: list[tuple[RunMeta, RunMetrics]] = []
  for rows in family_to_rows.values():
    all_rows.extend(rows)
  out_base = out_root / "all_models_combined_table"
  _make_combined_table_figure(all_rows, str(out_base), save_png=args.save_png, bar_logscale=args.bar_logscale, fig_scale=args.fig_scale)

  print(f"Wrote combined table for {len(family_to_rows)} families into: {out_root}")


if __name__ == '__main__':
  main()
