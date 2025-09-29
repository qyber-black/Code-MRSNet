#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2025
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
  model_name: str
  family: str
  dataset_name: str
  trainer: str
  path: str


@dataclass
class RunMetrics:
  val_abs_mean: float | None
  val_abs_min: float | None
  val_abs_max: float | None
  val_err_mean: float | None
  val_err_min: float | None
  val_err_max: float | None
  train_abs_mean: float | None
  train_abs_min: float | None
  train_abs_max: float | None
  train_err_mean: float | None
  train_err_min: float | None
  train_err_max: float | None


def _load_json(path: Path) -> dict | None:
  try:
    with open(path) as f:
      return json.load(f)
  except Exception:
    return None


def _extract_metrics(j: dict | None) -> dict[str, float | None]:
  res = {
    'abs_mean': None, 'abs_min': None, 'abs_max': None,
    'err_mean': None, 'err_min': None, 'err_max': None,
  }
  try:
    if not j:
      return res
    t = j.get('total', {})
    absj = t.get('abserror', {})
    errj = t.get('error', {})
    res['abs_mean'] = float(absj.get('mean')) if 'mean' in absj else None
    res['abs_min'] = float(absj.get('min')) if 'min' in absj else None
    res['abs_max'] = float(absj.get('max')) if 'max' in absj else None
    res['err_mean'] = float(errj.get('mean')) if 'mean' in errj else None
    res['err_min'] = float(errj.get('min')) if 'min' in errj else None
    res['err_max'] = float(errj.get('max')) if 'max' in errj else None
  except Exception:
    return res
  return res


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


def _load_meta(trainer_dir: Path, root: Path) -> RunMeta | None:
  try:
    dataset_name = trainer_dir.parent.name
    family = _detect_family(trainer_dir, root)
    model_name = trainer_dir.parents[8].name if len(trainer_dir.parents) > 8 else trainer_dir.name
  except Exception:
    return None
  return RunMeta(
    model_name=model_name,
    family=family,
    dataset_name=dataset_name,
    trainer=trainer_dir.name,
    path=str(trainer_dir),
  )


def _collect_train_val(trainer_dir: Path) -> RunMetrics:
  val_abs_vals: list[float] = []
  val_err_vals: list[float] = []
  train_abs_vals: list[float] = []
  train_err_vals: list[float] = []
  folds = sorted(trainer_dir.glob(f"{FOLD_PREFIX}*"))
  if folds:
    for f in folds:
      vj = _extract_metrics(_load_json(f / "validation_concentration_errors.json"))
      tj = _extract_metrics(_load_json(f / "train_concentration_errors.json"))
      if vj['abs_mean'] is not None:
        val_abs_vals.append(float(vj['abs_mean']))
      if vj['err_mean'] is not None:
        val_err_vals.append(float(vj['err_mean']))
      if tj['abs_mean'] is not None:
        train_abs_vals.append(float(tj['abs_mean']))
      if tj['err_mean'] is not None:
        train_err_vals.append(float(tj['err_mean']))
  else:
    vj = _extract_metrics(_load_json(trainer_dir / "validation_concentration_errors.json"))
    tj = _extract_metrics(_load_json(trainer_dir / "train_concentration_errors.json"))
    if vj['abs_mean'] is not None:
      val_abs_vals.append(float(vj['abs_mean']))
    if vj['err_mean'] is not None:
      val_err_vals.append(float(vj['err_mean']))
    if tj['abs_mean'] is not None:
      train_abs_vals.append(float(tj['abs_mean']))
    if tj['err_mean'] is not None:
      train_err_vals.append(float(tj['err_mean']))

  def safe_mean(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None

  def safe_min(xs: list[float]) -> float | None:
    return min(xs) if xs else None

  def safe_max(xs: list[float]) -> float | None:
    return max(xs) if xs else None

  return RunMetrics(
    val_abs_mean=safe_mean(val_abs_vals),
    val_abs_min=safe_min(val_abs_vals),
    val_abs_max=safe_max(val_abs_vals),
    val_err_mean=safe_mean(val_err_vals),
    val_err_min=safe_min(val_err_vals),
    val_err_max=safe_max(val_err_vals),
    train_abs_mean=safe_mean(train_abs_vals),
    train_abs_min=safe_min(train_abs_vals),
    train_abs_max=safe_max(train_abs_vals),
    train_err_mean=safe_mean(train_err_vals),
    train_err_min=safe_min(train_err_vals),
    train_err_max=safe_max(train_err_vals),
  )


def _collect_benchmarks(trainer_dir: Path) -> dict[str, dict[str, float | None]]:
  results: dict[str, dict[str, float | None]] = {}
  for file_path in sorted(trainer_dir.glob('*_concentration_errors.json')):
    fn = file_path.name
    if fn.startswith(('train_', 'validation_')):
      continue
    j = _load_json(file_path)
    metrics = _extract_metrics(j)
    bench_name = fn.replace('_concentration_errors.json', '')
    # Remove trailing norm suffix like '_max' from label only
    if bench_name.endswith('_max'):
      bench_label = bench_name[:-4]
    else:
      bench_label = bench_name
    results[bench_label] = metrics
  return results


def _shorten(text: str, max_len: int = 64) -> str:
  s = str(text)
  return s if len(s) <= max_len else (s[:max_len - 1] + '…')


def _make_model_figure(meta: RunMeta, m: RunMetrics, bench: dict[str, dict[str, float | None]], out_path: str, *, bar_logscale: bool, fig_scale: float) -> None:
  # Compose rows: Validation/Training abserror only, then each benchmark abserror only. Do not include 'All'.
  bench_items = list(bench.items())
  bench_items.sort(key=lambda x: x[0].lower())
  rows: list[tuple[str, float | None, float | None, float | None]] = []
  rows.append(('Validation', m.val_abs_mean, m.val_abs_min, m.val_abs_max))
  rows.append(('Training', m.train_abs_mean, m.train_abs_min, m.train_abs_max))
  for name, metrics in bench_items:
    rows.append((name, metrics.get('abs_mean'), metrics.get('abs_min'), metrics.get('abs_max')))

  # X limits — include error bar min/max to ensure full visibility
  means: list[float] = []
  mins: list[float] = []
  maxs: list[float] = []
  for _, abs_mean, abs_min, abs_max in rows:
    if abs_mean is not None:
      means.append(float(abs_mean))
    if abs_min is not None:
      mins.append(float(abs_min))
    if abs_max is not None:
      maxs.append(float(abs_max))
  if means or mins or maxs:
    candidate_max = max([*means, *maxs]) if (means or maxs) else None
    if bar_logscale:
      # Use the smallest positive among mins and means; fall back to 1e-12
      positives = [v for v in [*mins, *means] if v > 0]
      if positives:
        x_min = max(min(positives) / 10.0, 1e-12)
        x_max = max(candidate_max, 1e-12) * 1.5 if candidate_max is not None else 1.0
      else:
        x_min, x_max = 1e-12, 1.0
    else:
      lower_candidates = [0.0]
      if mins:
        lower_candidates.append(min(mins))
      if means:
        lower_candidates.append(min(means))
      x_min = min(lower_candidates)
      x_max = max(candidate_max, 1.0) * 1.15 if candidate_max is not None else 1.0
  else:
    x_min, x_max = (1e-12, 1.0) if bar_logscale else (0.0, 1.0)

  # Layout: four columns: Label | Mean | Max | Plot; n_rows = header + data rows
  n_data = len(rows)
  fig_w = 12.0 * max(1.0, fig_scale)
  fig_h = 0.56 * (n_data + 4.0)
  fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
  # Rows: 0 header (full-width), 1 column headings, 2 spacer, then data rows, and final spacer
  height_ratios = [2.6, 0.9, 0.24] + [1.0] * n_data + [0.6]
  total_rows = len(height_ratios)
  gs = gridspec.GridSpec(total_rows, 4, width_ratios=[3.2, 1.4, 1.4, 6.0], height_ratios=height_ratios, hspace=0.06, wspace=0.10)

  # Header
  # Full-width header cell
  ax_header = plt.subplot(gs[0, :])
  ax_header.axis('off')
  header_lines = [
    f"Model: {meta.model_name}",
    f"Trainer: {meta.trainer}",
    f"Dataset: {meta.dataset_name}",
  ]
  y_positions = [0.85, 0.55, 0.25]
  sizes = [14, 12, 12]
  weights = ['bold', None, None]
  for txt, y, sz, w in zip(header_lines, y_positions, sizes, weights, strict=False):
    ax_header.text(0.0, y, txt, fontsize=sz, fontweight=(w or 'normal'), va='center', ha='left')

  # Column headings row under header
  ax_lbl0 = plt.subplot(gs[1, 0])
  ax_lbl0.axis('off')
  ax_lbl0.text(0.0, 0.5, "Series", fontsize=12, fontweight='bold', va='center', ha='left')
  ax_lbl1 = plt.subplot(gs[1, 1])
  ax_lbl1.axis('off')
  ax_lbl1.text(0.0, 0.5, "Mean", fontsize=12, fontweight='bold', va='center', ha='left')
  ax_lbl2 = plt.subplot(gs[1, 2])
  ax_lbl2.axis('off')
  ax_lbl2.text(0.0, 0.5, "Max", fontsize=12, fontweight='bold', va='center', ha='left')
  ax_lbl3 = plt.subplot(gs[1, 3])
  ax_lbl3.axis('off')
  ax_lbl3.text(0.0, 0.5, "MAE", fontsize=12, fontweight='bold', va='center', ha='left')

  # Rows
  # Start after header row (0), headings row (1), and spacer row (2)
  current = 3
  for idx, (label, abs_mean, abs_min, abs_max) in enumerate(rows):
    ax_l = plt.subplot(gs[current, 0])
    ax_l.axis('off')
    ax_l.text(0.0, 0.5, label, fontsize=11, va='center', ha='left')

    # Mean and Max value cells
    ax_mean = plt.subplot(gs[current, 1])
    ax_mean.axis('off')
    if abs_mean is not None:
      ax_mean.text(0.0, 0.5, f"{float(abs_mean):.5f}", fontsize=10, va='center', ha='left')
    else:
      ax_mean.text(0.0, 0.5, "n/a", fontsize=10, va='center', ha='left')

    ax_max = plt.subplot(gs[current, 2])
    ax_max.axis('off')
    if abs_max is not None:
      ax_max.text(0.0, 0.5, f"{float(abs_max):.5f}", fontsize=10, va='center', ha='left')
    else:
      ax_max.text(0.0, 0.5, "n/a", fontsize=10, va='center', ha='left')

    ax_p = plt.subplot(gs[current, 3])
    if bar_logscale:
      ax_p.set_xscale('log')
    ax_p.set_xlim(x_min, x_max)
    ax_p.set_ylim(0.2, 0.8)
    ax_p.set_yticks([])
    # Show axis only on last row
    is_last = (idx == len(rows) - 1)
    if is_last:
      ax_p.spines['bottom'].set_visible(True)
      ax_p.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, pad=3, labelsize=10, length=3)
      if bar_logscale:
        log_locator = mticker.LogLocator(base=10)
        minor_locator = mticker.LogLocator(base=10, subs=(2, 5))
        ax_p.xaxis.set_major_locator(log_locator)
        ax_p.xaxis.set_minor_locator(minor_locator)
        ax_p.xaxis.set_major_formatter(mticker.LogFormatterExponent(base=10))
        ax_p.xaxis.set_minor_formatter(mticker.NullFormatter())
      else:
        ax_p.set_xticks(np.linspace(x_min, x_max, 6))
    else:
      ax_p.set_xticks([])
      ax_p.set_xticklabels([])
      ax_p.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
      ax_p.spines['bottom'].set_visible(False)

    # Guide lines aligned to ticks
    if bar_logscale:
      log_min = int(math.floor(math.log10(max(x_min, 1e-300))))
      log_max = int(math.ceil(math.log10(max(x_max, 1e-300))))
      for d in range(log_min, log_max + 1):
        t_major = 10 ** d
        if x_min <= t_major <= x_max:
          ax_p.axvline(t_major, color='lightgray', linewidth=0.3, alpha=0.6, zorder=0)
        for mult in (2, 5):
          t_minor = mult * (10 ** d)
          if x_min <= t_minor <= x_max:
            ax_p.axvline(t_minor, color='lightgray', linewidth=0.25, alpha=0.5, zorder=0)
    else:
      for t in np.linspace(x_min, x_max, 6):
        ax_p.axvline(t, color='lightgray', linewidth=0.3, alpha=0.6, zorder=0)

    # Draw a single bar per row: abserror mean with min-max whiskers
    height = 0.30
    y_pos = 0.5
    if abs_mean is not None:
      x_abs = max(float(abs_mean), x_min) if bar_logscale else float(abs_mean)
      ax_p.barh([y_pos], [x_abs], height=height, color=['steelblue'], zorder=1)
      if abs_min is not None and abs_max is not None:
        left = max((float(abs_mean) - float(abs_min)), 0.0)
        right = max((float(abs_max) - float(abs_mean)), 0.0)
        ax_p.errorbar([x_abs], [y_pos], xerr=[[left], [right]], fmt='none', ecolor='black', elinewidth=0.8, capsize=2, capthick=0.8, zorder=2)

    current += 1

  # Footer row blank for spacing / to avoid crop
  ax_f = plt.subplot(gs[-1, :])
  ax_f.axis('off')

  os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
  fig.savefig(out_path, bbox_inches='tight', pad_inches=0.2)
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser(description="Per-model figure: train/validation MAE and benchmark series MAEs.")
  parser.add_argument('--root', type=str, default='./data/model', help='Root folder containing model families and runs')
  parser.add_argument('--out', type=str, default='./out/model-benchmarks', help='Output folder for figures')
  parser.add_argument('--bar_logscale', action='store_true', help='Use logarithmic x-scale for the bar plots')
  parser.add_argument('--fig_scale', type=float, default=1.0, help='Figure width scale factor (>=1 widens)')
  args = parser.parse_args()

  root = Path(args.root).resolve()
  out_root = Path(args.out)
  if not root.exists():
    raise SystemExit(f"Root not found: {root}")

  count = 0
  for trainer_dir in sorted(_iter_trainer_dirs(root)):
    meta = _load_meta(trainer_dir, root)
    if meta is None:
      continue
    metrics = _collect_train_val(trainer_dir)
    bench = _collect_benchmarks(trainer_dir)
    out_file = out_root / f"{meta.family}__{meta.model_name}__{meta.dataset_name}__{meta.trainer}.pdf"
    _make_model_figure(meta, metrics, bench, str(out_file), bar_logscale=args.bar_logscale, fig_scale=args.fig_scale)
    count += 1

  print(f"Wrote {count} per-model figures into: {out_root}")


if __name__ == '__main__':
  main()


