#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (C) 2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Visualize Sim2Real comparison metrics across subfolders.

Given a Sim2Real ROOT folder that contains one or more basis/linewidth approach
subfolders (each with per-series "*_metrics.json" files), this script aggregates
and plots per-series and overall comparison metrics across those approaches.

Usage:
  python3 visualize_sim2real.py /path/to/data/sim2real
  # Optional arguments:
  #   --output <folder>     Where to save plots/CSV (default: the ROOT folder)
  #   --metrics <list>      Which summary metrics to plot (default: mae_overall rmse_overall corr_overall cosine_overall)

It will produce under ROOT (or --output):
  - plots/per_series/<series>_<metric>.png  (bar plots across approaches)
  - plots/overall/overall_<metric>.png      (bar plots across approaches)
  - plots/summary_metrics.csv               (table with per-series and overall summaries)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import re


def find_metrics_jsons(input_dir: Path):
  """Recursively find all *_metrics.json files under input_dir."""
  return sorted(input_dir.rglob("*_metrics.json"))


def parse_prefix_and_series(file_path: Path):
  """Extract series id and approach label from filename.

  Filename format from compare.save_prefix: "<b_id>_<variant>[_LWmed-XX.XX]_metrics.json"
  Special case: "all[_LWmed-XX.XX]_metrics.json" for combined.
  """
  name = file_path.stem  # without .json
  if name.endswith("_metrics"):
    name = name[:-8]
  # 'all' or 'b_id_variant' possibly with _LWmed-xx.xx suffix
  parts = name.split("_")
  if parts[0] == "all":
    series = "all"
    approach = "_".join(parts[1:]) if len(parts) > 1 else "default"
  else:
    # series is first two parts; remainder is approach tag
    if len(parts) >= 2:
      series = "_".join(parts[0:2])
      approach = "_".join(parts[2:]) if len(parts) > 2 else "default"
    else:
      series = parts[0]
      approach = "default"
  if approach == "":
    approach = "default"
  return series, approach


def load_metrics(file_path: Path, metrics_to_plot: list[str]):
  with open(file_path) as f:
    data = json.load(f)
  # Defensive extraction
  summary = data.get("summary", {})
  basis = data.get("basis", {})
  linewidth = data.get("linewidth", {})
  # Try to load LW-MC trials summary if present
  lw_mc = None
  try:
    sibling = file_path.parent / (file_path.stem.replace("_metrics", "_metrics_lw_mc") + file_path.suffix)
    if sibling.exists():
      with open(sibling) as f2:
        mc = json.load(f2)
      trials = mc.get("summary_trials", [])
      if isinstance(trials, list) and len(trials) > 0:
        import numpy as _np
        means = {}
        stds = {}
        for m in metrics_to_plot:
          vals = []
          for t in trials:
            v = t.get(m)
            if v is not None:
              vals.append(float(v))
          if len(vals) > 0:
            arr = _np.array(vals, dtype=float)
            means[m] = float(_np.mean(arr))
            stds[m] = float(_np.std(arr))
        lw_mc = {"mean": means, "std": stds}
  except Exception:
    lw_mc = None
  return {
    "file": str(file_path),
    "summary": summary,
    "basis": basis,
    "linewidth": linewidth,
    "lw_mc": lw_mc,
  }


def aggregate_overall(per_series: dict, approaches: set, metrics_to_plot: list):
  """Aggregate metrics across series for each approach.

  If an 'all' series exists for an approach, use that summary directly.
  Otherwise compute the mean across available series entries for that approach.
  """
  overall = {}
  for approach in sorted(approaches):
    # If explicit combined present
    if "all" in per_series and approach in per_series["all"]:
      overall[approach] = per_series["all"][approach]["summary"]
      continue
    # Else average across series that contain this approach
    vals = defaultdict(list)
    for series, appr_map in per_series.items():
      if series == "all":
        continue
      if approach in appr_map and "summary" in appr_map[approach]:
        for m in metrics_to_plot:
          v = appr_map[approach]["summary"].get(m, None)
          if v is not None:
            vals[m].append(float(v))
    overall[approach] = {m: (float(np.mean(vals[m])) if len(vals[m]) > 0 else None) for m in metrics_to_plot}
  return overall


def plot_bar(ax, labels, values, title, ylabel, yerr=None):
  x = np.arange(len(labels))
  if yerr is not None:
    try:
      yerr_arr = np.array(yerr, dtype=float)
    except Exception:
      yerr_arr = None
  else:
    yerr_arr = None
  ax.bar(x, values, yerr=yerr_arr, capsize=3)
  ax.set_xticks(x)
  ax.set_xticklabels(labels, rotation=45, ha='right', va='top', fontsize=8)
  ax.set_title(title)
  ax.set_ylabel(ylabel)
  ax.grid(axis='y', alpha=0.3)


def plot_per_series(per_series: dict, approaches: set, metrics_to_plot: list, out_dir: Path):
  out_dir.mkdir(parents=True, exist_ok=True)
  for series, appr_map in per_series.items():
    if series == "all":
      continue
    for metric in metrics_to_plot:
      labels = []
      values = []
      errs = []
      for approach in sorted(approaches):
        labels.append(approach if approach != "default" else "fixed")
        val = None
        if approach in appr_map and "summary" in appr_map[approach]:
          val = appr_map[approach]["summary"].get(metric, None)
        values.append(np.nan if val is None else float(val))
        # LW-MC std as error bar if available
        err = None
        try:
          lw_mc = appr_map[approach].get("lw_mc") if approach in appr_map else None
          if lw_mc and "std" in lw_mc:
            err = lw_mc["std"].get(metric)
        except Exception:
          err = None
        errs.append(err if err is not None else np.nan)
      # Dynamic width based on number of approaches
      fig_width = max(10, 0.6*len(labels) + 4)
      fig, ax = plt.subplots(figsize=(fig_width, 6.0))
      # If all error bars are nan, suppress them
      yerr = errs if np.any(np.isfinite(errs)) else None
      plot_bar(ax, labels, values, title=f"{series} - {metric}", ylabel=metric, yerr=yerr)
      fig.subplots_adjust(bottom=0.38)
      fig.savefig(out_dir / f"{series}_{metric}.png", dpi=200, bbox_inches='tight')
      plt.close(fig)


def plot_overall(per_series: dict, overall: dict, metrics_to_plot: list, out_dir: Path):
  out_dir.mkdir(parents=True, exist_ok=True)
  for metric in metrics_to_plot:
    labels = []
    values = []
    errs = []
    for approach, summ in sorted(overall.items()):
      labels.append(approach if approach != "default" else "fixed")
      val = summ.get(metric, None)
      values.append(np.nan if val is None else float(val))
      # Use 'all' lw_mc std if present
      err = None
      try:
        appr_map = per_series.get("all", {})
        lw_mc = appr_map.get(approach, {}).get("lw_mc", None)
        if lw_mc and "std" in lw_mc:
          err = lw_mc["std"].get(metric)
      except Exception:
        err = None
      errs.append(err if err is not None else np.nan)
    fig_width = max(10, 0.6*len(labels) + 4)
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))
    yerr = errs if np.any(np.isfinite(errs)) else None
    plot_bar(ax, labels, values, title=f"Overall - {metric}", ylabel=metric, yerr=yerr)
    fig.subplots_adjust(bottom=0.38)
    fig.savefig(out_dir / f"overall_{metric}.png", dpi=200, bbox_inches='tight')
    plt.close(fig)


def export_csv(per_series: dict, overall: dict, metrics_to_plot: list, out_dir: Path):
  out_dir.mkdir(parents=True, exist_ok=True)
  csv_path = out_dir / "summary_metrics.csv"
  series_list = sorted([s for s in per_series.keys() if s != "all"]) + ["overall"]
  # Build header
  approaches = set()
  for s in per_series:
    for a in per_series[s].keys():
      approaches.add(a)
  approaches = sorted(list(approaches))
  with open(csv_path, 'w') as f:
    # header
    header = ["series", "approach"] + metrics_to_plot
    f.write(",".join(header) + "\n")
    # per-series rows
    for series in sorted([s for s in per_series.keys() if s != "all"]):
      for approach in approaches:
        row = [series, approach if approach != "default" else "fixed"]
        summ = per_series[series].get(approach, {}).get("summary", {})
        for m in metrics_to_plot:
          v = summ.get(m, "")
          row.append(str(v) if v is not None else "")
        f.write(",".join(row) + "\n")
    # overall rows
    for approach in approaches:
      row = ["overall", approach if approach != "default" else "fixed"]
      summ = overall.get(approach, {})
      for m in metrics_to_plot:
        v = summ.get(m, "")
        row.append(str(v) if v is not None else "")
      f.write(",".join(row) + "\n")


def main():
  parser = argparse.ArgumentParser(description="Visualize Sim2Real comparison metrics across subfolders of a sim2real root folder")
  parser.add_argument("folder", help="Sim2Real root folder containing basis_tag subfolders")
  parser.add_argument("--output", default=None, help="Folder to save plots; defaults to the root folder")
  parser.add_argument("--metrics", nargs="*", default=["mae_overall", "rmse_overall", "corr_overall", "cosine_overall"],
                      help="Summary metrics to plot")
  parser.add_argument("--baseline_regex", default=r"fid-a-2d_.*_2\.0_.*",
                      help="Regex to select baseline approach (default: fid-a-2d @ 2.0 Hz)")
  parser.add_argument("--prefer_noise", action="store_true",
                      help="Prefer baseline folder that includes noise tag if multiple baselines match")
  args = parser.parse_args()

  root_dir = Path(args.folder)
  if not root_dir.exists():
    raise RuntimeError(f"Folder not found: {root_dir}")
  # Determine basis_tag subfolders (directories directly under root)
  input_dirs = [p for p in root_dir.iterdir() if p.is_dir()]
  if len(input_dirs) == 0:
    raise RuntimeError(f"No subfolders found under {root_dir}")
  # Output defaults to the root folder
  out_dir = Path(args.output) if args.output else root_dir
  plot_dir = out_dir

  # Load and organize per-series across all inputs
  per_series = defaultdict(dict)  # {series: {approach: data}}
  approaches = set()
  for in_dir in input_dirs:
    basis_tag = in_dir.name
    files = find_metrics_jsons(in_dir)
    for fp in files:
      series, approach = parse_prefix_and_series(fp)
      # Use subfolder (basis_tag) as the approach label for cross-folder comparison
      approach_key = basis_tag
      data = load_metrics(fp, args.metrics)
      per_series[series][approach_key] = data
      approaches.add(approach_key)

  # Aggregate overall
  overall = aggregate_overall(per_series, approaches, args.metrics)

  # Determine baseline approach key
  def select_baseline(approaches_set: set[str], regex: str, prefer_noise: bool) -> str | None:
    candidates = [a for a in approaches_set if re.search(regex, a) is not None]
    if not candidates:
      return None
    if prefer_noise:
      noisy = [a for a in candidates if "noiseT" in a]
      if noisy:
        return sorted(noisy)[0]
    return sorted(candidates)[0]

  baseline_key = select_baseline(approaches, args.baseline_regex, args.prefer_noise)

  # Compute and export gap summary vs baseline
  def export_gap_csv(per_series_map: dict, approaches_set: set[str], base_key: str | None,
                     metrics_to_plot: list[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gap_summary.csv"
    with open(path, "w") as f:
      header = ["series", "approach", "metric", "value", "baseline", "delta", "percent"]
      f.write(",".join(header) + "\n")
      for series, appr_map in per_series_map.items():
        for approach in sorted(approaches_set):
          summ = appr_map.get(approach, {}).get("summary", {})
          for m in metrics_to_plot:
            val = summ.get(m, None)
            if val is None:
              continue
            base_val = None
            if base_key is not None:
              base_val = appr_map.get(base_key, {}).get("summary", {}).get(m, None)
            delta = None
            pct = None
            try:
              if base_val is not None:
                delta = float(val) - float(base_val)
                pct = (delta / float(base_val) * 100.0) if float(base_val) != 0.0 else None
            except Exception:
              delta = None; pct = None
            row = [series, approach, m,
                   str(val if val is not None else ""),
                   str(base_val if base_val is not None else ""),
                   ("%.6f" % delta) if delta is not None else "",
                   ("%.2f" % pct) if pct is not None else ""]
            f.write(",".join(row) + "\n")

  export_gap_csv(per_series, approaches, baseline_key, args.metrics, plot_dir)

  # Plot overall delta vs baseline (from series='all') per metric
  def plot_overall_gap(per_series_map: dict, approaches_set: set[str], base_key: str | None,
                       metrics_to_plot: list[str], out_dir: Path):
    if base_key is None:
      return
    data_all = per_series_map.get("all", {})
    base_summary = data_all.get(base_key, {}).get("summary", {})
    if not base_summary:
      return
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in metrics_to_plot:
      labels = []
      deltas = []
      errs = []
      base_val = base_summary.get(metric, None)
      if base_val is None:
        continue
      for approach in sorted(approaches_set):
        if approach == base_key:
          continue
        summ = data_all.get(approach, {}).get("summary", {})
        val = summ.get(metric, None)
        if val is None:
          continue
        try:
          d = float(val) - float(base_val)
        except Exception:
          d = np.nan
        labels.append(approach)
        deltas.append(d)
        # error bars from LW-MC if available
        lw_mc = data_all.get(approach, {}).get("lw_mc", {})
        err = None
        if lw_mc and "std" in lw_mc:
          err = lw_mc["std"].get(metric)
        errs.append(err if err is not None else np.nan)
      if not labels:
        continue
      fig_width = max(10, 0.6*len(labels) + 4)
      fig, ax = plt.subplots(figsize=(fig_width, 6.0))
      yerr = errs if np.any(np.isfinite(errs)) else None
      plot_bar(ax, labels, deltas, title=f"Overall Δ vs baseline ({base_key}) - {metric}", ylabel=f"Δ {metric}", yerr=yerr)
      fig.subplots_adjust(bottom=0.38)
      fig.savefig(out_dir / f"overall_gap_{metric}.png", dpi=200, bbox_inches='tight')
      plt.close(fig)

  plot_overall_gap(per_series, approaches, baseline_key, args.metrics, plot_dir / "overall")

  # Plots
  # Matrix-style plots: one figure per metric with all series as rows and approaches as bars
  for metric in args.metrics:
    # Build series order (place 'all' at bottom if present)
    series_order = sorted([s for s in per_series.keys() if s != "all"]) + (["all"] if "all" in per_series else [])
    labels_series = series_order
    labels_approach = sorted(list(approaches))
    # Collect values matrix [num_series, num_approaches]
    values = np.full((len(series_order), len(labels_approach)), np.nan, dtype=float)
    for i, series in enumerate(series_order):
      for j, appr in enumerate(labels_approach):
        summ = per_series.get(series, {}).get(appr, {}).get("summary", {})
        val = summ.get(metric, None)
        if val is not None:
          try:
            values[i, j] = float(val)
          except Exception:
            values[i, j] = np.nan
    # Plot horizontal grouped bars
    fig, ax = plt.subplots(figsize=(max(10, 1 + 1.3*len(labels_approach)), max(6, 0.65*len(labels_series))))
    y = np.arange(len(labels_series))
    bar_h = 0.8 / max(1, len(labels_approach))
    for j, appr in enumerate(labels_approach):
      ax.barh(y + (j - (len(labels_approach)-1)/2)*bar_h, values[:, j], height=bar_h, label=appr)
    ax.set_yticks(y)
    ax.set_yticklabels(labels_series)
    ax.set_xlabel(metric)
    ax.set_title(f"{metric} per series across approaches")
    ax.grid(axis='x', alpha=0.3)
    # Place legend outside the plot (right side) to avoid covering content
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0., fontsize=8)
    (plot_dir / "matrix").mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "matrix" / f"metric_{metric}.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

  # Keep previous detailed plots as optional
  plot_per_series(per_series, approaches, args.metrics, plot_dir / "per_series")
  plot_overall(per_series, overall, args.metrics, plot_dir / "overall")

  # CSV export
  export_csv(per_series, overall, args.metrics, plot_dir)

  print(f"Saved plots and CSV to: {plot_dir}")


if __name__ == "__main__":
  main()


