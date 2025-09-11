#!/usr/bin/env python3
#
# compare-simulations.py
#
# SPDX-FileCopyrightText: Copyright (C) 2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compare two simulated basis sets discovered by FID-A style filename pattern.

Run from repository root, e.g.:
 ./etc/compare-simulations.py "FIDA2D_*_MEGAPRESS_EDITOFF_2.00_2000_4096_123.23" data/basis-dist/fid-a-2d data/basis/fid-a-2d
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from mrsnet.spectrum import Spectrum


def parse_pattern(pattern: str) -> tuple[str, str, float, int, int, float]:
  """Parse a FID-A filename-like pattern.

  Expected form: FIDA2D_*_MEGAPRESS_EDITOFF_2.00_2000_4096_123.23
  The metabolite token is ignored (wildcard). Acquisition token is unused here.

  Returns: (source_prefix, pulse_sequence, linewidth, sample_rate, samples, omega)
  """
  parts = pattern.split("_")
  if len(parts) < 8:
    raise RuntimeError(f"Pattern not understood: {pattern}")
  source_prefix = parts[0]  # e.g., FIDA2D
  pulse_sequence = parts[2].lower()  # MEGAPRESS -> megapress
  try:
    linewidth = float(parts[4])
    sample_rate = int(parts[5])
    samples = int(parts[6])
    omega = float(parts[7])
  except Exception as exc:
    raise RuntimeError(f"Cannot parse numeric fields from pattern: {pattern}") from exc
  return source_prefix, pulse_sequence, linewidth, sample_rate, samples, omega

def discover_metabolites(dir_path: str, source_prefix: str, pulse_sequence: str,
                         linewidth: float, sample_rate: int, samples: int, omega: float) -> list[str]:
  """List metabolites present in a basis folder by matching FID-A style filenames.

  Scans '<dir_path>/basis_files'. Returns sorted unique metabolite names.
  """
  basis_dir = os.path.join(dir_path, "basis_files")
  if not os.path.isdir(basis_dir):
    # Also allow SU-3T files directly in dir
    basis_dir = dir_path
    if not os.path.isdir(basis_dir):
      return []
  metabolites = set()
  for fname in os.listdir(basis_dir):
    if not fname.endswith(".mat"):
      continue
    if not fname.startswith(source_prefix + "_"):
      continue
    parts = fname[:-4].split("_")
    if len(parts) < 8:
      continue
    try:
      met = parts[1]
      seq = parts[2].lower()
      _ = parts[3]  # EDITON/OFF; not used here
      lw = float(parts[4])
      sr = int(parts[5])
      smp = int(parts[6])
      omg = float(parts[7])
    except Exception: # noqa: S112
      # Skip unparsable files quietly
      continue
    if seq != pulse_sequence:
      continue
    if abs(lw - linewidth) >= 1e-2:
      continue
    if sr != sample_rate or smp != samples:
      continue
    if abs(omg - omega) >= 1e-2:
      continue
    metabolites.add(met)
  return sorted(metabolites)

def load_spectra(dir_path: str, source_prefix: str, source_name: str,
                 pulse_sequence: str, linewidth: float, sample_rate: int,
                 samples: int, omega: float, metabolites: list[str]) -> tuple[dict[str, dict[str, Spectrum]], set[str]]:
  """Load spectra directly from FID-A .mat files matching parameters.

  Returns: (spectra_map, acquisitions)
  - spectra_map[metabolite][acquisition] = Spectrum
  - acquisitions: set of acquisition names discovered
  """
  basis_dir = os.path.join(dir_path, "basis_files")
  if not os.path.isdir(basis_dir):
    basis_dir = dir_path
    if not os.path.isdir(basis_dir):
      return {}, set()

  spectra: dict[str, dict[str, Spectrum]] = {}
  acquisitions: set[str] = set()

  for fname in os.listdir(basis_dir):
    if not fname.endswith(".mat"):
      continue
    if not fname.startswith(source_prefix + "_"):
      continue
    parts = fname[:-4].split("_")
    if len(parts) < 8:
      continue
    try:
      met = parts[1]
      seq = parts[2].lower()
      if parts[3] != "EDITON" and parts[3] != "EDITOFF":
        raise RuntimeError(f"Neither EDITON nor EDITOFF in filename: {fname}")
      lw = float(parts[4])
      sr = int(parts[5])
      smp = int(parts[6])
      omg = float(parts[7])
    except Exception:  # noqa: S112
      continue
    if met not in metabolites:
      continue
    if seq != pulse_sequence:
      continue
    if abs(lw - linewidth) >= 1e-2:
      continue
    if sr != sample_rate or smp != samples:
      continue
    if abs(omg - omega) >= 1e-2:
      continue

    fpath = os.path.join(basis_dir, fname)
    try:
      spec = Spectrum.load_fida(fpath, fname[:-4], source_name)
    except Exception as exc:
      print(f"Warning: failed to load {fpath}: {exc}")
      continue
    # Basic sanity checks
    if spec.pulse_sequence != pulse_sequence:
      continue
    if spec.metabolites and spec.metabolites[0] != met:
      # Keep filename-derived metabolite key to match across sets
      pass
    spectra.setdefault(met, {})[spec.acquisition] = spec
    acquisitions.add(spec.acquisition)

  return spectra, acquisitions

def rmse(a: np.ndarray, b: np.ndarray) -> float:
  """Root-mean-square error between complex arrays (magnitude-wise)."""
  d = np.ravel(np.abs(a - b))
  return float(np.sqrt(np.mean(d * d)))

def compare_spectra(sp1: dict[str, dict[str, Spectrum]],
                    sp2: dict[str, dict[str, Spectrum]],
                    acqs1: set[str], acqs2: set[str],
                    out_dir: str, title: str) -> dict[str, dict[str, float]]:
  """Compare overlapping metabolites and acquisitions from pre-loaded spectra; save plots.

  Returns: metrics[metabolite][acquisition] = rmse_value
  """
  os.makedirs(out_dir, exist_ok=True)
  common_mets = sorted(set(sp1.keys()) & set(sp2.keys()))
  common_acqs = sorted(acqs1 & acqs2)
  metrics: dict[str, dict[str, float]] = {}

  for met in common_mets:
    metrics[met] = {}
    met_acqs = sorted(set(sp1[met].keys()) & set(sp2[met].keys()))
    for acq in met_acqs:
      s1 = sp1[met][acq]
      s2 = sp2[met][acq]
      f1, nu1 = s1.rescale_fft()  # aligned ppm window
      f2, nu2 = s2.rescale_fft()
      if f1.shape != f2.shape or not np.allclose(nu1, nu2):
        re = np.interp(nu1, nu2, np.real(f2))
        im = np.interp(nu1, nu2, np.imag(f2))
        f2i = re + 1j * im
      else:
        f2i = f2
      metrics[met][acq] = rmse(f1, f2i)

      fig, ax = plt.subplots(1, 1)
      ax.plot(nu1, np.abs(f1), label="|Set1|", color="#1f77b4")
      ax.plot(nu1, np.abs(f2i), label="|Set2|", color="#ff7f0e", alpha=0.7)
      ax.set_title(f"{title}\n{met} - {acq}")
      ax.set_xlabel("Frequency (ppm)")
      ax.set_ylabel("Magnitude")
      ax.legend(loc="best")
      ax.invert_xaxis()
      fig.tight_layout()
      fig.savefig(os.path.join(out_dir, f"{met}_{acq}.png"), dpi=300)
      plt.close(fig)

  # Summary bar plot per acquisition
  for acq in common_acqs:
    vals = [metrics[m][acq] for m in common_mets if acq in metrics[m]]
    if len(vals) == 0:
      continue
    fig, ax = plt.subplots(1, 1)
    ax.bar(range(len(vals)), vals)
    ax.set_xticks(range(len([m for m in common_mets if acq in metrics[m]])))
    ax.set_xticklabels([m for m in common_mets if acq in metrics[m]], rotation=45, ha='right')
    ax.set_ylabel("RMSE")
    ax.set_title(f"RMSE across metabolites - {acq}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"RMSE_{acq}.png"), dpi=300)
    plt.close(fig)

  # Write a simple TSV summary
  with open(os.path.join(out_dir, "summary.tsv"), "w") as f:
    f.write("metabolite\t" + "\t".join(common_acqs) + "\n")
    for m in common_mets:
      f.write(m + "\t" + "\t".join(
        f"{metrics[m][a]:.6g}" if a in metrics[m] else "" for a in common_acqs
      ) + "\n")

  return metrics

def main():
  """CLI entrypoint to compare two basis directories by pattern."""
  parser = argparse.ArgumentParser(description="Compare two simulated basis sets by FID-A pattern")
  parser.add_argument("pattern", type=str,
                      help="Pattern like FIDA2D_*_MEGAPRESS_EDITOFF_2.00_2000_4096_123.23")
  parser.add_argument("dir1", type=str,
                      help="Directory of first basis set (e.g., data/basis-dist/fid-a-2d)")
  parser.add_argument("dir2", type=str,
                      help="Directory of second basis set (e.g., data/basis/fid-a-2d)")
  parser.add_argument("--out", dest="out", type=str, default="compare-simulations_out",
                      help="Output directory for plots and summary")
  args = parser.parse_args()

  source_prefix, pulse_sequence, linewidth, sample_rate, samples, omega = parse_pattern(args.pattern)

  # Determine Basis.source from prefix (FIDA2D -> fid-a-2d)
  m = re.match(r"FIDA([A-Z0-9]+)$", source_prefix)
  if not m:
    raise RuntimeError(f"Unrecognised source prefix: {source_prefix}")
  source_name = "fid-a-" + m.group(1).lower()

  mets1 = discover_metabolites(args.dir1, source_prefix, pulse_sequence, linewidth, sample_rate, samples, omega)
  mets2 = discover_metabolites(args.dir2, source_prefix, pulse_sequence, linewidth, sample_rate, samples, omega)
  if len(mets1) == 0:
    raise RuntimeError(f"No metabolites discovered in {args.dir1} for pattern {args.pattern}")
  if len(mets2) == 0:
    raise RuntimeError(f"No metabolites discovered in {args.dir2} for pattern {args.pattern}")

  common = sorted(set(mets1).intersection(set(mets2)))
  if len(common) == 0:
    raise RuntimeError("No overlapping metabolites between the two sets.")

  # Load spectra directly (no simulation) restricted to common metabolites
  sp1, acqs1 = load_spectra(args.dir1, source_prefix, source_name, pulse_sequence, linewidth, sample_rate, samples, omega, common)
  sp2, acqs2 = load_spectra(args.dir2, source_prefix, source_name, pulse_sequence, linewidth, sample_rate, samples, omega, common)
  if len(sp1) == 0:
    raise RuntimeError(f"No spectra loaded from {args.dir1} for pattern {args.pattern}")
  if len(sp2) == 0:
    raise RuntimeError(f"No spectra loaded from {args.dir2} for pattern {args.pattern}")

  title = f"{source_name} {pulse_sequence} @ {omega:.2f} MHz, LW {linewidth}, SR {sample_rate}, N {samples}\nSet1: {args.dir1} vs Set2: {args.dir2}"
  compare_spectra(sp1, sp2, acqs1, acqs2, args.out, title)

  print(f"Compared {len(common)} metabolites across acquisitions. Output in: {args.out}")

if __name__ == "__main__":
  main()
