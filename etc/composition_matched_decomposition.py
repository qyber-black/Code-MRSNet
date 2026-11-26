#!/usr/bin/env python3
#
# composition_matched_decomposition.py - MRSNet - linewidth/noise decomposition of the composition-matched gap
#
# SPDX-FileCopyrightText: Copyright (C) 2026 Frank C Langbein
# SPDX-License-Identifier: AGPL-3.0-or-later

r"""Decompose the composition-matched simulation-to-phantom gap into linewidth and noise parts.

The composition-matched control (etc/composition_matched.py) simulates the 144
phantom preparation compositions with the fixed 2 Hz FID-A basis and ten seeded
noise realisations (sigma U[0, 0.03] per mixture) and quantifies them with the
five released fixed models. This script reruns that control with frozen
checkpoints (inference only) in four variants, all with the same ten noise seeds:

* A-baseline: identical to the control. Its exported inputs and its noiseless
  and matched-noisy MAEs must reproduce the released control exactly (input
  hashes equal, MAEs within 1e-12); otherwise the run stops before any other
  variant is scored.
* B-linewidth: each composition is simulated with the FID-A basis whose
  linewidth is nearest to that spectrum's fitted effective linewidth (released
  sim2real basis-mixture OFF fit); noise as A.
* C-noise: 2 Hz basis; each spectrum gets one fixed sigma, calibrated so that
  the simulated inputs have the same noise level as the phantom input (below).
* D-linewidth-noise: B's basis per spectrum and a fixed sigma calibrated on B's
  basis.

Noise level (one definition for phantom and simulated inputs): the exported
OFF spectrum as quantified by the YAE (shared max(abs(OFF)) normalisation over
-4.5..-1 ppm, real and imaginary parts) restricted to the signed ppm window
NOISE_WINDOW_PPM, which contains no resonance of the five metabolites. A
polynomial of degree DETREND_DEGREE is fitted separately to the real and
imaginary parts and removed; the noise level q is the root mean square of both
residuals (divided by the residual degrees of freedom). The simulator adds
complex time-domain Gaussian noise of SD sigma relative to each acquisition's
peak time-domain amplitude; q is not an analytic function of sigma (it depends
on composition, linewidth and the OFF maximum), so sigma is calibrated per
spectrum and basis: q0 is the noiseless level, qref the RMS level over
CALIBRATION_SEEDS at CALIBRATION_REF_SIGMA, the start value is
sigma = sqrt(max(q_phantom^2 - q0^2, 0)) * CALIBRATION_REF_SIGMA / sqrt(qref^2 - q0^2),
followed by CALIBRATION_ITERATIONS multiplicative updates
sigma <- sigma * q_phantom / q_sim(sigma). The calibration is validated on
simulated inputs with known sigma (log-uniform, one realisation, VALIDATION_SEED), which
measures the error of the whole estimator-plus-calibration chain.

The fixed-sigma noise is Dataset.add_noise with the drawn sigma replaced: it
consumes the same random stream draw for draw (same seed, same uniform draws,
same Gaussian draws), recombines the difference spectrum and applies the same
B0 re-correction (Spectrum.correct_b0_multi) per mixture. Calibration,
validation and the C and D inputs all use this one path. Before calibration
it is compared with Dataset.add_noise byte for byte on the exported inputs, for
both the 2 Hz and the linewidth-matched bases (1.3-6.5 Hz),
all ten noise seeds and the sigma ranges MIRROR_SIGMA_MAX (the training range
and one that covers the calibrated sigmas, which must stay below the largest
sigma drawn there); the same comparison without the B0 re-correction is
recorded to show that the check can fail. MAEs use independent maximum
normalisation as in the control; the fraction of the phantom gap closed is
(MAE_variant - MAE_A) / (MAE_phantom - MAE_A), with MAE_phantom the CPU phantom
MAE of the released control.

The linewidth grid is pinned: the complete basis linewidths in the folder must
equal BASIS_LINEWIDTH_GRID, the linewidth table LINEWIDTH_CSV_SHA256, the
linewidths used BASIS_LINEWIDTHS_USED, and the SHA-256 of every basis file
loaded (recorded one by one) must give BASIS_USED_SHA256.

Known limitation of the noise level: the shared max(abs(OFF)) is the NAA or Cr
singlet in all but ten spectra with failed water suppression, nine in E6, where
residual water at the -4.5 ppm export edge is the maximum, and GEL02 in E11,
where a broad residual-water signal is (protocol.json lists them and the
singlet-to-maximum ratios), so the matched noise is lower relative to the
metabolites than in those phantom spectra. The phantom window noise is also more correlated than the white
simulator noise (lag-1 autocorrelation). Both are recorded in protocol.json
(noise.known_limitations); the estimator is kept as defined.

Run from the repository root (about 6 min on a 64-core CPU, most of it inference)::

  PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mrsnet-mpl CUDA_VISIBLE_DEVICES=-1 \
  TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  etc/composition_matched_decomposition.py

The thread settings are those of the recorded runs; other thread counts can
change the last bits of CPU inference and then fail the baseline gate. The run
refuses to start with other values (THREADS) unless --allow-thread-mismatch is
given, which warns and records the values. The calibrated sigmas depend on
np.linalg.lstsq (detrending), i.e. on the LAPACK/BLAS build of NumPy, its
runtime kernel (OpenBLAS DYNAMIC_ARCH selects one per CPU) and its thread
count; another BLAS can change their last bits and with them the C and D
inputs and predictions, which no gate covers (the A gate does not use them).
The BLAS build and runtime are recorded in protocol.json.

``--calibration-only`` maps linewidths, calibrates and validates the noise and
prints a summary without TensorFlow and without writing any file.
"""

import argparse
import copy
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# Run as a script from etc/: make the mrsnet library in the repository root importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_control():
  """Import etc/composition_matched.py by path, so its functions are reused, not copied."""
  path = Path(__file__).resolve().with_name("composition_matched.py")
  spec = importlib.util.spec_from_file_location("composition_matched", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


CM = _load_control()

VARIANTS = ("A-baseline", "B-linewidth", "C-noise", "D-linewidth-noise")
REFERENCE_DIR = "data/sim2real/paper-gaba-phantom/composition-matched"
DEFAULT_OUT = "data/sim2real/paper-gaba-phantom/composition-decomposition"
LINEWIDTH_CSV = "data/sim2real/linewidth/basis-mixture-off/linewidth_per_spectrum.csv"
REGISTRY = "data/model-dist/paper-model-registry/model_registry.json"
BASIS_NAME = "FIDA2D_{m}_MEGAPRESS_EDIT{a}_{lw:.2f}_2000_4096_123.23.mat"
BASIS_PATTERN = re.compile(r"FIDA2D_([A-Za-z0-9]+)_MEGAPRESS_EDIT(ON|OFF)_(\d+\.\d\d)_2000_4096_123\.23\.mat")
BASELINE_LINEWIDTH = 2.0
# Complete ON/OFF basis linewidths (Hz) of the released data/basis-dist; B and D map onto this grid.
BASIS_LINEWIDTH_GRID = (
  *(1.0, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.0, 3.2, 3.3, 3.4, 3.5),
  *(3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.8, 4.9, 5.0, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9),
  *(6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 7.0, 7.3, 7.5, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5),
  *(8.6, 8.7, 9.0, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0),
)
LINEWIDTH_CSV_SHA256 = "6fce6e458a3af7922edf69bf64c091b08abd722a556df14e31d6a120e16ac364"
# Basis linewidths loaded (2 Hz and those B and D map to), and the SHA-256 of the
# sorted lines "<file name> <sha256>\n" of their ten ON/OFF files each.
BASIS_LINEWIDTHS_USED = (
  *(1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.0, 3.3),
  *(3.4, 3.6, 3.7, 3.8, 3.9, 4.1, 4.2, 4.4, 4.5, 4.8, 4.9, 5.0, 5.3, 5.6, 5.8, 6.5),
)
BASIS_USED_SHA256 = "460ed4d4c15aeb31aa4ccade2da9180a6be7555384c3973758462edf7245c5e7"
# Thread settings of the recorded runs (see the module docstring).
THREADS = {"TF_NUM_INTRAOP_THREADS": "4", "TF_NUM_INTEROP_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
NOISE_SIGMA_MAX = 0.03  # Dataset.add_noise(1.0, "adc_normal", 0.0, NOISE_SIGMA_MAX, 0) in the control
# Export of the YAE inputs: the only released phantom inputs with real and imaginary parts.
NOISE_EXPORT = (["difference", "edit_off", "edit_on"], ["imaginary", "real"])
NOISE_WINDOW_PPM = (-1.6, -1.0)
DETREND_DEGREE = 2
CALIBRATION_SEEDS = tuple(range(20261140, 20261160))
CALIBRATION_REF_SIGMA = 0.01
CALIBRATION_ITERATIONS = 2
VALIDATION_SEED = 20261170
VALIDATION_SIGMA_RANGE = (0.002, 0.2)  # log-uniform; covers the calibrated phantom sigmas
# Upper ends of the U[0, max] sigma draws for which the fixed-sigma path is compared with Dataset.add_noise.
MIRROR_SIGMA_MAX = (NOISE_SIGMA_MAX, VALIDATION_SIGMA_RANGE[1])
# Signed ppm windows of the NAA (2.01 ppm) and Cr (3.015 ppm) singlets, one of which sets max(abs(OFF))
# unless residual water exceeds them (diagnostic of the max(abs(OFF)) normalisation).
SINGLET_WINDOWS_PPM = ((-2.11, -1.91), (-3.115, -2.915))
TOLERANCE = 1e-12


def _registry(code):
  """Read the five principal fixed-model entries and check their checkpoints."""
  entries = {
    e["paper_name"]: e
    for e in json.loads((code / REGISTRY).read_text())["instances"]
    if e["regime"] == "fixed" and e["paper_name"] in CM.MODELS
  }
  if set(entries) != set(CM.MODELS):
    raise ValueError("principal fixed checkpoints missing")
  sources = {REGISTRY: CM._sha256(code / REGISTRY)}
  for entry in entries.values():
    for kind in ("checkpoint", "metadata"):
      path = CM._child(code, entry[kind + "_path"])
      digest = CM._sha256(path)
      if digest != entry[kind + "_sha256"]:
        raise ValueError("deployed checkpoint or metadata changed")
      sources[str(path.relative_to(code))] = digest
  return entries, sources


def _available_linewidths(code):
  """Linewidths for which all five metabolites have ON and OFF basis files."""
  need = {(m, a) for m in CM.METABOLITES for a in ("ON", "OFF")}
  found = {}
  for name in os.listdir(code / CM.BASIS_DIR):
    match = BASIS_PATTERN.fullmatch(name)
    if match:
      found.setdefault(match[3], set()).add((match[1], match[2]))
  available = sorted(float(lw) for lw, have in found.items() if need <= have)
  if available != list(BASIS_LINEWIDTH_GRID) or sorted(float(lw) for lw in found) != available:
    raise ValueError("basis linewidth grid differs from BASIS_LINEWIDTH_GRID")
  return available, sorted(found)


def _linewidth_mapping(code, ids):
  """Map each spectrum's fitted effective linewidth to the nearest complete basis linewidth.

  Ties are resolved towards the smaller basis linewidth.
  """
  available, listed = _available_linewidths(code)
  if CM._sha256(code / LINEWIDTH_CSV) != LINEWIDTH_CSV_SHA256:
    raise ValueError("fitted linewidth table differs from LINEWIDTH_CSV_SHA256")
  with (code / LINEWIDTH_CSV).open(newline="") as stream:
    rows = {r["spectrum_id"]: r for r in csv.DictReader(stream)}
  if len(rows) != 144 or set(rows) != set(ids):
    raise ValueError("fitted linewidths do not cover the phantom cohort")
  grid = np.asarray(available)
  mapping = []
  for sid in ids:
    fitted = float(rows[sid]["linewidth_hz"])
    if not np.isfinite(fitted):
      raise ValueError("non-finite fitted linewidth for " + sid)
    basis = float(grid[np.argmin(np.round(np.abs(grid - fitted), 9))])
    mapping.append(
      {
        "spectrum_id": sid,
        "fitted_linewidth_hz": fitted,
        "basis_linewidth_hz": basis,
        "difference_hz": basis - fitted,
        "fit_accepted": rows[sid]["accepted"] == "True",
        "fit_flags": rows[sid]["rejection_reasons"],
      }
    )
  return mapping, available, listed


def _basis(code, lw, sources):
  """Load one released basis without allowing the library to simulate missing files.

  Records the SHA-256 of each of its ten files in sources.
  """
  from mrsnet.basis import Basis
  from mrsnet.cfg import Cfg

  folder = code / CM.BASIS_DIR
  names = [BASIS_NAME.format(m=m, a=a, lw=lw) for m in CM.METABOLITES for a in ("OFF", "ON")]
  for name in names:
    if not (folder / name).is_file():
      raise ValueError("basis file missing: " + name)
    digest = CM._sha256(folder / name)
    if name in CM.BASIS_SHA256 and digest != CM.BASIS_SHA256[name]:
      raise ValueError("released basis file differs: " + name)
    sources[CM.BASIS_DIR + name] = digest
  before = sorted(os.listdir(folder))
  basis = Basis(list(CM.METABOLITES), "fid-a-2d", "siemens", 123.23, lw, "megapress", 2000, 4096).setup(
    Cfg.val["path_basis"], []
  )
  if sorted(os.listdir(folder)) != before:
    raise RuntimeError("basis folder changed while loading")
  return basis


def _digest_lines(digests):
  """SHA-256 of the sorted lines "<name> <sha256>" of a set of file digests."""
  return hashlib.sha256("".join(f"{n} {digests[n]}\n" for n in sorted(digests)).encode("ascii")).hexdigest()


def _clean(bases, lws, ids, truth):
  """Noiseless composition-matched mixtures, one basis linewidth per spectrum."""
  from mrsnet.dataset import Dataset

  clean = Dataset("Phantom composition-matched simulations")
  clean.metabolites, clean.pulse_sequence = list(CM.METABOLITES), "megapress"
  for sid, concentrations, lw in zip(ids, truth, lws, strict=True):
    spectra, conc = bases[lw].combine(concentrations.tolist(), sid)
    clean.spectra.append(spectra)
    clean.concentrations.append(conc)
  return clean


def _control_noise(clean, seed, sigma_max=NOISE_SIGMA_MAX):
  """Noise exactly as in the control (sigma_max 0.03): sigma drawn U[0, sigma_max] per mixture."""
  ds = copy.deepcopy(clean)
  np.random.seed(seed)
  ds.add_noise(1.0, "adc_normal", 0.0, sigma_max, 0)
  return ds


def _fixed_noise(clean, sigmas, seed, correct_b0=True):
  """Dataset.add_noise with a given sigma per mixture instead of the uniform draw.

  The uniform draws of Dataset.add_noise (application, mean, sigma) are consumed
  in the same order, so the Gaussian draws equal those of the control for the
  same seed; only their scale differs. As in Dataset.add_noise, the difference
  spectrum is recombined from the noisy ON and OFF spectra and the mixture is
  B0-corrected again (Spectrum.correct_b0_multi). correct_b0=False leaves out
  that correction; it is used only to show that the mirror check can fail.
  """
  from mrsnet.spectrum import Spectrum

  ds = copy.deepcopy(clean)
  if ds.noise_added:
    raise RuntimeError("Noise added twice to dataset")
  num = len(ds.spectra)
  sigmas = np.asarray(sigmas, float)
  if sigmas.shape != (num,) or not np.all(np.isfinite(sigmas)) or np.any(sigmas < 0):
    raise ValueError("one finite non-negative sigma per mixture required")
  np.random.seed(seed)
  np.random.uniform(0.0, 1.0, num)
  mu = np.random.uniform(0.0, 0.0, num)
  np.random.uniform(0.0, NOISE_SIGMA_MAX, num)
  for idx in range(num):
    spectra = ds.spectra[idx]
    for a in spectra:
      if a != "difference":
        spectra[a].add_noise_adc_normal(mu=mu[idx], sigma=sigmas[idx])
    if "difference" in spectra:
      if "edit_off" not in spectra or "edit_on" not in spectra:
        raise RuntimeError("Difference spectrum without edit_off or edit_on")
      spectra["difference"] = Spectrum.comb(
        1.0,
        spectra["edit_on"],
        -1.0,
        spectra["edit_off"],
        spectra["edit_on"].id + "_+_" + spectra["edit_off"].id,
        "difference",
      )
    if correct_b0:
      Spectrum.correct_b0_multi(spectra)
  ds.noise_added = True
  return ds


def _control_sigmas(seed, num, sigma_max=NOISE_SIGMA_MAX):
  """Return the sigma values Dataset.add_noise(1.0, "adc_normal", 0.0, sigma_max, 0) draws for a seed."""
  state = np.random.RandomState(seed)
  state.uniform(0.0, 1.0, num)
  state.uniform(0.0, 0.0, num)
  return state.uniform(0.0, sigma_max, num)


def _mirror_check(clean, linewidths):
  """Compare the fixed-sigma path with Dataset.add_noise byte for byte on the exported inputs.

  For each basis, noise seed and MIRROR_SIGMA_MAX, the sigmas Dataset.add_noise
  draws are passed to _fixed_noise and the exports must be identical. The number
  of mixtures that differ without the B0 re-correction is recorded as evidence
  that the comparison is sensitive.
  """
  record = []
  for key, lws in linewidths.items():
    for sigma_max in MIRROR_SIGMA_MAX:
      drawn_max, above, unequal = 0.0, 0, 0
      for seed in CM.NOISE_SEEDS:
        drawn = _control_sigmas(seed, len(clean[key].spectra), sigma_max)
        library = _export(_control_noise(clean[key], seed, sigma_max), *NOISE_EXPORT)
        mirror = _export(_fixed_noise(clean[key], drawn, seed), *NOISE_EXPORT)
        if library.tobytes() != mirror.tobytes():
          raise RuntimeError(f"fixed-sigma noise does not mirror Dataset.add_noise ({key}, {sigma_max}, {seed})")
        without = _export(_fixed_noise(clean[key], drawn, seed, correct_b0=False), *NOISE_EXPORT)
        unequal += int(np.sum(np.any(library.reshape(len(drawn), -1) != without.reshape(len(drawn), -1), axis=1)))
        drawn_max = max(drawn_max, float(drawn.max()))
        above += int(np.sum(drawn > NOISE_SIGMA_MAX))
      record.append(
        {
          "basis": key,
          "basis_linewidth_hz_range": [min(lws), max(lws)],
          "sigma_draw": [0.0, sigma_max],
          "seeds": list(CM.NOISE_SEEDS),
          "realisations": len(CM.NOISE_SEEDS) * len(clean[key].spectra),
          "drawn_sigma_max": drawn_max,
          "drawn_sigma_above_0.03": above,
          "exports_identical": True,
          "realisations_differing_without_b0_recorrection": unequal,
        }
      )
  return record


def _export(ds, acquisitions, datatype, truth=None):
  """Export as in the control; optionally check the exported composition."""
  values, actual = ds.export(
    metabolites=list(CM.METABOLITES),
    acquisitions=list(acquisitions),
    datatype=list(datatype),
    norm="max",
    high_ppm=-4.5,
    low_ppm=-1,
    n_fft_pts=2048,
    verbose=0,
  )
  if truth is not None and not np.allclose(actual, truth, atol=TOLERANCE, rtol=0):
    raise ValueError("exported composition changed")
  return values


def _complex(values):
  """OFF and ON complex spectra from a YAE-layout export (acquisition, [imaginary, real])."""
  return values[:, 1, 1] + 1j * values[:, 1, 0], values[:, 2, 1] + 1j * values[:, 2, 0]


def _noise_level(spectra, mask):
  """Detrended RMS of real and imaginary parts in the noise window, per spectrum."""
  z = np.atleast_2d(spectra)[:, mask]
  design = np.vander(np.linspace(-1.0, 1.0, z.shape[1]), DETREND_DEGREE + 1)
  dof = z.shape[1] - design.shape[1]
  total = np.zeros(z.shape[0])
  for part in (z.real, z.imag):
    coef = np.linalg.lstsq(design, part.T, rcond=None)[0]
    total += np.sum((part - (design @ coef).T) ** 2, axis=1)
  return np.sqrt(total / (2 * dof))


def _lag1(spectra, mask):
  """Magnitude of the complex lag-1 autocorrelation of the detrended window (whiteness diagnostic)."""
  z = np.atleast_2d(spectra)[:, mask]
  design = np.vander(np.linspace(-1.0, 1.0, z.shape[1]), DETREND_DEGREE + 1)
  residual = np.empty_like(z)
  for i, row in enumerate(z):
    re = row.real - design @ np.linalg.lstsq(design, row.real, rcond=None)[0]
    im = row.imag - design @ np.linalg.lstsq(design, row.imag, rcond=None)[0]
    residual[i] = re + 1j * im
  num = np.abs(np.sum(np.conj(residual[:, :-1]) * residual[:, 1:], axis=1))
  return num / np.sum(np.abs(residual) ** 2, axis=1)


def _simulated_levels(clean, sigmas, mask):
  """RMS over the calibration seeds of the OFF and ON noise levels at given sigmas."""
  off2, on2, lag = 0.0, 0.0, 0.0
  for seed in CALIBRATION_SEEDS:
    off, on = _complex(_export(_fixed_noise(clean, sigmas, seed), *NOISE_EXPORT))
    off2 = off2 + _noise_level(off, mask) ** 2
    on2 = on2 + _noise_level(on, mask) ** 2
    lag = lag + _lag1(off, mask)
  n = len(CALIBRATION_SEEDS)
  return np.sqrt(off2 / n), np.sqrt(on2 / n), lag / n


def _calibrate(clean, mask, target):
  """Per-spectrum sigma whose simulated OFF noise level matches the target level."""
  num = len(target)
  q0 = _noise_level(_complex(_export(clean, *NOISE_EXPORT))[0], mask)
  qref = _simulated_levels(clean, np.full(num, CALIBRATION_REF_SIGMA), mask)[0]
  gain = np.sqrt(qref**2 - q0**2) / CALIBRATION_REF_SIGMA
  if not np.all(gain > 0):
    raise ValueError("noise level does not increase with sigma")
  sigma = np.sqrt(np.maximum(target**2 - q0**2, 0.0)) / gain
  history = []
  for _ in range(CALIBRATION_ITERATIONS):
    q = _simulated_levels(clean, sigma, mask)[0]
    history.append(float(np.max(np.abs(q / target - 1.0))))
    sigma = sigma * target / q
  q, q_on, lag = _simulated_levels(clean, sigma, mask)
  return sigma, {
    "q_noiseless": q0,
    "gain_per_sigma": gain,
    "q_simulated": q,
    "q_on_simulated": q_on,
    "lag1_simulated": lag,
    "max_relative_mismatch_by_iteration": [*history, float(np.max(np.abs(q / target - 1.0)))],
  }


def _validate(clean, mask):
  """Recover known sigmas from single simulated realisations through the same chain."""
  rng = np.random.default_rng(VALIDATION_SEED)
  truth = np.exp(rng.uniform(*np.log(VALIDATION_SIGMA_RANGE), len(clean.spectra)))
  off, _ = _complex(_export(_fixed_noise(clean, truth, VALIDATION_SEED), *NOISE_EXPORT))
  estimate, _ = _calibrate(clean, mask, _noise_level(off, mask))
  rel = estimate / truth - 1.0
  return {
    "sigma_range_log_uniform": list(VALIDATION_SIGMA_RANGE),
    "seed": VALIDATION_SEED,
    "n": len(truth),
    "relative_error_mean": float(rel.mean()),
    "relative_error_median": float(np.median(rel)),
    "relative_error_sd": float(rel.std(ddof=1)),
    "relative_error_p05_p95": [float(np.quantile(rel, 0.05)), float(np.quantile(rel, 0.95))],
    "relative_error_max_abs": float(np.max(np.abs(rel))),
    "correlation": float(np.corrcoef(truth, estimate)[0, 1]),
  }


def _summary(values):
  """Compact distribution summary of a per-spectrum quantity."""
  v = np.asarray(values, float)
  return {
    "min": float(v.min()),
    "median": float(np.median(v)),
    "mean": float(v.mean()),
    "max": float(v.max()),
  }


def prepare(code, ids, truth, entries, sources):
  """Build the four variants' datasets and the linewidth and noise records."""
  from mrsnet.cfg import Cfg

  Cfg.val.update(copy.deepcopy(CM.PREPROCESSING))
  Cfg.val.update(path_root=str(code), path_basis=str(code / "data/basis-dist"), search_basis=[])
  mapping, available, listed = _linewidth_mapping(code, ids)
  sources[LINEWIDTH_CSV] = CM._sha256(code / LINEWIDTH_CSV)
  lw_matched = [r["basis_linewidth_hz"] for r in mapping]
  loaded = sorted({BASELINE_LINEWIDTH, *lw_matched})
  if loaded != list(BASIS_LINEWIDTHS_USED):
    raise ValueError("basis linewidths used differ from BASIS_LINEWIDTHS_USED")
  bases = {lw: _basis(code, lw, sources) for lw in loaded}
  basis_files = {k.removeprefix(CM.BASIS_DIR): v for k, v in sources.items() if k.startswith(CM.BASIS_DIR)}
  if len(basis_files) != 10 * len(loaded) or _digest_lines(basis_files) != BASIS_USED_SHA256:
    raise ValueError("basis files used differ from BASIS_USED_SHA256")
  baseline = [BASELINE_LINEWIDTH] * len(ids)
  clean = {"2Hz": _clean(bases, baseline, ids, truth), "matched": _clean(bases, lw_matched, ids, truth)}

  # Fixed-sigma noise must reproduce Dataset.add_noise exactly when given the drawn sigmas.
  mirror = _mirror_check(clean, {"2Hz": baseline, "matched": lw_matched})

  # Phantom noise levels from the released YAE phantom inputs (OFF/ON, real and imaginary).
  phantom = CM._child(code, entries["YAE"]["phantom_result"].removeprefix("code/")) / "inputs.npz"
  sources[str(phantom.relative_to(code))] = CM._sha256(phantom)
  with np.load(phantom, allow_pickle=False) as z:
    pids, pvalues = z["id"].tolist(), z["input"]
  if pvalues.shape != (144, 3, 2, 2048) or set(pids) != set(ids):
    raise ValueError("unexpected phantom input layout")
  order = [pids.index(sid) for sid in ids]
  p_off, p_on = _complex(pvalues[order])
  _, nu = clean["2Hz"].spectra[0]["edit_off"].rescale_fft(high_ppm=-4.5, low_ppm=-1, npts=2048)
  mask = (nu >= NOISE_WINDOW_PPM[0]) & (nu <= NOISE_WINDOW_PPM[1])
  q_phantom = _noise_level(p_off, mask)
  q_on_phantom = _noise_level(p_on, mask)
  # Where the shared max(abs(OFF)) normalisation of the phantom inputs comes from.
  p_abs = np.abs(p_off)
  peak_ppm = nu[np.argmax(p_abs, axis=1)]
  in_range = np.zeros(nu.shape, dtype=bool)
  for low, high in SINGLET_WINDOWS_PPM:
    in_range |= (nu >= low) & (nu <= high)
  metabolite_peak = p_abs[:, in_range].max(axis=1) / p_abs.max(axis=1)
  outside = [i for i in range(len(ids)) if not in_range[np.argmax(p_abs[i])]]
  noise = {
    "estimator": (
      "detrended RMS (degree-2 polynomial removed from real and imaginary parts separately, "
      "residual sum of squares / (2 * (bins - 3))) of the max(abs(OFF))-normalised exported OFF spectrum "
      "in the signed ppm window; identical code for phantom and simulated inputs"
    ),
    "window_signed_ppm": list(NOISE_WINDOW_PPM),
    "window_bins": int(mask.sum()),
    "window_signed_ppm_actual": [float(nu[mask].min()), float(nu[mask].max())],
    "detrend_degree": DETREND_DEGREE,
    "phantom_inputs": str(phantom.relative_to(code)),
    "export": {"acquisitions": NOISE_EXPORT[0], "datatype": NOISE_EXPORT[1], "norm": "shared max(abs(OFF))"},
    "calibration": {
      "seeds": list(CALIBRATION_SEEDS),
      "reference_sigma": CALIBRATION_REF_SIGMA,
      "iterations": CALIBRATION_ITERATIONS,
      "method": (
        "sigma0 = sqrt(max(q_ph^2 - q0^2, 0)) / gain, gain = sqrt(qref^2 - q0^2) / sigma_ref; "
        "then sigma <- sigma * q_ph / q_sim(sigma); q_sim = RMS over calibration seeds"
      ),
    },
    "fixed_noise_mirrors_add_noise": True,
    "fixed_noise_mirror_check": mirror,
  }
  sigmas, per_spectrum = {}, {}
  for key, name in (("2Hz", "C-noise"), ("matched", "D-linewidth-noise")):
    sigma, detail = _calibrate(clean[key], mask, q_phantom)
    sigmas[name] = sigma
    per_spectrum[name] = detail
    noise[name] = {
      "basis": "2 Hz" if key == "2Hz" else "linewidth-matched",
      "sigma": _summary(sigma),
      "sigma_above_training_max_0.03": int(np.sum(sigma > NOISE_SIGMA_MAX)),
      "sigma_zero_noiseless_level_exceeds_phantom": int(np.sum(sigma == 0)),
      "q_noiseless_over_q_phantom": _summary(detail["q_noiseless"] / q_phantom),
      "max_relative_mismatch_by_iteration": detail["max_relative_mismatch_by_iteration"],
      "on_level_phantom_over_simulated": _summary(q_on_phantom / detail["q_on_simulated"]),
      "validation_known_sigma": _validate(clean[key], mask),
    }
  for check in mirror:
    name = "C-noise" if check["basis"] == "2Hz" else "D-linewidth-noise"
    if check["sigma_draw"][1] == MIRROR_SIGMA_MAX[-1] and sigmas[name].max() > check["drawn_sigma_max"]:
      raise RuntimeError("calibrated sigma above the mirror-checked range: " + name)
  lag_phantom = _lag1(p_off, mask)
  noise["whiteness_lag1_abs_autocorrelation"] = {
    "phantom": _summary(lag_phantom),
    "simulated_C": _summary(per_spectrum["C-noise"]["lag1_simulated"]),
    "simulated_D": _summary(per_spectrum["D-linewidth-noise"]["lag1_simulated"]),
  }
  normalisation = {
    "singlet_windows_signed_ppm": [list(w) for w in SINGLET_WINDOWS_PPM],
    "spectra_max_outside_singlet_windows": len(outside),
    "spectrum_ids": [ids[i] for i in outside],
    "max_signed_ppm": _summary(peak_ppm[outside]) if outside else None,
    "metabolite_peak_over_max": _summary(metabolite_peak[outside]) if outside else None,
    "phantom_over_simulated_noise_relative_to_metabolite_peak": _summary(1.0 / metabolite_peak[outside])
    if outside
    else None,
  }
  noise["max_off_normalisation"] = normalisation
  noise["known_limitations"] = [
    (
      f"In {len(outside)} phantom spectra ({', '.join(sorted({ids[i].split('/')[0] for i in outside}))}) "
      "max(abs(OFF)) is residual water (failed water suppression), not the NAA or Cr singlet (largest "
      f"singlet {metabolite_peak[outside].min():.2f}-{metabolite_peak[outside].max():.2f} of it). "
      "q is matched in the max-normalised domain, so for these spectra the simulated noise relative to the "
      f"metabolites is lower than the phantom's by a factor {1 / metabolite_peak[outside].max():.2f}-"
      f"{1 / metabolite_peak[outside].min():.2f}. The estimator is kept as defined."
    )
    if outside
    else "max(abs(OFF)) is the NAA or Cr singlet in every phantom spectrum",
    (
      "C and D match the per-bin noise SD only: the phantom window noise is more correlated than the white "
      f"simulator noise (median lag-1 abs autocorrelation {np.median(lag_phantom):.2f} phantom, "
      f"{np.median(per_spectrum['C-noise']['lag1_simulated']):.2f} simulated C, "
      f"{np.median(per_spectrum['D-linewidth-noise']['lag1_simulated']):.2f} simulated D)."
    ),
    (
      "The sigmas come from np.linalg.lstsq (detrending) and depend on the NumPy LAPACK/BLAS build, its "
      "runtime kernel and thread count (recorded in runtime); no gate covers their last bits."
    ),
  ]
  noise["per_spectrum"] = [
    {
      "spectrum_id": sid,
      "q_phantom_off": float(q_phantom[i]),
      "q_phantom_on": float(q_on_phantom[i]),
      "lag1_phantom_off": float(lag_phantom[i]),
      "phantom_off_max_signed_ppm": float(peak_ppm[i]),
      "phantom_off_metabolite_peak_over_max": float(metabolite_peak[i]),
      **{
        f"{name}_{k}": float(per_spectrum[name][v][i])
        for name in ("C-noise", "D-linewidth-noise")
        for k, v in (
          ("q_noiseless", "q_noiseless"),
          ("q_simulated", "q_simulated"),
          ("q_on_simulated", "q_on_simulated"),
        )
      },
      "C-noise_sigma": float(sigmas["C-noise"][i]),
      "D-linewidth-noise_sigma": float(sigmas["D-linewidth-noise"][i]),
    }
    for i, sid in enumerate(ids)
  ]

  seeds = list(CM.NOISE_SEEDS)
  datasets = {
    "A-baseline": [clean["2Hz"], *[_control_noise(clean["2Hz"], s) for s in seeds]],
    "B-linewidth": [clean["matched"], *[_control_noise(clean["matched"], s) for s in seeds]],
    "C-noise": [None, *[_fixed_noise(clean["2Hz"], sigmas["C-noise"], s) for s in seeds]],
    "D-linewidth-noise": [None, *[_fixed_noise(clean["matched"], sigmas["D-linewidth-noise"], s) for s in seeds]],
  }
  linewidth = {
    "source": LINEWIDTH_CSV + " (column linewidth_hz, keyed by spectrum_id)",
    "fitted_summary_hz": _summary([r["fitted_linewidth_hz"] for r in mapping]),
    "available_complete_basis_linewidths_hz": available,
    "basis_linewidths_in_folder": listed,
    "rule": "nearest complete basis linewidth; ties to the smaller",
    "max_abs_difference_hz": float(max(abs(r["difference_hz"]) for r in mapping)),
    "basis_linewidths_used": sorted(set(lw_matched)),
    "basis_linewidths_loaded": loaded,
    "basis_files_sha256": dict(sorted(basis_files.items())),
    "basis_files_sha256_lines_digest": BASIS_USED_SHA256,
    "pinned": "complete grid = BASIS_LINEWIDTH_GRID, table = LINEWIDTH_CSV_SHA256, "
    "loaded = BASIS_LINEWIDTHS_USED, files = BASIS_USED_SHA256 (checked before use)",
    "mapping": mapping,
  }
  return datasets, sigmas, lw_matched, linewidth, noise


def _runtime(allow_mismatch):
  """Check and record the thread settings and the NumPy BLAS build and runtime.

  Raises unless the thread variables equal THREADS; with allow_mismatch it
  prints a warning instead and the mismatch is recorded.
  """
  actual = {k: os.environ.get(k) for k in THREADS}
  record = {"threads_expected": THREADS, "threads": actual, "threads_match": actual == THREADS}
  if actual != THREADS:
    message = "thread settings " + json.dumps(actual) + " differ from the recorded runs' " + json.dumps(THREADS)
    if not allow_mismatch:
      raise SystemExit(message + "; set them or pass --allow-thread-mismatch")
    print("WARNING: " + message + "; results may differ in the last bits and fail the baseline gate", file=sys.stderr)
  build = np.show_config(mode="dicts")["Build Dependencies"]
  record["numpy_blas"] = {
    k: {v: build[k].get(v) for v in ("name", "version", "openblas configuration")} for k in ("blas", "lapack")
  }
  try:
    from threadpoolctl import threadpool_info

    record["blas_runtime"] = [
      {k: lib.get(k) for k in ("internal_api", "version", "architecture", "threading_layer", "num_threads")}
      for lib in threadpool_info()
    ]
  except ImportError:
    record["blas_runtime"] = "threadpoolctl not installed"
  return record


def _reference(code):
  """Released control metrics and predictions used by the baseline gate."""
  folder = code / REFERENCE_DIR
  with (folder / "metrics.csv").open(newline="") as stream:
    metrics = {(r["model"], r["domain"], r["metabolite"]): float(r["mae"]) for r in csv.DictReader(stream)}
  arrays = {}
  for alias in CM.MODELS:
    with np.load(folder / (alias + ".npz"), allow_pickle=False) as z:
      arrays[alias] = {k: z[k] for k in ("id", "truth", "prediction_raw", "input_sha256")}
  return metrics, arrays


def run(code, out, allow_thread_mismatch=False):
  """Run the four variants with frozen checkpoints and write the results."""
  if out.exists():
    raise ValueError("use a new output directory")
  runtime = _runtime(allow_thread_mismatch)
  start = time.time()
  import tensorflow as tf

  from mrsnet.cfg import Cfg

  tf.keras.utils.set_random_seed(CM.INFERENCE_SEED)
  entries, sources = _registry(code)
  sources["etc/composition_matched_decomposition.py"] = CM._sha256(Path(__file__))
  sources["etc/composition_matched.py"] = CM._sha256(Path(CM.__file__))
  ids, truth, _ = CM._released_labels(code, entries)
  reference, ref_arrays = _reference(code)
  for name in ["metrics.csv"] + [a + ".npz" for a in CM.MODELS]:
    sources[REFERENCE_DIR + "/" + name] = CM._sha256(code / REFERENCE_DIR / name)
  datasets, sigmas, lw_matched, linewidth, noise = prepare(code, ids, truth, entries, sources)
  Cfg.val.update(predict_batch_size=CM.BATCH_SIZE)
  for name in (
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
  ):
    sources["mrsnet/" + name] = CM._sha256(code / "mrsnet" / name)
  seeds = list(CM.NOISE_SEEDS)
  protocol = {
    "status": "incomplete",
    "models": list(CM.MODELS),
    "variants": {
      "A-baseline": "2 Hz basis; sigma U[0, 0.03] per mixture (Dataset.add_noise), as etc/composition_matched.py",
      "B-linewidth": "nearest complete basis linewidth to the fitted effective linewidth; noise as A",
      "C-noise": "2 Hz basis; fixed per-spectrum sigma calibrated to the phantom noise level; A's random stream",
      "D-linewidth-noise": "B's basis; fixed per-spectrum sigma calibrated on B's basis; A's random stream",
    },
    "spectra": len(ids),
    "noise_repeats": len(seeds),
    "noise_seeds": seeds,
    "inference_seed": CM.INFERENCE_SEED,
    "batch_size": CM.BATCH_SIZE,
    "preprocessing": CM.PREPROCESSING,
    "basis": "fid-a-2d Siemens 123.23 MHz, 2000 Hz, 4096 samples",
    "evaluation": "independent maximum normalisation; finite negatives retained; errors averaged over noise and spectra",
    "gap_fraction": "(MAE_variant - MAE_A) / (MAE_phantom_cpu - MAE_A), MAE_phantom_cpu from " + REFERENCE_DIR,
    "reference": REFERENCE_DIR,
    "labels": "phantom preparation concentrations from the released fixed-model phantom results",
    "linewidth": linewidth,
    "noise": noise,
    "source_sha256": sources,
    "runtime": runtime,
    "executed_runner": "etc/composition_matched_decomposition.py",
    "packages": {
      "python": sys.version.split()[0],
      "tensorflow": tf.__version__,
      "numpy": np.__version__,
      **{n: importlib.metadata.version(n) for n in ("keras", "scipy")},
    },
  }
  out.mkdir(parents=True)
  (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
  print("prepared variants in", round(time.time() - start), "s", flush=True)

  predictions = {v: {} for v in VARIANTS}
  hashes = {v: {} for v in VARIANTS}
  gate = {}
  for alias, (module, classname) in CM.MODELS.items():
    run_folder = CM._child(code, entries[alias]["checkpoint_path"]).parent
    model = getattr(importlib.import_module("mrsnet." + module), classname).load(str(run_folder))
    if (
      alias == "QNet-basis" and getattr(model.training_model.get_layer("basis_lls"), "basis_is_real", False) is not True
    ):
      raise ValueError("QNet-basis did not load the real basis")
    if alias == "YAE" and (list(model.acquisitions), list(model.datatype)) != NOISE_EXPORT:
      raise ValueError("YAE export layout differs from the noise estimator's")
    for variant in VARIANTS:
      preds, digests = [], []
      for ds in datasets[variant]:
        if ds is None:
          preds.append(np.full((len(ids), 5), np.nan))
          digests.append("")
          continue
        values = _export(ds, model.acquisitions, model.datatype, truth)
        digests.append(hashlib.sha256(values.tobytes()).hexdigest())
        preds.append(CM._predict(model, values, ids))
      predictions[variant][alias] = np.asarray(preds)
      hashes[variant][alias] = digests
      if variant == "A-baseline":
        ref = ref_arrays[alias]
        p = predictions[variant][alias]
        result = {
          "ids_equal": ref["id"].tolist() == ids,
          "input_hashes_equal": ref["input_sha256"].tolist() == digests,
          "max_abs_prediction_difference": float(np.max(np.abs(p - ref["prediction_raw"]))),
          "max_abs_mae_difference": 0.0,
        }
        for domain, values in (
          ("noiseless", CM._means(p[0], truth)),
          ("matched-noisy", CM._means(p[1:], np.tile(truth, (len(seeds), 1)))),
        ):
          for m, v in zip((*CM.METABOLITES, "Overall"), values, strict=True):
            diff = abs(float(v) - reference[alias, domain, m])
            result["max_abs_mae_difference"] = max(result["max_abs_mae_difference"], diff)
        gate[alias] = result
        print(alias, "baseline gate", json.dumps(result), flush=True)
        if not (result["ids_equal"] and result["input_hashes_equal"] and result["max_abs_mae_difference"] <= TOLERANCE):
          protocol.update(status="baseline-failed", baseline_gate=gate)
          (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
          raise RuntimeError("baseline does not reproduce the released control for " + alias)
    del model
    tf.keras.backend.clear_session()

  rows = []
  for alias in CM.MODELS:
    phantom = [reference[alias, "phantom-cpu", m] for m in (*CM.METABOLITES, "Overall")]
    base = CM._means(predictions["A-baseline"][alias][1:], np.tile(truth, (len(seeds), 1)))
    for variant in VARIANTS:
      p = predictions[variant][alias]
      domains = [("matched-noisy", CM._means(p[1:], np.tile(truth, (len(seeds), 1))))]
      if not np.isnan(p[0]).any():
        domains.insert(0, ("noiseless", CM._means(p[0], truth)))
      for domain, values in domains:
        for m, v, ph, b in zip((*CM.METABOLITES, "Overall"), values, phantom, base, strict=True):
          rows.append(
            {
              "model": alias,
              "variant": variant,
              "domain": domain,
              "metabolite": m,
              "mae": float(v),
              "mae_baseline_matched_noisy": float(b),
              "mae_phantom_cpu": ph,
              "gap_fraction_closed": float((v - b) / (ph - b)) if domain == "matched-noisy" else "",
            }
          )
  with (out / "metrics.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  for variant in VARIANTS:
    arrays = {
      "id": np.asarray(ids),
      "truth": truth,
      "noise_seeds": np.asarray(seeds),
      "basis_linewidth_hz": np.asarray(
        lw_matched if variant in ("B-linewidth", "D-linewidth-noise") else [BASELINE_LINEWIDTH] * len(ids)
      ),
    }
    if variant in sigmas:
      arrays["noise_sigma"] = sigmas[variant]
    for alias in CM.MODELS:
      arrays["prediction_raw_" + alias] = predictions[variant][alias]
      arrays["input_sha256_" + alias] = np.asarray(hashes[variant][alias])
    np.savez_compressed(out / (variant + ".npz"), **arrays)
  for relative, digest in sources.items():
    if relative == "etc/composition_matched_decomposition.py":
      path = Path(__file__)
    elif relative == "etc/composition_matched.py":
      path = Path(CM.__file__)
    else:
      path = CM._child(code, relative)
    if CM._sha256(path) != digest:
      raise ValueError("scientific source changed during the run: " + relative)
  protocol.update(status="complete", baseline_gate=gate, runtime_seconds=round(time.time() - start, 1))
  (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
  _print_table(rows)
  return True


def _print_table(rows):
  """Print GABA and overall matched-noisy MAEs and gap fractions."""
  print(f"{'model':<11} {'variant':<18} {'GABA':>9} {'frac':>7} {'Overall':>9} {'frac':>7}")
  table = {(r["model"], r["variant"], r["metabolite"]): r for r in rows if r["domain"] == "matched-noisy"}
  for alias in CM.MODELS:
    for variant in VARIANTS:
      g, o = table[alias, variant, "GABA"], table[alias, variant, "Overall"]
      print(
        f"{alias:<11} {variant:<18} {g['mae']:9.5f} {g['gap_fraction_closed']:7.4f}"
        f" {o['mae']:9.5f} {o['gap_fraction_closed']:7.4f}"
      )


def calibration_only(code):
  """Map linewidths and calibrate the noise without inference or output files."""
  print(json.dumps(_runtime(True), indent=2))
  entries, sources = _registry(code)
  ids, truth, _ = CM._released_labels(code, entries)
  _, _, _, linewidth, noise = prepare(code, ids, truth, entries, sources)
  print(json.dumps({k: v for k, v in linewidth.items() if k != "mapping"}, indent=2))
  print(json.dumps({k: v for k, v in noise.items() if k != "per_spectrum"}, indent=2))
  return True


def main():
  """Run the decomposition, or only its linewidth mapping and noise calibration."""
  root = Path(__file__).resolve().parents[1]
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--outdir", type=Path, default=root / DEFAULT_OUT, help="new result folder")
  parser.add_argument("--code-root", type=Path, default=root, help="MRSNet checkout with its data submodules")
  parser.add_argument(
    "--calibration-only", action="store_true", help="print linewidth mapping and noise calibration; write nothing"
  )
  parser.add_argument(
    "--allow-thread-mismatch",
    action="store_true",
    help="run although the thread variables differ from THREADS (warns; recorded in protocol.json)",
  )
  args = parser.parse_args()
  if args.calibration_only:
    ok = calibration_only(args.code_root.resolve())
  else:
    ok = run(args.code_root.resolve(), args.outdir.resolve(), args.allow_thread_mismatch)
  raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
  main()
