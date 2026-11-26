#!/usr/bin/env python3
#
# metabolite_gain_fit.py - MRSNet - per-metabolite signal gains of the GABA phantom spectra
#
# SPDX-FileCopyrightText: Copyright (C) 2026 Frank C Langbein
# SPDX-License-Identifier: AGPL-3.0-or-later

r"""Fit one signal gain per metabolite, relative to NAA, to the 144 GABA phantom spectra.

Descriptive data for the GABA phantom paper (Z Ma, SM Shermer, O Karakus,
FC Langbein, MELBA submission, 2026). The released effective-linewidth fit
(data/sim2real/linewidth/basis-mixture-off) matches each phantom OFF magnitude
spectrum with its known-composition FID-A mixture and one common amplitude.
Here each metabolite gets its own amplitude (gain) relative to NAA, shared by
all spectra of one acquisition series, to describe whether a metabolite's
measured signal is smaller or larger than the simulated one relative to NAA.

Spectra: exactly the 144 spectra of benchmark_manifest.json, loaded from
data/benchmark with Dataset.load_dicoms from the folder named in each spectrum
ID (checked against data/benchmark/benchmark_sequences.json), with the
preprocessing of the retained QNet phantom evaluation, whose exported real
OFF/ON inputs they must reproduce byte for byte. Series: the 14
acquisition_series_id values of the manifest (E4a-d and E9a/b are separate
acquisitions and fitted separately).

Model (per spectrum i, fit window signed ppm WINDOW_PPM of the 2048-point export
grid -4.5..-1 ppm, magnitude, exactly the grid, bins, broadening and shift of
mrsnet.robust_linewidth)::

  OFF:  y_i = a_i * | sum_m g_m c_im F_m(lw_i, shift_i) | / B_i + b_i
  DIFF: d_i = a_i * | sum_m h_m c_im D_m(lw_i, shift_i) | / B_i + bd_i

F_m is the FFT of the 1.0 Hz FID-A OFF basis FID of metabolite m, broadened by
exp(-pi (lw - 1) t) and shifted by exp(2 pi i shift t); D_m the same for the ON
minus OFF basis FID. c_im are the preparation concentrations, B_i the maximum
of the nominal (g = 1, 1 Hz) OFF mixture magnitude in the window, y_i and d_i
the observed OFF and DIFF magnitudes divided by max(y_i) (the DIFF thus on the
OFF scale; Dataset.export normalises OFF, ON and DIFF by one shared factor).

* OFF: g_NAA = 1; g_m >= 0 free for the metabolites present in the series (non-zero
  concentration in some spectrum; others NaN); per spectrum a_i >= 0,
  0 <= b_i <= cap_i, lw_i in LINEWIDTH_BOUNDS_HZ and shift_i in SHIFT_BOUNDS_HZ
  free. Starts: the released linewidth fit for the per-spectrum parameters,
  every free gain at each of GAIN_STARTS; the lowest cost is kept.
* OFF-fixed-lw: as OFF with lw_i and shift_i fixed at the released values.
* DIFF: a_i, lw_i, shift_i fixed from the OFF fit; h_m >= 0 free for the present
  metabolites whose DIFF basis has at least DIFF_SIGNAL_MIN of its OFF basis
  peak magnitude (all but Cr, see protocol.json), including NAA; Cr enters with
  h_Cr fixed to its OFF gain; own baseline 0 <= bd_i <= capd_i.

Loss: soft_l1 on residuals divided by each spectrum's noise proxy (f_scale 1;
for one spectrum this is the robust_linewidth objective up to a constant
factor). Noise proxy and baseline cap per spectrum follow robust_linewidth:
adjacent-difference MAD and 95th percentile of the observed magnitude in the
window bins where the nominal mixture magnitude is at most NOISE_BASIS_FRACTION
of its maximum (for DIFF the nominal DIFF mixture). Analytic Jacobian;
identifiability is the numerical rank of the Jacobian at the solution.
Uncertainty is descriptive only: the leave-one-spectrum-out (jackknife) range
of each gain, each refit started from the full-series solution.

Gates, in order:

1. Inputs: fit_robust_basis_linewidth (single amplitude) is rerun on every
   spectrum and must reproduce linewidth_hz and shift_hz of the released table
   to GATE1_TOLERANCE_HZ; this script's model with all gains 1 at the released
   parameters must reproduce its robust objective (relative GATE1_OBJECTIVE_RTOL).
   The run stops otherwise.
2. Recovery: spectra simulated from the same basis at the compositions of
   RECOVERY_SERIES with known gains (RECOVERY_OFF_GAINS, RECOVERY_DIFF_GAINS),
   the released per-spectrum linewidth, shift and amplitude, and complex white
   time-domain noise (first RECOVERY_ACQUIRED_POINTS samples, as acquired;
   independent for ON and OFF; DIFF = ON - OFF) are fitted by the identical
   procedure (single-amplitude start, OFF, OFF-fixed-lw, DIFF), RECOVERY_REPEATS
   realisations per noise case (RECOVERY_NOISE): sigma per spectrum such that the
   simulated OFF noise proxy equals the phantom's (the specified level), the same
   for the DIFF proxy, and one percent of the OFF proxy (procedure without
   noise). A gain is recovered if the mean over the realisations is within
   RECOVERY_TOLERANCE of the truth. The phantom gains are written whatever the
   outcome; recovery.csv and the README say which pass.

Water suppression failed in ten spectra (nine E6 spectra with GABA > 0 and E11
GEL02; composition-decomposition protocol.json noise.max_off_normalisation).
They are included; E6 and E11 are also fitted without them (cohort
excluding-failed-water-suppression).

Inputs (below --code-root): the manifest, data/benchmark, the 1.0 Hz FID-A basis
in data/basis-dist (BASIS_SHA256), the released linewidth table
(LINEWIDTH_CSV_SHA256), the retained QNet phantom inputs and report
(PHANTOM_SHA256) and the decomposition protocol (water-suppression list). No
simulation of a basis, no TensorFlow.

Outputs (--outdir, default data/sim2real/paper-gaba-phantom/metabolite-gains):
series_gains.csv, per_spectrum.csv, recovery.csv, gate1.json, protocol.json and
README.md. All but protocol.json (runtime) are byte-identical between runs with
the same packages, BLAS and THREADS.

Run from the repository root (about 10 min on a 64-core CPU)::

  OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  etc/metabolite_gain_fit.py

with an interpreter that has the mrsnet dependencies (NumPy, SciPy, pydicom,
sobol_seq). The run refuses other thread settings unless --allow-thread-mismatch
is given. Fits run in --jobs forked worker processes; each fit is computed in one
process, so the results do not depend on the number of workers.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

# Run as a script from etc/: make the mrsnet library in the repository root importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

METABOLITES = ("Cr", "GABA", "Gln", "Glu", "NAA")
REFERENCE = "NAA"
NAMES = {"creatine": "Cr", "gaba": "GABA", "glutamine": "Gln", "glutamate": "Glu", "n-acetylaspartate": "NAA"}
SIZES = {
  "E1": 13,
  "E2": 16,
  "E3": 15,
  "E4a": 8,
  "E4b": 8,
  "E4c": 8,
  "E4d": 8,
  "E6": 10,
  "E7": 16,
  "E8": 8,
  "E9a": 8,
  "E9b": 7,
  "E11": 8,
  "E14": 11,
}
DEFAULT_OUT = "data/sim2real/paper-gaba-phantom/metabolite-gains"
MANIFEST = "data/sim2real/paper-gaba-phantom/benchmark_manifest.json"
SEQUENCES = "data/benchmark/benchmark_sequences.json"
LINEWIDTH_CSV = "data/sim2real/linewidth/basis-mixture-off/linewidth_per_spectrum.csv"
LINEWIDTH_CSV_SHA256 = "6fce6e458a3af7922edf69bf64c091b08abd722a556df14e31d6a120e16ac364"
DECOMPOSITION_PROTOCOL = "data/sim2real/paper-gaba-phantom/composition-decomposition/protocol.json"
PHANTOM = (
  "data/model-dist/qnet_default/Cr-GABA-Gln-Glu-NAA/megapress/edit_off-edit_on/real/sum/16/2000/"
  "fid-a-2d_2000_4096_siemens_123.23_2.0_Cr-GABA-Gln-Glu-NAA_megapress_sobol_1.0-adc_normal-0.0-0.03_100000-1/"
  "NoValidation-1/evaluation/phantom/"
)
PHANTOM_SHA256 = {
  "report.json": "a4a6444bc2400b960b1e42aa42d134b6a65f47629da058d45b8de209af4f2288",
  "inputs.npz": "17ebb9710d847d6560ad4b2c23b031ae28e842f6846a9f9dd34b70895651d912",
}
BASIS_DIR = "data/basis-dist/fid-a-2d/basis_files/"
# The 1.0 Hz basis files of the released linewidth fit (its provenance.json).
BASIS_SHA256 = {
  "FIDA2D_Cr_MEGAPRESS_EDITOFF_1.00_2000_4096_123.23.mat": "1512129066bf36e584f13ac2313d8820f670fe0d5ba15bfa0c72f68b7ab3cf50",
  "FIDA2D_Cr_MEGAPRESS_EDITON_1.00_2000_4096_123.23.mat": "9d5014d105354eedea464817d3aa33ac6d1f490e79dac9dbad47bed8d4c7b466",
  "FIDA2D_GABA_MEGAPRESS_EDITOFF_1.00_2000_4096_123.23.mat": "8ace790a5a7b351cedf945f4cf6aa5b24d38a916c5c8f9f76e1bc11be7f61412",
  "FIDA2D_GABA_MEGAPRESS_EDITON_1.00_2000_4096_123.23.mat": "bd6fe92278a448a48b24215506d27a8e68683ca9471234c3da66e18c9d0151ae",
  "FIDA2D_Gln_MEGAPRESS_EDITOFF_1.00_2000_4096_123.23.mat": "e68786b5b0fa421f0d454695a8181c2d3ef1747bc3f0466535a5cb64fa4aeb3d",
  "FIDA2D_Gln_MEGAPRESS_EDITON_1.00_2000_4096_123.23.mat": "7d0f8c0df32d751913a488d2ca6fc12a0061ac2596aac40ca4851ac2648a72c3",
  "FIDA2D_Glu_MEGAPRESS_EDITOFF_1.00_2000_4096_123.23.mat": "1010a03f110c0bb904979c42f5a6f088231a31fc43b98ba266be8c4270c226a5",
  "FIDA2D_Glu_MEGAPRESS_EDITON_1.00_2000_4096_123.23.mat": "00b52fc47e679a9fa0529264d1bd936175222e83a23a04bf2cf60afc88ab538d",
  "FIDA2D_NAA_MEGAPRESS_EDITOFF_1.00_2000_4096_123.23.mat": "b065cfb46231ea10f77ad9094fa3eb27733203356a25fe4c0f1c66d55a8b56b2",
  "FIDA2D_NAA_MEGAPRESS_EDITON_1.00_2000_4096_123.23.mat": "a58922284e2ce4123d9be86ed755734da356fb242a905690968da54dbf30e51b",
}
MRSNET_MODULES = (
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
THREADS = {"OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"}
EXPORT_PPM = (-4.5, -1.0)
EXPORT_POINTS = 2048
WINDOW_PPM = (-4.0, -1.05)
SAMPLE_RATE = 2000.0
BASE_LINEWIDTH = 1.0
LINEWIDTH_BOUNDS_HZ = (1.0, 20.0)
SHIFT_BOUNDS_HZ = (-5.0, 5.0)
GAIN_STARTS = (0.5, 1.0, 1.5)
NOISE_BASIS_FRACTION = 0.02
NOISE_MIN_BINS = 20
DIFF_SIGNAL_MIN = 0.05
OPTIMISER = {"ftol": 1e-10, "xtol": 1e-10, "gtol": 1e-10, "max_nfev": 2000}
GATE1_TOLERANCE_HZ = 1e-6
GATE1_OBJECTIVE_RTOL = 1e-9
LINEARITY_RTOL = 1e-12
RECOVERY_SERIES = ("E3", "E4a", "E9a")
RECOVERY_OFF_GAINS = {"Cr": 1.15, "GABA": 0.7, "Gln": 1.0, "Glu": 0.9, "NAA": 1.0}
# DIFF truth: Cr is not fitted in DIFF (negligible DIFF basis) and enters with its OFF gain.
RECOVERY_DIFF_GAINS = {"Cr": 1.15, "GABA": 0.6, "Gln": 1.0, "Glu": 0.9, "NAA": 0.85}
RECOVERY_SEED = 20261127
RECOVERY_REPEATS = 5
# Noise cases: (proxy matched, factor). The OFF proxy is the specified level; in most series it
# is several times the DIFF proxy (see README), so the DIFF proxy is a second case; one percent of
# the OFF proxy checks that the procedure is exact when noise is negligible.
RECOVERY_NOISE = {"off-proxy": ("off", 1.0), "diff-proxy": ("diff", 1.0), "off-proxy-x0.01": ("off", 0.01)}
RECOVERY_ACQUIRED_POINTS = 2048
RECOVERY_CALIBRATION_ITERATIONS = 3
RECOVERY_TOLERANCE = 0.05
MODELS = ("OFF", "OFF-fixed-lw", "DIFF")
LOCAL = ("lw", "shift", "a", "b")


def _sha256(path):
  """Hash a file in blocks."""
  with path.open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def _child(root, relative):
  """Resolve a repository-relative path, rejecting absolute or escaping paths."""
  path = Path(relative)
  if path.is_absolute() or ".." in path.parts:
    raise ValueError("scientific source must use a repository-relative path")
  return root / path


def _float(value):
  """Plain float for CSV/JSON output (NaN kept)."""
  return float(value)


# ------------------------------------------------------------------ inputs
def _records(code):
  """Manifest records in manifest order, with concentrations in metabolite order."""
  manifest = json.loads((code / MANIFEST).read_text())
  records = manifest["records"]
  ids = [r["spectrum_id"] for r in records]
  if len(records) != 144 or len(set(ids)) != 144:
    raise ValueError("manifest does not hold 144 unique spectra")
  counts = {}
  for r in records:
    counts[r["acquisition_series_id"]] = counts.get(r["acquisition_series_id"], 0) + 1
    if set(r["concentrations_mM"]) - set(NAMES):
      raise ValueError("unknown metabolite in manifest")
  if counts != SIZES:
    raise ValueError("manifest series sizes differ from SIZES")
  return records


def _released_fits(code, ids):
  """Read the released single-amplitude linewidth fits, keyed by spectrum ID."""
  path = code / LINEWIDTH_CSV
  if _sha256(path) != LINEWIDTH_CSV_SHA256:
    raise ValueError("released linewidth table differs from LINEWIDTH_CSV_SHA256")
  with path.open(newline="") as stream:
    rows = {r["spectrum_id"]: r for r in csv.DictReader(stream)}
  if set(rows) != set(ids):
    raise ValueError("released linewidth fits do not cover the manifest")
  return rows


def _water_failures(code):
  """Spectra whose max(abs(OFF)) is residual water (decomposition protocol)."""
  record = json.loads((code / DECOMPOSITION_PROTOCOL).read_text())["noise"]["max_off_normalisation"]
  ids = list(record["spectrum_ids"])
  if len(ids) != 10 or sum(i.startswith("E6/") for i in ids) != 9 or "E11/MEGA_Combi_WS_ON/GEL02/" not in ids:
    raise ValueError("unexpected water-suppression failure list")
  return ids


def _basis(code, sources):
  """Load the released 1.0 Hz FID-A basis without allowing the library to simulate."""
  from mrsnet.basis import Basis
  from mrsnet.cfg import Cfg

  folder = code / BASIS_DIR
  for name, digest in BASIS_SHA256.items():
    if _sha256(folder / name) != digest:
      raise ValueError("released basis file differs: " + name)
    sources[BASIS_DIR + name] = digest
  before = sorted(os.listdir(folder))
  basis = Basis(list(METABOLITES), "fid-a-2d", "siemens", 123.23, BASE_LINEWIDTH, "megapress", 2000, 4096).setup(
    Cfg.val["path_basis"], []
  )
  if sorted(os.listdir(folder)) != before:
    raise RuntimeError("basis folder changed while loading")
  return basis


def _grid(reference):
  """FFT length, export grid and its indices exactly as robust-linewidth driver built them."""
  from mrsnet.cfg import Cfg

  bandwidth = np.asarray(reference.sample_rate / 2.0 / reference.omega).item()
  length = int(2 * bandwidth / ((EXPORT_PPM[1] - EXPORT_PPM[0]) / EXPORT_POINTS))
  for _ in range(Cfg.val["spectrum_rescale_fft_max_repeats"] + 1):
    grid = np.linspace(-1, 1, length) * bandwidth + reference.center_ppm + reference.b0_shift_ppm
    indices = np.flatnonzero((grid >= EXPORT_PPM[0]) & (grid <= EXPORT_PPM[1]))
    if len(indices) == EXPORT_POINTS:
      return length, grid[indices], indices
    length += 1
  raise ValueError("cannot construct the basis export grid")


def _pad(fid, length):
  """Zero-pad or truncate a FID to the FFT length."""
  return np.pad(fid, (0, max(0, length - len(fid))))[:length]


def _noise_controls(observed, nominal, mask):
  """Noise proxy and baseline cap of mrsnet.robust_linewidth (observed units)."""
  reference = float(observed[mask].max())
  quiet = mask & (nominal <= NOISE_BASIS_FRACTION * float(nominal[mask].max()))
  pairs = quiet[1:] & quiet[:-1]
  if np.count_nonzero(quiet) < NOISE_MIN_BINS or pairs.sum() < NOISE_MIN_BINS:
    raise ValueError("too few low-signal bins")
  differences = np.diff(observed)[pairs]
  noise = np.median(np.abs(differences - np.median(differences))) / (0.67448975 * np.sqrt(2))
  noise = max(float(noise), reference * 1e-6)
  cap = max(float(np.quantile(observed[quiet], 0.95)), reference * 1e-6)
  return noise, cap, int(quiet.sum())


def load(code, sources):
  """Load and check the 144 spectra and build their fit inputs (manifest order)."""
  from mrsnet.benchmark_data import concentrations_for_series
  from mrsnet.cfg import Cfg
  from mrsnet.dataset import Dataset

  report_path = code / PHANTOM / "report.json"
  input_path = code / PHANTOM / "inputs.npz"
  for name, digest in PHANTOM_SHA256.items():
    if _sha256(code / PHANTOM / name) != digest:
      raise ValueError("retained phantom source changed: " + name)
    sources[PHANTOM + name] = digest
  Cfg.val.update(json.loads(report_path.read_text())["preprocessing"])
  Cfg.val.update(path_root=str(code), path_basis=str(code / "data/basis-dist"), search_basis=[])
  basis = _basis(code, sources)
  records = _records(code)
  released = _released_fits(code, [r["spectrum_id"] for r in records])
  sequences = json.loads((code / SEQUENCES).read_text())
  with np.load(input_path, allow_pickle=False) as saved:
    saved_ids, saved_inputs = saved["id"].tolist(), saved["input"]
  if sorted(saved_ids) != sorted(r["spectrum_id"] for r in records):
    raise ValueError("retained phantom inputs do not cover the manifest")
  by_id = {sid: i for i, sid in enumerate(saved_ids)}
  water = set(_water_failures(code))
  spectra = []
  for series in SIZES:
    wanted = [r for r in records if r["acquisition_series_id"] == series]
    folders = {r["spectrum_id"].split("/")[1] for r in wanted}
    if len(folders) != 1 or sequences[series] != list(folders):
      raise ValueError("spectrum folder differs from benchmark_sequences.json: " + series)
    benchmark = code / "data/benchmark"
    dataset = Dataset(series).load_dicoms(
      str(benchmark / series / folders.pop()),
      concentrations=concentrations_for_series(str(benchmark), series),
      metabolites=list(METABOLITES),
      verbose=0,
    )
    loaded = {s["edit_off"].id: s for s in dataset.spectra}
    if set(loaded) != {r["spectrum_id"] for r in wanted} or len(dataset.spectra) != len(wanted):
      raise ValueError("loaded spectra differ from the manifest: " + series)
    order = [s["edit_off"].id for s in dataset.spectra]
    exported, _ = dataset.export(
      metabolites=list(METABOLITES),
      acquisitions=["edit_off", "edit_on"],
      datatype=["real"],
      norm="sum",
      high_ppm=EXPORT_PPM[0],
      low_ppm=EXPORT_PPM[1],
      n_fft_pts=EXPORT_POINTS,
      verbose=0,
    )
    if not np.array_equal(exported, saved_inputs[[by_id[sid] for sid in order]]):
      raise ValueError("exported phantom inputs differ from the retained QNet phantom inputs: " + series)
    for record in wanted:
      sid = record["spectrum_id"]
      row = loaded[sid]
      conc = np.array([float(record["concentrations_mM"].get(k, 0)) for k in NAMES])
      if [NAMES[k] for k in NAMES] != list(METABOLITES):
        raise RuntimeError("metabolite order")
      mixture, _ = basis.combine(dict(zip(METABOLITES, conc.tolist(), strict=True)), "ref_" + sid)
      reference = mixture["edit_off"]
      _, reference_grid = reference.rescale_fft(EXPORT_PPM[0], EXPORT_PPM[1], EXPORT_POINTS)
      length, grid, indices = _grid(reference)
      np.testing.assert_array_equal(grid, reference_grid)
      mixture_off = _pad(reference.get_t()[0], length)
      mixture_on = _pad(mixture["edit_on"].get_t()[0], length)
      off = np.array([_pad(basis.spectra[m]["edit_off"].get_t()[0], length) for m in METABOLITES])
      on = np.array([_pad(basis.spectra[m]["edit_on"].get_t()[0], length) for m in METABOLITES])
      for single, total in ((off, mixture_off), (on, mixture_on)):
        if np.max(np.abs(conc @ single - total)) > LINEARITY_RTOL * np.max(np.abs(total)):
          raise ValueError("mixture FID is not the concentration-weighted basis sum: " + sid)
      mask = (reference_grid >= WINDOW_PPM[0]) & (reference_grid <= WINDOW_PPM[1])
      observed = {}
      for acquisition in ("edit_off", "edit_on", "difference"):
        raw_fft, raw_grid = row[acquisition].rescale_fft(EXPORT_PPM[0], EXPORT_PPM[1], EXPORT_POINTS)
        if raw_grid[0] > reference_grid[mask].min() or raw_grid[-1] < reference_grid[mask].max():
          raise ValueError("observed FFT does not cover all fit bins")
        observed[acquisition] = (raw_fft, raw_grid)
      if not all(np.array_equal(observed[a][1], observed["edit_off"][1]) for a in observed):
        raise ValueError("OFF, ON and DIFF grids differ: " + sid)
      y_off = np.interp(reference_grid, observed["edit_off"][1], np.abs(observed["edit_off"][0]))
      y_diff = np.interp(reference_grid, observed["difference"][1], np.abs(observed["difference"][0]))
      raw = observed["edit_off"][0]
      scanner_diff = observed["difference"][0]
      window = (observed["edit_off"][1] >= WINDOW_PPM[0]) & (observed["edit_off"][1] <= WINDOW_PPM[1])
      on_minus_off = observed["edit_on"][0] - raw
      nominal_off = np.abs(np.fft.fftshift(np.fft.fft(mixture_off)))[indices]
      nominal_diff = np.abs(np.fft.fftshift(np.fft.fft(mixture_on - mixture_off)))[indices]
      noise_off, cap_off, quiet_off = _noise_controls(y_off, nominal_off, mask)
      noise_diff, cap_diff, quiet_diff = _noise_controls(y_diff, nominal_diff, mask)
      spectra.append(
        {
          "spectrum_id": sid,
          "series": series,
          "water_failed": sid in water,
          "conc": conc,
          "length": length,
          "sel": indices[mask],
          "mask": mask,
          "grid": reference_grid,
          "indices": indices,
          "fid_mixture": mixture_off,
          "nominal_off": nominal_off,
          "nominal_diff": nominal_diff,
          "fid_off": off,
          "fid_diff": on - off,
          "y_off_full": y_off,
          "y_off": y_off[mask],
          "y_diff": y_diff[mask],
          "reference": float(y_off[mask].max()),
          "B": float(nominal_off[mask].max()),
          "noise_off": noise_off,
          "cap_off": cap_off,
          "quiet_off": quiet_off,
          "noise_diff": noise_diff,
          "cap_diff": cap_diff,
          "quiet_diff": quiet_diff,
          "scanner_diff_vs_on_minus_off": float(
            np.linalg.norm((scanner_diff - on_minus_off)[window]) / np.linalg.norm(scanner_diff[window])
          ),
          "released": released[sid],
        }
      )
  for relative in (MANIFEST, SEQUENCES, LINEWIDTH_CSV, DECOMPOSITION_PROTOCOL):
    sources[relative] = _sha256(code / relative)
  return spectra, basis


def _diff_signal(spectra):
  """Peak DIFF over peak OFF basis magnitude (1 Hz, fit window) per metabolite."""
  s = spectra[0]
  off = np.abs(np.fft.fftshift(np.fft.fft(s["fid_off"], axis=1), axes=1)[:, s["sel"]]).max(axis=1)
  diff = np.abs(np.fft.fftshift(np.fft.fft(s["fid_diff"], axis=1), axes=1)[:, s["sel"]]).max(axis=1)
  return dict(zip(METABOLITES, (diff / off).tolist(), strict=True))


# ------------------------------------------------------------------ model
class SeriesModel:
  """Joint magnitude model of one series' spectra with shared metabolite gains.

  Parameter vector: the free gains (in METABOLITES order), then per spectrum the
  free local parameters among (lw, shift, a, b) in that order. Residuals are
  (a |S| / B + b - y) / noise per fit bin, in the normalisation y = observed /
  max(observed OFF).
  """

  def __init__(self, spectra, acquisition, free_gains, fixed_gains, free_local, local):
    """Set up the model; local is an (n, 4) array of lw, shift, a, b (fixed values and starts)."""
    self.spectra = spectra
    self.acquisition = acquisition
    self.free_gains = list(free_gains)
    self.fixed_gains = np.asarray(fixed_gains, float)
    self.free_local = [LOCAL.index(k) for k in free_local]
    self.local = np.asarray(local, float).copy()
    self.gain_index = [METABOLITES.index(m) for m in self.free_gains]
    self.fids = [s["fid_off"] if acquisition == "off" else s["fid_diff"] for s in spectra]
    self.time = [np.arange(s["length"]) / SAMPLE_RATE for s in spectra]
    ref = np.array([s["reference"] for s in spectra])
    self.target = [(s["y_off"] if acquisition == "off" else s["y_diff"]) / r for s, r in zip(spectra, ref, strict=True)]
    self.noise = np.array([s["noise_" + acquisition] for s in spectra]) / ref
    self.cap = np.array([s["cap_" + acquisition] for s in spectra]) / ref
    self.bins = [len(t) for t in self.target]
    self._cache = None

  def bounds(self):
    """Lower and upper bounds of the parameter vector."""
    lower = [0.0] * len(self.free_gains)
    upper = [np.inf] * len(self.free_gains)
    for i in range(len(self.spectra)):
      lo = (LINEWIDTH_BOUNDS_HZ[0], SHIFT_BOUNDS_HZ[0], 0.0, 0.0)
      hi = (LINEWIDTH_BOUNDS_HZ[1], SHIFT_BOUNDS_HZ[1], np.inf, self.cap[i])
      lower += [lo[k] for k in self.free_local]
      upper += [hi[k] for k in self.free_local]
    return np.array(lower), np.array(upper)

  def pack(self, gains, local):
    """Parameter vector from full gains and an (n, 4) local array."""
    return np.concatenate([np.asarray(gains, float)[self.gain_index], np.asarray(local)[:, self.free_local].ravel()])

  def unpack(self, x):
    """Full gains (5) and local (n, 4) arrays from a parameter vector."""
    gains = self.fixed_gains.copy()
    k = len(self.free_gains)
    gains[self.gain_index] = x[:k]
    local = self.local.copy()
    if self.free_local:
      local[:, self.free_local] = x[k:].reshape(len(self.spectra), len(self.free_local))
    return gains, local

  def _evaluate(self, x):
    """Residuals and Jacobian at x (cached for the last x)."""
    if self._cache is not None and np.array_equal(self._cache[0], x):
      return self._cache[1], self._cache[2]
    gains, local = self.unpack(x)
    k, nl = len(self.free_gains), len(self.free_local)
    residuals, jac = [], np.zeros((sum(self.bins), len(x)))
    row = 0
    for i, s in enumerate(self.spectra):
      lw, shift, a, b = local[i]
      t = self.time[i]
      envelope = np.exp(-np.pi * (lw - BASE_LINEWIDTH) * t + 2j * np.pi * shift * t)
      weights = gains * s["conc"]
      signal = (weights @ self.fids[i]) * envelope
      spectrum = np.fft.fftshift(np.fft.fft(signal))[s["sel"]]
      magnitude = np.abs(spectrum) / s["B"]
      r = (a * magnitude + b - self.target[i]) / self.noise[i]
      residuals.append(r)
      n = self.bins[i]
      safe = np.where(magnitude > 0, magnitude, 1.0)
      factor = a / (s["B"] ** 2 * self.noise[i] * safe)
      for j, m in enumerate(self.gain_index):
        if s["conc"][m] != 0:
          part = np.fft.fftshift(np.fft.fft(s["conc"][m] * self.fids[i][m] * envelope))[s["sel"]]
          jac[row : row + n, j] = factor * np.real(np.conj(spectrum) * part)
      for j, p in enumerate(self.free_local):
        col = k + i * nl + j
        if p == 0:
          part = np.fft.fftshift(np.fft.fft(signal * (-np.pi * t)))[s["sel"]]
          jac[row : row + n, col] = factor * np.real(np.conj(spectrum) * part)
        elif p == 1:
          part = np.fft.fftshift(np.fft.fft(signal * (2j * np.pi * t)))[s["sel"]]
          jac[row : row + n, col] = factor * np.real(np.conj(spectrum) * part)
        elif p == 2:
          jac[row : row + n, col] = magnitude / self.noise[i]
        else:
          jac[row : row + n, col] = 1.0 / self.noise[i]
      row += n
    residuals = np.concatenate(residuals)
    self._cache = (x.copy(), residuals, jac)
    return residuals, jac

  def residual(self, x):
    """Noise-scaled residual vector."""
    return self._evaluate(x)[0]

  def jacobian(self, x):
    """Analytic Jacobian of the noise-scaled residuals."""
    return self._evaluate(x)[1]

  def fit(self, x0):
    """Bounded soft_l1 least squares from x0."""
    lower, upper = self.bounds()
    return least_squares(
      self.residual,
      np.clip(x0, lower, upper),
      jac=self.jacobian,
      bounds=(lower, upper),
      loss="soft_l1",
      f_scale=1.0,
      x_scale="jac",
      **OPTIMISER,
    )

  def summary(self, result):
    """Gains, local parameters and fit statistics of a least-squares result."""
    gains, local = self.unpack(result.x)
    scaled = result.fun * np.repeat(self.noise, self.bins)
    targets = np.concatenate(self.target)
    per_spectrum = []
    row = 0
    for n, t in zip(self.bins, self.target, strict=True):
      per_spectrum.append(float(np.linalg.norm(scaled[row : row + n]) / np.linalg.norm(t)))
      row += n
    return {
      "gains": gains,
      "local": local,
      "cost": float(result.cost),
      "relative_residual": float(np.linalg.norm(scaled) / np.linalg.norm(targets)),
      "relative_residual_per_spectrum": per_spectrum,
      "rank": int(np.linalg.matrix_rank(result.jac)),
      "parameters": len(result.x),
      "success": bool(result.success),
      "status": int(result.status),
      "nfev": int(result.nfev),
    }


def single_amplitude_objective(s):
  """robust_linewidth objective and relative MSE of the released single-amplitude fit, this module's model."""
  model = SeriesModel([s], "off", [], np.ones(5), [], _released_local([s])[0:1])
  r = model.residual(np.zeros(0)) * model.noise[0]
  noise = model.noise[0]
  rho = 2 * (np.sqrt(1 + (r / noise) ** 2) - 1)
  return float(noise**2 * np.sum(rho) / len(r)), float(np.mean(r**2))


def _released_local(spectra):
  """Released lw, shift and normalised amplitude and baseline per spectrum."""
  local = []
  for s in spectra:
    r = s["released"]
    local.append(
      [
        float(r["linewidth_hz"]),
        float(r["shift_hz"]),
        float(r["scale"]) * s["B"] / s["reference"],
        float(r["baseline"]) / s["reference"],
      ]
    )
  return np.array(local)


def _present(spectra):
  """Metabolites with a non-zero concentration in some spectrum."""
  conc = np.array([s["conc"] for s in spectra])
  return [m for j, m in enumerate(METABOLITES) if np.any(conc[:, j] > 0)]


def fit_off(spectra, start_local, fixed_lw):
  """OFF gain fit of one series from several gain starts; returns the best summary and its model."""
  free = [m for m in _present(spectra) if m != REFERENCE]
  local_free = ("a", "b") if fixed_lw else LOCAL
  best = None
  for start in GAIN_STARTS:
    gains = np.where(np.isin(METABOLITES, free), start, 1.0)
    model = SeriesModel(spectra, "off", free, np.ones(5), local_free, start_local)
    result = model.fit(model.pack(gains, start_local))
    summary = model.summary(result)
    summary["start"] = start
    if best is None or summary["cost"] < best[0]["cost"]:
      best = (summary, model, result)
  best[0]["free"] = free
  best[0]["present"] = _present(spectra)
  return best


def fit_diff(spectra, off_summary, diff_metabolites):
  """DIFF gain fit with a, lw and shift fixed from the OFF fit; Cr-like gains fixed to OFF."""
  present = _present(spectra)
  free = [m for m in present if m in diff_metabolites]
  fixed = np.array(off_summary["gains"], float)
  local = off_summary["local"].copy()
  caps = np.array([s["cap_diff"] / s["reference"] for s in spectra])
  local[:, 3] = 0.5 * caps
  best = None
  for start in GAIN_STARTS:
    gains = fixed.copy()
    gains[[METABOLITES.index(m) for m in free]] = start
    model = SeriesModel(spectra, "diff", free, fixed, ("b",), local)
    result = model.fit(model.pack(gains, local))
    summary = model.summary(result)
    summary["start"] = start
    if best is None or summary["cost"] < best[0]["cost"]:
      best = (summary, model, result)
  best[0]["free"] = free
  best[0]["present"] = present
  return best


def jackknife(model, result):
  """Leave-one-spectrum-out refits started from the full-series solution: gains per refit."""
  n = len(model.spectra)
  if n < 2:
    return [], []
  gains, local = model.unpack(result.x)
  values, success = [], []
  for drop in range(n):
    keep = [i for i in range(n) if i != drop]
    sub = SeriesModel(
      [model.spectra[i] for i in keep],
      model.acquisition,
      model.free_gains,
      model.fixed_gains,
      [LOCAL[k] for k in model.free_local],
      local[keep],
    )
    fitted = sub.fit(sub.pack(gains, local[keep]))
    values.append(sub.unpack(fitted.x)[0])
    success.append(bool(fitted.success))
  return values, success


def fit_series(spectra, diff_metabolites, start_local, with_jackknife=True):
  """OFF, OFF-fixed-lw and DIFF fits (with jackknife) of one set of spectra."""
  out = {}
  off = fit_off(spectra, start_local, fixed_lw=False)
  fixed = fit_off(spectra, start_local, fixed_lw=True)
  diff = fit_diff(spectra, off[0], diff_metabolites)
  for name, (summary, model, result) in (("OFF", off), ("OFF-fixed-lw", fixed), ("DIFF", diff)):
    if with_jackknife:
      summary["jackknife"], summary["jackknife_success"] = jackknife(model, result)
    out[name] = summary
  return out


# ------------------------------------------------------------------ gates
def _single_fit(s):
  """Run the released single-amplitude fit (mrsnet.robust_linewidth, released settings) on one spectrum."""
  from mrsnet.robust_linewidth import fit_robust_basis_linewidth

  fitted = fit_robust_basis_linewidth(
    s["fid_mixture"],
    s["y_off_full"],
    s["indices"],
    SAMPLE_RATE,
    BASE_LINEWIDTH,
    LINEWIDTH_BOUNDS_HZ,
    shift_bounds=SHIFT_BOUNDS_HZ,
    mask=s["mask"],
  )
  return {k: v for k, v in fitted.items() if k != "profile"}


def gate1(spectra, mapper):
  """Rerun the released single-amplitude fit and evaluate this module's model at its parameters."""
  rows, worst = [], {}
  keys = ("linewidth_hz", "shift_hz", "scale", "baseline", "noise_scale", "baseline_limit", "robust_objective")
  for s, fitted in zip(spectra, mapper(_single_fit, spectra), strict=True):
    released = s["released"]
    diff = {k: abs(fitted[k] - float(released[k])) for k in keys}
    objective, mse = single_amplitude_objective(s)
    diff["model_objective_relative"] = abs(objective / float(released["robust_objective"]) - 1)
    diff["model_relative_mse_relative"] = abs(mse / float(released["relative_mse"]) - 1)
    diff["noise_proxy_relative"] = abs(s["noise_off"] / float(released["noise_scale"]) - 1)
    diff["baseline_cap_relative"] = abs(s["cap_off"] / float(released["baseline_limit"]) - 1)
    for k, v in diff.items():
      worst[k] = max(worst.get(k, 0.0), v)
    rows.append({"spectrum_id": s["spectrum_id"], **{k: _float(v) for k, v in diff.items()}})
  passed = (
    worst["linewidth_hz"] <= GATE1_TOLERANCE_HZ
    and worst["shift_hz"] <= GATE1_TOLERANCE_HZ
    and worst["model_objective_relative"] <= GATE1_OBJECTIVE_RTOL
    and worst["noise_proxy_relative"] <= GATE1_OBJECTIVE_RTOL
    and worst["baseline_cap_relative"] <= GATE1_OBJECTIVE_RTOL
  )
  return {
    "passed": bool(passed),
    "n": len(spectra),
    "tolerance_hz": GATE1_TOLERANCE_HZ,
    "objective_rtol": GATE1_OBJECTIVE_RTOL,
    "max_abs_difference": {k: _float(v) for k, v in worst.items()},
    "per_spectrum": rows,
    "definition": (
      "absolute differences of fit_robust_basis_linewidth rerun on this run's inputs from the released "
      "linewidth_per_spectrum.csv (linewidth_hz, shift_hz in Hz; scale, baseline, noise_scale, baseline_limit in "
      "observed units; robust_objective); model_objective_relative and model_relative_mse_relative: this "
      "script's model with all gains 1 at the released parameters against the released robust_objective and "
      "relative_mse; noise_proxy_relative, baseline_cap_relative: this script's noise proxy and cap against the "
      "released noise_scale and baseline_limit"
    ),
  }


def _noisy_magnitudes(clean_off, clean_on, unit_off, unit_on, indices, sigma):
  """OFF and DIFF (= ON - OFF) magnitudes at the export bins of noisy simulated FIDs."""
  off = np.fft.fftshift(np.fft.fft(clean_off + sigma * unit_off))[indices]
  on = np.fft.fftshift(np.fft.fft(clean_on + sigma * unit_on))[indices]
  return np.abs(off), np.abs(on - off)


def _simulate(spectra, rng, calibration):
  """Simulate OFF and DIFF magnitudes with known gains at the phantom parameters (copies of the spectra).

  calibration (proxy, factor): sigma such that the simulated OFF (proxy "off") or
  DIFF ("diff") noise proxy equals factor times the phantom's.
  """
  out = []
  g = np.array([RECOVERY_OFF_GAINS[m] for m in METABOLITES])
  h = np.array([RECOVERY_DIFF_GAINS[m] for m in METABOLITES])
  for s in spectra:
    released = s["released"]
    lw, shift, scale = float(released["linewidth_hz"]), float(released["shift_hz"]), float(released["scale"])
    t = np.arange(s["length"]) / SAMPLE_RATE
    envelope = np.exp(-np.pi * (lw - BASE_LINEWIDTH) * t + 2j * np.pi * shift * t)
    clean_off = scale * ((g * s["conc"]) @ s["fid_off"]) * envelope
    clean_on = clean_off + scale * ((h * s["conc"]) @ s["fid_diff"]) * envelope
    acquired = np.arange(s["length"]) < RECOVERY_ACQUIRED_POINTS
    unit_off = (rng.standard_normal(s["length"]) + 1j * rng.standard_normal(s["length"])) * acquired
    unit_on = (rng.standard_normal(s["length"]) + 1j * rng.standard_normal(s["length"])) * acquired
    signals = (clean_off, clean_on, unit_off, unit_on, s["indices"])

    proxy, factor = calibration
    target = factor * (s["noise_off"] if proxy == "off" else s["noise_diff"])
    sigma = target / np.sqrt(RECOVERY_ACQUIRED_POINTS)
    for _ in range(RECOVERY_CALIBRATION_ITERATIONS):
      y_off, y_diff = _noisy_magnitudes(*signals, sigma)
      if proxy == "off":
        sigma *= target / _noise_controls(y_off, s["nominal_off"], s["mask"])[0]
      else:
        sigma *= target / _noise_controls(y_diff, s["nominal_diff"], s["mask"])[0]
    y_off, y_diff = _noisy_magnitudes(*signals, sigma)
    noise_off, cap_off, quiet_off = _noise_controls(y_off, s["nominal_off"], s["mask"])
    noise_diff, cap_diff, quiet_diff = _noise_controls(y_diff, s["nominal_diff"], s["mask"])
    sim = dict(s)
    sim.update(
      y_off_full=y_off,
      y_off=y_off[s["mask"]],
      y_diff=y_diff[s["mask"]],
      reference=float(y_off[s["mask"]].max()),
      noise_off=noise_off,
      cap_off=cap_off,
      quiet_off=quiet_off,
      noise_diff=noise_diff,
      cap_diff=cap_diff,
      quiet_diff=quiet_diff,
      noise_ratio=float(noise_off / s["noise_off"]),
      noise_diff_ratio=float(noise_diff / s["noise_diff"]),
    )
    out.append(sim)
  return out


def _fit_task(task):
  """Fit one set of spectra (pool worker): OFF, OFF-fixed-lw and DIFF with or without jackknife."""
  spectra, diff_metabolites, with_jackknife = task
  return fit_series(spectra, diff_metabolites, _released_local(spectra), with_jackknife)


def gate2(spectra, diff_metabolites, mapper):
  """Recovery of known gains from simulated spectra of RECOVERY_SERIES by the identical procedure.

  For each noise calibration, series and realisation (RECOVERY_REPEATS, seeded
  by RECOVERY_SEED, calibration, series and realisation index) the spectra are
  simulated and fitted without jackknife; the single-amplitude fit of each
  simulated spectrum takes the place of the released linewidth fit (starts, and
  the fixed lw and shift of OFF-fixed-lw). A gain is recovered if the mean over
  the realisations is within RECOVERY_TOLERANCE of the truth.
  """
  cases = list(RECOVERY_NOISE)
  keys = [(c, k, r) for c in range(len(cases)) for k in range(len(RECOVERY_SERIES)) for r in range(RECOVERY_REPEATS)]
  simulated = [
    _simulate(
      [s for s in spectra if s["series"] == RECOVERY_SERIES[k]],
      np.random.default_rng([RECOVERY_SEED, c, k, r]),
      RECOVERY_NOISE[cases[c]],
    )
    for c, k, r in keys
  ]
  flat = [s for sims in simulated for s in sims]
  for s, single in zip(flat, mapper(_single_fit, flat), strict=True):
    s["released"] = single
  fits = mapper(_fit_task, [(sims, diff_metabolites, False) for sims in simulated])
  rows, details = [], {}
  for c, calibration in enumerate(cases):
    details[calibration] = {}
    for k, series in enumerate(RECOVERY_SERIES):
      chosen = [i for i, key in enumerate(keys) if key[:2] == (c, k)]
      sims = [simulated[i] for i in chosen]
      details[calibration][series] = {
        "n": len(sims[0]),
        "realisations": len(chosen),
        "off_noise_proxy_over_phantom_median": float(np.median([s["noise_ratio"] for x in sims for s in x])),
        "diff_noise_proxy_over_phantom_median": float(np.median([s["noise_diff_ratio"] for x in sims for s in x])),
      }
      for model in MODELS:
        summaries = [fits[i][model] for i in chosen]
        truth = RECOVERY_DIFF_GAINS if model == "DIFF" else RECOVERY_OFF_GAINS
        for j, m in enumerate(METABOLITES):
          if m not in summaries[0]["free"]:
            continue
          values = np.array([x["gains"][j] for x in summaries])
          mean = float(values.mean())
          rows.append(
            {
              "noise_calibration": calibration,
              "series": series,
              "model": model,
              "metabolite": m,
              "true_gain": truth[m],
              "recovered_mean": mean,
              "recovered_min": float(values.min()),
              "recovered_max": float(values.max()),
              "mean_error": mean - truth[m],
              "tolerance": RECOVERY_TOLERANCE,
              "recovered": abs(mean - truth[m]) <= RECOVERY_TOLERANCE,
              "all_realisations_within_tolerance": bool(np.all(np.abs(values - truth[m]) <= RECOVERY_TOLERANCE)),
              "realisations": len(values),
              "n_spectra": len(sims[0]),
              "all_identifiable": all(x["rank"] == x["parameters"] for x in summaries),
              "all_optimiser_success": all(x["success"] for x in summaries),
              "relative_residual_mean": float(np.mean([x["relative_residual"] for x in summaries])),
            }
          )
  return rows, details


# ------------------------------------------------------------------ output
def _gain_rows(series, cohort, fits, n):
  """series_gains.csv rows of one set of fits."""
  rows = []
  for model in MODELS:
    summary = fits[model]
    jack = np.array(summary.get("jackknife") or np.zeros((0, 5)))
    for j, m in enumerate(METABOLITES):
      if m in summary["free"]:
        role = "free"
      elif model != "DIFF" and m == REFERENCE:
        role = "reference"
      elif model == "DIFF" and m in summary["present"]:
        role = "fixed_to_off_gain"
      else:
        role = "absent"
      gain = float(summary["gains"][j]) if role != "absent" else float("nan")
      free = role == "free"
      rows.append(
        {
          "series": series,
          "cohort": cohort,
          "model": model,
          "metabolite": m,
          "role": role,
          "gain": gain,
          "jackknife_min": float(jack[:, j].min()) if free and len(jack) else float("nan"),
          "jackknife_max": float(jack[:, j].max()) if free and len(jack) else float("nan"),
          "jackknife_all_success": bool(all(summary.get("jackknife_success") or [False])) if free else "",
          "gain_at_zero": bool(free and gain <= 1e-8),
          "n_spectra": n,
          "identifiable": summary["rank"] == summary["parameters"],
          "jacobian_rank": summary["rank"],
          "n_parameters": summary["parameters"],
          "optimiser_success": summary["success"],
          "optimiser_status": summary["status"],
          "best_gain_start": summary["start"],
          "relative_residual": summary["relative_residual"],
        }
      )
  return rows


def _spectrum_rows(spectra, cohort, fits):
  """per_spectrum.csv rows of one set of fits."""
  rows = []
  single = _released_local(spectra)
  mse = [single_amplitude_objective(s)[1] for s in spectra]
  for model in MODELS:
    summary = fits[model]
    for i, s in enumerate(spectra):
      lw, shift, a, b = summary["local"][i]
      scale = "diff" if model == "DIFF" else "off"
      rows.append(
        {
          "spectrum_id": s["spectrum_id"],
          "series": s["series"],
          "cohort": cohort,
          "model": model,
          "water_suppression_failed": s["water_failed"],
          "amplitude_scale": float(a * s["reference"] / s["B"]),
          "baseline": float(b * s["reference"]),
          "baseline_limit": s["cap_" + scale],
          "baseline_at_limit": bool(b * s["reference"] >= s["cap_" + scale] * (1 - 1e-5)),
          "linewidth_hz": float(lw),
          "shift_hz": float(shift),
          "noise_proxy": s["noise_" + scale],
          "relative_residual": summary["relative_residual_per_spectrum"][i],
          "released_linewidth_hz": float(single[i][0]),
          "released_shift_hz": float(single[i][1]),
          "released_scale": float(s["released"]["scale"]),
          "single_amplitude_relative_residual": float(
            np.sqrt(mse[i] * len(s["y_off"])) / np.linalg.norm(s["y_off"] / s["reference"])
          ),
          "scanner_diff_vs_on_minus_off": s["scanner_diff_vs_on_minus_off"],
        }
      )
  return rows


def _write_csv(path, rows):
  """Deterministic CSV in the column order of the first row."""
  with path.open("w", newline="", encoding="ascii") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows([{k: _plain(v) for k, v in row.items()} for row in rows])


def _write_json(path, value):
  """Indented ASCII JSON (NaN not allowed)."""
  path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="ascii")


def _plain(value):
  """Python scalar of a NumPy scalar (csv writes NumPy floats with their repr)."""
  if isinstance(value, np.bool_):
    return bool(value)
  if isinstance(value, np.integer):
    return int(value)
  if isinstance(value, np.floating):
    return float(value)
  return value


def _fmt(value):
  """Three-decimal README number; NaN as n/a."""
  return "n/a" if not np.isfinite(value) else f"{value:.3f}"


def _readme(gains, recovery, gate, diff_signal, water, proxy_ratio):
  """README.md of the result folder (generated, deterministic)."""
  table = {(r["series"], r["cohort"], r["model"], r["metabolite"]): r for r in gains}
  lines = [
    "# Per-metabolite signal gains of the GABA phantom spectra",
    "",
    "> SPDX-FileCopyrightText: Copyright (C) 2026 Frank C Langbein",
    "> SPDX-License-Identifier: AGPL-3.0-or-later",
    "",
    "Written by `etc/metabolite_gain_fit.py` in the MRSNet repository (see its docstring",
    "for the exact model, settings and run command; `protocol.json` records them with",
    "the source hashes, package versions and thread settings). Descriptive data: nothing",
    "in the paper's model selection or evaluation used it.",
    "",
    "## What was fitted",
    "",
    "The released effective-linewidth fit (`../../linewidth/basis-mixture-off/`) matches",
    "each phantom OFF magnitude spectrum with its known-composition mixture of the 1.0 Hz",
    "FID-A basis and one common amplitude. Here every metabolite m gets a gain g_m",
    "relative to NAA (g_NAA = 1), shared by all spectra of one acquisition series (the 14",
    "`acquisition_series_id` values of `../benchmark_manifest.json`; E4a-d and E9a/b",
    "separately). Per spectrum the amplitude, a constant baseline (0 to the robust_linewidth",
    "cap), the exponential broadening and the frequency shift are free, as in the",
    "released fit (signed ppm window -4.0 to -1.05, soft_l1 loss scaled by each",
    "spectrum's noise proxy). Models:",
    "",
    "- `OFF`: OFF magnitude; gains of the metabolites present in the series free.",
    "- `OFF-fixed-lw`: as OFF with linewidth and shift fixed at the released values.",
    "- `DIFF`: scanner DIFF magnitude on the OFF scale (one shared export normalisation),",
    "  amplitude, linewidth and shift fixed from the OFF fit of the spectrum, DIFF basis =",
    "  ON minus OFF basis FID; gains free for every metabolite whose DIFF basis peak is",
    f"  at least {DIFF_SIGNAL_MIN} of its OFF basis peak (ratios: "
    + ", ".join(f"{m} {v:.4f}" for m, v in diff_signal.items())
    + "),",
    "  i.e. all but Cr, which enters with its OFF gain. The DIFF GABA gain is relative",
    "  to the OFF NAA scale and includes the editing efficiency.",
    "",
    "A gain g_m below one says that the measured signal of m, relative to NAA, is",
    "smaller than the simulated basis predicts for the preparation concentrations.",
    "It does not separate concentration errors, relaxation (the simulation has no T1",
    "or T2; phantoms differ from tissue and from each other), editing efficiency",
    "(DIFF), coil and lineshape differences, or fit mismatch.",
    "",
    "## Gates",
    "",
    f"1. Inputs (`gate1.json`): the released single-amplitude fit, rerun on these inputs,"
    f" reproduces the released linewidths and shifts with a maximum absolute difference of"
    f" {gate['max_abs_difference']['linewidth_hz']:.3g} Hz and"
    f" {gate['max_abs_difference']['shift_hz']:.3g} Hz; passed: {gate['passed']}.",
    "2. Recovery (`recovery.csv`): spectra simulated from the same basis at the compositions",
    f"   of {', '.join(RECOVERY_SERIES)} with known gains (OFF "
    + ", ".join(f"{m} {v}" for m, v in RECOVERY_OFF_GAINS.items())
    + "; DIFF "
    + ", ".join(f"{m} {v}" for m, v in RECOVERY_DIFF_GAINS.items() if m != "Cr")
    + "),",
    "   the released per-spectrum linewidth, shift and amplitude, and complex white",
    "   time-domain noise, fitted by the identical procedure, "
    f"{RECOVERY_REPEATS} noise realisations per case. Noise cases:",
    "   `off-proxy` (sigma per spectrum such that the simulated OFF noise proxy equals the",
    "   phantom's; the specified level), `diff-proxy` (the same for the DIFF proxy) and",
    "   `off-proxy-x0.01` (one percent of the OFF proxy; checks the procedure without",
    "   noise). A gain counts as recovered if the mean over the realisations is within",
    f"   {RECOVERY_TOLERANCE} of the truth. Mean (min-max) recovered gains:",
    "",
    "| Noise | Series | Model | Metabolite | True | Recovered | Recovered within tolerance |",
    "| :--- | :--- | :--- | :--- | ---: | :--- | :--- |",
  ]
  for r in recovery:
    lines.append(
      f"| {r['noise_calibration']} | {r['series']} | {r['model']} | {r['metabolite']} | {r['true_gain']} | "
      f"{_fmt(r['recovered_mean'])} ({_fmt(r['recovered_min'])}-{_fmt(r['recovered_max'])}) | "
      f"{'yes' if r['recovered'] else 'no'} |"
    )
  lines += ["", "Not recovered (mean outside tolerance) in at least one series:", ""]
  for case in RECOVERY_NOISE:
    failed = sorted(
      {(r["model"], r["metabolite"]) for r in recovery if r["noise_calibration"] == case and not r["recovered"]}
    )
    lines.append(f"- `{case}`: " + (", ".join(f"{m} {met}" for m, met in failed) if failed else "none") + ".")
  lines += [
    "",
    "Phantom gains of model/metabolite pairs that are not recovered at the phantom noise",
    "level should not be read as measurements of the signal ratio.",
    "",
    "Noise proxies: median phantom DIFF proxy over OFF proxy per series (independent ON and OFF",
    "noise alone would give about 1.4): "
    + ", ".join(f"{k} {v:.2f}" for k, v in proxy_ratio.items())
    + ". Where it is well below 1.4, the OFF proxy (the soft_l1 scale and the `off-proxy`",
    "noise case) is dominated by bin-to-bin structure common to OFF and ON rather than by",
    "random noise, and the `off-proxy` case is noisier than those phantoms.",
    "",
    "## GABA gains",
    "",
    "Gain (jackknife min-max, leave one spectrum out, each refit started from the",
    "full-series solution); n/a where the series has no GABA or one spectrum.",
    "",
    "| Series | Cohort | n | OFF | OFF-fixed-lw | DIFF |",
    "| :--- | :--- | ---: | :--- | :--- | :--- |",
  ]
  keys = sorted({(r["series"], r["cohort"]) for r in gains}, key=lambda k: (list(SIZES).index(k[0]), k[1]))
  for series, cohort in keys:
    cells = []
    for model in MODELS:
      r = table[series, cohort, model, "GABA"]
      cells.append(f"{_fmt(r['gain'])} ({_fmt(r['jackknife_min'])}-{_fmt(r['jackknife_max'])})")
    n = table[series, cohort, "OFF", "GABA"]["n_spectra"]
    lines.append(f"| {series} | {cohort} | {n} | " + " | ".join(cells) + " |")
  lines += [
    "",
    "## Files",
    "",
    "- `series_gains.csv`: one row per series, cohort (`all`, or",
    "  `excluding-failed-water-suppression` for E6 and E11), model and metabolite: `role`",
    "  (`free`, `reference` = NAA fixed at 1, `fixed_to_off_gain` = Cr in DIFF, `absent` =",
    "  zero concentration in every spectrum of the series, gain NaN), `gain`, jackknife",
    "  min/max and whether all jackknife refits succeeded, `gain_at_zero` (lower bound",
    "  active), `n_spectra`, `identifiable` (Jacobian rank equals the number of free",
    "  parameters at the solution), rank, parameters, optimiser success and status, the",
    "  gain start (of 0.5, 1, 1.5) that gave the lowest cost, and `relative_residual` =",
    "  ||model - data|| / ||data|| over all fit bins of the series (normalised units).",
    "- `per_spectrum.csv`: per spectrum, cohort and model the fitted amplitude (`amplitude_scale`,",
    "  same units as the released `scale`), baseline and its cap (observed units),",
    "  linewidth and shift (fixed values for OFF-fixed-lw and DIFF), noise proxy, relative",
    "  residual, the released single-amplitude values and relative residual, the",
    "  water-suppression flag, and the relative difference between the scanner DIFF and",
    "  ON minus OFF in the window (diagnostic; the DIFF fit uses the scanner DIFF).",
    "- `recovery.csv`: gate 2, one row per recovery series, model and free metabolite.",
    "- `gate1.json`: gate 1 per spectrum and maxima.",
    "- `protocol.json`: settings, DIFF basis signal ratios, gate summaries, the",
    "  water-suppression list, source SHA-256, packages, threads, BLAS, runtime and the",
    "  SHA-256 of the other files.",
    "",
    "## Limitations",
    "",
    "- Magnitude fit: phase is discarded; overlapping multiplets add in magnitude only",
    "  through the complex sum of the model, and a constant baseline absorbs the noise floor.",
    "- Fixed basis lineshape: one exponential broadening and shift per spectrum for all",
    "  metabolites; no Gaussian component, no per-metabolite shift, no macromolecules,",
    "  no relaxation. Lineshape mismatch can be absorbed by gains, most for broad",
    "  multiplets (GABA, Gln, Glu) that overlap Cr at 3.0 ppm and NAA/Glx near 2.0-2.4 ppm.",
    "- OFF and DIFF gains measure different things: OFF the unedited signal, DIFF the",
    "  edited signal including editing efficiency and subtraction artefacts (for example",
    "  residual Cr at 3.0 ppm, which the near-zero Cr DIFF basis cannot represent).",
    "- Water suppression failed in "
    + f"{len(water)} spectra ({', '.join(water)}); they are included, and E6 and E11 are also",
    "  fitted without them. Without them E6 keeps only its GABA-free spectrum.",
    "- Descriptive only, on the evaluation spectra themselves: the jackknife range is not a",
    "  confidence interval, there are no p-values, and nothing was selected or tuned with it.",
    "",
  ]
  return "\n".join(lines)


def _runtime(allow_mismatch):
  """Check and record the thread settings and the NumPy BLAS build and runtime."""
  actual = {k: os.environ.get(k) for k in THREADS}
  record = {"threads_expected": THREADS, "threads": actual, "threads_match": actual == THREADS}
  if actual != THREADS:
    message = "thread settings " + json.dumps(actual) + " differ from " + json.dumps(THREADS)
    if not allow_mismatch:
      raise SystemExit(message + "; set them or pass --allow-thread-mismatch")
    print("WARNING: " + message + "; results may differ in the last bits", file=sys.stderr)
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


def _packages():
  """Versions of the packages the fit depends on."""
  record = {"python": sys.version.split()[0], "numpy": np.__version__}
  for name in ("scipy", "pydicom"):
    try:
      record[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
      record[name] = None
  return record


def run(code, out, jobs, allow_thread_mismatch=False):
  """Gate 1, gate 2 and the phantom gain fits; write the result folder."""
  if out.exists():
    raise ValueError("use a new output directory")
  runtime = _runtime(allow_thread_mismatch)
  runtime["jobs"] = jobs
  start = time.time()
  sources = {"etc/metabolite_gain_fit.py": _sha256(Path(__file__))}
  for name in MRSNET_MODULES:
    sources["mrsnet/" + name] = _sha256(code / "mrsnet" / name)
  spectra, _ = load(code, sources)
  print(f"loaded {len(spectra)} spectra in {time.time() - start:.0f} s", flush=True)
  diff_signal = _diff_signal(spectra)
  diff_metabolites = [m for m, v in diff_signal.items() if v >= DIFF_SIGNAL_MIN]
  water = [s["spectrum_id"] for s in spectra if s["water_failed"]]
  protocol = {
    "status": "incomplete",
    "spectra": len(spectra),
    "series": dict(SIZES),
    "series_definition": "acquisition_series_id of the manifest",
    "metabolites": list(METABOLITES),
    "reference_metabolite": REFERENCE,
    "basis": "fid-a-2d Siemens 123.23 MHz, 2000 Hz, 4096 samples, 1.0 Hz (BASIS_SHA256)",
    "export_ppm": list(EXPORT_PPM),
    "export_points": EXPORT_POINTS,
    "fit_window_signed_ppm": list(WINDOW_PPM),
    "sample_rate_hz": SAMPLE_RATE,
    "base_linewidth_hz": BASE_LINEWIDTH,
    "linewidth_bounds_hz": list(LINEWIDTH_BOUNDS_HZ),
    "shift_bounds_hz": list(SHIFT_BOUNDS_HZ),
    "gain_bounds": [0.0, None],
    "gain_starts": list(GAIN_STARTS),
    "optimiser": {"method": "trf", "loss": "soft_l1", "f_scale": 1.0, "x_scale": "jac", **OPTIMISER},
    "residual_scaling": "per spectrum: (model - observed) / max(observed OFF) / noise proxy",
    "noise_policy": (
      "robust_linewidth: adjacent-difference MAD (/0.67449/sqrt 2) of the observed magnitude in window bins with "
      f"nominal mixture magnitude <= {NOISE_BASIS_FRACTION} of its maximum; cap = 95th percentile there; "
      "DIFF: the same on the DIFF magnitude with the nominal DIFF mixture"
    ),
    "identifiability": "numpy.linalg.matrix_rank of the (loss-weighted) Jacobian at the solution equals parameters",
    "jackknife": "leave one spectrum out, single refit started from the full-series solution (gains, local values)",
    "diff_signal_ratio": diff_signal,
    "diff_signal_min": DIFF_SIGNAL_MIN,
    "diff_free_metabolites": diff_metabolites,
    "diff_fixed": "metabolites below diff_signal_min enter DIFF with their OFF gain of the same series and cohort",
    "diff_observed": "scanner DIFF (Dataset difference acquisition), rescale_fft magnitude, interpolated as OFF",
    "water_suppression_failed": water,
    "water_suppression_source": DECOMPOSITION_PROTOCOL + " noise.max_off_normalisation.spectrum_ids",
    "recovery": {
      "series": list(RECOVERY_SERIES),
      "off_gains": RECOVERY_OFF_GAINS,
      "diff_gains": RECOVERY_DIFF_GAINS,
      "seed": RECOVERY_SEED,
      "seed_policy": "numpy default_rng([seed, noise case index, series index, realisation index])",
      "repeats": RECOVERY_REPEATS,
      "noise_cases": {k: {"proxy": v[0], "factor": v[1]} for k, v in RECOVERY_NOISE.items()},
      "acquired_points": RECOVERY_ACQUIRED_POINTS,
      "calibration_iterations": RECOVERY_CALIBRATION_ITERATIONS,
      "tolerance_abs": RECOVERY_TOLERANCE,
      "noise": (
        "independent complex white Gaussian noise on the first acquired_points samples of the OFF and ON FIDs "
        "(zero-filled as the basis), sigma scaled per spectrum until the chosen noise proxy equals factor times "
        "the phantom's; DIFF = ON - OFF"
      ),
      "truth": "released linewidth_hz, shift_hz and scale per spectrum; no baseline",
    },
    "packages": _packages(),
    "runtime": runtime,
    "source_sha256": sources,
  }
  out.mkdir(parents=True)
  with _Pool(jobs) as mapper:
    gate = gate1(spectra, mapper)
    _write_json(out / "gate1.json", gate)
    protocol["gate1"] = {k: gate[k] for k in ("passed", "n", "tolerance_hz", "objective_rtol", "max_abs_difference")}
    print("gate 1", json.dumps(protocol["gate1"]), flush=True)
    if not gate["passed"]:
      protocol["status"] = "gate1-failed"
      _write_json(out / "protocol.json", protocol)
      raise RuntimeError("gate 1 failed: inputs do not reproduce the released linewidth fit")
    recovery, details = gate2(spectra, diff_metabolites, mapper)
    protocol["gate2"] = {
      "rows": len(recovery),
      "recovered": sum(r["recovered"] for r in recovery),
      "not_recovered": [
        f"{r['noise_calibration']} {r['series']} {r['model']} {r['metabolite']}" for r in recovery if not r["recovered"]
      ],
      "details": details,
    }
    print("gate 2", json.dumps({k: v for k, v in protocol["gate2"].items() if k != "details"}), flush=True)
    tasks = [(series, "all", [s for s in spectra if s["series"] == series]) for series in SIZES]
    for series in ("E6", "E11"):
      kept = [s for s in spectra if s["series"] == series and not s["water_failed"]]
      tasks.append((series, "excluding-failed-water-suppression", kept))
    results = mapper(_fit_task, [(subset, diff_metabolites, True) for _, _, subset in tasks])
  gains, per_spectrum = [], []
  for (series, cohort, subset), fits in zip(tasks, results, strict=True):
    gains += _gain_rows(series, cohort, fits, len(subset))
    per_spectrum += _spectrum_rows(subset, cohort, fits)
  _write_csv(out / "series_gains.csv", gains)
  _write_csv(out / "per_spectrum.csv", per_spectrum)
  _write_csv(out / "recovery.csv", recovery)
  proxy_ratio = {
    series: float(np.median([s["noise_diff"] / s["noise_off"] for s in spectra if s["series"] == series]))
    for series in SIZES
  }
  (out / "README.md").write_text(_readme(gains, recovery, gate, diff_signal, water, proxy_ratio), encoding="ascii")
  for relative, digest in sources.items():
    path = Path(__file__) if relative == "etc/metabolite_gain_fit.py" else _child(code, relative)
    if _sha256(path) != digest:
      raise ValueError("scientific source changed during the run: " + relative)
  names = ("series_gains.csv", "per_spectrum.csv", "recovery.csv", "gate1.json", "README.md")
  protocol.update(
    status="complete",
    output_sha256={name: _sha256(out / name) for name in names},
    runtime_seconds=round(time.time() - start, 1),
  )
  _write_json(out / "protocol.json", protocol)
  _print_summary(gains, recovery)
  return True


class _Pool:
  """Context manager yielding an order-preserving map over a fork process pool (or the builtin map)."""

  def __init__(self, jobs):
    """Remember the number of worker processes."""
    self.jobs = jobs
    self.executor = None

  def __enter__(self):
    """Start the pool."""
    if self.jobs <= 1:
      return lambda function, items: list(map(function, items))
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    self.executor = ProcessPoolExecutor(self.jobs, mp_context=multiprocessing.get_context("fork"))
    return lambda function, items: list(self.executor.map(function, items))

  def __exit__(self, *_):
    """Stop the pool."""
    if self.executor is not None:
      self.executor.shutdown()


def _print_summary(gains, recovery):
  """Print the GABA, Cr, Gln and Glu gains and the recovery outcome."""
  for r in recovery:
    print(
      f"recovery {r['noise_calibration']:<15} {r['series']:<4} {r['model']:<12} {r['metabolite']:<4} "
      f"true {r['true_gain']:.2f} mean {r['recovered_mean']:.3f} [{r['recovered_min']:.3f}, {r['recovered_max']:.3f}] "
      f"{'ok' if r['recovered'] else 'NOT RECOVERED'}"
    )
  for r in gains:
    if r["role"] == "free" and r["metabolite"] in ("GABA", "Cr", "Gln", "Glu"):
      print(
        f"{r['series']:<4} {r['cohort'][:3]} {r['model']:<12} {r['metabolite']:<4} {r['gain']:.3f} "
        f"[{r['jackknife_min']:.3f}, {r['jackknife_max']:.3f}] id={r['identifiable']} ok={r['optimiser_success']}"
      )


def main():
  """Run the gain fit into a new folder."""
  root = Path(__file__).resolve().parents[1]
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--outdir", type=Path, default=root / DEFAULT_OUT, help="new result folder")
  parser.add_argument("--code-root", type=Path, default=root, help="MRSNet checkout with its data submodules")
  parser.add_argument(
    "--jobs", type=int, default=max(1, min(32, (os.cpu_count() or 2) // 2)), help="worker processes (fork)"
  )
  parser.add_argument(
    "--allow-thread-mismatch", action="store_true", help="run although the thread variables differ from THREADS"
  )
  args = parser.parse_args()
  ok = run(args.code_root.resolve(), args.outdir.resolve(), args.jobs, args.allow_thread_mismatch)
  raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
  main()
