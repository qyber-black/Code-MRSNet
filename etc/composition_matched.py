#!/usr/bin/env python3
#
# composition_matched.py - MRSNet - composition-matched simulation control
#
# SPDX-FileCopyrightText: Copyright (C) 2026 Frank C Langbein
# SPDX-License-Identifier: AGPL-3.0-or-later

r"""Rerun or verify the composition-matched simulation control of the GABA phantom paper.

Z Ma, SM Shermer, O Karakus, FC Langbein. A GABA Phantom Benchmark for
Simulation-Trained MRS Quantification at 3 T (MELBA submission, 2026); the
control is its Table "tab:transfer_mae" and the fold ranges in its abstract.

The control simulates the 144 phantom preparation compositions with the
released fixed 2 Hz FID-A basis (Siemens, 123.23 MHz, 2000 Hz, 4096 samples),
adds ten seeded noise realisations per composition (independent ON/OFF complex
time-domain Gaussian noise, sigma drawn from U[0, 0.03] per mixture), and
quantifies these simulations and the stored phantom inputs with the five
released fixed models (YAE, CNN, FCNN, QNet, QNet-basis). Errors use
independent maximum normalisation of truth and prediction, retain finite
negative predictions and all five metabolites, and are averaged over noise
realisations before averaging over phantom spectra.

All inputs come from the distributed data submodules: the basis from
data/basis-dist (checked against BASIS_SHA256), and the checkpoints, phantom
inputs and preparation labels from data/model-dist (checked against the model
registry). The seeds and preprocessing settings of the recorded run are fixed
below; no result data are stored in this repository.

Entry points (``etc/composition_matched.py --help``, from the repository root):

* ``--outdir DIR`` reruns inference on the CPU into a new folder, e.g.::

    MPLCONFIGDIR=/tmp/mrsnet-mpl TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 \
    etc/composition_matched.py --outdir composition-matched-rerun

  It writes the raw predictions, the 90 MAE summaries (metrics.csv), the label
  manifest and a protocol with package versions and source hashes, then
  verifies the folder as below.
* ``--verify DIR`` recomputes every summary in a result folder from its raw
  predictions and compares the matched and phantom GABA and overall MAEs with
  the values printed in the paper (PAPER_TABLE). It needs NumPy only. The exit
  status is non-zero if any value does not round to the printed digits.

The paper values were recorded with TensorFlow 2.21.0, Keras 3.15.1 and NumPy
2.5.1 on CPU; a rerun with Keras 3.15.0 and SciPy 1.18.0 gave the same 90 MAEs
to within 1e-16. The phantom MAEs of this CPU run differ by at most 1.01e-4 from
the released (GPU) phantom predictions used for the paper's main comparisons.
Neither entry point imports the paper repository or loads machine/user
configuration files.
"""

import argparse
import copy
import csv
import hashlib
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

# Run as a script from etc/: make the mrsnet library in the repository root importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrsnet.normalisation import normalise_concentrations

METABOLITES = ("Cr", "GABA", "Gln", "Glu", "NAA")
NAMES = ("creatine", "gaba", "glutamine", "glutamate", "n-acetylaspartate")
MODELS = {
  "YAE": ("ae_quantifier", "AutoencoderQuantifier"),
  "CNN": ("cnn", "CNN"),
  "FCNN": ("fcnn", "FoundationalCNN"),
  "QNet": ("qnet", "QNet"),
  "QNet-basis": ("qnet", "QNetBasis"),
}
# Settings of the recorded run (20261126), fixed here so that a rerun needs no
# bundled protocol file.
NOISE_SEEDS = tuple(range(20261126, 20261136))
INFERENCE_SEED = 20261126
BATCH_SIZE = 32
PREPROCESSING = {
  "b0_correct_ppm_range": 0.25,
  "fft_peak_location_estimator": "jain",
  "filter_dicom": None,
  "npfft_module": ["numpy", "fft"],
  "phase_correct": None,
  "water_peak_ppm_range": 0.75,
  "water_peak_under_max": 10,
}
BASIS_DIR = "data/basis-dist/fid-a-2d/basis_files/"
BASIS_SHA256 = {
  "FIDA2D_Cr_MEGAPRESS_EDITOFF_2.00_2000_4096_123.23.mat": "9bc7bb9f4c76ca1eee972e5f151057d22761ab80705f98927c9524e4d5d82b1d",
  "FIDA2D_Cr_MEGAPRESS_EDITON_2.00_2000_4096_123.23.mat": "556822439f4adc7dea83fb6532b0a21f5c2f7a24e9e7d67f8aa16a792bd69fbc",
  "FIDA2D_GABA_MEGAPRESS_EDITOFF_2.00_2000_4096_123.23.mat": "7a757162e6e96cbe9fa787139c7b5a75ea30135e5fdb48f512ba6822884bac47",
  "FIDA2D_GABA_MEGAPRESS_EDITON_2.00_2000_4096_123.23.mat": "7c42e38a0bd57f5e8f32adfe3917a427c8b30076a480ac491e1ad321d9110399",
  "FIDA2D_Gln_MEGAPRESS_EDITOFF_2.00_2000_4096_123.23.mat": "4e01861aedd4d757f2e185061ed4f4abf735922f6b83996036e480a627790f7a",
  "FIDA2D_Gln_MEGAPRESS_EDITON_2.00_2000_4096_123.23.mat": "a71941b10c89a7f4cb5e161bdb87c509a2481702c9129e5e34e622fbbcabe55c",
  "FIDA2D_Glu_MEGAPRESS_EDITOFF_2.00_2000_4096_123.23.mat": "43461863eae64850d074d1576c77c5840821fd4d96985372330a9b5009b9691d",
  "FIDA2D_Glu_MEGAPRESS_EDITON_2.00_2000_4096_123.23.mat": "cb7170a57af53bd9135630460908b97958d42d1c1700ea400ca41c3a11b1c489",
  "FIDA2D_NAA_MEGAPRESS_EDITOFF_2.00_2000_4096_123.23.mat": "6c0b3c8127291582af6374898a02def62795a04c0f3b0df9c6eee56a5db97ecb",
  "FIDA2D_NAA_MEGAPRESS_EDITON_2.00_2000_4096_123.23.mat": "da8c94afc4c09f98bc503cd62c2d88a0723e6e0da9169788279db6f9e51e274b",
}
# Matched and phantom columns of the paper's composition-control table, as
# printed; a rerun agrees when it rounds to the same digits.
PAPER_TABLE = {
  "YAE": {"GABA": ("0.00879", "0.167"), "Overall": ("0.00790", "0.139")},
  "CNN": {"GABA": ("0.00517", "0.203"), "Overall": ("0.00558", "0.147")},
  "FCNN": {"GABA": ("0.00587", "0.167"), "Overall": ("0.00593", "0.117")},
  "QNet": {"GABA": ("0.00628", "0.186"), "Overall": ("0.00629", "0.129")},
  "QNet-basis": {"GABA": ("0.00637", "0.135"), "Overall": ("0.00620", "0.139")},
}


def _sha256(path):
  """Hash a scientific input without loading it into memory at once."""
  with path.open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def _child(root, relative):
  """Resolve a repository-relative record while allowing data submodule links."""
  path = Path(relative)
  if path.is_absolute() or ".." in path.parts:
    raise ValueError("scientific source must use a repository-relative path")
  return root / path


def _errors(prediction, truth):
  """Retain negative predictions and independently max-normalise each vector."""
  prediction = normalise_concentrations(prediction, "max", role="prediction")[0]
  truth = normalise_concentrations(truth, "max", role="target")[0]
  if prediction.shape != truth.shape:
    raise ValueError("prediction and truth shapes differ")
  return np.abs(prediction - truth)


def _means(prediction, truth):
  """Average absolute errors, rather than averaging predictions before scoring."""
  errors = _errors(np.asarray(prediction).reshape(-1, 5), np.asarray(truth).reshape(-1, 5))
  return np.r_[errors.mean(axis=0), errors.mean()]


def _manifest(reference):
  """Read the bundled preparation labels in their retained spectrum order."""
  records = json.loads((reference / "benchmark_manifest.json").read_text())["records"]
  ids = [r["spectrum_id"] for r in records]
  truth = np.asarray([[r["concentrations_mM"].get(m, 0) for m in NAMES] for r in records], float)
  if len(ids) != 144 or len(set(ids)) != 144:
    raise ValueError("unexpected phantom identities")
  return ids, normalise_concentrations(truth, "max", role="target")[0]


def verify_results(folder):
  """Recompute all retained MAEs and headline ratios from raw predictions.

  Parameters
  ----------
  folder : pathlib.Path
    Result folder containing the protocol, manifest, CSV and NPZ arrays.

  Returns
  -------
  dict
    Number of checked metrics and unrounded GABA/overall fold ranges.
  """
  protocol = json.loads((folder / "protocol.json").read_text())
  if protocol["status"] != "complete" or protocol["models"] != list(MODELS):
    raise ValueError("incomplete matched evaluation")
  ids, truth = _manifest(folder)
  expected = dict(zip(ids, truth, strict=True))
  with (folder / "metrics.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))
  retained = {(r["model"], r["domain"], r["metabolite"]): float(r["mae"]) for r in rows}
  if len(rows) != 90 or len(retained) != 90:
    raise ValueError("missing or duplicate metric rows")
  ratios = {"GABA": [], "Overall": []}
  count = 0
  repeats = protocol["noise_repeats"]
  for alias in MODELS:
    with np.load(folder / (alias + ".npz"), allow_pickle=False) as z:
      matched_ids = z["id"].tolist()
      actual, prediction = z["truth"], z["prediction_raw"]
    if set(matched_ids) != set(ids) or len(matched_ids) != 144 or prediction.shape != (repeats + 1, 144, 5):
      raise ValueError("matched cohort or noise count differs")
    if not np.allclose(actual, [expected[sid] for sid in matched_ids], atol=1e-12, rtol=0):
      raise ValueError("matched concentrations differ from preparation labels")
    means = {
      "noiseless": _means(prediction[0], actual),
      "matched-noisy": _means(prediction[1:], np.tile(actual, (repeats, 1))),
    }
    with np.load(folder / (alias + "-phantom-runtime-control.npz"), allow_pickle=False) as z:
      phantom_ids = z["id"].tolist()
      if len(phantom_ids) != 144 or set(phantom_ids) != set(ids):
        raise ValueError("phantom cohort differs")
      if not np.allclose(
        normalise_concentrations(z["truth"], "max", role="target")[0],
        [expected[sid] for sid in phantom_ids],
        atol=1e-12,
        rtol=0,
      ):
        raise ValueError("phantom concentrations differ from preparation labels")
      means["phantom-cpu"] = _means(z["prediction_raw"], z["truth"])
    for domain, values in means.items():
      for metabolite, value in zip((*METABOLITES, "Overall"), values, strict=True):
        if not np.isclose(value, retained[alias, domain, metabolite], atol=1e-12, rtol=0):
          raise ValueError("retained MAE differs from raw predictions")
        count += 1
    for metabolite, index in (("GABA", 1), ("Overall", 5)):
      ratios[metabolite].append(float(means["phantom-cpu"][index] / means["matched-noisy"][index]))
  return {"checked_metric_rows": count, "spectra": 144, "fold_ranges": {m: [min(v), max(v)] for m, v in ratios.items()}}


def _saved_phantom(folder):
  """Support both released prediction-array layouts without paper helpers."""
  raw = folder / "raw_predictions.npz"
  if raw.exists():
    with np.load(raw, allow_pickle=False) as z:
      ids, prediction = z["id"].tolist(), z["prediction_raw"]
    with np.load(folder / "ground_truth.npz", allow_pickle=False) as z:
      if z["id"].tolist() != ids:
        raise ValueError("saved phantom prediction and truth identities differ")
      truth = z["truth_raw"]
  else:
    with np.load(folder / "predictions.npz", allow_pickle=False) as z:
      ids, prediction = z["id"].tolist(), z["canonical"]
    with np.load(folder / "inputs.npz", allow_pickle=False) as z:
      if z["id"].tolist() != ids:
        raise ValueError("saved phantom prediction and input identities differ")
      truth = z["truth"]
  return ids, prediction, truth


def _predict(model, values, ids):
  """Preserve acquisition-series boundaries from the retained phantom protocol."""
  prediction = np.empty((len(ids), 5))
  for name in dict.fromkeys(sid.split("/")[0] for sid in ids):
    ix = [i for i, sid in enumerate(ids) if sid.split("/")[0] == name]
    batch = np.asarray(model.predict(values[ix]))
    if batch.shape != (len(ix), 5):
      raise ValueError("unexpected prediction shape")
    prediction[ix] = batch
  normalise_concentrations(prediction, "max", role="prediction")
  return prediction


def _released_labels(code, entries):
  """Take phantom identities and preparation labels from the released results.

  Every released fixed model stores the same 144 phantom spectra in the same
  order with their preparation concentrations. The labels are required to agree
  across all five models after maximum normalisation, and the millimolar values
  are taken from a layout that stores them unnormalised.
  """
  ids, truth, millimolar = None, None, None
  for alias in MODELS:
    folder = _child(code, entries[alias]["phantom_result"].removeprefix("code/"))
    found, _, labels = _saved_phantom(folder)
    labels = np.asarray(labels, float)
    if len(found) != 144 or len(set(found)) != 144:
      raise ValueError("unexpected phantom identities")
    normalised = normalise_concentrations(labels, "max", role="target")[0]
    if ids is None:
      ids, truth = found, normalised
    elif found != ids or not np.allclose(normalised, truth, atol=1e-12, rtol=0):
      raise ValueError("released phantom labels differ between models")
    if millimolar is None and labels.max() > 1.0:
      millimolar = labels
  if millimolar is None:
    raise ValueError("no released millimolar phantom labels")
  return ids, truth, millimolar


def compare_with_paper(folder):
  """Compare matched and phantom MAEs of a result folder with the paper table.

  Returns
  -------
  list of dict
    One row per model, outcome and domain, with the paper value as printed,
    the unrounded result and whether the result rounds to the printed digits.
  """
  with (folder / "metrics.csv").open(newline="") as stream:
    mae = {(r["model"], r["domain"], r["metabolite"]): float(r["mae"]) for r in csv.DictReader(stream)}
  rows = []
  for alias, outcomes in PAPER_TABLE.items():
    for outcome, printed in outcomes.items():
      for domain, text in zip(("matched-noisy", "phantom-cpu"), printed, strict=True):
        value = mae[alias, domain, outcome]
        digits = len(text.split(".")[1])
        rows.append(
          {
            "model": alias,
            "outcome": outcome,
            "domain": domain,
            "paper": text,
            "result": value,
            "agrees": abs(value - float(text)) <= 0.5 * 10.0**-digits + 1e-12,
          }
        )
  return rows


def _report(folder):
  """Print the verification summary and the comparison with the paper."""
  print(json.dumps(verify_results(folder), indent=2))
  rows = compare_with_paper(folder)
  for r in rows:
    flag = "ok" if r["agrees"] else "DIFFERS"
    print(f"{r['model']:<11} {r['outcome']:<8} {r['domain']:<14} paper {r['paper']:<8} result {r['result']:.6f} {flag}")
  failed = [r for r in rows if not r["agrees"]]
  print(f"{len(rows) - len(failed)} of {len(rows)} paper values reproduced")
  return not failed


def run(code, out):
  """Rerun the recorded inference control with published basis files and weights.

  Parameters
  ----------
  code : pathlib.Path
    MRSNet repository containing its distributed data submodules.
  out : pathlib.Path
    New destination directory. Existing results are never overwritten.
  """
  if out.exists():
    raise ValueError("use a new output directory")
  import tensorflow as tf

  from mrsnet.basis import Basis
  from mrsnet.cfg import Cfg
  from mrsnet.dataset import Dataset

  for name, digest in BASIS_SHA256.items():
    if _sha256(code / BASIS_DIR / name) != digest:
      raise ValueError("released basis file differs: " + name)
  registry = code / "data/model-dist/paper-model-registry/model_registry.json"
  entries = {
    e["paper_name"]: e
    for e in json.loads(registry.read_text())["instances"]
    if e["regime"] == "fixed" and e["paper_name"] in MODELS
  }
  if set(entries) != set(MODELS):
    raise ValueError("principal fixed checkpoints missing")
  ids, truth, millimolar = _released_labels(code, entries)
  seeds = list(NOISE_SEEDS)
  tf.keras.utils.set_random_seed(INFERENCE_SEED)
  Cfg.val.update(copy.deepcopy(PREPROCESSING))
  Cfg.val.update(
    path_root=str(code),
    path_basis=str(code / "data/basis-dist"),
    search_basis=[],
    predict_batch_size=BATCH_SIZE,
  )
  sources = {
    "etc/composition_matched.py": _sha256(Path(__file__)),
    str(registry.relative_to(code)): _sha256(registry),
  }
  for entry in entries.values():
    for kind in ("checkpoint", "metadata"):
      path = _child(code, entry[kind + "_path"])
      digest = _sha256(path)
      if digest != entry[kind + "_sha256"]:
        raise ValueError("deployed checkpoint or metadata changed")
      sources[str(path.relative_to(code))] = digest
  for name in (
    "basis.py",
    "dataset.py",
    "spectrum.py",
    "normalisation.py",
    "cfg.py",
    "cnn.py",
    "ae_quantifier.py",
    "fcnn.py",
    "qnet.py",
    "train.py",
  ):
    sources["mrsnet/" + name] = _sha256(code / "mrsnet" / name)
  for name in BASIS_SHA256:
    sources[BASIS_DIR + name] = _sha256(code / BASIS_DIR / name)
  protocol = {
    "status": "incomplete",
    "models": list(MODELS),
    "spectra": len(ids),
    "noise_repeats": len(seeds),
    "noise_seeds": seeds,
    "inference_seed": INFERENCE_SEED,
    "batch_size": BATCH_SIZE,
    "preprocessing": PREPROCESSING,
    "basis": "fid-a-2d Siemens 123.23 MHz, 2000 Hz, 4096 samples, 2 Hz",
    "noise": "independent ON/OFF complex time-domain Gaussian; sigma U[0,0.03] per mixture",
    "evaluation": "independent maximum normalisation; finite negatives retained",
    "labels": "phantom preparation concentrations from the released fixed-model phantom results",
    "source_sha256": sources,
    "executed_runner": "etc/composition_matched.py",
    "packages": {
      "tensorflow": tf.__version__,
      "numpy": np.__version__,
      **{n: importlib.metadata.version(n) for n in ("keras", "scipy")},
    },
  }
  out.mkdir(parents=True)
  manifest = {
    "source": protocol["labels"],
    "records": [
      {"spectrum_id": sid, "concentrations_mM": dict(zip(NAMES, map(float, c), strict=True))}
      for sid, c in zip(ids, millimolar, strict=True)
    ],
  }
  (out / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
  basis = Basis(list(METABOLITES), "fid-a-2d", "siemens", 123.23, 2.0, "megapress", 2000, 4096).setup(
    Cfg.val["path_basis"], []
  )
  clean = Dataset("Phantom composition-matched simulations")
  clean.metabolites, clean.pulse_sequence = list(METABOLITES), "megapress"
  for sid, concentrations in zip(ids, truth, strict=True):
    spectra, conc = basis.combine(concentrations.tolist(), sid)
    clean.spectra.append(spectra)
    clean.concentrations.append(conc)
  datasets = [clean]
  for seed in seeds:
    ds = copy.deepcopy(clean)
    np.random.seed(seed)
    ds.add_noise(1.0, "adc_normal", 0.0, 0.03, 0)
    datasets.append(ds)
  rows, changes = [], {}
  cnn_folder = _child(code, entries["CNN"]["phantom_result"].removeprefix("code/"))
  for alias, (module, classname) in MODELS.items():
    entry = entries[alias]
    run_folder = _child(code, entry["checkpoint_path"]).parent
    model = getattr(importlib.import_module("mrsnet." + module), classname).load(str(run_folder))
    if (
      alias == "QNet-basis" and getattr(model.training_model.get_layer("basis_lls"), "basis_is_real", False) is not True
    ):
      raise ValueError("QNet-basis did not load the real basis")
    phantom_folder = _child(code, entry["phantom_result"].removeprefix("code/"))
    inputs = phantom_folder / "inputs.npz"
    if not inputs.exists():
      inputs = cnn_folder / "inputs.npz"
    with np.load(inputs, allow_pickle=False) as z:
      input_ids, values = z["id"].tolist(), z["input"]
    old_ids, archived, old_truth = _saved_phantom(phantom_folder)
    index = {sid: i for i, sid in enumerate(old_ids)}
    expected = dict(zip(ids, truth, strict=True))
    if len(index) != 144 or set(index) != set(ids) or len(input_ids) != 144 or set(input_ids) != set(ids):
      raise ValueError("saved phantom identities differ from manifest")
    if not np.allclose(
      normalise_concentrations(old_truth, "max", role="target")[0],
      [expected[sid] for sid in old_ids],
      atol=1e-12,
      rtol=0,
    ):
      raise ValueError("saved phantom concentrations differ from manifest")
    archived = archived[[index[sid] for sid in input_ids]]
    phantom_truth = np.asarray([expected[sid] for sid in input_ids])
    fresh = _predict(model, values, input_ids)
    delta = np.max(
      np.abs(
        normalise_concentrations(fresh, "max", role="prediction")[0]
        - normalise_concentrations(archived, "max", role="prediction")[0]
      )
    )
    changes[alias] = {
      "max_component_change": float(delta),
      "mae_changes": (_means(fresh, phantom_truth) - _means(archived, phantom_truth)).tolist(),
    }
    np.savez_compressed(
      out / (alias + "-phantom-runtime-control.npz"),
      id=np.asarray(input_ids),
      truth=phantom_truth,
      prediction_raw=fresh,
      archived_prediction_raw=archived,
    )
    predictions, hashes = [], []
    for ds in datasets:
      values, actual = ds.export(
        metabolites=list(METABOLITES),
        acquisitions=model.acquisitions,
        datatype=model.datatype,
        norm="max",
        high_ppm=-4.5,
        low_ppm=-1,
        n_fft_pts=2048,
        verbose=0,
      )
      if not np.allclose(actual, truth, atol=1e-12, rtol=0):
        raise ValueError("exported composition changed")
      hashes.append(hashlib.sha256(values.tobytes()).hexdigest())
      predictions.append(_predict(model, values, ids))
    predictions = np.asarray(predictions)
    np.savez_compressed(
      out / (alias + ".npz"), id=np.asarray(ids), truth=truth, prediction_raw=predictions, input_sha256=hashes
    )
    for domain, values in [
      ("phantom-cpu", _means(fresh, phantom_truth)),
      ("noiseless", _means(predictions[0], truth)),
      ("matched-noisy", _means(predictions[1:], np.tile(truth, (len(seeds), 1)))),
    ]:
      rows += [
        {"model": alias, "domain": domain, "metabolite": m, "mae": float(v)}
        for m, v in zip((*METABOLITES, "Overall"), values, strict=True)
      ]
    print(alias, "matched GABA/overall", rows[-5]["mae"], rows[-1]["mae"], flush=True)
    del model
    tf.keras.backend.clear_session()
  for relative, digest in sources.items():
    path = Path(__file__) if relative == "etc/composition_matched.py" else _child(code, relative)
    if _sha256(path) != digest:
      raise ValueError("scientific source changed during inference")
  with (out / "metrics.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=["model", "domain", "metabolite", "mae"])
    writer.writeheader()
    writer.writerows(rows)
  protocol.update(status="complete", saved_phantom_max_evaluation_difference=changes)
  (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
  return _report(out)


def main():
  """Rerun the CPU inference control, or verify an existing result folder."""
  root = Path(__file__).resolve().parents[1]
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument(
    "--verify", type=Path, help="verify a result folder and compare it with the paper, without inference"
  )
  group.add_argument("--outdir", type=Path, help="rerun inference into a new folder, then verify and compare")
  parser.add_argument("--code-root", type=Path, default=root, help="MRSNet checkout with its data submodules")
  args = parser.parse_args()
  if args.verify is not None:
    ok = _report(args.verify.resolve())
  else:
    ok = run(args.code_root.resolve(), args.outdir.resolve())
  raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
  main()
