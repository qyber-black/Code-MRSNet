"""Export decoder spectra from trained autoencoder-quantifier models."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from mrsnet.cfg import Cfg


def _path_parts(path: str) -> list[str]:
  drive, rest = os.path.splitdrive(path)
  parts: list[str] = []
  while True:
    rest, folder = os.path.split(rest)
    if folder:
      parts.append(folder)
    if rest == os.path.sep:
      parts.append(rest)
      break
    if not rest:
      break
  parts.reverse()
  if drive:
    parts.insert(0, drive)
  return parts


def _clean_name(value: str) -> str:
  return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def load_caeq_model(model_arg: str, verbose: int = 0):
  """Load a CAEQ/YAE model from an exact path or the configured model roots."""
  from mrsnet.ae_quantifier import AutoencoderQuantifier

  direct = Path(model_arg)
  if direct.is_dir() and (direct / "mrsnet.json").is_file():
    if verbose > 0:
      print(f"# Loading CAEQ model from {direct}")
    return AutoencoderQuantifier.load(str(direct)), str(direct)

  idl = _path_parts(model_arg)
  for k, part in enumerate(idl):
    if part.startswith("caeq"):
      if len(idl) <= k + 9:
        break
      name = os.path.join(*idl[k:k + 6])
      batchsize = idl[k + 6]
      epochs = idl[k + 7]
      train_model = idl[k + 8]
      trainer = idl[k + 9]
      rest = idl[k + 10] if len(idl) > k + 10 else ""
      prefix = os.path.join(*idl[:k]) if k > 0 else ""
      candidates = []
      if prefix:
        candidates.append(os.path.join(prefix, name, batchsize, epochs, train_model, trainer, rest))
      for spath in [Cfg.val["path_model"], *Cfg.val["search_model"]]:
        candidates.append(os.path.join(spath, name, batchsize, epochs, train_model, trainer, rest))
      for folder in candidates:
        try:
          if verbose > 0:
            print(f"# Trying CAEQ model {folder}")
          return AutoencoderQuantifier.load(folder), folder
        except Exception:
          continue
      break

  raise RuntimeError(f"CAEQ/YAE model not found: {model_arg}")


def load_dataset(dataset_arg: str, force_clean: bool = False):
  """Load a dataset from an exact path or the configured simulation roots."""
  from mrsnet.dataset import Dataset

  try:
    return Dataset.load(dataset_arg, force_clean=force_clean), dataset_arg
  except Exception:
    pass

  idl = _path_parts(dataset_arg)
  if len(idl) >= 9:
    name = os.path.join(*idl[-9:-1])
    rest = idl[-1]
    for spath in [Cfg.val["path_simulation"], *Cfg.val["search_simulation"]]:
      folder = os.path.join(spath, name, rest)
      if os.path.isdir(folder):
        return Dataset.load(folder, force_clean=force_clean), folder
  raise RuntimeError(f"Dataset not found: {dataset_arg}")


def _norm_factors(dataset, acquisitions, high_ppm, low_ppm, n_fft_pts, normalise: bool) -> np.ndarray:
  factors = np.ones(len(dataset.spectra), dtype=np.float64)
  if not normalise:
    return factors
  for idx, spectra in enumerate(dataset.spectra):
    fft = []
    for acq in acquisitions:
      values, _ = spectra[acq].rescale_fft(high_ppm=high_ppm, low_ppm=low_ppm, npts=n_fft_pts)
      fft.append(values)
    arr = np.asarray(fft)
    if "edit_off" in acquisitions:
      factors[idx] = float(np.max(np.abs(arr[acquisitions.index("edit_off"), :])))
    else:
      factors[idx] = float(np.max(np.abs(arr)))
  return factors


def export_dataset_decoder(
    quantifier,
    model_path: str,
    dataset,
    dataset_path: str,
    output_dir: str,
    label: str,
    norm: str = "default",
    normalise: bool = True,
    limit: int = 0,
    verbose: int = 0,
) -> tuple[str, str]:
  """Export original and decoded spectra tensors for one dataset."""
  if norm == "default":
    norm = quantifier.norm
  output = Path(output_dir)
  output.mkdir(parents=True, exist_ok=True)

  ids = []
  if dataset.spectra:
    if limit > 0:
      if verbose > 0:
        print(f"# Limiting dataset export to first {min(limit, len(dataset.spectra))} spectra")
      dataset.spectra = dataset.spectra[:limit]
      if dataset.concentrations:
        dataset.concentrations = dataset.concentrations[:limit]
    id_ref = sorted(dataset.spectra[0].keys())[0]
    ids = [spectra[id_ref].id for spectra in dataset.spectra]

  if verbose > 0:
    print(f"# Exporting dataset {label} ({len(dataset.spectra)} spectra)")
  original, concentrations = dataset.export(
      metabolites=quantifier.metabolites,
      n_fft_pts=quantifier.fft_samples,
      high_ppm=quantifier.high_ppm,
      low_ppm=quantifier.low_ppm,
      norm=norm,
      acquisitions=quantifier.acquisitions,
      datatype=quantifier.datatype,
      normalise=normalise,
      export_concentrations=True,
      verbose=verbose,
  )
  norm_factors = _norm_factors(
      dataset,
      quantifier.acquisitions,
      quantifier.high_ppm,
      quantifier.low_ppm,
      quantifier.fft_samples,
      normalise,
  )
  decoded, predicted_concentrations = quantifier.predict(original, verbose=verbose, return_both=True)

  safe_label = _clean_name(label)
  npz_path = output / f"{safe_label}_decoder_export.npz"
  meta_path = output / f"{safe_label}_decoder_export.json"
  np.savez_compressed(
      npz_path,
      original=original,
      decoded=decoded,
      concentrations=concentrations,
      predicted_concentrations=predicted_concentrations,
      spectral_norm_factors=norm_factors,
  )
  metadata = {
      "label": label,
      "model_path": model_path,
      "model": quantifier.model,
      "dataset_path": dataset_path,
      "dataset_name": dataset.name,
      "ids": ids,
      "metabolites": quantifier.metabolites,
      "pulse_sequence": quantifier.pulse_sequence,
      "acquisitions": quantifier.acquisitions,
      "datatype": quantifier.datatype,
      "concentration_norm": norm,
      "spectra_normalised": normalise,
      "high_ppm": quantifier.high_ppm,
      "low_ppm": quantifier.low_ppm,
      "fft_samples": quantifier.fft_samples,
      "arrays": {
          "original": "input spectra tensor [spectrum, acquisition, datatype, frequency]",
          "decoded": "decoder output tensor [spectrum, acquisition, datatype, frequency]",
          "concentrations": "ground-truth concentrations in metabolite order",
          "predicted_concentrations": "quantifier output in metabolite order",
          "spectral_norm_factors": "factor used to normalise spectra before model input",
      },
  }
  meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
  if verbose > 0:
    print(f"# Wrote {npz_path}")
    print(f"# Wrote {meta_path}")
  return str(npz_path), str(meta_path)


def export_decoder(args) -> None:
  """CLI implementation for exporting CAEQ/YAE decoder spectra."""
  import mrsnet.dataset as dataset_mod

  quantifier, model_path = load_caeq_model(args.model, verbose=args.verbose)
  output_dir = args.output or os.path.join(Cfg.val["path_root"], "data", "decoder-spectra")

  if args.dataset:
    ds, ds_path = load_dataset(args.dataset, force_clean=args.force_clean)
    export_dataset_decoder(
        quantifier,
        model_path,
        ds,
        ds_path,
        output_dir,
        args.label or ds.name,
        norm=args.norm,
        normalise=not args.no_normalise,
        limit=args.limit,
        verbose=args.verbose,
    )
    return

  if not args.benchmark:
    raise RuntimeError("Specify --dataset or --benchmark")

  with open(os.path.join(Cfg.val["path_benchmark"], "benchmark_sequences.json")) as f:
    benchmark_seqs = json.load(f)
  from mrsnet.benchmark_data import concentrations_for_series

  all_original = []
  all_decoded = []
  all_concentrations = []
  all_predicted = []
  all_norm_factors = []
  all_ids = []
  all_labels = []

  for b_id in benchmark_seqs:
    for variant in benchmark_seqs[b_id]:
      label = f"{b_id}_{variant}"
      bm = dataset_mod.Dataset(b_id).load_dicoms(
          os.path.join(Cfg.val["path_benchmark"], b_id, variant),
          concentrations=concentrations_for_series(Cfg.val["path_benchmark"], b_id),
          metabolites=quantifier.metabolites,
          verbose=args.verbose,
      )
      npz_path, _ = export_dataset_decoder(
          quantifier,
          model_path,
          bm,
          os.path.join(Cfg.val["path_benchmark"], b_id, variant),
          output_dir,
          label,
          norm=args.norm,
          normalise=not args.no_normalise,
          limit=args.limit,
          verbose=args.verbose,
      )
      arrays = np.load(npz_path)
      all_original.append(arrays["original"])
      all_decoded.append(arrays["decoded"])
      all_concentrations.append(arrays["concentrations"])
      all_predicted.append(arrays["predicted_concentrations"])
      all_norm_factors.append(arrays["spectral_norm_factors"])
      id_ref = sorted(bm.spectra[0].keys())[0]
      ids = [spectra[id_ref].id for spectra in bm.spectra]
      all_ids.extend(ids)
      all_labels.extend([label] * len(ids))

  if all_original:
    output = Path(output_dir)
    safe_label = _clean_name(args.label or "benchmark_all")
    npz_path = output / f"{safe_label}_decoder_export.npz"
    meta_path = output / f"{safe_label}_decoder_export.json"
    np.savez_compressed(
        npz_path,
        original=np.concatenate(all_original, axis=0),
        decoded=np.concatenate(all_decoded, axis=0),
        concentrations=np.concatenate(all_concentrations, axis=0),
        predicted_concentrations=np.concatenate(all_predicted, axis=0),
        spectral_norm_factors=np.concatenate(all_norm_factors, axis=0),
        benchmark_labels=np.asarray(all_labels),
    )
    meta_path.write_text(json.dumps({
        "label": args.label or "benchmark_all",
        "model_path": model_path,
        "model": quantifier.model,
        "dataset_path": Cfg.val["path_benchmark"],
        "dataset_name": "benchmark_all",
        "ids": all_ids,
        "benchmark_labels": all_labels,
        "metabolites": quantifier.metabolites,
        "pulse_sequence": quantifier.pulse_sequence,
        "acquisitions": quantifier.acquisitions,
        "datatype": quantifier.datatype,
        "concentration_norm": quantifier.norm if args.norm == "default" else args.norm,
        "spectra_normalised": not args.no_normalise,
        "high_ppm": quantifier.high_ppm,
        "low_ppm": quantifier.low_ppm,
        "fft_samples": quantifier.fft_samples,
    }, indent=2, sort_keys=True))
    if args.verbose > 0:
      print(f"# Wrote {npz_path}")
      print(f"# Wrote {meta_path}")
