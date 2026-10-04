# mrsnet/normalisation.py - MRSNet - canonical normalisation operations
#
# SPDX-FileCopyrightText: Copyright (C) 2026 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Canonical spectral, target, and evaluation normalisation operations."""

import numpy as np


def _as_rows(values, label):
  data = np.array(values, dtype=np.float64, copy=True)
  if data.ndim == 1:
    data = data.reshape(1, -1)
    squeeze = True
  elif data.ndim == 2:
    squeeze = False
  else:
    raise ValueError(f"{label} must be a one- or two-dimensional array")
  if data.shape[1] == 0:
    raise ValueError(f"{label} must contain at least one value per sample")
  if not np.all(np.isfinite(data)):
    raise ValueError(f"{label} contains non-finite values")
  return data, squeeze


def normalise_concentrations(values, norm, *, role):
  """Return a normalised copy and one scale factor per sample.

  Targets must be non-negative. Predictions may be negative and are not clipped.
  Any requested denominator must be finite and strictly positive.
  """
  if norm not in {"none", "sum", "max"}:
    raise ValueError(f"Unknown concentration normalisation {norm}")
  if role not in {"target", "prediction"}:
    raise ValueError(f"Unknown concentration role {role}")
  data, squeeze = _as_rows(values, role)
  if role == "target" and np.any(data < 0):
    raise ValueError("target concentrations must be non-negative")
  if norm == "none":
    factors = np.ones(data.shape[0], dtype=np.float64)
  elif norm == "sum":
    factors = np.sum(data, axis=1)
  else:
    factors = np.max(data, axis=1)
  if not np.all(np.isfinite(factors)) or np.any(factors <= 0):
    raise ValueError(f"{role} {norm} normalisation requires a finite positive denominator")
  result = data / factors[:, np.newaxis]
  if squeeze:
    return result[0], factors
  return result, factors


def normalise_spectra(values, acquisitions):
  """Normalise all acquisitions by one shared amplitude factor.

  The factor is max(abs(OFF)) when edit_off is present. Otherwise it is the
  maximum absolute amplitude over every requested acquisition.
  """
  spectra = np.array(values, copy=True)
  if spectra.ndim != 2 or spectra.shape[0] != len(acquisitions):
    raise ValueError("spectra must have one row per acquisition")
  if spectra.shape[1] == 0:
    raise ValueError("spectra must contain at least one frequency bin")
  if not np.all(np.isfinite(spectra)):
    raise ValueError("spectra contain non-finite values")
  if "edit_off" in acquisitions:
    reference = "edit_off"
    factor = np.max(np.abs(spectra[acquisitions.index(reference), :]))
  else:
    reference = "all_acquisitions"
    factor = np.max(np.abs(spectra))
  if not np.isfinite(factor) or factor <= 0:
    raise ValueError("spectral normalisation requires a finite positive denominator")
  return spectra / factor, float(factor), reference
