# mrsnet/benchmark_data.py - MRSNet benchmark metadata helpers
#
# SPDX-FileCopyrightText: Copyright (C) 2026 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Helpers for benchmark metadata and concentration lookup."""

from __future__ import annotations

import json
import os
from pathlib import Path

from mrsnet import molecules


def _normalise_concentrations(concentrations):
  """Return concentration table with MRSNet short metabolite names."""
  res = {}
  for sample_id, conc in concentrations.items():
    res[sample_id] = {}
    for metabolite, value in conc.items():
      res[sample_id][molecules.short_name(metabolite)] = float(value)
  return res


def load_concentrations(source):
  """Load concentration mapping from a legacy file, info.json, series dir, or dict."""
  if source is None:
    return None
  if isinstance(source, dict):
    if "concentrations" in source:
      return _normalise_concentrations(source["concentrations"])
    return _normalise_concentrations(source)

  path = Path(source)
  if path.is_dir():
    info = path / "info.json"
    legacy = path / "concentrations.json"
    if info.is_file():
      path = info
    elif legacy.is_file():
      path = legacy
    else:
      return None
  if not path.is_file():
    return None

  with path.open() as f:
    data = json.load(f)
  if path.name == "info.json":
    data = data.get("concentrations", {})
  return _normalise_concentrations(data)


def concentration_source_for(folder):
  """Find benchmark concentration metadata near a DICOM/CSV folder."""
  path = Path(folder).resolve()
  for candidate in [path, *path.parents]:
    info = candidate / "info.json"
    if info.is_file():
      return str(info)
    legacy = candidate / "concentrations.json"
    if legacy.is_file():
      return str(legacy)
  return None


def load_series_info(benchmark_root, series_id):
  """Load info.json for a benchmark series, returning an empty dict if absent."""
  info = Path(benchmark_root) / series_id / "info.json"
  if not info.is_file():
    return {}
  with info.open() as f:
    return json.load(f)


def selected_dicom_folder(benchmark_root, series_id):
  """Return the preferred DICOM folder from info.json, or a legacy fallback."""
  info = load_series_info(benchmark_root, series_id)
  folder = info.get("dicom_folder")
  if folder:
    return folder
  series = Path(benchmark_root) / series_id
  for fallback in ["MEGA_RAW_Combi_WS_ON", "MEGA_Combi_WS_ON"]:
    if (series / fallback).is_dir():
      return fallback
  raise RuntimeError(f"No benchmark DICOM folder configured for {series_id}")


def concentrations_for_series(benchmark_root, series_id):
  """Return concentration metadata source for a benchmark series."""
  series = os.path.join(benchmark_root, series_id)
  return concentration_source_for(series)
