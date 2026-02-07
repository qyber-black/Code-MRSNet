#!/usr/bin/env python3
#
# show_benchmark.py - Load and display benchmark spectra
#
# SPDX-FileCopyrightText: Copyright (C) 2026 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later
#
"""Load benchmark spectra and display them.

This script is intended as a small helper utility to quickly inspect the
measured spectra stored as CSV files for experiment E14.

Usage
-----

From the project root:

    ./etc/show_benchmark.py E14 GABA03
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt

import mrsnet.dataset as dataset
from mrsnet.cfg import Cfg


def _benchmark_root() -> str:
    """Return the root folder for benchmark data (Cfg.path_benchmark or fallback)."""
    root = Cfg.val.get("path_benchmark")
    if not root:
        # Fallback to repository-local benchmark folder
        root = os.path.join(Path(__file__).resolve().parents[1], "data", "benchmark")
    return str(root)


def _load_benchmark_series(b_id: str, variant: str, series_name: str, verbose: int):
    """Load a single benchmark series (folder) as a Dataset, similar to `benchmark`.

    This uses the same configuration paths as `mrsnet.benchmark`, but restricts
    loading to a specific series folder such as 'GABA03'.
    """
    seq_path = os.path.join(_benchmark_root(), b_id, variant, series_name)
    conc_path = os.path.join(_benchmark_root(), b_id, "concentrations.json")
    if not os.path.isdir(seq_path):
        raise SystemExit(f"Series folder not found: {seq_path}")
    bm = dataset.Dataset(f"{b_id}_{series_name}").load_dicoms(
        seq_path,
        concentrations=conc_path,
        metabolites=None,
        verbose=verbose,
    )
    if not bm.spectra:
        raise SystemExit(f"No spectra found in series folder: {seq_path}")
    return bm


def _plot_series(series, conc, verbose: int = 0) -> None:
    """Plot all acquisitions for a single series using `Spectrum.plot_spectrum`."""
    for acq in sorted(series.keys()):
        spec = series[acq]
        if verbose > 0:
            print(f"# Plotting {spec.id} / {acq}")
        spec.plot_spectrum(conc, screen_dpi=Cfg.val["screen_dpi"])
        plt.show(block=True)
        plt.close()


def main() -> None:
    """Parse arguments and display selected benchmark spectra."""
    parser = argparse.ArgumentParser(
        description=(
            "Display a specific benchmark spectrum (e.g. E14 / GABA03) "
            "using MRSNet's Dataset and Spectrum utilities."
        )
    )
    parser.add_argument(
        "experiment",
        metavar="B_ID",
        help="Benchmark experiment identifier, e.g. E14.",
    )
    parser.add_argument(
        "series",
        metavar="SERIES",
        help="Series folder name to display, e.g. GABA03.",
    )
    parser.add_argument(
        "--variant",
        default="MEGA_RAW_Combi_WS_ON",
        help="Benchmark variant / subfolder name, e.g. MEGA_RAW_Combi_WS_ON.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity for spectrum loading.",
    )
    args = parser.parse_args()

    bm = _load_benchmark_series(args.experiment, args.variant, args.series, verbose=args.verbose)
    # For a single series folder we expect exactly one (id -> Spectrum) mapping
    series_dict = bm.spectra[0]
    conc = bm.concentrations[0] if bm.concentrations else []

    _plot_series(series_dict, conc, verbose=args.verbose)


if __name__ == "__main__":
    main()
