# mrsnet/__init__.py - MRSNet - init package
#
# SPDX-FileCopyrightText: Copyright (C) 2019 Max Chandler, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2020-2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""MRSNet: Magnetic Resonance Spectra quantification using artificial neural networks.

MRSNet is a comprehensive framework for MRS spectra quantification that provides:
- Spectral basis management and simulation
- Deep learning models (CNN, Autoencoder, Autoencoder-Quantifier)
- Model training and validation strategies
- Hyperparameter optimization (Grid Search, QMC, GPO, GA)
- Model analysis and comparison utilities

This package contains all the core modules for MRS spectra analysis and quantification.
"""

__all__ = ["ae_quantifier", "analyse", "autoencoder", "basis",
           "cfg", "cnn", "compare", "dataset", "getfolder",
           "grid", "molecules", "selection", "spectrum", "train"]

version_info = (2,0,0)
__version__ = '.'.join(map(str, version_info))
