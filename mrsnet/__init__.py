# mrsnet/__init__.py - MRSNet - init package
#
# SPDX-FileCopyrightText: Copyright (C) 2019 Max Chandler, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2020-2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""MRSNet: Magnetic Resonance Spectra quantification using artificial neural networks.

MRSNet is a comprehensive framework for MRS spectra quantification that provides:
- Spectral basis management and simulation
- Deep learning models (CNN, Autoencoder, Autoencoder-Quantifier, EncDec, FCNN, QMRS, QNet)
- Model training and validation strategies
- Hyperparameter optimization (Grid Search, QMC, GPO, GA)
- Model analysis and comparison utilities
- Sim-to-real analysis and linewidth estimation

This package contains all the core modules for MRS spectra analysis and quantification.
"""

__all__ = ["ae_quantifier", "analyse", "autoencoder", "basis", "basis_lls",
           "cfg", "cnn", "compare", "dataset", "encdec", "fcnn", "getfolder",
           "grid", "molecules", "qmrs", "qnet", "selection", "spectrum", "train"]

version_info = (2,1,0)
__version__ = '.'.join(map(str, version_info))
