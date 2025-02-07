#!/usr/bin/env python3
#
# constrcut_spectrum.py - demo script to construct a spectrum for concentrations and a basis
#
# SPDX-FileCopyrightText: Copyright (C) 2023 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np

# Metabolites to be used (keep them sorted)
ms=sorted(['Cr', 'GABA', 'Glu', 'Gln', 'NAA'])
# Concentrations of the metabolites:
cs = {
  'Cr':  8,
  'GABA': 3,
  'Glu': 12,
  'Gln': 3,
  'NAA': 15,
}
# Normalise concentrations (here maximum is one; could also enforce sum of concentrations is 1)
s = np.sum([cs[m] for m in cs])
cs = { m: cs[m] / s for m in cs }
print(f"Requested concentrations: {cs}")

# We need the mrsnet configuration to find the basis files
import os
from mrsnet.cfg import Cfg
Cfg.init(os.path.dirname(os.path.realpath(__file__)))

# Get a basis set (search path from Cfg info)
import mrsnet.basis as basis
basis = basis.Basis(metabolites=ms, 
                    source="fid-a-2d",
                    manufacturer="siemens",
                    omega=123.23,
                    linewidth=2.0,
                    pulse_sequence="megapress",
                    sample_rate=2000,
                    samples=4096).setup(Cfg.val['path_basis'],
                                        search_basis=Cfg.val['search_basis'])

# Create a new  spectrum with the concentrations from the basis
s,c = basis.combine(cs,"test_spectrum_id")
print(f"Created concentrations: {c}")
# Plot all acquisitions and datatypes of the spectrum
import matplotlib.pyplot as plt
for ac in s:
  fig = s[ac].plot_spectrum()
  plt.show()
