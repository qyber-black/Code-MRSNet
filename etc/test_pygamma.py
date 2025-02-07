#!/usr/bin/env python3
#
# test_pygamma.py - test loading pygamma basis to check
#
# SPDX-FileCopyrightText: Copyright (C) 2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

from mrsnet.spectrum import Spectrum

# We need the mrsnet configuration
import os
from mrsnet.cfg import Cfg
bin_path = os.path.realpath(__file__)
if not os.path.isfile(bin_path):
  raise Exception("Cannot find location of mrsnet.py root folder")
Cfg.init(bin_path)
Cfg.dev_flags.add('spectrum_set_phase_correct') # Show phase correct effect

# Load pygamma spectrum
bs = Spectrum.load_pygamma('data/basis-dist/pygamma', [],
                           'Cr', 'megapress', 123.23, 2.0, 4096, 0.0005)

# Show spectra
import matplotlib.pyplot as plt
for s in bs:
  print(f"Loaded {s.id}")
  fig = s.plot_spectrum()
  plt.show()
