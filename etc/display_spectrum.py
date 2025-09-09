#!/usr/bin/env python3
#
# display_spectrum.py - demo script to display a simulated or dicom spectrum
#
# SPDX-FileCopyrightText: Copyright (C) 2023-2024 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

simulated = False # Switch between loading simulated and dicom spectra

import mrsnet.dataset as dataset
if simulated:
  # Path to a set of simulated spectra
  path='data/sim-spectra-megapress/fid-a-2d_2000_4096/siemens/123.23/2.0/Cr-GABA-Gln-Glu-NAA/megapress/sobol/1.0-adc_normal-0.0-0.03/10000-1/'
  # Load dataset
  ds = dataset.Dataset.load(path)
else:
  # Path to a dicom spectrum (this loads the whole set; can also pick a subfolder for only one)
  path = 'data/benchmark/E16/MEGA_RAW_Combi_WS_ON'
  # Load dataset (we do not know the metabolites, so they need to be specified)
  ds = dataset.Dataset("dicom dataset").load_dicoms(path,metabolites=sorted(['Cr', 'GABA', 'Glu', 'Gln', 'NAA']))

# Total number of spectra in dataset
num_spec = len(ds.spectra)
# Pick one to display
show_spec = num_spec // 2
show_spec = 0

# Spectrum is a dictionary of individual spectra (edit_off, edit_on, difference),
# plot each of them
import matplotlib.pyplot as plt
print(f"Showing spectrum {ds.spectra[show_spec]["edit_on"].id}")
for spec in ds.spectra[show_spec]:
  # Plot spectrum (all datatypes)
  fig = ds.spectra[show_spec][spec].plot_spectrum()
  plt.show()
