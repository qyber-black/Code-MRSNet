#!/usr/bin/env python3
#
# export_bases_json.py - export basis spectra in json
#
# SPDX-FileCopyrightText: Copyright (C) 2024 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import numpy as np
from mrsnet.cfg import Cfg
import mrsnet.basis as basis

# We need the mrsnet configuration to find the basis files
# Find mrsnet path via path to script, assuming it is in the etc folder of mrsnet
Cfg.init(os.path.dirname(os.path.realpath(__file__)))

# Metabolites to be used (keep them sorted)
ms=sorted(['Cr', 'GABA', 'Glu', 'Gln', 'NAA'])

# For basis sources: (source,linewidth) list
for b in [('lcmodel',None), ('pygamma',1.0), ('pygamma',2.0), ('pygamma',3.0), ('pygamma',4.0)]:
  # Get a basis set (search path from Cfg info)
  ba = basis.Basis(metabolites=ms, 
                   source=b[0],
                   manufacturer="siemens",
                   omega=123.23,
                   linewidth=b[1],
                   pulse_sequence="megapress",
                   sample_rate=2000,
                   samples=4096).setup(Cfg.val['path_basis'],
                                       search_basis=Cfg.val['search_basis'])
  for m in ms:
    print(f"{b[0]} - {b[1]}: {m}") 
    # Set concentration of metabolite to 1 for basis
    cs = {m: 0. for m in ms}
    cs[m] = 1
    name = "_".join([ba.source, ba.manufacturer, str(ba.omega),
                     str(ba.linewidth), ba.pulse_sequence,
                     str(ba.sample_rate), str(ba.samples), m])
    s, _ = ba.combine(cs,name)
    # Export acquisitions individually
    for ac in s:
      # Note the fft export in the json consists of a list of pairs of (real,imag) to encode complex128
      s[ac].save_json(f"{name}_{ac}.json")
