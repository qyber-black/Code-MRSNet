#/usr/bin/env python3
#
# simulators/pygamma/pygamma_simulator.py - MRSNet - basis simulation with pygamma
#
# Pulse sequence simulation code has been taken from VeSPA and implemented here with minor modifications.
# https://scion.duhs.duke.edu/vespa/
#
# Modifications by Max Chandler, PhD student at Cardiff University, 2019
#                  Frank C Langbein <frank@langbein.org>, Cardiff University, 2020-2021

from __future__ import division

import argparse
import os
import pdb
from datetime import datetime
import json
import numpy as np
import pygamma as pg

from pygamma_pulse_sequences import fid, press, steam, megapress
import mrsnet.molecules as molecules

def pygamma_spectra_sim(metabolite_name, omega, pulse_sequence, linewidth, cache_dir,
                        npts=4096, adc_dt=4e-4):
  # by having multiple linewidths it allows the use of the same 'mx' object to run the binning over,
  # rather than have to recalcualte the pulse sequence again. it would be more efficient to save the mx table
  # but this functionality is currently (Sept-2018) broken in PyGamma

  id = hash(datetime.now().strftime('%f%S%H%M%a%d%b%Y'))
  mol_spectra = []
  print('Cache miss. Simulating ' + metabolite_name + ' : ' + pulse_sequence)
  infile = os.path.join('mrsnet', 'simulators', 'pygamma', 'metabolite_models',
                        molecules.long_name(metabolite_name).lower() + '.sys')
  spin_system = pg.spin_system()
  spin_system.read(infile)
  spin_system.OmegaAdjust(omega)

  if pulse_sequence.lower() == 'fid':
    mx = [fid(spin_system)]
  elif pulse_sequence.lower() == 'press':
    mx = [press(spin_system)]
  elif pulse_sequence.lower() == 'steam':
    mx = [steam(spin_system)]
  elif pulse_sequence.lower() == 'megapress':
    mx = megapress(spin_system, omega=omega)
  else:
    raise Exception('Unrecognised PyGamma pulse sequence: ' + pulse_sequence)

  for count, acq in enumerate(mx):
    raw = {}
    adc = binning(acq, linewidth=linewidth, dt=adc_dt, npts=npts)
    raw["adc_re"] = adc.real.tolist()
    raw["adc_im"] = adc.imag.tolist()
    raw["count"] = count
    mol_spectra.append(raw)

  # Force gc to try and delete the C++ PyGamma mx object
  #for acq in mx:
  #    acq.disown()
  #    del acq
  #del mx

  # Finally we cache the spectra
  with open(os.path.join(cache_dir, metabolite_name + '.json'), 'w') as save_file:
    json.dump(mol_spectra, save_file)

def binning(mx, linewidth=1, dt=5e-4, npts=2048):
  # add some broadening and decay to the model
  mx.resolution(0.5)              # Combine transitions within 0.5 rad/sec
  mx.broaden(linewidth)
  acq = mx.T(npts, dt)
  ADC = []
  for ii in range(0, acq.pts()):
    ADC.extend([acq.get(ii).real() + (1j * acq.get(ii).imag())])
  return np.array(ADC)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Magnetic Resonance Spectra (MRS) quantification',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('metabolite', type=str, help='Metabolite')
  parser.add_argument('omega', type=float, help='Scanner frequency in Hz')
  parser.add_argument('pulse_sequence', type=str, help='Pulse sequence')
  parser.add_argument('linewidth', type=float, help='Linewidth')
  parser.add_argument('npts', type=int, help='Number of sampling points')
  parser.add_argument('adc_dt', type=float, help='ADC dt')
  parser.add_argument('cache', type=str, help='Cache folder')

  args = parser.parse_args()

  pygamma_spectra_sim(args.metabolite, args.omega, args.pulse_sequence,
                      args.linewidth, args.cache, args.npts, args.adc_dt)
