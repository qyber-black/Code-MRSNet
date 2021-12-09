# simulators/fida/fida_simulator.py - MRSNet - generated simulated FID-A basis spectrum
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2019, Max Chandler, PhD student at Cardiff University
# Copyright (C) 2020-2021, Frank C Langbein <frank@langbein.org>, Cardiff University

import os
import errno
import subprocess

from mrsnet.molecules import GYROMAGNETIC_RATIO

def fida_spectra(metabolite_names, omega, linewidth=1.0, npts=4096, adc_dt=4e-4,
                 save_dir=os.path.join('data', 'basis', 'fida', 'basis_files')):
  matlab_command = "addpath(genpath(fullfile(pwd,'simulators','fida'))); "
  matlab_command += "Bfield=" + str(omega/GYROMAGNETIC_RATIO) + "; "
  matlab_command += "Npts=" + str(npts) + "; "
  matlab_command += "sw="+str(1/adc_dt)+"; "

  matlab_command += "metabolites={"
  for m in metabolite_names:
    matlab_command += '\'' + fida_metabolite_name(m) + '\','
  matlab_command = matlab_command.rstrip(',')
  matlab_command+="}; "

  matlab_command += "linewidths=[" + str(linewidth) + "]; "
  matlab_command += "save_dir='" + save_dir + "'; "

  matlab_command += "run_custom_simMegaPressShapedEdit; exit; exit;"

  try:
    p = subprocess.Popen(['matlab', '-nosplash', '-nodisplay', '-r', matlab_command])
  except OSError as e:
    if e.errno == errno.ENOENT:
      raise Exception('Matlab is not installed on this system! Can\'t simulate FID-A spectra.\n'
                      'You can simulate them on another system, and put them into ' +
                      os.path.join('data', 'basis', 'fida')) from e
    else:
      raise
  p.wait()

def fida_metabolite_name(name):
  # Converts to the expected value for FID-A. See FID-A/simulationTools/metabolites for options"""
  name = name.lower()
  m_names = {'creatine': 'Cr', 'gaba': 'GABA', 'glutamate': 'Glu', 'glutamine': 'Gln', 'lactate': 'Lac',
             'myo-inositol': 'Ins', 'n-acetylaspartate': 'NAA', 'taurine': 'Tau'}
  return m_names[name]
