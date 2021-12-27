# simulators/fida/fida_simulator.py - MRSNet - generated simulated FID-A basis spectrum
#
# SPDX-FileCopyrightText: Copyright (C) 2019 Max Chandler, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2020-2021 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: BSD-3-Clause

import os
import errno
import subprocess

from mrsnet.cfg import Cfg
from mrsnet.molecules import GYROMAGNETIC_RATIO

def fida_spectra(metabolite_names, omega, linewidth, npts, adc_dt, save_dir):
  matlab_command = "addpath(genpath(fullfile('"+Cfg.val['path_root']+"','mrsnet','simulators','fida')));"
  matlab_command += "Bfield="+str(omega/GYROMAGNETIC_RATIO)+";"
  matlab_command += "Npts="+str(npts)+";"
  matlab_command += "sw="+str(1/adc_dt)+";"

  matlab_command += "metabolites={"
  for m in metabolite_names:
    matlab_command += "'"+fida_metabolite_name(m)+"',"
  matlab_command = matlab_command.rstrip(',')
  matlab_command += "};"

  matlab_command += "linewidths=["+str(linewidth)+"];"
  matlab_command += "save_dir='"+save_dir+"';"

  matlab_command += "run_custom_simMegaPressShapedEdit;exit;exit;"

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

fida_metabolite_names = {
  'naa': 'NAA',
  'cr': 'Cr',
  'gaba': 'GABA',
  'glu': 'Glu',
  'gln': 'Gln',
  'lac': 'Lac',
  'myi': 'Ins',
  'tau': 'Tau'
}

def fida_metabolite_name(name):
  # Converts to the expected value for FID-A. See FID-A/simulationTools/metabolites for options
  return fida_metabolite_names[name.lower()]
