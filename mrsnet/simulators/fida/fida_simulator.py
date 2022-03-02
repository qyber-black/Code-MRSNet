# simulators/fida/fida_simulator.py - MRSNet - generated simulated FID-A basis spectrum
#
# SPDX-FileCopyrightText: Copyright (C) 2019 Max Chandler, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2020-2021 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2021 S Shermer <lw1660@gmail.com> Swansea University
# SPDX-License-Identifier: BSD-3-Clause

import os
import errno
import subprocess

from mrsnet.cfg import Cfg

def fida_spectra(metabolite_names, omega, linewidth, npts, sample_rate, source, save_dir):
  if source == "fid-a":
    script="run_custom_simMegaPressShapedEdit"
  elif source =="fid-a-2d":
    script="run_custom_simMegaPress_2D"
  else:
    raise Exception(f"Unknown fid-a basis {source}")

  matlab_command = "addpath(genpath(fullfile('"+Cfg.val['path_root']+"','mrsnet','simulators','fida')));"
  matlab_command += "mrsnet_omega="+str(omega)+";"
  matlab_command += "npts="+str(npts)+";"
  matlab_command += "sw="+str(sample_rate)+";"

  matlab_command += "metabolites={"
  for m in metabolite_names:
    matlab_command += "'"+fida_metabolite_name(m)+"',"
  matlab_command = matlab_command.rstrip(',')
  matlab_command += "};"

  matlab_command += "linewidths=["+str(linewidth)+"];"
  matlab_command += "save_dir='"+save_dir+"';"

  matlab_command += script+";exit;exit;"

  try:
    p = subprocess.Popen(['matlab', '-nosplash', '-nodisplay', '-r', matlab_command])
  except OSError as e:
    if e.errno == errno.ENOENT:
      raise Exception('Matlab is not installed on this system! Can\'t simulate FID-A spectra.\n'
                      'You can simulate them on another system, and put them into ' +
                      os.path.join('data', 'basis', source)) from e
    else:
      raise
  p.wait()

# See FID-A/simulationTools/metabolites
fida_metabolite_names = {
  'ala': 'Ala',
  'asp': 'Asp',
  'cr': 'Cr',
  'gaba': 'GABA',
  'glu': 'Glu',
  'gln': 'Gln',
  'gsh': 'GSH',
  'gly': 'Gly',
  'h20': 'H2O',
  'lac': 'Lac',
  'myi': 'Ins',
  'naa': 'NAA',
  'naag': 'NAAG',
  'pcr': 'PCr',
  'scyllo': 'Scyllo',
  'tau': 'Tau',
}

def fida_metabolite_name(name):
  # Converts to the expected value for FID-A.
  return fida_metabolite_names[name.lower()]
