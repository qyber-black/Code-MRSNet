# simulators/fida/fida_simulator.py - MRSNet - generated simulated FID-A basis spectrum
#
# SPDX-FileCopyrightText: Copyright (C) 2019 Max Chandler, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2020-2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2021 S Shermer <lw1660@gmail.com> Swansea University
# SPDX-License-Identifier: BSD-3-Clause

"""FID-A simulator for MRSNet.

This module provides interfaces to the FID-A MATLAB toolbox for
simulating MRS basis spectra using various pulse sequences.
"""

import errno
import os
import subprocess

from mrsnet.cfg import Cfg


def fida_spectra(metabolite_names, omega, linewidth, npts, sample_rate, source, save_dir):
  """Generate FID-A spectra using MATLAB.

  Args:
      metabolite_names (list): List of metabolite names to simulate
      omega (float): B0 field strength in Hz
      linewidth (float): Linewidth parameter
      npts (int): Number of points in the spectrum
      sample_rate (float): Sample rate in Hz
      source (str): FID-A source type ("fid-a" or "fid-a-2d")
      save_dir (str): Directory to save generated spectra

  Raises
  ------
      RuntimeError: If source type is unknown or MATLAB is not installed
  """
  if source == "fid-a":
    script="run_custom_simMegaPressShapedEdit"
  elif source =="fid-a-2d":
    script="run_custom_simMegaPress_2D"
  else:
    raise RuntimeError(f"Unknown fid-a basis {source}")

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
  # Enable cached-unbroadened pathway for fid-a-2d and set cache_dir under basis path
  if source == "fid-a-2d":
    cache_dir = os.path.join(Cfg.val['path_basis'], 'fid-a-2d', 'cache_unbroadened')
    matlab_command += "use_cached_unbroadened=true;"
    matlab_command += "cache_dir='"+cache_dir+"';"

  matlab_command += script+";exit;exit;"

  try:
    p = subprocess.Popen(['matlab', '-nosplash', '-nodisplay', '-r', matlab_command])  # noqa: S603, S607
  except OSError as e:
    if e.errno == errno.ENOENT:
      raise RuntimeError('Matlab is not installed on this system! Can\'t simulate FID-A spectra.\n'
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
  """Convert metabolite name to FID-A expected format.

  Args:
      name (str): Metabolite name to convert

  Returns
  -------
      str: FID-A compatible metabolite name
  """
  # Converts to the expected value for FID-A.
  return fida_metabolite_names[name.lower()]
