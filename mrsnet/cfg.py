# mrsnet/cfg.py - MRSNet - config
#
# SPDX-FileCopyrightText: Copyright (C) 2021 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import matplotlib.pyplot as plt

class Cfg:
  # Default configuration - do not overwrite here but set alternatives in file
  # These are static variables for the class, accessed via the class. No object
  # of this class should be used; all methods are static.
  val = {
    'path_root': None,
    'path_basis': None,
    'path_simulation': None,
    'path_model': None,
    'path_benchmark': None,
    'figsize': (26.67,15.0),
    'fft_peak_location_estimator': 'jain' , # 'quadratic' or 'quinn2' or 'jain' or None
    'b0_correct_ppm_range': 0.25,
    'water_peak_ppm_range': 0.75,
    'water_peak_under_mean_max': 10,
    'spectrum_rescale_fft_max_repeats': 10,
    'filter_dicom': None,         # 'hamming' or 'hanning' or 'kaiser' or None dicom spectral leaking filter
    'filter_dicom_duration': 1.0, # in seconds; determines length of filter window
                                  # (we only use the right half of the filter)
    'filter_dicom_kaiser': 2.5,   # kaiser filter beta shape parameter
                                  # (0: rectangular; 5,6 ~hamming,hanning; 8.6 ~blackman; 14 default)
    'phase_correct': None,           # 'acme' or 'ernst' or None phase correction algorithm
    'phase_correct_acme_gamma': 100, # penalty weight for acme phase correction
    'npfft_module': ['pyfftw.interfaces','numpy_fft'], # ['pyfftw.interfaces','numpy_fft'] or
                                                       # ['numpy','fft'] or ['scipy','fft'], etc.
    'default_screen_dpi': 96,
    'screen_dpi': None,
    'image_dpi': [300]
  }
  # Development flags for extra functionalities and test (not relevant for use).
  # These are set via the environment vairbale MRSNET_DEV (colon separated list),
  # but all in use should be in the comments here for reference:
  # * selectgpo_optimise_noload: do not load existing datapoints for GPO model selection
  # * spectrum_set_phase_correct: show phase correction effect
  dev_flags = set()
  # check_dataset_export - Test if exporting dataset to tensors is correct in train
  # flag_plots - Show some test graphs for the activated checks
  # feature_selectgpo_optimse_noload - do not load existing results for SelectGPO
  file = os.path.expanduser(os.path.join('~','.config','mrsnet.json'))

  @staticmethod
  def init(bin_path):
    # Load cfg file - data folders and other Cfg values can be overwritten by config file
    if os.path.isfile(Cfg.file):
      import json
      with open(Cfg.file, "r") as fp:
        js = json.load(fp)
        for k in js.keys():
          if k in Cfg.val:
            Cfg.val[k] = js[k]
          else:
            raise Exception(f"Unknown config file entry {k} in {Cfg.file}")
    Cfg.val["path_root"] = os.path.dirname(bin_path)
    # Check data folders and create as needed
    data_dir = os.path.join(Cfg.val["path_root"],'data')
    paths = {
      "path_basis": "basis",
      "path_simulation": "sim-spectra",
      "path_model": "model",
      "path_benchmark": "benchmark"
    }
    for p in paths:
      if Cfg.val[p] == None:
        Cfg.val[p] = os.path.join(data_dir,paths[p])
      if not os.path.isdir(Cfg.val[p]):
        os.makedirs(Cfg.val[p])
    # Setup plot defaults
    if Cfg.val["screen_dpi"] == None:
      Cfg.val["screen_dpi"] = Cfg._screen_dpi()
    plt.rcParams["figure.figsize"] = Cfg.val['figsize']
    # Dev flags
    if 'MRSNET_DEV' in os.environ:
      for f in os.environ['MRSNET_DEV'].split(":"):
        Cfg.dev_flags.add(f)

  @staticmethod
  def dev(flag):
    return flag in Cfg.dev_flags

  @staticmethod
  def _screen_dpi():
    # DPI for plots on screen
    try:
      from screeninfo import get_monitors
    except ModuleNotFoundError:
      return Cfg.val['default_screen_dpi']
    try:
      m = get_monitors()[0]
    except:
      return Cfg.val['default_screen_dpi']
    from math import hypot
    try:
      dpi = hypot(m.width, m.height) / hypot(m.width_mm, m.height_mm) * 25.4
      if dpi > Cfg.val['image_dpi'][0]: # screen_dpi bound by image_dpi to avoid huge resolutions
        return Cfg.val['image_dpi'][0]
      return dpi
    except:
      return Cfg.val['default_screen_dpi']
