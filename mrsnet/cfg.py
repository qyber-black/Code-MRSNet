# mrsnet/cfg.py - MRSNet - config
#
# SPDX-FileCopyrightText: Copyright (C) 2021-2022 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import matplotlib.pyplot as plt
import json

class Cfg:
  # Default configuration - do not overwrite here but set alternatives in file
  # These are static variables for the class, accessed via the class. No object
  # of this class should be used; all methods are static.
  #
  # Change these values in ROOT_PATH/cfg.json (generated after first run; overwrites
  # defaults here) or ~/config/mrsnet.json (not generated; overwrites cfg.json and
  # defaults here).
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
    'water_peak_under_max': 10,
    'spectrum_rescale_fft_max_repeats': 10,
    'filter_dicom': None,         # 'hamming' or 'hanning' or 'kaiser' or None dicom spectral leaking filter
    'filter_dicom_duration': 1.0, # in seconds; determines length of filter window
                                  # (we only use the right half of the filter)
    'filter_dicom_kaiser': 2.5,   # kaiser filter beta shape parameter
                                  # (0: rectangular; 5,6 ~hamming,hanning; 8.6 ~blackman; 14 default)
    'phase_correct': None,           # 'acme' or 'ernst' or None phase correction algorithm
    'phase_correct_acme_gamma': 100, # penalty weight for acme phase correction
    'npfft_module': ['numpy','fft'], # ['pyfftw.interfaces','numpy_fft'] or
                                     # ['numpy','fft'] or ['scipy','fft'], etc.
    'lcmodel_megapress_difference_variant': 'kasier', # kasier or govindaraju
    'base_learning_rate': 1e-4, # Learning rate for batch size 16 (scaled linearly
                                # with batch_size - https://arxiv.org/abs/1706.02677)
    'beta1': 0.9, # Adam beta1
    'beta2': 0.999, # Adam beta2
    'epsilon': 1e-8, # Adam epsilon
    'ga_num_parents_mating': 5, # SelectGA number of parents selected for mating
    'ga_max_init_pop': 10, # SelectGA maximum initial population
    'default_screen_dpi': 96,
    'screen_dpi': None,
    'image_dpi': [300],
    'analysis_spectra_error_dist_sampling': 5, # Percentage of error samples to estimate
                                               # error distribution for spectra matching (for autoencoders)
    'analysis_predicted_spectra_samples': 10 # Max. number of predicted spectra plotted when ground truth available
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
    # Root path of mrsnet
    Cfg.val["path_root"] = os.path.dirname(bin_path)
    # Load cfg file - data folders and other Cfg values can be overwritten by config file
    # We first load ROOT/cfg.json, if it exists, then the user config file
    root_cfg_file = os.path.join(Cfg.val["path_root"],'cfg.json')
    root_cfg_vals = {}
    for fc in [root_cfg_file, Cfg.file]:
      if os.path.isfile(fc):
        with open(fc, "r") as fp:
          js = json.load(fp)
          if fc == root_cfg_file:
            root_cfg_vals = js
          for k in js.keys():
            if k in Cfg.val:
              Cfg.val[k] = js[k]
            else:
              if fc != root_cfg_file: # We fix this here later
                raise Exception(f"Unknown config file entry {k} in {fc}")
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
    # Store configs in ROOT/mrsnet.json if it does not exist
    changed = False
    del_keys = []
    for k in root_cfg_vals.keys(): # Do not store paths and remove old values
      if k[0:5] == 'path_' or k not in Cfg.val:
        del_keys.append(k)
        changed = True
    for k in del_keys:
      del root_cfg_vals[k]
    for k in Cfg.val: # Add any new values (except paths)
      if k[0:5] != 'path_' and k not in root_cfg_vals:
        root_cfg_vals[k] = Cfg.val[k]
        changed = True
    if changed:
      with open(root_cfg_file, "w") as fp:
        print(json.dumps(root_cfg_vals, indent=2, sort_keys=True), file=fp)
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
      return dpi # set in cfg.json if this is not working
    except:
      return Cfg.val['default_screen_dpi']
