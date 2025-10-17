# mrsnet/cfg.py - MRSNet - config
#
# SPDX-FileCopyrightText: Copyright (C) 2021-2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2022-2025 Zien Ma, PhD student at Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Configuration management module for MRSNet.

This module provides centralized configuration management for the MRSNet package.
It handles loading configuration values from multiple sources with a clear precedence
order, manages data directories, and provides development flags for testing.

The configuration system supports:
- Default values defined in the code
- Root configuration file (cfg.json) that overrides defaults
- User configuration file (~/.config/mrsnet.json) that overrides both
- Environment variables for development flags
- Automatic creation of required data directories

Key features:
- Static configuration class with no instances needed
- Hierarchical configuration loading with clear precedence
- Development flags for testing and debugging
- Automatic data directory management
- Matplotlib integration for plotting defaults
"""

import json
import os

import matplotlib.pyplot as plt


class Cfg:
  """Configuration management for MRSNet.

  This class provides static configuration management for the MRSNet package.
  All methods are static and no instances of this class should be created.

  Configuration values can be overridden by:
  1. ROOT_PATH/cfg.json (generated after first run; overwrites defaults)
  2. ~/.config/mrsnet.json (user config; overwrites cfg.json and defaults)

  Attributes
  ----------
      val (dict): Dictionary containing all configuration values
      dev_flags (set): Set of development flags from MRSNET_DEV environment variable
      file (str): Path to user configuration file
  """

  # Default configuration - do not overwrite here but set alternatives in file
  # These are static variables for the class, accessed via the class. No object
  # of this class should be used; all methods are static.
  #
  # Change these values in ROOT_PATH/cfg.json (generated after first run; overwrites
  # defaults here) or ~/config/mrsnet.json (not generated; overwrites cfg.json and
  # defaults here).
  val = {  # noqa: RUF012
    'path_root': None,        # MRSNet root path
    'path_basis': None,       # Save path for basis, searched first
    'path_simulation': None,  # Save path for simulation, searched first
    'path_model': None,       # Save path for model, searched first
    'path_benchmark': None,   # Save path for benchmark, searched first
    'path_sim2real': None,    # Save path for sim-to-real comparison outputs
    'search_basis': [],       # Basis path list for searching
    'search_simulation': [],  # Simulation path list for searching
    'search_model': [],       # Model path list for searching
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
    'lcmodel_megapress_difference_variant': 'kasier', # kasier (sic) or govindaraju
    'num_eps': 1e-8,            # numerical epsilon for float comparisons
    'base_learning_rate': 1e-4, # Learning rate for batch size 16 (scaled linearly
                                # with batch_size - https://arxiv.org/abs/1706.02677)
    'beta1': 0.9, # Adam beta1
    'beta2': 0.999, # Adam beta2
    'epsilon': 1e-8, # Adam epsilon
    'ga_num_parents_mating': 10, # SelectGA number of parents selected for mating
    'ga_max_init_pop': 10, # SelectGA maximum initial population
    'default_screen_dpi': 96,
    'screen_dpi': None,
    'image_dpi': [300],
    'analysis_spectra_error_dist_sampling': 5, # Percentage of error samples to estimate
                                               # error distribution for spectra matching (for autoencoders)
    'analysis_predicted_spectra_samples': 10, # Max. number of predicted spectra plotted when ground truth available
    'disable_gpu': False, # Disable gpu for tensorflow (for testing)
    # Training control defaults
    'early_stopping_patience': 25,
    'reduce_lr_min_lr': 1e-7,
    'reduce_lr_patience': 20,           # default patience for ReduceLROnPlateau
    # Prediction pipeline batch sizes
    'predict_batch_size': 32,
    # EarlyStopping monitors
    'es_monitor_metric_quant': 'mae',   # used by quantification models (cnn, fcnn, qnet, qmrs, encdec)
    'es_monitor_metric_caeq': 'q_mae',  # used by combined CAEQ model (autoencoder-quantifier)
    'es_monitor_metric_ae': 'loss',     # used by pure autoencoders
    'es_min_delta': 1e-8,               # min delta
    # Learning-rate scheduler monitors (ReduceLROnPlateau)
    'lr_monitor_metric_quant': 'loss',  # drive LR by optimised loss for quantifiers
    'lr_monitor_metric_caeq': 'loss',   # drive LR by total loss for CAEQ
    'lr_monitor_metric_ae': 'loss',     # drive LR by loss for autoencoders
    # CAEQ head loss weights (hardcoded interface for now; future: per-run)
    'caeq_weight_ae': 1.0,
    'caeq_weight_q': 1.0,
    # CAEQ quantifier weight ramp
    'caeq_weight_ramp': True,
    'caeq_weight_ramp_warmup_epochs': 25,
    'caeq_weight_ramp_cooldown': 5,
    'caeq_weight_q_start': 0.1,         # starting weight for quantifier during ramp (target: caeq_weight_q)
    # Training stability/perf
    'optimizer_clipnorm': 1.0,          # sensible default for gradient clipping by norm; set 0.0 to disable
    'optimizer_clipvalue': 0.0,         # >0 enables gradient clipping by value
    'cache_datasets': True,             # cache tf.data pipelines before shuffle/batch
    'mixed_precision': True,            # enable global mixed precision policy before model build
    'mixed_precision_policy': 'mixed_float16',
    'mixed_precision_auto_policy': True, # choose AMP policy automatically based on hardware
    # Determinism and AMP policy
    'deterministic_ops': False,         # enable deterministic TF ops (may reduce performance)
  }
  # Development flags for extra functionalities and test (not relevant for use).
  # These are set via the environment variable MRSNET_DEV (colon separated list),
  # but all in use should be in the comments here for reference:
  # * check_dataset_export - Test if exporting dataset to tensors is correct in train
  # * flag_plots - Show some test graphs for the activated checks
  # * selectgpo_optimse_noload - do not load existing results for SelectGPO
  # * selectgpo_no_search - only use existing results; do not execute GPO search
  # * spectrum_set_phase_correct: show phase correction effect
  dev_flags = set()  # noqa: RUF012
  file = os.path.expanduser(os.path.join('~','.config','mrsnet.json'))

  @staticmethod
  def init(bin_path):
    """Initialize MRSNet configuration.

    Sets up the configuration system by:
    1. Setting the root path from the binary path
    2. Loading configuration from cfg.json and user config files
    3. Creating necessary data directories
    4. Setting up matplotlib defaults
    5. Loading development flags from environment

    Parameters
    ----------
        bin_path (str): Path to the MRSNet binary/script
    """
    # Root path of mrsnet
    Cfg.val["path_root"] = os.path.dirname(bin_path)
    # Load cfg file - data folders and other Cfg values can be overwritten by config file
    # We first load ROOT/cfg.json, if it exists, then the user config file
    root_cfg_file = os.path.join(Cfg.val["path_root"],'cfg.json')
    root_cfg_vals = {}
    for fc in [root_cfg_file, Cfg.file]:
      if os.path.isfile(fc):
        with open(fc) as fp:
          js = json.load(fp)
          if fc == root_cfg_file:
            root_cfg_vals = js
          for k in js.keys():
            if k in Cfg.val:
              Cfg.val[k] = js[k]
            else:
              if fc != root_cfg_file: # We fix this here later
                raise RuntimeError(f"Unknown config file entry {k} in {fc}")
    # Check data folders and create as needed
    data_dir = os.path.join(Cfg.val["path_root"],'data')
    paths = {
      "path_basis": "basis",
      "path_simulation": "sim-spectra",
      "path_model": "model",
      "path_benchmark": "benchmark",
      "path_sim2real": "sim2real"
    }
    for p in paths:
      if Cfg.val[p] is None:
        Cfg.val[p] = os.path.join(data_dir,paths[p])
      if not os.path.isdir(Cfg.val[p]):
        os.makedirs(Cfg.val[p])
    changed = False # Any changes to cfg's so we need to save it again?
    # Extend search paths, to make sure -dist files are always available
    search = {
      "search_basis": "basis-dist",
      "search_model": "model-dist"
    }
    for s in search:
      sp = os.path.join(data_dir,search[s])
      if sp not in Cfg.val[s]:
        Cfg.val[s].append(sp)
        changed = True
    # Setup plot defaults
    if Cfg.val["screen_dpi"] is None:
      Cfg.val["screen_dpi"] = Cfg._screen_dpi()
    plt.rcParams["figure.figsize"] = Cfg.val['figsize']
    # Store configs in ROOT/mrsnet.json if it does not exist
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
  def get_su_bases(reload=False):
    """Get list of SU-* basis sets.

    Searches for directories starting with 'su-' in the basis search paths
    and returns a list of available SU basis sets.

    Parameters
    ----------
        reload (bool, optional): Force reload of SU bases. Defaults to False.

    Returns
    -------
        list: List of SU basis set names (e.g., ['su-3tskyra', 'su-7t'])
    """
    if reload or not hasattr(Cfg,'_su_bases'):
      Cfg._su_bases = []
    if len(Cfg._su_bases) == 0:
      for path in [Cfg.val['path_basis'],*Cfg.val['search_basis']]:
        for fldr in next(os.walk(path))[1]:
          if fldr[0:3] == 'su-' and fldr not in Cfg._su_bases:
            Cfg._su_bases.append(fldr)
    return Cfg._su_bases

  @staticmethod
  def dev(flag):
    """Check if a development flag is set.

    Parameters
    ----------
        flag (str): Development flag to check

    Returns
    -------
        bool: True if the flag is set, False otherwise
    """
    return flag in Cfg.dev_flags

  @staticmethod
  def _screen_dpi():
    """Calculate screen DPI for plotting.

    Attempts to detect the screen DPI using the screeninfo library.
    Falls back to default DPI if detection fails.

    Returns
    -------
        float: Screen DPI value
    """
    # DPI for plots on screen
    try:
      from screeninfo import get_monitors
    except ModuleNotFoundError:
      return Cfg.val['default_screen_dpi']
    try:
      m = get_monitors()[0]
    except Exception:
      return Cfg.val['default_screen_dpi']
    from math import hypot
    try:
      dpi = hypot(m.width, m.height) / hypot(m.width_mm, m.height_mm) * 25.4
      return dpi # set in cfg.json if this is not working
    except Exception:
      return Cfg.val['default_screen_dpi']
