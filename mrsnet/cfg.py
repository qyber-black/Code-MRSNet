# mrsnet/cfg.py - MRSNet - config
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2021, Frank C Langbein <frank@langbein.org>, Cardiff University

import os
import matplotlib.pyplot as plt

class Cfg:
  # Default configuration - do not overwrite here but set alternatives in file
  # These are static variables for the class, accessed via the class. No object
  # of this class should be used; all methods are static.
  val = {
    'path_basis': None,
    'path_simulation': None,
    'path_model': None,
    'path_benchmark': None,
    'figsize': (26.67,15.0),
    'default_screen_dpi': 96,
    'screen_dpi': None,
    'image_dpi': [96]#96
  }
  # Development flags for extra functionalities and test (not relevant for use).
  # These are set via the environment vairbale MRSNET_DEV (colon separated list),
  # but all in use should be in the comments here for reference.
  dev = set()
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
            raise Exception("Unknown config file entry %s in %s" % (k,Cfg.file))
    # Check data folders and create as needed
    data_dir = os.path.join(os.path.dirname(bin_path),'data')
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
        Cfg.dev.add(f)

  @staticmethod
  def _screen_dpi():
    # DPI for plots on screen
    #try:
    #  from screeninfo import get_monitors
    #except ModuleNotFoundError:
    #  return Cfg.val['default_screen_dpi']
    #try:
    #  m = get_monitors()[0]
    #except:
    #  return Cfg.val['default_screen_dpi']
    #from math import hypot
    #try:
    #  return hypot(m.width, m.height) / hypot(m.width_mm, m.height_mm) * 25.4
    #except:
    return Cfg.val['default_screen_dpi']
