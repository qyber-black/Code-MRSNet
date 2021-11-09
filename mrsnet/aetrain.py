# mrsnet/selection.py - MRSNet - selection
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2019, Max Chandler, PhD student at Cardiff University
# Copyright (C) 2020-2021, Frank C Langbein <frank@langbein.org>, Cardiff University

class AETrain:

  def __init__(self,
               model, metabolites, pulse_sequence,
               acquisitions, datatype, norm,
               validate,
               d_inp, d_out, epochs, batch_size,
               path_mode, dataset_name,
               image_dpi, screen_dpi,
               no_show, verbose):
    self.model = model
    self.metabolites = metabolites
    self.pulse_sequence = pulse_sequence
    self.acquisitions = acquisitions
    self.datatype = datatype
    self.norm = norm
    self.validate = validate
    self.d_inp = d_inp
    self.d_out = d_out
    self.epochs = epochs
    self.batch_size = batch_size
    self.path_mode = path_mode
    self.dataset_name = dataset_name
    self.image_dpi = image_dpi
    self.screen_dpi = screen_dpi
    self.no_show = no_show
    self.verbose = verbose

  def train(self):
    # FIXME: implement
    raise(Exception("Not implemented"))
