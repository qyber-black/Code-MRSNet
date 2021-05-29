#!/usr/bin/env python3
#
# utilities/utils.py - MRSNet - utility functions
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2019, Max Chandler, PhD student at Cardiff University
# Copyright (C) 2020, Frank C Langbein <frank@langbein.org>, Cardiff University

import numpy as np

from utilities.constants import MOLECULE_NAMES

def convert_molecule_names(molecules, shorten = False):
    """Standardise molecule names to short or long form"""
    new_names = []
    for name in molecules:
        name = name.lower()
        new_name = None
        for long_name in MOLECULE_NAMES:
            if long_name.lower() == name:
                new_name = MOLECULE_NAMES[long_name][0] if shorten else long_name
                break
            else:
                for short_name in MOLECULE_NAMES[long_name]:
                    if short_name.lower() == name:
                        new_name = MOLECULE_NAMES[long_name][0] if shorten else long_name
                        break
                if not new_name is None:
                    break
        if new_name is None:
            raise Exception('\n\nMetabolite name "' + name + '" not valid.')
        new_names.append(new_name)
    return sorted(new_names)

def reshape_data(data):
    n_channels = len(data[0])
    data = collapse_array(data)
    data = data.reshape(-1, len(data[0]), n_channels, 1)
    input_shape = (len(data[0]), n_channels, 1)
    return data, input_shape

def normalise_labels(labels, normalisation):
    if normalisation == 'max':
        with np.errstate(invalid='ignore'):
            if labels.ndim == 1:
                labels = labels / np.max(labels)
            elif labels.ndim == 2:
                labels = (labels.T / np.max(labels, 1)).T
    elif normalisation == 'sum':
        with np.errstate(invalid='ignore'):
            if labels.ndim == 1:
                labels = labels / np.sum(labels)
            elif labels.ndim == 2:
                labels = (labels.T / np.sum(labels, 1)).T
    else:
        raise Exception('Unknown output normalisation for model, please write a custom routine for ' + normalisation)
    # Replace the NaNs that come from [0,0,...]/0 with 0's
    labels[np.isnan(labels)] = 0
    return labels

def collapse_array(array):
    new_array = []
    for d in array:
        for a in d:
            new_array.append(a)
    return np.array(new_array)

def normalise_signal(signal):
    # the reason we expect multiple channels is to ensure that the normalisation is the same across all the FFTs.
    # this is important in the case of MEGA-PRESS where we have two acquisitions, where we want both to be normalise
    # to the same value otherwise as NAA is edited out, the Cr peak jumps a good few points!
    return signal / np.max(np.abs(signal))

def matlabify_string(string):
    # for exporting to matlab format, some characters are invalid
    to_replace = [' ', '-', '+', '  ', '/', '&', '&']
    for r in to_replace:
        string = string.replace(r, '_')
    return string
