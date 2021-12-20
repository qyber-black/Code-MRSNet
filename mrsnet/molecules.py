# mrsnet/metabolites.py - MRSNet - metabolites
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2019, Max Chandler, PhD student at Cardiff University
# Copyright (C) 2020-2021, Frank C Langbein <frank@langbein.org>, Cardiff University

# Molecule names understood by MRNet; mapping long-name to range of short-names.
# Preferred short name is the first in the array, e.g. for myo-inositol 'MyI' will be chosen over 'mi'
NAMES = {
  'N-Acetylaspartate': ['NAA'],
  'Creatine': ['Cr', 'cre'],
  'GABA': ['GABA'],
  'Choline-truncated': ['Cho'],
  'Glutamate': ['Glu'],
  'Glutamine': ['Gln'],
  'GlutaX': ['GlX'],
  'Glutathione': ['gsh'],
  'Glycine': ['Gly'],
  'Lactate': ['Lac'],
  'Myo-Inositol': ['MyI', 'mi', 'ins'],
  'N-Acetylaspartylglutamic': ['NAAG'],
  'NAAG-truncated-siemens': ['NAAG-SIE'],
  'Phosphocreatine': ['PCr', 'pch'],
  'Taurine': ['Tau'],
  'Water': ['H2O'],
  'DSS': ['DSS'],
  'Alanine': ['ala'],
  'Aspartate': ['asp'],
  'Scyllo-Inositol': ['scyllo']
}

NAA_REFERENCE = -2.01
CR_REFERENCE = -3.015
WATER_REFERENCE = -4.75             # Temperature dependant, avoid using if at all possible
GYROMAGNETIC_RATIO = 42.57747892    # 1H (MHz/T) : https://physics.nist.gov/cgi-bin/cuu/Value?gammapbar

def convert_names(molecules, shorten=False):
  # Standardise molecule names to short or long form
  new_names = []
  for name in molecules:
    name = name.lower()
    new_name = None
    for long_name in NAMES:
      if long_name.lower() == name:
        new_name = NAMES[long_name][0] if shorten else long_name
        break
      for short_name in NAMES[long_name]:
        if short_name.lower() == name:
          new_name = NAMES[long_name][0] if shorten else long_name
          break
    if new_name is None:
      raise Exception('Metabolite name "' + name + '" invalid')
    new_names.append(new_name)
  return new_names

def short_name(names):
  # Standard short molecule name
  if isinstance(names,list):
    return convert_names(names, shorten=True)
  return convert_names([names], shorten=True)[0]

def long_name(names):
  # Standard long molecule name
  if isinstance(names,list):
    return convert_names(names)
  return convert_names([names])[0]
