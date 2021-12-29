# mrsnet/basis.py - MRSNet - spectral basis
#
# SPDX-FileCopyrightText: Copyright (C) 2019 Max Chandler, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2020-2021 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import copy
import numpy as np
import matplotlib.pyplot as plt

import mrsnet.molecules as molecules
from mrsnet.spectrum import Spectrum

class BasisCollection:
  def __init__(self):
    self._bases = {}

  def __iter__(self):
    return BasisCollectionIterator(self)

  def __str__(self):
    str = "BasisCollection: \n" + "\n".join(["  "+idx for idx in self._bases])
    return str

  def add(self, metabolites, source, manufacturer, omega, linewidth, pulse_sequence,
          sample_rate, samples, path_basis=os.path.join('data','basis')):
    basis = Basis(metabolites=metabolites, source=source,
                  manufacturer=manufacturer, omega=omega,
                  linewidth=linewidth, pulse_sequence=pulse_sequence,
                  sample_rate=sample_rate, samples=samples).setup(path_basis)
    idx = basis.name()
    if idx in self._bases:
      raise Exception(f"Basis {idx} already in basis collection")
    self._bases[idx] = basis

  def get(self, metabolites, source, manufacturer, omega, linewidth, pulse_sequence):
    idx = "_".join(["-".join(sorted(metabolites)), source, manufacturer,
                    str(omega), str(linewidth), pulse_sequence])
    if idx in self._bases:
      return self._bases[idx]
    return None

  def describe(self):
    res = []
    for k in self._bases.keys():
      idx = k.split("_")
      res.append({
          'metabolites': idx[0].split("-"),
          'source': idx[1],
          'manufacturer': idx[2],
          'omega': idx[3],
          'linewidth': idx[4],
          'pulse_sequence': idx[5]
        })
    return res

class BasisCollectionIterator:
  def __init__(self, basis_collection):
    self._basis_collection = basis_collection
    self._indices = [k for k in basis_collection._bases]
    self._index = 0

  def __next__(self):
    if self._index < len(self._indices):
      res = self._basis_collection._bases[self._indices[self._index]]
      self._index += 1
      return res
    raise StopIteration

class Basis:
  # The Basis object is used to contain sets of single metabolite spectra,
  # and is used to generate combined spectra of them. We assume the spectra
  # are for one basis_source, manufacturer, pulse_sequence, omega, linewidth

  def __init__(self, metabolites=[], source=None, manufacturer=None,
               omega=None, linewidth=None, pulse_sequence=None, sample_rate=2500, samples=4096):
    self.metabolites = sorted(metabolites)
    self.source = source
    self.manufacturer = manufacturer
    self.omega = omega
    self.pulse_sequence = pulse_sequence

    if self.source == "lcmodel":
      self.linewidth = None # Unknown
      self.sample_rate = 2000 # Taken from lcmodel basis file
      self.samples = 8192 # Taken from lcmodel basis file
    else:
      self.linewidth = linewidth
      self.sample_rate = sample_rate
      self.samples = samples


    self.acquisitions = None # Indicates if setup or not and lists acqusitions
    self.spectra = {}  # Dict of spectra in basis: spectra['METABOLITE'] = {'ACQUISITION': Spectrum}

  def __str__(self):
    return ("Basis: \n  Metabolites: %s\n  Source: %s\n  Manufacturer: %s\n  Omega: %f\n  Linewidth: %s\n  Pulse Sequence: %s\n  Sample Rate: %f\n  Samples: %d"
            % ("-".join(self.metabolites), self.source, self.manufacturer,
               self.omega, str(self.linewidth), self.pulse_sequence,
               self.sample_rate, self.samples))

  def name(self):
    return "_".join(["-".join(self.metabolites), self.source, self.manufacturer,
                     str(self.omega), str(self.linewidth), self.pulse_sequence,
                     str(self.sample_rate), str(self.samples)])

  def setup(self, path_basis=os.path.join('data','basis')):
    if self.acquisitions is not None:
      return self
    # Load set
    self._load(path_basis)
    if len(self.spectra) < 1:
      raise Exception("No basis loaded")
    # Add missing spectra (for megapress)
    self._add_missing_spectra()
    # Check for consistency
    for m in self.metabolites:
      if m not in self.spectra.keys():
        raise Exception(f"Metabolite {m} not in basis")
    acqs = []
    for m in self.spectra.keys():
      if m not in self.metabolites:
        raise Exception(f"Basis spectra contain additional spectrum for {m}")
      if len(acqs) == 0:
        acqs = sorted(self.spectra[m].keys())
        if len(acqs) == 0:
          raise Exception("No acquisitons")
      else:
        if sorted(self.spectra[m].keys()) != acqs:
          raise Exception("Acquisitions between metabolite spectra for basis not consistent")
    self.acquisitions = acqs
    # Global normalisation
    self._normalise()
    # Global B0 correction
    self._correct_b0()
    return self

  def _load(self,path_basis):
    glx = False
    if 'GlX' in self.metabolites:
      self.metabolites.remove("GlX")
      self.metabolites.append("Glu")
      self.metabolites.append("Gln")
      self.metabolites.sort()
      glx = True
    if self.source == 'lcmodel':
      if self.linewidth != None:
        raise Exception('Cannot supply LCModel basis set with linewidths argument. It is not a simulator option; it has one fixed linewidth.')
      self._load_lcm(path_basis=os.path.join(path_basis,'lcmodel'))
    elif self.source == 'fid-a':
      if self.manufacturer == 'siemens':
        self._load_fida(path_basis=os.path.join(path_basis,'fid-a'))
      else:
        raise Exception('No FID-A simulator for ' + self.manufacturer + ' scanner.')
    elif self.source == 'pygamma':
      if self.manufacturer == 'siemens':
        self._load_pygamma(path_basis=os.path.join(path_basis,'pygamma'))
      else:
        raise Exception('No PyGamma simulator for ' + self.manufacturer + ' scanner.')
    else:
      raise Exception('Unknown basis source: ' + self.source)
    if glx:
      if 'Glu' not in self.spectra or 'Gln' not in self.spectra:
        raise Exception("Cannot use GlX as Glu/Gln not in basis")
      spec = {}
      for a in self.spectra['Gln'].keys():
        if a in self.spectra['Glu']:
          spec[a] = Spectrum.comb(1.0,self.spectra['Gln'],1.0,self.spectra['Glu'],
                                  self.spectra['Gln'][a].id+"_+_"+self.spectra['Glu'][a].id,
                                  a)
      self.spectra['GlX'] = spec
      del self.spectra['Gln']
      del self.spectra['Glu']
      self.metabolites.remove("Glu")
      self.metabolites.remove("Gln")
      self.metabolites.append("GlX")
      self.metabolites.sort()

  def _load_fida(self, path_basis, second_call=False):
    if not os.path.join(path_basis,'basis_files'):
      os.makedirs(os.path.join(path_basis,'basis_files'))
    to_simulate = copy.copy(self.metabolites)
    for file in os.listdir(os.path.join(path_basis,'basis_files')):
      if file.startswith('FIDA2D_') and file.endswith('.mat'): # FIXME: FIDA!
        vals = file.split("_")
        try:
          if vals[2].lower() == self.pulse_sequence \
             and (vals[3] == "EDITON" or vals[3] == "EDITOFF") \
             and np.abs(float(vals[4]) - self.linewidth) < 1e-2 \
             and int(vals[5]) == self.sample_rate \
             and int(vals[6]) == self.samples \
             and np.abs(float(vals[7][0:-4]) - self.omega) < 1e-2:
            spec = Spectrum.load_fida(os.path.join(path_basis,'basis_files',file),file[0:-4])
            if len(spec.metabolites) > 1:
              raise Exception("More than one metabolite in FID-A basis")
            if spec.metabolites[0].lower() in [x.lower() for x in self.metabolites] \
               and np.abs(spec.linewidth - self.linewidth) < 1e-2 \
               and np.abs(spec.omega - self.omega) < 1e-2 \
               and spec.sample_rate == self.sample_rate \
               and len(spec.fft) == self.samples:
              if spec.metabolites[0] not in self.spectra:
                self.spectra[spec.metabolites[0]] = {}
              self.spectra[spec.metabolites[0]][spec.acquisition] = spec
              if spec.metabolites[0] in to_simulate:
                to_simulate.remove(spec.metabolites[0])
        except:
          pass
    if len(to_simulate) > 0:
      if second_call:
        raise Exception('Recursion error, should have simulated spectra - but I can\'t seem to find it and '
                        'I\'m going to end up in an endless loop.')
      else:
        print('Some spectra are missing, simulating: ' + str(to_simulate))
        from mrsnet.simulators.fida.fida_simulator import fida_spectra
        fida_spectra(to_simulate, omega=self.omega, linewidth=self.linewidth,
                     npts=self.samples, sample_rate=self.sample_rate,
                     save_dir=os.path.join(path_basis,'basis_files'))
        self._load_fida(path_basis, second_call=True)

  def _load_pygamma(self, path_basis=os.path.join('data', 'basis', 'pygamma'), second_call=False):
    # Constants, synchronise with pygamma_simulator (passed as arguments, but defaults hardcoded
    # in pygamma simulator as well)
    for metabolite_name in self.metabolites:
      specs = Spectrum.load_pygamma(path_basis, metabolite_name,
                                    self.pulse_sequence, self.omega,
                                    self.linewidth, self.samples, 1.0/self.sample_rate)
      for s in specs:
        if s.metabolites[0] not in self.spectra:
          self.spectra[s.metabolites[0]] = {}
        self.spectra[s.metabolites[0]][s.acquisition] = s

  def _load_lcm(self, path_basis=os.path.join('data', 'basis', 'lcmodel')):
    if self.manufacturer == 'siemens':
      m_str = 'Siemens'
    elif self.manufacturer == 'ge':
      m_str = 'GE'
    elif self.manufacturer == 'phillips':
      m_str = 'Phillips'
    else:
      raise Exception('No LCModel basis set for ' + scanner_manufacturer + ' scanner')
    if self.pulse_sequence == 'megapress':
      p_str = 'MEGAPRESS'
      acqs = ['edit_off', 'difference']
    else:
      raise Exception('No LCModel basis set for ' + self.pulse_sequence + ' pulse sequence')
    for a in acqs:
      diff_var_str = '_kasier' if a == 'difference' else ''
      specs = Spectrum.load_lcm(os.path.join(path_basis,'basis_files',
                                p_str+"_"+a+"_"+m_str+"_3T"+diff_var_str+".basis"),
                                a, self.omega, self.metabolites)
      for s in specs:
        if len(s.metabolites) > 1:
          raise Exception("More than one metabolite in basis spectrum")
        if s.metabolites[0] in self.metabolites:
          if s.metabolites[0] not in self.spectra:
            self.spectra[s.metabolites[0]] = {}
          self.spectra[s.metabolites[0]][s.acquisition] = s
    for m in self.spectra.keys():
      if 'difference' not in self.spectra[m].keys():
        self.spectra[m]['difference'] = copy.deepcopy(self.spectra[m]['edit_off'])
        self.spectra[m]['difference'].acquisition = 'difference'
        self.spectra[m]['difference'].set_f(np.zeros_like(self.spectra[m]['edit_off'].fft),
                                            self.spectra[m]['edit_off'].sample_rate,
                                            center_ppm=self.spectra[m]['edit_off'].center_ppm,
                                            b0_shift_ppm=self.spectra[m]['edit_off'].b0_shift_ppm)

  def _add_missing_spectra(self):
    if self.pulse_sequence == 'megapress':
      for m in self.spectra.keys():
        if 'edit_on' in self.spectra[m] and 'edit_off' in self.spectra[m] and 'difference' in self.spectra[m]:
          pass
        elif 'edit_on' in self.spectra[m] and 'edit_off' in self.spectra[m]:
          self.spectra[m]['difference'] = Spectrum.comb(1.0,self.spectra[m]['edit_on'],-1.0,self.spectra[m]['edit_off'],
                                                        self.spectra[m]['edit_on'].id+":ON_-_OFF:"+self.spectra[m]['edit_off'].id,
                                                        "difference")
        elif 'edit_off' in self.spectra[m] and 'difference' in self.spectra[m]:
          self.spectra[m]['edit_on'] = Spectrum.comb(1.0,self.spectra[m]['difference'],1.0,self.spectra[m]['edit_off'],
                                                     self.spectra[m]['difference'].id+"_+_"+self.spectra[m]['edit_off'].id,
                                                     "edit_on")
        elif 'edit_on' in self.spectra[m] and 'difference' in self.spectra[m]:
          self.spectra[m]['edit_off'] = Spectrum.comb(1.0,self.spectra[m]['edit_on'],1.0,self.spectra[m]['difference'],
                                                      self.spectra[m]['edit_on'].id+"_-_"+self.spectra[m]['difference'].id,
                                                      "edif_off")
        else:
          raise Exception(f"Incomplete megapress spectrum for {m}")

  def _normalise(self):
    # All spectra are normalised against the maximum fft magnitude.
    # This is mainly to ensure the same noise is added (if it is added).
    global_max = 0.0
    for m in self.spectra.keys():
      for a in self.spectra[m].keys():
        fft, _ = self.spectra[m][a].get_f()
        global_max = np.max([global_max, np.max(np.abs(fft))])
    for m in self.spectra.keys():
      for a in self.spectra[m].keys():
        self.spectra[m][a].scale /= global_max

  def _correct_b0(self):
    # There's not going to be an individual shift per metabolite...
    # so we calibrate the entire set against the priority reference peak
    b0_shift = None
    peak_val = 0.0
    if self.pulse_sequence == "megapress":
      for pair in molecules.B0_CORRECTION:
        if pair[0] in self.spectra:
          shift, val = self.spectra[pair[0]]['edit_off'].correct_b0()
          if shift is not None and peak_val < val:
            b0_shift = shift
            peak_val = val
    else:
      raise Exception(f"No B0 correction for {self.pulse_sequence}")
    if b0_shift == None:
      raise Exception("B0 correction for basis failed")
    # Apply shift
    for m in self.spectra.keys():
      for a in self.spectra[m].keys():
        self.spectra[m][a].correct_b0(b0_shift)

  def combine(self, concentrations, id):
    if len(concentrations) != len(self.spectra.keys()):
      raise Exception("Concentrations do not match number of spectra in basis")
    spectra = {}
    con = {}
    if type(concentrations) is dict:
      for m in self.metabolites:
        con[m] = concentrations[m]
    else:
      l = 0
      for m in self.metabolites:
        con[m] = concentrations[l]
        l += 1
    concentrations = [con[m] for m in self.metabolites]
    shifts = []
    for a in self.acquisitions:
      spectra[a] = Spectrum.combs(concentrations,
                                  [self.spectra[m][a] for m in self.metabolites],
                                  id, a)
    Spectrum.correct_b0_multi(spectra)
    # Sanity check for difference
    if self.pulse_sequence == "megapress" and \
       np.max(np.abs(spectra['edit_on'].get_f()[0] - spectra['edit_off'].get_f()[0] - spectra['difference'].get_f()[0])) > 1e-8:
      raise Exception("Coimbined difference spectrum differs from edit_on - edit_off")
    return spectra, con

  def plot(self, data='magnitude', type='fft'):
    acqs = self.acquisitions
    num_m = len(self.spectra)
    num_a = len(acqs)
    fig, axes = plt.subplots(num_m, num_a, sharex=True, sharey='row')
    if len(axes.shape) == 1:
      axes = np.reshape(axes, (1,num_a))

    super_title = type.upper()+' Basis: ' + self.source + " (" + self.manufacturer + ") @ " + str(self.omega) + "Hz Linewidth: " + str(self.linewidth) + " - " + self.pulse_sequence.upper()
    plt.suptitle(super_title)

    from matplotlib.transforms import offset_copy
    pad = 5
    col = 0
    for a in sorted(acqs):
      row = 0
      for m in sorted(self.spectra.keys()):
        self.spectra[m][a].plot(axes[row,col], type=type, mode=data)
        ml = molecules.convert_names([m], shorten=True)[0]
        if col == 0:
          axes[row,col].annotate(ml, xy=(0, 0.5), xytext=(-axes[row,col].yaxis.labelpad - pad, 0),
                                 xycoords=axes[row,col].yaxis.label, textcoords='offset points',
                                 size='large', ha='right', va='center')
        else:
          axes[row,col].set_ylabel("")
        if row == 0:
          axes[row,col].annotate(a, xy=(0.5, 1), xytext=(0, pad),
                                 xycoords='axes fraction', textcoords='offset points',
                                 size='large', ha='center', va='baseline')
        if row != num_m-1:
          axes[row,col].set_xlabel("")
        row += 1
      col += 1

    return fig
