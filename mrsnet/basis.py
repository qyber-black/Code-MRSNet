# mrsnet/basis.py - MRSNet - spectral basis
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2019, Max Chandler, PhD student at Cardiff University
# Copyright (C) 2020-2021, Frank C Langbein <frank@langbein.org>, Cardiff University

import os
import copy
import numpy as np
import matplotlib.pyplot as plt

from .spectrum import Spectrum
from .molecules import convert_names

class BasisCollection:
  def __init__(self):
    self._bases = {}

  def __iter__(self):
    return BasisCollectionIterator(self)

  def __str__(self):
    str = "BasisCollection: \n" + "\n".join(["  "+idx for idx in self._bases])
    return str

  def add(self, metabolites, source, manufacturer, omega, linewidth, pulse_sequence,
          path_basis=os.path.join('data','basis')):
    basis = Basis(metabolites=metabolites, source=source,
                  manufacturer=manufacturer, omega=omega,
                  linewidth=linewidth, pulse_sequence=pulse_sequence).setup(path_basis)
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

class Basis(object):
  # The Basis object is used to contain sets of single metabolite spectra,
  # and is used to generate combined spectra of them. We assume the spectra
  # are for one basis_source, manufacturer, pulse_sequence, omega, linewidth

  def __init__(self, metabolites=[], source=None, manufacturer=None,
               omega=None, linewidth=None, pulse_sequence=None):
    self.metabolites = sorted(metabolites)
    # FIXME: Check GLX pseudo-spectrum
    self.source = source
    self.manufacturer = manufacturer
    self.omega = omega
    self.linewidth = linewidth
    self.pulse_sequence = pulse_sequence

    self.acquisitions = None # Indicates if setup or not and lists acqusitions
    self.spectra = {}  # Dict of spectra in basis: spectra['METABOLITE'] = {'ACQUISITION': Spectrum}

  def __str__(self):
    return ("Basis: \n  Metabolites: %s\n  Source: %s\n  Manufacturer: %s\n  Omega: %f\n  Lindewith: %f\n  Pulse Sequence: %s"
            % ("-".join(self.metabolites), self.source, self.manufacturer,
               self.omega, self.linewidth,self.pulse_sequence))

  def name(self):
    return "_".join(["-".join(self.metabolites), self.source, self.manufacturer,
                     str(self.omega), str(self.linewidth), self.pulse_sequence])

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
    # Normalise ADC signal globally
    self._normalise()
    # B0 correction
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
      if self.linewidth != 1.0:
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
          spec[a] = Spectrum(self.spectra['Gln'][a].id+":Gln_+_Glu:"+self.spectra['Glu'][a].id,
                             source=self.spectra['Gln'][a].source, # should be identical to Glu
                             metabolites=["GlX"],
                             pulse_sequence=self.spectra['Gln']['edit_off'].pulse_sequence, # should be identical to Glu
                             acquisition=a,
                             omega=self.spectra['Gln'][a].omega, # should be identical to Glu
                             linewidth=self.spectra['Gln'][a].linewidth, # should be identical to Glu
                             dt=self.spectra['Gln'][a].dt, # should be identical to Glu
                             center_ppm=self.spectra['Gln'][a].center_ppm, # should be identical to Glu
                             filter_fft=self.spectra['Gln'][a].filter_fft,
                             remove_water_peak=self.spectra['Gln'][a].remove_water_peak,
                             scale=1.0)
          spec[a].adc_noise_mu = self.spectra['Gln'][a].adc_noise_mu # should be identical to Glu
          spec[a].adc_noise_sigma = self.spectra['Gln'][a].adc_noise_sigma # should be identical to Glu
          spec[a].set_adc(self.spectra['Gln'][a].adc(pad=False) + self.spectra['Glu'][a].adc(pad=False))
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
      if file.endswith('.mat'):
        spec = Spectrum.load_fida(os.path.join(path_basis,'basis_files',file),file[0:-4])
        if len(spec.metabolites) > 1:
          raise Exception("More than one metabolite in FID-A basis")
        if (spec.metabolites[0].lower() in [x.lower() for x in self.metabolites]) \
            and (spec.linewidth == self.linewidth) \
            and (np.abs(spec.omega - self.omega) < 1e-8): # there are rounding errors for storing the B0
          if spec.metabolites[0] not in self.spectra:
            self.spectra[spec.metabolites[0]] = {}
          self.spectra[spec.metabolites[0]][spec.acquisition] = spec
          if spec.metabolites[0] in to_simulate:
            to_simulate.remove(spec.metabolites[0])
    if len(to_simulate) > 0:
      if second_call:
        raise Exception('Recursion error, should have simulated spectra - but I can\'t seem to find it and '
                        'I\'m going to end up in an endless loop.')
      else:
        print('Some spectra are missing, simulating: ' + str(to_simulate))
        from .simulators.fida.fida_simulator import fida_spectra
        fida_spectra(to_simulate, omega=self.omega, linewidth=self.linewidth,
                     save_dir=os.path.join(path_basis,'basis_files'))
        self._load_fida(path_basis, second_call=True)

  def _load_pygamma(self, path_basis=os.path.join('data', 'basis', 'pygamma'), second_call=False):
    # Constants, synchronise with pygamma_simulator (passed as arguments, but defaults hardcoded
    # in pygamma simulator as well)
    npts = 4096
    adc_dt = 4e-4
    for metabolite_name in self.metabolites:
      specs = Spectrum.load_pygamma(path_basis, metabolite_name,
                                    self.pulse_sequence, self.omega,
                                    self.linewidth, npts, adc_dt)
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
        self.spectra[m]['difference'].set_adc(np.zeros_like(self.spectra[m]['difference'].raw_adc))

  def _add_missing_spectra(self):
    if self.pulse_sequence == 'megapress':
      for m in self.spectra.keys():
        if 'edit_on' in self.spectra[m] and 'edit_off' in self.spectra[m] and 'difference' in self.spectra[m]:
          pass
        elif 'edit_on' in self.spectra[m] and 'edit_off' in self.spectra[m]:
          diff = Spectrum(self.spectra[m]['edit_on'].id+":ON_-_OFF:"+self.spectra[m]['edit_off'].id,
                          source=self.spectra[m]['edit_off'].source,
                          metabolites=self.spectra[m]['edit_off'].metabolites,
                          pulse_sequence=self.spectra[m]['edit_off'].pulse_sequence,
                          acquisition="difference",
                          omega=self.spectra[m]['edit_off'].omega,
                          linewidth=self.spectra[m]['edit_off'].linewidth,
                          dt=self.spectra[m]['edit_off'].dt,
                          center_ppm=self.spectra[m]['edit_off'].center_ppm,
                          filter_fft=self.spectra[m]['edit_off'].filter_fft,
                          remove_water_peak=self.spectra[m]['edit_off'].remove_water_peak,
                          scale=1.0)
          diff.adc_noise_mu = self.spectra[m]['edit_off'].adc_noise_mu
          diff.adc_noise_sigma = self.spectra[m]['edit_off'].adc_noise_sigma
          diff.set_adc(self.spectra[m]['edit_on'].adc(pad=False) - self.spectra[m]['edit_off'].adc(pad=False))
          self.spectra[m]['difference'] = diff
        elif 'edit_off' in self.spectra[m] and 'difference' in self.spectra[m]:
          eon = Spectrum(self.spectra[m]['difference'].id+":DIFF_+_OFF:"+self.spectra[m]['edit_off'].id,
                         source=self.spectra[m]['edit_off'].source,
                         metabolites=self.spectra[m]['edit_off'].metabolites,
                         pulse_sequence=self.spectra[m]['edit_off'].pulse_sequence,
                         acquisition="edit_on",
                         omega=self.spectra[m]['edit_off'].omega,
                         linewidth=self.spectra[m]['edit_off'].linewidth,
                         dt=self.spectra[m]['edit_off'].dt,
                         center_ppm=self.spectra[m]['edit_off'].center_ppm,
                         filter_fft=self.spectra[m]['edit_off'].filter_fft,
                         remove_water_peak=self.spectra[m]['edit_off'].remove_water_peak,
                         scale=1.0)
          eon.adc_noise_mu = self.spectra[m]['edit_off'].adc_noise_mu
          eon.adc_noise_sigma = self.spectra[m]['edit_off'].adc_noise_sigma
          eon.set_adc(self.spectra[m]['difference'].adc(pad=False) + self.spectra[m]['edit_off'].adc(pad=False))
          self.spectra[m]['edit_on'] = eon
        else:
          raise Exception(f"Incomplete megapress spectrum for {m}")

  def _normalise(self):
    # All spectra are normalised against the maximum absolute adc signal in the basis set.
    # There are a number of reasons, but it means that the noise added to the ADC has the same mu and sigma values.
    global_max = 0.0
    for m in self.spectra.keys():
      for a in self.spectra[m].keys():
        global_max = np.max([global_max, np.max(np.abs(self.spectra[m][a].raw_adc))])
    for m in self.spectra.keys():
      for a in self.spectra[m].keys():
        self.spectra[m][a].scale = 1.0/global_max

  def _correct_b0(self):
    if self.source in ['fid-a', 'lcmodel', 'pygamma']:
      # There's not going to be an individual shift per metabolite...
      # so we calibrate the entire set against Cr or NAA
      b0_shift = []
      for m in self.spectra.keys():
        for a in self.spectra[m].keys():
          shift = self.spectra[m][a].correct_b0()
          if shift is not None:
            b0_shift.append(shift)
      if b0_shift is None:
        raise Exception("B0 correction for basis failed")
      # Shift all by mean
      b0_shift = np.mean(b0_shift)
      for m in self.spectra.keys():
        for a in self.spectra[m].keys():
          self.spectra[m][a].correct_b0(ppm_shift=b0_shift)
    else:
      raise Exception('Unrecognised source for B0 correction routine')

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
    shifts = []
    for a in self.acquisitions:
      adc = None
      lw = None
      dt = None
      center_ppm = None
      ppm_shift = None
      for m in self.spectra.keys():
        if adc is None:
          adc = self.spectra[m][a].adc(pad=False) * con[m]
          lw = self.spectra[m][a].linewidth
          dt = self.spectra[m][a].dt
          center_ppm = self.spectra[m][a].center_ppm
          ppm_shift = self.spectra[m][a].b0_ppm_shift
        else:
          a_adc = self.spectra[m][a].adc(pad=False) * con[m]
          al = a_adc.shape[0]
          adcl = adc.shape[0]
          if al > adcl:
            adc = np.append(adc, np.zeros(al-adcl)) + a_adc
          elif al < adcl:
            adc += np.append(a_adc, np.zeros(adcl-al))
          else:
            adc += a_adc
          lw += self.spectra[m][a].linewidth
          if np.abs(dt - self.spectra[m][a].dt) > 1e-8:
            raise Exception("Cannot combine spectra with different dt")
          if np.abs(center_ppm - self.spectra[m][a].center_ppm) > 1e-8:
            raise Exception("Cannot combine spectra with different center_ppm")
          if np.abs(ppm_shift - self.spectra[m][a].b0_ppm_shift) > 1e-8:
            raise Exception("Cannot combine spectra with different b0_ppm_shift")
      lw = lw / len(self.spectra.keys()) # Linewidth should be identical, but just in case
      spectra[a] = Spectrum(id=id,
                            source=self.spectra[self.metabolites[0]][a].source,
                            metabolites=self.metabolites,
                            pulse_sequence=self.pulse_sequence,
                            acquisition=a,
                            omega=self.omega,
                            linewidth=lw,
                            dt=self.spectra[self.metabolites[0]][a].dt, # should be identical
                            center_ppm=self.spectra[self.metabolites[0]][a].center_ppm, # should be identical
                            raw_adc=adc)
      # B0 correction
      shift = spectra[a].correct_b0()
      if shift is not None:
        shifts.append(shift)
    # Shift all by mean b0 correction, if we have one
    if len(shifts) > 0:
      mean_shift = np.mean(shifts)
      for a in self.acquisitions:
        spectra[a].correct_b0(mean_shift)
    # Sanity check for difference
    if 'edit_on' in self.acquisitions and 'edit_off' in self.acquisitions and 'difference' in self.acquisitions:
      err = np.max(np.abs(spectra['edit_on'].raw_adc - spectra['edit_off'].raw_adc - spectra['difference'].raw_adc))
      if err > 1e-8:
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
        ml = convert_names([m], shorten=True)[0]
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
