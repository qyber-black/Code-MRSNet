# mrsnet/dataset.py - MRSNet - spectra dataset
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2019, Max Chandler, PhD student at Cardiff University
# Copyright (C) 2020-2021, Frank C Langbein <frank@langbein.org>, Cardiff University

import os
import math
import sobol_seq
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations

class Dataset(object):
  # A dataset is a collection of spectra for training, tesing or predicting.
  # The dataset object contains methods for how to handle and export the data
  # from the spectra objects.

  def __init__(self, name, high_ppm=-4.5, low_ppm=-1, n_fft_pts=2048):
    self.name = name
    self.metabolites = None
    self.spectra = []
    self.concentrations = []
    self.high_ppm = high_ppm
    self.low_ppm = low_ppm
    self.n_fft_pts = n_fft_pts
    self.pulse_sequence = None

  def load_dicoms(self, folder, concentrations=None, metabolites=[]):
    from .spectrum import Spectrum
    specs = {}
    concs = {}
    for dir, subdirs, files in os.walk(folder):
      for file in sorted(files):
        if file[-4:].lower() == '.ima':
          s, c = Spectrum.load_dicom(os.path.join(dir,file), concentrations, metabolites)
          if s.id not in specs:
            specs[s.id] = {}
          specs[s.id][s.acquisition] = s
          concs[s.id] = c
    for id in sorted(specs.keys()):
      b0_shift = []
      # FIXME: b0 correction as average over all acquisitions and peaks?
      for a in specs[id].keys():
        if 'NAA' in specs[id][a].metabolites or 'Cr' in self.specs[id][a].metabolites:
          shift = specs[id][a].correct_b0()
          if shift is not None:
            b0_shift.append(shift)
      if len(b0_shift) == 0:
        raise Exception("B0 correction for failed")
      b0_shift = np.mean(np.array(b0_shift, dtype=np.float64))
      for a in specs[id].keys():
        specs[id][a].correct_b0(ppm_shift=b0_shift)
      self.spectra.append(specs[id])
      self.concentrations.append(concs[id])
    return self

  def generate_spectra(self, basis, num, samplers, noise_p, noise_mu, noise_sigma, verbose):
    # Generate the dataset from the basis (assuming metabolites taken from those in the basis).
    # Does not add noise, but only generates clean combined ADC signal.
    if num <= 0:
      raise Exception('n_samples must be greater than 0, not %d!' % num)
    if self.metabolites == None:
      self.metabolites = basis.metabolites
    else:
      for m in basis.metabolites:
        if m not in self.metabolites:
          raise Exception("Basis metabolite not in dataset: %s" % m)
      for m in self.metabolites:
        if m not in basis.metabolites:
          raise Exception("Dataset metabolite not in basis: %s" % m)

    if self.pulse_sequence == None:
      self.pulse_sequence = basis.pulse_sequence
    elif self.pulse_sequence != basis.pulse_sequence:
      raise Exception("Dataset pulse sequence does not match basis pulse sequence")

    for s in basis.spectra.keys():
      for a in basis.spectra[s].keys():
        nu = basis.spectra[s][a].nu()
        if np.min(nu) > self.high_ppm:
          raise Exception('Spectra do not reach the required max frequency axis (%.2f) for export: %.2f' % (np.min(spectra.nu()), self.high_ppm))
        elif np.max(nu) < self.low_ppm:
          raise Exception('Spectra do not reach the required min frequency axis (%.2f) for export: %.2f' % (np.max(spectra.nu()), self.low_ppm))

    n0 = num // len(samplers)
    n1 = num % len(samplers)
    n_metabolites = len(self.metabolites)
    all_concentrations = np.empty((0,n_metabolites))
    for sampler in samplers:
      n = n0 + (1 if n1 > 0 else 0)
      n1 -= 1
      if verbose > 0:
        print("Generating %d concentrations with %s sampling" % (n,sampler))
      if sampler == 'random':
        # Random uniform concentrations
        concentrations = np.random.ranf((n, n_metabolites))
      elif sampler == 'dirichlet':
        # Dirichlet sampling, equal weight for all metablites
        concentrations = np.random.default_rng().dirichlet([1] * n_metabolites, n)
      elif sampler == 'sobol':
        # Sobol sampling
        skip = math.floor(math.log(n*n_metabolites,2)) # Skip broken in sobol package!
        concentrations = sobol_seq.i4_sobol_generate(n_metabolites, n+skip)[skip:,:]
      elif sampler[-6:] == '-zeros':
        # Select all zero concentration possibilities, equal weight for all remaining metablites according to sampling method
        concentrations = np.zeros((n, n_metabolites))
        groups = []
        groups_n = []
        n_per_combs = n // n_metabolites
        n_total = 0
        for n_excited in range(1,n_metabolites + 1):
          combs = list(combinations(list(range(0,n_metabolites)), n_excited))
          n_per_group = n_per_combs // len(combs)
          if verbose > 0:
            print("  For %d excited: %d samples per %d combinations" % (n_excited, n_per_group, len(combs)))
          if n_per_group < 1:
            raise Exception('Insufficient samples for *-zeros with %d groups' % len(groups))
          for comb in combs:
            groups.append(comb)
            groups_n.append(n_per_group)
            n_total += n_per_group
        groups.reverse()
        groups_n.reverse()
        n_remain = n - n_total
        idx = 0
        if sampler == 'sobol-zeros':
          skip = math.floor(math.log(n*n_metabolites,2))
        for g, n_g in zip(groups, groups_n):
          if n_remain > 0:
            n_g += 1
            n_remain -= 1
          if sampler == 'random-zeros':
            # Random uniform concentrations
            concentrations[idx:idx+n_g,g] = np.random.ranf((n_g, len(g)))
          elif sampler == 'dirichlet-zeros':
            # Dirichlet sampling, equal weight for all metablites
            concentrations[idx:idx+n_g,g] = np.random.default_rng().dirichlet([1]*len(g), n_g)
          elif sampler == 'sobol-zeros':
            # Sobol sampling
            concentrations[idx:idx+n_g,g] = sobol_seq.i4_sobol_generate(len(g), n_g+skip)[skip:,:]
            skip += n_g # Get differnt samples for the groups
          else:
            raise Exception('Unknown concentration generation method: ' + sampler)
          idx += n_g
      elif sampler[-4:] == '-one':
        # Set one concentration to one
        concentrations = np.zeros((n, n_metabolites))
        combs = list(combinations(list(range(0,n_metabolites)), n_metabolites-1))
        n_per_group = n // len(combs)
        if verbose > 0:
          print("  %d samples for %d combinations with one 1.0" % (n_per_group, len(combs)))
        if n_per_group < 1:
          raise Exception('Insufficient samples for *-one with %d combinations' % len(combs))
        n_total = 0
        groups = []
        groups_n = []
        for comb in combs:
          groups.append(comb)
          groups_n.append(n_per_group)
          n_total += n_per_group
        n_remain = n - n_total
        idx = 0
        if sampler == 'sobol-one':
          skip = math.floor(math.log(n*n_metabolites,2))
        for g, n_g in zip(groups, groups_n):
          if n_remain > 0:
            n_g += 1
            n_remain -= 1
          if sampler == 'random-one':
            # Random uniform concentrations
            concentrations[idx:idx+n_g,g] = np.random.ranf((n_g, len(g)))
          elif sampler == 'dirichlet-one':
            # Dirichlet sampling, equal weight for all metablites
            concentrations[idx:idx+n_g,g] = np.random.default_rng().dirichlet([1]*len(g), n_g)
          elif sampler == 'sobol-ones':
            # Sobol sampling
            concentrations[idx:idx+n_g,g] = sobol_seq.i4_sobol_generate(len(g), n_g+skip)[skip:,:]
            skip += n_g # Get differnt samples for the groups
          else:
            raise Exception('Unknown concentration generation method: ' + sampler)
          for one in range(0,n_metabolites):
            if one not in g:
              concentrations[idx:idx+n_g,one] = 1.0
          idx += n_g
      else:
        raise Exception('Unknown concentration generation method: ' + sampler)
      all_concentrations = np.concatenate((all_concentrations,concentrations),axis=0)

    if verbose > 0:
      print("Combining basis spectra")
      print("  Noise p: %f  max. mu: %f  max. sigma:%f " % (noise_p,noise_mu,noise_sigma))
    n_add = np.random.uniform(0.0,1.0,num)
    n_mu = np.random.uniform(0.0,noise_mu,num)
    n_sigma = np.random.uniform(0.0,noise_sigma,num)
    n_cnt = 0
    for count in range(num):
      s,c = basis.combine(all_concentrations[count],str(count))
      if n_add[count] <= noise_p:
        n_cnt += 1
        for a in s:
          s[a].add_noise(mu=n_mu[count], sigma=n_sigma[count])
      self.spectra.append(s)
      self.concentrations.append(c)
      # s['edit_off'].plot_spectrum(c)
      # s['edit_on'].plot_spectrum(c)
      # s['difference'].plot_spectrum(c)
    if verbose > 0:
      print("  Spectra with noise: %d" % n_cnt)

  def save(self):
    if not os.path.exists(self.name):
      os.makedirs(self.name)
    joblib.dump(self, os.path.join(self.name, "data"))

  @staticmethod
  def load(name):
    return joblib.load(os.path.join(name, "data"))

  def plot_concentrations(self, norm='none'):
    if len(self.concentrations) > 0:
      n_spec = len(self.spectra)
      n_hst = len(self.metabolites)
      n_col = int(np.floor(np.sqrt(n_hst)))
      n_row = n_col
      while n_row * n_col < n_hst:
        n_row += 1
      fig, axes = plt.subplots(n_row, n_col,  sharex=True, sharey=True,
                               figsize=(25.6, 14.4)) # 1440p@100dpi
      axes = axes.flatten()
      plt.suptitle('Concentrations %s of %s; %d spectra; %f - %f ppm @ %d pts'
                   % ('' if norm == 'none' else ("("+norm+" normalised)"),
                      self.name, len(self.spectra), self.low_ppm,
                      self.high_ppm, self.n_fft_pts))
      cs = np.ndarray((n_spec,n_hst),dtype=np.float64)
      k = 0
      for c in self.concentrations:
        cr = np.array([c[m] for m in self.metabolites], dtype=np.float64)
        if norm == 'sum':
          cs[k,:] = cr / np.sum(cr)
        elif norm == 'max':
          cs[k,:] = cr / np.max(cr)
        else:
          cs[k,:] = cr
        k += 1
      for m in range(0,n_hst):
        axes[m].hist(cs[:,m], bins=int(np.max([25, np.ceil(n_spec/100)])))
        axes[m].set_title(self.metabolites[m])
        axes[m].set_xlim([0, 1])
      return fig
    return None

  def export(self, metabolites=None, norm='sum', acquisitions=['edit_off','difference'],
             datatype='magnitude', verbose=0):
    if metabolites is None:
      metabolites = self.metabolites

    if len(self.spectra) > 0:
      if verbose > 0:
        print("Converting input spectra to tensor")
      d_inp = joblib.Parallel(n_jobs=-1, prefer="threads")(joblib.delayed(Dataset._export_spectra)(s,
                    acquisitions, datatype, self.high_ppm, self.low_ppm, self.n_fft_pts)
                for s in tqdm(self.spectra, disable=(verbose<1)))
      d_inp = np.array(d_inp, dtype=np.float64)
      if verbose > 0:
        print("  Shape: " + str(d_inp.shape) + " - [spectrum, acquisition, datatype, fft_value]")
    else:
      d_inp = np.ndarray()
    if len(self.concentrations) > 0:
      if verbose > 0:
        print("Converting output concentrations to tensor")
      d_out = joblib.Parallel(n_jobs=-1, prefer="threads")(joblib.delayed(Dataset._export_concentrations)(c,
                    metabolites, norm)
                for c in tqdm(self.concentrations, disable=(verbose<1)))
      d_out = np.array(d_out, dtype=np.float64)
      if verbose > 0:
        print("  Shape: " + str(d_out.shape) + " - [spectrum, metabolite_concentration]")
    else:
      d_out = np.ndarray()

    if d_out.shape[0] != d_inp.shape[0] or d_out.shape[1] != len(metabolites) or d_inp.shape[1] != len(acquisitions) or d_inp.shape[2] != len(datatype) or d_inp.shape[3] != self.n_fft_pts:
      raise Exception("Unexpected input/output tensor shape(s)")

    return d_inp, d_out

  @staticmethod
  def _export_spectra(s, acquisitions, datatypes, high_ppm, low_ppm, n_fft_pts,
                      mean_center=True,normalise=True):
    inp = np.ndarray((len(acquisitions),len(datatypes),n_fft_pts), dtype=np.float64)
    mean = []
    max = []
    fft = np.ndarray((len(acquisitions),n_fft_pts),dtype=np.complex64)
    a_idx = 0
    for a in acquisitions:
      a, _ = s[a].rescale_fft(high_ppm=high_ppm, low_ppm=low_ppm, npts=n_fft_pts)
      fft[a_idx,:] = a
      a_idx += 1
    if mean_center or normalise:
      # FIXME: different normalisation? any phase normalisation?
      m = np.abs(fft)
      p = np.angle(fft)
      if mean_center:
        m -= np.mean(m)
      if normalise:
        m /= np.max(m)
      fft = np.multiply(p, np.exp(1j*m))
    for a_idx in range(0,fft.shape[0]):
      d_idx = 0
      for d in datatypes:
        if d == 'real':
          inp[a_idx,d_idx,:] = np.real(fft[a_idx,:])
        elif d == 'imaginary':
          inp[a_idx,d_idx,:] = np.imag(fft[a_idx,:])
        elif d == 'magnitude':
          inp[a_idx,d_idx,:] = np.abs(fft[a_idx,:])
        else:
          raise Exception("Unknown datatype %s" % d)
        d_idx += 1
    return inp

  @staticmethod
  def _export_concentrations(c, metabolites, norm):
    m_idx = 0
    out = np.array([c[m] for m in metabolites], dtype=np.float64)
    if norm == 'max':
      out /= np.max(out)
    elif norm == 'sum':
      out /= np.sum(out)
    elif norm != 'none':
      raise Exception("Unknown norm %s" % norm)
    return out
