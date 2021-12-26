# mrsnet/dataset.py - MRSNet - spectra dataset
#
# SPDX-FileCopyrightText: Copyright (C) 2019 Max Chandler, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2020-2021 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import math
import sobol_seq
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations

from .spectrum import Spectrum
from .grid import Grid
from .cfg import Cfg

class Dataset(object):
  # A dataset is a collection of spectra for training, tesing or predicting.

  def __init__(self, name, high_ppm=-4.5, low_ppm=-1, n_fft_pts=2048): # FIXME: high/low ppm and pts only for export
    self.name = name
    self.metabolites = None
    self.spectra = []
    self.concentrations = []
    self.high_ppm = high_ppm
    self.low_ppm = low_ppm
    self.n_fft_pts = n_fft_pts
    self.pulse_sequence = None
    self.noise_added = False

  def load_dicoms(self, folder, concentrations=None, metabolites=[]):
    from .spectrum import Spectrum
    specs = {}
    concs = {}
    concs_ok = True
    if self.metabolites == None:
      self.metabolites = []
    for dir, subdirs, files in os.walk(folder):
      for file in sorted(files):
        if file[-4:].lower() == '.ima':
          s, c = Spectrum.load_dicom(os.path.join(dir,file), concentrations, metabolites)
          for m in s.metabolites:
            if m not in self.metabolites:
              self.metabolites.append(m)
          if s.id not in specs:
            specs[s.id] = {}
          specs[s.id][s.acquisition] = s
          concs[s.id] = c
          if len(c) == 0:
            concs_ok = False
    self.metabolites.sort()
    for id in sorted(specs.keys()):
      Spectrum.correct_b0_multi(specs[id])
      self.spectra.append(specs[id])
      if concs_ok:
        self.concentrations.append(concs[id])
    return self

  def generate_spectra(self, basis, num, samplers, verbose):
    # Generate the dataset from the basis (assuming metabolites taken from those in the basis).
    # Does not add noise, but only generates clean combined ADC signal.
    if num <= 0:
      raise Exception(f"n_samples must be greater than 0, not {num}!")
    if self.metabolites == None:
      self.metabolites = basis.metabolites
    else:
      for m in basis.metabolites:
        if m not in self.metabolites:
          raise Exception(f"Basis metabolite not in dataset: {m}")
      for m in self.metabolites:
        if m not in basis.metabolites:
          raise Exception(f"Dataset metabolite not in basis: {m}")

    if self.pulse_sequence == None:
      self.pulse_sequence = basis.pulse_sequence
    elif self.pulse_sequence != basis.pulse_sequence:
      raise Exception("Dataset pulse sequence does not match basis pulse sequence")

    for s in basis.spectra.keys():
      for a in basis.spectra[s].keys():
        _, nu = basis.spectra[s][a].get_f()
        if np.min(nu) > self.high_ppm:
          raise Exception(f"Spectra do not reach the required max frequency axis ({np.min(spectra.nu()):.2f}) for export: {self.high_ppm:.2f}")
        elif np.max(nu) < self.low_ppm:
          raise Exception(f"Spectra do not reach the required min frequency axis ({np.max(spectra.nu()):.2f}) for export: {self.low_ppm:.2f}")

    n0 = num // len(samplers)
    n1 = num % len(samplers)
    n_metabolites = len(self.metabolites)
    all_concentrations = np.empty((0,n_metabolites))
    for sampler in samplers:
      n = n0 + (1 if n1 > 0 else 0)
      n1 -= 1
      if verbose > 0:
        print(f"Generating {n} concentrations with {sampler} sampling")
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
            print(f"  For {n_excited} excited: {n_per_group} samples per {len(combs)} combinations")
          if n_per_group < 1:
            raise Exception(f"Insufficient samples for *-zeros with {len(groups)} groups")
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
            raise Exception(f"Unknown concentration generation method: {sampler}")
          idx += n_g
      elif sampler[-4:] == '-one':
        # Set one concentration to one
        concentrations = np.zeros((n, n_metabolites))
        combs = list(combinations(list(range(0,n_metabolites)), n_metabolites-1))
        n_per_group = n // len(combs)
        if verbose > 0:
          print(f"  {n_per_group} samples for {len(combs)} combinations with one 1.0")
        if n_per_group < 1:
          raise Exception(f"Insufficient samples for *-one with {len(combs)} combinations")
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
    n_cnt = 0
    for count in range(num):
      s,c = basis.combine(all_concentrations[count],str(count))
      self.spectra.append(s)
      self.concentrations.append(c)

  def add_noise(self, noise_p, noise_mu, noise_sigma, verbose): # FIXME: needs to use adc_gaussian!
    # Add noise to all spectra
    if self.noise_added:
      raise Exception("Noise added twice to dataset")
    if noise_p > 0.0:
      if verbose > 0:
        print(f"Adding noise Normal(mu={noise_mu},sigma={noise_sigma}) with probability {noise_p} to time signal")
      self.noise_added = True
      num = len(self.spectra)
      n_add = np.random.uniform(0.0,1.0,num)
      n_mu = np.random.uniform(0.0,noise_mu,num)
      n_sigma = np.random.uniform(0.0,noise_sigma,num)
      n_cnt = 0
      for idx in range(num):
        if n_add[idx] <= noise_p:
          n_cnt += 1
          # Add noise
          for a in self.spectra[idx]:
            if a != 'difference':
              self.spectra[idx][a].add_noise_adc_normal(mu=n_mu[idx], sigma=n_sigma[idx])
          if 'difference' in self.spectra[idx]:
            # Add difference of noisy spectra
            if 'edit_off' not in self.spectra[idx] or 'edit_on' not in self.spectra[idx]:
              raise Exception("Difference spectrum without edit_off or edit_on")
            self.spectra[idx]['difference'] = Spectrum.comb(1.0,self.spectra[idx]['edit_on'],-1.0,self.spectra[idx]['edit_off'],
                                                            self.spectra[idx]['edit_on'].id+"_+_"+self.spectra[idx]['edit_off'].id,
                                                            "difference")
          # B0 correction per spectrum
          Spectrum.correct_b0_multi(self.spectra[idx])
      if verbose > 1:
        print(f"  Added noise to {n_cnt} of {num} spectra")

  def save(self, path): # FIXME: save json/npy
    from .getfolder import get_folder
    for s in self.spectra:
      for a in s:
        s[a].fft_cache = None
    folder = get_folder(os.path.join(path,self.name),str(len(self.spectra))+"-%s")
    joblib.dump(self, os.path.join(folder, "spectra.joblib"))
    return folder

  @staticmethod
  def load(folder): # FIXME: load json/npy
    return joblib.load(os.path.join(folder, "spectra.joblib"))

  def plot_concentrations(self, norm='none'):
    if len(self.concentrations) > 0:
      n_spec = len(self.spectra)
      n_hst = len(self.metabolites)
      n_col = int(np.floor(np.sqrt(n_hst)))
      n_row = n_col
      while n_row * n_col < n_hst:
        n_row += 1
      fig, axes = plt.subplots(n_row, n_col,  sharex=True, sharey=True)
      axes = axes.flatten()
      norm_str = "" if norm == 'none' else ("("+norm+" normalised)"),
      plt.suptitle(f"Concentrations {norm_str} of {self.name}; {len(self.spectra)} spectra; {self.low_ppm} - {self.high_ppm} ppm @ {self.n_fft_pts} pts")
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
             datatype='magnitude', normalise=True, verbose=0):
    if metabolites is None:
      metabolites = self.metabolites

    if len(self.spectra) > 0:
      if verbose > 0:
        print("Converting spectra to tensor")
      d_inp = joblib.Parallel(n_jobs=-1, prefer="threads")(joblib.delayed(Dataset._export_spectra)(s,
                    acquisitions, datatype, self.high_ppm, self.low_ppm, self.n_fft_pts, normalise)
                for s in tqdm(self.spectra, disable=(verbose<1)))
      d_inp = np.array(d_inp, dtype=np.float64)
      if verbose > 0:
        print("  Shape: " + str(d_inp.shape) + " - [spectrum, acquisition, datatype, frequency]")
    else:
      d_inp = np.ndarray((0,0))
    if len(self.concentrations) > 0:
      if verbose > 0:
        print("Converting concentrations to tensor")
      d_out = joblib.Parallel(n_jobs=-1, prefer="threads")(joblib.delayed(Dataset._export_concentrations)(c,
                    metabolites, norm)
                for c in tqdm(self.concentrations, disable=(verbose<1)))
      d_out = np.array(d_out, dtype=np.float64)
      if verbose > 0:
        print("  Shape: " + str(d_out.shape) + " - [spectrum, metabolite_concentration]")
    else:
      d_out = np.ndarray((0,0))

    if np.sum(d_out.shape) > 0:
      if d_out.shape[0] != d_inp.shape[0] or d_out.shape[1] != len(metabolites) or d_inp.shape[1] != len(acquisitions) or d_inp.shape[2] != len(datatype) or d_inp.shape[3] != self.n_fft_pts:
        raise Exception("Unexpected input/output tensor shape(s)")

    if Cfg.dev("check_dataset_export"):
      self._check_export(d_inp,d_out,metabolites, norm, acquisitions, datatype, normalise)

    return d_inp, d_out

  def _check_export(self,d_inp,d_out,metabolites,norm,acquisitions,datatype,normalise): # FIXME: test this
    # Test mrsnet.dataset.export
    from colorama import Fore, Style
    print("# Testing mrsnet.dataset.export")
    if len(self.spectra) > 0:
      # Check if spectra tensor d_inp lists acquisitions and datatypes in
      # order of spectra and order given by acquisitions and datatype
      res = True
      for s in range(len(self.spectra)):
        print(f"## Spectra tensor export test: {s: 10d}", end='\r', flush=True)
        nl="\n"
        fft = np.ndarray((len(acquisitions),self.n_fft_pts),dtype=np.complex64)
        a_idx = 0
        a_norm = None
        for a in acquisitions:
          if a == "edit_off":
            a_norm = a_idx
          fft[a_idx,:], _ = self.spectra[s][a].rescale_fft(high_ppm=self.high_ppm,
                                                           low_ppm=self.low_ppm,
                                                           npts=self.n_fft_pts)
          a_idx += 1
        if normalise:
          m = np.abs(fft)
          p = np.angle(fft)
          if a_norm == None:
            no = np.max(m)
          else:
            no = np.max(m[a_norm,:])
          m /= no
          new_fft = np.multiply(m, np.exp(1j*p))
          diff = np.max(np.abs(np.abs(new_fft)*no - np.abs(fft)))
          if diff > 1e-4: # Magnitude errors can be in the 1e-5 range
            print(f"{nl}- Max. FFT center/normalise magnitude error: {diff}")
            nl=""
          diff = np.max(np.abs(np.angle(new_fft) - np.angle(fft)))
          if diff > 1e-6: # Phase errors can be in the 1e-7 range
            print(f"{nl}- Max. FFT center/normalise phase error: {diff}")
            nl=""
          if Cfg.dev('flag_plots'):
            for a_idx in range(len(acquisitions)):
              figure, axes = plt.subplots(2, 3)
              axes[0,0].plot(np.abs(fft[a_idx,:]))
              axes[0,0].set_title(f"FFT-Magnitude-{acquisitions[a_idx]}")
              axes[0,1].plot(np.abs(new_fft[a_idx,:]))
              axes[0,1].set_title(f"NORMALISED_FFT-Magnitude-{acquisitions[a_idx]}")
              axes[0,2].plot(np.abs(new_fft[a_idx,:])*no - np.abs(fft[a_idx,:]))
              axes[0,2].set_title(f"DIFF-Magnitude-{acquisitions[a_idx]}")
              axes[1,0].plot(np.angle(fft[a_idx,:]))
              axes[1,0].set_title(f"FFT-Phase-{acquisitions[a_idx]}")
              axes[1,1].plot(np.angle(new_fft[a_idx,:]))
              axes[1,1].set_title(f"NORMALISED-FFT-Phase-{acquisitions[a_idx]}")
              axes[1,2].plot(np.angle(new_fft[a_idx,:]) - np.angle(fft[a_idx,:]))
              axes[1,2].set_title(f"DIFF-Phase-{acquisitions[a_idx]}")
              plt.show()
              plt.close()
          fft = new_fft
        for a in range(len(acquisitions)):
          for d in range(len(datatype)):
            if datatype[d] == 'real':
              inp = np.real(fft[a,:])
            elif datatype[d] == 'imaginary':
              inp = np.imag(fft[a,:])
            elif datatype[d] == 'magnitude':
              inp = np.abs(fft[a,:])
            elif datatype[d] == 'phase':
              inp = np.angle(fft[a,:])
              if normalise:
                inp = (inp/np.pi+1.0)/2.0
            diff = np.max(np.abs(inp - d_inp[s,a,d,:]))
            if diff > 1e-6: # Careful with narrower error margin, due to fft/exp.
              print(f"{nl}- Spectrum {s}-{acquisitions[a]}-{datatype[d]} export error {diff}")
              nl=""
              res = False
      if res:
        print(f"## Spectra tensor export test: {Fore.GREEN}OK        {Style.RESET_ALL}")
      else:
        print(f"## Spectra tensor export test: {Fore.RED}FAILED    {Style.RESET_ALL}")
    if len(self.concentrations) > 0:
      # Check if concentration tensor d_out lists concentrations in order of
      # spectra and order given by metabolite
      res = True
      for c in range(len(self.concentrations)):
        print(f"## Concentration tensor export test: {c: 10d}", end='\r', flush=True)
        if norm == 'max':
          nf = np.max([self.concentrations[c][m] for m in metabolites])
        elif norm == 'sum':
          nf = np.sum([self.concentrations[c][m] for m in metabolites])
        else:
          nf = 1.0
        diff = np.max(np.abs([ self.concentrations[c][metabolites[m]] - d_out[c,m]*nf \
                               for m in range(len(metabolites)) ]))
        if diff > 1e-10:
          print(f"\n- Max. concentration error: {diff}")
          res = False
      if res:
        print(f"## Concentration tensor export test: {Fore.GREEN}OK        {Style.RESET_ALL}")
      else:
        print(f"## Concentration tensor export test: {Fore.RED}FAILED    {Style.RESET_ALL}")

  @staticmethod
  def _export_spectra(s, acquisitions, datatypes, high_ppm, low_ppm, n_fft_pts, normalise):
    inp = np.ndarray((len(acquisitions),len(datatypes),n_fft_pts), dtype=np.float64)
    fft = np.ndarray((len(acquisitions),n_fft_pts),dtype=np.complex64)
    a_idx = 0
    a_norm = None
    for a in acquisitions:
      if a == "edit_off":
        a_norm = a_idx
      fft[a_idx,:], _ = s[a].rescale_fft(high_ppm=high_ppm, low_ppm=low_ppm, npts=n_fft_pts)
      a_idx += 1
    if normalise:
      m = np.abs(fft)
      p = np.angle(fft)
      if a_norm == None:
        m /= np.max(m)
      else:
        m /= np.max(m[a_norm,:])
      fft = np.multiply(m, np.exp(1j*p))
    for a_idx in range(0,fft.shape[0]):
      d_idx = 0
      for d in datatypes:
        if d == 'real':
          inp[a_idx,d_idx,:] = np.real(fft[a_idx,:])
        elif d == 'imaginary':
          inp[a_idx,d_idx,:] = np.imag(fft[a_idx,:])
        elif d == 'magnitude':
          inp[a_idx,d_idx,:] = np.abs(fft[a_idx,:])
        elif d == 'phase':
          inp[a_idx,d_idx,:] = np.angle(fft[a_idx,:])
          if normalise:
            inp[a_idx,d_idx,:] = (inp[a_idx,d_idx,:]/np.pi+1.0)/2.0 # Normalise to (0..1]
        else:
          raise Exception(f"Unknown datatype {d}")
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
      raise Exception(f"Unknown norm {norm}")
    return out

Collections = {
  'single_source-sampler-noise': Grid({
    'metabolites': [['Cr','GABA','Gln','Glu','NAA']],
    'source': ['lcmodel','fid-a','pygamma'],
    'manufacturer': ['siemens'],
    'omega': [123.23],
    'linewidth': [1.0],
    'pulse_sequence': ['megapress'],
    'num': [10000],
    'sample': ['random','sobol','dirichlet'],
    'noise_p': [1.0],
    'noise_sigma': [0.05,0.1],
    'noise_mu': [0.0]
  }),
  'multi_source-linewidths': Grid({
    'metabolites': [['Cr','GABA','Gln','Glu','NAA']],
    'source': [['fid-a','lcmodel','pygamma'],['fid-a','pygamma']],
    'manufacturer': ['siemens'],
    'omega': [123.23],
    'linewidth': [1.0,[0.75,1.0,1.25]],
    'pulse_sequence': ['megapress'],
    'num': [30000],
    'sample': ['sobol'],
    'noise_p': [1.0],
    'noise_sigma': [0.1],
    'noise_mu': [0.0]
  })
}
