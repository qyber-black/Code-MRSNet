# mrsnet/dataset.py - MRSNet - spectra dataset
#
# SPDX-FileCopyrightText: Copyright (C) 2019 Max Chandler, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2020-2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2022-2024 Zien Ma, PhD student at Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Dataset management for MRSNet.

This module provides the Dataset class for managing collections of spectra
and their corresponding concentrations, including loading from files,
generating synthetic data, and exporting for model training.
"""

import math
import os
from itertools import combinations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import sobol_seq
from tqdm import tqdm

from mrsnet.cfg import Cfg
from mrsnet.spectrum import Spectrum


class Dataset:
  """Collection of spectra and concentrations for training, testing, or prediction.

  This class manages datasets containing multiple spectra with their corresponding
  metabolite concentrations. It supports loading from files, generating synthetic
  data, and exporting for model training.

  Attributes
  ----------
      name (str): Name of the dataset
      metabolites (list): List of metabolite names in this dataset
      spectra (list): List of spectrum dictionaries (one per sample)
      concentrations (list): List of concentration dictionaries (one per sample)
      pulse_sequence (str): Pulse sequence type for all spectra
      noise_added (bool): Whether noise has been added to the spectra
  """

  def __init__(self, name):
    """Initialize a new dataset.

    Parameters
    ----------
        name (str): Name of the dataset
    """
    self.name = name
    self.metabolites = None
    self.spectra = []
    self.concentrations = []
    self.pulse_sequence = None
    self.noise_added = False

  def load_dicoms(self, folder, concentrations=None, metabolites=[], verbose=0):
    """Load spectra from DICOM or CSV files in a folder.

    Recursively searches a folder for .ima (DICOM) and .csv files and loads
    them as spectra. Applies B0 correction to each set of spectra.

    Parameters
    ----------
        folder (str): Path to folder containing spectrum files
        concentrations (dict, optional): Dictionary mapping spectrum IDs to concentrations.
                                       Defaults to None
        metabolites (list, optional): List of metabolite names to extract. Defaults to []
        verbose (int, optional): Verbosity level. Defaults to 0

    Returns
    -------
        Dataset: Self for method chaining

    Raises
    ------
        RuntimeError: If duplicate spectrum IDs are found
    """
    from mrsnet.spectrum import Spectrum
    specs = {}
    concs = {}
    concs_ok = True
    if self.metabolites is None:
      self.metabolites = []
    for dir, _subdirs, files in os.walk(folder):
      for file in sorted(files):
        if file[-4:].lower() == '.ima':
          s, c = Spectrum.load_dicom(os.path.join(dir,file), concentrations, metabolites, verbose)
        elif file[-4:].lower() == '.csv':
          s, c = Spectrum.load_csv(os.path.join(dir,file), concentrations, metabolites, verbose)
        else:
          s = None
        if s is not None:
          for m in s.metabolites:
            if m not in self.metabolites:
              self.metabolites.append(m)
          if s.id not in specs:
            specs[s.id] = {}
          if s.acquisition in specs[s.id]:
            raise RuntimeError(f"Duplicate spectrum id would overwrite spectra: {s.id}-{s.acquisition}")
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

  def generate_spectra(self, basis, num, samplers, verbose, linewidth_mode=None, basis_pool=None, lw_values=None):
    """Generate synthetic spectra from a basis set.

    Creates a dataset by combining basis spectra with randomly sampled
    concentrations using various sampling strategies.

    Parameters
    ----------
        basis (Basis): Basis object containing metabolite spectra
        num (int): Number of spectra to generate
        samplers (list): List of sampling strategies ('random', 'dirichlet', 'sobol', etc.)
        verbose (int): Verbosity level

    Returns
    -------
        Dataset: Self for method chaining

    Raises
    ------
        RuntimeError: If num <= 0 or metabolite mismatch between dataset and basis
    """
    # Generate the dataset from the basis (assuming metabolites taken from those in the basis).
    if num <= 0:
      raise RuntimeError(f"n_samples must be greater than 0, not {num}!")
    if self.metabolites is None:
      self.metabolites = basis.metabolites
    else:
      for m in basis.metabolites:
        if m not in self.metabolites:
          raise RuntimeError(f"Basis metabolite not in dataset: {m}")
      for m in self.metabolites:
        if m not in basis.metabolites:
          raise RuntimeError(f"Dataset metabolite not in basis: {m}")

    if self.pulse_sequence is None:
      self.pulse_sequence = basis.pulse_sequence
    elif self.pulse_sequence != basis.pulse_sequence:
      raise RuntimeError("Dataset pulse sequence does not match basis pulse sequence")

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
        # Dirichlet sampling, equal weight for all metabolites
        concentrations = np.random.default_rng().dirichlet([1] * n_metabolites, n)
      elif sampler == 'sobol':
        # Sobol sampling
        skip = math.floor(math.log(n*n_metabolites,2)) # Skip broken in sobol package!
        concentrations = sobol_seq.i4_sobol_generate(n_metabolites, n+skip)[skip:,:]
      elif sampler[-6:] == '-zeros':
        # Select all zero concentration possibilities, equal weight for all remaining metabolites according to sampling method
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
            raise RuntimeError(f"Insufficient samples for *-zeros with {len(groups)} groups")
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
        for g, n_g in zip(groups, groups_n, strict=False):
          if n_remain > 0:
            n_g += 1
            n_remain -= 1
          if sampler == 'random-zeros':
            # Random uniform concentrations
            concentrations[idx:idx+n_g,g] = np.random.ranf((n_g, len(g)))
          elif sampler == 'dirichlet-zeros':
            # Dirichlet sampling, equal weight for all metabolites
            concentrations[idx:idx+n_g,g] = np.random.default_rng().dirichlet([1]*len(g), n_g)
          elif sampler == 'sobol-zeros':
            # Sobol sampling
            concentrations[idx:idx+n_g,g] = sobol_seq.i4_sobol_generate(len(g), n_g+skip)[skip:,:]
            skip += n_g # Get different samples for the groups
          else:
            raise RuntimeError(f"Unknown concentration generation method: {sampler}")
          idx += n_g
      elif sampler[-4:] == '-one':
        # Set one concentration to one
        concentrations = np.zeros((n, n_metabolites))
        combs = list(combinations(list(range(0,n_metabolites)), n_metabolites-1))
        n_per_group = n // len(combs)
        if verbose > 0:
          print(f"  {n_per_group} samples for {len(combs)} combinations with one 1.0")
        if n_per_group < 1:
          raise RuntimeError(f"Insufficient samples for *-one with {len(combs)} combinations")
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
        for g, n_g in zip(groups, groups_n, strict=False):
          if n_remain > 0:
            n_g += 1
            n_remain -= 1
          if sampler == 'random-one':
            # Random uniform concentrations
            concentrations[idx:idx+n_g,g] = np.random.ranf((n_g, len(g)))
          elif sampler == 'dirichlet-one':
            # Dirichlet sampling, equal weight for all metabolites
            concentrations[idx:idx+n_g,g] = np.random.default_rng().dirichlet([1]*len(g), n_g)
          elif sampler == 'sobol-one':
            # Sobol sampling
            concentrations[idx:idx+n_g,g] = sobol_seq.i4_sobol_generate(len(g), n_g+skip)[skip:,:]
            skip += n_g # Get different samples for the groups
          else:
            raise RuntimeError('Unknown concentration generation method: ' + sampler)
          for one in range(0,n_metabolites):
            if one not in g:
              concentrations[idx:idx+n_g,one] = 1.0
          idx += n_g
      else:
        raise RuntimeError('Unknown concentration generation method: ' + sampler)
      all_concentrations = np.concatenate((all_concentrations,concentrations),axis=0)

    if verbose > 0:
      print("Combining basis spectra")
    for count in range(num):
      cons = all_concentrations[count]
      id = str(count)
      if linewidth_mode is None or basis_pool is None or lw_values is None:
        s,c = basis.combine(cons,id)
      elif linewidth_mode == 'perSpectrum':
        # choose a single linewidth uniformly for this spectrum
        lw = float(np.random.choice(lw_values))
        b = basis_pool.get(lw, None)
        if b is None:
          raise RuntimeError(f"No basis for chosen linewidth {lw}")
        s,c = b.combine(cons, id)
      elif linewidth_mode == 'perMetabolite':
        # choose linewidth per metabolite uniformly
        lw_sel = [float(np.random.choice(lw_values)) for _ in self.metabolites]
        spectra = {}
        con = {m: float(cons[j]) for j, m in enumerate(self.metabolites)}
        for acq in basis.acquisitions:
          parts = []
          for j, m in enumerate(self.metabolites):
            b = basis_pool.get(lw_sel[j], None)
            if b is None:
              raise RuntimeError(f"No basis for chosen linewidth {lw_sel[j]}")
            parts.append(b.spectra[m][acq])
          spectra[acq] = Spectrum.combs([con[m] for m in self.metabolites], parts, id, acq)
        Spectrum.correct_b0_multi(spectra)
        if self.pulse_sequence == "megapress":
          if np.max(np.abs(spectra['edit_on'].get_f()[0] - spectra['edit_off'].get_f()[0] - spectra['difference'].get_f()[0])) >= Cfg.val['num_eps']:
            raise RuntimeError("Combined difference spectrum differs from edit_on - edit_off")
        s = spectra
      else:
        raise RuntimeError(f"Unknown linewidth_mode: {linewidth_mode}")
      self.spectra.append(s)
      self.concentrations.append(c)

  def add_noise(self, noise_p, noise_type, noise_mu, noise_sigma, verbose):
    """Add noise to all spectra in the dataset.

    Parameters
    ----------
        noise_p (float): Probability of adding noise to each spectrum (0-1)
        noise_type (str): Type of noise to add ('adc_normal' or 'none')
        noise_mu (float): Maximum noise mean parameter
        noise_sigma (float): Maximum noise standard deviation parameter
        verbose (int): Verbosity level

    Raises
    ------
        RuntimeError: If noise is added twice or unknown noise type
    """
    # Add noise to all spectra
    if self.noise_added:
      raise RuntimeError("Noise added twice to dataset")
    if noise_p > 0.0 and noise_type != "none":
      if verbose > 0:
        if noise_type == "adc_normal":
          print(f"Adding ADC noise Normal(mu={noise_mu},sigma={noise_sigma}) with probability {noise_p} to time signal")
        else:
          raise RuntimeError("Unknown noise type "+noise_type)
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
              if noise_type == "adc_normal":
                self.spectra[idx][a].add_noise_adc_normal(mu=n_mu[idx], sigma=n_sigma[idx])
              else:
                raise RuntimeError("Unknown noise type "+noise_type)
          if 'difference' in self.spectra[idx]:
            # Add difference of noisy spectra
            if 'edit_off' not in self.spectra[idx] or 'edit_on' not in self.spectra[idx]:
              raise RuntimeError("Difference spectrum without edit_off or edit_on")
            self.spectra[idx]['difference'] = Spectrum.comb(1.0,self.spectra[idx]['edit_on'],
                                                            -1.0,self.spectra[idx]['edit_off'],
                                                            self.spectra[idx]['edit_on'].id+"_+_"+self.spectra[idx]['edit_off'].id,
                                                            "difference")
          # B0 correction
          Spectrum.correct_b0_multi(self.spectra[idx])
      if verbose > 1:
        print(f"  Added noise to {n_cnt} of {num} spectra")

  def save(self, path, folder=None, spectra_only=False):
    """Save dataset to disk.

    Parameters
    ----------
        path (str): Base path for saving
        folder (str, optional): Specific folder name. Defaults to None
        spectra_only (bool, optional): Save only spectra, not metadata. Defaults to False

    Returns
    -------
        str: Path to saved dataset folder
    """
    from mrsnet.getfolder import get_folder
    if folder is None:
      folder = get_folder(os.path.join(path,self.name),str(len(self.spectra))+"-%s")
    if not spectra_only:
      joblib.dump({
          'name': self.name,
          'metabolites': self.metabolites,
          'concentrations': self.concentrations,
          'pulse_sequence': self.pulse_sequence
        }, os.path.join(folder, 'info.joblib'))
    if self.noise_added:
      fn = "spectra_noisy.joblib"
    else:
      fn = "spectra_clean.joblib"
    joblib.dump(self.spectra, os.path.join(folder, fn))
    return folder

  @staticmethod
  def load(folder, force_clean=False, info_only=False):
    """Load dataset from disk.

    Parameters
    ----------
        folder (str): Path to dataset folder
        force_clean (bool, optional): Force loading clean spectra. Defaults to False
        info_only (bool, optional): Load only metadata, not spectra. Defaults to False

    Returns
    -------
        Dataset: Loaded dataset object
    """
    info = joblib.load(os.path.join(folder, "info.joblib"))
    spectra = None
    if not force_clean and os.path.isfile(os.path.join(folder, "spectra_noisy.joblib")):
      if not info_only:
        spectra = joblib.load(os.path.join(folder, "spectra_noisy.joblib"))
      noise = True
    else:
      if not info_only:
        spectra = joblib.load(os.path.join(folder, "spectra_clean.joblib"))
      noise = False
    ds = Dataset(info['name'])
    ds.metabolites = info['metabolites']
    ds.spectra = spectra
    ds.concentrations = info['concentrations']
    ds.pulse_sequence = info['pulse_sequence']
    ds.noise_added = noise
    return ds

  def plot_concentrations(self, norm='none'):
    """Plot concentration histograms for all metabolites.

    Parameters
    ----------
        norm (str, optional): Normalization method ('none', 'sum', 'max'). Defaults to 'none'

    Returns
    -------
        matplotlib.figure.Figure or None: Figure object if concentrations exist, None otherwise
    """
    if len(self.concentrations) > 0:
      n_spec = len(self.spectra)
      n_hst = len(self.metabolites)
      n_col = int(np.floor(np.sqrt(n_hst)))
      n_row = n_col
      while n_row * n_col < n_hst:
        n_row += 1
      fig, axes = plt.subplots(n_row, n_col,  sharex=True, sharey=True)
      if isinstance(axes,np.ndarray):
        axes = axes.flatten()
      else:
        axes = np.asarray([axes])
      norm_str = "" if norm == 'none' else f"({norm} normalised) "
      plt.suptitle(f"Concentrations {norm_str}of {self.name}; {len(self.spectra)} spectra")
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

  def export(self, metabolites=None, high_ppm=-4.5, low_ppm=-1, n_fft_pts=2048, norm='sum',
             acquisitions=['edit_off','difference'],datatype='magnitude', normalise=True, export_concentrations=True, verbose=0):
    """Export dataset to tensor format for model training.

    Parameters
    ----------
        metabolites (list, optional): List of metabolites to export. Defaults to None
        high_ppm (float, optional): Upper PPM bound. Defaults to -4.5
        low_ppm (float, optional): Lower PPM bound. Defaults to -1
        n_fft_pts (int, optional): Number of FFT points. Defaults to 2048
        norm (str, optional): Concentration normalization method. Defaults to 'sum'
        acquisitions (list, optional): List of acquisition types. Defaults to ['edit_off','difference']
        datatype (str, optional): Data type for export. Defaults to 'magnitude'
        normalise (bool, optional): Whether to normalize spectra. Defaults to True
        export_concentrations (bool, optional): Whether to export concentrations. Defaults to True
        verbose (int, optional): Verbosity level. Defaults to 0

    Returns
    -------
        tuple: (input_tensor, output_tensor) for model training

    Raises
    ------
        RuntimeError: If tensor shapes are unexpected
    """
    if metabolites is None:
      metabolites = self.metabolites

    if len(self.spectra) > 0:
      if verbose > 0:
        print("Converting spectra to tensor")
      if Cfg.val['npfft_module'][0] == "pyfftw.interfaces":
        # pyfftw causes segmentation fault in parallel execution without this
        import pyfftw
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(60)
      d_inp = joblib.Parallel(n_jobs=-1, prefer="threads")(joblib.delayed(Dataset._export_spectra)(s,
                    acquisitions, datatype, high_ppm, low_ppm, n_fft_pts, normalise)
                for s in tqdm(self.spectra, disable=(verbose<1)))
      d_inp = np.array(d_inp, dtype=np.float64)
      if verbose > 0:
        print("  Shape: " + str(d_inp.shape) + " - [spectrum, acquisition, datatype, frequency]")
    else:
      d_inp = np.ndarray((0,0))
    if export_concentrations and len(self.concentrations) > 0:
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

    if np.sum(d_out.shape) > 0 and \
      (d_out.shape[0] != d_inp.shape[0] or d_out.shape[1] != len(metabolites) or \
       d_inp.shape[1] != len(acquisitions) or d_inp.shape[2] != len(datatype) or \
       d_inp.shape[3] != n_fft_pts):
        raise RuntimeError("Unexpected input/output tensor shape(s)")

    if verbose > 4:
      self._check_export(d_inp,d_out,metabolites, high_ppm, low_ppm, n_fft_pts, norm,
                         acquisitions, datatype, normalise, verbose)

    return d_inp, d_out

  def _check_export(self,d_inp,d_out,metabolites,high_ppm,low_ppm,n_fft_pts,norm,
                    acquisitions,datatype,normalise,verbose):
    """Check dataset export functionality.

    Parameters
    ----------
        d_inp: Input data
        d_out: Output data
        metabolites (list): List of metabolite names
        high_ppm (float): Upper PPM bound
        low_ppm (float): Lower PPM bound
        n_fft_pts (int): Number of FFT points
        norm (str): Normalization method
        acquisitions (list): List of acquisition types
        datatype (list): List of data types
        normalise (bool): Whether to normalize
        verbose (int): Verbosity level
    """
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
        fft = np.ndarray((len(acquisitions),n_fft_pts),dtype=np.complex64)
        a_idx = 0
        a_norm = None
        for a in acquisitions:
          if a == "edit_off":
            a_norm = a_idx
          fft[a_idx,:], _ = self.spectra[s][a].rescale_fft(high_ppm=high_ppm,
                                                           low_ppm=low_ppm,
                                                           npts=n_fft_pts)
          a_idx += 1
        if normalise:
          m = np.abs(fft)
          p = np.angle(fft)
          if a_norm is None:
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
          if verbose > 5:
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
    """Export spectrum data to tensor format.

    Parameters
    ----------
        s: Spectrum object
        acquisitions (list): List of acquisition types
        datatypes (list): List of data types
        high_ppm (float): Upper PPM bound
        low_ppm (float): Lower PPM bound
        n_fft_pts (int): Number of FFT points
        normalise (bool): Whether to normalize

    Returns
    -------
        numpy.ndarray: Exported spectrum tensor
    """
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
      if a_norm is None:
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
          raise RuntimeError(f"Unknown datatype {d}")
        d_idx += 1
    return inp

  @staticmethod
  def _export_concentrations(c, metabolites, norm):
    """Export concentration data to tensor format.

    Parameters
    ----------
        c: Concentration dictionary
        metabolites (list): List of metabolite names
        norm (str): Normalization method

    Returns
    -------
        numpy.ndarray: Exported concentration tensor
    """
    out = np.array([c[m] for m in metabolites], dtype=np.float64)
    if norm == 'max':
      out /= np.max(out)
    elif norm == 'sum':
      out /= np.sum(out)
    elif norm != 'none':
      raise RuntimeError(f"Unknown norm {norm}")
    return out
