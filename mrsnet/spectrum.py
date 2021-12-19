# mrsnet/spectrum.py - MRSNet - individual spectrum
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2019, Max Chandler, PhD student at Cardiff University
# Copyright (C) 2020-2021, Frank C Langbein <frank@langbein.org>, Cardiff University

import os
import random
import subprocess
import json
import numpy as np
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt

from . import molecules

class Spectrum(object):
  # Spectrum is a class that contains information about one single spectrum
  # loaded from any source. Time domain data is preferred over frequency
  # domain when loading.

  def __init__(self, id, source=None, metabolites=None, pulse_sequence=None,
               acquisition=None, omega=None, linewidth=None, dt=None, center_ppm=0,
               filter_fft=False, remove_water_peak=False, scale=1.0, raw_adc=[]):
    if acquisition is None:
      raise Exception('Please set acquisition for spectrum.')
    if pulse_sequence == 'megapress' and acquisition not in ['edit_off', 'edit_on', 'difference']:
      raise Exception('Pulse sequence is megapress, but spectrum_type not in [edit_off, edit_on, difference]: ' + acquisition)
    self.id = id
    self.source = source
    self.metabolites = molecules.short_name(metabolites)
    self.pulse_sequence = pulse_sequence
    self.acquisition = acquisition
    self.omega = omega
    self.linewidth = linewidth
    self.dt = dt
    self.center_ppm = center_ppm

    self.raw_adc = np.array(raw_adc)
    if any(np.isnan(self.raw_adc)):
      raise Exception('Raw adc contains nan values')
    self.scale = scale

    self.b0_ppm_shift = 0
    self.zero_pad = 0
    self.remove_water_peak = remove_water_peak
    self.filter_fft = filter_fft

    self.adc_noise_mu = 0
    self.adc_noise_sigma = 0

    self.fft_cache = None

  def adc(self, pad=True):
    if pad:
      adc = np.append(self.raw_adc * self.scale, np.zeros(self.zero_pad))
    else:
      adc = self.raw_adc * self.scale
    return adc

  def adc_len(self, pad=True):
    if pad:
      return len(self.raw_adc) + self.zero_pad
    return len(self.raw_adc)

  def nu(self, npts=None):
    if npts is None:
      npts = self.adc_len()
    nu = np.linspace(-1, 1, npts) / (2.0 * self.dt * self.omega) + self.center_ppm + self.b0_ppm_shift
    return nu

  def correct_b0(self, ppm_shift=None):
    # Find reference peak and adjust nu range via ppm_shift
    if ppm_shift is None:
      if self.b0_ppm_shift != 0:
        self.b0_ppm_shift = 0
      # No shift defined, find it
      reference_peaks = []
      if len(self.metabolites) > 0:
        if 'NAA' in self.metabolites:
          reference_peaks.append(molecules.NAA_REFERENCE)
        if 'Cr' in self.metabolites:
          reference_peaks.append(molecules.CR_REFERENCE)
      if len(reference_peaks) == 0:
        # no ref metabolites found, try them anyway.
        reference_peaks = [molecules.NAA_REFERENCE, molecules.CR_REFERENCE]
      ppm_shift = []
      for reference_signal in reference_peaks:
        peak = self._fft_peak_location(reference_signal, 0.25)
        if peak:
          ppm_shift.append(reference_signal - peak)
      if len(ppm_shift) == 0:
        return None
      # Average over all reference peaks
      ppm_shift = np.mean(np.array(ppm_shift, dtype=np.float64))
    # Apply Shift
    self.b0_ppm_shift = ppm_shift
    return ppm_shift

  def _fft_peak_location(self, location, ppm_range, fft=None):
    nu = self.nu()
    if fft is None:
      fft_abs = -np.abs(self.fft())
    else:
      fft_abs = -np.abs(fft) # Avoids loop if already called from fft (for remove_water_peak)
    median = np.median(fft_abs)
    # finds the highest peak from location +- ppm_range
    peak_idxs = fft_abs.argsort()
    for idx in peak_idxs:
      if abs(location - nu[idx]) < ppm_range:
        return nu[idx]
      if fft_abs[idx] < median:
        return None
    return None

  def fft(self):
    if self.fft_cache is None:
      # fft routines for different input sources
      fft = np.fft.fftshift(np.fft.fft(self.adc(), self.adc_len()))
      if self.source == 'pygamma' or self.source == 'fid-a':
        fft = np.conjugate(np.flip(fft, 0))
      if self.filter_fft:
        b, a = signal.butter(1, 0.7)
        fft = signal.filtfilt(b, a, fft, padlen=150)
      if self.remove_water_peak:
        fft = self._fft_remove_water_peak(fft)
      self.fft_cache = fft
    return self.fft_cache

  def rescale_fft(self, high_ppm=-4.5, low_ppm=-1, npts=2048):
    # Zero pads the time domain to fill the desired window with npts
    recursion_limit = 500
    nu = self.nu()
    if (np.max(nu) < low_ppm) or (np.min(nu) > high_ppm):
      raise Exception('Requested ppm rescale range out of range of nu of spectrum. Max:' + str(np.min(nu)) + ' Min: ' + str(np.max(nu)))
    # Calculate initially how many points in that range
    index = (nu >= high_ppm) & (nu <= low_ppm)
    nu_pts = len(nu[index])
    counter = 0
    while nu_pts != npts:
      if counter > recursion_limit:
        raise Exception('Iteration limit hit!')
      if counter > 100:
        print('Counter is getting high... ' + str(self.zero_pad) + ' : ' + str(nu_pts) + ' aiming for: ' + str(npts))
      # fine tune if that's not quite right
      percent_range = len(nu) / float(nu_pts)
      # random is added as there is sometimes some aliasing effects, this corrects it
      self._update_zero_pad(self.zero_pad + int(round(np.round((npts - nu_pts) * percent_range) + np.random.random())))
      if self.zero_pad < 0:
        raise Exception('Real data is too large to be input into the network, would have to reduce the '
                        'resolution of it. OR train a network with a higher resolution across the ppm range.')
      nu = self.nu()
      index = (nu >= high_ppm) & (nu <= low_ppm)
      nu_pts = len(nu[index])
      counter += 1
    return self.fft()[index], nu[index]

  def _update_zero_pad(self, new_zero_pad):
    if new_zero_pad != self.zero_pad:
      self.fft_cache = None
      self.zero_pad = new_zero_pad
      self.correct_b0(self.b0_ppm_shift)

  def _fft_remove_water_peak(self, fft, ppm_range=0.6):
    # find the peak then set the range centered around it to the median signal of the fft
    water_peak_loc = self._fft_peak_location(molecules.WATER_REFERENCE, 0.5, fft=fft)
    if water_peak_loc is not None:
      nu = self.nu()
      abs_fft = np.abs(fft)
      mean_abs = np.mean(abs_fft)
      under_mean = 0
      for jj in range(0, len(fft)):
        if np.abs(water_peak_loc - nu[jj]) < ppm_range/2.0:
          if abs_fft[jj] > mean_abs:
            under_mean = 0
            fft[jj] = mean_abs * np.exp(1j*np.angle(fft[jj]))
          elif nu[jj] > water_peak_loc:
            # trailing edge detection, stop when the water peak is over
            under_mean += 1
            if under_mean > 5:
              return fft
    return fft

  def add_noise(self, mu=0, sigma=0):
    if self.adc_noise_mu != 0.0 or self.adc_noise_sigma != 0.0:
      raise Exception("Adding noise twice is not advised")
    self.adc_noise_mu = mu
    self.adc_noise_sigma = sigma
    self.fft_cache = None
    self.raw_adc += (     np.random.normal(mu, sigma, len(self.raw_adc)) + \
                     1j * np.random.normal(mu, sigma, len(self.raw_adc))) / self.scale

  def plot(self, axes, type='fft', mode='magnitude'):
    if type == 'time':
      Y = self.adc()
      X = np.arange(0,len(Y)) * self.dt
      axes.set_xlabel('Time (s)')
    elif type == 'fft':
      Y, X = self.rescale_fft()
      axes.set_xlabel('Frequency (ppm)')
    else:
      raise Exception("Unknown plot type")
    if mode == 'magnitude':
      Y = np.abs(Y)
      axes.set_ylabel('Magn.')
    elif mode == 'phase':
      Y = np.angle(Y)
      axes.set_ylabel('Phase')
    elif mode == 'real':
      Y = np.real(Y)
      axes.set_ylabel('Re')
    elif mode == 'imaginary':
      Y = np.imag(Y)
      axes.set_ylabel('Im')
    else:
      raise Exception("Unkonw plot mode "+mode)
    axes.plot(X,Y)

  def plot_spectrum(self, concentrations={}, screen_dpi=96, type='fft'):
    n_cols = 1

    super_title = type.upper() + " "
    if len(concentrations) > 0:
      n_cols = 2
    else:
      super_title += "-".join(self.metabolites[0]) + ' '
    super_title += self.source + ' ' + self.pulse_sequence.upper() + ' ' + self.acquisition + " @ " + str(self.omega) + "Hz Linewidth: " + str(self.linewidth)
    if self.adc_noise_mu != 0.0 or self.adc_noise_sigma != 0.0:
      super_title += f" - Noise mu: {self.adc_noise_mu} sigma: {self.adc_noise_sigma}"

    figure, axes = plt.subplots(4, n_cols, sharex=True, dpi=screen_dpi)
    if len(axes.shape) == 1:
      axes = np.reshape(axes, (4, 1))
    else:
      axes[0,1].remove()
      axes[1,1].remove()
      axes[2,1].remove()
      axes[3,1].remove()

    plt.suptitle(super_title)

    self.plot(axes[0,0], type=type, mode='magnitude')
    axes[0,0].set_xlabel("")
    self.plot(axes[1,0], type=type, mode='phase')
    axes[1,0].set_xlabel("")
    self.plot(axes[2,0], type=type, mode='real')
    axes[2,0].set_xlabel("")
    self.plot(axes[3,0], type=type, mode='imaginary')

    if n_cols == 2:
      ax = plt.subplot(1, 2, 2)
      plt.title('Concentrations')
      cn = [n for n in concentrations.keys()]
      cv = [concentrations[v] for v in cn]
      ax.bar(np.linspace(0, len(concentrations) - 1, len(concentrations)), cv)
      ax.set_xticks(np.arange(len(self.metabolites)))
      ax.set_xticklabels(molecules.short_name(cn))

    return figure

  @staticmethod
  def plot_full_spectrum(spectra, concentrations={}, screen_dpi=96, type='fft'):
    n_cols = len(spectra)

    super_title = type.upper() + " "
    if len(concentrations) > 0:
      n_cols +=1
    else:
      super_title += "-".join(self.metabolites[0]) + " "
    source = set([spectra[a].source for a in spectra])
    pulse_sequence = set([spectra[a].pulse_sequence for a in spectra])
    omega = set([spectra[a].omega for a in spectra])
    linewidth = set([spectra[a].linewidth for a in spectra])
    adc_noise_mu = set([spectra[a].adc_noise_mu for a in spectra])
    adc_noise_sigma = set([spectra[a].adc_noise_sigma for a in spectra])
    if len(source) != 1 or len(pulse_sequence) != 1 or len(omega) != 1 or len(linewidth) != 1 or \
       len(adc_noise_mu) != 1 or len(adc_noise_sigma) != 1:
      raise Exception("Spectra differ in more than acqusition")

    super_title += next(iter(source)) + ' ' + next(iter(pulse_sequence)).upper() + ' ' + \
                   " @ " + str(next(iter(omega))) + "Hz Linewidth: " + str(next(iter(linewidth)))
    adc_noise_mu = next(iter(adc_noise_mu))
    adc_noise_sigma = next(iter(adc_noise_sigma))
    if adc_noise_mu != 0.0 or adc_noise_sigma != 0.0:
      super_title += f" - Noise mu: {adc_noise_mu} sigma: {adc_noise_sigma}"

    figure, axes = plt.subplots(4, n_cols, sharex=True, dpi=screen_dpi)
    if len(axes.shape) == 1:
      axes = np.reshape(axes, (4, 1))
    else:
      axes[0,n_cols-1].remove()
      axes[1,n_cols-1].remove()
      axes[2,n_cols-1].remove()
      axes[3,n_cols-1].remove()

    plt.suptitle(super_title)

    col = 0
    for a in sorted(spectra):
      axes[0,col].set_title(a.upper())
      spectra[a].plot(axes[0,col], type=type, mode='magnitude')
      axes[0,col].set_xlabel("")
      if col > 0:
        axes[0,0].get_shared_y_axes().join(axes[0,0], axes[0,col])
      spectra[a].plot(axes[1,col], type=type, mode='phase')
      axes[1,col].set_xlabel("")
      if col > 0:
        axes[1,0].get_shared_y_axes().join(axes[1,0], axes[1,col])
      spectra[a].plot(axes[2,col], type=type, mode='real')
      axes[2,col].set_xlabel("")
      if col > 0:
        axes[2,0].get_shared_y_axes().join(axes[2,0], axes[2,col])
      spectra[a].plot(axes[3,col], type=type, mode='imaginary')
      if col > 0:
        axes[3,0].get_shared_y_axes().join(axes[3,0], axes[3,col])
      col += 1

    if n_cols > col:
      ax = plt.subplot(1, n_cols, n_cols)
      plt.title('Concentrations')
      cn = [n for n in concentrations.keys()]
      cv = [concentrations[v] for v in cn]
      ax.bar(np.linspace(0, len(concentrations) - 1, len(concentrations)), cv)
      metabolites = spectra[next(iter(spectra))].metabolites
      ax.set_xticks(np.arange(len(metabolites)))
      ax.set_xticklabels(molecules.short_name(cn))

    return figure

  @staticmethod
  def load_fida(fida_file,id):
    fida_data = loadmat(fida_file)
    return Spectrum(id=id,
                    source='fid-a',
                    metabolites=[molecules.short_name(str(fida_data['m_name'][0]))],
                    pulse_sequence='megapress',
                    acquisition='edit_on' if fida_data['edit'][0][0] != 0 else 'edit_off',
                    omega=float(fida_data['omega'][0][0]) * molecules.GYROMAGNETIC_RATIO,
                    linewidth=float(fida_data['linewidth'][0][0]),
                    dt = np.abs(fida_data['t'][0][0] - fida_data['t'][0][1]),
                    center_ppm = -np.median(fida_data['nu']),
                    raw_adc = np.array(fida_data['fid']).flatten())

  @staticmethod
  def load_pygamma(pygamma_dir, metabolite, pulse_sequence, omega, linewidth,
                   npts, dt):
    cache_dir = os.path.join(pygamma_dir,
                             pulse_sequence + '_' + str(omega) + '_' +
                             str(linewidth) + '_' + str(npts) + '_' + str(dt))
    filename = os.path.join(cache_dir, metabolite+".json")
    if not os.path.exists(filename):
      if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
      pygamma_cmd = ['/usr/bin/env', 'python3', os.path.join('mrsnet',
                      'simulators', 'pygamma', 'pygamma_simulator.py'),
                     metabolite, str(omega), pulse_sequence, str(linewidth),
                     str(npts), str(dt), cache_dir]
      try:
        p = subprocess.Popen(pygamma_cmd)
      except OSError as e:
        raise Exception('PyGamma simulations failed') from e
      p.wait()
    specs = []
    with open(filename, 'rb') as load_file:
      for raw in json.load(load_file):
        if pulse_sequence == 'megapress':
          if raw["count"] == 0:
            acq = 'edit_off'
          elif raw["count"] == 1:
            acq = 'edit_on'
          else:
            raise Exception('More than 2 mx objects for megapress? Something is wrong here.')
        specs.append(Spectrum(id=filename.split("/")[-2:], source='pygamma',
                              metabolites=[molecules.short_name(metabolite)],
                              pulse_sequence=pulse_sequence,
                              acquisition=acq,
                              omega=omega,
                              linewidth=linewidth,
                              dt=dt,
                              center_ppm=0,
                              raw_adc = np.array(raw["adc_re"]) + 1j * np.array(raw["adc_im"])))
    return specs

  @staticmethod
  def load_lcm(basis_file, acquisition, req_omega, req_metabolites):
    # Load lcmodel basis
    # http://s-provencher.com/pub/LCModel/manual/manual.pdf
    if not os.path.exists(basis_file):
      raise Exception('Basis file does not exist: ' + basis_file)
    specs = []
    with open(basis_file) as file:
      line_buffer = []
      metadata = {}
      area = ""
      parse = True
      while parse:
        line = file.readline().strip()
        if len(line) == 0: # Process last block, then quite
          line="$END"
          parse = False
        if line[0] == '$':
          if area == "SEQPAR" or area == "BASIS" or area == "BASIS1" or area == "NMUSED":
            metadata[area] = {}
            for l in line_buffer:
              sl = l.split('=')
              if sl[1][-1] == ",":
                sl[1] = sl[1][0:-1]
              v = sl[1].strip()
              try:
                if v[0] == "\'":
                  v = v.strip("'")
                elif ' ' in v:
                  vals = v.split()
                  v = []
                  for vv in vals:
                    try:
                      if '.' in vv:
                        vv = float(vv)
                      else:
                        vv = int(vv)
                    except:
                      pass
                    v.append(vv)
                else:
                  if '.' in v:
                    v = float(v)
                  else:
                    v = int(v)
              except:
                pass
              metadata[area][sl[0].strip()] = v
          elif area == "spectrum":
            if np.abs(metadata['SEQPAR']['HZPPPM'] - req_omega) > (molecules.GYROMAGNETIC_RATIO/5):
              # more than a 0.2T difference, there's an issue
              raise Exception('LCModel basis set (%.2fT) is more than 0.2T different to prescibed '
                              'omega (%.2fT).' % (metadata['SEQPAR']['HZPPPM']/molecules.GYROMAGNETIC_RATIO,
                                                  req_omega/molecules.GYROMAGNETIC_RATIO))
            if metadata['SEQPAR']['SEQ'] != "MEGA-": # not megapress
              raise Exception('Unrecognised LCM pulse sequence: ' + metadata['SEQ'])
            try:
              metabolite = molecules.short_name(metadata["BASIS"]["METABO"])
            except:
              metabolite = "Unknown"
            if metabolite in req_metabolites:
              # All lcmodel spectra are stored as fourier transforms, so we convert them back to the ADC
              nums = " ".join(line_buffer).split()
              if len(nums) % 2 != 0:
                raise Exception('Uneven fft number, the real/imag switching does not work here or the file has been loaded in wrong!')
              fft = []
              for ii in range(0, len(nums), 2):
                fft.append(float(nums[ii]) + 1j*float(nums[ii + 1]))
              fft = np.roll(fft,-metadata["BASIS"]["ISHIFT"])
              adc = np.fft.ifft(np.array(fft,dtype=np.complex64))
              if "PPMSEP" in metadata["NMUSED"]:
                center_ppm = -metadata["NMUSED"]["PPMSEP"]
              else:
                center_ppm = -4.65
              specs.append(Spectrum(id=os.path.basename(basis_file),
                                    source='lcmodel',
                                    metabolites=[metabolite],
                                    pulse_sequence='megapress',
                                    acquisition=acquisition,
                                    omega=metadata["SEQPAR"]['HZPPPM'],
                                    linewidth=1,
                                    dt=metadata["BASIS1"]['BADELT'],
                                    center_ppm=center_ppm,
                                    raw_adc = adc,
                                    scale=metadata["BASIS"]["TRAMP"]/(metadata["BASIS"]["CONC"]*metadata["BASIS"]["VOLUME"])))
          elif area != "":
            raise IOError(f"Unknown section in LCModel basis file {area}")
          if line[1:] == "END":
            if area == "BASIS":
              area = "spectrum"
            else:
              area = ""
          else:
            area = line[1:]
          line_buffer = []
        else:
          if len(line_buffer) == 0 or '=' in line:
            line_buffer.append(line)
          else:
            line_buffer[-1] += " " + line
    file.close()
    return specs

  @staticmethod
  def load_dicom(file, concentrations=None, metabolites=[]):
    if not os.path.exists(file):
        raise Exception('Dicom file does not exist: ' + file)
    from .qdicom.read_dicom_siemens import read_dicom
    import struct
    # We assume it's a Siemens dicom spectrum, so do not check this
    dicom, info = read_dicom(file)
    # Read Siemens DICOM spectroscopy data tag (0x7fe1, 0x1010) as a list of complex numbers ReImReIm...; 4 byte floats; little endian
    TAG_SPECTROSCOPY_DATA = (0x7fe1, 0x1010)
    TAG_PATIENT_ID = (0x0010, 0x0020)
    data = dicom[TAG_SPECTROSCOPY_DATA].value
    data = struct.unpack("<%df" % (len(data) / 4), data)
    data = np.array([complex(data[l], data[l+1]) for l in range(0, len(data), 2)])

    # Get pulse sequence (this works for Siemens)
    pulse_sequence = info["[CSA Image Header Info]"]["SequenceName"]
    if pulse_sequence in ['svs_edit', 'svs_ed', 'megapress']:
      pulse_sequence = 'megapress'
    elif pulse_sequence == 'svs_se' or pulse_sequence == '*svs_se':
      pulse_sequence = 'press'
    elif pulse_sequence == 'svs_st':
      pulse_sequence = 'steam'
    else:
      raise Exception(f"{file} - Unrecognised dicom pulse sequence: {pulse_sequence}")
    # Acquisition
    if pulse_sequence == 'megapress':
      if 'EDIT_OFF' in file:
        acquisition = 'edit_off'
      elif 'EDIT_ON' in file:
        acquisition = 'edit_on'
      elif 'DIFF' in file:
        acquisition = 'difference'
      else:
        raise Exception('Loaded dicom file of type MEGA-PRESS, but I can\'t figure out which acquisition this '
                        'is (Edit On, Edit Off or Difference). \n'
                        'Please manuall specifiy it (add "EDIT_OFF", "EDIT_ON" or "DIFF" into the filepath anywhere).')
    elif pulse_sequence == 'press':
      if 'OFF' in file:
        acquisition = 'edit_off'
      else:
        raise Exception('Loaded dicom file of type PRESS, but I can\'t figure out which acquisition this '
                        'is (Edit Off?). \n'
                        'Please manuall specifiy it (add "OFF" into the filepath anywhere).')
    else:
      acquisition = 'unknown'
      raise Exception('Non-megapress spectra not supported')

    id = dicom[TAG_PATIENT_ID].value
    if len(id) < 1:
      id = "/".join([x for x in file.split("/")[-4:] if x[-4:].lower() != '.ima'])

    omega = float(info["[CSA Image Header Info]"]["ImagingFrequency"])
    dt = float(info["[CSA Image Header Info]"]["RealDwellTime"]) * 1e-9

    cs = np.array([])
    if concentrations is not None:
      ids = file.split('/')
      with open(concentrations, 'r') as f:
        js = json.load(f)
        if len(metabolites) == 0:
          metabolites = set()
          for k in js.keys():
            for l in js[k].keys():
              metabolites.add(l)
          metabolites = molecules.short_name(list(metabolites))
          metabolites.sort()
        cs = {}
        for m in metabolites:
          cs[m] = 0.0
        for l in range(len(ids)-1,-1,-1):
          if ids[l] in js:
            for m in js[ids[l]].keys():
              ms = molecules.short_name(m)
              if ms in metabolites:
                cs[ms] = float(js[ids[l]][m])

    spec = Spectrum(id=id,
                    source='dicom',
                    metabolites=metabolites,
                    pulse_sequence=pulse_sequence,
                    acquisition=acquisition,
                    omega=omega,
                    linewidth=-1.0, # Unknown
                    dt=dt,
                    center_ppm=-4.7,
                    raw_adc=data,
                    remove_water_peak=True)

    return spec, cs
