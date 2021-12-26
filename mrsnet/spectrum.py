# mrsnet/spectrum.py - MRSNet - individual spectrum
#
# SPDX-FileCopyrightText: Copyright (C) 2019 Max Chandler, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2020-2021 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import random
import subprocess
import json
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pyplot as plt

from . import molecules
from .cfg import Cfg

class Spectrum(object):
  # Spectrum is a class that contains information about a single spectrum.

  def __init__(self, id, pulse_sequence, acquisition, omega,
               source=None, metabolites=None, linewidth=None):
    self.id = id
    self.pulse_sequence = pulse_sequence
    self.acquisition = acquisition
    self.omega = omega

    self.source = source
    self.metabolites = molecules.short_name(metabolites)
    self.linewidth = linewidth

    self.sample_rate = None
    self.fft = None
    self.scale = None
    self.center_ppm = None
    self.b0_shift_ppm = None

    self.noise = None

  def set_f(self, fft, sample_rate, center_ppm=0, b0_shift_ppm=0, scale=1.0, filter_fft=False, remove_water_peak=False):
    self.fft = fft
    self._set(sample_rate, center_ppm, b0_shift_ppm, scale, filter_fft, remove_water_peak)

  def set_t(self, adc, sample_rate, center_ppm=0, b0_shift_ppm=0, scale=1.0, filter_fft=False, remove_water_peak=False):
    self.fft = np.fft.fftshift(np.fft.fft(adc))
    self._set(sample_rate, center_ppm, b0_shift_ppm, scale, filter_fft, remove_water_peak)

  def _set(self, sample_rate, center_ppm, b0_shift_ppm, scale, filter_fft, remove_water_peak):
    self.sample_rate = sample_rate
    self.center_ppm = center_ppm
    self.b0_shift_ppm = b0_shift_ppm
    self.scale = scale
    if filter_fft: # FIXME: check
      b, a = signal.butter(1, 0.7)
      self.fft = signal.filtfilt(b, a, self.fft, padlen=150)
    if remove_water_peak:
      self._fft_remove_water_peak()

  def get_f(self):
    return self.fft*self.scale, \
           np.linspace(-1, 1, len(self.fft)) * self.sample_rate/2.0 / self.omega + self.center_ppm + self.b0_shift_ppm

  def get_t(self):
    return np.fft.ifft(np.fft.ifftshift(self.fft * self.scale)), \
           np.arange(0, len(self.fft), 1) / self.sample_rate

  def correct_b0(self, shift=None):
    # Find reference peak and adjust nu range via ppm_shift
    if shift is None:
      # Pick shift from reference peak
      self.b0_shift_ppm = 0
      for pair in molecules.b0_correction:
        if pair[0] in self.metabolites:
          peak = self._fft_peak_location(pair[1], Cfg.val['b0_correct_ppm_range'])
          if peak:
            shift = pair[1] - peak
            break
    # Apply Shift
    if shift is not None:
      self.b0_shift_ppm = shift
    return shift

  def _fft_peak_location(self, location, ppm_range):
    fft_abs, nu = self.get_f()
    fft_abs = -np.abs(fft_abs)
    mean = np.mean(fft_abs)
    # finds the highest peak from location +- ppm_range
    # FIXME: speed up via index selection and max?
    peak_idxs = fft_abs.argsort()
    for idx in peak_idxs:
      if abs(location - nu[idx]) < ppm_range:
        # FIXME: interpolate?
        return nu[idx]
      if fft_abs[idx] < mean:
        break
    return None

  def rescale_fft(self, high_ppm=-4.5, low_ppm=-1, npts=2048): # FIXME: with get_f?
    # Interpolate oversampled (see Cfg and set_adc) spectrum to get fixed fft bins
    # (avoids issues with b0 correction and simpler than zero padding)
    fft, nu = self.get_f()
    if (np.max(nu) < low_ppm) or (np.min(nu) > high_ppm):
      raise Exception('Requested ppm rescale range out of range of nu of spectrum. Max:' + str(np.min(nu)) + ' Min: ' + str(np.max(nu)))
    freq_step = (low_ppm-high_ppm)/(npts-1)
    fp = interp1d(nu, fft, "cubic") # FIXME: better? zero filling
    return fp(np.arange(high_ppm,low_ppm+freq_step,freq_step)), np.arange(high_ppm,low_ppm+freq_step,freq_step)

  def _fft_remove_water_peak(self):
    # FIXME: check
    # find the peak then set the range centered around it to the median signal of the fft
    water_peak_loc = self._fft_peak_location(molecules.WATER_REFERENCE, Cfg.val['water_peak_ppm_range'])
    if water_peak_loc is not None:
      abs_fft, nu = self.get_f()
      abs_fft = np.abs(abs_fft)
      mean_abs = np.mean(abs_fft)
      under_mean = 0
      for jj in range(0, len(abs_fft)):
        if np.abs(water_peak_loc - nu[jj]) < ppm_range/2.0:
          if self.fft[jj] > mean_abs:
            under_mean = 0
            self.fft[jj] = mean_abs * np.exp(1j*np.angle(self.fft[jj]))
          elif nu[jj] > water_peak_loc:
            # trailing edge detection, stop when the water peak is over
            under_mean += 1
            if under_mean > 5:
              break

  def add_noise_adc_normal(self, mu=0, sigma=0):
    if self.noise != None:
      raise Exception("Adding noise twice is not advised")
    self.noise = ("adc", "normal", mu, sigma)
    adc, _ = self.get_t()
    m = np.max(np.abs(adc))
    noise = (     np.random.normal(mu, sigma, len(adc)) +
             1j * np.random.normal(mu, sigma, len(adc))) * m / self.scale
    self.fft += np.fft.fftshift(np.fft.fft(noise))

  def plot(self, axes, type='fft', mode='magnitude'):
    if type == 'time':
      Y, X = self.get_t()
      axes.set_xlabel('Time (s)')
    elif type == 'fft':
      Y, X = self.get_f() # FXIME: specify range / rescale_fft
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
    if self.noise != None:
      if self.noise[0] == "adc" and self.noise[1] == "normal":
        super_title += f" - ADC Noise N({self.noise[2]},{self.adc_noise[3]})"
      else:
        raise Exception("Unknown noise model")

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
    noise = set([spectra[a].noise for a in spectra])
    if len(source) != 1 or len(pulse_sequence) != 1 or len(omega) != 1 or len(linewidth) != 1 or \
       len(noise) != 1:
      raise Exception("Spectra differ in more than acqusition")

    super_title += next(iter(source)) + ' ' + next(iter(pulse_sequence)).upper() + ' ' + \
                   " @ " + str(next(iter(omega))) + "Hz Linewidth: " + str(next(iter(linewidth)))
    noise = next(iter(noise))
    if noise is not None and noise[0] == "adc" and noise[1] == "normal":
      super_title += f" - ADC Noise N({noise[2]},{noise[3]})"
    else:
      raise Exception("Unknonw noise model")

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
    s = Spectrum(id=id,
                 pulse_sequence='megapress',
                 acquisition='edit_on' if fida_data['edit'][0][0] != 0 else 'edit_off',
                 omega=float(fida_data['omega'][0][0]) * molecules.GYROMAGNETIC_RATIO,
                 source='fid-a',
                 metabolites=[molecules.short_name(str(fida_data['m_name'][0]))],
                 linewidth=float(fida_data['linewidth'][0][0]))
    s.set_t(np.conjugate(np.array(fida_data['fid']).flatten()), # FIXME: why conjugate?
            1/(np.abs(fida_data['t'][0][0] - fida_data['t'][0][1])),
            center_ppm = -np.median(fida_data['nu']))
    return s

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
        s = Spectrum(id=filename, source='pygamma',
                     pulse_sequence=pulse_sequence,
                     metabolites=[molecules.short_name(metabolite)],
                     acquisition=acq,
                     omega=omega,
                     linewidth=linewidth)
        s.set_t(np.array(raw["adc_re"]) - 1j * np.array(raw["adc_im"]),  # FIXME: why conjugate?
                1/dt)
        specs.append(s)
    return specs

  @staticmethod
  def load_lcm(basis_file, acquisition, req_omega, req_metabolites):
    # Load lcmodel basis
    # http://s-provencher.com/pub/LCModel/manual/manual.pdf
    # FIXME: lcmodel GABA spectrum - unusual?
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
              ishift = -metadata["BASIS"]["ISHIFT"]
              fft = np.roll(fft,ishift)
              if ishift > 0:
                fft[:ishift] = 0.0
              elif ishift < 0:
                fft[fft.shape[0]-ishift:] = 0.0
              fft_scale = metadata["BASIS"]["TRAMP"]/(metadata["BASIS"]["CONC"]*metadata["BASIS"]["VOLUME"])
              adc = np.fft.ifft(np.array(fft,dtype=np.complex64)) * fft_scale
              if "PPMSEP" in metadata["NMUSED"]:
                center_ppm = -metadata["NMUSED"]["PPMSEP"]
              else:
                center_ppm = -4.65
              s = Spectrum(id=os.path.basename(basis_file),
                           pulse_sequence='megapress',
                           acquisition=acquisition,
                           omega=metadata["SEQPAR"]['HZPPPM'],
                           source='lcmodel',
                           metabolites=[metabolite],
                           linewidth=1.0) # FIXME: linewidth?
              s.set_f(np.fft.fftshift(fft * fft_scale),1.0/metadata["BASIS1"]['BADELT'],center_ppm=center_ppm)
              specs.append(s)
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
        raise Exception('Loaded dicom file for MEGA-PRESS, but acquisition cannot be determined.\n'
                        'Add "EDIT_OFF", "EDIT_ON" or "DIFF" into the filepath anywhere.')
    else:
      raise Exception(f"Pulse sequence {pulse_sequence} not supported")

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
            if "GlX" in metabolites:
              for m in js[ids[l]].keys():
                ms = molecules.short_name(m)
                if ms == "Gln" or ms == "Glu":
                  cs["GlX"] += float(js[ids[l]][m])
    spec = Spectrum(id=id,
                    pulse_sequence=pulse_sequence,
                    acquisition=acquisition,
                    omega=omega,
                    source='dicom',
                    metabolites=metabolites,
                    linewidth=None) # Unknown
    spec.set_t(dat,1/dt,center_ppm=-4.7,filter_fft=True,remove_water_peak=True)
    return spec, cs
