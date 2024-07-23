# mrsnet/spectrum.py - MRSNet - individual spectrum
#
# SPDX-FileCopyrightText: Copyright (C) 2019 Max Chandler, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2020-2024 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import random
import subprocess
import json
import numpy as np
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from mrsnet import molecules
from mrsnet.cfg import Cfg

npfft = getattr(__import__(Cfg.val['npfft_module'][0], fromlist=[Cfg.val['npfft_module'][1]]),
                Cfg.val['npfft_module'][1])

class Spectrum:
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

  def set_f(self, fft, sample_rate, center_ppm=0, b0_shift_ppm=0, scale=1.0,
            remove_water_peak=False, phase_correct=False):
    self.fft = np.asarray(fft)
    self._set(sample_rate, center_ppm, b0_shift_ppm, scale, remove_water_peak, phase_correct)

  def set_t(self, adc, sample_rate, center_ppm=0, b0_shift_ppm=0, scale=1.0,
            remove_water_peak=False, phase_correct=False):
    self.fft = npfft.fftshift(npfft.fft(adc))
    self._set(sample_rate, center_ppm, b0_shift_ppm, scale, remove_water_peak, phase_correct)

  def _set(self, sample_rate, center_ppm, b0_shift_ppm, scale, remove_water_peak, phase_correct):
    self.sample_rate = sample_rate
    self.center_ppm = center_ppm
    self.b0_shift_ppm = b0_shift_ppm
    self.scale = scale
    if phase_correct and Cfg.val['phase_correct'] != None: # Only phase correct if also configured
      if Cfg.dev('spectrum_set_phase_correct'):
        fig, axs = plt.subplots(1,2)
        freq, nu = self.get_f()
        axs[0].plot(nu, np.real(freq), color='r')
        axs[0].plot(nu, np.imag(freq), color='g')
        axs[0].set_title("Raw Dicom Data")
      if Cfg.val['phase_correct'] == 'acme':
        self._phase_correct_acme()
      elif Cfg.val['phase_correct'] == 'ernst':
        self._phase_correct_ernst()
      else:
        raise RuntimeError(f"Unknown phase correction algorithm {Cfg.val['phase_correct']}")
      if Cfg.dev('spectrum_set_phase_correct'):
        freq, nu = self.get_f()
        axs[1].plot(nu, np.real(freq), color='r')
        axs[1].plot(nu, np.imag(freq), color='g')
        axs[1].set_title("Phase Corrected Dicom Data")
        plt.show()
    if remove_water_peak:
      self._fft_remove_water_peak()

  def get_f(self):
    return self.fft*self.scale, \
           np.linspace(-1, 1, len(self.fft)) * self.sample_rate/2.0 / self.omega + self.center_ppm + self.b0_shift_ppm

  def get_t(self):
    return npfft.ifft(npfft.ifftshift(self.fft * self.scale)), \
           np.arange(0, len(self.fft), 1) / self.sample_rate

  def correct_b0(self, shift=None):
    # Find reference peak and adjust nu range via ppm_shift
    peak_val = 0.0
    if shift is None:
      # Pick shift from reference peak
      self.b0_shift_ppm = 0
      for pair in molecules.B0_CORRECTION:
        if pair[0] in self.metabolites:
          peak, val = self._fft_peak_location(pair[1], Cfg.val['b0_correct_ppm_range'])
          if peak and peak_val < val:
            shift = pair[1] - peak
            peak_val = val
    # Apply Shift
    if shift is not None:
      self.b0_shift_ppm = shift
    return shift, peak_val

  def _fft_peak_location(self, location, ppm_range):
    fft, nu = self.get_f()
    fft_abs = -np.abs(fft)
    cut_off = np.median(fft_abs)
    # finds the highest peak from location +- ppm_range
    peak_idxs = fft_abs.argsort()
    for idx in peak_idxs:
      if abs(location - nu[idx]) < ppm_range:
        # BG Quinn, EJ Hannan. The Estimation and Tracking of Frequency, 2001.
        # https://dspguru.com/dsp/howtos/how-to-interpolate-fft-peak/
        if Cfg.val['fft_peak_location_estimator'] == None or idx < 1 or idx >= len(nu) - 1:
          return nu[idx], -fft_abs[idx] # at boundary (should never really be there)
        if Cfg.val['fft_peak_location_estimator'] == 'quadratic':
          # Quadratic method
          y1 = -fft_abs[idx-1]
          y2 = -fft_abs[idx]
          y3 = -fft_abs[idx+1]
          de = 2.0 * (2.0*y2 - y1 - y3)
          if np.abs(de) < 1e-10:
            return nu[idx], -fft_abs[idx] # zero denominator, return bucket freq.
          d = (y3-y1) / de
        elif Cfg.val['fft_peak_location_estimator'] == 'quinn2':
          # Quinn's second estimator (least RMS error)
          de = (fft[idx].real**2 + fft[idx].imag**2)
          if np.abs(de) < 1e-10:
            return nu[idx], -fft_abs[idx] # zero denominator, return bucket freq.
          ap = (fft[idx+1].real*fft[idx].real + fft[idx+1].imag*fft[idx].imag) / de
          dp = -ap / (1-ap)
          am = (fft[idx-1].real*fft[idx].real + fft[idx-1].imag*fft[idx].imag) / de
          dm = am / (1-am)
          dp2 = dp**2
          dm2 = dm**2
          f1 = np.sqrt(6.0)/24.0
          f2 = np.sqrt(2.0/3.0)
          tau_dp2 = np.log(3.0*(dp2**2)+6.0*dp2+1)/4.0 - f1*np.log((dp2+1.0-f2)/(dp2+1.0+f2))
          tau_dm2 = np.log(3.0*(dm2**2)+6.0*dm2+1)/4.0 - f1*np.log((dm2+1.0-f2)/(dm2+1.0+f2))
          # It seems d has to be negated to move estimator in right direction
          d = -1.0 * ((dp+dm)/2.0 + tau_dp2 - tau_dm2)
        elif Cfg.val['fft_peak_location_estimator'] == 'jain':
          # Jain's method
          y1 = -fft_abs[idx-1]
          y2 = -fft_abs[idx]
          y3 = -fft_abs[idx+1]
          if y1 > y3:
            a = y2/y1
            idx -= 1
          else:
            a = y3/y2
          d = a/(1.0+a)
        else:
          raise RuntimeError(f"Unknown fft peak location estimator {Cfg.val['fft_peak_location_estimator']}")
        return nu[idx] + (nu[1]-nu[0])*d, -fft_abs[idx]
      if fft_abs[idx] > cut_off:
        break
    return None, None

  def _phase_correct_ernst(self):
    # R Ernst. Numerical Hilbert transform and automatic phase correction in
    # magnetic resonance spectroscopy. J Magn Res 1(1):7-26, 1969.
    # https://www.sciencedirect.com/science/article/abs/pii/0022236469900031
    def err(para):
      val = para.valuesdict()
      # Adjust phase
      shift = val['phi0'] + val['phi1'] * (np.linspace(-1, 1, len(self.fft)) * self.sample_rate/2.0)
      s_fft = self.fft * np.exp(1j*shift)
      # Target function
      return np.sum(np.imag(s_fft))
    import lmfit
    para = lmfit.Parameters()
    para.add('phi0', value=0.0, min=-np.pi/4.0, max=np.pi/4.0)
    para.add('phi1', value=0.0, min=-0.1, max=0.1)
    res = lmfit.minimize(err, para, method='simplex')
    val = res.params
    # Adjust phase
    idx = np.argmax(np.abs(np.real(self.fft)))
    max_fft = np.real(self.fft[idx])
    shift = val['phc0'] + val['phc1'] * (np.linspace(-1, 1, len(self.fft)) * self.sample_rate/2.0)
    self.fft *= np.exp(1j*shift)
    idx = np.argmax(np.abs(np.real(self.fft)))
    max_s_fft = np.real(self.fft[idx])
    if max_s_fft * max_fft < 0.0: # avoid flipping the phase / turn real positive, if it should be negative
      self.fft *= np.exp(1j*np.pi)

  def _phase_correct_acme(self):
    # Automated phase Correction based on Minimization of Entropy
    # L Chen, Z Weng, LY Goh, M Garland. An efficient algorithm for automatic
    # phase correction of NMR spectra based on entropy minimization. J Magn
    # Res 158:164–168, 2002.
    def entropy(para):
      val = para.valuesdict()
      # Adjust phase
      shift = val['phc0'] + val['phc1'] * (np.linspace(-1, 1, len(self.fft)) * self.sample_rate/2.0)
      s_fft = self.fft * np.exp(1j*shift)
      # Target function
      r = np.real(s_fft)
      r1 = np.abs((r[1:] - r[:-1]))
      h = r1 / np.sum(r1)
      h[np.abs(h)<1e-8] = 1.0
      return -np.sum(h*np.log(h)) + Cfg.val['phase_correct_acme_gamma'] * np.sum(r[r<0]**2)
    import lmfit
    para = lmfit.Parameters()
    para.add('phc0', value=0.0, min=-np.pi, max=np.pi)
    para.add('phc1', value=0.0, min=-0.1, max=0.1)
    res = lmfit.minimize(entropy, para, method='simplex')
    val = res.params
    # Adjust phase
    idx = np.argmax(np.abs(np.real(self.fft)))
    max_fft = np.real(self.fft[idx])
    shift = val['phc0'] + val['phc1'] * (np.linspace(-1, 1, len(self.fft)) * self.sample_rate/2.0)
    self.fft *= np.exp(1j*shift)
    idx = np.argmax(np.abs(np.real(self.fft)))
    max_s_fft = np.real(self.fft[idx])
    if max_s_fft * max_fft < 0.0: # avoid flipping the phase / turn real positive, if it should be negative
      self.fft *= np.exp(1j*np.pi)

  def rescale_fft(self, high_ppm=-4.5, low_ppm=-1, npts=2048):
    # Resample fft to prescribed frequency bins via zero filling
    fft, nu = self.get_f()
    if (np.max(nu) < low_ppm) or (np.min(nu) > high_ppm):
      raise RuntimeError(f"Requested ppm rescale range out of range [{nu[0]},{nu[len(nu)-1]}]")
    freq_step = (low_ppm-high_ppm)/npts
    bw = self.sample_rate/2.0/self.omega
    t_samples = int(2.0*bw/freq_step)
    rnu = np.linspace(-1, 1, t_samples) * bw + self.center_ppm + self.b0_shift_ppm
    index = (rnu >= high_ppm) & (rnu <= low_ppm)
    repeats = 0
    while len(rnu[index]) != npts:
      t_samples += 1
      rnu = np.linspace(-1, 1, t_samples) * bw + self.center_ppm + self.b0_shift_ppm
      index = (rnu >= high_ppm) & (rnu <= low_ppm)
      repeats += 1
      if repeats > Cfg.val['spectrum_rescale_fft_max_repeats']:
        raise RuntimeError(f"Length error: got {len(rnu[index])}, expected {npts}")
    ifft = npfft.ifft(npfft.ifftshift(self.fft))
    if t_samples > len(self.fft):
      ifft = np.append(ifft, np.zeros(t_samples - len(self.fft)))
    else:
      ifft = ifft[0:t_samples]
    rfft = npfft.fftshift(npfft.fft(ifft))
    return rfft[index], rnu[index]

  def _fft_remove_water_peak(self):
    # find the peak then set the range centered around it to the median signal of the fft
    water_peak_loc, _ = self._fft_peak_location(molecules.WATER_REFERENCE, Cfg.val['water_peak_ppm_range'])
    if water_peak_loc is not None:
      fft, nu = self.get_f()
      abs_fft = np.abs(fft)
      cut_off = np.median(abs_fft)
      under_mean = 0
      for jj in range(0, len(abs_fft)):
        if nu[jj] >= water_peak_loc and nu[jj] < water_peak_loc + Cfg.val['water_peak_ppm_range']/2.0:
          if np.abs(self.fft[jj]) > cut_off:
            under_mean = 0
            self.fft[jj] = cut_off * np.exp(1j*np.angle(self.fft[jj]))
          elif nu[jj] > water_peak_loc:
            # trailing edge detection, stop when the water peak is over
            under_mean += 1
            if under_mean > Cfg.val['water_peak_under_max']:
              break
      under_mean = 0
      for jj in reversed(range(0, len(abs_fft))):
        if nu[jj] <= water_peak_loc and nu[jj] > water_peak_loc - Cfg.val['water_peak_ppm_range']/2.0:
          if np.abs(self.fft[jj]) > cut_off:
            under_mean = 0
            self.fft[jj] = cut_off * np.exp(1j*np.angle(self.fft[jj]))
          else:
            # trailing edge detection, stop when the water peak is over
            under_mean += 1
            if under_mean > Cfg.val['water_peak_under_max']:
              break

  def add_noise_adc_normal(self, mu=0, sigma=0):
    if self.noise != None:
      raise RuntimeError("Adding noise twice is not advised")
    self.noise = ("adc", "normal", mu, sigma)
    adc, _ = self.get_t()
    m = np.max(np.abs(adc))
    noise = (     np.random.normal(mu, sigma, len(adc)) +
             1j * np.random.normal(mu, sigma, len(adc))) * m / self.scale
    self.fft += npfft.fftshift(npfft.fft(noise))

  def plot(self, axes, type='fft', mode='magnitude'):
    if type == 'time':
      Y, X = self.get_t()
      axes.set_xlabel('Time (s)')
    elif type == 'fft':
      Y, X = self.rescale_fft()
      axes.set_xlabel('Frequency (ppm)')
      axes.xaxis.set_major_formatter(FuncFormatter(lambda x_val, tick_pos: "{:.8g}".format(np.abs(x_val))))
    else:
      raise RuntimeError("Unknown plot type")
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
      raise RuntimeError("Unknown plot mode "+mode)
    axes.plot(X,Y)

  def plot_spectrum(self, concentrations={}, screen_dpi=96, type='fft'):
    n_cols = 1
    super_title = type.upper() + " "
    if len(concentrations) > 0:
      n_cols = 2
    else:
      super_title += "-".join(self.metabolites[0]) + ' '
    super_title += self.source + ' ' + self.pulse_sequence.upper() + ' ' + self.acquisition + f" @ {self.omega:.2f} Hz"
    if self.linewidth != None:
      super_title += " Linewidth: " + str(self.linewidth)
    if self.noise != None:
      if self.noise[0] == "adc" and self.noise[1] == "normal":
        super_title += f" - ADC Noise N({self.noise[2]},{self.noise[3]})"
      else:
        raise RuntimeError("Unknown noise model")

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
    metabolites = spectra[next(iter(spectra))].metabolites # Metabolites have to be the same for all spectra

    super_title = type.upper() + " "
    if len(concentrations) > 0:
      n_cols +=1
    else:
      super_title += "-".join(metabolites[0]) + " "
    source = set([spectra[a].source for a in spectra])
    pulse_sequence = set([spectra[a].pulse_sequence for a in spectra])
    omega = set([spectra[a].omega for a in spectra])
    linewidth = set([spectra[a].linewidth for a in spectra])
    noise = set([spectra[a].noise for a in spectra])
    if len(source) != 1 or len(pulse_sequence) != 1 or len(omega) != 1 or len(linewidth) != 1 or \
       len(noise) != 1:
      raise RuntimeError("Spectra differ in more than acquisition")

    omega = next(iter(omega))
    super_title += next(iter(source)) + ' ' + next(iter(pulse_sequence)).upper() + ' ' + \
                   f" @ {omega:.2f} Hz"
    linewidth = next(iter(linewidth))
    if linewidth != None:
      super_title += " Linewidth: " + str(linewidth)
    noise = next(iter(noise))
    if noise is not None:
      if noise[0] == "adc" and noise[1] == "normal":
        super_title += f" - ADC Noise N({noise[2]},{noise[3]})"
      else:
        raise RuntimeError("Unknown noise model")

    figure, axes = plt.subplots(4, n_cols, dpi=screen_dpi)
    if len(axes.shape) == 1:
      axes = np.reshape(axes, (4, 1))

    plt.suptitle(super_title)
    col = 0
    for a in sorted(spectra):
      axes[0,col].set_title(a.upper())
      spectra[a].plot(axes[0,col], type=type, mode='magnitude')
      axes[0,col].set_xlabel("")
      if col > 0:
        axes[0,col].sharey(axes[0,0])
        axes[0,col].sharex(axes[0,0])
      spectra[a].plot(axes[1,col], type=type, mode='phase')
      axes[1,col].set_xlabel("")
      if col > 0:
        axes[1,col].sharey(axes[1,0])
      axes[1,col].sharex(axes[0,0])
      spectra[a].plot(axes[2,col], type=type, mode='real')
      axes[2,col].set_xlabel("")
      if col > 0:
        axes[2,col].sharey(axes[2,0])
      axes[2,col].sharex(axes[0,0])
      spectra[a].plot(axes[3,col], type=type, mode='imaginary')
      if col > 0:
        axes[3,col].sharey(axes[3,0])
      axes[3,col].sharex(axes[0,0])
      col += 1

    if n_cols > col:
      axes[3,n_cols-1].set_xticks([])
      ax = plt.subplot(1, n_cols, n_cols)
      plt.title('Concentrations')
      cn = [n for n in concentrations.keys()]
      cv = [concentrations[v] for v in cn]
      ax.bar(np.linspace(0, len(concentrations) - 1, len(concentrations)), cv, align='center')
      ax.set_xticks(np.arange(len(metabolites)),molecules.short_name(cn))
      ax.set_xticklabels(molecules.short_name(cn))

    # Remove tick labels from inner axes and fix tick labels
    for ax in figure.axes:
      try:
        ax.label_outer()
      except:
        pass

    return figure

  @staticmethod
  def comb(f1,s1,f2,s2,id,acq):
    # Weighted addition of two spectra
    if s1.pulse_sequence != s2.pulse_sequence:
      raise RuntimeError("Combing spectra from different pulse sequences")
    if np.abs(s1.omega - s2.omega) >= 1e-8:
      raise RuntimeError("Combing spectra with different omega")
    if s1.source != s2.source:
      raise RuntimeError("Combing spectra from different sources")
    metabolites = s1.metabolites
    for m in s2.metabolites:
      if m not in metabolites:
        metabolites.append(m)
    metabolites.sort()
    if s1.linewidth != s2.linewidth and np.abs(s1.linewidth - s2.linewidth) >= 1e-8:
      raise RuntimeError("Combing spectra with different linewidths")
    s = Spectrum(id=id,
                 pulse_sequence=s1.pulse_sequence,
                 acquisition=acq,
                 omega=(s1.omega+s2.omega)/2.0,
                 source=s1.source,
                 metabolites=metabolites,
                 linewidth=None if s1.linewidth == None else (s1.linewidth + s2.linewidth)/2.0)
    if s1.center_ppm != s2.center_ppm:
      raise RuntimeError("Combining spectra with different center_ppm")
    if s1.b0_shift_ppm != s2.b0_shift_ppm:
      raise RuntimeError("Combining spectra with different b0_shift_ppm")
    if s1.sample_rate != s2.sample_rate:
      raise RuntimeError("Combining spectra with different sample rates")
    if s1.noise != s2.noise:
      raise RuntimeError("Combining spectra with different added noise")
    fft1, _ = s1.get_f()
    fft2, _ = s2.get_f()
    s.set_f(f1*fft1 + f2*fft2, s1.sample_rate, center_ppm = s1.center_ppm, b0_shift_ppm = s1.b0_shift_ppm)
    s.noise = s1.noise
    return s

  @staticmethod
  def combs(fs,ss,id,acq):
    # Weighted sum of spectra
    for n in range(1,len(ss)):
      if ss[0].pulse_sequence != ss[n].pulse_sequence:
        raise RuntimeError("Combing spectra from different pulse sequences")
      if ss[0].source != ss[n].source:
        raise RuntimeError("Combing spectra from different sources")
    avg_omega = np.mean([s.omega for s in ss])
    if ss[0].linewidth == None:
      avg_linewidth = None
    else:
      avg_linewidth = np.mean([s.linewidth for s in ss])
    all_metabolites = []
    for n in range(len(ss)):
      if np.abs(ss[n].omega - avg_omega) >= 1e-8:
        raise RuntimeError("Combing spectra with different omega")
      if (avg_linewidth == None and ss[n].linewidth != None) or \
         (avg_linewidth != None and ss[n].linewidth == None) or \
         (avg_linewidth != None and np.abs(ss[n].linewidth - avg_linewidth) >= 1e-8):
        raise RuntimeError("Combing spectra with different linewidths")
      for m in ss[n].metabolites:
        if m not in all_metabolites:
          all_metabolites.append(m)
    all_metabolites.sort()
    s = Spectrum(id=id,
                 pulse_sequence=ss[0].pulse_sequence,
                 acquisition=acq,
                 omega=avg_omega,
                 source=ss[0].source,
                 metabolites=all_metabolites,
                 linewidth=avg_linewidth)
    fft = fs[0] * ss[0].get_f()[0]
    for n in range(1,len(ss)):
      if ss[0].b0_shift_ppm != ss[n].b0_shift_ppm:
        raise RuntimeError("Combining spectra with different b0_shift_ppm")
      if ss[0].sample_rate != ss[n].sample_rate:
        raise RuntimeError("Combining spectra with different sample rates")
      if ss[0].noise != ss[n].noise:
        raise RuntimeError("Combining spectra with different added noise")
      if ss[0].center_ppm != ss[n].center_ppm:
        # We need to frequency shift the FFT signal by shifting it in the time domain
        # x'[n] = exp(i\pi df/(Fs/2) n) x[n]
        # where x' is the new and x the original signal; Fs the sample rate and df the shift,
        # determined by the difference in center_ppm, in Hz.
        df = (ss[n].center_ppm - ss[0].center_ppm) * ss[n].omega
        D = 2j*np.pi * df / ss[n].sample_rate
        x = ss[n].get_t()[0]
        x = np.exp(D * np.arange(0,len(x))) * x
        F = npfft.fftshift(npfft.fft(x))
      else:
        F = ss[n].get_f()[0]
      fft +=  fs[n] * F
    s.set_f(fft, ss[0].sample_rate, center_ppm = ss[0].center_ppm, b0_shift_ppm = ss[0].b0_shift_ppm)
    s.noise = ss[0].noise
    return s

  @staticmethod
  def correct_b0_multi(spectra):
    # B0 correction across multiple acquisitions
    for a in spectra:
      if spectra[a].pulse_sequence != "megapress":
        raise RuntimeError("Multi-b0-correction only for megapress")
    b0_shift, _ = spectra['edit_off'].correct_b0()
    if b0_shift == None:
      b0_shift = 0.0 # No shift as no peak found
    for a in spectra:
      spectra[a].correct_b0(b0_shift)
    return b0_shift

  @staticmethod
  def load_fida(fida_file,id,source,su=False):
    fida_data = loadmat(fida_file)
    if 'linewidth' in fida_data:
      lw = float(fida_data['linewidth'][0][0])
    else:
      lw = None
    s = Spectrum(id=id,
                 pulse_sequence='megapress',
                 acquisition='edit_on' if fida_data['edit'][0][0] != 0 else 'edit_off',
                 omega=float(fida_data['omega'][0][0]),
                 source=source,
                 metabolites=[molecules.short_name(str(fida_data['m_name'][0]))],
                 linewidth=lw)
    # Time signal produced by fid-a seems mirrored, so need to take the complex conjugate
    if su:
      # Read SU-3TSkyra in a somewhat different format from actual fid-a
      # Specifically we've got an explicit center_ppm value from the file
      if 'scale_formate_peak' in fida_data:
        if 'scale_dss_peak' in fida_data or 'scaling' in fida_data:
          raise RuntimeError("Ambiguous scaling factors")
        scale = fida_data['scale_formate_peak'][0][0]
      elif 'scale_dss_peak' in fida_data:
        if 'scaling' in fida_data:
          raise RuntimeError("Ambiguous scaling factors")
        scale = fida_data['scale_dss_peak'][0][0]
      elif 'scaling' in fida_data:
        scale = fida_data['scaling'][0][0]
      else:
        scale = 1.0
      s.set_t(np.conjugate(np.array(fida_data['fid']).flatten())/scale,
              1/(np.abs(fida_data['t'][0][0] - fida_data['t'][0][1])),
              center_ppm = -fida_data['center_ppm'][0][0])
    else:
      # Read FID-A
      # Note, fid and fft are not on the same scale (due to combine)
      #s.set_t(np.conjugate(np.array(fida_data['fid']).flatten()),
      #        1/(np.abs(fida_data['t'][0][0] - fida_data['t'][0][1])),
      #        center_ppm = -np.median(fida_data['nu']))
      s.set_f(np.conjugate(np.array(fida_data['fft']).flatten()),
              1/(np.abs(fida_data['t'][0][0] - fida_data['t'][0][1])),
              center_ppm = -np.median(fida_data['nu']))
    return s

  @staticmethod
  def load_pygamma(pygamma_dir, search_path, metabolite, pulse_sequence, omega, linewidth,
                   npts, dt, simulate=True):
    for spath in [pygamma_dir, *search_path]:
      cache_dir = os.path.join(spath,
                               pulse_sequence + '_' + str(omega) + '_' +
                               str(linewidth) + '_' + str(npts) + '_' + str(dt))
      filename = os.path.join(cache_dir, metabolite+".json")
      if os.path.exists(filename):
        break
    if not os.path.exists(filename):
      if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
      from mrsnet.simulators.pygamma.pygamma_simulator import pygamma_spectra_sim
      pygamma_spectra_sim(metabolite, omega, pulse_sequence,
                          linewidth, cache_dir, npts, dt)
    specs = []
    with open(filename, 'rb') as load_file:
      for raw in json.load(load_file):
        if pulse_sequence == 'megapress':
          if raw["count"] == 0:
            acq = 'edit_off'
          elif raw["count"] == 1:
            acq = 'edit_on'
          else:
            raise RuntimeError('More than 2 mx objects for megapress? Something is wrong here.')
        s = Spectrum(id=filename, source='pygamma',
                     pulse_sequence=pulse_sequence,
                     metabolites=[molecules.short_name(metabolite)],
                     acquisition=acq,
                     omega=omega,
                     linewidth=linewidth)
        # Time signal produced by pygamma seems mirrored, so need to take the complex conjugate
        s.set_t(np.array(raw["adc_re"]) - 1j * np.array(raw["adc_im"]),
                1/dt)
        specs.append(s)
    return specs

  @staticmethod
  def load_lcm(basis_file, acquisition, req_omega, req_metabolites):
    # Load lcmodel basis
    # http://s-provencher.com/pub/LCModel/manual/manual.pdf
    if not os.path.exists(basis_file):
      raise RuntimeError('Basis file does not exist: ' + basis_file)
    specs = []
    with open(basis_file) as file:
      line_buffer = []
      metadata = {}
      area = ""
      parse = True
      while parse:
        line = file.readline().strip()
        if len(line) == 0: # Process last block, then quit
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
              raise RuntimeError('LCModel basis set (%.2fT) is more than 0.2T different to prescibed '
                                 'omega (%.2fT).' % (metadata['SEQPAR']['HZPPPM']/molecules.GYROMAGNETIC_RATIO,
                                                  req_omega/molecules.GYROMAGNETIC_RATIO))
            if metadata['SEQPAR']['SEQ'] != "MEGA-": # not megapress
              raise RuntimeError('Unrecognised LCM pulse sequence: ' + metadata['SEQ'])
            try:
              metabolite = molecules.short_name(metadata["BASIS"]["METABO"])
            except:
              metabolite = "Unknown"
            if metabolite in req_metabolites:
              # All lcmodel spectra are stored as fourier transforms, so we convert them back to the ADC
              nums = " ".join(line_buffer).split()
              if len(nums) % 2 != 0:
                raise RuntimeError('Uneven fft number, the real/imag switching does not work here or the file has been loaded in wrong!')
              fft = []
              for ii in range(0, len(nums), 2):
                fft.append(float(nums[ii]) + 1j*float(nums[ii + 1]))
              ishift = -metadata["BASIS"]["ISHIFT"]
              fft = np.roll(fft,ishift)
              if ishift > 0:
                fft[:ishift] = 0.0
              elif ishift < 0:
                fft[fft.shape[0]-ishift:] = 0.0
              # We have ON = OFF + 2*diff
              fft_scale = metadata["BASIS"]["TRAMP"]*100.0/(metadata["BASIS"]["CONC"]*metadata["BASIS"]["VOLUME"])
              adc = npfft.ifft(np.array(fft,dtype=np.complex64)) * fft_scale
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
                           linewidth=None)
              s.set_f(npfft.fftshift(fft * fft_scale),1.0/metadata["BASIS1"]['BADELT'],center_ppm=center_ppm)
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
  def load_dicom(file, concentrations=None, metabolites=None, verbose=0):
    if not os.path.exists(file):
        raise RuntimeError('Dicom file does not exist: ' + file)
    from mrsnet.qdicom.read_dicom_siemens import read_dicom
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
    if pulse_sequence in ['svs_edit', 'svs_ed', 'eja_svs_mpress', 'megapress']:
      pulse_sequence = 'megapress'
    else:
      raise RuntimeError(f"{file} - Unrecognised dicom pulse sequence: {pulse_sequence}")
    # Acquisition
    if pulse_sequence == 'megapress':
      if 'EDIT_OFF' in file:
        acquisition = 'edit_off'
      elif 'EDIT_ON' in file:
        acquisition = 'edit_on'
      elif 'DIFF' in file:
        acquisition = 'difference'
      else:
        raise RuntimeError('Loaded dicom file for MEGA-PRESS, but acquisition cannot be determined.\n'
                           'Add "EDIT_OFF", "EDIT_ON" or "DIFF" into the filepath anywhere.')
    else:
      raise RuntimeError(f"Pulse sequence {pulse_sequence} not supported")

    fn = os.path.abspath(file).split(os.sep)
    if len(fn) > 4:
      fn = fn[-4:]
    fn = fn[:-1]
    idl = "/".join(fn)
    pid = dicom[TAG_PATIENT_ID].value
    if len(pid) < 1:
      idl += "/"+pid

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
    spec = Spectrum(id=idl,
                    pulse_sequence=pulse_sequence,
                    acquisition=acquisition,
                    omega=omega,
                    source='dicom',
                    metabolites=metabolites,
                    linewidth=None) # Unknown
    if Cfg.val['filter_dicom'] != None:
      # Handle spectral leakage if requested via Cfg (possibly not the best idea; leave it to the NN)
      if verbose > 4:
        fig, axs = plt.subplot(1,2)
        axs[0].plot(data)
        axs[0].set_title("Dicom Data")
      filter_length = (int(Cfg.val['filter_dicom_duration']/dt)//2)*2
      filterv = np.zeros(len(data))
      if Cfg.val['filter_dicom'] == 'hamming':
        filterv[0:filter_length//2] = np.hamming(filter_length)[filter_length//2:]
      elif Cfg.val['filter_dicom'] == 'hanning':
        filterv[0:filter_length//2] = np.hanning(filter_length)[filter_length//2:]
      elif Cfg.val['filter_dicom'] == 'kaiser':
        filterv[0:filter_length//2] = np.kaiser(filter_length, Cfg.val['filter_dicom_kaiser'])[filter_length//2:]
      else:
        raise RuntimeError("Unknown dicom filter")
      data = np.multiply(data,filterv)
      if verbose > 4:
        fig, axs = plt.subplot(1,2)
        axs[0].plot(data)
        axs[0].set_title("Filtered Dicom Data")
    spec.set_t(data,1/dt,center_ppm=-4.7,remove_water_peak=True,phase_correct=True)
    return spec, cs
