# mrsnet/spectrum.py - MRSNet - individual spectrum
#
# SPDX-FileCopyrightText: Copyright (C) 2019 Max Chandler, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2020-2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2022-2024 Zien Ma, PhD student at Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Individual spectrum handling for MRSNet.

This module provides the Spectrum class for representing and manipulating
individual MRS spectra, including loading from various formats, signal
processing operations, and visualization.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.io import loadmat

from mrsnet import molecules
from mrsnet.cfg import Cfg

npfft = getattr(__import__(Cfg.val['npfft_module'][0], fromlist=[Cfg.val['npfft_module'][1]]),
                Cfg.val['npfft_module'][1])

class Spectrum:
  """Individual MRS spectrum representation and manipulation.

  This class represents a single MRS spectrum with associated metadata
  and provides methods for signal processing, visualization, and data
  export/import.

  Attributes
  ----------
      id (str): Unique identifier for the spectrum
      pulse_sequence (str): Pulse sequence type (e.g., 'megapress')
      acquisition (str): Acquisition type (e.g., 'edit_off', 'difference')
      omega (float): Larmor frequency in Hz
      source (str, optional): Source of the spectrum data
      metabolites (list): List of metabolite names in this spectrum
      linewidth (float, optional): Linewidth parameter
      sample_rate (float): Sample rate in Hz
      fft (numpy.ndarray): FFT data of the spectrum
      scale (float): Scaling factor for the spectrum
      center_ppm (float): Center PPM value
      b0_shift_ppm (float): B0 shift in PPM
      noise (dict, optional): Noise information if added
  """

  def __init__(self, id, pulse_sequence, acquisition, omega,
               source=None, metabolites=None, linewidth=None):
    """Initialize a new spectrum.

    Parameters
    ----------
        id (str): Unique identifier for the spectrum
        pulse_sequence (str): Pulse sequence type
        acquisition (str): Acquisition type
        omega (float): Larmor frequency in Hz
        source (str, optional): Source of the spectrum data. Defaults to None
        metabolites (list, optional): List of metabolite names. Defaults to None
        linewidth (float, optional): Linewidth parameter. Defaults to None
    """
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
            remove_water_peak=False, phase_correct=False, force_phase_correct=None):
    """Set spectrum data from frequency domain (FFT).

    Parameters
    ----------
        fft (array-like): FFT data of the spectrum
        sample_rate (float): Sample rate in Hz
        center_ppm (float, optional): Center PPM value. Defaults to 0
        b0_shift_ppm (float, optional): B0 shift in PPM. Defaults to 0
        scale (float, optional): Scaling factor. Defaults to 1.0
        remove_water_peak (bool, optional): Whether to remove water peak. Defaults to False
        phase_correct (bool, optional): Whether to apply phase correction. Defaults to False
        force_phase_correct (str, optional): Force specific phase correction method. Defaults to None
    """
    self.fft = np.asarray(fft)
    self._set(sample_rate, center_ppm, b0_shift_ppm, scale, remove_water_peak, phase_correct, force_phase_correct)

  def set_t(self, adc, sample_rate, center_ppm=0, b0_shift_ppm=0, scale=1.0,
            remove_water_peak=False, phase_correct=False, force_phase_correct=None):
    """Set spectrum data from time domain (ADC).

    Converts time domain data to frequency domain using FFT and sets up the spectrum.

    Parameters
    ----------
        adc (array-like): Time domain ADC data
        sample_rate (float): Sample rate in Hz
        center_ppm (float, optional): Center PPM value. Defaults to 0
        b0_shift_ppm (float, optional): B0 shift in PPM. Defaults to 0
        scale (float, optional): Scaling factor. Defaults to 1.0
        remove_water_peak (bool, optional): Whether to remove water peak. Defaults to False
        phase_correct (bool, optional): Whether to apply phase correction. Defaults to False
        force_phase_correct (str, optional): Force specific phase correction method. Defaults to None
    """
    self.fft = npfft.fftshift(npfft.fft(adc))
    self._set(sample_rate, center_ppm, b0_shift_ppm, scale, remove_water_peak, phase_correct, force_phase_correct)

  def _set(self, sample_rate, center_ppm, b0_shift_ppm, scale, remove_water_peak, phase_correct, force_phase_correct=None):
    """Set spectrum parameters and apply processing.

    Parameters
    ----------
        sample_rate (float): Sampling rate in Hz
        center_ppm (float): Center frequency in ppm
        b0_shift_ppm (float): B0 field shift in ppm
        scale (float): Scaling factor
        remove_water_peak (bool): Whether to remove water peak
        phase_correct (str): Phase correction method
        force_phase_correct (str, optional): Force specific phase correction method
    """
    self.sample_rate = sample_rate
    self.center_ppm = center_ppm
    self.b0_shift_ppm = b0_shift_ppm
    self.scale = scale
    if (phase_correct and Cfg.val['phase_correct'] is not None) or (force_phase_correct is not None):
      # Only phase correct if also configured or it is forced (specifically used by load_pygamma)
      if Cfg.dev('spectrum_set_phase_correct'):
        fig, axs = plt.subplots(1,2)
        freq, nu = self.get_f()
        axs[0].plot(nu, np.real(freq), color='r')
        axs[0].plot(nu, np.imag(freq), color='g')
        axs[0].set_title("Raw Dicom Data")
      if Cfg.val['phase_correct'] == 'acme' or force_phase_correct == 'acme':
        self._phase_correct_acme()
      elif Cfg.val['phase_correct'] == 'ernst' or force_phase_correct == 'ernst':
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
    """Get frequency domain data.

    Returns
    -------
        tuple: (fft_data, ppm_axis) where:
            - fft_data: Scaled FFT data
            - ppm_axis: PPM axis values
    """
    return self.fft*self.scale, \
           np.linspace(-1, 1, len(self.fft)) * self.sample_rate/2.0 / self.omega + self.center_ppm + self.b0_shift_ppm

  def get_t(self):
    """Get time domain data.

    Returns
    -------
        tuple: (adc_data, time_axis) where:
            - adc_data: Time domain ADC data
            - time_axis: Time axis values in seconds
    """
    return npfft.ifft(npfft.ifftshift(self.fft * self.scale)), \
           np.arange(0, len(self.fft), 1) / self.sample_rate

  def correct_b0(self, shift=None):
    """Apply B0 correction to the spectrum.

    Either finds a reference peak automatically or applies a given shift.

    Parameters
    ----------
        shift (float, optional): B0 shift in PPM. If None, automatically finds
                                reference peak. Defaults to None

    Returns
    -------
        tuple: (shift_applied, peak_value) where:
            - shift_applied: The B0 shift applied in PPM
            - peak_value: Value of the reference peak used
    """
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
    """Find peak location in FFT spectrum.

    Parameters
    ----------
        location (float): Expected peak location in ppm
        ppm_range (float): Search range around location in ppm

    Returns
    -------
        tuple: (peak_location, peak_value) or (None, None) if not found
    """
    fft, nu = self.get_f()
    fft_abs = -np.abs(fft)
    cut_off = np.median(fft_abs)
    # finds the highest peak from location +- ppm_range
    peak_idxs = fft_abs.argsort()
    for idx in peak_idxs:
      if abs(location - nu[idx]) < ppm_range:
        # BG Quinn, EJ Hannan. The Estimation and Tracking of Frequency, 2001.
        # https://dspguru.com/dsp/howtos/how-to-interpolate-fft-peak/
        if Cfg.val['fft_peak_location_estimator'] is None or idx < 1 or idx >= len(nu) - 1:
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
    """Apply Ernst angle phase correction.

    Corrects the phase of the spectrum using the Ernst angle method.
    Reference: R Ernst. Numerical Hilbert transform and automatic phase correction in
    magnetic resonance spectroscopy. J Magn Res 1(1):7-26, 1969.
    """
    # R Ernst. Numerical Hilbert transform and automatic phase correction in
    # magnetic resonance spectroscopy. J Magn Res 1(1):7-26, 1969.
    # https://www.sciencedirect.com/science/article/abs/pii/0022236469900031

    def err(para):
      """Error function for Ernst phase correction optimization.

      Parameters
      ----------
          para: Parameter object containing phase values

      Returns
      -------
          float: Error value for optimization
      """
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
    shift = val['phi0'] + val['phi1'] * (np.linspace(-1, 1, len(self.fft)) * self.sample_rate/2.0)
    self.fft *= np.exp(1j*shift)
    idx = np.argmax(np.abs(np.real(self.fft)))
    max_s_fft = np.real(self.fft[idx])
    if max_s_fft * max_fft < 0.0: # avoid flipping the phase / turn real positive, if it should be negative
      self.fft *= np.exp(1j*np.pi)

  def _phase_correct_acme(self):
    """Apply ACME phase correction.

    Automated phase Correction based on Minimization of Entropy.
    Reference: L Chen, Z Weng, LY Goh, M Garland. An efficient algorithm for automatic
    phase correction of NMR spectra based on entropy minimization. J Magn
    Res 158:164-168, 2002.
    """
    # Automated phase Correction based on Minimization of Entropy
    # L Chen, Z Weng, LY Goh, M Garland. An efficient algorithm for automatic
    # phase correction of NMR spectra based on entropy minimization. J Magn
    # Res 158:164-168, 2002.

    def entropy(para):
      """Entropy function for ACME phase correction optimization.

      Parameters
      ----------
          para: Parameter object containing phase values

      Returns
      -------
          float: Entropy value for optimization
      """
      val = para.valuesdict()
      # Adjust phase
      shift = val['phc0'] + val['phc1'] * (np.linspace(-1, 1, len(self.fft)) * self.sample_rate/2.0)
      s_fft = self.fft * np.exp(1j*shift)
      # Target function
      r = np.real(s_fft)
      r1 = np.abs(r[1:] - r[:-1])
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
    """Resample FFT to prescribed frequency bins via zero filling.

    Parameters
    ----------
        high_ppm (float, optional): Upper PPM bound. Defaults to -4.5
        low_ppm (float, optional): Lower PPM bound. Defaults to -1
        npts (int, optional): Number of frequency points. Defaults to 2048
    """
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
    """Remove water peak from FFT spectrum.

    Finds the water peak and sets the range centered around it to the median signal.
    """
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
    """Add normal ADC noise to the spectrum.

    Parameters
    ----------
        mu (float, optional): Mean of noise distribution. Defaults to 0
        sigma (float, optional): Standard deviation of noise distribution. Defaults to 0
    """
    if self.noise is not None:
      raise RuntimeError("Adding noise twice is not advised")
    self.noise = ("adc", "normal", mu, sigma)
    adc, _ = self.get_t()
    m = np.max(np.abs(adc))
    noise = (     np.random.normal(mu, sigma, len(adc)) +
             1j * np.random.normal(mu, sigma, len(adc))) * m / self.scale
    self.fft += npfft.fftshift(npfft.fft(noise))

  def plot(self, axes, type='fft', mode='magnitude'):
    """Plot spectrum on given axes.

    Parameters
    ----------
        axes: Matplotlib axes object
        type (str, optional): Plot type ('fft' or 'time'). Defaults to 'fft'
        mode (str, optional): Plot mode ('magnitude', 'real', 'imaginary'). Defaults to 'magnitude'
    """
    if type == 'time':
      Y, X = self.get_t()  # noqa: N806
      axes.set_xlabel('Time (s)')
    elif type == 'fft':
      Y, X = self.rescale_fft()  # noqa: N806
      axes.set_xlabel('Frequency (ppm)')
      # Note, we are using negative frequencies
      axes.xaxis.set_major_formatter(FuncFormatter(lambda x_val, tick_pos: f"{np.abs(x_val):.8g}"))
    else:
      raise RuntimeError("Unknown plot type")
    if mode == 'magnitude':
      Y = np.abs(Y)  # noqa: N806
      axes.set_ylabel('Magn.')
    elif mode == 'phase':
      Y = np.angle(Y)  # noqa: N806
      axes.set_ylabel('Phase')
    elif mode == 'real':
      Y = np.real(Y)  # noqa: N806
      axes.set_ylabel('Re')
    elif mode == 'imaginary':
      Y = np.imag(Y)  # noqa: N806
      axes.set_ylabel('Im')
    else:
      raise RuntimeError("Unknown plot mode "+mode)
    axes.plot(X,Y)

  def plot_spectrum(self, concentrations={}, screen_dpi=96, type='fft'):
    """Plot spectrum with optional concentration information.

    Parameters
    ----------
        concentrations (dict, optional): Dictionary of metabolite concentrations. Defaults to {}
        screen_dpi (int, optional): Screen DPI for display. Defaults to 96
        type (str, optional): Plot type ('fft' or 'time'). Defaults to 'fft'

    Returns
    -------
        matplotlib.figure.Figure: The created figure
    """
    n_cols = 1
    super_title = type.upper() + " "
    if len(concentrations) > 0:
      n_cols = 2
    else:
      super_title += "-".join(self.metabolites) + ' '
    super_title += self.source + ' ' + self.pulse_sequence.upper() + ' ' + self.acquisition + f" @ {self.omega:.2f} MHz"
    if self.linewidth is not None:
      super_title += " Linewidth: " + str(self.linewidth)
    if self.noise is not None:
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
      cn = list(concentrations.keys())
      cv = [concentrations[v] for v in cn]
      ax.bar(np.linspace(0, len(concentrations) - 1, len(concentrations)), cv)
      ax.set_xticks(np.arange(len(self.metabolites)))
      ax.set_xticklabels(molecules.short_name(cn))

    return figure

  def save_json(self, filename, version=1):
    """Save spectrum data to JSON file.

    Parameters
    ----------
        filename (str): Output filename
        version (int, optional): JSON format version. Defaults to 1
    """
    if version == 1:
      data = {
        'id': self.id,
        'pulse_sequence': self.pulse_sequence,
        'acquisition': self.acquisition,
        'omega': self.omega,
        'source': self.source,
        'metabolites': self.metabolites,
        'linewidth': self.linewidth,
        'sample_rate': self.sample_rate,
        'fft': [(np.real(v),np.imag(v)) for v in self.fft],
        'scale': self.scale,
        'center_ppm': self.center_ppm,
        'b0_shift_ppm': self.b0_shift_ppm,
        'noise': self.noise,
        'mrsnet_json_format': 1
      }
    else:
      raise RuntimeError(f"Unknown json format version {version}")
    with open(filename, 'w', encoding='utf-8') as f:
      json.dump(data, f, ensure_ascii=False, indent=2)

  @staticmethod
  def plot_full_spectrum(spectra, concentrations={}, screen_dpi=96, type='fft'):
    """Plot multiple spectra in a single figure.

    Parameters
    ----------
        spectra (dict): Dictionary of spectrum objects
        concentrations (dict, optional): Dictionary of metabolite concentrations. Defaults to {}
        screen_dpi (int, optional): Screen DPI for display. Defaults to 96
        type (str, optional): Plot type ('fft' or 'time'). Defaults to 'fft'

    Returns
    -------
        matplotlib.figure.Figure: The created figure
    """
    n_cols = len(spectra)
    metabolites = spectra[next(iter(spectra))].metabolites # Metabolites have to be the same for all spectra

    super_title = type.upper() + " "
    if len(concentrations) > 0:
      n_cols +=1
    else:
      super_title += "-".join(metabolites) + " "
    source = {spectra[a].source for a in spectra}
    pulse_sequence = {spectra[a].pulse_sequence for a in spectra}
    omega = {spectra[a].omega for a in spectra}
    linewidth = {spectra[a].linewidth for a in spectra}
    noise = {spectra[a].noise for a in spectra}
    if len(source) != 1 or len(pulse_sequence) != 1 or len(omega) != 1 or len(linewidth) != 1 or \
       len(noise) != 1:
      raise RuntimeError("Spectra differ in more than acquisition")

    omega = next(iter(omega))
    super_title += next(iter(source)) + ' ' + next(iter(pulse_sequence)).upper() + ' ' + \
                   f" @ {omega:.2f} MHz"
    linewidth = next(iter(linewidth))
    if linewidth is not None:
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
      cn = list(concentrations.keys())
      cv = [concentrations[v] for v in cn]
      ax.bar(np.linspace(0, len(concentrations) - 1, len(concentrations)), cv, align='center')
      ax.set_xticks(np.arange(len(metabolites)),molecules.short_name(cn))
      ax.set_xticklabels(molecules.short_name(cn))

    # Remove tick labels from inner axes and fix tick labels
    for ax in figure.axes:
      try:
        ax.label_outer()
      except Exception:  # noqa: S110
        pass

    return figure

  @staticmethod
  def comb(f1,s1,f2,s2,id,acq):
    """Combine two spectra with weighted addition.

    Parameters
    ----------
        f1 (float): Weight for first spectrum
        s1 (Spectrum): First spectrum
        f2 (float): Weight for second spectrum
        s2 (Spectrum): Second spectrum
        id (str): ID for the combined spectrum
        acq (str): Acquisition type for the combined spectrum

    Returns
    -------
        Spectrum: Combined spectrum

    Raises
    ------
        RuntimeError: If spectra have incompatible properties
    """
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
                 linewidth=None if s1.linewidth is None else (s1.linewidth + s2.linewidth)/2.0)
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
  def combs(fs,ss,id,acq, allow_mixed_linewidths=False):
    """Combine multiple spectra with weighted sum.

    Parameters
    ----------
        fs (list): List of weights for spectra
        ss (list): List of Spectrum objects
        id (str): ID for the combined spectrum
        acq (str): Acquisition type for the combined spectrum

    Returns
    -------
        Spectrum: Combined spectrum

    Raises
    ------
        RuntimeError: If spectra have incompatible properties
    """
    # Weighted sum of spectra
    for n in range(1,len(ss)):
      if ss[0].pulse_sequence != ss[n].pulse_sequence:
        raise RuntimeError("Combing spectra from different pulse sequences")
      if ss[0].source != ss[n].source:
        raise RuntimeError("Combing spectra from different sources")
    avg_omega = np.mean([s.omega for s in ss])
    if ss[0].linewidth is None:
      avg_linewidth = None
    else:
      avg_linewidth = np.mean([s.linewidth for s in ss])
    all_metabolites = []
    for n in range(len(ss)):
      if np.abs(ss[n].omega - avg_omega) >= 1e-8:
        raise RuntimeError("Combing spectra with different omega")
      if not allow_mixed_linewidths:
        if (avg_linewidth is None and ss[n].linewidth is not None) or \
           (avg_linewidth is not None and ss[n].linewidth is None) or \
           (avg_linewidth is not None and np.abs(ss[n].linewidth - avg_linewidth) >= 1e-8):
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
      if ss[0].sample_rate != ss[n].sample_rate:
        raise RuntimeError("Combining spectra with different sample rates")
      if ss[0].noise != ss[n].noise:
        raise RuntimeError("Combining spectra with different added noise")
      # Align center_ppm and (optionally) b0_shift_ppm via time-domain frequency shift
      need_shift = (ss[0].center_ppm != ss[n].center_ppm)
      df = 0.0
      if need_shift:
        df += (ss[n].center_ppm - ss[0].center_ppm) * ss[n].omega
      if ss[0].b0_shift_ppm != ss[n].b0_shift_ppm:
        if not allow_mixed_linewidths:
          raise RuntimeError("Combining spectra with different b0_shift_ppm")
        # incorporate b0 shift difference into frequency shift (in Hz)
        df += (ss[n].b0_shift_ppm - ss[0].b0_shift_ppm) * ss[n].omega
        need_shift = True
      if need_shift:
        D = 2j*np.pi * df / ss[n].sample_rate  # noqa: N806
        x = ss[n].get_t()[0]
        x = np.exp(D * np.arange(0,len(x))) * x
        F = npfft.fftshift(npfft.fft(x))  # noqa: N806
      else:
        F = ss[n].get_f()[0]  # noqa: N806
      fft +=  fs[n] * F
    s.set_f(fft, ss[0].sample_rate, center_ppm = ss[0].center_ppm, b0_shift_ppm = ss[0].b0_shift_ppm)
    s.noise = ss[0].noise
    return s

  @staticmethod
  def correct_b0_multi(spectra):
    """Apply B0 correction across multiple acquisitions.

    Parameters
    ----------
        spectra (dict): Dictionary of spectrum objects

    Returns
    -------
        float: B0 shift applied in ppm

    Raises
    ------
        RuntimeError: If pulse sequence is not megapress
    """
    # B0 correction across multiple acquisitions
    for a in spectra:
      if spectra[a].pulse_sequence != "megapress":
        raise RuntimeError("Multi-b0-correction only for megapress")
    b0_shift, _ = spectra['edit_off'].correct_b0()
    if b0_shift is None:
      b0_shift = 0.0 # No shift as no peak found
    for a in spectra:
      spectra[a].correct_b0(b0_shift)
    return b0_shift

  @staticmethod
  def load_fida(fida_file,id,source,su=False):
    """Load spectrum from FID-A MATLAB file.

    Parameters
    ----------
        fida_file (str): Path to FID-A .mat file
        id (str): Spectrum identifier
        source (str): Data source identifier
        su (bool, optional): Use SU-3TSkyra format. Defaults to False

    Returns
    -------
        Spectrum: Loaded spectrum object
    """
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
      # Conjugation due to negative frequencies
      s.set_t(np.conjugate(np.array(fida_data['fid']).flatten())/scale,
              1/(np.abs(fida_data['t'][0][0] - fida_data['t'][0][1])),
              center_ppm = -fida_data['center_ppm'][0][0])
    else:
      # Read FID-A
      # Note, fid and fft are not on the same scale (due to combine)
      #s.set_t(np.conjugate(np.array(fida_data['fid']).flatten()),
      #        1/(np.abs(fida_data['t'][0][0] - fida_data['t'][0][1])),
      #        center_ppm = -np.median(fida_data['nu']))
      # Conjugation due to negative frequencies
      s.set_f(np.conjugate(np.array(fida_data['fft']).flatten()),
              1/(np.abs(fida_data['t'][0] - fida_data['t'][1])),
              center_ppm = -np.median(fida_data['nu']))
    return s

  @staticmethod
  def load_pygamma(pygamma_dir, search_path, metabolite, pulse_sequence, omega, linewidth,
                   npts, dt, simulate=True, force_phase_correct="acme"):
    """Load spectrum from PyGamma simulation.

    Parameters
    ----------
        pygamma_dir (str): Directory containing PyGamma data
        search_path (list): Additional search paths
        metabolite (str): Metabolite name
        pulse_sequence (str): Pulse sequence name
        omega (float): Larmor frequency
        linewidth (float): Spectral linewidth
        npts (int): Number of data points
        dt (float): Dwell time
        simulate (bool, optional): Whether to simulate if not cached. Defaults to True
        force_phase_correct (str, optional): Phase correction method. Defaults to "acme"

    Returns
    -------
        list: List of Spectrum objects
        path (str): Directory containing PyGamma spectra loaded
    """
    filename = None
    for spath in [pygamma_dir, *search_path]:
      cache_dir = os.path.join(spath,
                               pulse_sequence + '_' + str(omega) + '_' +
                               str(linewidth) + '_' + str(npts) + '_' + str(dt))
      filename = os.path.join(cache_dir, metabolite+".json")
      if os.path.exists(filename):
        break
    if filename is None:
      cache_dir = os.path.join(pygamma_dir,
                               pulse_sequence + '_' + str(omega) + '_' +
                               str(linewidth) + '_' + str(npts) + '_' + str(dt))
      if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
      from mrsnet.simulators.pygamma.pygamma_simulator import pygamma_spectra_sim
      pygamma_spectra_sim(metabolite, omega, pulse_sequence, linewidth, cache_dir, npts, dt)
      filename = os.path.join(cache_dir, metabolite+".json")
      if not os.path.exists(filename):
        raise RuntimeError('PyGamma cache file does not exist: ' + filename)
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
        # Conjugation due to negative frequencies
        s.set_t(np.array(raw["adc_re"]) - 1j * np.array(raw["adc_im"]),
                1/dt, force_phase_correct=force_phase_correct)
        specs.append(s)
    return specs, os.path.dirname(cache_dir)

  @staticmethod
  def load_lcm(basis_file, acquisition, req_omega, req_metabolites):
    """Load spectrum from LCModel basis file.

    Parameters
    ----------
        basis_file (str): Path to LCModel basis file
        acquisition (str): Acquisition type
        req_omega (float): Required Larmor frequency
        req_metabolites (list): List of required metabolites

    Returns
    -------
        list: List of Spectrum objects

    Raises
    ------
        RuntimeError: If basis file doesn't exist or has incompatible properties
    """
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
                    except Exception:  # noqa: S110
                      pass
                    v.append(vv)
                else:
                  if '.' in v:
                    v = float(v)
                  else:
                    v = int(v)
              except Exception:  # noqa: S110
                pass
              metadata[area][sl[0].strip()] = v
          elif area == "spectrum":
            if np.abs(metadata['SEQPAR']['HZPPPM'] - req_omega) > (molecules.GYROMAGNETIC_RATIO/5):
              # more than a 0.2T difference, there's an issue
              raise RuntimeError('LCModel basis set (%.2fT) is more than 0.2T different to prescibed '  # noqa: UP031
                                 'omega (%.2fT).' % (metadata['SEQPAR']['HZPPPM']/molecules.GYROMAGNETIC_RATIO,
                                                  req_omega/molecules.GYROMAGNETIC_RATIO))
            if metadata['SEQPAR']['SEQ'] != "MEGA-": # not megapress
              raise RuntimeError('Unrecognised LCM pulse sequence: ' + metadata['SEQ'])
            try:
              metabolite = molecules.short_name(metadata["BASIS"]["METABO"])
            except Exception:
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
              # LCModel DIFF corresponds to the measured ON - OFF spectrum.
              # Hence DIFF already contains approximately twice the edited metabolite contribution (e.g. GABA),
              # and downstream we should consistently use DIFF = ON - OFF without additional rescaling.
              fft_scale = metadata["BASIS"]["TRAMP"]*100.0/(metadata["BASIS"]["CONC"]*metadata["BASIS"]["VOLUME"])
              #adc = npfft.ifft(np.array(fft,dtype=np.complex64)) * fft_scale
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
            raise OSError(f"Unknown section in LCModel basis file {area}")
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
    """Load spectrum from DICOM file.

    Parameters
    ----------
        file (str): Path to DICOM file
        concentrations (str, optional): Path to concentrations JSON file. Defaults to None
        metabolites (list, optional): List of metabolites. Defaults to None
        verbose (int, optional): Verbosity level. Defaults to 0

    Returns
    -------
        tuple: (Spectrum object, concentrations dict)

    Raises
    ------
        RuntimeError: If file doesn't exist or has unsupported format
    """
    if not os.path.exists(file):
        raise RuntimeError('Dicom file does not exist: ' + file)
    import struct

    from mrsnet.qdicom.read_dicom_siemens import read_dicom
    # We assume it's a Siemens dicom spectrum, so do not check this
    dicom, info = read_dicom(file)
    # Read Siemens DICOM spectroscopy data tag (0x7fe1, 0x1010) as a list of complex numbers ReImReIm...; 4 byte floats; little endian
    TAG_SPECTROSCOPY_DATA = (0x7fe1, 0x1010)  # noqa: N806
    TAG_PATIENT_ID = (0x0010, 0x0020)  # noqa: N806
    data = dicom[TAG_SPECTROSCOPY_DATA].value
    data = struct.unpack("<%df" % (len(data) / 4), data)  # noqa: UP031
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
      with open(concentrations) as f:
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
    if Cfg.val['filter_dicom'] is not None:
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

  @staticmethod
  def load_csv(file, concentrations=None, metabolites=None, verbose=0):
    """Load spectrum from CSV file.

    Parameters
    ----------
        file (str): Path to CSV file
        concentrations (str, optional): Path to concentrations JSON file. Defaults to None
        metabolites (list, optional): List of metabolites. Defaults to None
        verbose (int, optional): Verbosity level. Defaults to 0

    Returns
    -------
        tuple: (Spectrum object, concentrations dict)

    Raises
    ------
        RuntimeError: If file doesn't exist or has unsupported format
    """
    if not os.path.exists(file):
        raise RuntimeError('CSV file does not exist: ' + file)
    import csv
    with open(file) as csvfile:
      rows = csv.reader(csvfile)
      for row in rows:
        if row[0] == "WriteSpec2asc-v1":
          pass
        elif row[0] == "ID":
          pid = row[1]
        elif row[0] == "Pulse_sequence":
          pulse_sequence = row[1]
          if pulse_sequence in ["EJA_MPRESS"]:
            pulse_sequence = 'megapress'
          else:
            raise RuntimeError(f"{file} - Unrecognised dicom pulse sequence: {pulse_sequence}")
        elif row[0] == "Transmitter_frequency":
          omega = float(row[1]) / 1000000.
        elif row[0] == "Sampling_frequency":
          dt = 1. / float(row[1])
        elif row[0] == "FID":
          data = np.array(row[1:],dtype=np.float64)
        elif row[0] == "t":
          pass
        elif row[0] == "TE":
          pass
        elif row[0] == "Samples":
          pass
        elif row[0] == "Center_ppm":
          center_ppm = -float(row[1])
        else:
          raise RuntimeError(f"Unknown csv field {row[0]}")

    # Acquisition
    if pulse_sequence == 'megapress':
      if 'EDIT_OFF' in file:
        acquisition = 'edit_off'
      elif 'EDIT_ON' in file:
        acquisition = 'edit_on'
      elif 'DIFF' in file:
        acquisition = 'difference'
      else:
        raise RuntimeError('Loaded csv file for MEGA-PRESS, but acquisition cannot be determined.\n'
                            'Add "EDIT_OFF", "EDIT_ON" or "DIFF" into the filepath anywhere.')
    else:
      raise RuntimeError(f"Pulse sequence {pulse_sequence} not supported")

    fn = os.path.abspath(file).split(os.sep)
    if len(fn) > 4:
      fn = fn[-4:]
    fn = fn[:-1]
    idl = "/".join(fn)
    idl += pid

    cs = np.array([])
    if concentrations is not None:
      ids = file.split('/')
      with open(concentrations) as f:
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
    if Cfg.val['filter_dicom'] is not None:
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
    spec.set_t(data,1/dt,center_ppm=center_ppm,remove_water_peak=True,phase_correct=True)
    return spec, cs

  def estimate_linewidth(self, method='water_peak', verbose=0, min_snr=3.0, max_peaks=3):
    """Estimate linewidth from experimental spectrum.

    For MEGAPRESS spectra, this should ideally be called on edit_off spectra
    which have the best SNR and no editing artifacts.

    Parameters
    ----------
    method : str, optional
        Estimation method: 'water_peak', 'metabolite_peak', 'lorentzian', 'auto'. Defaults to 'water_peak'
    verbose : int, optional
        Verbosity level. Defaults to 0
    min_snr : float, optional
        Minimum SNR threshold for peak detection. Defaults to 3.0
    max_peaks : int, optional
        Maximum number of peaks to use for averaging. Defaults to 3

    Returns
    -------
    float
        Estimated linewidth in Hz, or None if estimation fails

    Raises
    ------
    RuntimeError
        If spectrum data is not available or method is not supported
    """
    if self.fft is None:
      raise RuntimeError("Spectrum data not available for linewidth estimation")

    if verbose > 2:
      print(f"# Estimating linewidth using method: {method} for {self.acquisition} spectrum")

    # Get frequency domain data
    fft_data, nu = self.get_f()
    magnitude = np.abs(fft_data)

    if method == 'water_peak':
      return self._estimate_linewidth_water_peak_robust(magnitude, nu, verbose, min_snr, max_peaks)
    elif method == 'metabolite_peak':
      return self._estimate_linewidth_metabolite_peak_robust(magnitude, nu, verbose, min_snr, max_peaks)
    elif method == 'lorentzian':
      return self._estimate_linewidth_lorentzian(magnitude, nu, verbose, min_snr, max_peaks)
    elif method == 'auto':
      # Try lorentzian, then water peak, then metabolite peak
      lw = self._estimate_linewidth_lorentzian(magnitude, nu, verbose, min_snr, max_peaks)
      if lw is None:
        lw = self._estimate_linewidth_water_peak_robust(magnitude, nu, verbose, min_snr, max_peaks)
      if lw is None:
        lw = self._estimate_linewidth_metabolite_peak_robust(magnitude, nu, verbose, min_snr, max_peaks)
      return lw
    else:
      raise RuntimeError(f"Unknown linewidth estimation method: {method}")

  def _estimate_linewidth_water_peak(self, magnitude, nu, verbose):
    """Estimate linewidth from water peak (4.7 ppm region).

    Parameters
    ----------
    magnitude : array
        Magnitude spectrum data
    nu : array
        Frequency axis in ppm
    verbose : int
        Verbosity level

    Returns
    -------
    float or None
        Estimated linewidth in Hz, or None if water peak not found
    """
    # Look for water peak around 4.7 ppm
    water_region = (nu >= 4.0) & (nu <= 5.5)
    if not np.any(water_region):
      if verbose > 2:
        print("# Water peak region (4.0-5.5 ppm) not found in spectrum")
      return None

    water_magnitude = magnitude[water_region]
    water_nu = nu[water_region]

    # Find peak maximum
    peak_idx = np.argmax(water_magnitude)
    peak_ppm = water_nu[peak_idx]
    peak_magnitude = water_magnitude[peak_idx]

    if verbose > 2:
      print(f"# Water peak found at {peak_ppm:.2f} ppm with magnitude {peak_magnitude:.2e}")

    # Find FWHM
    half_max = peak_magnitude / 2.0

    # Find left and right half-maximum points
    left_idx = None
    right_idx = None

    # Search left from peak
    for i in range(peak_idx - 1, -1, -1):
      if water_magnitude[i] <= half_max:
        left_idx = i
        break

    # Search right from peak
    for i in range(peak_idx + 1, len(water_magnitude)):
      if water_magnitude[i] <= half_max:
        right_idx = i
        break

    if left_idx is None or right_idx is None:
      if verbose > 2:
        print("# Could not determine FWHM of water peak")
      return None

    # Calculate FWHM in ppm
    fwhm_ppm = water_nu[right_idx] - water_nu[left_idx]

    # Convert to Hz
    fwhm_hz = fwhm_ppm * self.omega

    if verbose > 2:
      print(f"# Water peak FWHM: {fwhm_ppm:.3f} ppm ({fwhm_hz:.1f} Hz)")

    return fwhm_hz

  def _estimate_linewidth_metabolite_peak(self, magnitude, nu, verbose):
    """Estimate linewidth from metabolite peaks (NAA, Cr, etc.).

    Parameters
    ----------
    magnitude : array
        Magnitude spectrum data
    nu : array
        Frequency axis in ppm
    verbose : int
        Verbosity level

    Returns
    -------
    float or None
        Estimated linewidth in Hz, or None if suitable peaks not found
    """
    # Look for metabolite peaks in the typical MRS range
    metabolite_region = (nu >= -1.0) & (nu <= 4.5)
    if not np.any(metabolite_region):
      if verbose > 2:
        print("# Metabolite region (-1.0 to 4.5 ppm) not found in spectrum")
      return None

    metabolite_magnitude = magnitude[metabolite_region]
    metabolite_nu = nu[metabolite_region]

    # Find the strongest peak in the metabolite region
    peak_idx = np.argmax(metabolite_magnitude)
    peak_ppm = metabolite_nu[peak_idx]
    peak_magnitude = metabolite_magnitude[peak_idx]

    if verbose > 2:
      print(f"# Strongest metabolite peak found at {peak_ppm:.2f} ppm with magnitude {peak_magnitude:.2e}")

    # Find FWHM
    half_max = peak_magnitude / 2.0

    # Find left and right half-maximum points
    left_idx = None
    right_idx = None

    # Search left from peak
    for i in range(peak_idx - 1, -1, -1):
      if metabolite_magnitude[i] <= half_max:
        left_idx = i
        break

    # Search right from peak
    for i in range(peak_idx + 1, len(metabolite_magnitude)):
      if metabolite_magnitude[i] <= half_max:
        right_idx = i
        break

    if left_idx is None or right_idx is None:
      if verbose > 2:
        print("# Could not determine FWHM of metabolite peak")
      return None

    # Calculate FWHM in ppm
    fwhm_ppm = metabolite_nu[right_idx] - metabolite_nu[left_idx]

    # Convert to Hz
    fwhm_hz = fwhm_ppm * self.omega

    if verbose > 2:
      print(f"# Metabolite peak FWHM: {fwhm_ppm:.3f} ppm ({fwhm_hz:.1f} Hz)")

    return fwhm_hz

  def _estimate_linewidth_water_peak_robust(self, magnitude, nu, verbose, min_snr=3.0, max_peaks=3):
    """Robust estimate linewidth from water peak with improved peak detection and SNR filtering.

    Parameters
    ----------
    magnitude : array
        Magnitude spectrum data
    nu : array
        Frequency axis in ppm
    verbose : int
        Verbosity level
    min_snr : float
        Minimum SNR threshold for peak detection
    max_peaks : int
        Maximum number of peaks to use for averaging

    Returns
    -------
    float or None
        Estimated linewidth in Hz, or None if water peak not found
    """
    # Look for water peak around 4.7 ppm with dynamic range
    water_region = (nu >= 4.0) & (nu <= 5.5)
    if not np.any(water_region):
      if verbose > 2:
        print("# Water peak region (4.0-5.5 ppm) not found in spectrum")
      return None

    water_magnitude = magnitude[water_region]
    water_nu = nu[water_region]

    # Estimate noise level from baseline
    noise_region = (nu >= 8.0) & (nu <= 12.0)  # High ppm noise region
    if np.any(noise_region):
      noise_level = np.std(magnitude[noise_region])
    else:
      noise_level = np.std(water_magnitude) * 0.1  # Fallback estimate

    # Find multiple peaks above SNR threshold
    peaks = self._find_peaks_robust(water_magnitude, water_nu, noise_level, min_snr, max_peaks, verbose)

    if len(peaks) == 0:
      if verbose > 2:
        print("# No water peaks found above SNR threshold")
      return None

    # Calculate FWHM for each peak and average
    fwhm_values = []
    for peak_ppm, peak_magnitude, peak_idx in peaks:
      fwhm_hz = self._calculate_fwhm_interpolated(water_magnitude, water_nu, peak_idx, peak_magnitude, verbose)
      if fwhm_hz is not None:
        fwhm_values.append(fwhm_hz)
        if verbose > 2:
          print(f"# Water peak at {peak_ppm:.2f} ppm: {fwhm_hz:.1f} Hz")

    if len(fwhm_values) == 0:
      if verbose > 2:
        print("# Could not determine FWHM for any water peaks")
      return None

    # Use median for robustness against outliers
    median_fwhm = np.median(fwhm_values)
    if verbose > 2:
      print(f"# Water peak FWHM: {median_fwhm:.1f} Hz (from {len(fwhm_values)} peaks)")

    return median_fwhm

  def _estimate_linewidth_metabolite_peak_robust(self, magnitude, nu, verbose, min_snr=3.0, max_peaks=3):
    """Robust estimate linewidth from metabolite peaks with improved peak detection.

    Parameters
    ----------
    magnitude : array
        Magnitude spectrum data
    nu : array
        Frequency axis in ppm
    verbose : int
        Verbosity level
    min_snr : float
        Minimum SNR threshold for peak detection
    max_peaks : int
        Maximum number of peaks to use for averaging

    Returns
    -------
    float or None
        Estimated linewidth in Hz, or None if suitable peaks not found
    """
    # Look for metabolite peaks in the typical MRS range
    metabolite_region = (nu >= -1.0) & (nu <= 4.5)
    if not np.any(metabolite_region):
      if verbose > 2:
        print("# Metabolite region (-1.0 to 4.5 ppm) not found in spectrum")
      return None

    metabolite_magnitude = magnitude[metabolite_region]
    metabolite_nu = nu[metabolite_region]

    # Estimate noise level from baseline
    noise_region = (nu >= 8.0) & (nu <= 12.0)  # High ppm noise region
    if np.any(noise_region):
      noise_level = np.std(magnitude[noise_region])
    else:
      noise_level = np.std(metabolite_magnitude) * 0.1  # Fallback estimate

    # Find multiple peaks above SNR threshold
    peaks = self._find_peaks_robust(metabolite_magnitude, metabolite_nu, noise_level, min_snr, max_peaks, verbose)

    if len(peaks) == 0:
      if verbose > 2:
        print("# No metabolite peaks found above SNR threshold")
      return None

    # Calculate FWHM for each peak and average
    fwhm_values = []
    for peak_ppm, peak_magnitude, peak_idx in peaks:
      fwhm_hz = self._calculate_fwhm_interpolated(metabolite_magnitude, metabolite_nu, peak_idx, peak_magnitude, verbose)
      if fwhm_hz is not None:
        fwhm_values.append(fwhm_hz)
        if verbose > 1:
          print(f"# Metabolite peak at {peak_ppm:.2f} ppm: {fwhm_hz:.1f} Hz")

    if len(fwhm_values) == 0:
      if verbose > 2:
        print("# Could not determine FWHM for any metabolite peaks")
      return None

    # Use median for robustness against outliers
    median_fwhm = np.median(fwhm_values)
    if verbose > 2:
      print(f"# Metabolite peak FWHM: {median_fwhm:.1f} Hz (from {len(fwhm_values)} peaks)")

    return median_fwhm

  def _find_peaks_robust(self, magnitude, nu, noise_level, min_snr, max_peaks, verbose):
    """Find peaks above SNR threshold with proper peak detection.

    Parameters
    ----------
    magnitude : array
        Magnitude spectrum data
    nu : array
        Frequency axis in ppm
    noise_level : float
        Estimated noise level
    min_snr : float
        Minimum SNR threshold
    max_peaks : int
        Maximum number of peaks to return
    verbose : int
        Verbosity level

    Returns
    -------
    list
        List of (peak_ppm, peak_magnitude, peak_idx) tuples
    """
    # Find local maxima
    from scipy.signal import find_peaks

    # Ensure minimum peak height based on SNR
    min_height = noise_level * min_snr

    # Find peaks with minimum distance between them
    min_distance = max(1, len(magnitude) // 100)  # At least 1% of spectrum width

    peaks_idx, properties = find_peaks(magnitude, height=min_height, distance=min_distance)

    if len(peaks_idx) == 0:
      return []

    # Sort peaks by magnitude (strongest first)
    peak_magnitudes = magnitude[peaks_idx]
    sort_indices = np.argsort(peak_magnitudes)[::-1]  # Descending order

    # Select top peaks
    selected_peaks = []
    for i in sort_indices[:max_peaks]:
      peak_idx = peaks_idx[i]
      peak_ppm = nu[peak_idx]
      peak_magnitude = magnitude[peak_idx]
      snr = peak_magnitude / noise_level

      if snr >= min_snr:
        selected_peaks.append((peak_ppm, peak_magnitude, peak_idx))
        if verbose > 2:
          print(f"# Peak at {peak_ppm:.2f} ppm: SNR={snr:.1f}, magnitude={peak_magnitude:.2e}")

    return selected_peaks

  def _calculate_fwhm_interpolated(self, magnitude, nu, peak_idx, peak_magnitude, verbose):
    """Calculate FWHM with interpolation for better accuracy.

    Parameters
    ----------
    magnitude : array
        Magnitude spectrum data
    nu : array
        Frequency axis in ppm
    peak_idx : int
        Index of peak maximum
    peak_magnitude : float
        Peak magnitude
    verbose : int
        Verbosity level

    Returns
    -------
    float or None
        FWHM in Hz, or None if calculation fails
    """
    half_max = peak_magnitude / 2.0

    # Find left and right half-maximum points with interpolation
    left_ppm = self._find_half_max_interpolated(magnitude, nu, peak_idx, half_max, direction='left')
    right_ppm = self._find_half_max_interpolated(magnitude, nu, peak_idx, half_max, direction='right')

    if left_ppm is None or right_ppm is None:
      return None

    # Calculate FWHM in ppm
    fwhm_ppm = right_ppm - left_ppm

    # Convert to Hz
    fwhm_hz = fwhm_ppm * self.omega

    return fwhm_hz

  def _find_half_max_interpolated(self, magnitude, nu, peak_idx, half_max, direction='left'):
    """Find half-maximum point with linear interpolation.

    Parameters
    ----------
    magnitude : array
        Magnitude spectrum data
    nu : array
        Frequency axis in ppm
    peak_idx : int
        Index of peak maximum
    half_max : float
        Half-maximum value
    direction : str
        'left' or 'right' to search from peak

    Returns
    -------
    float or None
        Interpolated ppm value, or None if not found
    """
    if direction == 'left':
      search_range = range(peak_idx - 1, -1, -1)
    else:
      search_range = range(peak_idx + 1, len(magnitude))

    for i in search_range:
      if direction == 'left' and magnitude[i] <= half_max:
        # Linear interpolation between i and i+1
        if i + 1 < len(magnitude):
          x1, y1 = nu[i], magnitude[i]
          x2, y2 = nu[i + 1], magnitude[i + 1]
          # Interpolate to find x where y = half_max
          if y2 != y1:
            x_interp = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
            return x_interp
          else:
            return x1
        else:
          return nu[i]
      elif direction == 'right' and magnitude[i] <= half_max:
        # Linear interpolation between i-1 and i
        if i - 1 >= 0:
          x1, y1 = nu[i - 1], magnitude[i - 1]
          x2, y2 = nu[i], magnitude[i]
          # Interpolate to find x where y = half_max
          if y2 != y1:
            x_interp = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
            return x_interp
          else:
            return x1
        else:
          return nu[i]

    return None

  def _estimate_linewidth_lorentzian(self, magnitude, nu, verbose, min_snr=3.0, max_peaks=3):
    """Estimate linewidth using Lorentzian peak fitting with FWHM fallback.

    This method fits Lorentzian functions to identified peaks and uses the fitted
    linewidth parameter. If fitting fails, it falls back to FWHM calculation.

    Parameters
    ----------
    magnitude : array
        Magnitude spectrum data
    nu : array
        Frequency axis in ppm
    verbose : int
        Verbosity level
    min_snr : float
        Minimum SNR threshold for peak detection
    max_peaks : int
        Maximum number of peaks to use for averaging

    Returns
    -------
    float or None
        Estimated linewidth in Hz, or None if estimation fails
    """
    try:
      from scipy.optimize import curve_fit
    except ImportError:
      if verbose > 2:
        print("# scipy not available for Lorentzian fitting, falling back to FWHM")
      return self._estimate_linewidth_metabolite_peak_robust(magnitude, nu, verbose, min_snr, max_peaks)

    # Look for metabolite peaks in the typical MRS range
    metabolite_region = (nu >= -1.0) & (nu <= 4.5)
    if not np.any(metabolite_region):
      if verbose > 2:
        print("# Metabolite region (-1.0 to 4.5 ppm) not found in spectrum")
      return None

    metabolite_magnitude = magnitude[metabolite_region]
    metabolite_nu = nu[metabolite_region]

    # Estimate noise level from baseline
    noise_region = (nu >= 8.0) & (nu <= 12.0)  # High ppm noise region
    if np.any(noise_region):
      noise_level = np.std(magnitude[noise_region])
    else:
      noise_level = np.std(metabolite_magnitude) * 0.1  # Fallback estimate

    # Find multiple peaks above SNR threshold
    peaks = self._find_peaks_robust(metabolite_magnitude, metabolite_nu, noise_level, min_snr, max_peaks, verbose)

    if len(peaks) == 0:
      if verbose > 2:
        print("# No metabolite peaks found above SNR threshold for Lorentzian fitting")
      return None

    # Define Lorentzian function
    def lorentzian(x, amplitude, center, linewidth, baseline):
      """Lorentzian function: amplitude / (1 + ((x - center) / (linewidth/2))^2) + baseline."""
      return amplitude / (1 + ((x - center) / (linewidth / 2))**2) + baseline

    linewidth_values = []

    for peak_ppm, peak_magnitude, _ in peaks:
      try:
        # Define fitting region around the peak (±0.5 ppm or 20% of spectrum width)
        region_width = min(0.5, (metabolite_nu[-1] - metabolite_nu[0]) * 0.2)
        region_mask = (metabolite_nu >= peak_ppm - region_width) & (metabolite_nu <= peak_ppm + region_width)

        if not np.any(region_mask):
          continue

        fit_nu = metabolite_nu[region_mask]
        fit_magnitude = metabolite_magnitude[region_mask]

        # Initial parameter estimates
        baseline_est = np.median(fit_magnitude)
        amplitude_est = peak_magnitude - baseline_est
        center_est = peak_ppm

        # Estimate linewidth from FWHM as initial guess
        half_max = peak_magnitude / 2.0
        left_idx = None
        right_idx = None
        for i in range(len(fit_magnitude)):
          if fit_magnitude[i] <= half_max and left_idx is None:
            left_idx = i
          if fit_magnitude[i] <= half_max:
            right_idx = i
        if left_idx is not None and right_idx is not None:
          fwhm_ppm_est = fit_nu[right_idx] - fit_nu[left_idx]
          linewidth_est = fwhm_ppm_est * 0.5  # Initial guess based on FWHM
        else:
          linewidth_est = 0.1  # Fallback

        # Parameter bounds - more generous bounds
        bounds = (
          [0, peak_ppm - region_width/2, 0.001, baseline_est * 0.5],  # Lower bounds
          [amplitude_est * 3, peak_ppm + region_width/2, 1.0, baseline_est * 1.5]  # Upper bounds
        )

        # Try fitting without bounds first
        try:
          popt, pcov = curve_fit(
            lorentzian, fit_nu, fit_magnitude,
            p0=[amplitude_est, center_est, linewidth_est, baseline_est],
            maxfev=2000
          )
        except Exception:
          # If that fails, try with bounds
          popt, pcov = curve_fit(
            lorentzian, fit_nu, fit_magnitude,
            p0=[amplitude_est, center_est, linewidth_est, baseline_est],
            bounds=bounds,
            maxfev=2000
          )

        # Extract fitted linewidth and convert to Hz
        fitted_linewidth_ppm = popt[2]
        fitted_linewidth_hz = fitted_linewidth_ppm * self.omega

        # Validate the fit
        fitted_curve = lorentzian(fit_nu, *popt)
        r_squared = 1 - np.sum((fit_magnitude - fitted_curve)**2) / np.sum((fit_magnitude - np.mean(fit_magnitude))**2)

        if verbose > 2:
          print(f"# Lorentzian fit details at {peak_ppm:.2f} ppm: R²={r_squared:.3f}, linewidth={fitted_linewidth_ppm:.3f} ppm, amplitude={popt[0]:.1f}")

        if r_squared > 0.5 and 0.01 < fitted_linewidth_ppm < 2.0:  # More lenient criteria
          linewidth_values.append(fitted_linewidth_hz)
          if verbose > 1:
            print(f"# Lorentzian fit at {peak_ppm:.2f} ppm: {fitted_linewidth_hz:.1f} Hz (R²={r_squared:.3f})")
        else:
          if verbose > 2:
            print(f"# Lorentzian fit at {peak_ppm:.2f} ppm rejected: R²={r_squared:.3f}, linewidth={fitted_linewidth_ppm:.3f} ppm")

      except Exception as e:
        if verbose > 2:
          print(f"# Lorentzian fitting failed for peak at {peak_ppm:.2f} ppm: {e}")

    # If Lorentzian fitting failed for all peaks, fall back to FWHM
    if len(linewidth_values) == 0:
      if verbose > 2:
        print("# All Lorentzian fits failed, falling back to FWHM method")
      return self._estimate_linewidth_metabolite_peak_robust(magnitude, nu, verbose, min_snr, max_peaks)

    # Use median for robustness against outliers
    median_linewidth = np.median(linewidth_values)
    if verbose > 2:
      print(f"# Lorentzian linewidth: {median_linewidth:.1f} Hz (from {len(linewidth_values)} peaks)")

    return median_linewidth
