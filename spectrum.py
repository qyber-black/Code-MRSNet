#!/usr/bin/env python3
#
# spectrum.py - MRSNet - process individual spectrum
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2019, Max Chandler, PhD student at Cardiff University
# Copyright (C) 2020, Frank C Langbein <frank@langbein.org>, Cardiff University

import os
import dill
import random
import struct
import numpy as np
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt

from utilities.utils import convert_molecule_names
from utilities.constants import NAA_REFERENCE, CR_REFERENCE, WATER_REFERENCE, DICOM_METADATA, GYROMAGNETIC_RATIO
from utilities.read_dicom_siemens import read_dicom, TAG_SPECTROSCOPY_DATA, TAG_PATIENT_ID

class Spectrum(object):
    """
        Spectrum is a class that contains information about one single spectrum
        loaded from any source. Time domain data is preferred over frequency
        domain when loading.
    """
    def __init__(self):
        self.id = None
        self.source = None              # pygamma, fid-a, lcmodel,...
        self.type = None                # simulated or real
        self.pulse_sequence = None      # megapress, press, steam, fid, etc...
        self.spectrum_type = None       # for megapress: edit on, edit off, difference
        self.metabolite_names = []
        self.concentrations = []
        self.scale = 1
        self.center_ppm = 0
        self.omega = None
        self.dt = None
        self.sw = None
        self.te = None
        self.linewidth = None
        self.metadata = {}
        self.acquisition = None         # for basis sets with multiple acquisitons of the same metabolite, e.g.
                                        # MEGA-PRESS with edit on/off

        self.add_adc_noise = False
        self._adc_noise_mu = None
        self._adc_noise_sigma = None
        self._adc_noise = None

        self._source_filepath = None
        self._filter_fft = False        # this is only applied by default to dicom files
        self._raw_adc = []
        self._zero_pad = 0

        self.remove_water_peak = True

        self._correct_b0 = True         # do we allow b0 correction for this spectrum?
        self._b0_reference_frequency = None
        self._b0_ppm_shift = 0

        self._b0_fft_pts_shift = 0      # explanation for these two can be found in correct_b0
        self._b0_nu_shift = 0

        self._fft_cache = None
        self._need_fft_cache_update = False

    def check(self):
        self.metabolite_names = convert_molecule_names(self.metabolite_names)
        if not len(self._raw_adc):
            raise Exception('Spectrum has no raw adc associalted with it')
        if self.acquisition is None:
            raise Exception('Please set acquisition number for spectrum.')
        if self.type is None:
            raise Exception('Please set type of spectrum')
        if self.spectrum_type is None and self.pulse_sequence == 'megapress':
            raise Exception('Spectrum pulse sequence is megapress but spectrum type is not set, please set it!')
        if self.pulse_sequence == 'megapress' and self.spectrum_type not in ['edit off', 'edit on', 'difference']:
            raise Exception('Pulse sequence is megapress, but spectrum_type not in [edit off, edit on, difference]: ' + self.spectrum_type)
        if any(np.isnan(self._raw_adc)):
            raise Exception('Raw adc contains nan values')

    def zero_signal(self):
        return np.zeros(len(self._raw_adc) + self._zero_pad, dtype=complex)

    def adc(self):
        if self.scale > 0:
            adc = self.scale * self._raw_adc
            if self.add_adc_noise:
                adc += self._adc_noise  # (self._adc_noise * np.max(self._raw_adc))
            adc = np.append(adc, np.zeros(self._zero_pad))
        elif self.scale == 0:
            adc = self.zero_signal()
        else:
            raise Exception('Spectrum scale cannot be less than 0')
        return adc

    def raw_adc(self):
        if self.scale == 0:
            return np.zeros(len(self._raw_adc), dtype=complex)
        else:
            return self._raw_adc * self.scale

    def nu(self, npts=None):
        if npts is None:
            npts = len(self.adc())
        if self.source == 'dicom':
            ppm_range = (self.sw / self.omega) / 2
            nu = np.linspace(-ppm_range, ppm_range, npts) + self.center_ppm
        elif self.source in ['pygamma', 'fid-a', 'lcmodel'] or self.source[0:4] == 'sim_':
            nu = ((np.linspace(-1, 1, npts) * (1 / self.dt / 2)) / self.omega) + self.center_ppm
        else:
            raise Exception('Please write custom nu routine for input source: ' + self.source)
        # Finally apply the fine grained b0 correction, this is on the order of < delta_nu
        nu += self._b0_nu_shift
        return nu

    def delta_nu(self):
        return np.abs(self.nu()[0] - self.nu()[1])

    def ppm_to_nu_pts(self, ppm_shift):
        return int(np.floor(ppm_shift / self.delta_nu()))

    def nu_pts_to_ppm(self, n_nu_pts):
        return self.delta_nu() * n_nu_pts

    def update_zero_pad(self, new_zero_pad):
        if new_zero_pad != self._zero_pad:
            self._zero_pad = new_zero_pad
            if self._b0_ppm_shift:
                self.update_b0_shift(self._b0_ppm_shift)

    def correct_b0(self, ppm_shift=None):
        # the way this works is twofold
        # the major B0 correction will be done by padding and trimming the fft, shifting it on the frequency axis nu
        # the issue is that this then only has a granularity of delta_nu, so the remaining b0 shit has to be corrected
        # by offsetting the entire nu axis by a value < delta_nu
        if self._correct_b0:
            if ppm_shift is not None:
                # we've defined the ppm shift
                self.update_b0_shift(ppm_shift)
            else:
                reference_peaks = []
                if len(self.metabolite_names):
                    if 'n-acetylaspartate' in self.metabolite_names:
                        reference_peaks.append(NAA_REFERENCE)
                    if 'creatine' in self.metabolite_names:
                        reference_peaks.append(CR_REFERENCE)
                else:
                    if not reference_peaks and ppm_shift is None:
                        # no ref metabolites found, try them anyway.
                        reference_peaks = [NAA_REFERENCE, CR_REFERENCE]
                if reference_peaks:
                    fft = self.fft()
                    nu = self.nu()
                    for reference_signal in reference_peaks:
                        peak = self.peak_location(fft, nu, reference_signal, 0.25)
                        if peak:
                            self.update_b0_shift(self._b0_ppm_shift + (peak - reference_signal))
                            break

    def clear_b0(self):
        self._correct_b0 = False
        self._b0_ppm_shift = 0
        self._b0_fft_pts_shift = 0
        self._b0_nu_shift = 0
        self._need_fft_cache_update = True

    def update_b0_shift(self, new_b0_shift):
        self._b0_ppm_shift = new_b0_shift
        self._b0_fft_pts_shift = self.ppm_to_nu_pts(self._b0_ppm_shift)
        self._b0_nu_shift = self._b0_ppm_shift - self.nu_pts_to_ppm(self._b0_fft_pts_shift)
        self._need_fft_cache_update = True

    def fft(self, update_cache=False):
        if not update_cache:
            # do some checking here, just in case we do need to forcibly recalculate the fft
            if self._need_fft_cache_update:
                update_cache = True
            elif self._fft_cache is None:
                update_cache = True
            elif len(self.adc()) != len(self._fft_cache):
                update_cache = True
        if update_cache:
            self._need_fft_cache_update = False
            if self.scale != 0:
                adc = self.adc()
                # fft routines for different input sources
                if self.source == 'dicom' or self.source == 'lcmodel' or self.source[0:12] == 'sim_lcmodel_':
                    fft = np.fft.fftshift(np.fft.fft(adc, len(adc)))
                elif self.source == 'pygamma' or self.source == 'fid-a'  or self.source[0:12] == 'sim_pygamma_' or self.source[0:10] == 'sim_fid-a_':
                    fft = np.flip(np.fft.fftshift(np.fft.fft(adc, len(adc)), 0))
                else:
                    raise Exception('Please write custom raw_fft routine for input source: ' + self.source)
                if self._filter_fft:
                    b, a = signal.butter(1, 0.7)
                    fft = signal.filtfilt(b, a, fft, padlen=150)
            else:
                fft = self.zero_signal()
            # b0 correction is required
            if self._b0_fft_pts_shift != 0:
                if self._b0_fft_pts_shift > 0:
                    fft = np.pad(fft, (0, self._b0_fft_pts_shift), 'constant', constant_values=np.mean(fft))[self._b0_fft_pts_shift:]
                elif self._b0_fft_pts_shift < 0:
                    fft = np.pad(fft, (np.abs(self._b0_fft_pts_shift), 0), 'constant', constant_values=np.mean(fft))[:self._b0_fft_pts_shift]
            if self.type == 'real' and self.remove_water_peak:
                fft = Spectrum.remove_water_peak(fft, self.nu(), ppm_range=1)
            self._fft_cache = fft
        return self._fft_cache

    def raw_fft(self):
        # No scaling, zero padding, shifting or correction at all
        if self.source == 'dicom' or self.source == 'lcmodel':
            return np.fft.fftshift(np.fft.fft(self._raw_adc, len(self._raw_adc)))
        elif self.source == 'pygamma' or self.source == 'fid-a':
            return np.flip(np.fft.fftshift(np.fft.fft(self._raw_adc, len(self._raw_adc)), 0))
        else:
            raise Exception('Please write custom raw_fft routine for input source: ' + self.source)

    def rescale_fft(self, high_ppm=-4.5, low_ppm=-1, npts=2048):
        # zero pads the time domain to fill the desired window with npts
        recursion_limit = 500
        nu = self.nu()
        if (np.max(nu) < low_ppm) or (np.min(nu) > high_ppm):
            raise Exception('Requested ppm rescale range out of range of nu of spectrum. Max:' + str(np.min(nu)) +
                            ' Min: ' + str(np.max(nu)))

        # calculate initially how many points in that range
        index = (nu >= high_ppm) & (nu <= low_ppm)
        nu_pts = len(nu[index])
        counter = 0
        while nu_pts != npts:
            if counter > recursion_limit:
                raise Exception('Recursion limit hit!')
            if counter > 100:
                print('Counter is getting high... ' + str(self._zero_pad) + ' : ' + str(nu_pts) + ' aiming for: ' + str(npts))

            # fine tune if that's not quite right
            # nu_pts = len(self.nu()[(self.nu() > high_ppm) & (self.nu() < low_ppm)])
            percent_range = len(nu) / float(nu_pts)
            # random is added as there is sometimes some aliasing effects, this corrects it
            self.update_zero_pad(self._zero_pad + int(round(np.round((npts - nu_pts) * percent_range) + np.random.random())))
            # print(str(self._zero_pad) + ' : ' + str(nu_pts) + ' aiming for: ' + str(npts) + ' dt:' + str(self.dt) + ' n:' + str(len(self._raw_adc)))

            if self._zero_pad < 0:
                raise Exception('Real data is too large to be input into the network, would have to reduce the '
                                'resolution of it. OR train a network with a higher resolution across the ppm range.')
            nu = self.nu()
            index = (nu >= high_ppm) & (nu <= low_ppm)
            nu_pts = len(nu[index])
            counter += 1

        return self.fft(update_cache=True)[index], nu[index]

    def trim_fft(self, high_ppm=-4.5, low_ppm=-1):
        nu = self.nu()
        if (np.max(nu) < low_ppm) or (np.min(nu) > high_ppm):
            raise Exception('Requested ppm trim range out of range of nu of spectrum.')
        index = (nu > high_ppm) & (nu < low_ppm)
        return self.fft()[index], nu[index]

    def max_metabolite_amplitude(self, magnitude=False):
        metabolite_range = [-4.2, -1.5]
        fft = self.fft()
        nu = self.nu()
        m_fft = fft[(nu < np.max(metabolite_range)) & (nu > np.min(metabolite_range))]
        if magnitude:
            return np.max(np.abs(m_fft))
        else:
            return np.max([np.abs(np.real(m_fft)), np.abs(np.imag(m_fft))])

    def adc_noise(self):
        if self._adc_noise is None:
            self.generate_adc_noise()
        return self._adc_noise

    def setup_adc_noise_parameters(self, noise_p, max_sigma):
        if self.add_adc_noise:
            if self._adc_noise_mu is not None and self._adc_noise_sigma is not None:
                return
            # So we make a choice for mu and sigma and store that option, with certain probability
            if random.random() <= noise_p:
                self._adc_noise_mu = 0 # random.choice([0])
                self._adc_noise_sigma = random.choice(np.linspace(0.0, max_sigma, 1000)) # 1000 is probably overkill
            else:
                self._adc_noise_mu = 0
                self._adc_noise_sigma = 0
        else:
            self._adc_noise_mu = None
            self._adc_noise_sigma = None

    def generate_adc_noise(self, overwrite=False):
        # These were intially stored with the spectrum, but that meant they weren't as dynamic as I'd like.
        if self._adc_noise is None or overwrite:
            if not self.add_adc_noise:
                self._adc_noise = None
            elif (self._adc_noise_mu is None or self._adc_noise_mu <= 0) and \
                 (self._adc_noise_sigma is None or self._adc_noise_sigma <= 0):
                self._adc_noise = np.zeros(len(self._raw_adc))
            else:
                self._adc_noise = np.random.normal(self._adc_noise_mu, self._adc_noise_sigma, len(self._raw_adc)) + \
                                  (1j * np.random.normal(self._adc_noise_mu, self._adc_noise_sigma, len(self._raw_adc)))
        else:
            raise Exception('Tried to overwrite adc noise, this is not allowed by default. '
                            'If you want to do this, pass overwrite=true')

    def plot(self):
        fft, nu = self.rescale_fft()
        n_cols = 1

        super_title = ''
        if len(self.concentrations) > 1:
            n_cols = 2
        else:
            super_title += self.metabolite_names[0] + ' '
        super_title += self.source + ' ' + self.pulse_sequence
        if self.spectrum_type is not None:
            super_title += ' ' + str(self.spectrum_type)
        if self.add_adc_noise:
            super_title += " Noise mu: %.1f, sigma: %.4f" % (self._adc_noise_mu, self._adc_noise_sigma)

        figure = plt.subplots(3, n_cols, sharex=True, sharey=True, figsize=(19.2, 19.2), dpi=100)

        plt.suptitle(super_title)

        plt.subplot(3, n_cols, 1)
        plt.title('Magnitude')
        plt.plot(nu, np.abs(fft))
        plt.xlim([-4.5, -1])

        plt.subplot(3, n_cols, 1 + n_cols)
        plt.title('Real')
        plt.plot(nu, np.real(fft))
        plt.xlim([-4.5, -1])

        plt.subplot(3, n_cols, 1 + (n_cols * 2))
        plt.title('Imaginary')
        plt.plot(nu, np.imag(fft))
        plt.xlim([-4.5, -1])
        # to do, reverse sign on axis

        if n_cols == 2:
            ax = plt.subplot(1, 2, 2)
            plt.title('Concentrations')
            ax.bar(np.linspace(0, len(self.concentrations) - 1, len(self.concentrations)), self.concentrations)
            ax.set_xticks(np.arange(len(metabolite_names) + 1))
            ax.set_xticklabels(convert_molecule_names(metabolite_names, shorten=True))
        plt.show()

        return figure

    def save(self, directory):
        self.check()
        filename = '-'.join(self.metabolite_names) + 'acq_' + self.acquisition + '.dill'
        with open(os.path.join(directory, filename), 'wb') as out_file:
            dill.dump(self, out_file)

    @staticmethod
    def load(filepath):
        with open(filepath) as in_file:
            return dill.load(in_file)

    def load_metadata(self):
        # matches the ima files to the dict of metadata in Utilities->constants.property
        # matching is based on the filepath
        split_filepath = self._source_filepath.split('/')
        series_name = [x for x in split_filepath if x in DICOM_METADATA.keys()]
        found = False
        # first we find which series
        if len(series_name)  == 1:
            series_name = series_name[0]
            spectrum_id = [x for x in split_filepath if x in DICOM_METADATA[series_name].keys()]
            if len(spectrum_id) == 1:
                # found it! Now add the data to self
                spectrum_id = spectrum_id[0];
                found = True
                if 'b0_ppm_shift' in DICOM_METADATA[series_name][spectrum_id].keys():
                    self.correct_b0(DICOM_METADATA[series_name][spectrum_id]['b0_ppm_shift'])
                if 'concentrations' in DICOM_METADATA[series_name][spectrum_id].keys():
                    self.concentrations = [float(i) for i in DICOM_METADATA[series_name][spectrum_id]['concentrations']]
                if 'metabolite_names' in DICOM_METADATA[series_name][spectrum_id].keys():
                    self.metabolite_names = DICOM_METADATA[series_name][spectrum_id]['metabolite_names']
                    if self.metabolite_names:
                        self.id = '_'.join(self.metabolite_names) + '_' + self.patient_id + '_' + spectrum_id
                if self.pulse_sequence == 'megapress':
                    self.id = spectrum_id
            elif len(spectrum_id) > 1:
                raise Exception('Spectrum matched more than one ID: ' + spectrum_id)
            else:
                raise Exception('Spectrum matched series, but did not match any ID: ' + self._source_filepath)
        elif len(series_name) > 1:
            raise Exception('Dicom data matches more than one series (match done on foler names).\nMatching series: ' + series_name +'\nIMA Filepath: '+self._source_filepath)
        elif len(series_name) == 0:
            # no data for the dicom was found
            pass

        if self.pulse_sequence == 'megapress':
            if 'EDIT_OFF' in self._source_filepath:
                self.spectrum_type = 'edit off'
                self.acquisition = 0
            elif 'EDIT_ON' in self._source_filepath:
                self.spectrum_type = 'edit on'
                self.acquisition = 1
            elif 'DIFF' in self._source_filepath:
                self.spectrum_type = 'difference'
                self.acquisition = 2
            else:
                # There is no good reliable way to identify these without guessing based off the time and dates, then working backwards.
                raise Exception('Loaded dicom file of type MEGA-PRESS, but I can\'t figure out which acquisition this '
                                'is (Edit On, Edit Off or Difference). \n'
                                'Please manuall specifiy it (add "EDIT_OFF", "EDIT_ON" or "DIFF" into the filepath anywhere).')

        if self.pulse_sequence == 'megapress' and not found:
            raise Exception('Spectrum ID has not been set successfully for spectrum. \n This is required for MEGA-PRESS '
                            'spectra to group them properly. Please add a unique ID to your dicom file names, and add '
                            'this ID to the dictionary in Utilties.constants to group the scans together. \n'
                            'See Utilities.constants and Spectrum.load_dicom for more information.')

    @staticmethod
    def load_dicom(ima_file):
        if not os.path.exists(ima_file):
            raise Exception('ima file does not exist: ' + ima_file)
        spectrum = Spectrum()
        # We assume it's a Siemens dicom spectrum, so do not check this
        dicom, info = read_dicom(ima_file)
        # Read Siemens DICOM spectroscopy data tag (0x7fe1, 0x1010) as a list of complex numbers ReImReIm...; 4 byte floats; little endian
        data = dicom[TAG_SPECTROSCOPY_DATA].value
        data = struct.unpack("<%df" % (len(data) / 4), data)
        data = [complex(data[i], data[i+1]) for i in range(0, len(data), 2)]
        # Patient ID
        patient_id = dicom[TAG_PATIENT_ID].value
        # Sweep width
        #remove_oversample_flag = (info["dicom"]["sSpecPara"]["ucRemoveOversampling"].strip() == "0x1")
        #readout_os = float(info["dicom"]["ReadoutOS"], 1.0))
        sweep_width = 1 / (float(info["info"]["RealDwellTime"]) * 1e-9)
        # if not remove_oversample_flag:
        #   sweep_width *= readout_os
        # Frequency
        frequency = float(info["info"]["ImagingFrequency"])
        # Sequence filename
        sequence_filename = info["protocol"]["tSequenceFileName"].split('\\')[-1]
        # Echo time
        echo_time = float(info["info"]["EchoTime"])
        ## dicom_data = util_dicom_siemens.read(ima_file)
        spectrum._source_filepath = ima_file
        spectrum.patient_id = patient_id
        spectrum.source = 'dicom'
        spectrum.id = ima_file.split('/')[-1]
        spectrum._raw_adc = np.array(data)
        spectrum.sw = sweep_width
        spectrum.omega = frequency
        spectrum.dt = -(spectrum.omega / spectrum.sw) / 1e+2
        spectrum.center_ppm = -4.7
        spectrum.pulse_sequence = sequence_filename
        spectrum.te = echo_time
        spectrum.acquisition = 0
        spectrum.type = 'real'
        spectrum.linewidth = -1      # Unknown!
        spectrum._filter_fft = True

        # do some translation of the pulse sequence types - this works for Siemens
        if spectrum.pulse_sequence in ['svs_edit', 'svs_ed', 'megapress']:
            spectrum.pulse_sequence = 'megapress'
        elif spectrum.pulse_sequence == 'svs_se':
            spectrum.pulse_sequence = 'press'
        elif spectrum.pulse_sequence == 'svs_st':
            spectrum.pulse_sequence = 'steam'
        else:
            raise Exception('Unrecognised dicom pulse sequence: ' + spectrum.pulse_sequence + '. Please add an \'elif\' statement for it. ')

        spectrum.load_metadata()
        spectrum.check()

        return spectrum


    @staticmethod
    def load_fida(fida_file):
        spectrum = Spectrum()
        spectrum.source = 'fid-a'
        spectrum.type = 'simulated'
        fida_data = loadmat(fida_file)
        spectrum._source_filepath = fida_file
        spectrum.pulse_sequence = 'megapress'
        spectrum.metabolite_names = convert_molecule_names([str(fida_data['m_name'][0])])
        spectrum.dt = np.abs(fida_data['t'][0][0] - fida_data['t'][0][1])
        spectrum._raw_adc = np.transpose(fida_data['fid'])[0]
        spectrum.omega = float(fida_data['omega'][0][0]) * GYROMAGNETIC_RATIO
        spectrum.linewidth = float(fida_data['linewidth'][0][0])
        spectrum.center_ppm = -np.median(fida_data['nu'])
        spectrum.id = spectrum.metabolite_names[0]

        if bool(fida_data['edit'][0][0]):
            spectrum.acquisition = 1
            spectrum.spectrum_type = 'edit on'
        else:
            spectrum.acquisition = 0
            spectrum.spectrum_type = 'edit off'

        spectrum.check()
        return spectrum

    @staticmethod
    def load_lcm_basis(file_buffer, dt, omega):
        # http://s-provencher.com/pub/LCModel/manual/manual.pdf
        spectrum = Spectrum()
        spectrum.source = 'lcmodel'
        spectrum.type = 'simulated'
        spectrum.dt = dt
        spectrum.omega = omega
        spectrum.linewidth = 1
        area = None
        var_buffer = ''

        for counter, line in enumerate(file_buffer):
            if '$NMUSED' in line:
                area = 'nmused'
                continue
            elif "$BASIS" in line:
                area = 'basis'
                continue
            elif '$END' in line:
                # automatically assume that the next area is the spectrum region, it may be the basis section
                if len(var_buffer):
                    spectrum.add_lcm_metadata(var_buffer)
                    var_buffer = ''
                area = 'spectrum'
                continue

            # var buffer is used as some variables span multiple lines, the only break is either commas or $END
            # there's also no end marker to the spectrum in most cases
            var_buffer += line
            if ',' in var_buffer or (len(file_buffer) - 1) == counter:
                if area == 'nmused' or area == 'basis':
                    spectrum.add_lcm_metadata(var_buffer)
                elif area == 'spectrum':
                    spectrum.add_lcm_fft(var_buffer)
                else:
                    raise Exception('No area set ' + str(counter) + ' :' + var_buffer)
                var_buffer = ''

        if 'PPMSEP' in spectrum.metadata:
            spectrum.center_ppm = -spectrum.metadata['PPMSEP']
        else:
            spectrum.center_ppm = -4.65      # defualt LCM value

        # Convert the LCM basis set names to something more universal, see utils.convert_mol_names.
        spectrum.metabolite_names = convert_molecule_names([spectrum.metadata['METABO'].replace('\'', '')])
        spectrum.acquisition = 0
        spectrum.check()
        return spectrum

    def add_lcm_metadata(self, var_buffer):
        if '=' in var_buffer:
            for to_remove in [' ', ',', '   ', '\n']:
                var_buffer = var_buffer.replace(to_remove, '')
            split_line = var_buffer.split('=')
            if split_line[0] in ['CONCSC', 'PPMPK', 'PPMPHA', 'PPMSCA', 'PPMSEP', 'PPMSCA', 'HWDPHA', 'HWDSCA',
                                 'PPMBAS', 'FWHMSM', 'CONCSC', 'CONC']:
                self.metadata[split_line[0]] = float(split_line[1])
            elif split_line[0] in ['NDATAB', 'ISHIFT']:
                self.metadata[split_line[0]] = int(split_line[1])
            elif split_line[0] in ['AUTOPH', 'AUTOSC', 'SCALE1', 'CONSISTENT_SCALING', 'NOSHIF', 'DO_CONTAM', 'FLATEN']:
                if 'F' in split_line[1]:
                    self.metadata[split_line[0]] = False
                elif 'T' in split_line[1]:
                    self.metadata[split_line[0]] = False
                else:
                    raise IOError('Unknown boolean(?) type: ' + split_line[1])
            else:
                # not sure what to do with these lines, keep them as strings...?
                self.metadata[split_line[0]] = split_line[1]
        else:
            raise Exception('Not sure how to handle line :' + var_buffer)

    def add_lcm_fft(self, var_buffer):
        fft = []
        nums = var_buffer.replace('\n', '').split()

        if len(nums) % 2 != 0:
            raise Exception('Uneven fft number, the real/imag switching does not work here or the file has been loaded '
                            'in wrong!')

        for ii in range(0, len(nums), 2):
            fft.append(float(nums[ii]) + (1j * float(nums[ii + 1])))

        # All lcmodel spectra are stored as fourier transforms, so we convert them back to the ADC
        self._raw_adc = np.flip(np.fft.fft(fft), 0)

    @staticmethod
    def peak_location(fft, nu, location, ppm_range):
        # finds the highest peak from location +- ppm_range
        peaks = (-np.abs(fft)).argsort()[:int(len(nu) / 100)]
        for peak in nu[peaks]:
            if location + ppm_range >= peak >= location - ppm_range:
                return peak
        return None

    @staticmethod
    def water_peak_location(fft, nu):
        return Spectrum.peak_location(fft, nu, WATER_REFERENCE, 0.5)

    @staticmethod
    def naa_peak_location(fft, nu):
        return Spectrum.peak_location(fft, nu, NAA_REFERENCE, 0.5)

    @staticmethod
    def remove_water_peak(fft, nu, ppm_range=0.6):
        # find the peak then set the range centered around it to the median signal of the fft
        water_peak_loc = Spectrum.water_peak_location(fft, nu)
        ppm_range = float(ppm_range)
        mean_abs = np.mean(np.abs(fft))
        abs_fft = np.abs(fft)
        under_mean = 0
        if water_peak_loc is not None:
            mean = np.mean(np.real(fft)) + 1j * np.mean(np.imag(fft))
            for jj in range(0, len(fft)):
                if (water_peak_loc - (ppm_range / 2) < nu[jj]) and (water_peak_loc + (ppm_range / 2) > nu[jj]):
                    if abs_fft[jj] > mean_abs:
                        under_mean = 0
                        fft[jj] = mean
                    if nu[jj] > water_peak_loc and abs_fft[jj] < mean_abs:
                        # trailing edge detection, clip it as soon as the water peak (in absolute terms is over).
                        under_mean += 1
                        if under_mean > 5:
                            return fft
        return fft
