#!/usr/bin/env python3
#
# basis.py - MRSNet - spectral basis set processing
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2019, Max Chandler, PhD student at Cardiff University
# Copyright (C) 2020, Frank C Langbein <frank@langbein.org>, Cardiff University

import os
import copy
import random
import subprocess
import dill
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from spectrum import Spectrum
from utilities.constants import GYROMAGNETIC_RATIO
from simulators.fida.fida_simulator import fida_spectra

def basis(args):
    bases = generate_bases(args.basis_source, args.scanner_manufacturer, args.omega, args.linewidths, args.metabolites, args.verbose)
    if args.plot or args.save_plot:
        for b in bases:
            b.plot(args.save_plot)

def generate_bases(basis_sources, scanner_manufacturer, omega, linewidths, metabolites, verbose):
    """Get basis set for metabolites, one per linewidth in list"""
    if not isinstance(basis_sources,list):
        basis_sources=[basis_sources]
    basis = []
    for basis_source in basis_sources:
        if basis_source == 'lcmodel':
            if len(linewidths) != 1 or linewidths[0] != 1.0:
                raise Exception('Cannot supply LCModel basis set with linewidths argument. It is not a simulator option; it has one fixed linewidth.')
            if scanner_manufacturer == 'siemens':
                basis.append(Basis.load_lcm_basis(omega=omega, metabolite_names=metabolites, megapress=True,
                                              edit_off=os.path.join('data', 'basis', 'lcmodel', 'MEGAPRESS_edit_off_Siemens_3T.basis'),
                                              difference=os.path.join('data', 'basis', 'lcmodel', 'MEGAPRESS_difference_Siemens_3T_kasier.basis')))
            elif scanner_manufacturer == 'ge':
                basis.append(Basis.load_lcm_basis(omega=omega, metabolite_names=metabolites, megapress=True,
                                              edit_off=os.path.join('data', 'basis', 'lcmodel', 'MEGAPRESS_edit_off_GE_3T.basis'),
                                              difference=os.path.join('data', 'basis', 'lcmodel', 'MEGAPRESS_difference_GE_3T_kasier.basis')))
            elif scanner_manufacturer == 'phillips':
                basis.append(Basis.load_lcm_basis(omega=omega, metabolite_names=metabolites, megapress=True,
                                              edit_off=os.path.join('data', 'basis', 'lcmodel', 'MEGAPRESS_edit_off_Phillips_3T.basis'),
                                              difference=os.path.join('data', 'basis', 'lcmodel', 'MEGAPRESS_difference_Phillips_3T_kasier.basis')))
            else:
                raise Exception('No LCModel basis set for ' + scanner_manufacturer + ' scanner.')
        elif basis_source == 'fid-a':
            if scanner_manufacturer == 'siemens':
                for linewidth in linewidths:
                    basis.append(Basis.load_fida(metabolites, omega=omega, linewidth=linewidth))
            else:
                raise Exception('No FID-A simulator for ' + scanner_manufacturer + ' scanner.')
        elif basis_source == 'pygamma':
            if scanner_manufacturer == 'siemens':
                for linewidth in linewidths:
                    basis.append(Basis.load_pygamma(metabolites, omega=omega, pulse_sequence='megapress', linewidth=linewidth))
            else:
                raise Exception('No PyGamma simulator for ' + scanner_manufacturer + ' scanner.')
        else:
            raise Exception('Unknown basis source: ' + basis_source)
    return basis

class Basis(object):
    """
        The Basis object is used to contain sets of single metabolite spectra,
        and is used to generate combined spectra of them.

        It contains methods for checking that the basis is valid, and exporting
        combined spectra.
    """

    def __init__(self):
        self.spectra = [] # List of spectra in basis
        self.metadata = {} # For LCM basis only
        # Cache for global values for basis from the spectra in the basis list
        self._source = None
        self._pulse_sequence = None
        self._linewidths = None

    def source(self):
        if self._source is None:
            self._source = self.spectra[0].source
            for s in self.spectra:
                if s.source != self._source:
                    self._source = 'mixed'
                    break
        return self._source

    def pulse_sequence(self):
        if self._pulse_sequence is None:
            self._pulse_sequence = self.spectra[0].pulse_sequence
            for s in self.spectra:
                if s.pulse_sequence != self._pulse_sequence:
                    self._pulse_sequence = 'mixed'
                    break
        return self._pulse_sequence

    def linewidths(self):
        if self._linewidths == None:
            self._linewidths = list(set([s.linewidth for s in self.spectra]))
        return self._linewidths

    def acquisition_numbers(self):
        return list(set([s.acquisition for s in self.spectra]))

    def spectra_ids(self):
        return list(set([s.id for s in self.spectra]))

    def metabolite_names(self):
        names = []
        for s in self.spectra:
            names.extend(s.metabolite_names)
        return list(set(names))

    def invalidate_spectra_fft_cache(self):
        for ii in range(len(self.spectra)):
            self.spectra[ii]._need_fft_cache_update = True

    def add_spectra(self, spectra):
        self.spectra.append(spectra)
        self._source = None
        self._pulse_sequence = None
        self._linewidths = None

    def remove_spectra(self, metabolites_to_keep_names=None):
        if metabolites_to_keep_names:
            metabolites_to_keep_names = [x.lower() for x in metabolites_to_keep_names]
            for spectrum in self.spectra:
                if spectrum.metabolite_names[0].lower() not in metabolites_to_keep_names:
                    self.spectrum.remove(spectra)
            self.validate(metabolites_to_keep_names)

    def validate(self, metabolite_names):
        # Verify we have a complete basis set
        if metabolite_names is not None:
            if len(self.metabolite_names()) != len(metabolite_names) or \
                   len(self.spectra) % len(self.metabolite_names()) != 0:
                raise Exception(
                    'Loaded basis does not have enough metabolites or acquistions to complete list of metabolites. \n' + \
                    '   Required: ' + str(metabolite_names) + '\n' + \
                    '   Available: ' + str(self.metabolite_names()) + '\n' \
                    '   Missing: ' + str(list(set(metabolite_names).difference(self.metabolite_names()))))

    def assign_ids(self):
        metabolite_names = []
        acquisitions = []
        for ii in range(0, len(self.spectra)):
            metabolite_names.append(self.spectra[ii].metabolite_names)
            acquisitions.append(self.spectra[ii].acquisition)
        acquisitions = max(acquisitions) + 1  # indexing starts at 0, but counting starts at 1

        # get unique list of lists
        metabolite_names = [list(x) for x in set(tuple(x) for x in metabolite_names)]

        for metab_names in metabolite_names:
            id = hash(random.random())
            count = 0
            for ii in range(0, len(self.spectra)):
                if self.spectra[ii].metabolite_names == metab_names:
                    self.spectra[ii].id = id
                    count += 1
            if count != acquisitions:
                raise Exception('Could not find the correct number of acquisitions for spectra containing metabolites: ' + str(metab_names))

    def group_spectra_by_acquisition(self):
        # Group spectra in basis by acquisition number
        spectra = {}
        for ii in range(0, len(self.spectra)):
            acq = self.spectra[ii].acquisition
            if acq not in spectra:
                spectra[acq] = []
            spectra[acq].append(self.spectra[ii])

        # Sort the acquisitions so the ids are aligned
        for acquisition in spectra.keys():
            ids = []
            for spec in spectra[acquisition]:
                ids.append(spec.id)
            spectra[acquisition] = np.array(spectra[acquisition])
            spectra[acquisition] = spectra[acquisition][np.argsort(ids)]

        # Sort the spectra in each group alphabetically
        metabolite_names = []
        for spec in spectra[list(spectra.keys())[0]]:
            metabolite_names.append(spec.metabolite_names[0])
        name_index = np.argsort(metabolite_names)
        for key in spectra.keys():
            spectra[key] = spectra[key][name_index]

        # Check consistency
        n_in_groups = len(spectra[list(spectra.keys())[0]])
        for key in spectra.keys():
            if len(spectra[key]) != n_in_groups:
                raise Exception('Number of spectra in each acquisition group does not match!')
        for ii in range(0, len(spectra[0])):
            s_id = spectra[0][ii].id
            for key in spectra.keys():
                if spectra[key][ii].id != s_id:
                    raise Exception('Spectra IDs across acquisitions do not align')

        return spectra

    def group_spectra_by_id(self):
        ids = self.spectra_ids()
        spec = self.group_spectra_by_acquisition()
        spectra = []
        for id in ids:
            id_spectra = []
            for acq in sorted(spec.keys()):
                for acq_spectra in spec[acq]:
                    if acq_spectra.id == id:
                        id_spectra.append(acq_spectra)
            spectra.append(id_spectra)
        return spectra

    def group_spectra_by_id_and_acq(self):
        ids = self.spectra_ids()
        spec = self.group_spectra_by_acquisition()
        spectra = {}
        for id in ids:
            spectra[id] = []
            for acq in sorted(spec.keys()):
                id_spectra = []
                for acq_spectra in spec[acq]:
                    if acq_spectra.id == id:
                        id_spectra.append(acq_spectra)
                spectra[id].append(id_spectra)
        return spectra

    def add_difference_spectra(self):
        if self.acquisition_numbers() != [0, 1, 2]:
            self.check()
            if self.pulse_sequence() != 'megapress':
                raise Exception('This is only designed to be used with the megapress pulse sequence.')

            acq_numbers = self.acquisition_numbers()
            if self.acquisition_numbers() not in [[0, 1], [0, 2]]:
                raise Exception('The acquisition numbers for this basis set do not match [0,1] or [0,2]: '
                                + str(acq_numbers) + '. Please define the behaviour for the mix you have supplied.')

            self.normalise()
            acqs = self.group_spectra_by_acquisition()

            new_spectra_list = []
            if acq_numbers == [0, 1]:
                for ii, edit_off in enumerate(acqs[0]):
                    edit_on = None
                    for spectra in acqs[1]:
                        if spectra.id == edit_off.id:
                            edit_on = spectra
                            break

                    if edit_on is None:
                        raise Exception('Could not find corresponding edit on spectra for edit on with id: ' + str(edit_off.id))

                    # Now we have both, create the difference spectra object then edit it.
                    difference = copy.deepcopy(acqs[0][ii])
                    # https: // www.ncbi.nlm.nih.gov / pmc / articles / PMC3825742 /
                    difference._raw_adc = edit_on._raw_adc - edit_off._raw_adc

                    difference.acquisition = 2
                    difference.spectrum_type = 'difference'
                    difference.scale = 1.0
                    new_spectra_list.extend([edit_off, edit_on, difference])
            else:
                raise Exception('Cannot add difference spectra if acq numbers are not [0,1]')

            self.spectra = new_spectra_list
            self.check()
            self.normalise()

    def setup(self, metabolite_names=None):
        self.remove_spectra(metabolite_names)
        self.check()
        self.normalise()
        self.correct_b0()

    def check(self):
        if len(self.spectra) == 0:
            raise Exception('Basis has no spectra associated with it!')

        # Check if spectra in basis are consistent
        if self.pulse_sequence() not in ['fid', 'steam', 'press', 'megapress']:
            raise Exception('Unknown basis pulse sequence: ' + self.pulse_sequence())
        if len(self.linewidths()) != 1:
            raise Exception('More than one linewidth in this basis!')
        omega = self.spectra[0].omega
        for spectrum in self.spectra:
            spectrum.check()
            if spectrum.linewidth is None:
                raise Exception('Linewidth is not set for spectra in basis!')
            if abs(spectrum.omega - omega) > 1:
                raise Exception('Not all spectra in this basis share the same scanner frequency. '
                                '(Difference in B0 is larger than 1MHz)')
            if spectrum.id is None:
                raise Exception('Please set spectra IDs. This is used to verify that spectra of the same type but with different '
                                'acquisitions can be grouped together.')

        # Checking that each spectrum in the basis has the same number of acquisitons,
        # and that they don't have the same number.
        n_acq = {}
        for id in self.spectra_ids():
            for spectrum in self.spectra:
                if spectrum.id == id:
                    if id not in n_acq.keys():
                        n_acq[id] = 0
                    else:
                        n_acq[id] += 1
        n_acq = set(n_acq.values())
        if len(n_acq) != 1:
            raise Exception('Spectra do not have the same number of acquisitons! ' + str(n_acq))
        n_acq = list(n_acq)[0] + 1

        for id in self.spectra_ids():
            # Check to make sure each spec with a unique ID has the same number of acquistions as the rest
            # e.g. GABA, Cr,... all have acq [0,1,2]
            seen_acq = []
            for s in self.spectra:
                if s.id == id:
                    seen_acq.append(s.acquisition)
            if not all(np.sort(seen_acq) == np.linspace(0, n_acq-1, n_acq)):
                raise Exception('Looks like acusition numbers for id:"' + str(id) + '" are not linearly increasing: ' +
                                str(seen_acq) + ' should be [0, 1, ..., n]')
            if len(set(seen_acq)) != len(seen_acq):
                raise Exception('ID: ' + str(id) + ' has more than one spectrum with the same acquisition number')
            if len(set(seen_acq)) != n_acq:
                raise Exception('ID: ' + str(id) + ' doesn\'t have enough acquisitions compared to the rest of the basis set: '
                                + str(len(set(seen_acq))) + '/' + str(n_acq))

        if self.pulse_sequence() == 'megapress':
            on_count = 0
            off_count = 0
            diff_count = 0
            for spectrum in self.spectra:
                if spectrum.spectrum_type == 'edit off':
                    if spectrum.acquisition != 0:
                        raise Exception('MEGA-PRESS spectra of type "edit off" does not have an acquisition of 0... It should do!')
                    off_count += 1
                elif spectrum.spectrum_type == 'edit on':
                    if spectrum.acquisition != 1:
                        raise Exception('MEGA-PRESS spectra of type "edit on" does not have an acquisition of 1... It should do!')
                    on_count += 1
                elif spectrum.spectrum_type == 'difference':
                    if spectrum.acquisition != 2:
                        raise Exception('MEGA-PRESS spectra of type "diff" does not have an acquisition of 2... It should do!')
                    diff_count += 1
            if on_count != off_count:
                raise Exception('Megapress basis set does not have an even number of edit on and edit off spectra!')
            if diff_count > 0 and on_count != diff_count:
                raise Exception('Megapress basis set does not have an even number of edit on and difference spectra!')

    def normalise(self, mode='normal'):
        # All spectra are normalised against the maximum absloute adc signal in the basis set.
        # There are a number of reasons, but it means that the noise added to the ADC has the same mu and sigma values.
        if mode == 'normal':
            adcs = []
            global_max = 0.0
            for ii in range(len(self.spectra)):
                global_max = np.max([global_max, np.max(np.abs(self.spectra[ii]._raw_adc))])
            for ii in range(len(self.spectra)):
                self.spectra[ii]._raw_adc = self.spectra[ii]._raw_adc / global_max
        elif mode == 'flatten':
            # This is used for the basis rescaling routines, so we have a reference starting point of 1 for each of the spectra
            for ii in range(len(self.spectra)):
                self.spectra[ii]._raw_adc = self.spectra[ii]._raw_adc / np.max(np.abs(self.spectra[ii]._raw_adc))
        else:
            raise Exception('Unknown basis normalisaiton mode: %s' % mode)
        self.invalidate_spectra_fft_cache()

    def correct_b0(self):
        if self.source() in ['fid-a', 'lcmodel', 'pygamma']:
            # There's not going to be an individual shift per metabolite...
            # so we calibrate the entire set against Cr or Naa
            b0_shift = None
            for ii in range(len(self.spectra)):
                if ('creatine' in self.spectra[ii].metabolite_names) or \
                   ('n-acetylaspartate' in self.spectra[ii].metabolite_names):
                    self.spectra[ii].correct_b0()
                    b0_shift = self.spectra[ii]._b0_ppm_shift
                    break
            if b0_shift is not None:
                for ii in range(len(self.spectra)):
                    self.spectra[ii].correct_b0(b0_shift)
            else:
                for ii in range(len(self.spectra)):
                    self.spectra[ii].correct_b0()
        elif self.source() == 'dicom':
            for ii in range(len(self.spectra)):
                self.spectra[ii].correct_b0()
        elif self.source() == 'mixed':
            raise Exception('Cannot B0 correct a mixed basis')
        else:
            raise Exception('Unrecognised source for B0 correction routine')

    def correct_baseline(self):
        for s in self.spectra:
            s._correct_baseline = True
            s.correct_baseline()

    def update_spectra_scale(self, concentrations=None, metabolite_names=None):
        if concentrations is None:
            if metabolite_names is not None:
                raise Exception('Undefined behaviour - why pass metabolite names but not concentrations?')
            else:
                concentrations = np.ones(len(self.spectra), dtype=float)

        if metabolite_names is None:
            if len(concentrations) != len(self.spectra):
                raise Exception('Length of concentraiton array does not match number of spectra')
            for ii in range(0, len(self.spectra)):
                self.spectra[ii].scale = concentrations[ii]
        else:
            if len(concentrations) != len(metabolite_names):
                raise Exception('Length of concentraiton array does not match number of metabolite names')
            for ii in range(0, len(self.spectra)):
                if isinstance(metabolite_names, np.ndarray):
                    metabolite_names = metabolite_names.tolist()
                if self.spectra[ii].metabolite_names[0] in metabolite_names:
                    self.spectra[ii].scale = concentrations[metabolite_names.index(self.spectra[ii].metabolite_names[0])]
                else:
                    self.spectra[ii].scale = 0.0

    def export_combination(self, concentrations, metabolite_names=None, acquisitions=None):
        linewidth = self.linewidths()
        if len(linewidth) != 1:
            raise Exception('Basis set has more than one linewidth: ' + str(linewidth) +
                            '. Mixing linewidths in an export is not allowed! (Why would you want to do this?)')
        linewidth = linewidth[0]

        if acquisitions is not None and not (set(acquisitions) <= set(self.acquisition_numbers())):
            raise Exception('Acqusition numbers ' + str(set(acquisitions)) + ' not in basis set ' + str(self.acquisition_numbers()))

        if metabolite_names is None:
            metabolite_names = self.metabolite_names()

        self.update_spectra_scale(concentrations, metabolite_names)

        total_adc = self.total_raw_adc()
        combinations = []
        id = hash(random.random())
        grouped_spectra = self.group_spectra_by_acquisition()
        for ii, acq_number in enumerate(sorted(grouped_spectra.keys())):
            if acquisitions is None or  acq_number in acquisitions:
                spec = copy.deepcopy(grouped_spectra[acq_number][0])
                spec.id = id
                spec.scale = 1.0
                spec.metabolite_names = metabolite_names
                spec.concentrations = concentrations
                spec.acquisition = acq_number
                spec.linewidth = linewidth
                spec._raw_adc = total_adc[ii]
                spec._adc_noise = None
                spec._need_fft_cache_update = True
                spec.check()
                combinations.append(spec)
        return combinations

    def total_rescaled_fft(self, high_ppm=-4.5, low_ppm=-0.5, npts=2048):
        grouped_spectra = self.group_spectra_by_acquisition()
        total_fft = []
        for ii, key in enumerate(sorted(grouped_spectra.keys())):
            for jj, s in enumerate(grouped_spectra[key]):
                fft, nu = s.rescale_fft(high_ppm=high_ppm, low_ppm=low_ppm, npts=npts)
                if jj == 0:
                    total_fft.append(fft)
                else:
                    total_fft[ii] += fft
        return normalise_signal(total_fft), nu

    def total_trimmed_fft(self, high_ppm=-4.5, low_ppm=-0.5):
        grouped_spectra = self.group_spectra_by_acquisition()
        total_fft = []
        for ii, key in enumerate(sorted(grouped_spectra.keys())):
            for jj, s in enumerate(grouped_spectra[key]):
                re_fft, nu = s.trim_fft(high_ppm=high_ppm, low_ppm=low_ppm)
                if jj == 0:
                    total_fft.append(re_fft)
                else:
                    total_fft[ii] += re_fft
        return normalise_signal(total_fft), nu

    def total_fft(self):
        grouped_spectra = self.group_spectra_by_acquisition()
        total_fft = []
        for ii, key in enumerate(sorted(grouped_spectra.keys())):
            for jj, s in enumerate(grouped_spectra[key]):
                re_fft = s.fft()
                if jj == 0:
                    total_fft.append(re_fft)
                else:
                    total_fft[ii] += re_fft
        return normalise_signal(total_fft)

    def total_adc(self):
        grouped_spectra = self.group_spectra_by_acquisition()
        total_adc = []
        for ii, key in enumerate(sorted(grouped_spectra.keys())):
            for jj, s in enumerate(grouped_spectra[key]):
                adc = s.adc()
                if jj == 0:
                    total_adc.append(adc)
                else:
                    total_adc[ii] += adc
        return total_adc

    def total_raw_adc(self):
        # returns n arrays, where n is the number of acquisitions. They are in numerical order : [0,1,2,...]
        grouped_spectra = self.group_spectra_by_acquisition()
        total_adc = []
        for ii, key in enumerate(sorted(grouped_spectra.keys())):
            for jj, s in enumerate(grouped_spectra[key]):
                adc = s.raw_adc()
                if jj == 0:
                    total_adc.append(adc)
                else:
                    total_adc[ii] += adc
        return total_adc

    def save(self, directory, name):
        if not os.path.isdir(directory):
            os.makedirs(directory)
        if not name.endswith('.dill'):
            name += '.dill'
        filepath = os.path.join(directory, name)
        with open(filepath, 'wb') as out_file:
            dill.dump(self, out_file)
        return filepath

    def save_to_mat(self, name):
        if not name.endswith('.mat'):
            name += '.mat'
        dict = {}
        dict[self.source() + '_nu'] = self.spectra[0].nu()
        for s in self.spectra:
            name = self.source() + '_' + matlabify_string(s.metabolite_names[0] + '_' + s.spectrum_type)
            dict[name + '_fft'] = s.fft()
        savemat(name, dict)

    @staticmethod
    def load(filepath, metabolite_names=None):
        if not os.path.exists(filepath):
            raise Exception('Basis file does not exist: ')
        with open(filepath) as in_file:
            basis = dill.load(in_file)
        basis._filename = filepath.split('/')[-1]
        if metabolite_names is not None:
            for spectra in basis.spectra:
                spectra_to_keep = []
                if spectra.metabolite_names[0] in metabolite_names:
                    spectra_to_keep.append(spectra)
            basis.spectra = spectra_to_keep
            basis.setup(metabolite_names=metabolite_names)
        return basis

    @staticmethod
    def load_fida(metabolite_names, omega, linewidth=1.0, directory=os.path.join('data', 'basis', 'fida'), second_call=False):
        if not os.path.exists(directory):
            os.makedirs(directory)

        basis = Basis()
        to_simulate = copy.copy(metabolite_names)
        for file in os.listdir(directory):
            if file.endswith('.mat'):
                spec = Spectrum.load_fida(os.path.join(directory,file))
                if (spec.metabolite_names[0].lower() in [x.lower() for x in metabolite_names]) \
                   and (spec.linewidth == linewidth) \
                   and (np.abs(spec.omega - omega) < 1e-8):   # there are rounding errors for storing the B0
                    basis.add_spectra(spec)
                    if spec.metabolite_names[0] in to_simulate:
                        to_simulate.remove(spec.metabolite_names[0])

        if to_simulate:
            if second_call:
                raise Exception('Recursion error, should have simulated spectra - but I can\'t seem to find it and '
                                'I\'m going to end up in an endless loop.')
            else:
                print('Some spectra are missing, simulating: ' + str(to_simulate))
                fida_spectra(to_simulate, omega=omega, linewidth=linewidth)
                basis = Basis.load_fida(metabolite_names, omega, linewidth, directory, second_call=True)

        basis.assign_ids()
        basis.add_difference_spectra()
        basis.setup(metabolite_names=metabolite_names)
        return basis

    @staticmethod
    def load_pygamma(metabolite_names, omega, pulse_sequence='megapress', linewidth=1.0,
                     directory=os.path.join('data', 'basis', 'pygamma'), second_call=False):
        linewidth = float(linewidth)
        to_sim = list(metabolite_names)
        basis = Basis()
        # Constants, synchronise with pygamma_simulator (passed as arguments, but defaults hardcoded in pygamma simulator as well)
        npts = 4096
        adc_dt = 4e-4
        for metabolite_name in metabolite_names:
            cache_name = pulse_sequence + '_' + str(omega) + '_' + str(linewidth) + '_' + str(npts) + '_' + str(adc_dt)
            spectra_cache_dir = os.path.join(directory, cache_name)
            if os.path.isdir(spectra_cache_dir):
                for file in os.listdir(spectra_cache_dir):
                    if file == metabolite_name + '.json':
                        with open(os.path.join(spectra_cache_dir, file), 'rb') as load_file:
                            for raw in json.load(load_file):
                                spectrum = Spectrum()
                                spectrum.source = 'pygamma'
                                spectrum.type = 'simulated'
                                spectrum.pulse_sequence = pulse_sequence.lower()
                                spectrum.metabolite_names = [metabolite_name]
                                spectrum.concentrations = [1]
                                spectrum.omega = omega
                                spectrum.npts = npts
                                spectrum.dt = adc_dt
                                spectrum.linewidth = linewidth
                                spectrum._raw_adc = np.array(raw["adc_re"]) + 1j * np.array(raw["adc_im"])
                                spectrum.center_ppm = 0
                                spectrum.acquisition = raw["count"]
                                if spectrum.pulse_sequence == 'megapress':
                                    if raw["count"] == 0:
                                        spectrum.spectrum_type = 'edit off'
                                    elif raw["count"] == 1:
                                        spectrum.spectrum_type = 'edit on'
                                    else:
                                        raise Exception('More than 2 mx objects for megapress? Something is wrong here...')
                                basis.add_spectra(spectrum)
                        to_sim.remove(metabolite_name)
            else:
                os.makedirs(spectra_cache_dir)

        if to_sim:
            if second_call:
                raise Exception('Recursion error, should have simulated spectra - but I can\'t seem to find it and '
                                'I\'m going to end up in an endless loop.')
            for metabolite_name in to_sim:
                pygamma_cmd = ['/usr/bin/env', 'python2', os.path.join('simulators', 'pygamma', 'pygamma_simulator.py'),
                               metabolite_name, str(omega), pulse_sequence, str(linewidth), str(npts), str(adc_dt), spectra_cache_dir]
                try:
                    p = subprocess.Popen(pygamma_cmd)
                except OSError as e:
                    raise Exception('PyGamma simulations failed!') from e
                p.wait()
            basis = Basis.load_pygamma(metabolite_names, omega, pulse_sequence, linewidth, directory, second_call=True)

        basis.assign_ids()
        if pulse_sequence.lower() == 'megapress':
            basis.add_difference_spectra()
        basis.setup(metabolite_names=metabolite_names)
        return basis

    @staticmethod
    def load_lcm_basis(omega, basis_files=None, metabolite_names=None, megapress=False, edit_off=None, difference=None):
        # Warning, this is an ugly one...
        # http://s-provencher.com/pub/LCModel/manual/manual.pdf
        # each one of the basis files is an aquisition
        # feed the basis files in the order the of the acusitisions
        total_basis = None
        other_basi = []
        if megapress:
            if edit_off is None:
                raise Exception('If importing megapress, please pass the edit off basis file using the edit_off variable.')
            if difference is None:
                raise Exception('If importing megapress, please pass the difference basis file using the difference variable.')
            basis_files = [edit_off, difference]

        for acquisition, basis_file in enumerate(basis_files):
            if not os.path.exists(basis_file):
                raise Exception('Basis file does not exist: ' + basis_file)
            basis = Basis()

            with open(basis_file) as file:
                line_buffer = []
                for count, line in enumerate(file):
                    if count == 0 and "$SEQPAR" not in line:
                        raise IOError('File does not appear to be a valid ".basis" file, no "$SEQPAR" found at start.')
                    if '$NMUSED' in line and len(basis.metadata) == 0:
                        # setup the metadata from the basis file - this is the first section
                        for b_line in line_buffer:
                            if '=' in b_line:
                                for to_remove in [' ', ',', '   ', '\n', '$END']:
                                    b_line = b_line.replace(to_remove, '')
                                split_line = b_line.split('=')
                                if split_line[0] in ['ECHOT', 'HZPPPM', 'FWHMBA', 'BADELT']:
                                    basis.metadata[split_line[0]] = float(split_line[1])
                                elif split_line[0] in ['NDATAB']:
                                    basis.metadata[split_line[0]] = int(split_line[1])
                                else:
                                    basis.metadata[split_line[0]] = split_line[1]
                        line_buffer = []

                        # set up the pulse sequence parameter
                        pulse_sequence = basis.metadata['SEQ']
                        if pulse_sequence in ["'MEGA-'"]:
                            pulse_sequence = 'megapress'
                        else:
                            raise Exception('Unrecognised LCM pulse sequence: ' + pulse_sequence)

                        if np.abs(basis.metadata['HZPPPM'] - omega) > (GYROMAGNETIC_RATIO/5):
                            # more than a 0.2T difference, there's an issue
                            raise Exception('LCModel basis set (%.2fT) is more than 0.2T different to prescibed '
                                            'omega (%.2fT).' % (basis.metadata['HZPPPM']/GYROMAGNETIC_RATIO,
                                                                omega/GYROMAGNETIC_RATIO))

                    elif len(line_buffer) > 1 and '$NMUSED' in line:
                        # from the second NMUSED is the marker for the start of the metabolite section
                        # I belive these sections are the same as individual ".BASIS" files.
                        # We collect a line buffer until we hit the next one, and pass the buffer to Spectrum.load_lcm
                        spec = Spectrum.load_lcm_basis(line_buffer, basis.metadata['BADELT'], basis.metadata['HZPPPM'])
                        spec.pulse_sequence = pulse_sequence
                        if spec.pulse_sequence == 'megapress':
                            if acquisition == 0:
                                spec.spectrum_type = 'edit off'
                                spec.acquisition = acquisition
                            elif acquisition == 1:
                                spec.spectrum_type = 'difference'
                                spec.acquisition = 2
                        if metabolite_names:
                            if spec.metabolite_names[0].lower() in [x.lower() for x in metabolite_names]:
                                basis.add_spectra(spec)
                        else:
                            basis.add_spectra(spec)
                        line_buffer = []
                    line_buffer.append(line)
                file.close()

                if len(basis.spectra) == 0:
                    raise Exception('No spectra loaded from basis file.')

            if megapress:
                if acquisition == 0:
                    total_basis = copy.deepcopy(basis)
                elif acquisition == 1:
                    other_basi.append(basis)
                else:
                    raise Exception('To many input acquisitions for megapress, should only be 2 (edit off, difference)')
            else:
                if acquisition == 0:
                    total_basis = copy.deepcopy(basis)
                else:
                    if len(total_basis.spectra) != len(basis.spectra):
                        raise Exception('One of the other basis has a different number of spectra')
                other_basi.append(basis)

        # now we have to merge the different loaded basis sets...
        # we need to set the IDs to be the same and the acquisiton numbers in the same order as the basis sets
        # were added

        if megapress:
            # first things first, if this is a megapress basis set, we need to sort this out.
            # LCM basis sets are edit off and difference, so we need to work backwards to get the edit on and fill
            # in the blanks
            edit_on_basis = Basis()
            difference_basis = other_basi[0]
            edit_off_metabolites = [x.lower() for x in total_basis.metabolite_names()]

            for difference_spectra in difference_basis.spectra:
                if difference_spectra.metabolite_names[0].lower() in edit_off_metabolites:
                    # the idea is to remove them from the list, so we know which spectra we need to invert
                    edit_off_metabolites.remove(difference_spectra.metabolite_names[0].lower())
                    for edit_off in total_basis.spectra:
                        if edit_off.metabolite_names[0].lower() == difference_spectra.metabolite_names[0].lower():
                            edit_on = copy.deepcopy(edit_off)
                            edit_on.spectrum_type = 'edit on'
                            edit_on.acquisition = 1
                            # diff = on - off. So on = diff + off
                            edit_on._raw_adc = difference_spectra._raw_adc + edit_off._raw_adc
                            edit_on_basis.add_spectra(edit_on)
                            break

            for metabolite_name in edit_off_metabolites:
                # these metabolites were not found in the difference basis set, meaning they have the same signal
                # for edit on & off (Cr as an example). so we copy the object from the edit off to edit on basis
                for edit_off in total_basis.spectra:
                    if edit_off.metabolite_names[0].lower() == metabolite_name:
                        edit_on = copy.deepcopy(edit_off)
                        edit_on.acquisition = 1
                        edit_on.spectrum_type = 'edit on'
                        edit_on_basis.add_spectra(edit_on)

            other_basi = [edit_on_basis]

        # setup the IDs for the metabolite spectra
        spec_info = {}
        for ii in range(0, len(total_basis.spectra)):
            total_basis.spectra[ii].id = hash(random.random())
            spec_info[total_basis.spectra[ii].metabolite_names[0].lower()] = {'id': total_basis.spectra[ii].id, 'acq': 0}

        # copy the IDs to the other loaded basis sets
        for ii in range(0, len(other_basi)):
            names_cpy = [x.lower() for x in metabolite_names]
            for jj in range(0, len(other_basi[ii].spectra)):
                spectra = other_basi[ii].spectra[jj]
                m_name = spectra.metabolite_names[0].lower()
                spectra.id = spec_info[m_name]['id']
                spec_info[m_name]['acq'] += 1
                spectra.acquisition = spec_info[m_name]['acq']
                if names_cpy:
                    if m_name in names_cpy:
                        total_basis.add_spectra(spectra)
                        names_cpy.remove(m_name.lower())
                else:
                    total_basis.add_spectra(spectra)

        if len(names_cpy):
            raise Exception('Basis set is missing metabolites from metabolite_list: ' + str(names_cpy))

        if pulse_sequence == 'megapress':
            total_basis.add_difference_spectra()
        total_basis.setup(metabolite_names=metabolite_names)
        return total_basis

    @staticmethod
    def load_dicom(directory, metabolite_names=None, pulse_sequence='megapress'):
        basis = Basis()
        dicom_files = []
        for dir, subdirs, files in os.walk(directory):
            for file in sorted(files):
                if file[-4:].lower() == '.ima':
                    dicom_files.append(os.path.join(dir, file))
        if not len(dicom_files):
            raise Exception('No IMA files found')

        for count, dicom_file in enumerate(dicom_files):
            spectrum = Spectrum.load_dicom(dicom_file)
            if metabolite_names is not None:
                if spectrum.metabolite_names:
                    if not all([name.lower() in [x.lower() for x in metabolite_names] for name in spectrum.metabolite_names]):
                        continue
            if spectrum.pulse_sequence == pulse_sequence:
                basis.add_spectra(spectrum)

        if basis.pulse_sequence() == 'megapress':
            basis.add_difference_spectra()
        basis.setup(metabolite_names=metabolite_names)
        return basis

    def plot(self, save_plot=False, save_name=None):
        acquisitions = self.group_spectra_by_acquisition()
        for acq_number, acquisition in enumerate(acquisitions):
            n_rows = len(acquisitions[acquisition])
            n_cols = 3
            fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey='row', figsize=(19.2, 10.8), dpi=300)
            if len(axes.shape) == 1:
                axes = np.reshape(axes, (1,n_cols))

            super_title = 'Basis: ' + self.source() + ' ' + self.pulse_sequence()
            fname = self.source() + '_' + self.pulse_sequence()
            if self.pulse_sequence() == 'megapress':
                super_title += ' ' + str(acquisitions[acquisition][0].spectrum_type)
                fname += '_' + str(acquisitions[acquisition][0].spectrum_type).replace(' ','-')
            super_title +=  ' (Acq ' + str(acq_number) + '), ' + str(round((self.spectra[0].nu()[-1] - self.spectra[0].nu()[0]) * self.spectra[0].omega)) + 'Hz' + ', Linewidth:' + str(self.spectra[0].linewidth)
            fname += '-' + str(acq_number) + '_' + str(round((self.spectra[0].nu()[-1] - self.spectra[0].nu()[0]) * self.spectra[0].omega)) + 'Hz' + '_' + str(self.spectra[0].linewidth)
            plt.suptitle(super_title)

            axes[0, 0].set_xlim([-4.5, -1])
            for jj in range(0, len(acquisitions[acquisition])):
                spectra = acquisitions[acquisition][jj]
                if jj == 0:
                    axes[jj, 0].set_title('Magnitude')
                axes[jj, 0].set_ylabel(spectra.metabolite_names[0], rotation=0, labelpad=50, size='large')
                axes[jj, 0].plot(spectra.nu(), np.abs(spectra.fft()), label='FFT')
                if jj == 0:
                    axes[jj, 1].set_title('Real')
                axes[jj, 1].plot(spectra.nu(), np.real(spectra.fft()), label='FFT')
                if jj == 0:
                    axes[jj, 2].set_title('Imaginary')
                axes[jj, 2].plot(spectra.nu(), np.imag(spectra.fft()), label='FFT')
            axes[n_rows - 1, 0].set_xlabel('PPM')
            axes[n_rows - 1, 1].set_xlabel('PPM')
            axes[n_rows - 1, 2].set_xlabel('PPM')

            if save_plot:
                if save_name is None:
                    sname = fname
                else:
                    sname = save_name
                if not sname.endswith('.png'):
                    sname += ".png"
                plt.savefig(os.path.join('data','basis',sname))
                plt.close()
            else:
                plt.show()
