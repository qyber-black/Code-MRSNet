#!/usr/bin/env python3
#
# dataset.py - MRSNet - spectra dataset processing
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2019, Max Chandler, PhD student at Cardiff University
# Copyright (C) 2020, Frank C Langbein <frank@langbein.org>, Cardiff University

import math
import numpy as np
import multiprocessing
import sobol_seq
import json
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
from tqdm import tqdm
from itertools import combinations

from utilities.utils import normalise_labels, convert_molecule_names
from basis import generate_bases

def spectra(args):
    bases = generate_bases(args.basis_source, args.scanner_manufacturer, args.omega, args.linewidths, args.metabolites, args.verbose)
    datasets = generate_datasets(bases, 'testing', args.num, args.gen, args.metabolites, args.acquisitions, args.verbose)
    if args.save is not None:
        js = []
        n = 1
        for ds in datasets:
            for sp in ds.spectra:
                jsp = {}
                jsp["raw_adc_re"] = np.real(sp._raw_adc).tolist()
                jsp["raw_adc_im"] = np.imag(sp._raw_adc).tolist()
                jsp["metabolite_names"] = sp.metabolite_names
                jsp["pulse_sequence"] = sp.pulse_sequence
                jsp["id"] = str(sp.id)
                jsp["source"] = 'sim_' + args.basis_source + "_" + args.scanner_manufacturer + "_" + args.gen
                jsp["sw"] = sp.sw
                jsp["omega"] = sp.omega
                jsp["dt"] = sp.dt
                jsp["center_ppm"] = sp.center_ppm
                jsp["te"] = sp.te
                jsp["acquisition"] = sp.acquisition
                jsp["type"] = sp.type
                jsp["spectrum_type"] = sp.spectrum_type
                jsp["linewidth"] = sp.linewidth
                jsp["concentrations"] = sp.concentrations.tolist()
                js.append(jsp)
        with open(args.save, 'w') as f:
            print(json.dumps(js, indent=4, sort_keys=True), file=f)
    if args.plot:
        for ds in datasets:
            d_in, d_out, _ = ds.export_to_keras(ds.basis.metabolite_names(),
                                                adc_noise=True,
                                                adc_noise_p=args.adc_noise_p,
                                                max_sigma=args.adc_noise_sigma,
                                                conc_normalisation=args.norm,
                                                datatype=args.datatype)
            # Plot concentration distribution
            plt.figure(figsize=(19.2, 10.8), dpi=100)
            plt.suptitle('Concentration distribution')
            for ii in range(len(args.metabolites)):
                plt.subplot(1, len(args.metabolites), ii + 1)
                plt.hist(d_out[:, ii], bins=50)
                plt.title('%s' % args.metabolites[ii])
                plt.xlim([0, 1])
            plt.show()
            # Plot spectra
            for id, acqs in group_spectra(ds.spectra).items():
                n_rows = 0
                acq_idx = []
                for k in range(0,len(acqs)):
                    if acqs[k] is not None:
                        n_rows += 1
                        acq_idx.append(k)
                n_cols = len(args.datatype)
                if len(acqs[acq_idx[0]].concentrations) > 1:
                    n_cols += 1
                fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey='row', figsize=(19.2, 10.8), dpi=100)

                super_title = ds.basis.source() + ' ' + ds.basis.pulse_sequence()
                super_title +=  ' ' + str(round((ds.basis.spectra[0].nu()[-1] - ds.basis.spectra[0].nu()[0]) * ds.basis.spectra[0].omega)) + 'Hz' + ', Linewidth:' + str(ds.basis.spectra[0].linewidth)
                if acqs[acq_idx[0]].add_adc_noise:
                    super_title += ", Noise mu: %.1f, sigma: %.4f" % (acqs[acq_idx[0]]._adc_noise_mu, acqs[acq_idx[0]]._adc_noise_sigma)
                plt.suptitle(super_title)

                if n_rows > 1:
                    axes[0, 0].set_xlim([-4.5, -1])
                    for l in range(0, n_rows):
                        axes[l, 0].set_ylabel(acqs[acq_idx[l]].spectrum_type, rotation=0, labelpad=50, size='large')
                        c = 0
                        if 'magnitude' in args.datatype:
                            if l == 0:
                                axes[l, c].set_title('Magnitude')
                            axes[l, c].plot(acqs[acq_idx[l]].nu(), np.abs(acqs[acq_idx[l]].fft()), label='FFT')
                            c += 1
                        if 'real' in args.datatype:
                            if l == 0:
                                axes[l, c].set_title('Real')
                            axes[l, c].plot(acqs[acq_idx[l]].nu(), np.real(acqs[acq_idx[l]].fft()), label='FFT')
                            c += 1
                        if 'imaginary' in args.datatype:
                            if l == 0:
                                axes[l, c].set_title('Imaginary')
                            axes[l, c].plot(acqs[acq_idx[l]].nu(), np.imag(acqs[acq_idx[l]].fft()), label='FFT')
                            c += 1
                    axes[n_rows - 1, 0].set_xlabel('PPM')
                    if c > 0:
                        axes[n_rows - 1, 1].set_xlabel('PPM')
                        if c > 1:
                            axes[n_rows - 1, 2].set_xlabel('PPM')
                else:
                    axes[0].set_xlim([-4.5, -1])
                    axes[0].set_ylabel(acqs[acq_idx[0]].spectrum_type, rotation=0, labelpad=50, size='large')
                    c = 0
                    if 'magnitude' in args.datatype:
                        axes[c].set_title('Magnitude')
                        axes[c].plot(acqs[acq_idx[0]].nu(), np.abs(acqs[acq_idx[0]].fft()), label='FFT')
                        c += 1
                    if 'real' in args.datatype:
                        axes[c].set_title('Real')
                        axes[c].plot(acqs[acq_idx[0]].nu(), np.real(acqs[acq_idx[0]].fft()), label='FFT')
                        c += 1
                    if 'imaginary' in args.datatype:
                        axes[c].set_title('Imaginary')
                        axes[c].plot(acqs[acq_idx[0]].nu(), np.imag(acqs[acq_idx[0]].fft()), label='FFT')
                        c += 1
                    axes[0].set_xlabel('PPM')
                    if c > 0:
                        axes[1].set_xlabel('PPM')
                        if c > 1:
                            axes[2].set_xlabel('PPM')

                if len(acqs[acq_idx[0]].concentrations) > 1:
                    ax = plt.subplot(1, n_cols, n_cols)
                    plt.title('Concentrations')
                    ax.bar(np.linspace(0, len(acqs[acq_idx[0]].concentrations) - 1, len(acqs[acq_idx[0]].concentrations)), acqs[acq_idx[0]].concentrations)
                    ax.set_xticks(np.arange(len(acqs[acq_idx[0]].metabolite_names)))
                    ax.set_xticklabels(convert_molecule_names(acqs[acq_idx[0]].metabolite_names, shorten=True))

                plt.show()

def generate_datasets(bases, name, num, gen, metabolites, acquisitions, verbose):
    # Generate datasets
    datasets = []
    n0 = num // len(bases)
    n1 = num % len(bases)
    for basis in bases:
        dataset = Dataset()
        dataset._name = name
        dataset.basis = basis
        dataset.acquisitions = acquisitions
        dataset.generate_dataset(n0+(1 if n1 > 0 else 0), gen, metabolites, verbose=verbose)
        n1 -= 1
        datasets.append(dataset)
    return datasets

def group_spectra(ds_spectra):
    spectra = {}
    for s in ds_spectra:
        if s.id not in spectra:
            spectra[s.id] = []
        if s.acquisition >= len(spectra[s.id]):
            spectra[s.id] += [None] * (1 + s.acquisition - len(spectra[s.id]))
        spectra[s.id][s.acquisition] = s
    return spectra

class Dataset(object):
    """
       A dataset is a collection of spectra for training, tesing or predicting.
       The dataset object contains methods for how to handle and export the data
       from the spectra objects.
    """
    def __init__(self):
        self._name = 'dataset'

        self.basis = None # Set if generated from basis only
        self.spectra = []
        self._pulse_sequence = None # Set if no basis (real data)
        self.acquisitions = None

        # Fixed frequency bins
        self.high_ppm = -4.5
        self.low_ppm = -1
        self.n_fft_pts = 2048

    def pulse_sequence(self):
        if self.basis == None:
            return self._pulse_sequence
        return self.basis.pulse_sequence()

    def linewidth(self):
        if self.basis == None:
            return None;
        lw = self.basis.linewidths()
        if len(lw) > 1:
            raise Exception('Dataset basis should only have one linewidth')
        return lw[0]

    def add_spectra(self, spectra):
        self.spectra.extend(spectra)

    def invalidate_spectra_fft_cache(self):
        for spectra in self.spectra:
            spectra._need_fft_cache_update = True

    def group_spectra_by_id(self):
        ids = []
        for ii in range(0, len(self.spectra)):
            ids.append(self.spectra[ii].id)

        n_acquisitions = ids.count(ids[0])
        ids = list(set(ids))
        acquisitions = [np.array([]) for _ in range(len(ids))]

        # Group the acquisitions by ID, and also sort them by acquisition
        for count, id in enumerate(ids):
            acquisition_numbers = []
            for ii in range(0, len(self.spectra)):
                if self.spectra[ii].id == id:
                    acquisition_numbers.append(self.spectra[ii].acquisition)
                    acquisitions[count] = np.append(acquisitions[count], self.spectra[ii])
                    if len(acquisition_numbers) == n_acquisitions:
                        # we've found all the acquisitions for this ID
                        break
            # sort the acquisitions
            acquisitions[count] = acquisitions[count][np.argsort(acquisition_numbers)]
        return acquisitions

    def check(self):
        if len(self.spectra) == 0:
            raise Exception('Dataset has no spectra associated with it')
        if self.basis is None:
            if self._pulse_sequence is None:
                raise Exception('Dataset basis and _pulse_seqeunce is none!')
        else:
            self.basis.check()

        if self.high_ppm > self.low_ppm:
            raise Exception('High ppm needs to be less than low ppm.. E.G. high_ppm = -5, low_ppm = 0 as the'
                            ' frequency axis is backwards.')

        for ii in range(0, len(self.spectra)):
            self.spectra[ii].check()

    def prime_rescale_fft(self):
        self.spectra[0].rescale_fft(high_ppm=self.high_ppm, low_ppm=self.low_ppm, npts=self.n_fft_pts)
        for ii in range(len(self.spectra)):
            self.spectra[ii].update_zero_pad(self.spectra[0]._zero_pad)

    def generate_dataset(self, n_samples, conc_gen_method, metabolites, overwrite=False, verbose=0):
        # Generate the dataset from the basis (assuming metabolites taken from those in the basis).
        # Does not add noise, but only generates clean combined ADC signal.
        if n_samples <= 0:
            raise Exception('n_samples must be greater than 0, not %d!' % n_samples)
        if len(self.spectra):
            if overwrite:
                print('Overwriting current dataset, as overwrite set to true.')
                self.spectra = []
            else:
                raise Exception('This dataset already has spectra associated with it, either set overwrite=True or '
                                'make a new dataset with the same basis for a different config.')

        metabolite_names = []
        for spectrum in self.basis.spectra:
            if len(spectrum.metabolite_names) != 1:
                raise Exception('Spectra in basis set cannot have more or less than one metabolite if you want to generate a dataset.')
            if np.min(spectrum.nu()) > self.high_ppm:
                raise Exception('Spectra do not reach the required max frequency axis (%.2f) for export: %.2f' % (np.min(spectra.nu()), self.high_ppm))
            elif np.max(spectrum.nu()) < self.low_ppm:
                raise Exception('Spectra do not reach the required min frequency axis (%.2f) for export: %.2f' % (np.max(spectra.nu()), self.low_ppm))
            metabolite_names.extend(spectrum.metabolite_names)
        metabolite_names = list(set(metabolite_names)) # Assuming things are consistent here
        if len(metabolite_names) != len(metabolites):
            raise Exception('Metabolites in basis sets different from metabolites requests for dataset generation')
        for m in metabolites:
            if m not in metabolite_names:
                raise Exception('Request metabolite for data set generation not in basis set: ' + m)
        n_metabolites = len(metabolites)

        if conc_gen_method == 'random':
            # Random uniform concentrations
            concentrations = np.random.ranf((n_samples, n_metabolites))
        elif conc_gen_method == 'dirichlet':
            # Dirichlet sampling, equal weight for all metablites
            concentrations = np.random.default_rng().dirichlet([1] * n_metabolites, n_samples)
        elif conc_gen_method == 'sobol':
            # Sobol sampling
            skip = math.floor(math.log(n_samples*n_metabolites,2)) # Skip broken in sobol package! (silly seed assignment in some versions)
            concentrations = sobol_seq.i4_sobol_generate(n_metabolites, n_samples+skip)[skip:,:]
        elif conc_gen_method[-6:] == '-zeros':
            # Select all zero concentration possibilities, equal weight for all remaining metablites according to sampling method
            concentrations = np.zeros((n_samples, n_metabolites))
            groups = []
            groups_n = []
            n_per_combs = n_samples // n_metabolites
            n_total = 0
            for n_excited in range(1,n_metabolites + 1):
                combs = list(combinations(list(range(0,n_metabolites)), n_excited))
                n_per_group = n_per_combs // len(combs)
                if verbose > 0:
                    print("For %d excited: %d samples per %d combinations" % (n_excited, n_per_group, len(combs)))
                if n_per_group < 1:
                    raise Exception('Insufficient samples for dirichlet-zeros with %d groups' % len(groups))
                for comb in combs:
                    groups.append(comb)
                    groups_n.append(n_per_group)
                    n_total += n_per_group
            groups.reverse()
            groups_n.reverse()
            n_remain = n_samples - n_total
            idx = 0
            if conc_gen_method == 'sobol-zeros':
                skip = math.floor(math.log(n_samples*n_metabolites,2))
            for g, n_g in zip(groups, groups_n):
                if n_remain > 0:
                    n_g += 1
                    n_remain -= 1
                if conc_gen_method == 'random-zeros':
                    # Random uniform concentrations
                    concentrations[idx:idx+n_g,g] = np.random.ranf((n_g, len(g)))
                elif conc_gen_method == 'dirichlet-zeros':
                    # Dirichlet sampling, equal weight for all metablites
                    concentrations[idx:idx+n_g,g] = np.random.default_rng().dirichlet([1]*len(g), n_g)
                elif conc_gen_method == 'sobol-zeros':
                    # Sobol sampling
                    concentrations[idx:idx+n_g,g] = sobol_seq.i4_sobol_generate(len(g), n_g+skip)[skip:,:]
                    skip += n_g # Get differnt samples for the groups
                else:
                    raise Exception('Unknown concentration generation method: ' + self.conc_gen_method)
                idx += n_g
        elif conc_gen_method == 'single':
            # Single metabolite concentration (for testing)
            concentrations = np.zeros((n_samples, n_metabolites),dtype=np.float64)
            nm = 0
            for n in range(0,n_samples):
                concentrations[n,nm] = 1.0
                nm += 1
                if nm >= n_metabolites:
                    nm = 0
        else:
            raise Exception('Unknown concentration generation method: ' + self.conc_gen_method)

        for count in range(n_samples):
            self.add_spectra(self.basis.export_combination(concentrations[count], metabolites, self.acquisitions))

        self.check()

    def export_to_keras(self,
                        metabolites=None,
                        adc_noise=False,
                        adc_noise_p=0.5,
                        max_sigma=0.25,
                        conc_normalisation='sum', # max, sum
                        datatype=['magnitude']): # real, imaginary, magnitude
        if metabolites is None:
            # Organise the output mapping for the metabolite names and the concentrations
            metabolites = []
            for s in self.spectra:
                metabolites.extend(s.metabolite_names)
        else:
            if isinstance(metabolites, np.ndarray):
                metabolites = metabolites.tolist()
            elif not isinstance(metabolites, list):
                raise Exception('metabolites must be of type list or np.ndarray.')
        metabolites = sorted(list(set(metabolites)))

        # Add noise?
        for ii in range(0, len(self.spectra)):
            self.spectra[ii].add_adc_noise = adc_noise
        self.check()
        if adc_noise:
            for group in self.group_spectra_by_id():
                group[0].setup_adc_noise_parameters(adc_noise_p, max_sigma)
                for spectra in group:
                    spectra._adc_noise_sigma = group[0]._adc_noise_sigma
                    spectra._adc_noise_mu = group[0]._adc_noise_mu
                    spectra.generate_adc_noise(overwrite=True)

        for ii in tqdm(range(len(self.spectra)), desc='Verifying concentration mappings', leave=False, total=len(self.spectra) - 1):
            self.spectra[ii].metabolite_names = np.array(self.spectra[ii].metabolite_names)
            self.spectra[ii].concentrations = np.array(self.spectra[ii].concentrations)
            if (not len(self.spectra[ii].metabolite_names) == len(metabolites)) or \
               (not all(self.spectra[ii].metabolite_names == metabolites)):
                # Either the ordering is wrong, or there are metabolites missing.
                # Then replace the concentration array length to accommodate this,
                # while mapping the old concentrations to the new ones.

                # Get the indexes of metabolites that are present, in relation to metabolites
                index = []
                for name in [x.lower() for x in self.spectra[ii].metabolite_names]:
                    if name in [x.lower() for x in metabolites]:
                        index.append([x.lower() for x in metabolites].index(name))

                # Create a new label array, with the correct model->label arrangement
                new_conc = np.zeros(len(metabolites))
                for c, i in enumerate(index):
                    new_conc[i] = self.spectra[ii].concentrations[c]

                # Update the spectra labels & model lables (no need to regen the FFTs as it's the same)
                self.spectra[ii].concentrations = new_conc
                self.spectra[ii].metabolite_names = metabolites

        # In theory, we should only have to do the zero_pad calculation once.
        # It checks the length of the zero pad needed on one of the spectra and copies it to all of the output.
        self.prime_rescale_fft()

        # Now we group all of the spectra together by ID and in the correct order of acquisition.
        scans = self.group_spectra_by_id()

        if len(scans) < 200:
            data = []
            labels = []
            for scan in scans:
                temp_data, temp_labels = parallel_export_function(scan,
                                                                  high_ppm=self.high_ppm, low_ppm=self.low_ppm, n_fft_pts=self.n_fft_pts,
                                                                  datatype=datatype)
                data.append(temp_data)
                labels.append(temp_labels)
        else:
            try:
                pool = multiprocessing.Pool()
                func = partial(parallel_export_function,
                               high_ppm=self.high_ppm, low_ppm=self.low_ppm, n_fft_pts=self.n_fft_pts,
                               datatype=datatype)
                data, labels = zip(*list(tqdm(pool.imap(func, iterable=scans, chunksize=32),
                                              desc='Preparing data',
                                              total=len(scans),
                                              leave=False)))
                pool.close()
                pool.join()
            except:
                print('\n\nFound error in parallel export function, running without multi-processing pool to find and '
                      'raise the underlying error:\n')
                for scan in scans:
                    temp_data, temp_labels = parallel_export_function(scan,
                                                                      high_ppm=self.high_ppm, low_ppm=self.low_ppm, n_fft_pts=self.n_fft_pts,
                                                                      datatype=datatype)
                    data.append(temp_data)
                    labels.append(temp_labels)

        if len(np.shape(data)) != 3:
            raise Exception('Export does not have dim(3), when it should...')

        labels = normalise_labels(np.array(labels), conc_normalisation)

        data = np.array(data)

        return data, labels, metabolites

def parallel_export_function(acquisition, high_ppm, low_ppm, n_fft_pts, datatype, mean_center=True):
    # Map the spectra metabolite names to the correct output labels on the given network.
    labels = acquisition[0].concentrations
    for acq in acquisition:
        if not all(acq.concentrations == labels):
            raise Exception('Concentrations are not consistent across acquisitions.')

    ffts = []
    # Gather all the ffts for the acquisitions
    for jj in range(len(acquisition)):  # for each acquisiton of each sample
        fft, nu = acquisition[jj].rescale_fft(high_ppm=high_ppm, low_ppm=low_ppm, npts=n_fft_pts)
        if mean_center:
            fft = fft - np.mean(fft)
        ffts.append(fft)

    # We normalise the signals once we have them all as we want them relative to each other
    # this could be more compact, but keeping it expanded out for clarity
    # the idea here is to try and preserver the order of the acquisitions e.g:
    #           | real      |
    #  acq 0 :  | imaginary |
    # __________| magnitude |
    #           | real      |
    #  acq 1 :  | imaginary |
    # __________| magnitude |
    #           | real      |
    #  acq 2 :  | imaginary |
    #           | magnitude |
    spec_data = []
    for fft in ffts:
        if any([x in datatype for x in ['r', 'real']]):
            spec_data.append(np.real(fft))
        if any([x in datatype for x in ['i', 'imaginary']]):
            spec_data.append(np.imag(fft))
        if any([x in datatype for x in ['m', 'magnitude', 'a', 'absolute']]):
            spec_data.append(np.abs(fft))

    # Normalse the spectra data to fall in the -1:1 range
    spec_data = spec_data / np.max(np.abs(spec_data))

    if any(np.isnan(np.ndarray.flatten(np.array(spec_data)))):
        raise Exception('"spec_data" export array contains NaNs. Something has gone wrong here...')

    return spec_data, labels
