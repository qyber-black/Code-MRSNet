#!/usr/bin/env python3
#
# quantify.py - MRSNet - quantify spectra
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2019, Max Chandler, PhD student at Cardiff University
# Copyright (C) 2020, Frank C Langbein <frank@langbein.org>, Cardiff University

import os
import numpy as np
import json
from tensorflow.keras.models import load_model

from utilities.utils import reshape_data, normalise_labels, convert_molecule_names
from spectrum import Spectrum
from basis import Basis
from dataset import Dataset
from analyse import analyse_model

def quantify(args):
    ima_dir = args.spectra
    json_file = args.spectra_json
    model_dir = args.model
    metabolites = args.metabolites
    verbose = args.verbose

    model = load_model(os.path.join(model_dir, 'model'))
    for s in model.name.split("_"):
        if s[0:2] == "M-":
            model.metabolites = sorted(convert_molecule_names(s[2:].split("-")))
        elif s[0:2] == "A-":
            model.acquisitions = [int(a) for a in s[2:].split("-")]
        elif s[0:2] == "T-":
            model.datatype = []
            for c in s[2:]:
                if c == "m":
                    model.datatype.append('magnitude')
                elif c == "r":
                    model.datatype.append('real')
                elif c == "i":
                    model.datatype.append('imaginary')
        elif s[0:2] == "P-":
            model.pulse_sequence = s[2:]

    if args.verbose > 0:
        print("Model: " + model.name)
        print("Pulse sequence: " + model.pulse_sequence)
        print("Metabolites: " + ', '.join(map(str, model.metabolites)))
        print("Acquisitions: " + ', '.join(map(str, model.acquisitions)))
        print("Datatypes: " + ', '.join(map(str, model.datatype)))

    # check to see if the network can actually quantify the requested metabolites
    if not all([m_name.lower() in [x.lower() for x in model.metabolites] for m_name in metabolites]):
        raise Exception('Network is unable to quantify one or more metabolites suplied.'
                        'Network is able to quantify: ' + str(model.metabolites) + '\n'
                        'Requested metabolites: ' + str(metabolites))

    # generate a dataset from the loaded dicoms / json
    if json_file is not None:
        with open(json_file) as f:
            data = json.load(f)
        basis = Basis()
        for s in data:
            b = Spectrum()
            b._raw_adc = np.float64(s["raw_adc_re"]) + 1j * np.float64(s["raw_adc_im"])
            b.metabolite_names = s["metabolite_names"]
            b._pulse_sequence = s["pulse_sequence"]
            b.id = s["id"]
            b.source =  s["source"]
            b.sw = s["sw"]
            b.omega = s["omega"]
            b.dt = s["dt"]
            b.center_ppm = s["center_ppm"]
            b.te = s["te"]
            b.acquisition = s["acquisition"]
            b.type = s["type"]
            b.spectrum_type = s["spectrum_type"]
            b._linewidths = s["linewidth"]
            b.concentrations = np.float64(s["concentrations"])
            basis.spectra.append(b)
    else:
        basis = Basis.load_dicom(ima_dir)

    ds = Dataset()
    ds._name = 'quantify'
    for s in basis.spectra:
        if s.acquisition in model.acquisitions:
            ds.spectra.append(s)
    ds._pulse_sequence = model.pulse_sequence
    ds.acquisitions = model.acquisitions

    # export the dataset into the format we're looking
    d_in, _, d_labels = ds.export_to_keras(model.metabolites,
                                           adc_noise=False,
                                           max_sigma=None,
                                           conc_normalisation=model.name[-3:],
                                           datatype=model.datatype)
    d_in, _ = reshape_data(d_in)

    # quantify
    predictions = model.predict(d_in)

    # trim the output data to match the metabolites of interest
    cols_index = [[x.lower() for x in model.metabolites].index(m_name.lower()) for m_name in metabolites]
    predictions = predictions[:, cols_index]
    # renormalise the output labels
    predictions = normalise_labels(predictions,model.name[-3:])

    # get the spectra in the order they were exported in dataset.export_to_keras
    # so we can match the predictions to the spectra
    spectra = np.array(ds.group_spectra_by_id())

    # print the results table!
    print('# Quantifying %d MEGA-PRESS Spectra' % (len(spectra)))
    print('\nMetabolites quantified: ' + ", ".join(model.metabolites))
    print('Model path: ' + model_dir)
    if json_file is not None:
        print('Spectra file: ' + json_file)
    else:
        print('Spectra path: ' + ima_dir)
    print('\n                       Concentrations')
    if sum(spectra[0,0].concentrations):
        print('  %-*s %s  %s    %s' % (20, 'Metabolite', 'Predicted', 'Actual', 'Error'))
    else:
        print('  %-*s %s' % (20, 'Metabolite', 'Predicted'))
    m_names = convert_molecule_names(metabolites)
    for spec, prediction in zip(spectra, predictions):
        spec[0].prediction = prediction
    spectra = sorted(spectra, key=lambda x: x[0].id)
    for spec in spectra:
        if sum(spec[0].concentrations):
            print('Spectrum ID: ' + spec[0].id)
            actual_concentrations = normalise_labels(spec[0].concentrations, model.name[-3:])
            for p, a, m_name in zip(spec[0].prediction, actual_concentrations, m_names):
                print('  %-*s %.6f   %.6f  %.6f' % (20, m_name, p, a, np.abs(p-a)))
        else:
            print('Spectrum ID: ' + spec[0].id)
            print('  %-*s %s' % (20, 'Metabolite', 'Predicted'))
            for p, m_name in zip(spec[0].prediction, m_names):
                print('  %-*s %.6f' % (20, m_name, p))

def benchmark(args):
    model_dir = args.model
    verbose = args.verbose

    model = load_model(os.path.join(model_dir, 'model'))
    for s in model.name.split("_"):
        if s[0:2] == "M-":
            model.metabolites = sorted(convert_molecule_names(s[2:].split("-")))
        elif s[0:2] == "A-":
            model.acquisitions = [int(a) for a in s[2:].split("-")]
        elif s[0:2] == "T-":
            model.datatype = []
            for c in s[2:]:
                if c == "m":
                    model.datatype.append('magnitude')
                elif c == "r":
                    model.datatype.append('real')
                elif c == "i":
                    model.datatype.append('imaginary')
        elif s[0:2] == "P-":
            model.pulse_sequence = s[2:]

    if args.verbose > 0:
        print("Model: " + model.name)
        print("Pulse sequence: " + model.pulse_sequence)
        print("Metabolites: " + ', '.join(map(str, model.metabolites)))
        print("Acquisitions: " + ', '.join(map(str, model.acquisitions)))
        print("Datatypes: " + ', '.join(map(str, model.datatype)))

    benchmark_spectra, benchmark_names = load_benchmark_datasets()
    for basis, name in zip(benchmark_spectra, benchmark_names):
        benchmark_dataset = Dataset()
        benchmark_dataset._name = name
        for s in basis.spectra:
            if s.acquisition in model.acquisitions:
                benchmark_dataset.spectra.append(s)
        benchmark_dataset._pulse_sequence = 'megapress' # Works for now
        benchmark_dataset.acquisitions = model.acquisitions
        d_in, d_out, d_labels = benchmark_dataset.export_to_keras(model.metabolites,
                                                                  adc_noise=False,
                                                                  max_sigma=None,
                                                                  conc_normalisation=model.name[-3:],
                                                                  datatype=model.datatype)
        analyse_model(model, d_in, d_out, d_labels, save_dir=model_dir, prefix=benchmark_dataset._name)

def load_benchmark_datasets():
    test_basis = [Basis.load_dicom(os.path.join('data','benchmark', 'E1', 'MEGA_Combi_WS_ON')),
                  Basis.load_dicom(os.path.join('data','benchmark', 'E2', 'MEGA_Combi_WS_ON')),
                  Basis.load_dicom(os.path.join('data','benchmark', 'E3', 'MEGA_Combi_WS_ON')),
                  Basis.load_dicom(os.path.join('data','benchmark', 'E4a', 'MEGA_Combi_WS_ON')),
                  Basis.load_dicom(os.path.join('data','benchmark', 'E4b', 'MEGA_Combi_WS_ON')),
                  Basis.load_dicom(os.path.join('data','benchmark', 'E4c', 'MEGA_Combi_WS_ON')),
                  Basis.load_dicom(os.path.join('data','benchmark', 'E4d', 'MEGA_Combi_WS_ON'))]
    test_names = ['E1', 'E2', 'E3', 'E4a', 'E4b', 'E4c', 'E4d']
    return test_basis, test_names
