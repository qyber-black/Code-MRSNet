#!/usr/bin/env python3
#
# mrsnet.py - MRSNet - command line MRSNet interface
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2019, Max Chandler, PhD student at Cardiff University
# Copyright (C) 2020, Frank C Langbein <frank@langbein.org>, Cardiff University
#
# See --help for arguments, uses sub-commands

import os
import sys
import argparse
import datetime
import matplotlib

# Only print warnings and errors for tf (set before importing tf)
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from utilities.utils import convert_molecule_names
from utilities.constants import MOLECULE_NAMES
from quantify import quantify, benchmark
from basis import basis
from dataset import spectra
from model import train

def main():
    """Main function of MRSNet: parse arguments, setup basic environment and initiate functionality"""

    # headless mode
    if not "DISPLAY" in os.environ:
        matplotlib.use("Agg")

    # Process arguments
    parser = argparse.ArgumentParser(description='Magnetic Resonance Spectra (MRS) quantification',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(title="Valid sub-commands")

    # Quantify
    p_quantify = subparsers.add_parser('quantify', help='Quantify spectra in dicoms.',
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arguments_default(p_quantify)
    add_arguments_metabolites(p_quantify)
    p_quantify.add_argument('-m', '--model', help='Folder containing the model to quantifiy spectra.',
                          default=os.path.join('data', 'model', 'MRSNet_LCModel'))
    p_quantify.add_argument('-s', '--spectra', help='Folder of spectra dicom files to quantify; searched recursively.',
                          default=os.path.join('data', 'benchmark', 'E1', 'MEGA_Combi_WS_ON'))
    p_quantify.add_argument('-j', '--spectra_json', help='Json file of spectra (from spectra simulation).', default=None)
    p_quantify.set_defaults(func=quantify)

    # Benchmark
    p_benchmark = subparsers.add_parser('benchmark', help='Run benchmark on model.')
    add_arguments_default(p_benchmark)
    p_benchmark.add_argument('-m', '--model', help='Folder containing the model to quantifiy spectra.',
                             default=os.path.join('data', 'model', 'MRSNet_LCModel'))
    p_benchmark.set_defaults(func=benchmark)

    # Generate basis
    p_basis = subparsers.add_parser('basis', help='Generate basis for spectra generation, if it does not exist.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arguments_default(p_basis)
    add_arguments_metabolites(p_basis)
    add_arguments_basis(p_basis)
    p_basis.set_defaults(func=basis)

    # Generate spectra
    p_spectra = subparsers.add_parser('spectra', help='Generate simulated spectra dataset (for testing only).',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arguments_default(p_spectra)
    add_arguments_metabolites(p_spectra)
    add_arguments_basis(p_spectra)
    add_arguments_spectra(p_spectra)
    p_spectra.add_argument('-s', '--save', type=str, default=None, help='File to store spectra in (if given).')
    p_spectra.set_defaults(func=spectra)

    # Train
    p_train = subparsers.add_parser('train', help='Train model (always generates dataset).',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arguments_default(p_train)
    add_arguments_metabolites(p_train)
    add_arguments_basis(p_train)
    add_arguments_spectra(p_train)
    p_train.add_argument('-f', '--model_folder', type=str, default=os.path.join('data', 'model', datetime.datetime.now().strftime('%d%m%y_%H%M%S')),
                         help='Folder to store model.')
    p_train.add_argument('-p', '--validate_per', type=int, default=30,
                         help='Percentage of dataset for testing split.')
    p_train.add_argument('-m', '--model', type=str, default='mrsnet_cnn_small',
                         help='Model architecture - function name from cnns.py.')
    p_train.add_argument('-e', '--epochs', type=int, default=100,
                         help='Number of training epochs.')
    p_train.add_argument('-b', '--batch_size', type=int, default=64,
                         help='Batch size.')
    p_train.set_defaults(func=train)

    args = parser.parse_args()
    if hasattr(args, 'metabolites'):
        args.metabolites = sorted(args.metabolites) # Make sure metabolite names are sorted
    if hasattr(args,"func"):
        args.func(args)
    else:
        print("%s: illegal sub-command or sub-command not specified, see help [-h]" % sys.argv[0], file=sys.stderr)

def add_arguments_default(p):
    p.add_argument('-v', '--verbose', action='count', help='Increase output verbosity.', default=0)
    p.add_argument('--plot', action='store_true', default=False, help='Generate graphs, if any.')
    p.add_argument('--save_plot', action='store_true', default=False, help='Save graphs.')

def add_arguments_metabolites(p):
    p.add_argument('--metabolites', type=lambda s : convert_molecule_names([s])[0], nargs='+',
                   default=sorted(['Creatine', 'GABA', 'Glutamate', 'Glutamine', 'N-Acetylaspartate']),
                   help='List of metabolites to use. Use full names as defined in utilities.constants: '+str(MOLECULE_NAMES)+'.')

def add_arguments_basis(p):
    p.add_argument('--basis_source', type=lambda s : s.lower(), choices=['lcmodel', 'fid-a', 'pygamma'], default='lcmodel', nargs='+',
                   help='Data source for the basis spectra (FID-A requires Matlab; pygamma requires python2 pygamma).')
    p.add_argument('--scanner_manufacturer', type=lambda s : s.lower(), choices=['siemens', 'ge', 'phillips'], default='siemens',
                   help='Scanner manufacturer (fid-a and pygamma only support siemens).')
    p.add_argument('--omega', type=float, default=123.23,
                   help='Scanner frequency in MHz (default 123.23 MHz for 2.98 T Siemens scanner).')
    p.add_argument('--linewidths', type=float, nargs='+', dest='linewidths', default=[1.0],
                   help='Linewidths to be used for simulation (not possible for lcmodel).')

def add_arguments_spectra(p):
    p.add_argument('-n', '--num', type=int, default=5000,
                   help='Total dataset size.')
    p.add_argument('--norm', choices=['sum', 'max'], default='sum', help='Concentration normalisation: sum or max equal to 1')
    p.add_argument('--gen', choices=['random', 'random-zeros', 'dirichlet', 'dirichlet-zeros', 'sobol', 'sobol-zeros', 'single'],
                   default='sobol', help='Concentration generation method.')
    p.add_argument('--datatype', type=lambda s : s.lower(), nargs='+', default=['magnitude'],
                   help='Datatype representation of spectrum (magnitude, real, imaginary).')
    p.add_argument('--acquisitions', type=int, nargs='+', default=[0, 2],
                   help='Acquisitions from pulse sequence used (acquisition numbers; megapress: 0=edit-off, 1=edit-on, 2=difference).')
    p.add_argument('--adc_noise_p', type=float, default=0.5,
                   help='Probability of ADC noise applied to spectrum.')
    p.add_argument('--adc_noise_sigma', type=float, default=0.025,
                   help='Maximum sigma for simulated ADC noise.')

if __name__ == '__main__':
    main()
