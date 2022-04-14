#!/usr/bin/env python3
#
# mrsnet.py - MRSNet - command line MRSNet interface
#
# SPDX-FileCopyrightText: Copyright (C) 2019 Max Chandler, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2020-2022 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2022 Zien Ma, PhD student at Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# See --help for arguments, uses sub-commands

import os
import glob
import sys
import argparse
import matplotlib.pyplot as plt

import mrsnet.molecules as molecules
from mrsnet.cfg import Cfg

def main():
  # Main function of MRSNet: parse arguments, setup basic environment and run

  # Process arguments
  parser = argparse.ArgumentParser(description='Magnetic Resonance Spectra (MRS) Quantification',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  subparsers = parser.add_subparsers(title="Valid sub-commands")

  # Generate basis
  p_basis = subparsers.add_parser('basis', help='Generate basis, if it does not exist.',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  add_arguments_default(p_basis)
  add_arguments_metabolites(p_basis)
  add_arguments_basis(p_basis)
  add_arguments_fft(p_basis)
  p_basis.set_defaults(func=basis)

  # Generate simulated dataset
  p_simulate = subparsers.add_parser('simulate', help='Generate simulated spectra dataset.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  add_arguments_default(p_simulate)
  add_arguments_metabolites(p_simulate)
  add_arguments_basis(p_simulate)
  add_arguments_simulate(p_simulate)
  add_arguments_fft(p_simulate)
  p_simulate.set_defaults(func=simulate)

  # Generate all datasets
  p_gen_ds = subparsers.add_parser('generate_datasets', help='Generate standard datasets.',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  add_arguments_default(p_gen_ds)
  p_gen_ds.add_argument('collection', type=str, help='Dataset collection json filename')
  p_gen_ds.set_defaults(func=generate_datasets)

  # Compare spectra
  p_compare = subparsers.add_parser('compare', help='Compare spectra.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  add_arguments_default(p_compare)
  add_arguments_compare(p_compare)
  add_arguments_fft(p_compare)
  p_compare.set_defaults(func=compare)

  # Train
  p_train = subparsers.add_parser('train', help='Train model on dataset.',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  add_arguments_default(p_train)
  add_arguments_metabolites(p_train)
  add_arguments_train_select(p_train)
  add_arguments_train(p_train)
  p_train.set_defaults(func=train)

  # Model selection
  p_select = subparsers.add_parser('select', help='Model selection on dataset.',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  add_arguments_default(p_select)
  add_arguments_metabolites(p_select)
  add_arguments_train_select(p_select)
  p_select.add_argument('--method', choices=['grid', 'qmc', 'gpo', 'ga'], default='random',
                        help='Model selection approach')
  p_select.add_argument('-r', '--repeats', type=int, default=100,
                        help='Maximum number of repeats (for qmc, gpo, ga).')
  p_select.add_argument('--remote', type=str, default='',
                        help='Remote execution: scheduler:user:[max_parallel_tasks=10:[wait_minutes=15]]')
  p_select.add_argument('collection', type=str, help='Model collection json filename')
  p_select.set_defaults(func=model_selection)

  # Quantify
  p_quantify = subparsers.add_parser('quantify', help='Quantify spectra in dicoms.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  add_arguments_default(p_quantify)
  add_arguments_quantify(p_quantify)
  p_quantify.set_defaults(func=quantify)

  # Benchmark
  p_benchmark = subparsers.add_parser('benchmark', help='Run benchmark on model.')
  add_arguments_default(p_benchmark)
  add_arguments_benchmark(p_benchmark)
  p_benchmark.set_defaults(func=benchmark)

  args = parser.parse_args()
  if hasattr(args,"metabolites"):
    if "GlX" in args.metabolites and ("Gln" in args.metabolites or "Glu" in args.metabolites):
      raise Exception("GlX with Gln or Glu is not possible")
  if hasattr(args,"noise_p"):
    if args.noise_p <= 0.0 or (args.noise_sigma <= 0.0 and args.noise_mu <= 0.0) or args.noise_type == "none":
      args.noise_p = 0.0
      args.noise_sigma = 0.0
      args.noise_mu = 0.0
      args.noise_type = "none"

  if hasattr(args,"func"):
    args.func(args)
  else:
    print(f"{sys.argv[0]}: illegal sub-command or sub-command not specified, see help [-h]", file=sys.stderr)

def add_arguments_default(p):
  # Add argumnets for all sub-commands
  p.add_argument('-v', '--verbose', action='count', help='Increase output verbosity (0: none; 1: main text; 2: +main plots; 3: detailed text; 4: +detailed plots; 5: +tests and debug; 6: +extra plots).', default=0)

def add_arguments_metabolites(p):
  # Add metabolites argument
  p.add_argument('--metabolites', type=lambda s : molecules.short_name(s), nargs='+',
                 default=sorted(['Cr', 'GABA', 'Glu', 'Gln', 'NAA']),
                 help='List of metabolites to use, as defined in mrsnet.molecules: '+str(molecules.NAMES)+'.')

def add_arguments_basis(p):
  # Add basis source arguments
  p.add_argument('--source', type=lambda s : s.lower(),
                 choices=['lcmodel', 'fid-a', 'fid-a-2d', 'pygamma', 'su-3tskyra'], default=['fid-a'],
                 nargs='+',
                 help='Data source(s) for the basis spectra (fid-a* requires Matlab).')
  p.add_argument('--manufacturer', type=lambda s : s.lower(),
                 choices=['siemens', 'ge', 'phillips'], default=['siemens'],
                 nargs='+',
                 help='Scanner manufacturer (fid-a* and pygamma only support siemens).')
  p.add_argument('--omega', type=float, default=[123.23], nargs='+',
                 help='Scanner frequency in MHz (default 123.23 MHz for 2.89 T Siemens scanner).')
  p.add_argument('--linewidth', type=float, nargs='+', default=[2.0],
                 help='Linewidths to be used for simulation (not possible for lcmodel, su-3tskyra).')
  p.add_argument('--pulse_sequence', type=lambda s : s.lower(), nargs='+',
                 choices=['megapress'], default=["megapress"],
                 help='Pulse sequence (placeholder).')

def add_arguments_fft(p):
  # Add fft arguments
  p.add_argument('--sample_rate', type=lambda v : (abs(int(v))//2)*2, default=2000,
                 help='FFT sample rate for basis/simulation in Hz (even, positive integer; ignored for lcmodel, su-3tskyra).')
  p.add_argument('--samples', type=lambda v : (abs(int(v))//2)*2, default=4096,
                 help='FFT time samples for basis/simluation (even, positive integer; ignored for lcmodel, su-3tskyra).')

def add_arguments_simulate(p):
  # Add dataset simulation arguments
  p.add_argument('-n', '--num', type=int, default=10000, help='Dataset size.')
  p.add_argument('--sample', nargs='+', choices=['random', 'dirichlet', 'sobol', 'dirichlet-zeros', 'random-zeros', 'sobol-zeros', 'random-one', 'dirichlet-one', 'sobol-one'],
                 default=['sobol'], help='Concentration sampling method(s).')
  p.add_argument('--noise_p', type=float, default=1.0,
                 help='Probability of ADC noise applied to spectrum. If noise is requested, clean spectra are also saved.')
  p.add_argument('--noise_type', choices=['none', 'adc_normal'], default="adc_normal",
                 help='Type of noise to be added.')
  p.add_argument('--noise_sigma', type=float, default=0.03,
                 help='Maximum sigma for simulated ADC noise (uniform distribution).')
  p.add_argument('--noise_mu', type=float, default=0.0,
                 help='Maximum mu for simulated ADC noise (uniform distribution).')

def add_arguments_compare(p):
  # Add compare arguments
  p.add_argument('-d', '--dataset', type=str, help='Dataset comparison (path ending SOURCE/MANUFACTURER/OEMGA/LINEWIDTH/METABOLITES/PULSE_SEQUENCE/NOISE_P-NOISE_TYPE-NOISE_MU-NOISE_SIGMA/SIZE-ID or dicom folder)')
  p.add_argument('--metabolites', type=lambda s : molecules.short_name(s), nargs='+',
                 default=sorted(['Cr', 'GABA', 'Glu', 'Gln', 'NAA']),
                 help='List of metabolites to use, as defined in mrsnet.molecules: '+str(molecules.NAMES)+'.')
  p.add_argument('--source', type=lambda s : s.lower(),
                 choices=['lcmodel', 'fid-a', 'fid-a-2d', 'pygamma', 'su-3tskyra'], default='lcmodel',
                 help='Data source for the basis spectra (fid-a* requires Matlab).')
  p.add_argument('--manufacturer', type=lambda s : s.lower(),
                 choices=['siemens', 'ge', 'phillips'], default='siemens',
                 help='Scanner manufacturer (fid-a* and pygamma only support siemens).')
  p.add_argument('--omega', type=float, default=123.23, nargs=1,
                 help='Scanner frequency in MHz (default 123.23 MHz for 2.98 T Siemens scanner).')
  p.add_argument('--linewidth', type=float, default=2.0,
                 help='Linewidths to be used for simulation (ignored for lcmodel, su-3tskyra).')
  p.add_argument('--pulse_sequence', type=lambda s : s.lower(), nargs=1,
                 choices=['megapress'], default="megapress",
                 help='Pulse sequence (placeholder).')

def add_arguments_train_select(p):
  # Add training/selection arguments
  p.add_argument('-d', '--dataset', type=str, help='Folder with dataset for training (path ending SOURCE/MANUFACTURER/OEMGA/LINEWIDTH/METABOLITES/PULSE_SEQUENCE/NOISE_P-NOISE_TYPE-NOISE_MU-NOISE_SIGMA/SIZE-ID).')
  p.add_argument('-e', '--epochs', type=int, default=500,
                 help='Number of training epochs.')
  p.add_argument('-k', '--validate', type=float, default=0.7,
                 help='Validation (k>1: k-fold cross-validation; k<-1: duplex k-fold cross-validation; 0..1: train percentage split; -1..0: duplex train percentage split; 0: no split/testing).')
  p.add_argument('-l', '--load_ae', type=str,
                 help='Autoencoder model folder, only for aeq_ model training (path ending MODEL/METABOLITES/PULSE_SEQUENCE/ACQUISITIONS/DATATYPE/NORM/BATCH_SIZE/EPOCHS/TRAIN_DATASET/TRAINER-ID[/fold-N]).')

def add_arguments_train(p):
  # Add training arguments
  p.add_argument('--norm', choices=['sum', 'max'], default='sum',
                 help='Concentration normalisation: sum or max equal to 1')
  p.add_argument('--acquisitions', type=str, nargs='+', default=['edit_off', 'edit_on', 'difference'],
                 help='Acquisitions from pulse sequence used (megapress: edit_off, edit_on, difference).')
  p.add_argument('--datatype', type=lambda s : s.lower(), nargs='+',
                 choices=['magnitude', 'phase', 'real', 'imaginary'], default=['magnitude', 'phase'],
                 help='Data representation of spectrum.')
  p.add_argument('-m', '--model', type=str, default='cnn_small_softmax',
                 help='Model architecture: cnn_[small,medium,large]_[softmax,sigmoid][_pool] or cnn_S1_S2_C1_C2_C3_C4_O1_O2_F1_F2_D_[softmax,sigmoid]- see mrsnet/models.py for details. Or  ae_cnn_FILTER_LATENT_[pool|stride]_DO, ae_fc_LIN_LOUT_ACT_DO, aeq_fc_UNITS_LAYERS_ACT_ACT-LAST_DO for autoencoder (see mrsnet/autoencoder.py)')
  p.add_argument('-a', '--autoencoder', type=str,
                 help='Autoencoder model folder, only for aeq_ model training (path ending MODEL/METABOLITES/PULSE_SEQUENCE/ACQUISITIONS/DATATYPE/NORM/BATCH_SIZE/EPOCHS/TRAIN_DATASET/TRAINER-ID[/fold-N]).')
  p.add_argument('-b', '--batchsize', type=int, default=16,
                 help='Batch size (per GPU if multi-GPU).')

def add_arguments_quantify(p):
  # Add quantification arguments
  p.add_argument('-d', '--dataset', type=str, help='Dataset for quantification (path ending SOURCE/MANUFACTURER/OEMGA/LINEWIDTH/METABOLITES/PULSE_SEQUENCE/NOISE_P-NOISE_TYPE-NOISE_MU-NOISE_SIGMA-SIZE-ID or dicom folder)')
  p.add_argument('-m', '--model', help='Model to quantifiy spectra (path ending MODEL/METABOLITES/PULSE_SEQUENCE/ACQUISITIONS/DATATYPE/NORM/BATCH_SIZE/EPOCHS/TRAIN_DATASET/TRAINER-ID[/fold-N]).')

def add_arguments_benchmark(p):
  # Add benchmark arguments
  p.add_argument('-m', '--model', help='Model to quantify spectra (path ending MODEL/METABOLITES/PULSE_SEQUENCE/ACQUISITIONS/DATATYPE/NORM/BATCH_SIZE/EPOCHS/TRAIN_DATASET/TRAINER-ID[/fold-N]).')

def basis(args):
  # Basis sub-command
  import mrsnet.basis as basis
  if args.verbose > 0:
    print("# Setting up basis")
  bases = basis.BasisCollection()
  args.metabolites.sort()
  args.manufacturer.sort()
  args.omega.sort(key=float)
  args.linewidth.sort(key=float)
  args.pulse_sequence.sort()
  for s in args.source:
    for m in args.manufacturer:
      for o in args.omega:
        for l in args.linewidth:
          for p in args.pulse_sequence:
            bases.add(args.metabolites, s, m, o, l, p,
                      args.sample_rate, args.samples,
                      path_basis=Cfg.val['path_basis'])
  if args.verbose > 0:
    print("# Generating plots")
  for b in bases:
    if args.verbose > 0:
      print(b)
    for f in glob.glob(os.path.join(Cfg.val['path_basis'],b.source,b.name()+"*.png")):
      os.remove(f)
    for d in ['magnitude','phase','real','imaginary']:
      fig = b.plot(data=d, type='fft')
      if args.verbose > 0:
        print(f"Saving figure {b.name()}")
      for dpi in Cfg.val['image_dpi']:
        plt.savefig(os.path.join(Cfg.val['path_basis'],b.source,b.name()+"-"+d+"@"+str(dpi)+".png"), dpi=dpi)
      if args.verbose > 1:
        fig.set_dpi(Cfg.val['screen_dpi'])
        plt.show(block=True)
      plt.close()

def simulate(args):
  # Simulate sub-command
  import mrsnet.basis as basis
  import mrsnet.dataset as dataset
  import mrsnet.spectrum as spectrum
  if args.verbose > 0:
    print("# Setting up bases")
  args.source.sort()
  args.manufacturer.sort()
  args.omega = sorted(args.omega, key=float)
  args.linewidth = sorted(args.linewidth, key=float)
  args.metabolites.sort()
  args.pulse_sequence.sort()
  lw = args.linewidth
  if "lcmodel" in args.source or "su-3tskyra" in args.source:
    lw = [None]
    if len(args.source) > 1:
      for l in args.linewidths:
        lw.append(l)
  name=os.path.join("-".join(args.source)+"_"+str(args.sample_rate)+"_"+str(args.samples),
                    "-".join(args.manufacturer),
                    "-".join([str(k) for k in args.omega]),
                    "-".join([str(k) for k in lw]),
                    "-".join(args.metabolites),
                    "-".join(args.pulse_sequence),
                    "-".join(args.sample),
                    str(args.noise_p)+"-"+args.noise_type+"-"+str(args.noise_mu)+"-"+str(args.noise_sigma))

  bases = basis.BasisCollection()
  num_bases = 0
  for s in args.source:
    for m in args.manufacturer:
      for o in args.omega:
        for l in args.linewidth:
          for ps in args.pulse_sequence:
            bases.add(args.metabolites, s, m, o, l, ps,
                      args.sample_rate, args.samples,
                      path_basis=Cfg.val['path_basis'])
          num_bases += 1
  if args.verbose > 0:
    print("# Generating dataset")
  # Generate datasets
  dataset = dataset.Dataset(name)
  n0 = args.num // num_bases
  n1 = args.num % num_bases
  for b in bases:
    n = n0 + (1 if n1 > 0 else 0)
    n1 -= 1
    if args.verbose > 0:
      print(f"Generating {n} spectra for {b}")
    dataset.generate_spectra(b, n, args.sample, verbose=args.verbose)
  # Save without noise
  if args.verbose > 0:
    print(f"Saving dataset without noise: {dataset.name}")
  path = dataset.save(Cfg.val["path_simulation"])
  # Save with noise
  if args.noise_p > 0.0:
    dataset.add_noise(args.noise_p, args.noise_type, args.noise_mu, args.noise_sigma, verbose=args.verbose)
    if args.verbose > 0:
      print(f"Saving dataset with noise: {dataset.name}")
    path = dataset.save(Cfg.val["path_simulation"], path, spectra_only=True)
  # Plots
  if args.verbose > 0:
    print("Plotting concentrations")
  for norm in ['none', 'sum', 'max']:
    fig = dataset.plot_concentrations(norm=norm)
    if args.verbose > 0:
      print(f"Saving {norm}-normalised concentration figure")
    for f in glob.glob(os.path.join(path,f"concentrations-{norm}@*.png")):
      os.remove(f)
    for dpi in Cfg.val['image_dpi']:
      plt.savefig(os.path.join(path,f"concentrations-{norm}@{dpi}.png"), dpi=dpi)
    if args.verbose > 3 or (args.verbose > 1 and norm == "none"):
      fig.set_dpi(Cfg.val['screen_dpi'])
      plt.show(block=True)
    plt.close()
  if args.verbose > 3:
    for s,c in zip(dataset.spectra,dataset.concentrations):
      spectrum.Spectrum.plot_full_spectrum(s,c,screen_dpi=Cfg.val['screen_dpi'])
      plt.show(block=True)
      plt.close()

def generate_datasets(args):
  # Generate datasets sub-command
  import subprocess
  import mrsnet.grid as grid
  datasets = grid.Grid.load(args.collection)
  k = [str(k) for k in datasets.values.keys()]
  for v in datasets:
    # Check if it exists already
    na = {}
    for ki in range(0,len(k)):
      if isinstance(v[ki],list):
        na[k[ki]] = [str(val) for val in v[ki]]
      else:
        na[k[ki]] = [str(v[ki])]
    name=os.path.join("-".join(na['source'])+"_"+str(na['sample_rate'][0])+"_"+str(na['samples'][0]),
                      "-".join(na['manufacturer']),
                      "-".join(na['omega']),
                      "-".join(na['linewidth']),
                      "-".join(na['metabolites']),
                      "-".join(na['pulse_sequence']),
                      "-".join(na['sample']),
                      na['noise_p'][0]+"-"+na['noise_type'][0]+"-"+na['noise_mu'][0]+"-"+na['noise_sigma'][0])
    if (os.path.exists(os.path.join(Cfg.val['path_simulation'],name,na['num'][0]+"-1","spectra_clean.joblib")) or
        os.path.exists(os.path.join(Cfg.val['path_simulation'],name,na['num'][0]+"-1","spectra_noisy.joblib"))):
      if na['noise_p'][0] > 0.0:
        if not os.path.exists(os.path.join(Cfg.val['path_simulation'],name,na['num'][0]+"-1","spectra_noisy.joblib")):
          raise Exception("No noisy dataset, even if requested: "+Cfg.val['path_simulation'],name,na['num'][0]+"-1")
      if not os.path.exists(os.path.join(Cfg.val['path_simulation'],name,na['num'][0]+"-1","spectra_clean.joblib")):
        raise Exception("Only noisy dataset exists for "+Cfg.val['path_simulation'],name,na['num'][0]+"-1")
      if args.verbose > 0:
        print(f"# Exists: {name}:{na['num'][0]}")
    else:
      # Create
      if args.verbose > 0:
        print(f"# Creating {name}")
      cmd = ['/usr/bin/env', 'python3', 'mrsnet.py', 'simulate']
      if args.verbose > 0:
        cmd += ['-v']*args.verbose
      skip_lw = False
      linewidth1 = False
      for ki in range(0,len(k)):
        if k[ki] == 'source':
          if ((isinstance(v[ki],list) and 'lcmodel' in v[ki]) or
              (not isinstance(v[ki],list) and v[ki] == 'lcmodel') or
              (isinstance(v[ki],list) and 'su-3tskyra' in v[ki]) or
              (not isinstance(v[ki],list) and v[ki] == 'su-3tskyra')):
            skip_lw = True
        if k[ki] == 'linewidth' and not isinstance(v[ki],list) and v[ki] == 1.0: # 1.0 for lcmodel/su-3tskyra interpreated a None linewidth
          linewidth1 = True
        cmd.append("--"+k[ki])
        if isinstance(v[ki],list):
          for val in v[ki]:
            cmd.append(str(val))
        else:
          cmd.append(str(v[ki]))
      if not skip_lw or linewidth1: # Skip unsupported linwidths for lcmodel
        if args.verbose > 0:
          print('# Run '+' '.join(cmd[3:]))
        try:
          p = subprocess.Popen(cmd)
        except OSError as e:
          raise Exception('MRSNet simulations failed') from e
        p.wait()
      else:
        if args.verbose > 0:
          print("Skipping non-1.0 linewidth for lcmodel")

def compare(args):
  # Compare sub-command
  import mrsnet.dataset as dataset
  import numpy as np
  if (os.path.isfile(os.path.join(args.dataset,"spectra_noisy.joblib")) or
      os.path.isfile(os.path.join(args.dataset,"spectra_clean.joblib"))):
    id = get_std_name(args.dataset)
    name = os.path.join(*id[-9:-1])
    ds_rest = id[-1]
    if args.verbose > 0:
      print(f"# Loading dataset {name} : {ds_rest}")
    ds = dataset.Dataset.load(os.path.join(Cfg.val['path_simulation'],name,ds_rest))
  else:
    if args.verbose > 0:
      print(f"# Loading dicom data {args.dataset}")
    concentrations = os.path.join(args.dataset,"concentrations.json")
    if not os.path.isfile(concentrations):
      concentrations = os.path.join(os.path.dirname(args.dataset),"concentrations.json")
      if not os.path.isfile(concentrations):
        concentrations = os.path.join(os.path.dirname(os.path.dirname(args.dataset)),"concentrations.json")
        if not os.path.isfile(concentrations):
          concentrations = None
    ds= dataset.Dataset(args.dataset).load_dicoms(args.dataset,
                                                  concentrations=concentrations,
                                                  metabolites=args.metabolites,
                                                  verbose=args.verbose)
  if len(ds.concentrations) > 0:
    # Get basis
    import mrsnet.basis as basis
    if args.verbose > 0:
      print("# Setting up basis")
    basis = basis.Basis(metabolites=sorted(ds.metabolites), source=args.source,
                        manufacturer=args.manufacturer, omega=args.omega,
                        linewidth=args.linewidth, pulse_sequence=args.pulse_sequence,
                        sample_rate=args.sample_rate, samples=args.samples).setup(Cfg.val['path_basis'])
    # Analyse with given concentrations
    from mrsnet.compare import compare_basis
    compare_basis(ds, basis,
                  verbose=args.verbose, image_dpi=Cfg.val['image_dpi'], screen_dpi=Cfg.val['screen_dpi'])
  else:
    if args.verbose > 0:
      print("Nothing to compare, as no concentrations available/found")

def train(args):
  # Train sub-command
  import mrsnet.dataset as dataset
  # Standardise name, but could be path anyway
  id = get_std_name(args.dataset)
  name = os.path.join(*id[-9:-1])
  ds_rest = id[-1]
  args.metabolites.sort()
  args.acquisitions.sort()
  args.datatype.sort()

  if args.model[0:4] == 'cnn_':
    from mrsnet.cnn import CNN
    if args.verbose > 0:
      print(f"# Loading dataset {name} : {ds_rest}")
    ds = dataset.Dataset.load(os.path.join(Cfg.val['path_simulation'],name,ds_rest))
    model = CNN(args.model, args.metabolites, ds.pulse_sequence,
                args.acquisitions, args.datatype, args.norm)
    d_inp, d_out = ds.export(metabolites=args.metabolites, norm=args.norm,
                             acquisitions=args.acquisitions, datatype=args.datatype,
                             high_ppm=model.high_ppm, low_ppm=model.low_ppm, n_fft_pts=model.fft_samples,
                             verbose=args.verbose)
    data = [d_inp, d_out] # output last
    data_name = ds.name+"_"+ds_rest
  elif args.model[0:3] == 'ae_' or args.model[0:4] == 'aeq_':
    from mrsnet.autoencoder import Autoencoder
    if args.verbose > 0:
      print(f"# Loading dataset {name} : {ds_rest}")
    # Load noisy dataset first
    ds_noisy = dataset.Dataset.load(os.path.join(Cfg.val['path_simulation'],name,ds_rest))
    if args.model[0:3] == 'ae_':
      # If we train the autoencoder, not the quantifier, load clean dataset, if
      # dataset loaded was actualy noisy; otherwise we loaded clean dataset and
      # use it as input and output for the autoencoder
      if ds_noisy.noise_added:
        if args.verbose > 2:
          print("Noisy dataset loaded")
        ds_clean = dataset.Dataset.load(os.path.join(Cfg.val['path_simulation'],
                                                     name,ds_rest), force_clean=True)
        if args.verbose > 2:
          print("Clean dataset loaded")
      else:
        if args.verbose > 0:
          print("Training on clean dataset")
      model = Autoencoder(args.model, args.metabolites, ds_noisy.pulse_sequence,
                          args.acquisitions, args.datatype, args.norm)
      d_noise, _ = ds_noisy.export(metabolites=args.metabolites, norm=args.norm,
                                   acquisitions=args.acquisitions, datatype=args.datatype,
                                   high_ppm=model.high_ppm, low_ppm=model.low_ppm, n_fft_pts=model.fft_samples,
                                   export_concentrations=False, verbose=args.verbose)
      if ds_noisy.noise_added:
        d_clean, _ = ds_clean.export(metabolites=args.metabolites, norm=args.norm,
                                     acquisitions=args.acquisitions, datatype=args.datatype,
                                     high_ppm=model.high_ppm, low_ppm=model.low_ppm, n_fft_pts=model.fft_samples,
                                     export_concentrations=False, verbose=args.verbose)
      else:
        d_clean = d_noise
      data = [d_noise, d_clean] # output last
      data_name = ds_noisy.name+"_"+ds_rest
    else:
      # Or we train the autoencoder as quantifier, meaning the autoencoder model
      # needs to exist already and we just load it.
      #
      # Load the autoencoder model
      id = get_std_name(args.autoencoder)
      name = []
      for k in range(0,len(id)):
        if id[k][0:3] == 'ae_' or id[k][0:4] == 'aeq_':
          name = os.path.join(*id[k:k+6])
          batchsize = id[k+6]
          epochs = id[k+7]
          train_model = id[k+8]
          trainer = id[k+9]
          rest = id[k+10] if len(id) > k+10 else '' # Folds
          break
      if len(name) == 0:
        raise Exception("Cannot get model name from model argument")
      if args.verbose > 0:
        print(f"# Loading autoencoder model {name} : {batchsize} : {epochs} {train_model} : {trainer} : {rest}")
      folder = os.path.join(Cfg.val['path_model'], name, batchsize, epochs, train_model, trainer, rest)
      ae_model = Autoencoder.load(folder)
      encoder = ae_model.ae.encoder
      # Prepare quantifier conversion - need to reconstruct and only set encoder
      # and make sure other parameters ar ethe same than in the autoencoder model
      model = Autoencoder(args.model, ae_model.metabolites, ae_model.pulse_sequence,
                          ae_model.acquisitions, ae_model.datatype, ae_model.norm,
                          encoder=encoder, encoder_model=ae_model.model,
                          encoder_train_dataset_name=ae_model.train_dataset_name)
      # Get data
      d_noise, d_conc = ds_noisy.export(metabolites=args.metabolites, norm=args.norm,
                                        acquisitions=args.acquisitions, datatype=args.datatype,
                                        high_ppm=model.high_ppm, low_ppm=model.low_ppm, n_fft_pts=model.fft_samples,
                                        verbose=args.verbose)
      data = [d_noise, d_conc] # output last
      data_name = ds_noisy.name+"_"+ds_rest
  else:
    raise Exception(f"Unknown model {args.model}")
  if args.verbose > 0:
    print(f"# Model:\n  {str(model)}")

  if args.validate > 1.0:
    from mrsnet.train import KFold
    trainer = KFold(k=int(args.validate))
  elif args.validate < -1.0:
    from mrsnet.train import DuplexKFold
    trainer = DuplexKFold(k=int(-args.validate))
  elif args.validate > 0.0:
    from mrsnet.train import Split
    trainer = Split(p=args.validate)
  elif args.validate < 0.0:
    from mrsnet.train import DuplexSplit
    trainer = DuplexSplit(p=-args.validate)
  elif args.validate == 0.0:
    from mrsnet.train import NoValidation
    trainer = NoValidation()
  else:
    raise Exception(f"Unknown validation {args.validate}")

  trainer.train(model, data, args.epochs, args.batchsize,
                Cfg.val['path_model'], train_dataset_name=data_name,
                image_dpi=Cfg.val['image_dpi'], screen_dpi=Cfg.val['screen_dpi'],
                verbose=args.verbose)

def model_selection(args):
  # Select sub-command
  # FIXME: not tested with ae and aeq models; may not work / need expansion, load_ae need to be stay "None" when using selection for other model(cnn_, ae_fc, ae_cnn)
  import subprocess
  import mrsnet.grid as grid
  args.metabolites.sort()
  models = grid.Grid.load(args.collection)
  if args.method == "grid":
    from mrsnet.selection import SelectGrid
    selector = SelectGrid(args.metabolites,args.dataset,args.epochs,args.validate,args.remote,args.load_ae,
                          Cfg.val['screen_dpi'],Cfg.val['image_dpi'],args.verbose)
  elif args.method == "qmc":
    from mrsnet.selection import SelectQMC
    selector = SelectQMC(args.metabolites,args.dataset,args.epochs,args.validate,args.repeats,args.remote,
                         Cfg.val['screen_dpi'],Cfg.val['image_dpi'],args.verbose)
  elif args.method == "gpo":
    from mrsnet.selection import SelectGPO
    selector = SelectGPO(args.metabolites,args.dataset,args.epochs,args.validate,args.repeats,args.remote,
                         Cfg.val['screen_dpi'],Cfg.val['image_dpi'],args.verbose)
  elif args.method == "ga":
    from mrsnet.selection import SelectGA
    selector = SelectGA(args.metabolites,args.dataset,args.epochs,args.validate,args.repeats,args.remote,
                        Cfg.val['screen_dpi'],Cfg.val['image_dpi'],args.verbose)
  else:
    raise Exception(f"Unknown model selection method {args.method}")
  selector.optimise(args.collection, models, Cfg.val['path_model'])

def quantify(args):
  # Quantify sub-command
  import mrsnet.dataset as dataset
  import numpy as np
  ds_type = ""
  if os.path.isfile(os.path.join(args.dataset,"spectra_noisy.joblib")):
    id = get_std_name(args.dataset)
    ds_name = os.path.join(*id[-9:-1])
    ds_rest = id[-1]
    if args.verbose > 0:
      print(f"# Loading dataset {ds_name} : {ds_rest}")
    ds = dataset.Dataset.load(os.path.join(Cfg.val['path_simulation'],ds_name,ds_rest))
    ds_type = "joblib_noisy"
  elif os.path.isfile(os.path.join(args.dataset,"spectra_clean.joblib")):
    id = get_std_name(args.dataset)
    ds_name = os.path.join(*id[-9:-1])
    ds_rest = id[-1]
    if args.verbose > 0:
      print(f"# Loading dataset {ds_name} : {ds_rest}")
    ds = dataset.Dataset.load(os.path.join(Cfg.val['path_simulation'],ds_name,ds_rest))
    ds_type = "joblib_clean"
  else:
    ds = None # Load later, as dicom and we don't know metabolites
  id = get_std_name(args.model)
  name = []
  for k in range(0,len(id)):
    if id[k][0:4] == 'cnn_' or id[k][0:3] == 'ae_' or id[k][0:4] == 'aeq_':
      name = os.path.join(*id[k:k+6])
      batchsize = id[k+6]
      epochs = id[k+7]
      train_model = id[k+8]
      trainer = id[k+9]
      rest = id[k+10] if len(id) > k+10 else '' # Folds
      break
  if len(name) == 0:
    raise Exception("Cannot get model name from model argument")
  if args.verbose > 0:
    print(f"# Loading model {name} : {batchsize} : {epochs} {train_model} : {trainer} : {rest}")
  folder = os.path.join(Cfg.val['path_model'], name, batchsize, epochs, train_model, trainer, rest)
  if name[0:4] == "cnn_":
    from mrsnet.cnn import CNN
    quantifier = CNN.load(folder)
  elif name[0:3] == "ae_" or name[0:4] == "aeq_":
    from mrsnet.autoencoder import Autoencoder
    quantifier = Autoencoder.load(folder)
  else:
    raise Exception("Unknown model "+name)
  if ds is None:
    if args.verbose > 0:
      print(f"# Loading dicom data {args.dataset}")
    concentrations = os.path.join(args.dataset,"concentrations.json")
    if not os.path.isfile(concentrations):
      concentrations = None
    ds = dataset.Dataset(args.dataset).load_dicoms(args.dataset,
                                                   concentrations=concentrations,
                                                   metabolites=quantifier.metabolites,
                                                   verbose=args.verbose)
    id = get_std_name(args.dataset)
    while id[0] == '.' or id[0] == '..':
      id = id[1:]
    ds_name = os.path.join(*id[:-2])
    ds_rest = id[-1]
    ds_type = "dicom"
  # Export for quantification
  d_inp, d_out = ds.export(metabolites=quantifier.metabolites, norm=quantifier.norm,
                           acquisitions=quantifier.acquisitions, datatype=quantifier.datatype,
                           low_ppm=quantifier.low_ppm, high_ppm=quantifier.high_ppm,
                           n_fft_pts=quantifier.fft_samples, verbose=args.verbose)
  if name[0:3] == "ae_": # Only for autoencoder, not quantifiers
    if ds_type == "joblib_clean":
      d_out = d_inp
    elif ds_type == "joblib_noisy":
      dsc = dataset.Dataset.load(os.path.join(Cfg.val['path_simulation'],ds_name,ds_rest),force_clean=True)
      d_out, _ = ds.export(metabolites=quantifier.metabolites, norm=quantifier.norm,
                           acquisitions=quantifier.acquisitions, datatype=quantifier.datatype,
                           low_ppm=quantifier.low_ppm, high_ppm=quantifier.high_ppm,
                           n_fft_pts=quantifier.fft_samples, verbose=args.verbose,
                           export_concentrations=False)
    else:
      d_out = []
  id_ref = sorted([a for a in ds.spectra[0].keys()])[0]
  # Store results in data repository
  from mrsnet.analyse import analyse_model
  analyse_model(quantifier, d_inp, d_out, folder,
                id=[s[id_ref].id for s in ds.spectra],
                show_conc=True, save_conc=True,
                verbose=args.verbose, prefix=os.path.join(ds_name,ds_rest).replace("/","_"),
                image_dpi=Cfg.val['image_dpi'], screen_dpi=Cfg.val['screen_dpi'])

def benchmark(args):
  # Benchmark sub-command
  if args.verbose > 0:
    print(f"# Loading model {args.model}")
  id = get_std_name(args.model)
  name = []
  for k in range(0,len(id)):
    if id[k][0:4] == 'cnn_' or id[k][0:4] == 'aeq_':
      print(id)
      name = os.path.join(*id[k:k+6])
      batchsize = id[k+6]
      epochs = id[k+7]
      train_model = id[k+8]
      trainer = id[k+9]
      rest = id[k+10] if len(id) > k+10 else '' # Folds
      break
  if len(name) == 0:
    raise Exception("Cannot get model name from model argument")
  if args.verbose > 0:
    print(f"# Model {name} : {batchsize} : {epochs} : {train_model} : {trainer} : {rest}")
  if name[0:4] == "cnn_":
    from mrsnet.cnn import CNN
    quantifier = CNN.load(os.path.join(Cfg.val['path_model'], name, batchsize, epochs, train_model, trainer, rest))
  elif name[0:4] == "aeq_":
    from mrsnet.autoencoder import Autoencoder
    quantifier = Autoencoder.load(os.path.join(Cfg.val['path_model'], name, batchsize, epochs, train_model, trainer, rest))
  elif name[0:3] == "ae_":
    raise Exception("No concentration prediction implemented")
  else:
    raise Exception("Unknown model "+name)
  import mrsnet.dataset as dataset
  for id in ['E1', 'E2', 'E3', 'E4a', 'E4b', 'E4c', 'E4d']:
    for variant in ['MEGA_Combi_WS_ON', 'MEGA_RAW_Combi_WS_ON']:
      if args.verbose > 0:
        print(f"# Loading Benchmark {id}")
      bm = dataset.Dataset(id).load_dicoms(os.path.join(Cfg.val['path_benchmark'], id, variant),
                                           concentrations=os.path.join(Cfg.val['path_benchmark'],
                                                                       id, 'concentrations.json'),
                                           metabolites=quantifier.metabolites,
                                           verbose=args.verbose)
      if args.verbose > 3:
        for s,c in zip(bm.spectra,bm.concentrations):
          for a in s.keys():
            s[a].plot_spectrum(c,screen_dpi=Cfg.val['screen_dpi'])
            plt.show(block=True)
            plt.close()
      d_inp, d_out = bm.export(metabolites=quantifier.metabolites, n_fft_pts=quantifier.fft_samples,
                               high_ppm=quantifier.high_ppm, low_ppm=quantifier.low_ppm,
                               norm=quantifier.norm, acquisitions=quantifier.acquisitions,
                               datatype=quantifier.datatype,
                               verbose=args.verbose)
      from mrsnet.analyse import analyse_model
      id_ref = sorted([a for a in bm.spectra[0].keys()])[0]
      analyse_model(quantifier, d_inp, d_out, os.path.join(Cfg.val['path_model'], name, batchsize, epochs,
                                                           train_model, trainer, rest),
                    id=[s[id_ref].id for s in bm.spectra],
                    show_conc=True, save_conc=True,
                    verbose=args.verbose, prefix=id+":"+variant, image_dpi=Cfg.val['image_dpi'],
                    screen_dpi=Cfg.val['screen_dpi'])

def get_std_name(name):
  _, path = os.path.splitdrive(name)
  id = []
  while True:
    path, folder = os.path.split(path)
    if folder != "":
      id.append(folder)
    if path == "" or path == '/':
      break
  id.reverse()
  return id

if __name__ == '__main__':
  # Find base folder
  if not os.name == 'posix':
    print("**WARNING - MRSNet only runs reliably and is only supported on Linux/POSIX**")
  bin_path = os.path.realpath(__file__)
  if not os.path.isfile(bin_path):
    raise Exception("Cannot find location of mrsnet.py root folder")
  Cfg.init(bin_path)
  # Only print warnings and errors for tf (set before importing tf)
  if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  # Headless mode
  if not "DISPLAY" in os.environ:
    from matplotlib import use
    use("Agg")
  main()
