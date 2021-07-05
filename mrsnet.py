#!/usr/bin/env python3
#
# mrsnet.py - MRSNet - command line MRSNet interface
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2019, Max Chandler, PhD student at Cardiff University
# Copyright (C) 2020-2021, Frank C Langbein <frank@langbein.org>, Cardiff University
#
# See --help for arguments, uses sub-commands

import os
import sys
import argparse
import datetime

import mrsnet.molecules as molecules

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
  p_basis.set_defaults(func=basis)

  # Generate simulated dataset
  p_spectra = subparsers.add_parser('simulate', help='Generate simulated spectra dataset.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  add_arguments_default(p_spectra)
  add_arguments_metabolites(p_spectra)
  add_arguments_basis(p_spectra)
  add_arguments_simulate(p_spectra)
  p_spectra.add_argument('-d', '--dataset', type=str, default=os.path.join('data', 'sim-spectra', datetime.datetime.now().strftime('%d%m%y_%H%M%S')),
                         help='Folder to store dataset in.')
  p_spectra.add_argument('--show-spectra', action='store_true', default=False,
                         help='Shows plots of all generated spectra.')
  p_spectra.set_defaults(func=simulate)

  # Train
  p_train = subparsers.add_parser('train', help='Train model on dataset.',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  add_arguments_default(p_train)
  add_arguments_metabolites(p_train)
  p_train.add_argument('-d', '--dataset', type=str, help='Folder with dataset for training.')
  p_train.add_argument('--norm', choices=['sum', 'max'], default='sum',
                       help='Concentration normalisation: sum or max equal to 1')
  p_train.add_argument('--acquisitions', type=str, nargs='+', default=['edit_off', 'difference'],
                       help='Acquisitions from pulse sequence used (megapress: edit_off, edit_on, difference).')
  p_train.add_argument('--datatype', type=lambda s : s.lower(), nargs='+',
                       choices=['magnitude', 'real', 'imaginary'], default=['magnitude'],
                       help='Data representation of spectrum.')
  p_train.add_argument('--model_folder', type=str, default=os.path.join('data', 'model', datetime.datetime.now().strftime('%d%m%y_%H%M%S')),
                       help='Folder to store model.')
  p_train.add_argument('-m', '--model', type=str, default='cnn_medium_softmax',
                       help='Model architecture: cnn_[small,medium,large]_[softmax,sigmoid][_pool] or cnn_[S1]_[S2]_[C1]_[C2]_[C3]_[C4]_[O1]_[O2]_[F1]_[F2]_[D]_[softmax,sigmoid][_pool]- see mrsnet/models.py for details.')
  p_train.add_argument('-b', '--batch_size', type=int, default=64,
                       help='Batch size.')
  p_train.add_argument('-e', '--epochs', type=int, default=100,
                       help='Number of training epochs.')
  p_train.add_argument('--validate', type=float, default=10,
                       help='Validation (k>1: k-fold cross-validation; k<-1: duplex k-fold cross-validation; 0..1: train percentage split; -1..0: duplex train percentage split; 0: no split/testing).')
  p_train.set_defaults(func=train)

  # Quantify
  p_quantify = subparsers.add_parser('quantify', help='Quantify spectra in dicoms.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  add_arguments_default(p_quantify)
  p_quantify.add_argument('-d', '--dataset', type=str, help='Folder with dataset for quantification.')
  p_quantify.add_argument('-m', '--model', help='Folder containing the model to quantifiy spectra.',
                          default=os.path.join('data', 'model', 'MRSNet_LCModel'))
  p_quantify.set_defaults(func=quantify)

  # Benchmark
  p_benchmark = subparsers.add_parser('benchmark', help='Run benchmark on model.')
  add_arguments_default(p_benchmark)
  p_benchmark.add_argument('-m', '--model', help='Folder containing the model to quantify spectra.',
                           default=os.path.join('data', 'model', 'MRSNet_LCModel'))
  p_benchmark.add_argument('--show-spectra', action='store_true', default=False,
                           help='Shows plots of all spectra.')
  p_benchmark.set_defaults(func=benchmark)

  args = parser.parse_args()
  if hasattr(args, 'metabolites'):
    args.metabolites = sorted(args.metabolites) # Make sure metabolite names are sorted
  if hasattr(args,"func"):
    args.func(args)
  else:
    print("%s: illegal sub-command or sub-command not specified, see help [-h]" % sys.argv[0], file=sys.stderr)

def add_arguments_default(p):
  # Add argumnets for all sub-commands
  p.add_argument('-v', '--verbose', action='count', help='Increase output verbosity.', default=0)
  p.add_argument('--no-show', action='store_true', default=False, help='Do not show any plots.')

def add_arguments_metabolites(p):
  # Add metabolites argument
  p.add_argument('--metabolites', type=lambda s : molecules.short_name(s), nargs='+',
                 default=sorted(['Cr', 'GABA', 'Glu', 'Gln', 'NAA']),
                 help='List of metabolites to use, as defined in mrsnet.molecules: '+str(molecules.NAMES)+'.')

def add_arguments_basis(p):
  # Add basis source arguments
  p.add_argument('--source', type=lambda s : s.lower(),
                 choices=['lcmodel', 'fid-a', 'pygamma'], default=['lcmodel'],
                 nargs='+',
                 help='Data source(s) for the basis spectra (fid-a required Matlab).')
  p.add_argument('--manufacturer', type=lambda s : s.lower(),
                 choices=['siemens', 'ge', 'phillips'], default=['siemens'],
                 nargs='+',
                 help='Scanner manufacturer (fid-a and pygamma only support siemens).')
  p.add_argument('--omega', type=float, default=[123.23], nargs='+',
                 help='Scanner frequency in MHz (default 123.23 MHz for 2.98 T Siemens scanner).')
  p.add_argument('--linewidth', type=float, nargs='+', default=[1.0],
                 help='Linewidths to be used for simulation (not possible for lcmodel).')
  p.add_argument('--pulse_sequence', type=lambda s : s.lower(), nargs='+',
                 choices=['megapress'], default="megapress",
                 help='Pulse sequence (placeholder).')

def add_arguments_simulate(p):
  # Add dataset generation arguments
  p.add_argument('-n', '--num', type=int, default=10000, help='Dataset size.')
  p.add_argument('--sample', nargs='+', choices=['random', 'dirichlet', 'sobol', 'dirichlet-zeros', 'random-zeros', 'sobol-zeros', 'random-one', 'dirichlet-one', 'sobol-one'],
                 default=['sobol'], help='Concentration sampling method(s).')
  p.add_argument('--noise_p', type=float, default=1.0,
                 help='Probability of ADC noise applied to spectrum.')
  p.add_argument('--noise_sigma', type=float, default=0.1,
                 help='Maximum sigma for simulated ADC noise (uniform distribution).')
  p.add_argument('--noise_mu', type=float, default=0.0,
                 help='Maximum mu for simulated ADC noise (uniform distribution).')

def basis(args):
  # Basis sub-command
  import mrsnet.basis as basis
  if args.verbose > 0:
    print("# Setting up bases")
  bases = basis.BasisCollection()
  args.metabolites.sort()
  for s in args.source:
    for m in args.manufacturer:
      for o in args.omega:
        for l in args.linewidth:
          for p in args.pulse_sequence:
            bases.add(args.metabolites, s, m, o, l, p)
  if args.verbose > 0:
    print(bases)
    print("# Generating plots")
  import matplotlib.pyplot as plt
  for b in bases:
    if args.verbose > 0:
      print(b)
    fig = b.plot()
    for dpi in Cfg.val['image_dpi']:
      if args.verbose > 0:
        print("Saving figure %s @ %ddpi" % (b.name(),res))
      plt.savefig(os.path.join('data','basis',b.name()+"@"+str(dpi)+".png"), dpi=dpi)
    if not args.no_show:
      fig.set_dpi(Cfg.val['screen_dpi'])
      plt.show(block=True)
    plt.close()

def simulate(args):
  # Simulate sub-command
  import mrsnet.basis as basis
  import mrsnet.dataset as dataset
  if args.verbose > 0:
    print("# Setting up bases")
  bases = basis.BasisCollection()
  args.metabolites.sort()
  num_bases = 0
  for s in args.source:
    for m in args.manufacturer:
      for o in args.omega:
        for l in args.linewidth:
          bases.add(args.metabolites, s, m, o, l, args.pulse_sequence)
          num_bases += 1
  if args.verbose > 0:
    print(bases)
    print("# Generating dataset")
  # Generate datasets
  dataset = dataset.Dataset(args.dataset)
  n0 = args.num // num_bases
  n1 = args.num % num_bases
  for b in bases:
    n = n0 + (1 if n1 > 0 else 0)
    n1 -= 1
    if args.verbose > 0:
      print("Generating %d spectra for %s" % (n,b))
    dataset.generate_spectra(b, n, args.sample, args.noise_p, args.noise_mu,
                             args.noise_sigma, verbose=args.verbose)
  if args.verbose > 0:
    print("Saving dataset %s" % dataset.name)
  dataset.save()
  if args.verbose > 0:
    print("Plotting concentrations")
  import matplotlib.pyplot as plt
  for norm in ['none', 'sum', 'max']:
    fig = dataset.plot_concentrations(norm=norm)
    for res in Cfg.val['image_dpi']:
      if args.verbose > 0:
        print("Saving %s-normalised concentration figure %s @ %dpi" % (norm,dataset.name,dpi))
      plt.savefig(os.path.join(dataset.name,"concentrations-%s@%d.png" % (norm,dpi)), dpi=dpi)
    if not args.no_show:
      fig.set_dpi(Cfg.val['screen_dpi'])
      plt.show(block=True)
      plt.close()
      if args.show_spectra:
        for s,c in zip(dataset.spectra,dataset.concentrations):
          for a in s.keys():
            s[a].plot_spectrum(c)
            plt.show(block=True)
            plt.close()

def train(args):
  # Train sub-command
  import mrsnet.dataset as dataset
  if args.verbose > 0:
    print("# Loading dataset %s" % args.dataset)
  dataset = dataset.Dataset.load(args.dataset)
  args.metabolites.sort()
  args.acquisitions.sort()
  args.datatype.sort()
  d_inp, d_out = dataset.export(metabolites=args.metabolites, norm=args.norm,
                                acquisitions=args.acquisitions, datatype=args.datatype,
                                verbose=args.verbose)

  if args.model[0:4] == 'cnn_':
    from mrsnet.model import CNN
    model = CNN(args.model, args.metabolites, dataset.pulse_sequence,
                args.acquisitions, args.datatype, args.norm)
  else:
    raise Exception("Unknown model %s" % args.model)
  if args.verbose > 0:
    print("# Model setup:\n  %s" % str(model))

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
    # TODO: only training
  else:
    raise Exception("Unknown validation %f" % args.validate)
  trainer.train(model, d_inp, d_out, args.epochs, args.batch_size,
                args.model_folder, train_dataset_name=dataset.name,
                image_dpi=Cfg.val['image_dpi'], screen_dpi=Cfg.val['screen_dpi'],
                no_show=args.no_show, verbose=args.verbose)

def quantify(args):
  # Quantify sub-command
  import mrsnet.dataset as dataset
  if args.verbose > 0:
    print("# Loading dataset %s" % args.dataset)
  # FIXME: Load dicoms as alternative
  dataset = dataset.Dataset.load(args.dataset)
  # Load model (FIXME: issue when more models than CNN class)
  if args.verbose > 0:
    print("# Loading model %s" % args.model)
  from mrsnet.model import CNN
  cnn = CNN.load(args.model)
  # Export for quantification
  d_inp, d_out = dataset.export(metabolites=cnn.metabolites, norm=cnn.norm,
                                acquisitions=cnn.acquisitions, datatype=cnn.datatype,
                                verbose=args.verbose)
  from mrsnet.analyse import analyse_model
  analyse_model(cnn, d_inp, d_out, args.dataset,
                id=[s['edit_off'].id for s in dataset.spectra], # FIXME: maybe cannot assume edit_off here
                show_conc=True, save_conc=True, no_show=args.no_show,
                verbose=args.verbose, prefix='analyse_'+str(cnn), image_dpi=Cfg.val['image_dpi'], screen_dpi=Cfg.val['screen_dpi'])

def benchmark(args):
  # Benchmark sub-command
  if args.verbose > 0:
    print("# Loading model %s" % args.model)
  from mrsnet.model import CNN
  cnn = CNN.load(args.model)
  import mrsnet.dataset as dataset
  for id in ['E1', 'E2', 'E3', 'E4a', 'E4b', 'E4c', 'E4d']:
    if args.verbose > 0:
      print("# Loading Benchmark %s" % id)
    bm = dataset.Dataset(id).load_dicoms(os.path.join('data','benchmark', id, 'MEGA_Combi_WS_ON'),
                                         concentrations=os.path.join('data','benchmark', id, 'concentrations.json'),
                                         metabolites=cnn.metabolites)
    if args.show_spectra:
      import matplotlib.pyplot as plt
      for s,c in zip(bm.spectra,bm.concentrations):
        for a in s.keys():
          s[a].plot_spectrum(c)
          plt.show(block=True)
          plt.close()
    d_inp, d_out = bm.export(metabolites=cnn.metabolites, norm=cnn.norm,
                             acquisitions=cnn.acquisitions, datatype=cnn.datatype,
                             verbose=args.verbose)
    from mrsnet.analyse import analyse_model
    analyse_model(cnn, d_inp, d_out, os.path.dirname(args.model),
                  id=[s['edit_off'].id for s in bm.spectra], # FIXME: maybe cannot assume edit_off here
                  show_conc=True, save_conc=True, no_show=args.no_show,
                  verbose=args.verbose, prefix='benchmark_'+id+'_'+str(cnn), image_dpi=Cfg.val['image_dpi'],
                  screen_dpi=Cfg.val['screen_dpi'])

class Cfg:
  # Default configuration - do not overwrite here but set alternatives in file
  val = {
    'default_screen_dpi': 96,
    'image_dpi': [300]
  }
  file = os.path.expanduser("~/.config/mrsnet.json")

  @staticmethod
  def init():
    if os.path.isfile(Cfg.file):
      import json
      with open(Cfg.file, "r") as fp:
        js = json.load(fp)
        for k in js.keys():
          if k in Cfg.val:
            Cfg.val[k] = js[k]
          else:
            raise Exception("Unknown config file entry %s in %s" % (k,Cfg.file))
    Cfg.val["screen_dpi"] = Cfg.get_screen_dpi()

  @staticmethod
  def get_screen_dpi():
    # DPI for plots on screen
    try:
      from screeninfo import get_monitors
    except ModuleNotFoundError:
      return Cfg.val['default_screen_dpi']
    from math import hypot
    m = get_monitors()[0]
    return hypot(m.width, m.height) / hypot(m.width_mm, m.height_mm) * 25.4

if __name__ == '__main__':
  # Only print warnings and errors for tf (set before importing tf)
  if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
  # Headless mode
  if not "DISPLAY" in os.environ:
    from matplotlib import use
    use("Agg")
  Cfg.init()
  main()
