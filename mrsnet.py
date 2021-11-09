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
import glob
import sys
import argparse
import matplotlib.pyplot as plt

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
  p_simulate = subparsers.add_parser('simulate', help='Generate simulated spectra dataset.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  add_arguments_default(p_simulate)
  add_arguments_metabolites(p_simulate)
  add_arguments_basis(p_simulate)
  add_arguments_simulate(p_simulate)
  p_simulate.set_defaults(func=simulate)

  # Generate all datasets
  p_gen_ds = subparsers.add_parser('generate_datasets', help='Generate standard datasets.',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  add_arguments_default(p_gen_ds)
  p_gen_ds.add_argument('collection', type=str, help='Dataset collection name (basic-1, mixed-1; see mrsnet.dataset.Collection)')
  p_gen_ds.set_defaults(func=generate_datasets)

  # Train
  p_train = subparsers.add_parser('train', help='Train model on dataset.',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  add_arguments_default(p_train)
  add_arguments_metabolites(p_train)
  add_arguments_train_select(p_train)
  add_arguments_train(p_train)
  p_train.set_defaults(func=train)

  # AETrain
  p_aetrain = subparsers.add_parser('aetrain', help='Train autoencoder model on dataset.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  add_arguments_default(p_aetrain)
  add_arguments_metabolites(p_aetrain)
  add_arguments_train_select(p_aetrain)
  add_arguments_train(p_aetrain)
  p_aetrain.set_defaults(func=aetrain)

  # Model selection
  p_select = subparsers.add_parser('select', help='Model selection on dataset.',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  add_arguments_default(p_select)
  add_arguments_metabolites(p_select)
  add_arguments_train_select(p_select)
  p_select.add_argument('--method', choices=['grid', 'qmc', 'gpo', 'evo'], default='random',
                        help='Model selection approach')
  p_select.add_argument('-r', '--repeats', type=int, default=100,
                        help='Maximum number of repeats (for qmc, gpo, evo).')
  p_select.add_argument('--remote', type=str, default='',
                        help='Remote execution: scheduler:user:[max_parallel_tasks=10:[wait_minutes=15]]')
  p_select.add_argument('collection', type=str, help='Model collection name (see mrsnet.selection.Collection)')
  p_select.set_defaults(func=model_selection)

  # Quantify
  p_quantify = subparsers.add_parser('quantify', help='Quantify spectra in dicoms.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  add_arguments_default(p_quantify)
  add_arguments_train(p_quantify)
  p_quantify.set_defaults(func=quantify)

  # Benchmark
  p_benchmark = subparsers.add_parser('benchmark', help='Run benchmark on model.')
  add_arguments_default(p_benchmark)
  add_arguments_benchmark(p_benchmark)
  p_benchmark.set_defaults(func=benchmark)

  args = parser.parse_args()
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
                 choices=['megapress'], default=["megapress"],
                 help='Pulse sequence (placeholder).')

def add_arguments_simulate(p):
  # Add dataset simulation arguments
  p.add_argument('-n', '--num', type=int, default=10000, help='Dataset size.')
  p.add_argument('--sample', nargs='+', choices=['random', 'dirichlet', 'sobol', 'dirichlet-zeros', 'random-zeros', 'sobol-zeros', 'random-one', 'dirichlet-one', 'sobol-one'],
                 default=['sobol'], help='Concentration sampling method(s).')
  p.add_argument('--noise_p', type=float, default=1.0,
                 help='Probability of ADC noise applied to spectrum.')
  p.add_argument('--noise_sigma', type=float, default=0.1,
                 help='Maximum sigma for simulated ADC noise (uniform distribution).')
  p.add_argument('--noise_mu', type=float, default=0.0,
                 help='Maximum mu for simulated ADC noise (uniform distribution).')
  p.add_argument('--show-spectra', action='store_true', default=False,
                 help='Shows plots of all generated spectra.')

def add_arguments_train_select(p):
  # Add training/selection arguments
  p.add_argument('-d', '--dataset', type=str, help='Folder with dataset for training (path ending SOURCE/MANUFACTURER/OEMGA/LINEWIDTH/METABOLITES/PULSE_SEQUENCE/NOISE_P-NOISE_MU-NOISE_SIGMA/SIZE-ID).')
  p.add_argument('-e', '--epochs', type=int, default=500,
                 help='Number of training epochs.')
  p.add_argument('--validate', type=float, default=5,
                 help='Validation (k>1: k-fold cross-validation; k<-1: duplex k-fold cross-validation; 0..1: train percentage split; -1..0: duplex train percentage split; 0: no split/testing).')

def add_arguments_train(p):
  # Add training arguments
  p.add_argument('--norm', choices=['sum', 'max'], default='sum',
                 help='Concentration normalisation: sum or max equal to 1')
  p.add_argument('--acquisitions', type=str, nargs='+', default=['edit_off', 'difference'],
                 help='Acquisitions from pulse sequence used (megapress: edit_off, edit_on, difference).')
  p.add_argument('--datatype', type=lambda s : s.lower(), nargs='+',
                 choices=['magnitude', 'phase', 'real', 'imaginary'], default=['magnitude'],
                 help='Data representation of spectrum.')
  p.add_argument('-m', '--model', type=str, default='cnn_medium_softmax',
                 help='Model architecture: cnn_[small,medium,large]_[softmax,sigmoid][_pool] or cnn_[S1]_[S2]_[C1]_[C2]_[C3]_[C4]_[O1]_[O2]_[F1]_[F2]_[D]_[softmax,sigmoid][_pool]- see mrsnet/models.py for details.')
  p.add_argument('-b', '--batch-size', type=int, default=32,
                 help='Batch size.')

def add_arguments_quantify(p):
  # Add quantification arguments
  p.add_argument('-d', '--dataset', type=str, help='Dataset for quantification (path ending SOURCE/MANUFACTURER/OEMGA/LINEWIDTH/METABOLITES/PULSE_SEQUENCE/NOISE_P-NOISE_MU-NOISE_SIGMA-SIZE-ID)')
  p.add_argument('-m', '--model', help='Model to quantifiy spectra (path ending MODEL/METABOLITES/PULSE_SEQUENCE/ACQUISITIONS/DATATYPE/NORM/BATCH_SIZE/EPOCHS/TRAIN_DATASET/TRAINER-ID[/fold-N]).',
                 default=os.path.join('data', 'model', 'MRSNet_LCModel'))

def add_arguments_benchmark(p):
  # Add benchmark arguments
  p.add_argument('-m', '--model', help='Model to quantify spectra (path ending MODEL/METABOLITES/PULSE_SEQUENCE/ACQUISITIONS/DATATYPE/NORM/BATCH_SIZE/EPOCHS/TRAIN_DATASET/TRAINER-ID[/fold-N]).',
                 default=os.path.join('data', 'model', 'MRSNet_LCModel'))
  p.add_argument('--show-spectra', action='store_true', default=False,
                 help='Shows plots of all spectra.')

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
            bases.add(args.metabolites, s, m, o, l, p, path_basis=Cfg.val['path_basis'])
  if args.verbose > 0:
    print(bases)
    print("# Generating plots")
  for b in bases:
    if args.verbose > 0:
      print(b)
    fig = b.plot()
    if args.verbose > 0:
      print("Saving figure %s" % b.name())
    for f in glob.glob(os.path.join(Cfg.val['path_basis'],b.source,b.name()+"@*.png")):
      os.remove(f)
    for dpi in Cfg.val['image_dpi']:
      plt.savefig(os.path.join(Cfg.val['path_basis'],b.source,b.name()+"@"+str(dpi)+".png"), dpi=dpi)
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
  args.source.sort()
  args.manufacturer.sort()
  args.omega = sorted(args.omega, key=float)
  args.linewidth = sorted(args.linewidth, key=float)
  args.metabolites.sort()
  args.pulse_sequence.sort()
  name=os.path.join("-".join(args.source),
                    "-".join(args.manufacturer),
                    "-".join([str(k) for k in args.omega]),
                    "-".join([str(k) for k in args.linewidth]),
                    "-".join(args.metabolites),
                    "-".join(args.pulse_sequence),
                    "-".join(args.sample),
                    str(args.noise_p)+"-"+str(args.noise_mu)+"-"+str(args.noise_sigma))

  bases = basis.BasisCollection()
  num_bases = 0
  for s in args.source:
    for m in args.manufacturer:
      for o in args.omega:
        for l in args.linewidth:
          for ps in args.pulse_sequence:
            bases.add(args.metabolites, s, m, o, l, ps)
          num_bases += 1
  if args.verbose > 0:
    print(bases)
    print("# Generating dataset")
  # Generate datasets
  dataset = dataset.Dataset(name)
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
  path = dataset.save(Cfg.val["path_simulation"])
  if args.verbose > 0:
    print("Plotting concentrations")
  for norm in ['none', 'sum', 'max']:
    fig = dataset.plot_concentrations(norm=norm)
    if args.verbose > 0:
      print("Saving %s-normalised concentration figure" % norm)
    for f in glob.glob(os.path.join(path,"concentrations-%s@*.png" % norm)):
      os.remove(f)
    for dpi in Cfg.val['image_dpi']:
      plt.savefig(os.path.join(path,"concentrations-%s@%d.png" % (norm,dpi)), dpi=dpi)
    if not args.no_show:
      fig.set_dpi(Cfg.val['screen_dpi'])
      plt.show(block=True)
      plt.close()
  if args.show_spectra:
    for s,c in zip(dataset.spectra,dataset.concentrations):
      for a in s.keys():
        s[a].plot_spectrum(c,screen_dpi=Cfg.val['screen_dpi'])
        plt.show(block=True)
        plt.close()

def generate_datasets(args):
  # Generate datasets sub-command
  import subprocess
  import mrsnet.grid as grid
  from mrsnet.dataset import Collections
  datasets = Collections[args.collection]
  k = [str(k) for k in datasets.values.keys()]
  for v in datasets:
    # Check if it exists already
    na = {}
    for ki in range(0,len(k)):
      if isinstance(v[ki],list):
        na[k[ki]] = [str(val) for val in v[ki]]
      else:
        na[k[ki]] = [str(v[ki])]
    name=os.path.join("-".join(na['source']),
                      "-".join(na['manufacturer']),
                      "-".join(na['omega']),
                      "-".join(na['linewidth']),
                      "-".join(na['metabolites']),
                      "-".join(na['pulse_sequence']),
                      "-".join(na['sample']),
                      na['noise_p'][0]+"-"+na['noise_mu'][0]+"-"+na['noise_sigma'][0])
    if os.path.exists(os.path.join(Cfg.val['path_simulation'],name,na['num'][0]+"-1","spectra.joblib")):
      if args.verbose > 0:
        print("Exists: %s:%s" % (name,na['num'][0]))
    else:
      # Create
      cmd = ['/usr/bin/env', 'python3', 'mrsnet.py', 'simulate', '--no-show']
      if args.verbose > 0:
        cmd += ['-v']*args.verbose
      lcmodel = False
      linewidth1 = False
      for ki in range(0,len(k)):
        if k[ki] == 'source' and ((isinstance(v[ki],list) and 'lcmodel' in v[ki]) or
                                  (not isinstance(v[ki],list) and v[ki] == 'lcmodel')):
          lcmodel = True
        if k[ki] == 'linewidth' and not isinstance(v[ki],list) and v[ki] == 1.0:
          linewidth1 = True
        cmd.append("--"+k[ki])
        if isinstance(v[ki],list):
          for val in v[ki]:
            cmd.append(str(val))
        else:
          cmd.append(str(v[ki]))
      if not lcmodel or linewidth1: # Skip unsupported linwidths for lcmodel
        if args.verbose > 0:
          print('# Run '+' '.join(cmd[3:]))
        try:
          p = subprocess.Popen(cmd)
        except OSError as e:
          raise Exception('MRSNet simulations failed') from e
        p.wait()

def train(args):
  # Train sub-command
  import mrsnet.dataset as dataset
  # Standardise name, but could be path anyway
  id = get_std_name(args.dataset)
  name = os.path.join(*id[-9:-1])
  ds_rest = id[-1]
  if args.verbose > 0:
    print("# Loading dataset %s : %s" % (name,ds_rest))
  dataset = dataset.Dataset.load(os.path.join(Cfg.val['path_simulation'],name,ds_rest))
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
  else:
    raise Exception("Unknown validation %f" % args.validate)
  trainer.train(model, d_inp, d_out, args.epochs, args.batch_size,
                Cfg.val['path_model'], train_dataset_name=dataset.name+"_"+ds_rest,
                image_dpi=Cfg.val['image_dpi'], screen_dpi=Cfg.val['screen_dpi'],
                no_show=args.no_show, verbose=args.verbose)

def aetrain(args):
  # Aerain sub-command
  import mrsnet.dataset as dataset
  # Standardise name, but could be path anyway
  id = get_std_name(args.dataset)
  name = os.path.join(*id[-9:-1])
  ds_rest = id[-1]
  if args.verbose > 0:
    print("# Loading dataset %s : %s" % (name,ds_rest))
  dataset = dataset.Dataset.load(os.path.join(Cfg.val['path_simulation'],name,ds_rest))
  args.metabolites.sort()
  args.acquisitions.sort()
  args.datatype.sort()
  d_inp, d_out = dataset.export(metabolites=args.metabolites, norm=args.norm,
                                acquisitions=args.acquisitions, datatype=args.datatype,
                                verbose=args.verbose)
  import mrsnet.aetrain as aetrain
  # Call to autoencoder train function - adjust arguments, etc. as necesary (those listed are all available with the current arguments from the mrsnet call)
  # Create ae-trainer class to store arguments
  trainer = aetrain.AETrain(args.model, args.metabolites, dataset.pulse_sequence,
                            args.acquisitions, args.datatype, args.norm,
                            args.validate,
                            d_inp, d_out, args.epochs, args.batch_size,
                            Cfg.val['path_model'], dataset.name+"_"+ds_rest,
                            Cfg.val['image_dpi'], Cfg.val['screen_dpi'],
                            args.no_show, args.verbose)
  # ...and execute trainer (to be implemented - FIXME)
  trainer.train()

def model_selection(args):
  # Select sub-command
  import subprocess
  from mrsnet.selection import Collections
  args.metabolites.sort()
  models = Collections[args.collection]
  if args.method == "grid":
    from mrsnet.selection import SelectGrid
    selector = SelectGrid(args.metabolites,args.dataset,args.epochs,args.validate,args.remote,
                          Cfg.val['screen_dpi'],Cfg.val['image_dpi'],args.no_show,args.verbose)
  elif args.method == "qmc":
    from mrsnet.selection import SelectQMC
    selector = SelectQMC(args.metabolites,args.dataset,args.epochs,args.validate,args.repeats,args.remote,
                         Cfg.val['screen_dpi'],Cfg.val['image_dpi'],args.no_show,args.verbose)
  elif args.method == "gpo":
    from mrsnet.selection import SelectGPO
    selector = SelectGPO(args.metabolites,args.dataset,args.epochs,args.validate,args.repeats,args.remote,
                         Cfg.val['screen_dpi'],Cfg.val['image_dpi'],args.no_show,args.verbose)
  elif args.method == "evo":
    from mrsnet.selection import SelectEvo
    selector = SelectEvo(args.metabolites,args.dataset,args.epochs,args.validate,args.repeats,args.remote,
                         Cfg.val['screen_dpi'],Cfg.val['image_dpi'],args.no_show,args.verbose)
  else:
    raise Exception("Unknown model selection method %s" % args.method)
  selector.optimise(args.collection, models, Cfg.val['path_model'])

def quantify(args):
  # Quantify sub-command
  import mrsnet.dataset as dataset
  import numpy as np
  if os.path.isfile(os.path.join(args.dataset,"spectra.joblib")):
    id = get_std_name(args.dataset)
    name = os.path.join(*id[-9:-1])
    ds_rest = id[-1]
    if args.verbose > 0:
      print("# Loading dataset %s : %s" % (name,ds_rest))
    ds = dataset.Dataset.load(os.path.join(Cfg.val['path_simulation'],name,ds_rest))
  else:
    ds = None # Load later, as dicom and we don't know metabolites
  id = get_std_name(args.model)
  name = []
  for k in range(0,len(id)):
    if id[k][0:4] == 'cnn_':
      name = os.path.join(*id[k:k+6])
      train_model = id[k+6]
      batch_size = id[k+7]
      epochs = id[k+8]
      trainer = id[k+9]
      rest = id[k+10] if len(id) > k+10 else '' # Folds
      break
  if len(name) == 0:
    raise Exception("Cannot get model name from model argument")
  if args.verbose > 0:
    print("# Loading model %s : %s : %s %s : %s : %s" % (name,batch_size,epochs,train_model,trainer,rest))
  if name[0:4] == "cnn_":
    from mrsnet.model import CNN
    quantifier = CNN.load(os.path.join(Cfg.val['path_model'], name, batch_size, epochs, train_model, trainer, rest))
  else:
    raise Exception("Unknown model "+name)
  if ds is None:
    if args.verbose > 0:
      print("# Loading dicom data %s" % args.dataset)
    ds= dataset.Dataset(args.dataset).load_dicoms(args.dataset,
                                                  metabolites=quantifier.metabolites)
    store_in_data = True
  else:
    store_in_data = False
  # Export for quantification
  d_inp, d_out = ds.export(metabolites=quantifier.metabolites, norm=quantifier.norm,
                           acquisitions=quantifier.acquisitions, datatype=quantifier.datatype,
                           verbose=args.verbose)
  if np.sum(d_out.shape) == 0:
    # No concentrations, so for analysis of data and store results with data, not model
    store_in_data = True
  from mrsnet.analyse import analyse_model
  id_ref = sorted([a for a in ds.spectra[0].keys()])[0]
  if store_in_data:
    # Store results in data repository, as dicoms or no concentrations (so for actual quantification)
    analyse_model(quantifier, d_inp, d_out, args.dataset,
                  id=[s[id_ref].id for s in ds.spectra],
                  show_conc=True, save_conc=True, no_show=args.no_show,
                  verbose=args.verbose, prefix=str(quantifier).replace("/","_"),
                  image_dpi=Cfg.val['image_dpi'], screen_dpi=Cfg.val['screen_dpi'])
  else:
    # Store results with model, as concentrations and dataset is for testing model
    analyse_model(quantifier, d_inp, d_out, os.path.join(Cfg.val['path_model'], name, batch_size,
                                                         epochs, train_model, trainer, rest),
                  id=[s[id_ref].id for s in ds.spectra],
                  show_conc=True, save_conc=True, no_show=args.no_show,
                  verbose=args.verbose, prefix=ds.name.replace("/","_"),
                  image_dpi=Cfg.val['image_dpi'], screen_dpi=Cfg.val['screen_dpi'])

def benchmark(args):
  # Benchmark sub-command
  if args.verbose > 0:
    print("# Loading model %s" % args.model)
  id = get_std_name(args.model)
  name = []
  for k in range(0,len(id)):
    if id[k][0:4] == 'cnn_':
      # FIXME: index issues!
      name = os.path.join(*id[k:k+6])
      batch_size = id[k+6]
      epochs = id[k+7]
      train_model = id[k+8]
      trainer = id[k+9]
      rest = id[k+10] if len(id) > k+10 else '' # Folds
      print(k,id,name,train_model,batch_size)
      break
  if len(name) == 0:
    raise Exception("Cannot get model name from model argument")
  if args.verbose > 0:
    print("# Loading model %s : %s : %s : %s : %s : %s" % (name,batch_size,epochs,train_model,trainer,rest))
  if name[0:4] == "cnn_":
    from mrsnet.model import CNN
    quantifier = CNN.load(os.path.join(Cfg.val['path_model'], name, batch_size, epochs, train_model, trainer, rest))
  else:
    raise Exception("Unknown model "+name)
  import mrsnet.dataset as dataset
  for id in ['E1', 'E2', 'E3', 'E4a', 'E4b', 'E4c', 'E4d']:
    if args.verbose > 0:
      print("# Loading Benchmark %s" % id)
    bm = dataset.Dataset(id).load_dicoms(os.path.join(Cfg.val['path_benchmark'], id, 'MEGA_Combi_WS_ON'),
                                         concentrations=os.path.join(Cfg.val['path_benchmark'],
                                                                     id, 'concentrations.json'),
                                         metabolites=quantifier.metabolites)
    if args.show_spectra:
      for s,c in zip(bm.spectra,bm.concentrations):
        for a in s.keys():
          s[a].plot_spectrum(c,screen_dpi=Cfg.val['screen_dpi'])
          plt.show(block=True)
          plt.close()
    d_inp, d_out = bm.export(metabolites=quantifier.metabolites, norm=quantifier.norm,
                             acquisitions=quantifier.acquisitions, datatype=quantifier.datatype,
                             verbose=args.verbose)
    from mrsnet.analyse import analyse_model
    id_ref = sorted([a for a in bm.spectra[0].keys()])[0]
    analyse_model(quantifier, d_inp, d_out, os.path.join(Cfg.val['path_model'], name, batch_size, epochs,
                                                         train_model, trainer, rest),
                  id=[s[id_ref].id for s in bm.spectra],
                  show_conc=True, save_conc=True, no_show=args.no_show,
                  verbose=args.verbose, prefix=id, image_dpi=Cfg.val['image_dpi'],
                  screen_dpi=Cfg.val['screen_dpi'])

def get_std_name(name):
  _, path = os.path.splitdrive(name)
  id = []
  while True:
    path, folder = os.path.split(path)
    if folder != "":
      id.append(folder)
    if path == "":
      break
  id.reverse()
  return id

class Cfg:
  # FIXME: PATHS!
  # Default configuration - do not overwrite here but set alternatives in file
  val = {
    'path_basis': os.path.expanduser('~/.local/share/mrsnet/basis'),
    'path_simulation': os.path.expanduser('~/.local/share/mrsnet/sim-spectra'),
    'path_model': os.path.expanduser('~/.local/share/mrsnet/model'),
    'path_benchmark': os.path.expanduser('~/.local/share/mrsnet/benchmark'),
    'figsize': (26.67,15.0),
    'default_screen_dpi': 96,
    'image_dpi': [300]
  }
  file = os.path.expanduser("~/.config/mrsnet.json")

  @staticmethod
  def init():
    if os.path.isfile('mrsnet.py') and os.path.isdir('data') and os.path.isdir('.git'):
      # Dev mode
      Cfg.val['path_basis'] = os.path.join('.','data','basis')
      Cfg.val['path_simulation'] = os.path.join('.','data','sim-spectra')
      Cfg.val['path_model'] = os.path.join('.','data','model')
      Cfg.val['path_benchmark'] = os.path.join('.','data','benchmark')
    # Load cfg file
    if os.path.isfile(Cfg.file):
      import json
      with open(Cfg.file, "r") as fp:
        js = json.load(fp)
        for k in js.keys():
          if k in Cfg.val:
            Cfg.val[k] = js[k]
          else:
            raise Exception("Unknown config file entry %s in %s" % (k,Cfg.file))
    # Setup defaults
    Cfg.val["screen_dpi"] = Cfg._screen_dpi()
    plt.rcParams["figure.figsize"] = Cfg.val['figsize']

  @staticmethod
  def _screen_dpi():
    # DPI for plots on screen
    try:
      from screeninfo import get_monitors
    except ModuleNotFoundError:
      return Cfg.val['default_screen_dpi']
    try:
      m = get_monitors()[0]
    except:
      return Cfg.val['default_screen_dpi']
    from math import hypot
    return hypot(m.width, m.height) / hypot(m.width_mm, m.height_mm) * 25.4

if __name__ == '__main__':
  # Only print warnings and errors for tf (set before importing tf)
  if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  # Headless mode
  if not "DISPLAY" in os.environ:
    from matplotlib import use
    use("Agg")
  Cfg.init()
  main()
