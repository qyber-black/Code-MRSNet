# mrsnet/selection.py - MRSNet - model selection
#
# SPDX-FileCopyrightText: Copyright (C) 2020-2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2022-2024 Zien Ma, PhD student at Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Model selection and hyperparameter optimization for MRSNet.

This module provides various model selection strategies including grid search,
Quasi-Monte Carlo (QMC), Gaussian Process Optimization (GPO), and Genetic
Algorithms (GA) for finding optimal hyperparameters.
"""

import csv
import json
import os
import random
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
import sobol_seq

from mrsnet.cfg import Cfg


class Select:
  """Main model selection class for hyperparameter optimization.

  This class provides a unified interface for various model selection strategies
  including grid search, QMC, GPO, and GA optimization methods.

  Attributes
  ----------
      metabolites (list): List of metabolite names
      dataset (str): Path to dataset
      epochs (int): Number of training epochs
      validate (str): Validation strategy
      screen_dpi (int): DPI for screen display
      image_dpi (list): DPI for saved images
      verbose (int): Verbosity level
      dataset_name (str): Name of the dataset
      pulse_sequence (str): Pulse sequence type
      error_type (str): Type of error to use for model selection (concentration, spectra)
      remote (str, optional): Remote execution configuration
      remote_user (str, optional): Remote username
      remote_tasks (int, optional): Number of remote tasks
      remote_wait (int, optional): Remote wait time
  """

  def __init__(self,remote,metabolites,dataset,epochs,validate,screen_dpi,image_dpi,search_model,verbose):
    """Initialize model selection.

    Parameters
    ----------
        remote (str): Remote execution configuration (format: "host:user:tasks:wait")
        metabolites (list): List of metabolite names
        dataset (str): Path to dataset
        epochs (int): Number of training epochs
        validate (str): Validation strategy
        screen_dpi (int): DPI for screen display
        image_dpi (list): DPI for saved images
        search_model (list): List of search model paths
        verbose (int): Verbosity level
    """
    from mrsnet.dataset import Dataset
    if os.path.isfile(os.path.join(dataset,"spectra_noisy.joblib")) or \
       os.path.isfile(os.path.join(dataset,"spectra_clean.joblib")):
      idl = _get_std_name(dataset)
      name = os.path.join(*idl[-9:-1])
      self.ds_rest = idl[-1]
      if verbose > 0:
        print(f"# Loading dataset {name} : {self.ds_rest}")
      ds = Dataset.load(dataset, info_only=True)
      self.pulse_sequence = ds.pulse_sequence
    else:
      raise RuntimeError("Cannot find dataset")
    # Search paths: path_model first, then search_model paths
    self.search_model = search_model
    self.metabolites = metabolites
    self.dataset = dataset
    self.epochs = epochs
    self.validate = validate
    self.screen_dpi = screen_dpi
    self.image_dpi = image_dpi
    self.verbose = verbose
    self.dataset_name = ds.name
    self.error_type = None
    if len(remote) > 0:
      remote = remote.split(":")
      self.remote = remote[0]
      self.remote_user = remote[1]
      self.remote_tasks = int(remote[2]) if len(remote) > 2 else 10
      self.remote_wait = int(remote[3]) if len(remote) > 3 else 15
      if self.verbose > 0:
        print("# Remote Sync")
      cmd = ['/usr/bin/env', 'bash', self.remote, self.remote_user, "sync", self.dataset]
      p = subprocess.run(cmd, capture_output=True)  # noqa: S603
      output = p.stdout.decode("utf-8").split("\n")
      if self.verbose > 0:
        for l in output:
          print(l)
    else:
      self.remote = 'local'
      self.remote_user = None
      self.remote_tasks = None

    self.tasks = []
    self.key_vals = []
    self.val_performance = []
    self.train_performance = []

  def _find_existing_model(self, model_name, args, train_model, trainer, fold, path_model):
    """Find existing model across path_model and search_model paths.

    Parameters
    ----------
        model_name (str): Name of the model
        args (dict): Model arguments including batchsize
        train_model (str): Training model identifier
        trainer (str): Trainer type
        fold (str): Fold identifier
        path_model (str): Base path for model storage

    Returns
    -------
        model_path if found, None if not found
    """
    for search_path in [path_model, *self.search_model]:
      base_path = os.path.join(search_path, model_name, args['batchsize'], str(self.epochs), train_model)

      if os.path.isdir(base_path):
        # Find model folder, assuming there may be multiple repeats.
        # We assume the last valid repeat is the best option/latest result.
        selected_id = -1
        for fn in os.listdir(base_path):
          ffn = os.path.join(base_path, fn)
          if os.path.isdir(ffn):
            file_s = fn.split("-")
            repeat_id = -1
            if len(file_s) > 1:
              try:
                repeat_id = int(file_s[-1])
              except Exception:  # noqa: S110
                pass
            if (repeat_id > 0 and
                fn == trainer+"-"+str(repeat_id) and
                os.path.exists(os.path.join(base_path, fn, fold, f"train_{self.error_type}_errors.json")) and
                os.path.exists(os.path.join(base_path, fn, fold, f"validation_{self.error_type}_errors.json"))):
              if repeat_id > selected_id:
                selected_id = repeat_id
            else:
              if self.verbose > 0:
                print(f"# WARNING: {ffn} - broken/incomplete model")
          else:
            if self.verbose > 0:
              print(f"# WARNING: {ffn} - this file should not be there")

        if selected_id > 0:
          model_path = os.path.join(base_path, trainer+"-"+str(selected_id))
          if self.verbose > 4:
            print(f"# Found existing model at: {model_path}")
          return model_path

    return None

  def _add_task(self,key_vals,path_model):
    """Add a new optimization task.

    Parameters
    ----------
        key_vals (dict): Dictionary of parameter values for the task
        path_model (str): Base path for model storage
    """
    # Add new task given by key_vals arguments and model storage path base path_model

    # Convert arguments, interpret model string
    args = key_vals.copy()
    if args['model'] == 'cnn':
      # cnn_[S1]_[S2]_[C1]_[C2]_[C3]_[C4]_[O1]_[O2]_[F1]_[F2]_[D]_[softmax,sigmoid]
      model_str = 'cnn'
      self.error_type = 'concentration'
      for marg in ["S1", "S2", "C1", "C2", "C3", "C4", "O1", "O2", "F1", "F2", "D", "ACTIVATION"]:
        model_str += "_" + str(args['model_'+marg])
        del args['model_'+marg]
      args['model'] = model_str
      from mrsnet.cnn import CNN
      model_name = str(CNN(model_str, self.metabolites, self.pulse_sequence,
                           args['acquisitions'], args['datatype'], args['norm']))
    elif args['model'][0:4] == 'cnn_':
      model_str = args['model']
      self.error_type = 'concentration'
      from mrsnet.cnn import CNN
      model_name = str(CNN(model_str, self.metabolites, self.pulse_sequence,
                           args['acquisitions'], args['datatype'], args['norm']))
    elif args['model'][0:5] == 'ae_fc':
      # AE-FC model fully parameterised
      # ae_fc_[LIN]_[LOUT]_[ACT]_[ACT-LAST]_[DO]
      model_str = 'ae_fc'
      self.error_type = 'spectra'
      for marg in ["LIN", "LOUT", "ACT", "ACT-LAST", "DO"]:
        model_str += "_"
        model_str += str(args['model_' + marg])
        del args['model_' + marg]
      args['model'] = model_str
      from mrsnet.autoencoder import Autoencoder
      model_name = str(Autoencoder(model_str,self.metabolites, self.pulse_sequence,
                                   args['acquisitions'], args['datatype'], args['norm']))
    elif args['model'][0:6] == 'aeq_fc':
      # AEQ-FC model fully parameterised
      # aeq_fc_[UNITS]_[LAYERS]_[ACT]_[ACT-LAST]_[DO]
      model_str = 'aeq_fc'
      self.error_type = 'spectra'
      for marg in ["UNITS", "LAYERS", "ACT", "ACT-LAST", "DO"]:
        model_str += "_" + str(args['model_' + marg])
        del args['model_' + marg]
      args['model'] = model_str
      from mrsnet.autoencoder import Autoencoder
      model_name = str(Autoencoder(model_str,self.metabolites, self.pulse_sequence,
                                   args['acquisitions'], args['datatype'], args['norm']))
    elif args['model'][0:7] == 'caeq_fc':
      # CAEQ-FC model fully parameterised
      # caeq_fc_[LIN]_[LOUT]_[ACT]_[ACT-LAST]_[DO]_[UNITS]_[LAYERS]_[ACTIVATION]_[ACTIVATION-LAST]_[DP]
      model_str = 'caeq_fc'
      self.error_type = 'concentration'
      for marg in ["LIN", "LOUT", "ACT", "ACT-LAST", "DO", "UNITS", "LAYERS", "ACTIVATION", "ACTIVATION-LAST", "DP"]:
        model_str += "_" + str(args['model_' + marg])
        del args['model_' + marg]
      args['model'] = model_str
      from mrsnet.ae_quantifier import AutoencoderQuantifier
      model_name = str(AutoencoderQuantifier(model_str,self.metabolites, self.pulse_sequence,
                                             args['acquisitions'], args['datatype'], args['norm']))

    else:
      raise RuntimeError(f"Unknown model string {args['model']}")

    # Make all arguments strings
    for a in args:
      if isinstance(args[a],list):
        args[a] = [str(v) for v in args[a]]
      else:
        args[a] = str(args[a])
    # Same for key_vals (can be different due to model parameters, but used for
    # plots and must be all strings for sorting, etc)
    for kv in key_vals:
      if isinstance(key_vals[kv],list):
        key_vals[kv] = [str(v) for v in key_vals[kv]]
      else:
        key_vals[kv] = str(key_vals[kv])

    # Find model_path and fold for model
    train_model = self.dataset_name.replace("/","_")+"_"+self.ds_rest
    fold=""
    if self.validate > 1.0:
      trainer = "KFold_"+str(int(self.validate))
      fold = "fold-0"
    elif self.validate < -1.0:
      trainer = "DuplexKFold_"+str(int(-self.validate))
      fold = "fold-0"
    elif self.validate > 0.0:
      trainer = "Split_"+str(self.validate)
    elif self.validate < 0.0:
      trainer = "DuplexSplit_"+str(-self.validate)
    elif self.validate == 0.0:
      trainer = "NoValidation"
    else:
      raise RuntimeError(f"Unknown validation {args.validate}")

    # Check for existing model across path_model and search_model paths
    model_path = self._find_existing_model(model_name, args, train_model, trainer, fold, path_model)
    if model_path is None:
      # No existing model found, create new one
      base_path = os.path.join(path_model, model_name, args['batchsize'], str(self.epochs), train_model)
      model_path = os.path.join(base_path, trainer+"-1")

    # Create task
    self.tasks.append({
        'model_path': model_path,
        'fold': fold,
        'args': args,
        'key_vals': key_vals.copy()
      })

  def _run_tasks(self, load_only=False):
    """Run all tasks in the selection process.

    Parameters
    ----------
        load_only (bool, optional): If True, only load existing results without running new tasks. Defaults to False.
    """
    counter = 1
    remote_run = []
    for t in self.tasks:
      global ga_aux
      if not load_only and self.verbose > 0:
        print(f"# Task {counter} / {len(self.tasks)}")
      val_p = None
      if (os.path.exists(os.path.join(t['model_path'],t['fold'],"model.keras")) or
          (os.path.exists(os.path.join(t['model_path'],t['fold'],f"train_{self.error_type}_errors.json")) and
           os.path.exists(os.path.join(t['model_path'],t['fold'],f"validation_{self.error_type}_errors.json")))): # Can use old results here as we only need the evaluation data
        if self.verbose > 0:
          print(f"Exists {t['model_path']}:{t['fold']}")
        val_p, train_p = self._load_performance(t['model_path'], t['fold'])
        if val_p is not None:
          self.key_vals.append(t['key_vals'])
          self.val_performance.append(val_p)
          self.train_performance.append(train_p)
      if val_p is None and not load_only:
        # Train
        if self.remote == 'local':
          self._run(t['args'])
          val_p, train_p = self._load_performance(t['model_path'], t['fold'])
          self.key_vals.append(t['key_vals'])
          if val_p is not None:
            self.val_performance.append(val_p)
            self.train_performance.append(train_p)
          else:
            self.val_performance.append([999999.0]*len(self.val_performance[0])) # Let's hope the first one does not fail
            self.train_performance.append([999999.0]*len(self.val_performance[0]))
            print("**Error:** Local job failed: "+str(t))
        else:
          remote_run.append(['wait', t['args'], t['model_path'], t['fold'], t['key_vals']])
      counter += 1
    if len(remote_run) > 0:
      if self.verbose > 0:
        print("# Running remotely")
      # Remote tasks scheduling
      all_done = False
      while not all_done:
        all_done = True
        for k in range(0,len(remote_run)):
          if remote_run[k][0] != 'complete':
            print(f"## Job {k+1} / {len(remote_run)}")
            status = self._run_remote(k,remote_run)
            if status == 'done':
              val_p, train_p = self._load_performance(remote_run[k][2], remote_run[k][3])
              self.key_vals.append(remote_run[k][4])
              if val_p is not None:
                self.val_performance.append(val_p)
                self.train_performance.append(train_p)
              else:
                self.val_performance.append([999999.0]*len(self.val_performance[0])) # Let's hope the first one does not fail
                self.train_performance.append([999999.0]*len(self.val_performance[0]))
                print("**Error:** Remote job failed: "+str(remote_run[k]))
              remote_run[k][0] = 'complete'
            else:
              all_done = False
        running = len([l for l in remote_run if l[0] == 'run'])
        waiting = len([l for l in remote_run if l[0] == 'wait'])
        if self.verbose > 0:
          print(f"  {running} running; {waiting} waiting")
        if running >= self.remote_tasks or (waiting < 1 and running > 0):
          time.sleep(self.remote_wait*60)

    self.tasks = []

  def _run(self,args):
    """Run a single training task.

    Parameters
    ----------
        args: Task arguments containing model parameters
    """
    cmd = ['/usr/bin/env', 'python3', 'mrsnet.py', 'train',
           '--metabolites', *list(self.metabolites),
           '--dataset', self.dataset,
           '--epochs', str(self.epochs),
           '--validate', str(self.validate)]
    for a in args:
      cmd.append("--"+a)
      if isinstance(args[a], list):
        for l in range(0,len(args[a])):
          if isinstance(args[a][l],str):
            cmd.append(args[a][l])
          else:
            cmd.append(str(args[a][l]))
      elif isinstance(args[a], str):
        cmd.append(args[a])
      else:
        cmd.append(str(args[a]))
    if self.verbose > 0:
      cmd += ['-v']*self.verbose
      print('# Run '+' '.join(cmd[3:]))
    try:
      p = subprocess.Popen(cmd)  # noqa: S603
    except OSError as e:
      raise RuntimeError('MRSNet training failed') from e
    p.wait()

  def _run_remote(self,id,all):
    """Run a task remotely via SSH.

    Parameters
    ----------
        id: Task identifier
        all: List of all tasks with their status and commands
    """
    cmd = ['/usr/bin/env', 'bash', self.remote, self.remote_user, "X", all[id][2]]

    if all[id][0] == 'run':
      cmd[4] = "check"
      p = subprocess.run(cmd, capture_output=True)  # noqa: S603
      output = p.stdout.decode("utf-8").split("\n")
      if self.verbose > 0:
        for l in output[:-2]:
          print(l)
      status = output[-2]
      if status[:4] == "DONE":
        all[id][0] = 'done'
      elif status[:4] != "WAIT":
        all[id][0] = 'wait'
    elif all[id][0] == 'wait':
      if len([l for l in all if l[0] == 'run']) < self.remote_tasks:
        cmd[4] = "run"
        cmd += ['--metabolites', *list(self.metabolites),
                '--dataset', self.dataset,
                '--epochs', str(self.epochs),
                '--validate', str(self.validate)]
        for a in all[id][1]:
          cmd.append("--"+a)
          if isinstance(all[id][1][a], list):
            for l in range(0,len(all[id][1][a])):
              if isinstance(all[id][1][a][l],str):
                cmd.append(all[id][1][a][l])
              else:
                cmd.append(str(all[id][1][a][l]))
          elif isinstance(all[id][1][a], str):
            cmd.append(all[id][1][a])
          else:
            cmd.append(str(all[id][1][a]))
        p = subprocess.run(cmd, capture_output=True)  # noqa: S603
        output = p.stdout.decode("utf-8").split("\n")
        if self.verbose > 0:
          for l in output[:-2]:
            print(l)
        all[id][0] = 'run'
    return all[id][0]

  def _load_performance(self, model_path, fold):
    """Load comprehensive performance metrics from a trained model.

    Parameters
    ----------
        model_path (str): Path to the trained model
        fold (str): Fold identifier for cross-validation

    Returns
    -------
        tuple: (validation_performance, training_performance) lists of MAE values
    """
    try:
      if len(fold) == 0:
        # Single fold case
        with open(os.path.join(model_path, f"train_{self.error_type}_errors.json")) as f:
          train_data = json.load(f)
        with open(os.path.join(model_path, f"validation_{self.error_type}_errors.json")) as f:
          val_data = json.load(f)

        # Extract comprehensive metrics
        train_p = [train_data['total']['abserror']['mean']]  # total MAE
        val_p = [val_data['total']['abserror']['mean']]      # total MAE

        # FIXME: Store additional metrics for potential future use, for single and cross-validation below:
        #        Currently we only use MAE, but we could extend this to include other metrics like R^2, std (stability: 1/(std + 1e-6)), generalization (1/(|train_mae-val_mae| + 1e-6)) etc. based on the JSON error file data.

      else:
        # Cross-validation case
        train_p = []
        val_p = []
        f_cnt = 0
        while os.path.exists(os.path.join(model_path,"fold-"+str(f_cnt))):
          with open(os.path.join(model_path,"fold-"+str(f_cnt),f"train_{self.error_type}_errors.json")) as f:
            train_data = json.load(f)
          with open(os.path.join(model_path,"fold-"+str(f_cnt),f"validation_{self.error_type}_errors.json")) as f:
            val_data = json.load(f)

          train_p.append(train_data['total']['abserror']['mean'])  # total MAE
          val_p.append(val_data['total']['abserror']['mean'])      # total MAE
          f_cnt += 1

    except Exception as e:
      if self.verbose > 0:
        print(f"# WARNING: {model_path} - model broken")
        print(e)
      return None, None
    return val_p, train_p

  def _save_performance(self, collection_name, var_keys, fix_keys):
    """Save comprehensive model selection performance results to CSV file.

    Parameters
    ----------
        collection_name (str): Name of the collection being optimized
        var_keys (list): List of variable parameter keys
        fix_keys (list): List of fixed parameter keys
    """
    var_keys.sort()
    fix_keys.sort()

    # Results folder
    basename = os.path.basename(collection_name).replace(".json","")
    from mrsnet.getfolder import get_folder
    folder = get_folder(self.dataset,basename+"-"+str(self.epochs)+"-"+str(self.validate)+"-%s")
    os.makedirs(folder, exist_ok=True)

    # Calculate comprehensive performance metrics
    n_models = len(self.val_performance)
    n_folds = len(self.val_performance[0]) if n_models > 0 else 0

    # Calculate metrics for each model
    model_metrics = []
    for i in range(n_models):
      val_perf = np.array(self.val_performance[i])
      train_perf = np.array(self.train_performance[i])

      # Basic statistics
      val_mean = np.mean(val_perf)
      val_std = np.std(val_perf)
      val_min = np.min(val_perf)
      val_max = np.max(val_perf)
      val_median = np.median(val_perf)

      train_mean = np.mean(train_perf)
      train_std = np.std(train_perf)

      # Overfitting metrics
      overfitting_gap = train_mean - val_mean  # Positive = overfitting
      generalization_ratio = val_mean / train_mean if train_mean > 0 else float('inf')

      # Stability metrics
      val_cv = val_std / val_mean if val_mean > 0 else float('inf')  # Coefficient of variation
      fold_range = val_max - val_min

      # Robustness metrics
      val_iqr = np.percentile(val_perf, 75) - np.percentile(val_perf, 25)

      model_metrics.append({
        'iteration': i,
        'val_mean': val_mean,
        'val_std': val_std,
        'val_min': val_min,
        'val_max': val_max,
        'val_median': val_median,
        'val_cv': val_cv,
        'val_iqr': val_iqr,
        'fold_range': fold_range,
        'train_mean': train_mean,
        'train_std': train_std,
        'overfitting_gap': overfitting_gap,
        'generalization_ratio': generalization_ratio,
        'val_folds': val_perf.tolist(),
        'train_folds': train_perf.tolist()
      })

    # Sort by validation mean (best first)
    sorted_models = sorted(model_metrics, key=lambda x: x['val_mean'])

    # Assign ranks
    for rank, model in enumerate(sorted_models):
      model['rank'] = rank + 1

    # Create comprehensive CSV with iteration order preserved
    with open(os.path.join(folder,"model_performance.csv"), "w") as f:
      writer = csv.writer(f, delimiter=",")

      # Header information
      writer.writerow([f"Results from {self.__class__.__name__}"])
      writer.writerow([])
      writer.writerow(["Dataset", self.dataset])
      writer.writerow(["Epochs", self.epochs])
      writer.writerow(["Validate", self.validate])
      writer.writerow(["Error Type", self.error_type])
      writer.writerow([])

      # Fixed parameters
      writer.writerow(["Fixed Parameters", *fix_keys])
      row=[""]
      for k in fix_keys:
        if isinstance(self.key_vals[0][k],list):
          self.key_vals[0][k] = "-".join(self.key_vals[0][k])
        row.append(self.key_vals[0][k])
      writer.writerow(row)
      writer.writerow([])

      # Column headers
      header = (["Iteration", "Rank"] +
                var_keys +
                ["Val_Mean", "Val_Std", "Val_Min", "Val_Max", "Val_Median", "Val_CV", "Val_IQR", "Fold_Range"] +
                ["Train_Mean", "Train_Std", "Overfitting_Gap", "Generalization_Ratio"] +
                [f"Val_Fold_{i}" for i in range(n_folds)] +
                [f"Train_Fold_{i}" for i in range(n_folds)])
      writer.writerow(header)

      # Data rows (preserving iteration order)
      for i in range(n_models):
        model = model_metrics[i]  # Use original iteration order
        row = ([model['iteration'], model['rank']] +
               [self.key_vals[i][k] if not isinstance(self.key_vals[i][k], list)
                else "-".join(self.key_vals[i][k]) for k in var_keys] +
               [model['val_mean'], model['val_std'], model['val_min'], model['val_max'],
                model['val_median'], model['val_cv'], model['val_iqr'], model['fold_range']] +
               [model['train_mean'], model['train_std'], model['overfitting_gap'],
                model['generalization_ratio']] +
               model['val_folds'] + model['train_folds'])
        writer.writerow(row)
    # # E.g. after model selection analyze the CSV:
    # import pandas as pd
    # df = pd.read_csv('model_performance.csv')
    # # Find most stable models (low coefficient of variation)
    # stable_models = df.nsmallest(10, 'Val_CV')
    # # Find models with least overfitting
    # generalizable_models = df.nsmallest(10, 'Overfitting_Gap')
    # # Compare top 5 models across all metrics
    # top_models = df.nsmallest(5, 'Val_Mean')[['Iteration', 'Rank', 'Val_Mean', 'Val_CV', 'Overfitting_Gap']]

    # Create summary statistics file
    with open(os.path.join(folder,"model_summary.json"), "w") as f:
      summary = {
        'total_models': n_models,
        'n_folds': n_folds,
        'best_model': {
          'iteration': sorted_models[0]['iteration'],
          'rank': 1,
          'val_mean': sorted_models[0]['val_mean'],
          'val_std': sorted_models[0]['val_std'],
          'overfitting_gap': sorted_models[0]['overfitting_gap']
        },
        'performance_stats': {
          'val_mean_range': [min(m['val_mean'] for m in model_metrics),
                           max(m['val_mean'] for m in model_metrics)],
          'avg_overfitting_gap': np.mean([m['overfitting_gap'] for m in model_metrics]),
          'models_with_overfitting': len([m for m in model_metrics if m['overfitting_gap'] > 0.01]),
          'most_stable_model': min(model_metrics, key=lambda x: x['val_cv'])['iteration']
        }
      }
      json.dump(summary, f, indent=2)

    # Bar plot of validate/train performance (sorted by performance)
    fig, ax = plt.subplots()
    parameters = [":".join([str(self.key_vals[sorted_models[p]['iteration']][k]) for k in var_keys])
                  for p in range(0,len(sorted_models))]
    val_error = [sorted_models[p]['val_mean'] for p in range(0,len(sorted_models))]
    train_error = [sorted_models[p]['train_mean'] for p in range(0,len(sorted_models))]
    top_n = np.min([len(sorted_models),50])
    top_n = np.max(np.argwhere(np.array(val_error[:top_n]) < 999999.0)) # Don't plot failed
    Y = np.arange(2*len(val_error),1,-2)  # noqa: N806
    ax.barh(y=Y[:top_n], width=val_error[:top_n], height=0.9, left=0, align='center',
            label="Val. Error", color="#4878D0", zorder=1)
    ax.barh(y=Y[:top_n]-0.75, width=train_error[:top_n], height=0.6, left=0, align='center',
            label="Train Error", color="#A1C9F4", zorder=0)
    ax.scatter(y=[Y[:top_n]]*len(self.val_performance[0]),
               x=[sorted_models[p]['val_folds'][q]
                  for q in range(0,len(self.val_performance[0])) for p in range(0,top_n)],
               color="#000000", s=5.0, zorder=2)
    ax.scatter(y=[Y[:top_n]-0.75]*len(self.train_performance[0]),
               x=[sorted_models[p]['train_folds'][q]
                  for q in range(0,len(self.train_performance[0])) for p in range(0,top_n)],
               color="#000000", s=5.0, zorder=2)
    plt.yticks(Y[:top_n], parameters[:top_n])
    ax.legend(loc="upper right", frameon=True)
    ax.set(xlim=(0,np.max([np.max([sorted_models[p]['val_folds'] for p in range(0,top_n)]),
                           np.max([sorted_models[p]['train_folds'] for p in range(0,top_n)])])),
           ylabel="", xlabel="Mean Absolute Concentration Error")
    fig.tight_layout()
    for dpi in self.image_dpi:
      plt.savefig(os.path.join(folder,"errors@"+str(dpi)+".png"), dpi=dpi)
    if self.verbose > 1:
      fig.set_dpi(self.screen_dpi)
      plt.show(block=True)
    plt.close()
    # Plot distributions across single-parameter groups
    x_max=1
    for group_id in var_keys:
      m = np.max(len(set([str(self.key_vals[p][group_id]) for p in range(0,len(self.val_performance))])))  # noqa: C403
      if m > x_max:
        x_max = m
    X = np.arange(1,x_max+1)  # noqa: N806
    fig, ax = plt.subplots(len(var_keys),1)
    if len(var_keys) == 1:
      ax = [ax]
    for k in range(0,len(var_keys)):
      group_id = var_keys[k]
      key_vals = sorted(list(set([str(self.key_vals[p][group_id])  # noqa: C403, C414
                                  for p in range(0,len(self.val_performance))])))
      key_vals = [str(v) for v in key_vals]
      for ki in range(0,len(key_vals)):
        val_per = [sorted_models[p]['val_mean']
                    for p in range(0,len(sorted_models))
                      if key_vals[ki] == str(self.key_vals[sorted_models[p]['iteration']][group_id])]
        train_per = [sorted_models[p]['train_mean']
                     for p in range(0,len(sorted_models))
                       if key_vals[ki] == str(self.key_vals[sorted_models[p]['iteration']][group_id])]
        offset=0.1
        bp=ax[k].boxplot(x=val_per,positions=[X[ki]-offset],patch_artist=True,labels=["Val. Err"],zorder=0)
        bp['boxes'][0].set_facecolor("#4878D0")
        ax[k].scatter(x=[X[ki]-offset]*len(val_per), y=val_per, color="#000000", s=5.0, zorder=2)
        bp=ax[k].boxplot(x=train_per,positions=[X[ki]+offset],patch_artist=True,labels=["Train Err"],zorder=1)
        bp['boxes'][0].set_facecolor("#A1C9F4")
        ax[k].scatter(x=[X[ki]+offset]*len(train_per), y=train_per, color="#000000", s=5.0, zorder=2)
      ax[k].set_xlim(X[0]-1,X[-1]+1)
      ax[k].set_ylabel("WDE")
      ax[k].set_xticks(X)
      ax[k].set_xticklabels([group_id+": "+v+" (val.,train)" for v in key_vals]+[""]*(x_max-len(key_vals)))
    fig.tight_layout()
    for dpi in self.image_dpi:
      plt.savefig(os.path.join(folder,"error-dists@"+str(dpi)+".png"), dpi=dpi)
    if self.verbose > 1:
      fig.set_dpi(self.screen_dpi)
      plt.show(block=True)
    plt.close()
    return folder

class SelectGrid(Select):
  """Grid search model selection.

  Performs exhaustive grid search over all parameter combinations
  to find the optimal hyperparameters.
  """

  def __init__(self,metabolites,dataset,epochs,validate,remote,screen_dpi,image_dpi,search_model,verbose):
    """Initialize grid search selector.

    Parameters
    ----------
        metabolites (list): List of metabolite names
        dataset (str): Path to dataset
        epochs (int): Number of training epochs
        validate (str): Validation strategy
        remote (str): Remote execution configuration
        screen_dpi (int): DPI for screen display
        image_dpi (list): DPI for saved images
        verbose (int): Verbosity level
    """
    super().__init__(remote,metabolites,dataset,epochs,validate,screen_dpi,image_dpi,search_model,verbose)

  def optimise(self, collection_name, models, path_model):
    """Perform grid search optimization.

    Parameters
    ----------
        collection_name (str): Name of the collection
        models (Grid): Grid of model parameters to search
        path_model (str): Path to save model results

    Returns
    -------
        str: Path to results folder
    """
    if self.verbose > 0:
      print("# Grid Model Selection")
    keys = list(models.values.keys())
    var_keys = [k for k in keys if len(models.values[k]) > 1]
    fix_keys = [k for k in keys if len(models.values[k]) == 1]
    total = np.prod([len(models.values[k]) for k in keys])
    counter = 1
    for model in models:
      if self.verbose > 0:
        print(f"# Model {counter} / {total}")
      key_vals = {}
      for l in range(0,len(keys)):
        key_vals[keys[l]] = model[l]
      self._add_task(key_vals, path_model)
      counter += 1
    self._run_tasks()
    # Store performance info
    self._save_performance(collection_name+"-grid", var_keys, fix_keys)

class SelectQMC(Select):
  """Quasi-Monte Carlo model selection.

  Uses Sobol sequences for efficient sampling of the parameter space
  to find optimal hyperparameters.
  """

  def __init__(self,metabolites,dataset,epochs,validate,repeats,remote,screen_dpi,image_dpi,search_model,verbose):
    """Initialize QMC selector.

    Parameters
    ----------
        metabolites (list): List of metabolite names
        dataset (str): Path to dataset
        epochs (int): Number of training epochs
        validate (str): Validation strategy
        repeats (int): Number of QMC samples
        remote (str): Remote execution configuration
        screen_dpi (int): DPI for screen display
        image_dpi (list): DPI for saved images
        verbose (int): Verbosity level
    """
    super().__init__(remote,metabolites,dataset,epochs,validate,screen_dpi,image_dpi,search_model,verbose)
    self.repeats = repeats

  def optimise(self, collection_name, models, path_model):
    """Perform QMC optimization.

    Parameters
    ----------
        collection_name (str): Name of the collection
        models (Grid): Grid of model parameters to search
        path_model (str): Path to save model results

    Returns
    -------
        str: Path to results folder
    """
    if self.verbose > 0:
      print("# QMC Model Selection")
    keys = list(models.values.keys())
    var_keys = [k for k in keys if len(models.values[k]) > 1]
    fix_keys = [k for k in keys if len(models.values[k]) == 1]
    total = np.prod([len(models.values[k]) for k in keys])
    counter = 1
    # QMC sequence
    dim = len(var_keys)
    import math
    skip = math.floor(math.log(self.repeats*dim,2))
    select = sobol_seq.i4_sobol_generate(dim, self.repeats+skip)[skip:,:]
    while counter <= self.repeats:
      if self.verbose > 0:
        print(f"# Sample {counter} / {self.repeats} in space of size {total}")
      # Sample from parameter space
      key_vals = {}
      for k in keys:
        if k in fix_keys:
          key_vals[k] = models.values[k][0]
        else:
          sel = select[counter-1][var_keys.index(k)]
          n_para = len(models.values[k])
          key_vals[k] = models.values[k][np.floor(sel*n_para).astype(np.int64)]
      self._add_task(key_vals, path_model)
      counter += 1
    self._run_tasks()
    # Store performance info
    self._save_performance(collection_name+"-qmc", var_keys, fix_keys)

class SelectGPO(Select):
  """Gaussian Process Optimization model selection.

  Uses Gaussian Process Optimization (GPO) to efficiently search
  the parameter space and find optimal hyperparameters.
  """

  def __init__(self,metabolites,dataset,epochs,validate,repeats,remote,screen_dpi,image_dpi,search_model,verbose):
    """Initialize GPO selector.

    Parameters
    ----------
        metabolites (list): List of metabolite names
        dataset (str): Path to dataset
        epochs (int): Number of training epochs
        validate (str): Validation strategy
        repeats (int): Number of optimization iterations
        remote (str): Remote execution configuration
        screen_dpi (int): DPI for screen display
        image_dpi (list): DPI for saved images
        verbose (int): Verbosity level
    """
    super().__init__(remote,metabolites,dataset,epochs,validate,screen_dpi,image_dpi,search_model,verbose)
    self.repeats = repeats

  def optimise(self, collection_name, models, path_model):
    """Perform GPO optimization.

    Parameters
    ----------
        collection_name (str): Name of the collection
        models (Grid): Grid of model parameters to search
        path_model (str): Path to save model results

    Returns
    -------
        str: Path to results folder
    """
    if self.verbose > 0:
      print("# GPO Model Selection")
    keys = list(models.values.keys())
    var_keys = [k for k in keys if len(models.values[k]) > 1]
    fix_keys = [k for k in keys if len(models.values[k]) == 1]
    total = np.prod([len(models.values[k]) for k in keys])
    if self.verbose > 0:
      print(f"Search space size: {total}")
    # Prefer GPyOpt if available; otherwise fall back to Optuna (NumPy 2 compatible)
    use_gpyopt = False
    try:
      import GPyOpt as gpo  # noqa: N813
      use_gpyopt = True
    except Exception:
      gpo = None  # type: ignore
    domain = []
    self.values = {}
    for k in var_keys:
      if isinstance(models.values[k][0],int) or isinstance(models.values[k][0],float):
        dtype = 'discrete'
      else:
        dtype = 'categorical'
      domain.append({
          'name': k,
          'type': dtype,
          'domain': np.arange(0,len(models.values[k])),
          'dimensionality': 1
        })
    if self.verbose > 1:
      print("Domain: "+str(domain))

    # Initialise values from those we have
    if self.verbose > 0:
      print("# Init GPO")
    key_vals = {}
    for k in fix_keys:
      key_vals[k] = models.values[k][0]
    # Load Existing - can be disabled, so we can run repeatedly and use existing results
    if not Cfg.dev('selectgpo_optimise_noload'):
      if self.verbose > 0:
        print("# Loading existing samples")
      for model in models:
        for ki,k in enumerate(keys):
          if k in var_keys:
            key_vals[k] = models.values[k][models.values[k].index(model[ki])]
        self._add_task(key_vals, path_model)
      self._run_tasks(load_only=True)
      if self.verbose > 0:
        if len(self.key_vals) > 0:
          print(f"  Models loaded: {len(self.key_vals)} x {len(self.val_performance[0])}")
        else:
          print("  Models loaded: None")
    # Evaluate first, if none available so far
    if len(self.key_vals) < 1:
      for _ki,k in enumerate(keys):
        if k in var_keys:
          key_vals[k] = models.values[k][random.randrange(len(models.values[k]))]  # noqa: S311
      self._add_task(key_vals, path_model)
      self._run_tasks()
    # Convert eval. data to optimizer format (indices in [0..len(values)-1])
    Xdata = np.ndarray((0,len(var_keys)))  # noqa: N806
    Ydata = np.ndarray((0,1))  # noqa: N806
    res_n = len(self.val_performance[0])
    Xnext = np.ndarray((len(self.val_performance)*res_n,len(var_keys)))  # noqa: N806
    Ynext = np.ndarray((Xnext.shape[0],1))  # noqa: N806
    ll = 0
    for l in range(0,len(self.key_vals)):
      # Add results; multiple times if multiple evluations due to KFold validation, etc.
      for ri in range(0,res_n):
        for ki,k in enumerate(var_keys):
          if isinstance(models.values[k][0], list):
            if isinstance(models.values[k][0][0], int):
              Xnext[ll,ki] = models.values[k].index([int(v) for v in self.key_vals[l][k]])
            elif isinstance(models.values[k][0][0], float):
              Xnext[ll,ki] = models.values[k].index([float(v) for v in self.key_vals[l][k]])
            else:
              Xnext[ll,ki] = models.values[k].index(self.key_vals[l][k])
          elif isinstance(models.values[k][0], int):
            Xnext[ll,ki] = models.values[k].index(int(self.key_vals[l][k]))
          elif isinstance(models.values[k][0], float):
            Xnext[ll,ki] = models.values[k].index(float(self.key_vals[l][k]))
          else:
            Xnext[ll,ki] = models.values[k].index(self.key_vals[l][k])
        Ynext[ll,0] = self.val_performance[l][ri]
        ll += 1
    Xdata = np.vstack((Xdata,Xnext))  # noqa: N806
    Ydata = np.vstack((Ydata,Ynext))  # noqa: N806
    Perf_data_pos = len(self.val_performance)  # noqa: N806
    # Init GPO performance data
    remaining_samples = total - Xdata.shape[0] // res_n
    # Calculate average performance for each unique model (accounting for k-fold)
    unique_models = Xdata.shape[0] // res_n
    model_avg_performance = []
    for model_idx in range(unique_models):
      start_idx = model_idx * res_n
      end_idx = start_idx + res_n
      avg_perf = np.mean(Ydata[start_idx:end_idx, 0])
      model_avg_performance.append(avg_perf)

    best_model_idx = np.argmin(model_avg_performance)
    best_avg_performance = model_avg_performance[best_model_idx]
    Ybest = [best_avg_performance]  # noqa: N806
    XDiff = [0]  # noqa: N806
    XLast = Xdata[-1,:]  # noqa: N806
    best_data_idx = best_model_idx * res_n
    if self.verbose > 0:
      print(f"## Best: Model {best_model_idx} avg = {best_avg_performance:.8f} {Xdata[best_data_idx,:]!s}  of {unique_models} models")

    # If using Optuna, initialise study and inject existing observations
    if not use_gpyopt:
      try:
        import datetime

        import optuna
        from optuna.distributions import IntDistribution
        try:
          from optuna.samplers import GPSampler  # Gaussian Process sampler
          # GPSampler configuration to match GPyOpt behavior
          sampler = GPSampler(
              warm_starting_trial=None,  # No warm starting
              n_startup_trials=max(5, len(var_keys)),  # More startup trials like GPyOpt
              n_ei_candidates=200,  # More candidates for better acquisition optimization
              independent_sampler=None,  # Use full GP for all parameters
              warn_independent_sampling=True
            )
          # Note: Optuna GPSampler uses Expected Improvement (EI) by default
          # and optimizes acquisition function internally (similar to GPyOpt's L-BFGS)
          if self.verbose > 0:
            print("# Using Optuna GPSampler (Gaussian Process Bayesian Optimization)")
            print(f"#   Startup trials: {max(5, len(var_keys))}, EI candidates: 200")
            print("#   Acquisition: Expected Improvement (EI) with L-BFGS optimization")
        except Exception as gps_error:
          from optuna.samplers import TPESampler
          # Enhanced TPESampler as fallback
          sampler = TPESampler(
              multivariate=True,
              group=True,  # Enable multivariate modeling
              warn_independent_sampling=True,
              n_startup_trials=min(20, max(5, len(var_keys) * 2)),
              n_ei_candidates=100
            )
          if self.verbose > 0:
            print("# WARNING: Using Optuna TPESampler instead of GPSampler")
            print("#   TPESampler uses Tree-structured Parzen Estimator with multivariate modeling")
            print(f"#   Startup trials: {min(20, max(5, len(var_keys) * 2))}, EI candidates: 100")
            if "torch" in str(gps_error).lower():
              print("#   GPSampler requires PyTorch - install with: pip install torch")
      except Exception as e:
        raise RuntimeError("Optuna is required for SelectGPO with NumPy 2; please install optuna") from e
      # Minimization of validation error
      study = optuna.create_study(directions=["minimize"], sampler=sampler)
      distributions = {vk: IntDistribution(0, len(models.values[vk]) - 1) for vk in var_keys}
      # Seed the study with historical data (use average performance across folds)
      for model_idx in range(unique_models):
        start_idx = model_idx * res_n
        end_idx = start_idx + res_n
        avg_perf = np.mean(Ydata[start_idx:end_idx, 0])
        params = {vk: int(Xdata[start_idx, i]) for i, vk in enumerate(var_keys)}
        # Add historical trial to the study
        now = datetime.datetime.now()
        study.add_trial(optuna.trial.FrozenTrial(
          trial_id=model_idx,
          number=model_idx,
          state=optuna.trial.TrialState.COMPLETE,
          value=float(avg_perf),
          datetime_start=now,
          datetime_complete=now,
          params=params,
          distributions=distributions,
          user_attrs={},
          system_attrs={},
          intermediate_values={}
        ))

    # Optimisation iterations
    current_iter = len(self.key_vals)
    if self.remote == 'local':
      eval_per_step = 1
    else:
      eval_per_step = self.remote_tasks
    if Cfg.dev('selectgpo_no_search'):
      current_iter = self.repeats

    # Track evaluated parameter combinations to prevent duplicates
    # Use both set and numpy array for fast lookups
    evaluated_combinations = set()
    evaluated_arrays = []  # For fast numpy-based checking
    for model_idx in range(unique_models):
      start_idx = model_idx * res_n
      param_tuple = tuple(int(Xdata[start_idx, i]) for i in range(len(var_keys)))
      evaluated_combinations.add(param_tuple)
      evaluated_arrays.append(np.array(param_tuple))

    # Convert to numpy array for vectorized operations
    if evaluated_arrays:
      evaluated_matrix = np.array(evaluated_arrays)
    else:
      evaluated_matrix = np.array([]).reshape(0, len(var_keys))

    def is_duplicate_fast(sample_params):
      """Fast duplicate checking using numpy vectorized operations."""
      if evaluated_matrix.size == 0:
        return False
      sample_array = np.array(sample_params)
      return np.any(np.all(evaluated_matrix == sample_array, axis=1))

    # Original GPyOpt-like strategy: alternate between evaluator types
    # GPyOpt alternates between 'sequential', 'random', and 'thompson_sampling'
    startup_trials = max(5, len(var_keys))  # Match GPSampler startup trials

    while current_iter < self.repeats and remaining_samples > 0:
      # Original GPyOpt strategy: alternate evaluator types
      if current_iter < startup_trials:
        # Startup phase: random sampling for initialization
        evaluator = 'random'
        if self.verbose > 0:
          print(f"### Iteration {current_iter+1} / {self.repeats} - STARTUP PHASE [evaluator: {evaluator}]")
      else:
        # Main optimization: alternate between sequential and thompson_sampling (like original GPyOpt)
        # GPyOpt alternates between sequential (exploitation) and thompson_sampling (exploration)
        if current_iter % 2 == 0:
          evaluator = 'sequential'  # Use acquisition function optimization (exploitation)
        else:
          evaluator = 'thompson_sampling'  # Use Thompson sampling (exploration)
        if self.verbose > 0:
          print(f"### Iteration {current_iter+1} / {self.repeats} - OPTIMIZATION PHASE [evaluator: {evaluator}]")

      # Optimiser to get next evaluations
      batch_size = int(np.min([eval_per_step,remaining_samples]))
      if use_gpyopt:
        bop = gpo.methods.BayesianOptimization(f=None, domain=domain,
                                               X=Xdata, Y=Ydata,
                                               normalize_Y=False,
                                               batch_size=batch_size,
                                               evaluator_type=evaluator,
                                               verbosity=(self.verbose>1),
                                               acquisition_type='EI',
                                               acquisition_optimizer_type='lbfgs',
                                               exact_feval=False,
                                               de_duplication=True)
        Xnext = bop.suggest_next_locations()  # noqa: N806
      else:
        # Optimized Optuna-based suggestion with efficient duplicate prevention
        Xnext = np.ndarray((batch_size, len(var_keys)))  # noqa: N806

        if evaluator == 'random':
          # Fast random sampling with duplicate checking
          for bi in range(batch_size):
            sample_params = []
            for vk in var_keys:
              sample_params.append(random.randrange(len(models.values[vk])))  # noqa: S311
            Xnext[bi, :] = sample_params
            # Update tracking structures
            param_tuple = tuple(sample_params)
            evaluated_combinations.add(param_tuple)
            evaluated_arrays.append(np.array(sample_params))
            evaluated_matrix = np.vstack([evaluated_matrix, np.array(sample_params)])
        else:
          # Original GPyOpt-like behavior: alternate between sequential and thompson_sampling
          if evaluator == 'thompson_sampling':
            # Thompson sampling: use GPSampler with more diverse sampling (exploration)
            candidate_samples = []
            max_candidates = batch_size * 15  # More candidates for Thompson sampling

            attempts = 0
            max_attempts = max_candidates * 3  # Allow more attempts for Thompson sampling
            while len(candidate_samples) < batch_size and attempts < max_attempts:
              trial = study.ask(distributions)  # type: ignore[name-defined]
              sample_params = [int(trial.params[vk]) for vk in var_keys]
              attempts += 1

              # Use fast numpy-based duplicate checking
              if not is_duplicate_fast(sample_params):
                candidate_samples.append(sample_params)
                # Update both tracking structures
                param_tuple = tuple(sample_params)
                evaluated_combinations.add(param_tuple)
                evaluated_arrays.append(np.array(sample_params))
                evaluated_matrix = np.vstack([evaluated_matrix, np.array(sample_params)])

            # Fill Xnext with unique samples
            for bi in range(batch_size):
              if bi < len(candidate_samples):
                Xnext[bi, :] = candidate_samples[bi]
              else:
                # Fallback to random if not enough unique samples from GPSampler
                sample_params = []
                for vk in var_keys:
                  sample_params.append(random.randrange(len(models.values[vk])))  # noqa: S311
                Xnext[bi, :] = sample_params
                # Update tracking structures
                param_tuple = tuple(sample_params)
                evaluated_combinations.add(param_tuple)
                evaluated_arrays.append(np.array(sample_params))
                evaluated_matrix = np.vstack([evaluated_matrix, np.array(sample_params)])
          else:
            # Sequential evaluator: pure GPSampler with acquisition function optimization (exploitation)
            candidate_samples = []
            max_candidates = batch_size * 10  # More candidates for better acquisition optimization

            attempts = 0
            max_attempts = max_candidates * 2  # Allow more attempts for GPSampler
            while len(candidate_samples) < batch_size and attempts < max_attempts:
              trial = study.ask(distributions)  # type: ignore[name-defined]
              sample_params = [int(trial.params[vk]) for vk in var_keys]
              attempts += 1

              # Use fast numpy-based duplicate checking
              if not is_duplicate_fast(sample_params):
                candidate_samples.append(sample_params)
                # Update both tracking structures
                param_tuple = tuple(sample_params)
                evaluated_combinations.add(param_tuple)
                evaluated_arrays.append(np.array(sample_params))
                evaluated_matrix = np.vstack([evaluated_matrix, np.array(sample_params)])

            # Fill Xnext with unique samples
            for bi in range(batch_size):
              if bi < len(candidate_samples):
                Xnext[bi, :] = candidate_samples[bi]
              else:
                # Fallback to random if not enough unique samples from GPSampler
                sample_params = []
                for vk in var_keys:
                  sample_params.append(random.randrange(len(models.values[vk])))  # noqa: S311
                Xnext[bi, :] = sample_params
                # Update tracking structures
                param_tuple = tuple(sample_params)
                evaluated_combinations.add(param_tuple)
                evaluated_arrays.append(np.array(sample_params))
                evaluated_matrix = np.vstack([evaluated_matrix, np.array(sample_params)])

      # Evaluate next data points
      for x in Xnext:
        remaining_samples -= 1
        for l in range(0,len(var_keys)):
          key_vals[var_keys[l]] = models.values[var_keys[l]][int(x[l])]
        if self.verbose > 0:
          print("# Evaluate "+str(x))
          print("  "+str([str(key_vals[k]) for k in var_keys]))
        self._add_task(key_vals, path_model)
      self._run_tasks()

      # Add results; multiple times if multiple evluations due to KFold validation, etc.
      for ri in range(0,len(self.val_performance[0])):
        Ynext = np.ndarray((Xnext.shape[0],1))  # noqa: N806
        for l in range(0,Ynext.shape[0]):
          Ynext[l,0] = self.val_performance[Perf_data_pos+l][ri]
        Xdata = np.vstack((Xdata,Xnext))  # noqa: N806
        Ydata = np.vstack((Ydata,Ynext))  # noqa: N806
      Perf_data_pos = len(self.val_performance)  # noqa: N806

      # Inform Optuna about observed performances (use mean across folds)
      # This is crucial for GP model updates - GPyOpt updates after each evaluation
      if not use_gpyopt:
        res_n = len(self.val_performance[0])
        for l in range(0, Xnext.shape[0]):
          vals = [self.val_performance[Perf_data_pos - Xnext.shape[0] + l][ri] for ri in range(0, res_n)]
          mean_val = float(np.mean(vals))
          params = {vk: int(Xnext[l, i]) for i, vk in enumerate(var_keys)}
          # Add new trial to the study (this triggers GP model update)
          trial_number = len(study.trials)
          now = datetime.datetime.now()
          study.add_trial(optuna.trial.FrozenTrial(
            trial_id=trial_number,
            number=trial_number,
            state=optuna.trial.TrialState.COMPLETE,
            value=mean_val,
            datetime_start=now,
            datetime_complete=now,
            params=params,
            distributions=distributions,
            user_attrs={},
            system_attrs={},
            intermediate_values={}
          ))

        # Force GP model update by accessing the sampler's internal state
        # This ensures the GP model is updated after each batch of evaluations
        if hasattr(sampler, '_gaussian_process'):
          # Trigger GP model retraining
          try:
            sampler._gaussian_process._fit_gp_model()
          except Exception as e:
            if self.verbose > 1:
              print(f"# GP model update failed: {e}")

      # Update results - find best model based on average performance across folds
      # Calculate average performance for each unique model (accounting for k-fold)
      unique_models = Xdata.shape[0] // res_n
      model_avg_performance = []
      for model_idx in range(unique_models):
        start_idx = model_idx * res_n
        end_idx = start_idx + res_n
        avg_perf = np.mean(Ydata[start_idx:end_idx, 0])
        model_avg_performance.append(avg_perf)

      best_model_idx = np.argmin(model_avg_performance)
      best_avg_performance = model_avg_performance[best_model_idx]
      Ybest.append(best_avg_performance)

      # Get the actual data point index for the best model (first fold)
      best_data_idx = best_model_idx * res_n
      XDiff.append(np.linalg.norm(XLast-Xdata[-1,:]))
      XLast = Xdata[-1,:]  # noqa: N806
      if self.verbose > 0:
        print(f"## Best: Model {best_model_idx} avg = {best_avg_performance:.8f} {Xdata[best_data_idx,:]!s}  of {unique_models} models")
        for l in range(0,len(var_keys)):
          key_vals[var_keys[l]] = models.values[var_keys[l]][int(Xdata[best_data_idx,l])]
        print("   "+str([str(key_vals[k]) for k in var_keys]))
      # Next iter
      current_iter += 1

    # Store performance info
    folder = self._save_performance(collection_name+"-gpo", var_keys, fix_keys)

    # Print optimization summary
    if self.verbose > 0:
      print("\n# GPO Optimization Summary:")
      print(f"#   Total iterations: {current_iter}")
      print(f"#   Models evaluated: {unique_models}")
      print(f"#   Best performance: {best_avg_performance:.8f}")
      print("#   Strategy: Original GPyOpt alternating sequential/thompson_sampling evaluators")

    # GPO convergence
    fig, ax = plt.subplots(1,2)
    ax[0].plot(np.arange(0,len(XDiff))*eval_per_step,XDiff, 'ro-')
    ax[0].set_title("Distance between consecutive samples")
    ax[0].set_xlabel("Samples")
    ax[0].set_ylabel("L2 Distance")
    ax[0].xaxis.get_major_locator().set_params(integer=True)
    ax[1].plot(np.arange(0,len(Ybest))*eval_per_step,Ybest, 'bo-')
    ax[1].set_title("Best selected sample")
    ax[1].set_xlabel("Samples")
    ax[1].set_ylabel("Best Wasserstein distance error")
    ax[1].xaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    for dpi in self.image_dpi:
      plt.savefig(os.path.join(folder,"gpo_convergence@"+str(dpi)+".png"), dpi=dpi)
    if self.verbose > 1:
      fig.set_dpi(self.screen_dpi)
      plt.show(block=True)
    plt.close()

class SelectGA(Select):
  """Genetic Algorithm model selection.

  Uses Genetic Algorithm (GA) to evolve solutions and find
  optimal hyperparameters through evolutionary optimization.
  """

  def __init__(self,metabolites,dataset,epochs,validate,repeats,remote,screen_dpi,image_dpi,search_model,verbose):
    """Initialize GA selector.

    Parameters
    ----------
        metabolites (list): List of metabolite names
        dataset (str): Path to dataset
        epochs (int): Number of training epochs
        validate (str): Validation strategy
        repeats (int): Number of generations
        remote (str): Remote execution configuration
        screen_dpi (int): DPI for screen display
        image_dpi (list): DPI for saved images
        verbose (int): Verbosity level
    """
    super().__init__(remote,metabolites,dataset,epochs,validate,screen_dpi,image_dpi,search_model,verbose)
    self.repeats = repeats
    self.last_fitness = 0

  def optimise(self, collection_name, models, path_model):
    """Perform GA optimization.

    Parameters
    ----------
        collection_name (str): Name of the collection
        models (Grid): Grid of model parameters to search
        path_model (str): Path to save model results

    Returns
    -------
        str: Path to results folder
    """
    import pygad
    if self.verbose > 0:
      print("# GA Model Selection")
    keys = list(models.values.keys())
    var_keys = [k for k in keys if len(models.values[k]) > 1]
    fix_keys = [k for k in keys if len(models.values[k]) == 1]
    total = np.prod([len(models.values[k]) for k in keys])
    global ga_aux
    ga_aux = {
      'select': self,
      'keys': keys,
      'var_keys': var_keys,
      'fix_keys': fix_keys,
      'models': models,
      'path_model': path_model,
      'last_fitness': 0
    }
    # GA Setup
    pop = int(min(0.1*total,Cfg.val["ga_max_init_pop"]))
    num_parents_mating = Cfg.val["ga_num_parents_mating"]
    gene_space = []
    gene_len = []
    for k in range(0,len(var_keys)):
      gene_space.append(np.arange(0,len(models.values[var_keys[k]])))
      gene_len.append(len(gene_space[-1]))
    dim = len(var_keys)
    import math
    skip = math.floor(math.log(pop*dim,2))
    initial_population = np.ndarray((pop,dim))
    select = sobol_seq.i4_sobol_generate(dim, pop+skip)[skip:,:]
    for l in range(0,select.shape[0]):
      initial_population[l,:] = np.floor(select[l] * gene_len)
    ga_instance = pygad.GA(num_generations=self.repeats,
                           num_parents_mating=num_parents_mating,
                           initial_population=initial_population,
                           gene_type=int,
                           gene_space=gene_space,
                           parent_selection_type="sus",
                           crossover_type="single_point",
                           crossover_probability=0.5,
                           mutation_type="adaptive",
                           mutation_probability=(0.5,0.1),
                           mutation_percent_genes=(25,10),
                           fitness_func=_ga_fitness_func,
                           on_generation=_ga_on_generation,
                           suppress_warnings=(self.verbose<5),
                           allow_duplicate_genes=False)
    # Running GA
    ga_instance.run()
    # Best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    if self.verbose > 0:
      print(f"Best solution: {solution} - fitness {solution_fitness} @ gen {ga_instance.best_solution_generation}")
      l = 0
      for k in keys:
        if k in ga_aux['fix_keys']:
          print(f"   {k} = {ga_aux['models'].values[k][0]}")
        else:
          print(f"  *{k} = {ga_aux['models'].values[k][solution[l]]}")
          l += 1
    # Store performance info
    self._save_performance(collection_name+"-ga", var_keys, fix_keys)

def _ga_fitness_func(solution, solution_idx):
  """Fitness function for Genetic Algorithm.

  Parameters
  ----------
      solution (numpy.ndarray): GA solution (parameter indices)
      solution_idx (int): Index of the solution

  Returns
  -------
      float: Fitness value (lower is better)
  """
  global ga_aux
  # Setup arguments
  key_vals = {}
  l = 0
  for k in ga_aux['keys']:
    if k in ga_aux['fix_keys']:
      key_vals[k] = ga_aux['models'].values[k][0]
    else:
      key_vals[k] = ga_aux['models'].values[k][solution[l]]
      l += 1
  ga_aux['select']._add_task(key_vals, ga_aux['path_model'])
  if ga_aux['select'].verbose > 0:
    print(f"Evaluating {solution}")
    print(f"  {[key_vals[k] for k in ga_aux['keys']]}")
  # Run
  if ga_aux['select'].verbose < 5:
    v = ga_aux['select'].verbose
    ga_aux['select'].verbose = 0
  ga_aux['select']._run_tasks()
  if ga_aux['select'].verbose < 5:
    ga_aux['select'].verbose = v
  # Gather result
  val = None
  for k in range(0,len(ga_aux['select'].key_vals)):
    match = True
    kv = ga_aux['select'].key_vals[k]
    for l in range(0,len(ga_aux['var_keys'])):
      lv = ga_aux['var_keys'][l]
      if isinstance(ga_aux['models'].values[lv][solution[l]],list):
        if isinstance(ga_aux['models'].values[lv][solution[l]][0],int):
          xcmp = [int(v) for v in kv[lv]]
        elif isinstance(ga_aux['models'].values[lv][solution[l]][0],float):
          xcmp = [float(v) for v in kv[lv]]
        else:
          xcmp = kv[lv]
      elif isinstance(ga_aux['models'].values[lv][solution[l]],int):
        xcmp = int(kv[lv])
      elif isinstance(ga_aux['models'].values[lv][solution[l]],float):
        xcmp = float(kv[lv])
      else:
        xcmp = kv[lv]
      if xcmp != ga_aux['models'].values[lv][solution[l]]:
        match = False
        break
    if match:
      val = ga_aux['select'].val_performance[k][0]
      break
  if val is None:
    raise RuntimeError("Could not find result")
  if ga_aux['select'].verbose > 0:
    print(f" = {val}")
  # Avoid division by zero and ensure positive fitness
  return 1/(max(val, 1e-8))

def _ga_on_generation(ga_instance):
  """Callback function for GA generation completion.

  Parameters
  ----------
      ga_instance: PyGAD GA instance
  """  # noqa: D401
  global ga_aux
  if ga_aux['select'].verbose > 0:
    print(f"# Generation: {ga_instance.generations_completed}")
    print(f"  Fitness: {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]} - Delta:  {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - ga_aux['last_fitness']}")
  ga_aux['last_fitness'] = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

def _get_std_name(name):
  """Get standard name from path.

  Converts a file path to a standardized list of path components.

  Parameters
  ----------
      name (str): File path to standardize

  Returns
  -------
      list: List of path components in order
  """
  _, path = os.path.splitdrive(name)
  idl = []
  while True:
    path, folder = os.path.split(path)
    if folder != "":
      idl.append(folder)
    if path == "":
      break
  idl.reverse()
  return idl
