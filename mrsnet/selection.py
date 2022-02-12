# mrsnet/selection.py - MRSNet - model selection
#
# SPDX-FileCopyrightText: Copyright (C) 2020-2022 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import subprocess
import time
import json
import csv
import numpy as np
import sobol_seq
import random
import matplotlib.pyplot as plt

from mrsnet.grid import Grid
from mrsnet.dataset import Dataset
from mrsnet.cfg import Cfg

class Select:
  def __init__(self,remote,metabolites,dataset,epochs,validate,screen_dpi,image_dpi,verbose):
    from mrsnet.dataset import Dataset
    if os.path.isfile(os.path.join(dataset,"spectra_noisy.joblib")):
      id = _get_std_name(dataset)
      name = os.path.join(*id[-9:-1])
      self.ds_rest = id[-1]
      if verbose > 0:
        print(f"# Loading dataset {name} : {self.ds_rest}")
      ds = Dataset.load(dataset, info_only=True)
      self.pulse_sequence = ds.pulse_sequence
    elif os.path.isfile(os.path.join(dataset,"spectra_clean.joblib")):
      id = _get_std_name(dataset)
      name = os.path.join(*id[-9:-1])
      self.ds_rest = id[-1]
      if verbose > 0:
        print(f"# Loading dataset {name} : {self.ds_rest}")
      ds = Dataset.load(dataset, info_only=True)
      self.pulse_sequence = ds.pulse_sequence
    else:
      raise Exception("Cannot find dataset")
    self.metabolites = metabolites
    self.dataset = dataset
    self.epochs = epochs
    self.validate = validate
    self.screen_dpi = screen_dpi
    self.image_dpi = image_dpi
    self.verbose = verbose
    self.dataset_name = ds.name

    if len(remote) > 0:
      remote = remote.split(":")
      self.remote = remote[0]
      self.remote_user = remote[1]
      self.remote_tasks = int(remote[2]) if len(remote) > 2 else 10
      self.remote_wait = int(remote[3]) if len(remote) > 3 else 15
      cmd = ['/usr/bin/env', 'bash', self.remote, self.remote_user, "sync"]
      p = subprocess.run(cmd, capture_output=True)
      output = p.stdout.decode("utf-8").split("\n")
      if self.verbose > 0:
        print("# Remote Sync")
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

  def _model_path(self,key_vals,path_model):
    na = {}
    for k in key_vals:
      if isinstance(key_vals[k],list):
        na[k] = [str(val) for val in key_vals[k]]
      else:
        na[k] = [str(key_vals[k])]
    # Get model info
    if na['model'][0] == 'cnn':
      # Model fully parameterised
      from mrsnet.cnn import CNN
      # cnn_[S1]_[S2]_[C1]_[C2]_[C3]_[C4]_[O1]_[O2]_[F1]_[F2]_[D]_[softmax,sigmoid]
      model_str = ('cnn_' + na['model_S1'][0] +
                      '_' + na['model_S2'][0] +
                      '_' + na['model_C1'][0] +
                      '_' + na['model_C2'][0] +
                      '_' + na['model_C3'][0] +
                      '_' + na['model_C4'][0] +
                      '_' + na['model_O1'][0] +
                      '_' + na['model_O2'][0] +
                      '_' + na['model_F1'][0] +
                      '_' + na['model_F2'][0] +
                      '_' + na['model_D'][0] +
                      '_' + na['model_ACTIVATION'][0])
      model_name = str(CNN(model_str, self.metabolites, self.pulse_sequence,
                           na['acquisitions'], na['datatype'], na['norm'][0]))
    elif na['model'][0][0:4] == 'cnn_':
      # CNN standard models
      from mrsnet.cnn import CNN
      model_name = str(CNN(na['model'][0], self.metabolites, self.pulse_sequence,
                           na['acquisitions'], na['datatype'], na['norm'][0]))
    else:
      # FIXME: add ae_[cnn,fc] models with parameters
      raise Exception(f"Unknown model {na['model']}")
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
      raise Exception(f"Unknown validation {args.validate}")
    # Check if sane, delete otherwise
    base_path = os.path.join(path_model, model_name, na['batch_size'][0], str(self.epochs),
                             train_model)
    # Find model folder, assuming there may be multiple repeats.
    # We assume the last valid repeat is the best option/latest result.
    # iterate over files in
    selected_id = -1
    if os.path.isdir(base_path):
      for fn in os.listdir(base_path):
        ffn = os.path.join(base_path,fn)
        if os.path.isdir(ffn):
          file_s = fn.split("-")
          repeat_id = -1
          if len(file_s) > 1:
            try:
              repeat_id = int(file_s[-1])
            except:
              pass
          if (repeat_id > 0 and
              fn == trainer+"-"+str(repeat_id) and
              os.path.exists(os.path.join(base_path,fn,
                             fold,"train_concentration_errors.json")) and
              os.path.exists(os.path.join(base_path,fn,
                             fold,"validation_concentration_errors.json"))):
            if repeat_id > selected_id:
              selected_id = repeat_id
          else:
            if self.verbose > 0:
              print(f"# WARNING: {fn} - broken/incomplete model")
        else:
          if self.verbose > 0:
            print(f"# WARNING: {ffn} - this file should not be there")
    if selected_id < 0:
      selected_id = 1
    return os.path.join(base_path,trainer+"-"+str(selected_id)), fold

  def _add_task(self,key_vals,path_model):
    model_path, fold = self._model_path(key_vals,path_model)
    self.tasks.append({
        'model_path': model_path,
        'fold': fold,
        'args': key_vals.copy()
      })

  def _run_tasks(self, load_only=False):
    counter = 1
    remote_run = []
    for t in self.tasks:
      if not load_only and self.verbose > 0:
        print(f"# Task {counter} / {len(self.tasks)}")
      val_p = None
      if os.path.exists(os.path.join(t['model_path'],t['fold'],"tf_model")):
        if self.verbose > 0:
          print(f"Exists {t['model_path']}:{t['fold']}")
        val_p, train_p = self._load_performance(t['model_path'], t['fold'])
        if val_p is not None:
          self.key_vals.append(t['args'])
          self.val_performance.append(val_p)
          self.train_performance.append(train_p)
      if val_p is None and not load_only:
        # Train
        if t['args']['model'] == 'cnn':
          # cnn_[S1]_[S2]_[C1]_[C2]_[C3]_[C4]_[O1]_[O2]_[F1]_[F2]_[D]_[softmax,sigmoid]
          model_str = ('cnn_' + str(t['args']['model_S1']) +
                          '_' + str(t['args']['model_S2']) +
                          '_' + str(t['args']['model_C1']) +
                          '_' + str(t['args']['model_C2']) +
                          '_' + str(t['args']['model_C3']) +
                          '_' + str(t['args']['model_C4']) +
                          '_' + str(t['args']['model_O1']) +
                          '_' + str(t['args']['model_O2']) +
                          '_' + str(t['args']['model_F1']) +
                          '_' + str(t['args']['model_F2']) +
                          '_' + str(t['args']['model_D']) +
                          '_' + t['args']['model_ACTIVATION'])
        elif t['args']['model'][0:4] == 'cnn_':
          model_str = t['args']['model']
        else:
          raise Exception(f"Unknown model string {t['args']['model']}")
        if self.remote == 'local':
          self._run(t['args']['norm'],t['args']['acquisitions'],t['args']['datatype'],
                    model_str,t['args']['batch_size'])
          val_p, train_p = self._load_performance(t['model_path'], t['fold'])
          if val_p is not None:
            self.key_vals.append(t['args'])
            self.val_performance.append(val_p)
            self.train_performance.append(train_p)
          else:
            raise Exception("Local job failed: "+str(t)+" : "+model_str)
        else:
          remote_run.append(['wait',
                             [t['args']['norm'],
                              t['args']['acquisitions'],
                              t['args']['datatype'],
                              model_str,
                              t['args']['batch_size']],
                             t['model_path'],
                             t['fold'],
                             t['args']])
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
              if val_p is not None:
                self.key_vals.append(remote_run[k][4])
                self.val_performance.append(val_p)
                self.train_performance.append(train_p)
              else:
                raise Exception("Job failed: "+str(remote[k]))
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

  def _run(self,norm,acquisitions,datatype,model,batch_size):
    cmd = ['/usr/bin/env', 'python3', 'mrsnet.py', 'train',
           '--metabolites', *[m for m in self.metabolites],
           '--dataset', self.dataset,
           '--epochs', str(self.epochs),
           '--validate', str(self.validate),
           '--norm', norm,
           '--acquisitions', *[a for a in acquisitions],
           '--datatype', *[d for d in datatype],
           '--model', model,
           '--batchsize', str(batch_size)]
    if self.verbose > 0:
      cmd += ['-v']*self.verbose
      print('# Run '+' '.join(cmd[3:]))
    try:
      p = subprocess.Popen(cmd)
    except OSError as e:
      raise Exception('MRSNet training failed') from e
    p.wait()

  def _run_remote(self,id,all):
    cmd = ['/usr/bin/env', 'bash', self.remote, self.remote_user, "X",
           self.dataset,
           "-".join([m for m in self.metabolites]),
           self.pulse_sequence,
           str(self.epochs),
           str(self.validate),
           all[id][1][0],
           "-".join([a for a in all[id][1][1]]),
           "-".join([d for d in all[id][1][2]]),
           all[id][1][3],
           str(all[id][1][4])]
    if all[id][0] == 'run':
      cmd[4] = "check"
      p = subprocess.run(cmd, capture_output=True)
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
        p = subprocess.run(cmd, capture_output=True)
        output = p.stdout.decode("utf-8").split("\n")
        if self.verbose > 0:
          for l in output[:-2]:
            print(l)
        all[id][0] = 'run'
    return all[id][0]

  def _load_performance(self, model_path, fold):
    try:
      if len(fold) == 0:
        with open(os.path.join(model_path,"train_concentration_errors.json"), 'r') as f:
          data = json.load(f)
        train_p = [data['total']['abserror']['mean']] # total MAE
        with open(os.path.join(model_path,"validation_concentration_errors.json"), 'r') as f:
          data = json.load(f)
        val_p = [data['total']['abserror']['mean']] # total MAE
      else:
        train_p = []
        val_p = []
        f_cnt = 0
        while os.path.exists(os.path.join(model_path,"fold-"+str(f_cnt))):
          with open(os.path.join(model_path,"fold-"+str(f_cnt),"train_concentration_errors.json"), 'r') as f:
            data = json.load(f)
          train_p.append(data['total']['abserror']['mean']) # total MAE
          with open(os.path.join(model_path,"fold-"+str(f_cnt),"validation_concentration_errors.json"), 'r') as f:
            data = json.load(f)
          val_p.append(data['total']['abserror']['mean']) # total MAE
          f_cnt += 1
    except Exception as e:
      if self.verbose > 0:
        print(f"# WARNING: {model_path} - model broken")
        print(e)
      return None, None
    return val_p, train_p

  def _save_performance(self, collection_name, var_keys, fix_keys):
    var_keys.sort()
    fix_keys.sort()
    # Results folder
    basename = os.path.basename(collection_name).replace(".json","")
    from mrsnet.getfolder import get_folder
    folder = get_folder(self.dataset,basename+"-"+str(self.epochs)+"-%s")
    if not os.path.exists(folder):
      os.makedirs(folder)
    # Store performance data
    idx = [l[0] for l in sorted(enumerate(self.val_performance), key=lambda x:np.mean(x[1]))]
    with open(os.path.join(folder,"model_performance.csv"), "w") as f:
      writer = csv.writer(f, delimiter=",")
      writer.writerow([f"Results from {self.__class__.__name__}"])
      writer.writerow([])
      writer.writerow(["Dataset", self.dataset])
      writer.writerow(["Epochs", self.epochs])
      writer.writerow(["Validate", self.validate])
      writer.writerow([])
      writer.writerow(["Fixed Parameters", *fix_keys])
      row=[""]
      for k in fix_keys:
        if isinstance(self.key_vals[0][k],list):
          self.key_vals[0][k] = "-".join(self.key_vals[0][k])
        row.append(self.key_vals[0][k])
      writer.writerow(row)
      writer.writerow([])
      writer.writerow(var_keys
                      + ["Val. Perf."]*len(self.val_performance[0])
                      + ["Train Perf."]*len(self.train_performance[0]))
      for l in range(0,len(self.val_performance)):
        ll = idx[l]
        row = []
        for k in var_keys:
          if isinstance(self.key_vals[ll][k],list):
            self.key_vals[ll][k] = "-".join(self.key_vals[ll][k]) # Merge lists for later (plot)
          row.append(self.key_vals[ll][k])
        row += self.val_performance[ll]
        row += self.train_performance[ll]
        writer.writerow(row)
    # Bar plot of validate/train performance
    fig, ax = plt.subplots()
    parameters = [":".join([str(self.key_vals[idx[p]][k]) for k in var_keys])
                  for p in range(0,len(self.key_vals))]
    val_error = [np.mean(self.val_performance[idx[p]]) for p in range(0,len(self.val_performance))]
    train_error = [np.mean(self.train_performance[idx[p]]) for p in range(0,len(self.val_performance))]
    top_n = np.min([len(self.val_performance),50])
    Y = np.arange(2*len(val_error),1,-2)
    ax.barh(y=Y[:top_n], width=val_error[:top_n], height=0.9, left=0, align='center',
            label="Val. Error", color="#4878D0", zorder=1)
    ax.barh(y=Y[:top_n]-0.75, width=train_error[:top_n], height=0.6, left=0, align='center',
            label="Train Error", color="#A1C9F4", zorder=0)
    ax.scatter(y=[Y[:top_n]]*len(self.val_performance[0]),
               x=[self.val_performance[idx[p]][q]
                  for q in range(0,len(self.val_performance[0])) for p in range(0,top_n)],
               color="#000000", s=5.0, zorder=2)
    ax.scatter(y=[Y[:top_n]-0.75]*len(self.train_performance[0]),
               x=[self.train_performance[idx[p]][q]
                   for q in range(0,len(self.train_performance[0])) for p in range(0,top_n)],
               color="#000000", s=5.0, zorder=2)
    plt.yticks(Y[:top_n], parameters[:top_n])
    ax.legend(loc="upper right", frameon=True)
    ax.set(xlim=(0,np.max([np.max([self.val_performance[idx[p]] for p in range(0,top_n)]),
                           np.max([self.train_performance[idx[p]] for p in range(0,top_n)])])),
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
      m = np.max(len(set([str(self.key_vals[p][group_id]) for p in range(0,len(self.val_performance))])))
      if m > x_max:
        x_max = m
    X = np.arange(1,x_max+1)
    fig, ax = plt.subplots(len(var_keys),1)
    for k in range(0,len(var_keys)):
      group_id = var_keys[k]
      key_vals = sorted(list(set([self.key_vals[p][group_id]
                                  for p in range(0,len(self.val_performance))])))
      key_vals = [str(v) for v in key_vals]
      for ki in range(0,len(key_vals)):
        val_per = [val_error[p]
                    for p in range(0,len(self.val_performance))
                      if key_vals[ki] == str(self.key_vals[idx[p]][group_id])]
        train_per = [train_error[p]
                     for p in range(0,len(self.train_performance))
                       if key_vals[ki] == str(self.key_vals[idx[p]][group_id])]
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

class SelectGrid(Select):
  def __init__(self,metabolites,dataset,epochs,validate,remote,screen_dpi,image_dpi,verbose):
    super(SelectGrid, self).__init__(remote,metabolites,dataset,epochs,validate,screen_dpi,image_dpi,verbose)

  def optimise(self, collection_name, models, path_model):
    if self.verbose > 0:
      print("# Grid Model Selection")
    keys = [k for k in models.values.keys()]
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
  def __init__(self,metabolites,dataset,epochs,validate,repeats,remote,screen_dpi,image_dpi,verbose):
    super(SelectQMC, self).__init__(remote,metabolites,dataset,epochs,validate,screen_dpi,image_dpi,verbose)
    self.repeats = repeats

  def optimise(self, collection_name, models, path_model):
    if self.verbose > 0:
      print("# QMC Model Selection")
    keys = [k for k in models.values.keys()]
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
  def __init__(self,metabolites,dataset,epochs,validate,repeats,remote,screen_dpi,image_dpi,verbose):
    super(SelectGPO, self).__init__(remote,metabolites,dataset,epochs,validate,screen_dpi,image_dpi,verbose)
    self.repeats = repeats

  def optimise(self, collection_name, models, path_model):
    if self.verbose > 0:
      print("# GPO Model Selection")
    keys = [k for k in models.values.keys()]
    var_keys = [k for k in keys if len(models.values[k]) > 1]
    fix_keys = [k for k in keys if len(models.values[k]) == 1]
    total = np.prod([len(models.values[k]) for k in keys])
    if self.verbose > 0:
      print(f"Search space size: {total}")
    import GPyOpt as gpo
    domain = []
    self.values = {}
    for k in var_keys:
      if isinstance(models.values[k][0],int) or isinstance(models.values[k][0],float):
        type = 'discrete'
      else:
        type = 'categorical'
      domain.append({
          'name': k,
          'type': type,
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
    if not Cfg.dev('selectgpo_optimse_noload'):
      for model in models:
        for ki,k in enumerate(keys):
          if k in var_keys:
            key_vals[k] = models.values[k][models.values[k].index(model[ki])]
        self._add_task(key_vals, path_model)
        self._run_tasks(load_only=True)
    # Evaluate first, if none available so far
    if len(self.key_vals) < 1:
      for ki,k in enumerate(keys):
        if k in var_keys:
          key_vals[k] = models.values[k][random.randrange(len(models.values[k]))]
      self._add_task(key_vals, path_model)
      self._run_tasks()
    # Convert eval. data to GPO format
    Xdata = np.ndarray((0,len(var_keys)))
    Ydata = np.ndarray((0,1))
    res_n = len(self.val_performance[0])
    Xnext = np.ndarray((len(self.key_vals)*res_n,len(var_keys)))
    Ynext = np.ndarray((Xnext.shape[0],1))
    ll = 0
    for l in range(0,len(self.key_vals)):
      # Add results; multiple times if multiple evluations due to KFold validation, etc.
      for ri in range(0,res_n):
        for ki,k in enumerate(var_keys):
          Xnext[ll,ki] = models.values[k].index(self.key_vals[l][k])
        Ynext[ll,0] = self.val_performance[l][ri]
        ll += 1
    Xdata = np.vstack((Xdata,Xnext))
    Ydata = np.vstack((Ydata,Ynext))
    # Init GPO performance data
    remaining_samples = total - Xdata.shape[0] // res_n
    idx_best = np.argmin(Ydata,axis=0)[0]
    Ybest = [Ydata[idx_best,0]]
    XDiff = [0]
    XLast = Xdata[-1,:]
    if self.verbose > 0:
      print(f"## Best: Y[{idx_best}] = {Ybest[-1]} {str(Xdata[idx_best,:])}  of {Ydata.shape[0]}/{res_n} = {Ydata.shape[0]//res_n} samples")

    # Optimisation iterations
    current_iter = len(self.key_vals)
    if self.remote == 'local':
      eval_per_step = 1
    else:
      eval_per_step = self.remote_tasks
    while current_iter < self.repeats and remaining_samples > 0:
      if eval_per_step  == 1:
        evaluator = 'sequential'
      else:
        # Switch to avoid posterior sampling bias
        evaluator = 'thompson_sampling' if current_iter % 2 == 0 else 'random'
      if self.verbose > 0:
        print(f"### Iteration {current_iter+1} / {self.repeats} - samples remaining: {remaining_samples} [eval: {evaluator}]")
      # Optimiser to get next evaluations
      bop = gpo.methods.BayesianOptimization(f=None, domain=domain,
                                             X=Xdata, Y=Ydata,
                                             normalize_Y=False,
                                             batch_size=np.min([eval_per_step,remaining_samples]),
                                             evaluator_type=evaluator,
                                             verbosity=(self.verbose>1),
                                             acquisition_type='EI',
                                             acquisition_optimizer_type='lbfgs',
                                             exact_feval=False,
                                             de_duplication=True)
      Xnext = bop.suggest_next_locations()
      # Evaluate next data points
      Y_data_pos = len(self.val_performance)
      for x in Xnext:
        remaining_samples -= 1
        for l in range(0,len(var_keys)):
          key_vals[var_keys[l]] = models.values[var_keys[l]][int(x[l])]
        if self.verbose > 0:
          print("# Evaluate "+str(x))
          print("  "+str([str(key_vals[k]) for k in var_keys]))
        self._add_task(key_vals, path_model)
      self._run_tasks()
      Ynext = np.ndarray((Xnext.shape[0],1))
      # Add results; multiple times if multipel evluations due to KFold validation, etc.
      for ri in range(0,len(self.val_performance[0])):
        for l in range(0,Ynext.shape[0]):
          Ynext[l,0] = self.val_performance[Y_data_pos+l][ri]
        Xdata = np.vstack((Xdata,Xnext))
        Ydata = np.vstack((Ydata,Ynext))
      idx_best = np.argmin(Ydata,axis=0)[0]
      Ybest.append(Ydata[idx_best,0])
      XDiff.append(np.linalg.norm(XLast-Xdata[-1,:]))
      XLast = Xdata[-1,:]
      if self.verbose > 0:
        print(f"## Best: Y[{idx_best}] = {Ybest[-1]} {str(Xdata[idx_best,:])}  of {Ydata.shape[0]}/{res_n} = {Ydata.shape[0]//res_n} samples")
        for l in range(0,len(var_keys)):
          key_vals[var_keys[l]] = models.values[var_keys[l]][int(Xdata[idx_best,l])]
        print("   "+str([str(key_vals[k]) for k in var_keys]))
      # Next iter
      current_iter += 1

    # Store performance info
    self._save_performance(collection_name+"-gpo", var_keys, fix_keys)

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
    folder = os.path.join(self.dataset,collection_name+"-gpo-"+str(self.epochs))
    for dpi in self.image_dpi:
      plt.savefig(os.path.join(folder,"gpo_convergence@"+str(dpi)+".png"), dpi=dpi)
    if self.verbose > 1:
      fig.set_dpi(self.screen_dpi)
      plt.show(block=True)
    plt.close()

class SelectGA(Select):
  def __init__(self,metabolites,dataset,epochs,validate,repeats,remote,screen_dpi,image_dpi,verbose):
    super(SelectGA, self).__init__(remote,metabolites,dataset,epochs,validate,screen_dpi,image_dpi,verbose)
    self.repeats = repeats
    self.last_fitness = 0

  def optimise(self, collection_name, models, path_model):
    import pygad
    if self.verbose > 0:
      print("# GA Model Selection")
    keys = [k for k in models.values.keys()]
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
    pop = int(min(0.01*total,Cfg.val["ga_max_init_pop"]))
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
                           crossover_probability=0.25,
                           mutation_type="adaptive",
                           mutation_probability=(0.25,0.1),
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
      if kv[lv] != ga_aux['models'].values[lv][solution[l]]:
        match = False
        break
    if match:
      val = ga_aux['select'].val_performance[k][0]
      break
  if val == None:
    raise Exception("Could not find result")
  if ga_aux['select'].verbose > 0:
    print(f" = {val}")
  return 1/(val+1e-8)

def _ga_on_generation(ga_instance):
  global ga_aux
  if ga_aux['select'].verbose > 0:
    print(f"# Generation: {ga_instance.generations_completed}")
    print(f"  Fitness: {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]} - Delta:  {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - ga_aux['last_fitness']}")
  ga_aux['last_fitness'] = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

def _get_std_name(name):
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
