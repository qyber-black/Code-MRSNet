# mrsnet/selkection.py - MRSNet - model selection
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2021, Frank C Langbein <frank@langbein.org>, Cardiff University

import os
import subprocess
import time
import json
import csv
import numpy as np
import sobol_seq
import matplotlib.pyplot as plt

from .grid import Grid
from .dataset import Dataset

class Select:
  def __init__(self,remote,metabolites,dataset,epochs,validate,screen_dpi,image_dpi,no_show,verbose):
    from .dataset import Dataset
    if os.path.isfile(os.path.join(dataset,"spectra.joblib")):
      id = _get_std_name(dataset)
      name = os.path.join(*id[-9:-1])
      self.ds_rest = id[-1]
      if verbose > 0:
        print("# Loading dataset %s : %s" % (name,self.ds_rest))
      ds = Dataset.load(dataset) # We trust the path is in Cfg{path_simulation},
                                 # so no re-construction here (as in ../mrsnet.py)
      self.pulse_sequence = ds.pulse_sequence
    else:
      raise Exception("Cannot find dataset")
      ds = None # Load later, as dicom and we don't know metabolites
    self.metabolites = metabolites
    self.dataset = dataset
    self.epochs = epochs
    self.validate = validate
    self.screen_dpi = screen_dpi
    self.image_dpi = image_dpi
    self.no_show = no_show
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
        for l in output[:-2]:
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
      from .model import CNN
      # cnn_[S1]_[S2]_[C1]_[C2]_[C3]_[C4]_[O1]_[O2]_[F1]_[F2]_[D]_[softmax,sigmoid][_pool]
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
      if na['model_POOL'][0] == True:
        model_str += "_pool"
      model_name = str(CNN(model_str, self.metabolites, self.pulse_sequence,
                           na['acquisitions'], na['datatype'], na['norm'][0]))
    elif na['model'][0][0:4] == 'cnn_':
      # CNN standard models
      from .model import CNN
      model_name = str(CNN(na['model'][0], self.metabolites, self.pulse_sequence,
                           na['acquisitions'], na['datatype'], na['norm'][0]))
    else:
      raise Exception("Unknown model %s" % na['model'])
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
      raise Exception("Unknown validation %f" % args.validate)
    return os.path.join(path_model, model_name, na['batch_size'][0], str(self.epochs),
                        train_model, trainer+"-1"), fold

  def _add_task(self,key_vals,path_model):
    model_path, fold = self._model_path(key_vals,path_model)
    self.tasks.append({
        'model_path': model_path,
        'fold': fold,
        'args': key_vals
      })

  def _run_tasks(self):
    counter = 1
    remote_run = []
    for t in self.tasks:
      print("# Task %d / %d" % (counter,len(self.tasks)))
      if os.path.exists(os.path.join(t['model_path'],t['fold'],"tf_model")):
        if self.verbose > 0:
          print("Exists %s:%s" % (t['model_path'],t['fold']))
        val_p, train_p = self._load_performance(t['model_path'], t['fold'])
        self.key_vals.append(t['args'])
        self.val_performance.append(val_p)
        self.train_performance.append(train_p)
      else:
        # Train
        if t['args']['model'] == 'cnn':
          # cnn_[S1]_[S2]_[C1]_[C2]_[C3]_[C4]_[O1]_[O2]_[F1]_[F2]_[D]_[softmax,sigmoid][_pool]
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
          if t['args']['model_POOL'] == True:
            model_str += "_pool"
        elif t['args']['model'][0:4] == 'cnn_':
          model_str = t['args']['model']
        else:
          raise Exception("Unknown model string %s" % t['args']['model'])
        if self.remote == 'local':
          self._run(t['args']['norm'],t['args']['acquisitions'],t['args']['datatype'],
                    model_str,t['args']['batch_size'])
          val_p, train_p = self._load_performance(t['model_path'], t['fold'])
          self.key_vals.append(t['args'])
          self.val_performance.append(val_p)
          self.train_performance.append(train_p)
        else:
          remote_run.append(['run',
                             [t['args']['norm'],
                              t['args']['acquisitions'],
                              t['args']['datatype'],
                              model_str,
                              t['args']['batch_size']],
                             t['model_path'],t['fold']])
      counter += 1
    if len(remote_run) > 0:
      if self.verbose > 0:
        print("# Running remotely")
      # Remote tasks scheduling
      while len([l for l in remote_run if l[0] == 'done']) != len(remote_run):
        for k in range(0,len(remote_run)):
          print("## Job %d / %d" % (k+1,len(remote_run)))
          status = self._run_remote(k,remote_run)
          if status == 'done':
            val_p, train_p = self._load_performance(remote_run[k][2], remote_run[k][3])
            self.key_vals.append(t['args'])
            self.val_performance.append(val_p)
            self.train_performance.append(train_p)
        running = len([l for l in remote_run if l[0] == 'run'])
        waiting = len([l for l in remote_run if l[0] == 'wait'])
        if self.verbose > 0:
          print("  %d running; %d waiting" % (running,waiting))
        if running >= self.remote_tasks or (waiting < 1 and running > 0):
          time.sleep(self.remote_wait*60)

    self.tasks = []

  def _run(self,norm,acquisitions,datatype,model,batch_size):
    cmd = ['/usr/bin/env', 'python3', 'mrsnet.py', 'train', '--no-show',
           '--metabolites', *[m for m in self.metabolites],
           '--pulse_sequence', self.pulse_sequence,
           '--dataset', self.dataset,
           '--epochs', str(self.epochs),
           '--validate', str(self.validate),
           '--norm', norm,
           '--acquisitions', *[a for a in acquisitions],
           '--datatype', *[d for d in datatype],
           '--model', model,
           '--batch-size', str(batch_size)]
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
    if len(fold) == 0:
      with open(os.path.join(model_path,"train_concentration_errors.json"), 'r') as f:
        data = json.load(f)
      train_p = [data['wasserstein_distance_quality']]
      with open(os.path.join(model_path,"validation_concentration_errors.json"), 'r') as f:
        data = json.load(f)
      val_p = [data['wasserstein_distance_quality']]
    else:
      train_p = []
      val_p = []
      f_cnt = 0
      while os.path.exists(os.path.join(model_path,"fold-"+str(f_cnt))):
        with open(os.path.join(model_path,"fold-"+str(f_cnt),"train_concentration_errors.json"), 'r') as f:
          data = json.load(f)
        train_p.append(data['wasserstein_distance_quality'])
        with open(os.path.join(model_path,"fold-"+str(f_cnt),"validation_concentration_errors.json"), 'r') as f:
          data = json.load(f)
        val_p.append(data['wasserstein_distance_quality'])
        f_cnt += 1
    return val_p, train_p

  def _save_performance(self, collection_name, var_keys, fix_keys):
    var_keys.sort()
    fix_keys.sort()
    # Results folder
    folder = os.path.join(self.dataset,collection_name+"-"+str(self.epochs))
    if not os.path.exists(folder):
      os.makedirs(folder)
    # Store performance data
    idx = [l[0] for l in sorted(enumerate(self.val_performance), key=lambda x:np.mean(x[1]))]
    with open(os.path.join(folder,"model_performance.csv"), "w") as f:
      writer = csv.writer(f, delimiter=",")
      writer.writerow(["Results from SelectGrid"])
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
      writer.writerow([*var_keys,"Val. Perf."*len(self.val_performance[0]),
                                 "Train Perf."*len(self.train_performance[0])])
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
    Y = np.arange(2*len(val_error),1,-2)
    ax.barh(y=Y, width=val_error, height=0.9, left=0, align='center',
            label="Val. Error", color="#4878D0", zorder=1)
    ax.barh(y=Y-0.75, width=train_error, height=0.6, left=0, align='center',
            label="Train Error", color="#A1C9F4", zorder=0)
    ax.scatter(y=Y, x=np.array([self.val_performance[idx[p]]
                                for p in range(0,len(self.val_performance))]).flatten(),
               color="#000000", s=5.0, zorder=2)
    ax.scatter(y=Y-0.75, x=np.array([self.train_performance[idx[p]]
                                     for p in range(0,len(self.train_performance))]).flatten(),
               color="#000000", s=5.0, zorder=2)
    plt.yticks(Y, parameters)
    ax.legend(loc="upper right", frameon=True)
    ax.set(xlim=(0,np.max([np.max(self.val_performance[idx[-1]]),
                           np.max(self.train_performance[idx[-1]])])),
           ylabel="", xlabel="Wasserstein Distance Error")
    fig.tight_layout()
    for dpi in self.image_dpi:
      plt.savefig(os.path.join(folder,"errors@"+str(dpi)+".png"), dpi=dpi)
    if not self.no_show:
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
      ax[k].set_ylabel("Wasserstein Dist. Error")
      ax[k].set_xticks(X)
      ax[k].set_xticklabels([group_id+": "+v+" (val.,train)" for v in key_vals]+[""]*(x_max-len(key_vals)))
    fig.tight_layout()
    for dpi in self.image_dpi:
      plt.savefig(os.path.join(folder,"error-dists@"+str(dpi)+".png"), dpi=dpi)
    if not self.no_show:
      fig.set_dpi(self.screen_dpi)
      plt.show(block=True)
    plt.close()

class SelectGrid(Select):
  def __init__(self,metabolites,dataset,epochs,validate,remote,screen_dpi,image_dpi,no_show,verbose):
    super(SelectGrid, self).__init__(remote,metabolites,dataset,epochs,validate,screen_dpi,image_dpi,no_show,verbose)

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
        print("# Model %d / %d" % (counter,total))
      key_vals = {}
      for l in range(0,len(keys)):
        key_vals[keys[l]] = model[l]
      self._add_task(key_vals, path_model)
      counter += 1
    self._run_tasks()
    # Store performance info
    self._save_performance(collection_name+"-grid", var_keys, fix_keys)

class SelectQMC(Select):
  def __init__(self,metabolites,dataset,epochs,validate,repeats,remote,screen_dpi,image_dpi,no_show,verbose):
    super(SelectQMC, self).__init__(remote,metabolites,dataset,epochs,validate,screen_dpi,image_dpi,no_show,verbose)
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
        print("# Sample %d / %d in space of size %d" % (counter,self.repeats,total))
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
  def __init__(self,metabolites,dataset,epochs,validate,repeats,remote,mscreen_dpi,image_dpi,no_show,verbose):
    super(SelecttGPO, self).__init__(remote,metabolites,dataset,epochs,validate,screen_dpi,image_dpi,no_show,verbose)
    self.repeats = repeats

  def optimise(self, collection_name, models, path_model):
    # FIXME: ...
    pass

class SelectEvo(Select):
  def __init__(self,metabolites,dataset,epochs,validate,repeats,remote,screen_dpi,image_dpi,no_show,verbose):
    super(SelectEvo, self).__init__(remote,metabolites,dataset,epochs,validate,screen_dpi,image_dpi,no_show,verbose)
    self.repeats = repeats

  def optimise(self, collection_name, models, path_model):
    # FIXME: ...
    pass

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

Collections = {
  # Parameter lists (i.e. lists for single arguments) must be sorted!
  'test-1': Grid({
    'norm':         ['sum'],
    'acquisitions': [['difference','edit_off'], ['difference','edit_on']],
    'datatype':     [['magnitude']],
    'model':        ['cnn_small_softmax', 'cnn_medium_softmax', 'cnn_large_softmax'],
    'batch_size':   [16, 32, 64]
  }),
  'test-2': Grid({
    'norm':             ['sum'],
    'acquisitions':     [['difference','edit_on']],
    'datatype':         [['magnitude','phase']],
    'model':            ['cnn'],
    'model_S1':         [2],
    'model_S2':         [3],
    'model_C1':         [4, 8, 16],
    'model_C2':         [7],
    'model_C3':         [5],
    'model_C4':         [3],
    'model_O1':         [0.4, 0.25],
    'model_O2':         [0.25],
    'model_F1':         [256],
    'model_F2':         [512],
    'model_D':          [1024],
    'model_ACTIVATION': ['softmax'],
    'model_POOL':       [False],
    'batch_size':       [32]
  }),
  'test-3': Grid({
    'norm':         ['sum'],
    'acquisitions': Grid.all_combinations_sort(['edit_off', 'edit_on', 'difference']),
    'datatype':     Grid.all_combinations_sort(['magnitude', 'phase', 'real', 'imaginary']),
    'model':        ['cnn_small_softmax', 'cnn_medium_softmax', 'cnn_large_softmax'],
    'batch_size':   [16, 32, 64]
  })
}
