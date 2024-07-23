# mrsnet/train.py - MRSNet - training
#
# SPDX-FileCopyrightText: Copyright (C) 2019 Max Chandler, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2020-2023 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2022-2024 Zien Ma, PhD student at Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
from scipy.stats import wasserstein_distance

from mrsnet.analyse import analyse_model
from mrsnet.getfolder import get_folder
from mrsnet.cfg import Cfg

class Train:

  def __init__(self,k):
    self.k=k
    self._bucket_idx = []

  def _plot_distributions(self,d_out,folder,image_dpi,screen_dpi,verbose):
    # Plot distributions
    if len(d_out.shape) != 2:
      return # Nothing to plot, d_out is not a concentration tensor
    data_dim = d_out.shape[0]
    out_dim = d_out.shape[-1]
    if self.k > 1:
      fig, axes = plt.subplots(self.k+1,out_dim)
    else:
      fig, axes = plt.subplots(self.k,out_dim)
      axes = np.reshape(axes, (1, out_dim))
    fig.suptitle("Output value distributions in buckets")
    for dim in range(0,out_dim):
      axes [0,dim].hist(d_out[:,dim],
                        bins=int(np.max([25, np.ceil(data_dim/(self.k*100))])))
      if dim == 0:
        axes[0,dim].set_ylabel("Full dist.")
      if len(self._bucket_idx) > 0:
        for b in range(0,self.k):
          axes[b+1,dim].hist(d_out[np.where(self._bucket_idx==b)[0],dim],
                             bins=int(np.max([25, np.ceil(data_dim/(self.k*100))])))
          if b == self.k-1:
            axes[b+1,dim].set_xlabel(f"Output value {dim}")
          if dim == 0:
            axes[b+1,dim].set_ylabel(f"{b} Buckets")
    if not os.path.isdir(folder):
      os.makedirs(folder)
    for f in glob.glob(os.path.join(folder,"output-distribution@*.png")):
      os.remove(f)
    for dpi in image_dpi:
      plt.savefig(os.path.join(folder,"output-distribution@"+str(dpi)+".png"), dpi=dpi)
    if verbose > 1:
      fig.set_dpi(screen_dpi)
      plt.show(block=True)
    plt.close()

  def _cross_validate(self, model, epochs, batch_size, data, folder,
                      train_dataset_name, verbose, image_dpi, screen_dpi):
    # Cross validation
    train_res = { 'error': [None]*self.k }
    val_res = { 'error': [None]*self.k }
    has_error = True # analyse_model produces error distribtuions if this is true
    for val_fold in range(0,self.k):
      if verbose > 0:
        print(f"# Fold {val_fold+1} of {self.k}")
      fold_folder = os.path.join(folder,"fold-"+str(val_fold))
      val_sel = (self._bucket_idx == val_fold)
      train_sel = np.logical_not(val_sel)
      train_score, val_score = model.train([data[l][train_sel] for l in range(0,len(data))],
                                           [data[l][val_sel] for l in range(0,len(data))],
                                           epochs,batch_size,
                                           fold_folder,verbose=verbose,
                                           image_dpi=image_dpi,screen_dpi=screen_dpi,
                                           train_dataset_name=train_dataset_name)
      model.save(fold_folder)
      # Analyse result of fold
      for k in train_score.keys():
        if k not in train_res:
          train_res[k] = []
        train_res[k].append(train_score[k])
        if k not in val_res:
          val_res[k] = []
        val_res[k].append(val_score[k])
      _, info, err = analyse_model(model, data[0][train_sel], data[-1][train_sel], fold_folder,
                                   verbose=verbose, prefix='train', image_dpi=image_dpi, screen_dpi=screen_dpi)
      if err is not None:
        train_res['error'][val_fold] = err
      else:
        has_error = False # Should be the same across all calls, but set it each time anyway
      _, info, err = analyse_model(model, data[0][val_sel], data[-1][val_sel], fold_folder,
                                   verbose=verbose, prefix='validation', image_dpi=image_dpi, screen_dpi=screen_dpi)
      for dpi in image_dpi:
        if os.path.exists(os.path.join(fold_folder,"architecture@"+str(dpi)+".png")):
          if val_fold == 0:
            os.rename(os.path.join(fold_folder,"architecture@"+str(dpi)+".png"),
                      os.path.join(folder,"architecture@"+str(dpi)+".png"))
          else:
            os.remove(os.path.join(fold_folder,"architecture@"+str(dpi)+".png"))
      model.reset()

    # Pairwise Wasserstein distance between validation error distributions
    if has_error:
      if verbose > 0:
        print("# Wasserstein distance between fold error distributions")
      max_wd_err = 0.0
      max_wd_aerr = 0.0
      for k1 in range(1,self.k):
        # Compare distributions
        for k2 in range(0,k1):
          if model.model[0:3] == 'ae_' and Cfg.val['analysis_spectra_error_dist_sampling'] < 100:
            # Sample distributions for large number of values, depending on cfg.
            l = len(val_res['error'][k1])
            sel1 = np.random.randint(0,l,size=l*Cfg.val['analysis_spectra_error_dist_sampling']//100)
            sel2 = np.random.randint(0,l,size=l*Cfg.val['analysis_spectra_error_dist_sampling']//100)
            wd = wasserstein_distance(val_res['error'][k1][sel1],val_res['error'][k2][sel2])
            wda = wasserstein_distance(np.abs(val_res['error'][k1][sel1]), np.abs(val_res['error'][k2][sel2]))
          else:
            wd = wasserstein_distance(val_res['error'][k1],val_res['error'][k2])
            wda = wasserstein_distance(np.abs(val_res['error'][k1]), np.abs(val_res['error'][k2]))
          if verbose > 1:
            print(f"    {k1} - {k2} = {wd}")
            print(f"    |{k1}| - |{k2}| = {wda}")
          if wd > max_wd_err:
            max_wd_err = wd
          if wda > max_wd_aerr:
            max_wd_aerr = wda
      if verbose > 0:
        print(f"  Max 1st order Wasserstein distance between validation error: {max_wd_err}")
        print(f"  Max 1st order Wasserstein distance between absolute validation error: {max_wd_aerr}")
    else:
      max_wd_err = None
      max_wd_aerr = None

    # Plot cross-validation results
    self._plot_cross_validate(model, train_res, val_res, has_error, folder, verbose, image_dpi, screen_dpi)

    # Save cross-validation result
    if has_error:
      del train_res['error']
      del val_res['error']
    with open(os.path.join(folder, "cv_result.json"), 'w') as f:
      print(json.dumps({
          'folds': self.k,
          'validation': val_res,
          'train': train_res,
          'max_wasserstein_distance_error': max_wd_err,
          'max_wasserstein_distance_absolute_error': max_wd_aerr,
        }, indent=2, sort_keys=True), file=f)

  def _plot_cross_validate(self, model, train_res, val_res, has_error, folder, verbose, image_dpi, screen_dpi):
    # Plot cross validation results

    # Error distributions
    if has_error:
      err_min = np.min([np.min([np.min(train_res['error'][l]),np.min(val_res['error'][l])])
                        for l in range(0,self.k)])
      err_max = np.max([np.max([np.max(train_res['error'][l]),np.max(val_res['error'][l])])
                        for l in range(0,self.k)])
      aerr_min = np.min([np.min([np.min(np.abs(train_res['error'][l])),np.min(np.abs(val_res['error'][l]))])
                         for l in range(0,self.k)])
      aerr_max = np.max([np.max([np.max(np.abs(train_res['error'][l])),np.max(np.abs(val_res['error'][l]))])
                         for l in range(0,self.k)])
      err_d = (err_max-err_min)*0.025
      err_min -= err_d
      err_max += err_d
      err_d = (aerr_max-aerr_min)*0.025
      aerr_min -= err_d
      aerr_max += err_d

    import matplotlib.gridspec as gridspec
    fig = plt.figure()
    if has_error:
      gs = gridspec.GridSpec(ncols=4,nrows=3,figure=fig,width_ratios=[0.499,0.001,0.499,0.001])
      ax00 = fig.add_subplot(gs[0,0:2])
      ax01 = fig.add_subplot(gs[0,2:])
      ax10 = fig.add_subplot(gs[1,0:2])
      ax11 = fig.add_subplot(gs[1,2:])
      ax20 = fig.add_subplot(gs[2,0:1])
      ax21 = fig.add_subplot(gs[2,2:3])
    else:
      gs = gridspec.GridSpec(ncols=4,nrows=1,figure=fig,width_ratios=[0.499,0.001,0.499,0.001])
      ax20 = fig.add_subplot(gs[0,0:1])
      ax21 = fig.add_subplot(gs[0,2:3])

    if has_error:
      ax00.set_title(f"Train Error Distributions")
      if model.model[0:3] == 'ae_' and Cfg.val['analysis_spectra_error_dist_sampling'] < 100:
        # Sample distributions for large number of values, depending on cfg.
        sns.boxplot(data=[train_res['error'][k][np.random.randint(0,len(train_res['error'][k]),
                                                                  size=len(train_res['error'][k])
                                                                       *Cfg.val['analysis_spectra_error_dist_sampling']//100)]
                          for k in range(0,self.k)], ax=ax00)
      else:
        sns.boxplot(data=train_res['error'], ax=ax00)
      ax00.set_ylim([err_min,err_max])
      ax00.set_xlabel("Fold")
      ax00.set_ylabel("Error")

      ax01.set_title(f"Validation Error Distributions")
      if model.model[0:3] == 'ae_' and Cfg.val['analysis_spectra_error_dist_sampling'] < 100:
        # Sample distributions for large number of values, depending on cfg.
        sns.boxplot(data=[val_res['error'][k][np.random.randint(0,len(val_res['error'][k]),
                                                                size=len(val_res['error'][k])
                                                                     *Cfg.val['analysis_spectra_error_dist_sampling']//100)]
                          for k in range(0,self.k)], ax=ax01)
      else:
        sns.boxplot(data=val_res['error'], ax=ax01)
      ax01.set_ylim([err_min,err_max])
      ax01.set_xlabel("Fold")
      ax01.set_ylabel("Error")

      ax10.set_title("Train Absolute Error Distributions")
      if model.model[0:3] == 'ae_' and Cfg.val['analysis_spectra_error_dist_sampling'] < 100:
        # Sample distributions for large number of values, depending on cfg.
        sns.boxplot(data=[np.abs(train_res['error'][k][np.random.randint(0,len(train_res['error'][k]),
                                                                         size=len(train_res['error'][k])
                                                                              *Cfg.val['analysis_spectra_error_dist_sampling']//100)])
                          for k in range(0,self.k)], ax=ax10)
      else:
        sns.boxplot(data=[np.abs(train_res['error'][k]) for k in range(0,self.k)],
                    ax=ax10)
      ax10.set_ylim([aerr_min,aerr_max])
      ax10.set_xlabel("Fold")
      ax10.set_ylabel("Abs. Error")

      ax11.set_title("Validation Absolute Error Distributions")
      if model.model[0:3] == 'ae_' and Cfg.val['analysis_spectra_error_dist_sampling'] < 100:
        # Sample distributions for large number of values, depending on cfg.
        sns.boxplot(data=[np.abs(np.array(val_res['error'][k][np.random.randint(0,len(val_res['error'][k]),
                                                                                size=len(val_res['error'][k])
                                                                                     *Cfg.val['analysis_spectra_error_dist_sampling']//100)]))
                          for k in range(0,self.k)], ax=ax11)
      else:
        sns.boxplot(data=[np.abs(np.array(val_res['error'][k])) for k in range(0,self.k)],
                    ax=ax11)
      ax11.set_ylim([aerr_min,aerr_max])
      ax11.set_ylabel("Abs. Error")
      ax11.set_xlabel("Fold")

    ax20.set_title("Train Metrics")
    keys = [k for k in sorted(train_res.keys())]
    ymin2x = []
    keys.remove("error")
    ymax2x = []
    cols = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown',
            'tab:pink','tab:gray','tab:olive','tab:cyan']
    for l in range(0,len(keys)):
      ymin2x.append(np.min([np.min(train_res[keys[l]]),np.min(val_res[keys[l]])]))
      ymax2x.append(np.max([np.max(train_res[keys[l]]),np.max(val_res[keys[l]])]))
      if l == 0:
        ax20.plot(range(0,self.k), train_res[keys[l]], color=cols[l])
        ax20.set_ylim([ymin2x[l],ymax2x[l]])
        ax20.set_ylabel(keys[0])
        ax20.tick_params(axis='y', labelcolor=cols[l])
      else:
        ax2 = ax20.twinx()
        ax2.plot(range(0,self.k), train_res[keys[l]], color=cols[l])
        ax2.set_ylim([ymin2x[l],ymax2x[l]])
        ax2.set_ylabel(keys[l])
        ax2.tick_params(axis='y', labelcolor=cols[l])
        ax2.spines.right.set_position(("axes", 0.8+(0.2*l)))
    ax20.set_xlabel("Fold")

    ax21.set_title("Validation Metrics")
    for l in range(0,len(keys)):
      if l == 0:
        ax21.plot(range(0,self.k), train_res[keys[l]], color=cols[l])
        ax21.set_ylim([ymin2x[l],ymax2x[l]])
        ax21.set_ylabel(keys[0])
        ax21.tick_params(axis='y', labelcolor=cols[l])
      else:
        ax2 = ax21.twinx()
        ax2.plot(range(0,self.k), train_res[keys[l]], color=cols[l])
        ax2.set_ylim([ymin2x[l],ymax2x[l]])
        ax2.set_ylabel(keys[l])
        ax2.tick_params(axis='y', labelcolor=cols[l])
        ax2.spines.right.set_position(("axes", 0.8+(0.2*l)))
    ax21.set_xlabel("Fold")

    fig.tight_layout()

    for f in glob.glob(os.path.join(folder, 'folds@*.png')):
      os.remove(f)
    for dpi in image_dpi:
      plt.savefig(os.path.join(folder, 'folds@'+str(dpi)+'.png'), dpi=dpi)
    if verbose > 1:
      fig.set_dpi(screen_dpi)
      plt.show(block=True)
    plt.close()

class NoValidation(Train):
  def __init__(self):
    Train.__init__(self,1)

  def train(self, model, data, epochs, batch_size, path_model, train_dataset_name="",
            image_dpi=[300], screen_dpi=96, shuffle=True, verbose=0):
    if verbose > 0:
      print("# No Validation")
    folder = get_folder(os.path.join(path_model,str(model),str(batch_size),str(epochs),
                                     train_dataset_name.replace("/","_")),
                        "NoValidation-%s")
    # Plot output distribution
    self._plot_distributions(data[-1], folder, image_dpi, screen_dpi, verbose)
    # Train
    model.train(data,None,epochs,batch_size,
                folder,verbose=verbose,image_dpi=image_dpi,screen_dpi=screen_dpi,
                train_dataset_name=train_dataset_name)
    model.save(folder)
    # Analyse model with training/test datasets
    analyse_model(model, data[0], data[-1], folder,
                  verbose=verbose, prefix='train', image_dpi=image_dpi, screen_dpi=screen_dpi)

class Split(Train):
  def __init__(self,p):
    Train.__init__(self,2)
    self.p = p

  def train(self, model, data, epochs, batch_size, path_model, train_dataset_name="",
            image_dpi=[300], screen_dpi=96, shuffle=True, verbose=0):
    # Split train/validation by percentage
    data_dim = data[0].shape[0]
    out_dim = data[-1].shape[-1]
    if verbose > 0:
      print(f"# Creating {self.p} split for {out_dim}-dimensional output data for {data_dim} inputs")
    idx = np.arange(0,data_dim)
    if shuffle:
      if verbose > 0:
        print("  Suffle data")
      rng = np.random.default_rng()
      rng.shuffle(idx)
    # Split
    split = np.round(data_dim * self.p).astype(np.int64)
    if verbose > 0:
      print(f"  Split {split} / {data_dim-split}")
    self._bucket_idx = np.ndarray(data_dim,dtype=np.int16)
    self._bucket_idx[idx[0:split]] = 1
    self._bucket_idx[idx[split:]] = 0
    if verbose > 1:
      print(f"    Bucket 0: {np.where(self._bucket_idx == 0)[0].shape[0]}")
      print(f"    Bucket 1: {np.where(self._bucket_idx == 1)[0].shape[0]}")
    folder = get_folder(os.path.join(path_model,str(model),str(batch_size),str(epochs),
                                     train_dataset_name.replace("/","_")),
                        "Split_"+str(self.p)+"-%s")
    # Plot distributions
    self._plot_distributions(data[-1], folder, image_dpi, screen_dpi, verbose)
    # Train
    val_sel = (self._bucket_idx == 0)
    train_sel = np.logical_not(val_sel)
    model.train([data[l][train_sel] for l in range(0,len(data))],
                [data[l][val_sel] for l in range(0,len(data))],
                epochs,batch_size,
                folder,verbose=verbose,image_dpi=image_dpi,screen_dpi=screen_dpi,
                train_dataset_name=train_dataset_name)
    model.save(folder)
    # Analyse model with training/test datasets
    # data = [d_noise, d_conc, d_clean]
    analyse_model(model, data[0][train_sel], data[-1][train_sel], folder,
                  verbose=verbose, prefix='train', image_dpi=image_dpi,screen_dpi=screen_dpi)
    analyse_model(model, data[0][val_sel], data[-1][val_sel], folder,
                  verbose=verbose, prefix='validation',
                  image_dpi=image_dpi,screen_dpi=screen_dpi)

class DuplexSplit(Train):
  def __init__(self,p):
    Train.__init__(self,2)
    self.p = p

  def train(self, model, data, epochs, batch_size, path_model, train_dataset_name="",
            image_dpi=[300], screen_dpi=96, shuffle=True, verbose=0):
    # Duplex split
    data_dim = data[0].shape[0]
    out_dim = data[-1].shape[-1]
    if verbose > 0:
      print(f"# Creating {self.p} duplex split for {out_dim}-dimensional output data for {data_dim} points")
    # Assign pairs to bucket in order of largest distance
    if verbose > 0:
      print("  Duplex split")
    from scipy.spatial.distance import pdist, squareform
    self._bucket_idx = -np.ones((data_dim),dtype=np.int64)
    if verbose > 0:
      print("    Pairwise distances")
    dm = squareform(pdist(data[-1],'mahalanobis'))
    dm[np.isnan(dm)] = -1.0
    if verbose > 0:
      print("    Init buckets")
    last_pt_idx = np.ndarray(2,dtype=np.int64)
    k_b = 0
    ddm = dm.copy()
    while k_b < 2:
      idx = np.where(ddm == np.amax(ddm))
      row_i = idx[0][0]
      col_i = idx[1][0]
      if ddm[row_i,col_i] > 0.0:
        self._bucket_idx[row_i] = k_b
        self._bucket_idx[col_i] = k_b
        last_pt_idx[k_b] = col_i
        if verbose > 3:
          print(f"      Pair for bucket {k_b}: {row_i}-{col_i} dist: {ddm[row_i,col_i]}; points left: {len(np.where(self._bucket_idx == -1)[0])}")
        dm[:,row_i] = -1.0
        dm[:,col_i] = -1.0
        ddm[:,row_i] = -1.0
        ddm[:,col_i] = -1.0
        ddm[row_i,:] = -1.0
        ddm[col_i,:] = -1.0
        k_b += 1
    del ddm
    if verbose > 0:
      print("    Adding points to buckets")
    k_b = 0
    n_selected_small = 2
    if self.p <= 0.5:
      bucket_small = 0
      n_expected_small = data_dim * self.p
    else:
      bucket_small = 1
      n_expected_small = data_dim * (1.0-self.p)
    while n_selected_small < n_expected_small:
      row_i = last_pt_idx[k_b]
      col_i = np.argmax(dm[row_i,:],axis=-1)
      if dm[row_i,col_i] > 0.0:
        if k_b == bucket_small:
          n_selected_small += 1
        self._bucket_idx[col_i] = k_b
        last_pt_idx[k_b] = col_i
        if verbose > 3:
          print(f"      Point for bucket {k_b}: {row_i}->{col_i} dist: {dm[row_i,col_i]}; points left: {len(np.where(self._bucket_idx == -1)[0])}")
        dm[:,col_i] = -1.0
        dm[row_i,:] = -1.0
        k_b = (k_b + 1) % 2
    sel = np.where(self._bucket_idx < 0)
    self._bucket_idx[sel] = (bucket_small + 1) % 2
    del dm
    if verbose > 0:
      print(f"  Split {np.where(self._bucket_idx==0)[0].shape[0]} / {np.where(self._bucket_idx==1)[0].shape[0]}")
    folder = get_folder(os.path.join(path_model,str(model),str(batch_size),str(epochs),
                                     train_dataset_name.replace("/","_")),
                        "DuplexSplit_"+str(self.p)+"-%s")
    # Plot distributions
    self._plot_distributions(data[-1], folder, image_dpi, screen_dpi, verbose)
    # Train
    val_sel = (self._bucket_idx == 1)
    train_sel = np.logical_not(val_sel)
    model.train([data[l][train_sel] for l in range(0,len(data))],
                [data[l][val_sel] for l in range(0,len(data))],
                epochs,batch_size,
                folder,verbose=verbose,image_dpi=image_dpi,screen_dpi=screen_dpi,
                train_dataset_name=train_dataset_name)
    model.save(folder)
    # Analyse model with training/test datasets
    analyse_model(model, data[0][train_sel], data[-1][train_sel], folder,
                  verbose=verbose, prefix='train', image_dpi=image_dpi,screen_dpi=screen_dpi)
    analyse_model(model, data[0][val_sel], data[-1][val_sel], folder,
                  verbose=verbose, prefix='validation', image_dpi=image_dpi,screen_dpi=screen_dpi)

class KFold(Train):
  def __init__(self,k):
    Train.__init__(self,k)

  def train(self, model, data, epochs, batch_size, path_model, train_dataset_name="",
            image_dpi=[300], screen_dpi=96, shuffle=True, verbose=0):
    # Stratify data into k folds
    data_dim = data[0].shape[0]
    out_dim = data[1].shape[-1]
    if verbose > 0:
      print(f"# Creating {self.k} folds for {out_dim}-dimensional output data for {data_dim} points")
    idx = np.arange(0,data_dim)
    if shuffle:
      if verbose > 0:
        print("  Suffle data")
      rng = np.random.default_rng()
      rng.shuffle(idx)
    # Venetian blinds splitting into buckets
    if verbose > 0:
      print("  Venetian blinds, step=1 mod k")
    self._bucket_idx = np.floor(idx % self.k).astype(np.int64)
    if verbose > 1:
      for b in range(0,self.k):
        print(f"    Bucket {b}: {np.where(self._bucket_idx == b)[0].shape[0]}")
    folder = get_folder(os.path.join(path_model,str(model),str(batch_size),str(epochs),
                                     train_dataset_name.replace("/","_")),
                        "KFold_"+str(self.k)+"-%s")
    # Plot distributions
    self._plot_distributions(data[-1], folder, image_dpi, screen_dpi, verbose)
    # Run cross validation
    self._cross_validate(model, epochs, batch_size, data, folder,
                         train_dataset_name, verbose, image_dpi, screen_dpi)

class DuplexKFold(Train):
  def __init__(self,k):
    Train.__init__(self,k)

  def train(self, model, data, epochs, batch_size, path_model, train_dataset_name="",
            image_dpi=[300], screen_dpi=96, shuffle=True, verbose=0):
    # Stratify data into k folds
    data_dim = data[0].shape[0]
    out_dim = data[-1].shape[-1]
    if verbose > 0:
      print(f"# Creating {self.k} folds for {out_dim}-dimensional output data for {data_dim} points")
    # Assign pairs to bucket in order of largest distance
    if verbose > 0:
      print("  k-Duplex splitting")
    from scipy.spatial.distance import pdist, squareform
    self._bucket_idx = -np.ones((data_dim),dtype=np.int64)
    if verbose > 0:
      print("    Pairwise distances")
    dm = squareform(pdist(data[-1],'mahalanobis'))
    dm[np.isnan(dm)] = -1.0
    if verbose > 0:
      print("    Init buckets")
    n_selected = 0
    last_pt_idx = np.ndarray((self.k),dtype=np.int64)
    k_b = 0
    ddm = dm.copy()
    while k_b < self.k:
      idx = np.where(ddm == np.amax(ddm))
      row_i = idx[0][0]
      col_i = idx[1][0]
      if ddm[row_i,col_i] > 0.0:
        n_selected += 2
        self._bucket_idx[row_i] = k_b
        self._bucket_idx[col_i] = k_b
        last_pt_idx[k_b] = col_i
        if verbose > 3:
          print(f"      Pair for bucket {k_b}: {row_i}-{col_i} dist: {ddm[row_i,col_i]}; points left: {len(np.where(self._bucket_idx == -1)[0])}")
        dm[:,row_i] = -1.0
        dm[:,col_i] = -1.0
        ddm[:,row_i] = -1.0
        ddm[:,col_i] = -1.0
        ddm[row_i,:] = -1.0
        ddm[col_i,:] = -1.0
        if n_selected > data_dim:
          raise RuntimeError("Insufficient data-points for finding buckets")
        k_b += 1
    del ddm
    if verbose > 2:
      print("      Selected:", n_selected)
    if verbose > 0:
      print("    Adding points to buckets")
    k_b = 0
    while n_selected < data_dim:
      row_i = last_pt_idx[k_b]
      col_i = np.argmax(dm[row_i,:],axis=-1)
      if dm[row_i,col_i] > 0.0:
        n_selected += 1
        self._bucket_idx[col_i] = k_b
        last_pt_idx[k_b] = col_i
        if verbose > 3:
          print(f"      Point for bucket {k_b}: {row_i}->{col_i} dist: {dm[row_i,col_i]}; points left: {len(np.where(self._bucket_idx == -1)[0])}")
        dm[:,col_i] = -1.0
        dm[row_i,:] = -1.0
        k_b = (k_b + 1) % self.k
    del dm
    if verbose > 1:
      for b in range(0,self.k):
        print(f"    Bucket {b}: {np.where(self._bucket_idx == b)[0].shape[0]}")
    folder = get_folder(os.path.join(path_model,str(model),str(batch_size),str(epochs),
                                     train_dataset_name.replace("/","_")),
                        "DuplexKFold_"+str(self.k)+"-%s")
    # Plot distributions
    self._plot_distributions(data[-1], folder, image_dpi, screen_dpi, verbose)
    # Run cross validation
    self._cross_validate(model, epochs, batch_size, data, folder,
                         train_dataset_name, verbose, image_dpi, screen_dpi)
