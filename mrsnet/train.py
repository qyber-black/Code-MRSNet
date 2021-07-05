# mrsnet/selection.py - MRSNet - selection
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2019, Max Chandler, PhD student at Cardiff University
# Copyright (C) 2020-2021, Frank C Langbein <frank@langbein.org>, Cardiff University

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
from scipy.stats import wasserstein_distance

from .analyse import analyse_model

class Train:

  def __init__(self,k):
    self.k=k
    self._bucket_idx = []

  def _plot_distributions(self,d_out,folder,screen_dpi,no_show):
    # Plot distributions
    data_dim = d_out.shape[0]
    out_dim = d_out.shape[-1]
    if self.k > 1:
      fig, axes = plt.subplots(self.k+1,out_dim,figsize=(25.6, 14.4))
    else:
      fig, axes = plt.subplots(self.k,out_dim,figsize=(25.6, 14.4))
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
            axes[b+1,dim].set_xlabel("Output value %d" % dim)
          if dim == 0:
            axes[b+1,dim].set_ylabel("%d Buckets" % b)
    if not os.path.isdir(folder):
      os.makedirs(folder)
    plt.savefig(os.path.join(folder,"output-distribution@300.png"), dpi=300)
    if not no_show:
      fig.set_dpi(screen_dpi)
      plt.show(block=True)
    plt.close()

  def _cross_validate(self, model, epochs, batch_size, d_inp, d_out, folder,
                      train_dataset_name, verbose, no_show, screen_dpi):
    # Cross valildation
    train_res = { 'error': [None]*self.k, 'wdq': [None]*self.k }
    val_res = { 'error': [None]*self.k, 'wdq': [None]*self.k }
    for val_fold in range(0,self.k):
      if verbose > 0:
        print("# Fold %d of %d" % (val_fold, self.k))
      fold_folder = os.path.join(folder,"fold-"+str(val_fold))
      val_sel = (self._bucket_idx == val_fold)
      train_sel = np.logical_not(val_sel)
      train_score, val_score = model.train(d_inp[train_sel],d_out[train_sel],
                                           d_inp[val_sel],d_out[val_sel],
                                           epochs,batch_size,
                                           fold_folder,verbose=verbose,
                                           no_show=no_show,screen_dpi=screen_dpi,
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
      _, info, err = analyse_model(model, d_inp[train_sel], d_out[train_sel],
                                   fold_folder, no_show=no_show,
                                   verbose=verbose, prefix='train', screen_dpi=screen_dpi)

      train_res['wdq'][val_fold] = info['wasserstein_distance_quality']
      train_res['error'][val_fold] = err
      _, info, err = analyse_model(model, d_inp[val_sel], d_out[val_sel],
                                   fold_folder, no_show=no_show,
                                   verbose=verbose, prefix='validation', screen_dpi=screen_dpi)
      val_res['wdq'][val_fold] = info['wasserstein_distance_quality']
      val_res['error'][val_fold] = err
      if val_fold == 0:
        os.rename(os.path.join(fold_folder,"architecture@300.png"),
                  os.path.join(folder,"architectecture@300.png"))
      else:
        os.remove(os.path.join(fold_folder,"architecture@300.png"))
      model.reset()

    # Pairwise Wasserstein distance between validation error distributions
    if verbose > 0:
      print("# Wasserstein distance")
    max_wd_err = 0.0
    max_wd_aerr = 0.0
    for k1 in range(1,self.k):
      # Compare distributions
      for k2 in range(0,k1):
        wd = wasserstein_distance(val_res['error'][k1],val_res['error'][k2],
                                  np.ones(len(val_res['error'][k1])) / len(val_res['error'][k1]),
                                  np.ones(len(val_res['error'][k2])) / len(val_res['error'][k2]))
        if verbose > 1:
          print("    %d - %d = %f" % (k1,k2,wd))
        if wd > max_wd_err:
          max_wd_err = wd
        wd = wasserstein_distance(np.abs(val_res['error'][k1]), np.abs(val_res['error'][k2]),
                                  np.ones(len(val_res['error'][k1])) / len(val_res['error'][k1]),
                                  np.ones(len(val_res['error'][k2])) / len(val_res['error'][k2]))
        if verbose > 1:
          print("    |%d| - |%d| = %f" % (k1,k2,wd))
        if wd > max_wd_aerr:
          max_wd_aerr = wd
    if verbose > 0:
      print("  Max 1st order Wasserstein distance between error: %f" % max_wd_err)
      print("  Max 1st order Wasserstein distance betweeb absolute error: %f" % max_wd_aerr)

    # Plot corss-validation results
    self._plot_cross_validate(train_res, val_res, folder, no_show, screen_dpi)

    # Save cross-validation result
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

  def _plot_cross_validate(self, train_res, val_res, folder, no_show, screen_dpi):
    # Plot cross validation results
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

    fig, axes = plt.subplots(3,2,figsize=(25.6,14.4), sharex=True)

    axes[0,0].set_title("Train Error Distributions")
    sns.boxplot(data=train_res['error'], ax=axes[0,0])
    axes[0,0].set_ylim([err_min,err_max])
    axes[0,0].set_xlabel("Fold")
    axes[0,0].set_ylabel("Error")

    sns.boxplot(data=val_res['error'], ax=axes[0,1])
    axes[0,1].set_ylim([err_min,err_max])
    axes[0,1].set_title("Validation Error Distributions")
    axes[0,1].set_xlabel("Fold")
    axes[0,1].set_ylabel("Error")

    sns.boxplot(data=[np.abs(train_res['error'][k]) for k in range(0,self.k)],
                ax=axes[1,0])
    axes[1,0].set_ylim([aerr_min,aerr_max])
    axes[1,0].set_title("Train Absolute Error Distributions")
    axes[1,0].set_xlabel("Fold")
    axes[1,0].set_ylabel("Abs. Error")

    sns.boxplot(data=[np.abs(np.array(val_res['error'][k])).tolist() for k in range(0,self.k)],
                ax=axes[1,1])
    axes[1,1].set_ylim([aerr_min,aerr_max])
    axes[1,1].set_title("Validation Absolute Error Distributions")
    axes[1,1].set_ylabel("Abs. Error")
    axes[1,1].set_xlabel("Fold")

    axes[2,0].set_title("Train Metrics")
    keys = [k for k in sorted(train_res.keys())]
    ymin2x = []
    keys.remove("error")
    keys.remove("wdq") # Mostly like MAE, if distribution is sound, FIXME?
    ymax2x = []
    cols = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown',
            'tab:pink','tab:gray','tab:olive','tab:cyan']
    for l in range(0,len(keys)):
      ymin2x.append(np.min([np.min(train_res[keys[l]]),np.min(val_res[keys[l]])]))
      ymax2x.append(np.max([np.max(train_res[keys[l]]),np.max(val_res[keys[l]])]))
      if l == 0:
        axes[2,0].plot(range(0,self.k), train_res[keys[l]], color=cols[l])
        axes[2,0].set_ylim([ymin2x[l],ymax2x[l]])
        axes[2,0].set_ylabel(keys[0])
        axes[2,0].tick_params(axis='y', labelcolor=cols[l])
      else:
        ax2 = axes[2,0].twinx()
        ax2.plot(range(0,self.k), train_res[keys[l]], color=cols[l])
        ax2.set_ylim([ymin2x[l],ymax2x[l]])
        ax2.set_ylabel(keys[l])
        ax2.tick_params(axis='y', labelcolor=cols[l])
    axes[2,0].set_xlabel("Fold")

    axes[2,1].set_title("Validation Metrics")
    for l in range(0,len(keys)):
      if l == 0:
        axes[2,0].plot(range(0,self.k), train_res[keys[l]], color=cols[l])
        axes[2,1].set_ylim([ymin2x[l],ymax2x[l]])
        axes[2,1].set_ylabel(keys[0])
        axes[2,0].tick_params(axis='y', labelcolor=cols[l])
      else:
        ax2 = axes[2,1].twinx()
        ax2.plot(range(0,self.k), train_res[keys[l]], color=cols[l])
        ax2.set_ylim([ymin2x[l],ymax2x[l]])
        ax2.set_ylabel(keys[l])
        ax2.tick_params(axis='y', labelcolor=cols[l])
    axes[2,1].set_xlabel("Fold")

    plt.savefig(os.path.join(folder, 'folds@300.png'), dpi=300)
    if not no_show:
      fig.set_dpi(screen_dpi)
      plt.show(block=True)
    plt.close()

class NoValidation(Train):
  def __init__(self):
    Train.__init__(self,1)

  def train(self, model, d_inp, d_out, epochs, batch_size, folder, train_dataset_name="",
            screen_dpi=100, no_show=False,shuffle=True, verbose=False):
    # No validation
    data_dim = d_out.shape[0]
    out_dim = d_out.shape[-1]
    if verbose > 0:
      print("# No Validation")
    # Plot distribution
    self._plot_distributions(d_out, folder, screen_dpi, no_show)
    # Train
    model.train(d_inp,d_out,np.array([]),np.array([]),
                epochs,batch_size,
                folder,verbose=verbose,no_show=no_show,screen_dpi=screen_dpi,
                train_dataset_name=train_dataset_name)
    model.save(folder)
    # Analyse model with training/test datasets
    analyse_model(model, d_inp, d_out, folder, no_show=no_show,
                  verbose=verbose, prefix='train', screen_dpi=screen_dpi)

class Split(Train):
  def __init__(self,p):
    Train.__init__(self,2)
    self.p = p

  def train(self, model, d_inp, d_out, epochs, batch_size, folder, train_dataset_name="",
            screen_dpi=100, no_show=False,shuffle=True, verbose=False):
    # Split train/validation by percentage
    data_dim = d_out.shape[0]
    out_dim = d_out.shape[-1]
    if verbose > 0:
      print("# Creating %f split for %d-dimensional output data for %d points" % (self.p,out_dim,data_dim))
    idx = np.arange(0,data_dim)
    if shuffle:
      if verbose > 0:
        print("  Suffle data")
      rng = np.random.default_rng()
      rng.shuffle(idx)
    # Split
    split = np.round(data_dim * self.p).astype(np.int64)
    if verbose > 0:
      print("  Split %d / %d" % (split,data_dim-split))
    self._bucket_idx = np.ndarray(data_dim,dtype=np.int16)
    self._bucket_idx[idx[0:split]] = 1
    self._bucket_idx[idx[split:]] = 0
    if verbose > 1:
      print("    Bucket 0: %d" % np.where(self._bucket_idx == 0)[0].shape[0])
      print("    Bucket 1: %d" % np.where(self._bucket_idx == 1)[0].shape[0])
    # Plot distributions
    self._plot_distributions(d_out, folder, screen_dpi, no_show)
    # Train
    val_sel = (self._bucket_idx == 0)
    train_sel = np.logical_not(val_sel)
    model.train(d_inp[train_sel],d_out[train_sel],d_inp[val_sel],d_out[val_sel],
                epochs,batch_size,
                folder,verbose=verbose,no_show=no_show,screen_dpi=screen_dpi,
                train_dataset_name=train_dataset_name)
    model.save(folder)
    # Analyse model with training/test datasets
    analyse_model(model, d_inp[train_sel], d_out[train_sel], folder, no_show=no_show,
                  verbose=verbose, prefix='train', screen_dpi=screen_dpi)
    analyse_model(model, d_inp[val_sel], d_out[val_sel], folder, no_show=no_show,
                  verbose=verbose, prefix='validation', screen_dpi=screen_dpi)

class DuplexSplit(Train):
  def __init__(self,p):
    Train.__init__(self,2)
    self.p = p

  def train(self, model, d_inp, d_out, epochs, batch_size, folder, train_dataset_name="",
            screen_dpi=100, no_show=False,shuffle=True, verbose=False):
    # Duplex split
    data_dim = d_out.shape[0]
    out_dim = d_out.shape[-1]
    if verbose > 0:
      print("# Creating %f duplex split for %d-dimensional output data for %d points" % (self.p,out_dim,data_dim))
    # Assign pairs to bucket in order of largest distance
    if verbose > 0:
      print("  Duplex split")
    from scipy.spatial.distance import pdist, squareform
    self._bucket_idx = -np.ones((data_dim),dtype=np.int64)
    if verbose > 0:
      print("    Pairwise distances")
    dm = squareform(pdist(d_out,'mahalanobis'))
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
        k_b += 1
        if verbose > 3:
          print("      Pair for",k_b,":", row_i, col_i, ddm[row_i,col_i], len(np.where(self._bucket_idx == -1)[0]))
        dm[:,row_i] = -1.0
        dm[:,col_i] = -1.0
        ddm[:,row_i] = -1.0
        ddm[:,col_i] = -1.0
        ddm[row_i,:] = -1.0
        ddm[col_i,:] = -1.0
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
          print("      Point for", kb,":", row_i, col_i, dm[row_i,col_i], len(np.where(self._bucket_idx == -1)[0]))
        dm[:,col_i] = -1.0
        dm[row_i,:] = -1.0
        k_b = (k_b + 1) % 2
    sel = np.where(self._bucket_idx < 0)
    self._bucket_idx[sel] = (bucket_small + 1) % 2
    del dm
    if verbose > 0:
      print("  Split %d / %d" % (np.where(self._bucket_idx==0)[0].shape[0], np.where(self._bucket_idx==1)[0].shape[0]))
    # Plot distributions
    self._plot_distributions(d_out, folder, screen_dpi, no_show)
    # Train
    val_sel = (self._bucket_idx == 1)
    train_sel = np.logical_not(val_sel)
    model.train(d_inp[train_sel],d_out[train_sel],d_inp[val_sel],d_out[val_sel],
                epochs,batch_size,
                folder,verbose=verbose,no_show=no_show,screen_dpi=screen_dpi,
                train_dataset_name=train_dataset_name)
    model.save(folder)
    # Analyse model with training/test datasets
    analyse_model(model, d_inp[train_sel], d_out[train_sel], folder, no_show=no_show,
                  verbose=verbose, prefix='train', screen_dpi=screen_dpi)
    analyse_model(model, d_inp[val_sel], d_out[val_sel], folder, no_show=no_show,
                  verbose=verbose, prefix='validation', screen_dpi=screen_dpi)

class KFold(Train):
  def __init__(self,k):
    Train.__init__(self,k)

  def train(self, model, d_inp, d_out, epochs, batch_size, folder, train_dataset_name="",
            screen_dpi=100, no_show=False,shuffle=True, verbose=False):
    # Stratify data into k folds
    data_dim = d_out.shape[0]
    out_dim = d_out.shape[-1]
    if verbose > 0:
      print("# Creating %d folds for %d-dimensional output data for %d points" % (self.k,out_dim,data_dim))
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
        print("    Bucket %d: %d" % (b,np.where(self._bucket_idx == b)[0].shape[0]))
    # Plot distributions
    self._plot_distributions(d_out, folder, screen_dpi, no_show)
    # Run cross validation
    self._cross_validate(model, epochs, batch_size, d_inp, d_out, folder,
                         train_dataset_name, verbose, no_show, screen_dpi)

class DuplexKFold(Train):
  def __init__(self,k):
    Train.__init__(self,k)

  def train(self, model, d_inp, d_out, epochs, batch_size, folder, train_dataset_name="",
            screen_dpi=100, no_show=False,shuffle=True, verbose=False):
    # Stratify data into k folds
    data_dim = d_out.shape[0]
    out_dim = d_out.shape[-1]
    if verbose > 0:
      print("# Creating %d folds for %d-dimensional output data for %d points" % (self.k,out_dim,data_dim))
    # Assign pairs to bucket in order of largest distance
    if verbose > 0:
      print("  k-Duplex splitting")
    from scipy.spatial.distance import pdist, squareform
    self._bucket_idx = -np.ones((data_dim),dtype=np.int64)
    if verbose > 0:
      print("    Pairwise distances")
    dm = squareform(pdist(d_out,'mahalanobis'))
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
        k_b += 1
        if verbose > 3:
          print("      Pair for",k_b,":", row_i, col_i, ddm[row_i,col_i], len(np.where(self._bucket_idx == -1)[0]))
        dm[:,row_i] = -1.0
        dm[:,col_i] = -1.0
        ddm[:,row_i] = -1.0
        ddm[:,col_i] = -1.0
        ddm[row_i,:] = -1.0
        ddm[col_i,:] = -1.0
        if n_selected >= data_dim:
          raise Exception("Insufficient data-points for finding buckets")
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
          print("      Point for", k_b,":", row_i, col_i, dm[row_i,col_i], len(np.where(self._bucket_idx == -1)[0]))
        k_b = (k_b + 1) % self.k
        dm[:,col_i] = -1.0
        dm[row_i,:] = -1.0
    del dm
    if verbose > 1:
      for b in range(0,self.k):
        print("    Bucket %d: %d" % (b,np.where(self._bucket_idx == b)[0].shape[0]))
    # Plot distributions
    self._plot_distributions(d_out, folder, screen_dpi, no_show)
    # Run cross validation
    self._cross_validate(model, epochs, batch_size, d_inp, d_out, folder,
                         train_dataset_name, verbose, no_show, screen_dpi)
