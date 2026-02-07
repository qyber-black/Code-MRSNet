# mrsnet/train.py - MRSNet - training
#
# SPDX-FileCopyrightText: Copyright (C) 2019 Max Chandler, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2020-2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2022-2024 Zien Ma, PhD student at Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Training utilities and validation strategies for MRSNet.

This module provides various training strategies including cross-validation,
data splitting, and model training utilities.
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from scipy.stats import kruskal, wasserstein_distance

from mrsnet.analyse import analyse_model
from mrsnet.cfg import Cfg
from mrsnet.getfolder import get_folder


def enable_deterministic_ops_if_configured(verbose=0):
  """Enable deterministic TF ops when configured in Cfg."""
  if Cfg.val.get('deterministic_ops', False):
    try:
      import tensorflow as _tf
      try:
        _tf.config.experimental.enable_op_determinism(True)
      except Exception:
        # Older TF versions
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    except Exception as e:
      if verbose > 0:
        print(f"WARNING: Could not enable deterministic ops: {e}")

class LrHistory(tf.keras.callbacks.Callback):
  """Callback to record the optimizer learning rate into logs/history."""

  def on_epoch_end(self, epoch, logs=None):
    """Record the current learning rate into the training logs.

    This callback method is called at the end of each training epoch
    to capture the optimizer's learning rate and add it to the logs
    for monitoring and analysis purposes.

    Parameters
    ----------
        epoch (int): The current epoch number (0-indexed)
        logs (dict, optional): Dictionary of logs from the training epoch.
            The learning rate will be added to this dictionary under the 'lr' key.
            Defaults to None
    """
    try:
      lr = self.model.optimizer.learning_rate
      try:
        lr = float(lr.numpy())
      except Exception:
        lr = float(tf.keras.backend.get_value(lr))
    except Exception:
      lr = None
    if logs is not None and lr is not None:
      logs['lr'] = lr

def set_auto_mixed_precision_policy_if_enabled(verbose=0):
  """Enable mixed precision and set policy automatically or from config.

  - Requires Cfg.mixed_precision to be True to activate
  - If Cfg.mixed_precision_auto_policy is True, selects policy from hardware
    (mixed_bfloat16 on CPU/TPU, mixed_float16 on GPU)
  - Otherwise uses Cfg.mixed_precision_policy
  """
  if not Cfg.val.get('mixed_precision', False):
    return
  try:
    import tensorflow as _tf
    from tensorflow.keras import mixed_precision as _mp
    if Cfg.val.get('mixed_precision_auto_policy', True):
      devs = _tf.config.list_physical_devices()
      names = [d.device_type.lower() for d in devs]
      policy = None
      # Prefer bfloat16 if CPU/TPU present
      if any('tpu' in n for n in names) or any('cpu' in n for n in names):
        policy = 'mixed_bfloat16'
      # Prefer float16 on GPUs
      if any('gpu' in n for n in names):
        policy = 'mixed_float16'
      if policy is not None:
        _mp.set_global_policy(policy)
    else:
      _mp.set_global_policy(Cfg.val.get('mixed_precision_policy', 'mixed_float16'))
  except Exception as e:
    if verbose > 0:
      print(f"WARNING: AMP policy selection failed: {e}")

def reshape_spectra_data(spectra_tensor, add_channel_dim=False):
    """Reshape spectra data from (batch, acquisition, datatype, frequency) to (batch, acquisition*datatype, frequency).

    This utility function standardizes the data reshaping used across all model types.

    Parameters
    ----------
        spectra_tensor (tensorflow.Tensor): Input spectra tensor with shape (batch, acquisition, datatype, frequency)
        add_channel_dim (bool, optional): Whether to add a channel dimension for CNN models. Defaults to False

    Returns
    -------
        tensorflow.Tensor: Reshaped tensor with shape (batch, acquisition*datatype, frequency) or (batch, acquisition*datatype, frequency, 1) if add_channel_dim=True
    """
    if add_channel_dim:
        return tf.reshape(spectra_tensor,
                         (spectra_tensor.shape[0],
                          spectra_tensor.shape[1] * spectra_tensor.shape[2],
                          spectra_tensor.shape[3], 1))
    else:
        return tf.reshape(spectra_tensor,
                         (spectra_tensor.shape[0],
                          spectra_tensor.shape[1] * spectra_tensor.shape[2],
                          spectra_tensor.shape[3]))


class Train:
  """Base class for training strategies.

  This class provides the foundation for various training strategies
  including cross-validation and data splitting.

  Attributes
  ----------
      k (int): Number of folds or buckets
      _bucket_idx (list): Bucket indices for data splitting
  """

  def __init__(self,k):
    """Initialize training strategy.

    Parameters
    ----------
        k (int): Number of folds or buckets
    """
    self.k=k
    self._bucket_idx = []
    # Generate and store a per-run seed for deterministic splits; this is
    # persisted with the model so runs can be reproduced.
    try:
      # Prefer high-quality seed from OS entropy via SeedSequence
      self.seed = int(np.random.SeedSequence().generate_state(1)[0])
    except Exception:
      # Fallback to default_rng
      self.seed = int(np.random.default_rng().integers(0, 2**31 - 1))
    # Derive a separate shuffle seed deterministically from split seed
    try:
      rng = np.random.default_rng(self.seed)
      self.shuffle_seed = int(rng.integers(0, 2**31 - 1))
    except Exception:
      self.shuffle_seed = int(np.random.default_rng().integers(0, 2**31 - 1))

  def _plot_distributions(self,d_out,folder,image_dpi,screen_dpi,verbose):
    """Plot output value distributions across buckets.

    Creates histograms showing the distribution of output values
    across different data buckets for visualization.

    Parameters
    ----------
        d_out (numpy.ndarray): Output data tensor
        folder (str): Folder to save plots
        image_dpi (int): DPI for saved images
        screen_dpi (int): DPI for screen display
        verbose (int): Verbosity level
    """
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
    """Perform k-fold cross-validation.

    Paramters
    ---------
        model: Model to train
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        data: Training data
        folder (str): Output folder for results
        train_dataset_name (str): Name of training dataset
        verbose (int): Verbosity level
        image_dpi (list): Image DPI settings
        screen_dpi (int): Screen DPI setting

    Returns
    -------
        tuple: (train_results, validation_results, has_error)
    """
    # Cross validation
    train_res = { 'error': [None]*self.k }
    val_res = { 'error': [None]*self.k }
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
      # Persist seed on the model for reproducibility in saved metadata
      try:
        model.seed = getattr(self, 'seed', None)
        model.shuffle_seed = getattr(self, 'shuffle_seed', None)
      except Exception:  # noqa: S110
        pass
      model.save(fold_folder)
      # Analyse result of fold
      for k in train_score.keys():
        if k not in train_res:
          train_res[k] = []
        train_res[k].append(train_score[k])
        if k not in val_res:
          val_res[k] = []
        val_res[k].append(val_score[k])
      # Check if clean spectra are available (different from noisy spectra)
      clean_spectra_train = None
      clean_spectra_val = None
      if len(data) == 3:
        clean_spectra_train = data[1][train_sel]
        clean_spectra_val = data[1][val_sel]
      _, info, err = analyse_model(model, data[0][train_sel], data[-1][train_sel], fold_folder, 'train',
                                   norm='max', verbose=verbose, image_dpi=image_dpi, screen_dpi=screen_dpi,
                                   clean_spectra=clean_spectra_train)
      train_res['error'][val_fold] = err
      _, info, err = analyse_model(model, data[0][val_sel], data[-1][val_sel], fold_folder, 'validation',
                                   norm='max', verbose=verbose, image_dpi=image_dpi, screen_dpi=screen_dpi,
                                   clean_spectra=clean_spectra_val)
      val_res['error'][val_fold] = err
      for dpi in image_dpi:
        if os.path.exists(os.path.join(fold_folder,"architecture@"+str(dpi)+".png")):
          if val_fold == 0:
            os.rename(os.path.join(fold_folder,"architecture@"+str(dpi)+".png"),
                      os.path.join(folder,"architecture@"+str(dpi)+".png"))
          else:
            os.remove(os.path.join(fold_folder,"architecture@"+str(dpi)+".png"))
      model.reset()

    # Pairwise Wasserstein distance between validation error distributions
    # Check availability of error distributions across all folds (should
    has_error = all(e is not None for e in train_res['error']) and all(e is not None for e in val_res['error'])
    if has_error:
      if verbose > 0:
        print("# Wasserstein distance between fold error distributions")
      max_wd_err = 0.0
      max_wd_aerr = 0.0
      sum_wd_err = 0.0
      sum_wd_aerr = 0.0
      cnt_wd_pairs = 0
      for k1 in range(1,self.k):
        # Compare distributions
        for k2 in range(0,k1):
          if model.model[0:3] == 'ae_' and Cfg.val['analysis_spectra_error_dist_sampling'] < 100:
            # Sample distributions for large number of values, depending on cfg.
            l = len(val_res['error'][k1])
            sel1 = np.random.randint(0,l,size=l*Cfg.val['analysis_spectra_error_dist_sampling']//100)
            sel2 = np.random.randint(0,l,size=l*Cfg.val['analysis_spectra_error_dist_sampling']//100)
            wd = wasserstein_distance(val_res['error'][k1][sel1].ravel(), val_res['error'][k2][sel2].ravel())
            wda = wasserstein_distance(np.abs(val_res['error'][k1][sel1]).ravel(), np.abs(val_res['error'][k2][sel2]).ravel())
          else:
            wd = wasserstein_distance(val_res['error'][k1].ravel(), val_res['error'][k2].ravel())
            wda = wasserstein_distance(np.abs(val_res['error'][k1]).ravel(), np.abs(val_res['error'][k2]).ravel())
          if verbose > 1:
            print(f"    {k1} - {k2} = {wd}")
            print(f"    |{k1}| - |{k2}| = {wda}")
          if wd > max_wd_err:
            max_wd_err = wd
          if wda > max_wd_aerr:
            max_wd_aerr = wda
          sum_wd_err += wd
          sum_wd_aerr += wda
          cnt_wd_pairs += 1
      mean_wd_err = (sum_wd_err / cnt_wd_pairs) if cnt_wd_pairs > 0 else None
      mean_wd_aerr = (sum_wd_aerr / cnt_wd_pairs) if cnt_wd_pairs > 0 else None

      # Kruskal-Wallis test across folds (validation errors)
      if model.model[0:3] == 'ae_' and Cfg.val['analysis_spectra_error_dist_sampling'] < 100:
        # Sample each fold consistently with above when very large
        signed_groups = []
        abs_groups = []
        for k in range(0,self.k):
          l = len(val_res['error'][k])
          sel = np.random.randint(0,l,size=l*Cfg.val['analysis_spectra_error_dist_sampling']//100)
          signed_groups.append(val_res['error'][k][sel].ravel())
          abs_groups.append(np.abs(val_res['error'][k][sel]).ravel())
      else:
        signed_groups = [val_res['error'][k].ravel() for k in range(0,self.k)]
        abs_groups = [np.abs(val_res['error'][k]).ravel() for k in range(0,self.k)]

      try:
        _, kw_p_err = kruskal(*signed_groups)
      except Exception:
        kw_p_err = None
      try:
        _, kw_p_aerr = kruskal(*abs_groups)
      except Exception:
        kw_p_aerr = None
      if verbose > 0:
        print(f"  Max 1st order Wasserstein distance between validation error: {max_wd_err}")
        print(f"  Mean 1st order Wasserstein distance between validation error: {mean_wd_err}")
        if kw_p_err is not None:
          print(f"  Kruskal-Wallis p-value (signed validation error): {kw_p_err}")
        print(f"  Max 1st order Wasserstein distance between absolute validation error: {max_wd_aerr}")
        print(f"  Mean 1st order Wasserstein distance between absolute validation error: {mean_wd_aerr}")
        if kw_p_aerr is not None:
          print(f"  Kruskal-Wallis p-value (absolute validation error): {kw_p_aerr}")
    else:
      max_wd_err = None
      max_wd_aerr = None
      mean_wd_err = None
      mean_wd_aerr = None
      kw_p_err = None
      kw_p_aerr = None

    # Plot cross-validation results
    self._plot_cross_validate(model, train_res, val_res, has_error, folder, verbose, image_dpi, screen_dpi)

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
          'mean_wasserstein_distance_error': mean_wd_err,
          'mean_wasserstein_distance_absolute_error': mean_wd_aerr,
          'kruskal_pvalue_error': kw_p_err,
          'kruskal_pvalue_absolute_error': kw_p_aerr,
        }, indent=2, sort_keys=True), file=f)

  def _plot_cross_validate(self, model, train_res, val_res, has_error, folder, verbose, image_dpi, screen_dpi):
    """Plot cross-validation results.

    Parameters
    ----------
        model: Trained model
        train_res (dict): Training results
        val_res (dict): Validation results
        has_error (bool): Whether error distributions are available
        folder (str): Output folder for plots
        verbose (int): Verbosity level
        image_dpi (list): Image DPI settings
        screen_dpi (int): Screen DPI setting
    """
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
      ax00.set_title("Train Error Distributions")
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

      ax01.set_title("Validation Error Distributions")
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
    keys = sorted(train_res.keys())
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
        ax21.plot(range(0,self.k), val_res[keys[l]], color=cols[l])
        ax21.set_ylim([ymin2x[l],ymax2x[l]])
        ax21.set_ylabel(keys[0])
        ax21.tick_params(axis='y', labelcolor=cols[l])
      else:
        ax2 = ax21.twinx()
        ax2.plot(range(0,self.k), val_res[keys[l]], color=cols[l])
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
  """Training without validation split.

  This class implements training without any validation split,
  using all available data for training.
  """

  def __init__(self):
    """Initialize no-validation trainer."""
    Train.__init__(self,1)

  def train(self, model, data, epochs, batch_size, path_model, train_dataset_name="",
            image_dpi=[300], screen_dpi=96, shuffle=True, verbose=0):
    """Train model without validation split.

    Parameters
    ----------
        model: Model to train
        data: Training data
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        path_model (str): Path for saving model
        train_dataset_name (str, optional): Name of training dataset. Defaults to ""
        image_dpi (list, optional): Image DPI settings. Defaults to [300]
        screen_dpi (int, optional): Screen DPI setting. Defaults to 96
        shuffle (bool, optional): Whether to shuffle data. Defaults to True
        verbose (int, optional): Verbosity level. Defaults to 0
    """
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
    # Persist seed on the model for reproducibility in saved metadata
    try:
      model.seed = getattr(self, 'seed', None)
      model.shuffle_seed = getattr(self, 'shuffle_seed', None)
    except Exception:  # noqa: S110
      pass
    model.save(folder)
    # Analyse model with training/test datasets
    # Check if clean spectra are available (different from noisy spectra)
    clean_spectra = None
    if len(data) == 3:
      clean_spectra = data[1]
    analyse_model(model, data[0], data[-1], folder, 'train',
                  norm='max', verbose=verbose, image_dpi=image_dpi, screen_dpi=screen_dpi,
                  clean_spectra=clean_spectra)

class Split(Train):
  """Train/validation split strategy.

  This class implements a simple train/validation split based on
  a specified percentage of data for validation.

  Attributes
  ----------
      p (float): Percentage of data to use for validation
  """

  def __init__(self,p):
    """Initialize split trainer.

    Parameters
    ----------
        p (float): Percentage of data to use for training (0-1)
    """
    Train.__init__(self,2)
    self.p = p

  def train(self, model, data, epochs, batch_size, path_model, train_dataset_name="",
            image_dpi=[300], screen_dpi=96, shuffle=True, verbose=0):
    """Train model with percentage-based split.

    Parameters
    ----------
        model: Model to train
        data: Training data
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        path_model (str): Path for saving model
        train_dataset_name (str, optional): Name of training dataset. Defaults to ""
        image_dpi (list, optional): Image DPI settings. Defaults to [300]
        screen_dpi (int, optional): Screen DPI setting. Defaults to 96
        shuffle (bool, optional): Whether to shuffle data. Defaults to True
        verbose (int, optional): Verbosity level. Defaults to 0
    """
    # Split train/validation by percentage
    data_dim = data[0].shape[0]
    out_dim = data[-1].shape[-1]
    if verbose > 0:
      print(f"# Creating {self.p} split for {out_dim}-dimensional output data for {data_dim} inputs")
    idx = np.arange(0,data_dim)
    if shuffle:
      if verbose > 0:
        print("  Shuffle data")
      # Deterministic shuffle using per-run seed
      rng = np.random.default_rng(getattr(self, 'seed', None))
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
    # Persist seed on the model for reproducibility in saved metadata
    try:
      model.seed = getattr(self, 'seed', None)
    except Exception:  # noqa: S110
      pass
    model.save(folder)
    # Analyse model with training/test datasets
    # data = [d_noise, d_clean, d_conc] - concentrations are always last
    # Check if clean spectra are available (different from noisy spectra)
    clean_spectra_train = None
    clean_spectra_val = None
    if len(data) == 3:
      clean_spectra_train = data[1][train_sel]
      clean_spectra_val = data[1][val_sel]
    analyse_model(model, data[0][train_sel], data[-1][train_sel], folder, 'train',
                  norm='max', verbose=verbose, image_dpi=image_dpi, screen_dpi=screen_dpi,
                  clean_spectra=clean_spectra_train)
    analyse_model(model, data[0][val_sel], data[-1][val_sel], folder, 'validation',
                  norm='max', verbose=verbose, image_dpi=image_dpi, screen_dpi=screen_dpi,
                  clean_spectra=clean_spectra_val)

class DuplexSplit(Train):
  """Duplex split strategy for train/validation splitting.

  This class implements duplex splitting, which uses distance-based
  selection to create representative train/validation splits.

  Attributes
  ----------
      p (float): Percentage of data to use for validation
  """

  def __init__(self,p):
    """Initialize duplex split trainer.

    Paramters
    ---------
        p (float): Percentage of data to use for validation (0-1)
    """
    Train.__init__(self,2)
    self.p = p

  def train(self, model, data, epochs, batch_size, path_model, train_dataset_name="",
            image_dpi=[300], screen_dpi=96, shuffle=True, verbose=0):
    """Train model with duplex split.

    Parameters
    ----------
        model: Model to train
        data: Training data
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        path_model (str): Path for saving model
        train_dataset_name (str, optional): Name of training dataset. Defaults to ""
        image_dpi (list, optional): Image DPI settings. Defaults to [300]
        screen_dpi (int, optional): Screen DPI setting. Defaults to 96
        shuffle (bool, optional): Whether to shuffle data. Defaults to True
        verbose (int, optional): Verbosity level. Defaults to 0
    """
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
    try:
      dm = squareform(pdist(data[-1],'mahalanobis'))
    except Exception as e:
      if verbose > 0:
        print("    WARNING: Mahalanobis distance failed; falling back to seeded random split:", e)
      dm = None
    if dm is not None:
      dm[~np.isfinite(dm)] = -1.0
      np.fill_diagonal(dm, -1.0)
    if verbose > 0:
      print("    Init buckets")
    last_pt_idx = np.ndarray(2,dtype=np.int64)
    k_b = 0
    success = True
    if dm is not None:
      ddm = dm.copy()
      # Select initial pairs for both buckets by farthest distance (allowing zero)
      guard = 0
      while k_b < 2:
        guard += 1
        if guard > data_dim * 4:
          success = False
          break
        max_val = np.max(ddm)
        if max_val < 0:
          success = False
          break
        pairs = np.argwhere(ddm == max_val)
        # Pick first off-diagonal pair
        row_i, col_i = -1, -1
        for r, c in pairs:
          if r != c:
            row_i, col_i = int(r), int(c)
            break
        if row_i < 0 or col_i < 0:
          success = False
          break
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
    if dm is not None and success:
      while n_selected_small < n_expected_small:
        row_i = int(last_pt_idx[k_b])
        # Candidate columns: not assigned, not the same row, and with non-negative distance
        candidates = np.where((self._bucket_idx < 0) & (dm[row_i,:] >= 0.0))[0]
        candidates = candidates[candidates != row_i]
        if candidates.size == 0:
          success = False
          break
        # Pick the farthest candidate from current row
        best_idx = candidates[np.argmax(dm[row_i, candidates])]
        col_i = int(best_idx)
        if k_b == bucket_small:
          n_selected_small += 1
        self._bucket_idx[col_i] = k_b
        last_pt_idx[k_b] = col_i
        if verbose > 3:
          print(f"      Point for bucket {k_b}: {row_i}->{col_i} dist: {dm[row_i,col_i]}; points left: {len(np.where(self._bucket_idx == -1)[0])}")
        dm[:,col_i] = -1.0
        dm[row_i,:] = -1.0
        k_b = (k_b + 1) % 2
    if not success or dm is None:
      # Seeded random fallback maintaining requested split ratio
      if verbose > 0:
        print("    WARNING: Falling back to seeded random split (duplex selection failed)")
      idx = np.arange(0, data_dim)
      if shuffle:
        rng = np.random.default_rng(getattr(self, 'seed', None))
        rng.shuffle(idx)
      n_small = int(np.round(n_expected_small))
      self._bucket_idx[idx[:n_small]] = bucket_small
      self._bucket_idx[idx[n_small:]] = (bucket_small + 1) % 2
    else:
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
    # Persist seed on the model for reproducibility in saved metadata
    try:
      model.seed = getattr(self, 'seed', None)
    except Exception:  # noqa: S110
      pass
    model.save(folder)
    # Analyse model with training/test datasets
    # data = [d_noise, d_clean, d_conc] - concentrations are always last
    # Check if clean spectra are available (different from noisy spectra)
    clean_spectra_train = None
    clean_spectra_val = None
    if len(data) == 3:
      clean_spectra_train = data[1][train_sel]
      clean_spectra_val = data[1][val_sel]
    analyse_model(model, data[0][train_sel], data[-1][train_sel], folder, 'train',
                  norm='max', verbose=verbose, image_dpi=image_dpi, screen_dpi=screen_dpi,
                  clean_spectra=clean_spectra_train)
    analyse_model(model, data[0][val_sel], data[-1][val_sel], folder, 'validation',
                  norm='max', verbose=verbose, image_dpi=image_dpi, screen_dpi=screen_dpi,
                  clean_spectra=clean_spectra_val)

class KFold(Train):
  """K-fold cross-validation strategy.

  This class implements k-fold cross-validation using a "Venetian blinds"
  approach for data splitting.
  """

  def __init__(self,k):
    """Initialize K-fold trainer.

    Parameters
    ----------
        k (int): Number of folds for cross-validation
    """
    Train.__init__(self,k)

  def train(self, model, data, epochs, batch_size, path_model, train_dataset_name="",
            image_dpi=[300], screen_dpi=96, shuffle=True, verbose=0):
    """Train model with K-fold cross-validation.

    Parameters
    ----------
        model: Model to train
        data: Training data
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        path_model (str): Path for saving model
        train_dataset_name (str, optional): Name of training dataset. Defaults to ""
        image_dpi (list, optional): Image DPI settings. Defaults to [300]
        screen_dpi (int, optional): Screen DPI setting. Defaults to 96
        shuffle (bool, optional): Whether to shuffle data. Defaults to True
        verbose (int, optional): Verbosity level. Defaults to 0
    """
    # Stratify data into k folds
    data_dim = data[0].shape[0]
    out_dim = data[-1].shape[-1]
    if verbose > 0:
      print(f"# Creating {self.k} folds for {out_dim}-dimensional output data for {data_dim} points")
    idx = np.arange(0,data_dim)
    if shuffle:
      if verbose > 0:
        print("  Shuffle data")
      # Deterministic shuffle using per-run seed
      rng = np.random.default_rng(getattr(self, 'seed', None))
      rng.shuffle(idx)
    # Venetian blinds splitting into buckets
    if verbose > 0:
      print("  Venetian blinds, step=1 mod k (respecting shuffled order)")
    # Assign folds round-robin in the (possibly shuffled) order
    self._bucket_idx = np.empty(data_dim, dtype=np.int64)
    self._bucket_idx[idx] = (np.arange(data_dim) % self.k).astype(np.int64)
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
  """Duplex K-fold cross-validation strategy.

  This class implements duplex K-fold cross-validation, which uses
  distance-based selection to create representative folds.
  """

  def __init__(self,k):
    """Initialize duplex K-fold trainer.

    Parameters
    ----------
        k (int): Number of folds for cross-validation
    """
    Train.__init__(self,k)

  def train(self, model, data, epochs, batch_size, path_model, train_dataset_name="",
            image_dpi=[300], screen_dpi=96, shuffle=True, verbose=0):
    """Train model with duplex K-fold cross-validation.

    Parameters
    ----------
        model: Model to train
        data: Training data
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        path_model (str): Path for saving model
        train_dataset_name (str, optional): Name of training dataset. Defaults to ""
        image_dpi (list, optional): Image DPI settings. Defaults to [300]
        screen_dpi (int, optional): Screen DPI setting. Defaults to 96
        shuffle (bool, optional): Whether to shuffle data. Defaults to True
        verbose (int, optional): Verbosity level. Defaults to 0
    """
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
    try:
      dm = squareform(pdist(data[-1],'mahalanobis'))
    except Exception as e:
      if verbose > 0:
        print("    WARNING: Mahalanobis distance failed; falling back to seeded random k-fold:", e)
      dm = None
    if dm is not None:
      dm[~np.isfinite(dm)] = -1.0
      np.fill_diagonal(dm, -1.0)
    if verbose > 0:
      print("    Init buckets")
    n_selected = 0
    last_pt_idx = np.ndarray((self.k),dtype=np.int64)
    k_b = 0
    success = True
    if dm is not None:
      ddm = dm.copy()
      guard = 0
      while k_b < self.k:
        guard += 1
        if guard > data_dim * 4:
          success = False
          break
        max_val = np.max(ddm)
        if max_val < 0:
          success = False
          break
        pairs = np.argwhere(ddm == max_val)
        row_i, col_i = -1, -1
        for r, c in pairs:
          if r != c:
            row_i, col_i = int(r), int(c)
            break
        if row_i < 0 or col_i < 0:
          success = False
          break
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
          success = False
          break
        k_b += 1
      del ddm
    if verbose > 2:
      print("      Selected:", n_selected)
    if verbose > 0:
      print("    Adding points to buckets")
    k_b = 0
    if dm is not None and success:
      while n_selected < data_dim:
        row_i = int(last_pt_idx[k_b])
        candidates = np.where((self._bucket_idx < 0) & (dm[row_i,:] >= 0.0))[0]
        candidates = candidates[candidates != row_i]
        if candidates.size == 0:
          success = False
          break
        best_idx = candidates[np.argmax(dm[row_i, candidates])]
        col_i = int(best_idx)
        n_selected += 1
        self._bucket_idx[col_i] = k_b
        last_pt_idx[k_b] = col_i
        if verbose > 3:
          print(f"      Point for bucket {k_b}: {row_i}->{col_i} dist: {dm[row_i,col_i]}; points left: {len(np.where(self._bucket_idx == -1)[0])}")
        dm[:,col_i] = -1.0
        dm[row_i,:] = -1.0
        k_b = (k_b + 1) % self.k
      del dm
    if not success or dm is None:
      # Seeded random k-fold fallback respecting shuffled order
      if verbose > 0:
        print("    WARNING: Falling back to seeded random k-fold (duplex selection failed)")
      idx = np.arange(0, data_dim)
      if shuffle:
        rng = np.random.default_rng(getattr(self, 'seed', None))
        rng.shuffle(idx)
      self._bucket_idx[idx] = (np.arange(data_dim) % self.k).astype(np.int64)
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


def calculate_flops(model, input_shape):
  """Calculate FLOPs (Floating Point Operations) for the model.

  Parameters
  ----------
      model: Model to calculate FLOPs for
      input_shape (tuple): Input tensor shape (excluding batch dimension)

  Returns
  -------
      int: Total number of FLOPs, or 0 if calculation fails
  """
  if model is None:
    return 0

  import traceback

  import tensorflow as tf

  try:
    # Trace the model to obtain a concrete graph function
    @tf.function(jit_compile=False)
    def model_fn(x):
      return model(x, training=False)

    concrete_fn = model_fn.get_concrete_function(
      tf.TensorSpec(shape=(1, *input_shape), dtype=tf.float32)
    )

    # FIXME: tensorflow/python/ops/nn_ops.py:5261: tensor_shape_from_node_def_name (from tensorflow.python.framework.graph_util_impl) is deprecated and will be removed in a future version. Instructions for updating: This API was designed for TensorFlow v1. See https://www.tensorflow.org/guide/migrate for instructi ons on how to migrate your code to TensorFlow v2.

    # Primary: use TF compat v1 profiler directly on the traced graph
    try:
      opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
      prof = tf.compat.v1.profiler.profile(
        graph=concrete_fn.graph,
        options=opts
      )
      if prof is not None and hasattr(prof, 'total_float_ops') and prof.total_float_ops is not None:
        return int(prof.total_float_ops)
    except Exception as e:
      print("FLOPs primary profiler failed:", e)

    # Fallback: freeze variables to constants and profile the imported graph
    try:
      from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
      frozen = convert_variables_to_constants_v2(concrete_fn)
      graph_def = frozen.graph.as_graph_def()
      with tf.Graph().as_default() as graph:
        tf.compat.v1.graph_util.import_graph_def(graph_def, name="")
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        prof = tf.compat.v1.profiler.profile(
          graph=graph,
          options=opts
        )
        if prof is not None and hasattr(prof, 'total_float_ops') and prof.total_float_ops is not None:
          return int(prof.total_float_ops)
    except Exception as e:
      print("FLOPs frozen-graph profiler failed:", e)

    # Optional: try TensorFlow Model Garden utility if available
    try:
      import importlib
      if importlib.util.find_spec('tfm') is not None:
        from tfm.core.train_utils import try_count_flops
        dummy_input = tf.random.normal((1, *input_shape))
        counted = try_count_flops(model_fn, dummy_input)
        if counted is not None:
          return int(counted)
    except Exception as e:
      print("FLOPs Model Garden fallback failed:", e)

  except Exception as e:
    print("Error calculating FLOPs:", e)
    traceback.print_exc()

  # If all methods fail, return 0
  return 0
