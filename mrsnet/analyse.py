# mrsnet/analyse.py - MRSNet - analyse model performance
#
# SPDX-FileCopyrightText: Copyright (C) 2019 Max Chandler, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2020-2022 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import glob
import warnings
import numpy as np
import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

from mrsnet.cfg import Cfg

def analyse_model(model, inp, out, folder, prefix, id=None, save_conc=False, show_conc=False,
                  verbose=0, image_dpi=[300], screen_dpi=96):
  pred_op = getattr(model, "predict", None)
  if not callable(pred_op):
    if verbose > 0:
      print("# Warning, model cannot be analysed as it has no prediction method")
    return None, None, None
  # Analyse data (assumed to be exported from dataset in format as used by model for train call)
  if not os.path.exists(folder):
    os.makedirs(folder)

  pre = model.predict(inp,verbose=verbose)

  if model.model[0:3] == "ae_":
    # Analyse output spectra and errors, if possible, as we have a pure autoencoder
    # FIXME: have to check later if the autoencoder produceS concentrations via the model string and then use the concentration analysis instead (code after this condition)
    return _analyse_spectra_error(model, pre, inp, out, folder, prefix, id, verbose, image_dpi, screen_dpi)

  # Analyse if we have concentrations
  if len(out) > 0:
    info, error = _analyse_model_error(model, pre, inp, out, folder, prefix, verbose, image_dpi, screen_dpi)
  else:
    info = None
    error = None
  # Store/print quantification results
  if save_conc:
    with open(os.path.join(folder, prefix+'_quantify.csv'), "w") as out_file:
      writer = csv.writer(out_file, delimiter=",")
      writer.writerows([[str(model).upper()+" Quantification Results"],
                        ["Metabolites", *model.metabolites],
                        ["Pulse Sequence", model.pulse_sequence],
                        [""]])
      if len(out) > 0:
        writer.writerow(['Metabolite', 'Predicted', 'Actual', 'Error'])
      else:
        writer.writerow(['Metabolite', 'Predicted'])
      for l in range(0,inp.shape[0]):
        if id is None:
          writer.writerow([f"Spectrum: {l}"])
        else:
          writer.writerow([f"Spectrum: {id[l]}"])
        if len(out) > 0:
          for k,m in enumerate(model.metabolites):
            writer.writerow([m, pre[l,k], out[l,k], pre[l,k]-out[l,k]])
        else:
          for k,m in enumerate(model.metabolites):
            writer.writerow([m, pre[l,k]])
  if show_conc:
    print(f"\n# {str(model).upper()} Quantification Results")
    print("Metabolites: "+", ".join(model.metabolites))
    print("Pulse Sequence: "+model.pulse_sequence)
    print('\n                       Concentrations')
    if len(out) > 0:
      print('  %12s  %8s    %8s  %8s' % ('Metabolite', 'Predicted', 'Actual', 'Error'))
    else:
      print('  %12s  %8s' % ('Metabolite', 'Predicted'))
    for l in range(0,inp.shape[0]):
      if id is None:
        print(f"Spectrum: {l}")
      else:
        print(f"Spectrum: {id[l]}")
      if len(out) > 0:
        for k,m in enumerate(model.metabolites):
          print('  %12s  %.8f    %.8f  %.8f' % (m, pre[l,k], out[l,k], pre[l,k]-out[l,k]))
      else:
        for k,m in enumerate(model.metabolites):
          print('  %12s  %.8f' % (m, pre[l,k]))

  return pre, info, error

def _analyse_model_error(model, pre, inp, out, folder, prefix, verbose, image_dpi, screen_dpi):
  error = pre - out
  error_mean = np.mean(error,axis=0)
  error_std = np.std(error,axis=0)
  error_min = np.min(error,axis=0)
  error_max = np.max(error,axis=0)
  abserror = np.abs(error)
  abserror_mean = np.mean(abserror,axis=0)
  abserror_std = np.std(abserror,axis=0)
  abserror_min = np.min(abserror,axis=0)
  abserror_max = np.max(abserror,axis=0)
  info = { 'prefix': prefix }

  # Per metabolite plots/data
  fig, axes =  plt.subplots(2,len(model.metabolites)+1)
  fig.suptitle(f"Concentration Error Analysis ({prefix})")
  for l,m in enumerate(model.metabolites):
    with warnings.catch_warnings():
      warnings.filterwarnings('ignore')
      # ignores division by zero warnings, which will happen in the analysis
      slope, intercept, r_value, p_value, std_err = linregress(out[:,l], pre[:,l])
    info[m] = {
        'error': {
          'mean': error_mean[l],
          'std': error_std[l],
          'min': error_min[l],
          'max': error_max[l],
        },
        'abserror': {
          'mean': abserror_mean[l],
          'std': abserror_std[l],
          'min': abserror_min[l],
          'max': abserror_max[l],
        },
        'linreg': {
          'slope': slope,
          'intercept': intercept,
          'r_value': r_value,
          'p_value': p_value,
          'std_err': std_err
        }
      }
    axes[0,l].plot([0, 1], [0, 1], label='true line')
    sns.regplot(y=pre[:,l], x=out[:,l], ax=axes[0,l])
    axes[0,l].set_title(m)
    axes[0,l].set_xlabel("True")
    if l == 0:
      axes[0,l].set_ylabel("Predicted")
    if l == len(model.metabolites)//2:
      axes[0,l].set_title("True vs. Predicted\n\n"+m)
    axes[0,l].set_xlim([0,1])
    axes[0,l].set_ylim([0,1])

    sns.histplot(error[:,l], kde=True, ax=axes[1,l])
    axes[1,l].set_xlabel("Error")
    if l == 0:
      axes[1,l].set_ylabel("Count")
    if l == len(model.metabolites)//2:
      axes[1,l].set_title("Error Distributions")
  info['true'] = out.tolist()
  info['predicted'] = pre.tolist()
  info['error'] = pre.tolist()

  # Total plots/data
  error = np.reshape(error,np.prod(error.shape))
  terror_mean = np.mean(error)
  terror_std = np.std(error)
  terror_min = np.min(error)
  terror_max = np.max(error)
  abserror = np.reshape(abserror,np.prod(abserror.shape))
  tabserror_mean = np.mean(abserror)
  tabserror_std = np.std(abserror)
  tabserror_min = np.min(abserror)
  tabserror_max = np.max(abserror)

  out_all = out.reshape(np.prod(out.shape))
  pre_all = pre.reshape(np.prod(pre.shape))
  with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    # ignores division by zero warnings, which will happen in the analysis
    slope, intercept, r_value, p_value, std_err = linregress(out_all,pre_all)
  info['total'] = {
      'error': {
        'mean': terror_mean,
        'std': terror_std,
        'min': terror_min,
        'max': terror_max,
      },
      'abserror': {
        'mean': tabserror_mean,
        'std': tabserror_std,
        'min': tabserror_min,
        'max': tabserror_max,
      },
      'linreg': {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err
      },
    }
  if verbose > 0:
    print(f"  Total mean absolute error ({prefix}): {info['total']['abserror']['mean']}")

  axes[0,len(model.metabolites)].plot([0, 1], [0, 1], label='true line')
  sns.regplot(y=pre_all, x=out_all, ax=axes[0,len(model.metabolites)])
  axes[0,len(model.metabolites)].set_title('Total')
  axes[0,len(model.metabolites)].set_xlabel("True")
  axes[0,len(model.metabolites)].set_xlim([0,1])
  axes[0,len(model.metabolites)].set_ylim([0,1])
  sns.histplot(error, kde=True, ax=axes[1,len(model.metabolites)])
  axes[1,len(model.metabolites)].set_xlabel("Error")

  with open(os.path.join(folder, prefix+"_concentration_errors.json"), 'w') as f:
    print(json.dumps(info, indent=2, sort_keys=True), file=f)
  for f in glob.glob(os.path.join(folder, prefix + '_concentration_errors@*.png')):
    os.remove(f)
  for dpi in image_dpi:
    plt.savefig(os.path.join(folder, prefix + '_concentration_errors@'+str(dpi)+'.png'), dpi=dpi)
  if verbose > 1:
    fig.set_dpi(screen_dpi)
    plt.show()
  plt.close()

  return info, error

def _analyse_spectra_error(model, pre, inp, out, folder, prefix, id, verbose, image_dpi, screen_dpi):
  # Analyse spectra output from autoencoder
  if len(out) > 0:
    # Difference between predicted and actual spectra
    # Stats are over all spectra and frequency bins, hence tuple as axis
    # (first axis is spectrum id and 3rd/last axis is frequency bin; 1,2 is acquisition,datatype)
    if verbose > 0:
      print("# Analysing difference between predicted and actual spectra")
    error = pre - out
    error_mean = np.mean(error,axis=(0,3))
    error_std = np.std(error,axis=(0,3))
    error_min = np.min(error,axis=(0,3))
    error_max = np.max(error,axis=(0,3))
    abserror = np.abs(error)
    abserror_mean = np.mean(abserror,axis=(0,3))
    abserror_std = np.std(abserror,axis=(0,3))
    abserror_min = np.min(abserror,axis=(0,3))
    abserror_max = np.max(abserror,axis=(0,3))
    info = { 'prefix': prefix }
    # Per acquisition-datatype signal
    fig, axes =  plt.subplots(pre.shape[1],pre.shape[2]+1)
    fig.suptitle(f"Spectra Signal Prediction Error Analysis ({prefix})")
    for ac in range(0,pre.shape[1]):
      for dt in range(0,pre.shape[2]):
        if verbose > 0:
          print(f"Signals for {model.acquisitions[ac]}-{model.datatype[dt]}")
        info[f"Signal-{model.acquisitions[ac]}-{model.datatype[dt]}"] = {
            'error': {
              'mean': error_mean[ac,dt],
              'std': error_std[ac,dt],
              'min': error_min[ac,dt],
              'max': error_max[ac,dt],
            },
            'abserror': {
              'mean': abserror_mean[ac,dt],
              'std': abserror_std[ac,dt],
              'min': abserror_min[ac,dt],
              'max': abserror_max[ac,dt],
            }
          }
        d = np.copy(error[:,ac,dt,:]).flatten()
        # Histogram over the full error distribution is expensive; so we sample to estimate, if configured
        if Cfg.val['analysis_spectra_error_dist_sampling'] < 100:
          sel = np.random.randint(0,d.shape[0],size=d.shape[0]*Cfg.val['analysis_spectra_error_dist_sampling']//100)
          d = d[sel]
        sns.histplot(d.flatten(), ax=axes[ac,dt])
        axes[ac,dt].set_title(f"Error Dist. Signal {model.acquisitions[ac]}-{model.datatype[dt]}")
        axes[ac,dt].set_xlabel("Error")
        if ac == 0:
          axes[ac,dt].set_ylabel("Count")
    # Total error
    error = np.reshape(error,np.prod(error.shape))
    terror_mean = np.mean(error)
    terror_std = np.std(error)
    terror_min = np.min(error)
    terror_max = np.max(error)
    abserror = np.reshape(abserror,np.prod(abserror.shape))
    tabserror_mean = np.mean(abserror)
    tabserror_std = np.std(abserror)
    tabserror_min = np.min(abserror)
    tabserror_max = np.max(abserror)
    info['total'] = {
        'error': {
          'mean': terror_mean,
          'std': terror_std,
          'min': terror_min,
          'max': terror_max,
        },
        'abserror': {
          'mean': tabserror_mean,
          'std': tabserror_std,
          'min': tabserror_min,
          'max': tabserror_max,
        }
      }
    if verbose > 0:
      print(f"  Total mean absolute error ({prefix}): {info['total']['abserror']['mean']}")
    d = error.flatten()
    # Histogram over the full error distribution is expensive; so we sample to estimate, if configured
    if Cfg.val['analysis_spectra_error_dist_sampling'] < 100:
      sel = np.random.randint(0,d.shape[0],size=d.shape[0]*Cfg.val['analysis_spectra_error_dist_sampling']//100)
      d = d[sel]
    sns.histplot(d, ax=axes[0,pre.shape[2]])
    axes[0,pre.shape[2]].set_title(f"Total Error Dist.")
    axes[0,pre.shape[2]].set_xlabel("Error")

    with open(os.path.join(folder, prefix+"_spectra_errors.json"), 'w') as f:
      print(json.dumps(info, indent=2, sort_keys=True), file=f)
    for f in glob.glob(os.path.join(folder, prefix + '_spectra_errors@*.png')):
      os.remove(f)
    for dpi in image_dpi:
      plt.savefig(os.path.join(folder, prefix + '_spectra_errors@'+str(dpi)+'.png'), dpi=dpi)
    if verbose > 1:
      fig.set_dpi(screen_dpi)
      plt.show()
    plt.close()
  else:
    info = None
    error = None

  # Plot and compare spectra
  path = os.path.join(folder, prefix + "_spectra")
  if not os.path.exists(path):
    os.makedirs(path)
  if verbose > 0:
    print("# Plot predicted spectra: "+path)
  if len(out) > 0:
    ss = min(pre.shape[0],Cfg.val['analysis_predicted_spectra_samples']) # Limit spectra if we have ground truth
  else:
    ss = pre.shape[0] # All spectra if no ground truth
  for s in range(0,ss):
    fig = _plot_predicted_spectra(model, prefix, s, inp[s,:,:,:], pre[s,:,:,:], out[s,:,:,:] if len(out) > 0 else [])
    for dpi in image_dpi:
      plt.savefig(os.path.join(path, f'spectrum_{s}@'+str(dpi)+'.png'), dpi=dpi)
    if verbose > 1:
      fig.set_dpi(screen_dpi)
      plt.show()
    plt.close()
  plt.show()

  return pre, info, error

# Plot difference
def _plot_predicted_spectra(model, prefix, s, inp, pre, out):
  rs = pre.shape[0]*pre.shape[1]
  fig, axs = plt.subplots(rs,3)
  fig.suptitle(f"Spectra Signal Prediction ({prefix})")

  X = np.arange(model.high_ppm,model.low_ppm,(model.low_ppm-model.high_ppm)/pre.shape[2])

  r = 0
  for ac in range(0,pre.shape[0]):
    for dt in range(0,pre.shape[1]):
      axs[r,0].plot(X,inp[ac,dt,:])
      axs[r,0].set_title("Input "+model.acquisitions[ac])
      axs[r,0].set_ylabel(model.datatype[dt])
      if r == rs:
        axs[r,0].set_xlabel("Frequency (ppm)")

      axs[r,1].plot(X,pre[ac,dt,:])
      axs[r,1].set_title("Pred. "+model.acquisitions[ac])
      if r == rs:
        axs[r,1].set_xlabel("Frequency (ppm)")

      if len(out) > 0:
        axs[r,2].plot(X,out[ac,dt,:],color='#DC143C',label="True")
        axs[r,2].plot(X,pre[ac,dt,:],color='#4169E1',label="Pred")
      else:
        axs[r,2].plot(X,inp[ac,dt,:],color='#DC143C',label="Input")
        axs[r,2].plot(X,pre[ac,dt,:],color='#4169E1',label="Pred")
      axs[r,2].set_title("Diff. "+model.acquisitions[ac])
      if r == rs:
        axs[r,2].set_xlabel("Frequency (ppm)")
      axs[r,2].legend(loc='best')

      r += 1

  # plt.legend()

  return fig
