# mrsnet/analyse.py - MRSNet - analyse model performance
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2019, Max Chandler, PhD student at Cardiff University
# Copyright (C) 2020-2021, Frank C Langbein <frank@langbein.org>, Cardiff University

import os
import glob
import warnings
import numpy as np
import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from scipy.stats import wasserstein_distance

def analyse_model(model, inp, out, folder, prefix, id=None, save_conc=False, show_conc=False,
                  no_show=False, verbose=0, image_dpi=[300], screen_dpi=96):
  # Analyse data (assumed to be exported from dataset in format as used by model for train call)
  if not os.path.exists(folder):
    os.makedirs(folder)
  pre = model.predict(inp,verbose=verbose)
  # Analyse if we have concentrations
  if len(out) > 0:
    info, error = _analyse_model_error(model, pre, inp, out, folder, prefix, no_show, verbose, image_dpi, screen_dpi)
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
          writer.writerow(["Spectrum: %d" % l])
        else:
          writer.writerow(["Spectrum: %s" % id[l]])
        if len(out) > 0:
          for k,m in enumerate(model.metabolites):
            writer.writerow([m, pre[l,k], out[l,k], pre[l,k]-out[l,k]])
        else:
          for k,m in enumerate(model.metabolites):
            writer.writerow([m, pre[l,k]])
  if show_conc:
    print('\n# %s Quantifcation Results' % str(model).upper())
    print("Metabolites: "+", ".join(model.metabolites))
    print("Pulse Sequence: "+model.pulse_sequence)
    print('\n                       Concentrations')
    if len(out) > 0:
      print('  %12s  %8s    %8s  %8s' % ('Metabolite', 'Predicted', 'Actual', 'Error'))
    else:
      print('  %12s  %8s' % ('Metabolite', 'Predicted'))
    for l in range(0,inp.shape[0]):
      if id is None:
        print("Spectrum: %d" % l)
      else:
        print("Spectrum: %s" % id[l])
      if len(out) > 0:
        for k,m in enumerate(model.metabolites):
          print('  %12s  %.8f    %.8f  %.8f' % (m, pre[l,k], out[l,k], pre[l,k]-out[l,k]))
      else:
        for k,m in enumerate(model.metabolites):
          print('  %12s  %.8f' % (m, pre[l,k]))

  return pre, info, error


def _analyse_model_error(model, pre, inp, out, folder, prefix, no_show, verbose, image_dpi, screen_dpi):
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
  info = {}

  # Per metabolite plots/data
  fig, axes =  plt.subplots(2,len(model.metabolites)+1)
  fig.suptitle("Concentration Error Analysis (%s)" % prefix)
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

  axes[0,len(model.metabolites)].plot([0, 1], [0, 1], label='true line')
  sns.regplot(y=pre_all, x=out_all, ax=axes[0,len(model.metabolites)])
  axes[0,len(model.metabolites)].set_title('Total')
  axes[0,len(model.metabolites)].set_xlabel("True")
  axes[0,len(model.metabolites)].set_xlim([0,1])
  axes[0,len(model.metabolites)].set_ylim([0,1])
  sns.histplot(error, kde=True, ax=axes[1,len(model.metabolites)])
  axes[1,len(model.metabolites)].set_xlabel("Error")

  # Quality of distribution - wasserstein distance to ideal distribution
  wd = wasserstein_distance(error,
                            np.array([0]),
                            np.ones(error.shape[0])/error.shape[0],
                            np.ones(1))
  info["wasserstein_distance_quality"] = wd
  if verbose > 0:
    print("  Wasserstein distance quality: %f" % wd)

  with open(os.path.join(folder, prefix+"_concentration_errors.json"), 'w') as f:
    print(json.dumps(info, indent=2, sort_keys=True), file=f)
  for f in glob.glob(os.path.join(folder, prefix + '_concentration_errors@*.png')):
    os.remove(f)
  for dpi in image_dpi:
    plt.savefig(os.path.join(folder, prefix + '_concentration_errors@'+str(dpi)+'.png'), dpi=dpi)
  if not no_show:
    fig.set_dpi(screen_dpi)
    plt.show()
  plt.close()

  return info, error
