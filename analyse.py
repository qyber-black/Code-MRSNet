#!/usr/bin/env python3
#
# analyse.py - MRSNet - analyse model performance
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2019, Max Chandler, PhD student at Cardiff University
# Copyright (C) 2020, Frank C Langbein <frank@langbein.org>, Cardiff University

import os
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

from utilities.utils import normalise_labels, reshape_data, convert_molecule_names

def analyse_model(model, inp, out, metabolites, save_dir, prefix):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    inp = np.array(inp)
    out = np.array(out)
    inp, input_shape = reshape_data(inp)

    prediction = normalise_labels(model.predict(inp, verbose=1, batch_size=32), model.name[-3:])
    s_metabolites = convert_molecule_names(metabolites, shorten=True)

    info = {}
    info['match'] = plot_labels_vs_labels(out, prediction, s_metabolites, save_dir, prefix)
    info['error'], info['total'] = plot_metabolite_error_dist(out, prediction, s_metabolites, save_dir, prefix)

    save_analytics(out, prediction, s_metabolites, info, save_dir, prefix)

def save_analytics(true_labels, predicted_labels, output_labels, info, save_dir, prefix):
    save_file = os.path.join(save_dir, 'analytics.json')
    if os.path.exists(save_file):
        try:
            with open(save_file) as f:
                data = json.load(f)
        except:
            data = {}
    else:
        data = {}
    if prefix in data.keys():
        print("Warning, overwriting analytics for "+prefix)
    data[prefix] = { 'metabolites': output_labels,
                     'predicted_concentrations': predicted_labels.tolist(),
                     'true_concentrations': true_labels.tolist(),
                     'total_error': info['total'],
                     'metabolite_match': info['match'],
                     'metabolite_error': info['error'] }
    with open(save_file, 'w') as f:
        print(json.dumps(data, indent=4, sort_keys=True), file=f)

def plot_labels_vs_labels(true_labels, predicted_labels, output_labels, save_dir, prefix):
    import warnings
    metabolite_names = output_labels
    n_cols = 3
    n_rows = int(np.ceil(len(metabolite_names) / float(n_cols)))
    fig, axes = plt.subplots(int(n_rows), int(n_cols), sharex=True, sharey=True, figsize=(19.2, 19.2), dpi=300)
    axes = axes.flatten()
    fig.suptitle('Predicted vs actual concentrations')
    fig.text(0.5, 0.04, 'Actual', ha='center')
    fig.text(0.04, 0.5, 'Predicted', va='center', rotation='vertical')

    info = {}

    xlim = [np.min(true_labels), np.max(true_labels)]
    ylim = [np.min(predicted_labels), np.max(predicted_labels)]

    for ii in range(0, len(metabolite_names)):
        sort_index = np.argsort(true_labels[:, ii])
        axes[ii].plot([0, 1], [0, 1], label='true line')
        sns.regplot(y=predicted_labels[sort_index, ii], x=true_labels[sort_index, ii], ax=axes[ii])
        xm = np.min(true_labels[sort_index, ii])

        # plot slope analysis
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            # ignores division by zero warnings, which will happen in the analysis
            slope, intercept, r_value, p_value, std_err = linregress(true_labels[sort_index, ii], predicted_labels[sort_index, ii])
        info[metabolite_names[ii]] = {
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'p_value': p_value,
            'std_err': std_err
        }
        axes[ii].set_title('%s $R^2$:%.4f Sl:%.4f Int:%.4f p:%.4f Err:%.4f' % (metabolite_names[ii], r_value, slope, intercept, p_value, std_err))

        if ii == 0:
            axes[ii].legend(loc=2)
            axes[ii].set_xlim(xlim)
            axes[ii].set_ylim(ylim)

    plt.savefig(os.path.join(save_dir, prefix + '_pred_vs_true_concentrations.png'))
    plt.close()

    return info

def plot_metabolite_error_dist(true_labels, predicted_labels, output_labels, save_dir, prefix):
    metabolite_names = output_labels
    n_cols = 3
    n_rows = int(np.ceil(len(metabolite_names) / float(n_cols)))
    fig, axes = plt.subplots(int(n_rows), int(n_cols), sharex=True, sharey=True, figsize=(19.2, 19.2), dpi=300)
    fig.suptitle('Concentration error distribution')
    axes = axes.flatten()
    fig.text(0.5, 0.04, 'Error', ha='center')

    info = {}

    total_error = []
    total_pred = []
    for ii in range(0, len(metabolite_names)):
        axes[ii].set_title(metabolite_names[ii])
        error = predicted_labels[:, ii] - true_labels[:, ii]
        e = np.sum(error) / error.shape[0]
        s = np.sqrt(np.sum(np.square(error - e)) / error.shape[0])
        h = np.abs(error)
        a_e = np.sum(h) / h.shape[0]
        a_s = np.sqrt(np.sum(np.square(h - a_e)) / h.shape[0])
        info[metabolite_names[ii]] = {
            'abs_error_mean': a_e,
            'abs_error_std': a_s,
            'error_mean': e,
            'error_std': s,
            'errors': error.tolist()
        }
        total_error.extend(error)
        try:
            sns.distplot(error, ax=axes[ii])
        except np.linalg.LinAlgError:
            print('Singular matrix found in dist plot - not saving.')
            return

    a = np.max([np.abs(np.min(total_error)), np.abs(np.max(total_error))])
    for ii in range(0, len(metabolite_names)):
        axes[ii].set_xlim([-a, a])

    h = np.asarray(total_error)
    e = np.sum(h) / h.shape[0]
    s = np.sqrt(np.sum(np.square(h - e)) / h.shape[0])
    h = np.abs(h)
    a_e = np.sum(h) / h.shape[0]
    a_s = np.sqrt(np.sum(np.square(h - a_e)) / h.shape[0])
    info_total = {
        'abs_error_mean': a_e,
        'abs_error_std': a_s,
        'error_mean': e,
        'error_std': s
    }

    plt.savefig(os.path.join(save_dir, prefix + '_concentration_error_distribution.png'))
    plt.close()

    return info, info_total
