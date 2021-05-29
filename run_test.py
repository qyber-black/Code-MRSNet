#!/usr/bin/env python3
#
# run_test.py - MRSNet - generate data for papers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2020, Frank C Langbein <frank@langbein.org>, Cardiff University
#
# See --help for arguments

import os
import argparse
import shutil
import subprocess
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

datasets = {}
datasets_compare = {}

# Comparing performance for different dataset generation methods
datasets['MRSNet-v1-Dataset'] = {
    "metabolites": [['Creatine', 'GABA', 'Glutamate', 'Glutamine', 'N-Acetylaspartate']],
    "basis_source": ['FID-A'],
    "scanner_manufacturer": ['Siemens'],
    "omega": [123.23],
    "linewidths": [1.0],
    "num": [10000],
    "norm": ["sum", "max"],
    "gen": ["random", "sobol", "dirichlet", "random-zeros", "sobol-zeros", "dirichlet-zeros"],
    "datatype": [['real', 'imaginary', 'magnitude']],
    "acquisitions": [[0,1,2]],
    "adc_noise_p": [1.0],
    "adc_noise_sigma": [0.1],
    "model": ["mrsnet_cnn_small", "mrsnet_cnn_medium", "mrsnet_cnn_large"],
    "validate_per": [30],
    "epochs": [500],
    "batch_size": [32]
}
datasets_compare['MRSNet-v1-Dataset'] = {
    'all' : [ 'gen', 'norm', 'model' ],
    'norm' : [ 'gen', 'model' ],
    'gen' : [ 'norm', 'model' ],
    'model' : [ 'gen', 'norm' ]
}

# Comparing performance over different noise levels
datasets['MRSNet-v1-Noise'] = {
    "metabolites": [['Creatine', 'GABA', 'Glutamate', 'Glutamine', 'N-Acetylaspartate']],
    "basis_source": ['LCModel', 'FID-A', 'PyGamma'],
    "scanner_manufacturer": ['Siemens'],
    "omega": [123.23],
    "linewidths": [1.0],
    "num": [5000],
    "norm": ["sum"],
    "gen": ["sobol", "dirichlet"],
    "datatype": [['real', 'imaginary', 'magnitude']],
    "acquisitions": [[0,1,2]],
    "adc_noise_p": [1.0],
    "adc_noise_sigma": [0.05, 0.1, 0.15, 0.2, 0.25],
    "model": ["mrsnet_cnn_small"],
    "validate_per": [30],
    "epochs": [500],
    "batch_size": [32]
}
datasets_compare['MRSNet-v1-Noise'] = {
    'all' : [ 'basis_source', 'gen', 'adc_noise_sigma' ],
    'noise' : [ 'basis_source', 'gen' ],
    'src' : [ 'gen', 'adc_noise_sigma' ],
    'gen' : [ 'basis_source', 'adc_noise_sigma' ]
}

# Comparing over model and dataset parameters: (model,batch_size), (norm,gen,adc_noise_p)
datasets['MRSNet-v1-Models_DS'] = {
    "metabolites": [['Creatine', 'GABA', 'Glutamate', 'Glutamine', 'N-Acetylaspartate']],
    "basis_source": ['LCModel'],
    "scanner_manufacturer": ['Siemens'],
    "omega": [123.23],
    "linewidths": [[1.0]],
    "num": [5000],
    "norm": ["sum", "max"],
    "gen": ["sobol", "dirichlet"],
    "datatype": [['real', 'imaginary', 'magnitude']],
    "acquisitions": [[0,1,2]],
    "adc_noise_p": [0.5, 1.0],
    "adc_noise_sigma": [0.25], # FIXME: Lower?
    "validate_per": [30],
    "model": ["mrsnet_cnn_small", "mrsnet_cnn_small_pool",
              "mrsnet_cnn_medium", "mrsnet_cnn_medium_pool",
              "mrsnet_cnn_large", "mrsnet_cnn_large_pool"],
    "epochs": [150],
    "batch_size": [16, 32, 64, 96]
}
datasets_compare['MRSNet-v1-Models_DS'] = {
    'dataset' : [ 'norm',  'gen', 'adc_noise_p', 'Repeat' ],
    'model' : [ 'model', 'batch_size' ]
}

# Comparing over model and dataset parameters: (model,batch_size), (basis_source,gen); multi-linewidth witohut pooling models
datasets['MRSNet-v1-Models_MLW'] = {
    "metabolites": [['Creatine', 'GABA', 'Glutamate', 'Glutamine', 'N-Acetylaspartate']],
    "basis_source": [['FID-A'], ['PyGamma'], ['FID-A', 'PyGamma']],
    "scanner_manufacturer": ['Siemens'],
    "omega": [123.23],
    "linewidths": [[0.75,1.0,1.25]],
    "num": [10000],
    "norm": ["sum"],
    "gen": ["sobol", "dirichlet"],
    "datatype": [['real', 'imaginary', 'magnitude']],
    "acquisitions": [[0,1,2]],
    "adc_noise_p": [1.0],
    "adc_noise_sigma": [0.1], # FIXME: ?
    "validate_per": [30],
    "model": ["mrsnet_cnn_small", "mrsnet_cnn_medium", "mrsnet_cnn_large"],
    "epochs": [150],
    "batch_size": [16, 32, 64, 96]
}
datasets_compare['MRSNet-v1-Models_MLW'] = {
    'dataset' : [ 'basis_source', 'gen', 'Repeat' ],
    'model#' : [ 'model', 'batch_size' ]
}

# Comparing over dataset parameters: (basis_source); single-linewidth for small/large model
datasets['MRSNet-v1-DS_BS'] = {
    "metabolites": [['Creatine', 'GABA', 'Glutamate', 'Glutamine', 'N-Acetylaspartate']],
    "basis_source": [['FID-A'], ['LCModel'], ['PyGamma'], ['FID-A', 'LCModel'], ['FID-A', 'PyGamma'], ['LCModel', 'PyGamma'], ['FID-A', 'LCModel', 'PyGamma']],
    "scanner_manufacturer": ['Siemens'],
    "omega": [123.23],
    "linewidths": [[1.0]],
    "num": [10000], # FIXME: Other value?
    "norm": ["sum"],
    "gen": ["sobol"],
    "datatype": [['real', 'imaginary', 'magnitude']],
    "acquisitions": [[0,1,2]],
    "adc_noise_p": [1.0],
    "adc_noise_sigma": [0.25], # FIXME: Other value?
    "validate_per": [30],
    "model": ["mrsnet_cnn_small", "mrsnet_cnn_large"],
    "epochs": [150],
    "batch_size": [32]
}
datasets_compare['MRSNet-v1-DS_BS'] = {
    'dataset' : [ 'basis_source', 'Repeat' ],
    'model' : [ 'model' ]
}

# Single linewidth, all bases, CNN models
datasets['MRSNet-v1-SLW-1'] = {
    "metabolites": [['Creatine', 'GABA', 'Glutamate', 'Glutamine', 'N-Acetylaspartate']],
    "basis_source": ['LCModel','FID-A','PyGamma'],
    "scanner_manufacturer": ['Siemens'],
    "omega": [123.23],
    "linewidths": [[1.0]],
    "num": [10000],
    "norm": ["sum","max"],
    "gen": ["sobol", "dirichlet"],
    "datatype": [['real'], ['imaginary'], ['magnitude'], ['real', 'imaginary'], ['real', 'magnitude'], ['imaginary', 'magnitude'], ['real', 'imaginary', 'magnitude']],
    "acquisitions": [[0,2]],
    "adc_noise_p": [0.5],
    "adc_noise_sigma": [0.025],
    "validate_per": [30],
    "model": ["mrsnet_cnn_small", "mrsnet_cnn_small_pool",
              "mrsnet_cnn_medium", "mrsnet_cnn_medium_pool",
              "mrsnet_cnn_large", "mrsnet_cnn_large_pool"],
    "epochs": [500],
    "batch_size": [16, 32, 64, 128]
}
datasets_compare['MRSNet-v1-SLW-1'] = {
    'dataset' : [ 'basis_source', 'norm', 'gen', 'datatype', 'Repeat' ],
    'model' : [ 'model', 'batch_size' ]
}

# Multi linewidth, FID-A/PyGamma bases, CNN models
datasets['MRSNet-v1-MLW-1'] = {
    "metabolites": [['Creatine', 'GABA', 'Glutamate', 'Glutamine', 'N-Acetylaspartate']],
    "basis_source": ['FID-A','PyGamma'],
    "scanner_manufacturer": ['Siemens'],
    "omega": [123.23],
    "linewidths": [[0.75,1.0,1.25]],
    "num": [10000],
    "norm": ["sum","max"],
    "gen": ["sobol"],
    "datatype": [['real'], ['imaginary'], ['magnitude'], ['real', 'imaginary'], ['real', 'magnitude'], ['imaginary', 'magnitude'], ['real', 'imaginary', 'magnitude']],
    "acquisitions": [[0,2]],
    "adc_noise_p": [0.5],
    "adc_noise_sigma": [0.025],
    "validate_per": [30],
    "model": ["mrsnet_cnn_small", "mrsnet_cnn_small_pool",
              "mrsnet_cnn_medium", "mrsnet_cnn_medium_pool",
              "mrsnet_cnn_large", "mrsnet_cnn_large_pool"],
    "epochs": [250],
    "batch_size": [16, 32, 64, 128]
}
datasets_compare['MRSNet-v1-MLW-1'] = {
    'dataset' : [ 'basis_source', 'norm',  'datatype', 'Repeat' ],
    'model' : [ 'model', 'batch_size' ]
}

# Final comparison
datasets['MRSNet-v1-Final'] = {
    "metabolites": [['Creatine', 'GABA', 'Glutamate', 'Glutamine', 'N-Acetylaspartate']],
    "basis_source": ['LCModel', 'FID-A'],
    "scanner_manufacturer": ['Siemens'],
    "omega": [123.23],
    "linewidths": [1.0],
    "num": [20000],
    "norm": ["sum"],
    "gen": ["sobol", "dirichlet"],
    "datatype": ['magnitude'],
    "acquisitions": [[0,2]],
    "adc_noise_p": [0.5, 1.0],
    "adc_noise_sigma": [0.05, 0.075],
    "validate_per": [30],
    "model": ["mrsnet_cnn_small"],
    "epochs": [500],
    "batch_size": [64]
}
datasets_compare['MRSNet-v1-Final'] = {
    'dataset' : [ 'basis_source', 'gen' ],
    'model' : [ 'adc_noise_p', 'adc_noise_sigma' ]
}

def main():
    # headless mode
    if not "DISPLAY" in os.environ:
        matplotlib.use("Agg")

    # Process arguments
    parser = argparse.ArgumentParser(description='Generate MRSNet data sets for papers',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', type=str, choices=sorted(datasets.keys()), help='ID for dataset to be generated as specified in the scripts.')
    parser.add_argument('repeats', type=int, help='Repeats for each set', default=1)
    args = parser.parse_args()

    if args.dataset in datasets:
        ds = datasets[args.dataset]
        keys = list(ds.keys())

        all_err = {}
        all_std = {}

        szs = [len(ds[k]) for k in keys]
        total_iter = np.product(szs)
        cur_iter = 0
        szs.append(args.repeats)
        all_time = np.zeros(szs)

        iters = [0] * len(keys)
        while True:
            cur_iter += 1
            print(iters, "%d of %d" % (cur_iter, total_iter))

            # Construct arguments and  folder name
            cmd_args = []
            folder = os.path.join('data', 'model', args.dataset)
            for l in range(len(keys)):
                cmd_args.append("--"+keys[l])
                if isinstance(ds[keys[l]][iters[l]],list):
                    for s in ds[keys[l]][iters[l]]:
                        cmd_args.append(str(s))
                    if len(ds[keys[l]]) > 1:
                        folder = os.path.join(folder,keys[l]+str(iters[l]))
                else:
                    v = str(ds[keys[l]][iters[l]])
                    cmd_args.append(v)
                    if len(ds[keys[l]]) > 1:
                        folder = os.path.join(folder,v)
            os.makedirs(folder,exist_ok=True)

            # Check number of repeats for training
            rep = [0] * args.repeats
            for d in os.listdir(folder):
                try:
                    num = int(d)-1
                except ValueError:
                    continue
                if num < args.repeats:
                    if os.path.isdir(os.path.join(folder, str(num+1), "model")):
                        rep[num] = 1
                    else:
                        shutil.rmtree(os.path.join(folder, str(num+1)))
            rep_id = args.repeats

            # Train model
            trained = False
            for rep_id in range(args.repeats):
                if rep[rep_id] == 0:
                    trained = True
                    # Execute
                    rep[rep_id] = 1
                    m_folder = os.path.join(folder, str(rep_id+1))
                    cmd = ['/usr/bin/env', 'python3', os.path.join('./mrsnet.py'), "train", "--model_folder", m_folder, *cmd_args]
                    print("  Training, repeat %d" % (rep_id+1))
                    print(cmd)
                    try:
                        p = subprocess.Popen(cmd)
                        p.wait()
                    except OSError as e:
                        p.wait()
                        raise Exception('MRSNet train failed!') from e
            if not trained:
                print("  Training complete")

            # Run Benchmark
            for d in range(args.repeats):
                m_folder = os.path.join(folder, str(d+1))
                if os.path.isdir(os.path.join(m_folder, "model")):
                    # Get data
                    json_file = os.path.join(m_folder, 'analytics.json')
                    if os.path.exists(json_file):
                        try:
                            with open(json_file) as f:
                                data = json.load(f)
                        except:
                            data = {}
                    else:
                        data = {}
                    # Compute benchmark, if not already there
                    if 'E1' not in data.keys() or 'E3' not in data.keys() or 'E4a' not in data.keys() or 'E4b' not in data.keys():
                        print("  Benchmark, repeat %d" % (d+1))
                        cmd = ['/usr/bin/env', 'python3', os.path.join('./mrsnet.py'), "benchmark", "--model", m_folder]
                        print(cmd)
                        try:
                            p = subprocess.Popen(cmd)
                            p.wait()
                        except OSError as e:
                            p.wait()
                            raise Exception('MRSNet benchmark failed!') from e
                        # Update data with results
                        if os.path.exists(json_file):
                            with open(json_file) as f:
                                data = json.load(f)
                    # Get data
                    print("  Data, repeat %d" % (d+1))
                    idx = iters.copy()
                    idx.append(d)
                    # Errors
                    for seq in data:
                        if 'total_error' in data[seq]:
                            if seq not in all_err:
                                szs = [len(ds[k]) for k in keys]
                                szs.append(args.repeats)
                                all_err[seq] = np.zeros(szs)
                                all_std[seq] = np.zeros(szs)
                            np.put(all_err[seq], np.ravel_multi_index(idx,all_err[seq].shape), data[seq]['total_error']['abs_error_mean'])
                            np.put(all_std[seq], np.ravel_multi_index(idx,all_std[seq].shape), data[seq]['total_error']['abs_error_std'])
                    # Timing data
                    h_file = os.path.join(m_folder, 'history.csv')
                    if os.path.exists(h_file):
                        with open (h_file, newline='') as f:
                            reader = csv.reader(f, delimiter=',')
                            t = 0.0
                            n = -1.0
                            k = -1
                            for row in reader:
                                if n < 0.0:
                                    for l in range(len(row)):
                                        if row[l][0:4] == 'time':
                                            k = l
                                            n = 0.0
                                else:
                                    v = np.float64(row[k])
                                    if v > 0.0:
                                      t += v
                                      n += 1.0
                            t = t / n
                            np.put(all_time, np.ravel_multi_index(idx,all_time.shape), t)

            # Increase iteration index
            k = len(iters) - 1
            while True:
                iters[k] += 1
                if iters[k] >= len(ds[keys[k]]):
                    iters[k] = 0
                    k -= 1
                    if k < 0:
                        ds['Repeat'] = range(1,args.repeats+1)
                        save_errors(os.path.join('data', 'model', args.dataset, 'errors.csv'), all_err, all_std, all_time, ds)
                        if args.dataset in datasets_compare:
                            for cmp_name, cmp_iters in datasets_compare[args.dataset].items():
                                create_plots(os.path.join('data', 'model', args.dataset), all_err, all_std, all_time, ds, cmp_name, cmp_iters)
                                scatter_plot(os.path.join('data', 'model', args.dataset), all_err, all_std, ds, cmp_name, cmp_iters)
                        return
                else:
                    break

def save_errors(file, err, std, tme, ds):
    print("Saving results to "+file)
    with open(file, 'w', newline='') as f:
        w = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        keys = list(ds.keys())
        # Header
        r = []
        for k in keys:
            r.append(k)
        for seq in err.keys():
            r.append(seq+"_err")
            r.append(seq+"_std")
        r.append('time')
        w.writerow(r)

        # Process all rows (iterate over dataset fields, incl. repeats)
        iters = [0] * len(keys)
        while True:
            # Construct row entries to identify result
            r = []
            for l in range(len(keys)):
                if isinstance(ds[keys[l]][iters[l]],list):
                    r.append(keys[l]+str(iters[l]))
                else:
                    r.append(str(ds[keys[l]][iters[l]]))

            # Error, timing for benchmark sequences
            print(iters)
            for seq in err.keys():
                r.append(np.take(err[seq], np.ravel_multi_index(iters, err[seq].shape)))
                r.append(np.take(std[seq], np.ravel_multi_index(iters, std[seq].shape)))
            r.append(np.take(tme, np.ravel_multi_index(iters, tme.shape)))

            w.writerow(r)

            # Increase iteration index
            k = len(iters) - 1
            while True:
                iters[k] += 1
                if iters[k] >= len(ds[keys[k]]):
                    iters[k] = 0
                    k -= 1
                    if k < 0:
                        return
                else:
                    break

def create_plots(folder, err, std, tme, ds, cmp_name, cmp_iters):
    print("Creating plots in " + folder)
    # Split keys into constant (fixed for all graphs), var_within cmp_iters (to compare over in the graph) and var_across (to compare across graphs)
    keys = list(ds.keys())
    cnst = []
    var_within = []
    var_across = []
    cnst_str = ""
    for ki in range(0,len(keys)):
        if len(ds[keys[ki]]) > 1:
            if keys[ki] in cmp_iters:
                var_within.append(ki)
            else:
                var_across.append(ki)
        else:
            cnst.append(ki)
            if len(cnst_str) > 0:
                cnst_str += ", "
            cnst_str += keys[ki]+": "+str(ds[keys[ki]][0])
    # Generate graphs
    across_iters = [0] * len(var_across)
    while True:
        # String/index describing fixed dataset part for graph
        iters = [0] * len(keys)
        across_str = ""
        fn = ""
        for ki in range(0,len(var_across)):
            iters[var_across[ki]] = across_iters[ki]
            if len(across_str) > 0:
                across_str += ", "
                fn += "-"
            across_str += keys[var_across[ki]]+": "+str(ds[keys[var_across[ki]]][across_iters[ki]])
            fn += str(ds[keys[var_across[ki]]][across_iters[ki]])
        if len(var_across) > 0:
            print("Graph comparing for "+across_str)
        else:
            print("Graph comparing all")

        # Collect data for graph
        n_rows = int(np.product([len(ds[keys[k]]) for k in var_within]))
        n_cols = len(err.keys())
        err_data = np.ndarray((n_rows,n_cols))
        std_data = np.ndarray((n_rows,n_cols))
        tme_data = np.ndarray((n_rows,1))
        i_row = 0
        labels = []
        within_iters = [0] * len(var_within)
        done = False
        while not done:
            # Full index
            label = ""
            for ki in range(0,len(var_within)):
                iters[var_within[ki]] = within_iters[ki]
                if len(label) > 0:
                    label += "-"
                if isinstance(ds[keys[var_within[ki]]][within_iters[ki]],list):
                    label += keys[var_within[ki]]+str(within_iters[ki])
                    fn += str(within_iters[ki])
                else:
                    label += str(ds[keys[var_within[ki]]][within_iters[ki]])
            label = label.replace("mrsnet_cnn_","")
            labels.append(label)

            # Get data for index
            i_col = 0
            for seq in err.keys():
                err_data[i_row,i_col] = np.take(err[seq], np.ravel_multi_index(iters, err[seq].shape))
                std_data[i_row,i_col] = np.take(std[seq], np.ravel_multi_index(iters, std[seq].shape))
                i_col += 1
            tme_data[i_row,0] = np.take(tme, np.ravel_multi_index(iters, tme.shape))
            i_row += 1

            # Increase iteration index
            k = len(within_iters) - 1
            while True:
                within_iters[k] += 1
                if within_iters[k] >= len(ds[keys[var_within[k]]]):
                    within_iters[k] = 0
                    k -= 1
                    if k < 0:
                        done = True
                        break
                else:
                    break

        # Plot
        nc = 4
        nr = int(np.ceil((len(err.keys())+1.0)/nc))
        fig, axs = plt.subplots(nr,nc, figsize=(19.2, 10.8), dpi=300)
        ri = 0
        ci = 0
        ki = 0
        y_max = 0
        for seq in err.keys():
            ax = axs[ri,ci]
            eb = np.ndarray((2,std_data.shape[0]))
            eb[0,:] = std_data[:,ki]
            eb[1,:] = std_data[:,ki]
            sel = (err_data[:,ki] - std_data[:,ki] < 0)
            eb[0,sel] = err_data[sel,ki]
            ax.bar(np.arange(n_rows), err_data[:,ki], yerr=eb, align='center', alpha=0.5, color='black', capsize=3, clip_on=True)
            a = ax.get_ylim()
            if a[1] > y_max:
                y_max = a[1]
            if ci == 0:
                ax.set_ylabel('Loss with std-dev error bars')
            else:
                ax.tick_params(axis='y', which='both', bottom=False, top=False)
                ax.set_yticklabels([])
            ax.set_xticks(np.arange(n_rows))
            if ri == nr-1:
                ax.set_xticklabels(labels, rotation=90, ha='center')
            else:
                ax.tick_params(axis='x', which='both', bottom=False, top=False)
                ax.set_xticklabels([])
            ax.set_title(seq)
            ci += 1
            if ci >= nc:
                for ci in range(0,nc):
                    axs[ri,ci].set_ylim([0,y_max])
                y_max = 0
                ci = 0
                ri += 1
            ki += 1
        axs[nr-1,nc-1].bar(np.arange(n_rows), tme_data[:,0], align='center', alpha=0.5, color='black', capsize=10)
        axs[nr-1,nc-1].set_ylim([0,np.max(tme_data[:,0])])
        axs[nr-1,nc-1].set_xticks(np.arange(n_rows))
        axs[nr-1,nc-1].set_xticklabels(labels, rotation=90, ha='center')
        axs[nr-1,nc-1].set_title("Time per Epoch (ms)")
        fig.suptitle('Loss for ' + across_str + '\n (' + cnst_str + ')', wrap=True)
        plt.savefig(os.path.join(folder, cmp_name + '_' + fn +'.png'))
        plt.close()

        if len(across_iters) < 1:
            return

        # Increase iteration index
        k = len(across_iters) - 1
        while True:
            across_iters[k] += 1
            if across_iters[k] >= len(ds[keys[var_across[k]]]):
                across_iters[k] = 0
                k -= 1
                if k < 0:
                    return
            else:
                break

def scatter_plot(folder, err, std, ds, cmp_name, cmp_iters):
    print("Creating scatter plot " + cmp_name)
    cm = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff007f', '#ffff33', '#a65628', '#f781bf']
    mm = ['o', 's', 'D', '*', '+', 'x', 'v', '^', '<', '>' ]
    keys = list(ds.keys())
    n_points = np.product([len(ds[k]) for k in keys])
    nc = 4
    nr = int(np.ceil((len(err.keys()))/nc))
    fig, axs = plt.subplots(nr,nc, figsize=(19.2, 10.8), dpi=300)
    ri = 0
    ci = 0
    ki = 0
    for seq in err.keys():

        x = np.ndarray((n_points))
        y = np.ndarray((n_points))
        mar = []
        col = []
        siz = []
        i_pt = 0

        iters = [0] * len(keys)
        done = False
        while not done:
            x[i_pt] = np.take(err[seq], np.ravel_multi_index(iters, err[seq].shape))
            y[i_pt] = np.take(std[seq], np.ravel_multi_index(iters, std[seq].shape))
            i_pt += 1
            for li in range(0,len(iters)):
                if len(cmp_iters) > 0 and keys[li] == cmp_iters[0]:
                    mar.append(mm[iters[li]])
                elif len(cmp_iters) > 1 and keys[li] == cmp_iters[1]:
                    col.append(cm[iters[li]])
                elif len(cmp_iters) > 2 and keys[li] == cmp_iters[2]:
                    siz.append((iters[li]+1)**2*10)
            # Increase iteration index
            k = len(iters) - 1
            while True:
                iters[k] += 1
                if iters[k] >= len(ds[keys[k]]):
                    iters[k] = 0
                    k -= 1
                    if k < 0:
                        done = True
                        break
                else:
                    break

        ax = axs[ri,ci]
        if len(cmp_iters) < 1:
            ax.scatter(x,y)
        elif len(cmp_iters) < 2:
            mar = np.asarray(mar)
            for m in mm:
                sel = (mar == m)
                ax.scatter(x[sel],y[sel],marker=m)
        elif len(cmp_iters) < 3:
            mar = np.asarray(mar)
            col = np.asarray(col)
            for m in mm:
                sel = (mar == m)
                ax.scatter(x[sel],y[sel],marker=m,c=col[sel])
        else:
            mar = np.asarray(mar)
            col = np.asarray(col)
            siz = np.asarray(siz)
            for m in mm:
                sel = (mar == m)
                ax.scatter(x[sel],y[sel],marker=m,c=col[sel],s=siz[sel])

        if ci == 0:
            ax.set_ylabel('Std-Dev')
        if ri == nr-1:
            ax.set_xlabel('Mean Error')
        ax.set_title(seq)
        ci += 1
        if ci >= nc:
            ci = 0
            ri += 1
        ki += 1

    ax = axs[nr-1,nc-1]

    c = 1.0
    mr = 0.0
    if len(mar) > 0 and len(mm) > mr:
        mr = len(mm)
    if len(col) > 0 and len(cm) > mr:
        mr = len(cm)
    if len(siz) > 0 and len(ds[cmp_iters[2]]) > mr:
        mr = len(ds[cmp_iters[2]])
    if len(mar) > 0:
        r = mr
        ax.text(c, r, cmp_iters[0])
        if mr < r:
            mr = r
        for i in range(len(mm)):
            if len(ds[cmp_iters[0]]) > i:
                r -= 1
                ax.plot([c], [r], mm[i], color='blue')
                ax.text(c+0.5, r, ds[cmp_iters[0]][i])
        c += 5
    if len(col) > 0:
        r = mr
        ax.text(c, r, cmp_iters[1])
        if mr < r :
            mr = r
        for i in range(len(cm)):
            if len(ds[cmp_iters[1]]) > i:
                r -= 1
                ax.plot([c], [r], 'o', markerfacecolor=cm[i], markeredgecolor=cm[i])
                ax.text(c+0.5, r, ds[cmp_iters[1]][i])
        c += 5
    if len(siz) > 0:
        r = mr
        ax.text(c, r, cmp_iters[2])
        if mr < r :
            mr = r
        for i in range(len(ds[cmp_iters[2]])):
            r -= 1
            ax.plot([c], [r], 'o', markersize=(ds[cmp_iters[2]][i]+1)**2*10, markerfacecolor='green', markeredgecolor='green')
            ax.text(c+0.5, r, ds[cmp_iters[2]][i])
        c += 5

    ax.set_ylim([0, mr+1])
    ax.set_xlim([0, c+5])

    fig.suptitle('Mean absolute error vs. std-dev of all results')
    plt.savefig(os.path.join(folder, 'scatter-' + cmp_name + '.png'))
    plt.close()

if __name__ == '__main__':
    main()
