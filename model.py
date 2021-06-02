#!/usr/bin/env python3
#
# model.py - MRSNet - model training
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2019, Max Chandler, PhD student at Cardiff University
# Copyright (C) 2020, Frank C Langbein <frank@langbein.org>, Cardiff University

import os
import shutil
import datetime
import time
import numpy as np
import csv
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import plot_model

from utilities.utils import collapse_array, normalise_labels, reshape_data, convert_molecule_names
from basis import generate_bases
from dataset import Dataset, generate_datasets
from analyse import analyse_model
from cnns import *

# Disable annoying warnings from default TF compilation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
    def on_epoch_begin(self, batch, logs={}):
        self.tic = time.time()
    def on_epoch_end(self, batch, logs={}):
        self.times.append((time.time() - self.tic)*1000)

def train(args):
    if tf.test.gpu_device_name():
      print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
      print("WARNING, we do not have a default GPU for Tensorflow!")

    bases = generate_bases(args.basis_source, args.scanner_manufacturer, args.omega, args.linewidths, args.metabolites, args.verbose)
    datasets = generate_datasets(bases, 'testing', args.num, args.gen, args.metabolites, args.acquisitions, args.verbose)

    model_name=args.model
    validate_per=args.validate_per
    epochs=args.epochs
    batch_size=args.batch_size
    conc_norm=args.norm
    datatype=args.datatype
    save_folder=args.model_folder
    plot=args.plot
    verbose=args.verbose

    # Prepare dataset for training
    in_spectra = [] # Spectra (:,acquisition,frequency)
    out_concentrations = [] # Concentrations (:,concentration)
    out_labels = None # Metabolite names
    acqs = []
    for ds in datasets:
        acqs.extend(ds.acquisitions)
        d_in, d_out, d_labels = ds.export_to_keras(ds.basis.metabolite_names(),
                                                   adc_noise=True,
                                                   adc_noise_p=args.adc_noise_p,
                                                   max_sigma=args.adc_noise_sigma,
                                                   conc_normalisation=conc_norm,
                                                   datatype=datatype)
        in_spectra.extend(d_in)
        out_concentrations.extend(d_out)
        out_labels = check_output_labels(out_labels, d_labels)
    in_spectra = np.array(in_spectra)
    out_concentrations = np.array(out_concentrations)
    acqs = list(set(acqs))

    # Shuffle dataset (split is in fit, but due to sobol sampling, shuffle seems a good idea,
    # as validation_split is done before shuffle from the last samples in the data)
    perm = np.random.permutation(len(in_spectra))
    in_spectra = in_spectra[perm]
    out_concentrations = out_concentrations[perm]

    # Model name extensions:
    model_str = 'M-' + "-".join(convert_molecule_names(args.metabolites, shorten=True)) + \
                '_A-' + "-".join([str(a) for a in args.acquisitions]) + \
                '_T-' + "-".join([t[0] for t in datatype]) + \
                '_P-' + datasets[0].pulse_sequence()

    # Train the model
    model = train_cnn(in_spectra,
                      out_concentrations,
                      out_labels,
                      validate_per,
                      model_name,
                      model_str,
                      epochs,
                      batch_size,
                      conc_norm=conc_norm,
                      acquisitions=acqs,
                      datatype=datatype,
                      save_folder=save_folder)

    # Analyse model with training/test datasets
    n_train = int(np.round(np.float64(validate_per)/100.0 * in_spectra.shape[0]))
    analyse_model(model, in_spectra, out_concentrations, out_labels, save_folder, prefix='traintest')
    analyse_model(model, in_spectra[:n_train], out_concentrations[:n_train], out_labels, save_folder, prefix='train')
    analyse_model(model, in_spectra[n_train:], out_concentrations[n_train:], out_labels, save_folder, prefix='test')

def check_output_labels(model_output_labels, proposed_output_labels):
    if model_output_labels is None:
        return proposed_output_labels
    if not len(model_output_labels) == len(proposed_output_labels):
        raise Exception('Model output labels and proposed output labels are of different lengths! This means '
                        'that there are not the same number of metabolites in different datasets...')
    if not model_output_labels == proposed_output_labels:
        raise Exception('Export labels are not aligned between datasets. '
                        'Either they are out of order, or have different lengths.')
    return model_output_labels

def train_cnn(inp, out, labels, validate_per, model_name, model_str, epochs, batch_size,
              conc_norm='sum', datatype=None, acquisitions=None,
              save_folder='model'):
    inp, input_shape = reshape_data(inp)

    _MODEL_NAME = datetime.datetime.now().strftime('%d%m%y_%H:%M:%S') + '_' + model_name + '_'

    n_out = len(out[0])  # don't remove me - used in the following eval statements
    model = eval(model_name + '(input_shape, n_out, model_str, conc_norm)')
    if conc_norm == 'sum':
        if model.layers[-1].activation.__name__ != 'softmax':
            raise Exception('When using "sum" label norm, please use softmax activation for final layer.')
    elif conc_norm == 'max':
        if model.layers[-1].activation.__name__ != 'sigmoid':
            raise Exception('When using "max" label norm, please use sigmoid activation for final layer.')

    if os.path.isdir(save_folder):
        raise Exception('Model folder ' + save_folder + " already exists")
    elif os.path.isfile(save_folder):
        os.remove(save_folder)
    os.makedirs(save_folder)
    save_path = os.path.join(save_folder+".in_progress")
    if os.path.isdir(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    elif os.path.isfile(save_path):
        os.remove(save_path)
    os.makedirs(save_path)

    plot_model(model,
               to_file=os.path.join(save_path, 'architecture.png'),
               show_shapes=True,
               show_layer_names=True,
               rankdir='TB',
               dpi=300)
    plot_label_distribution(out, save_path, labels)

    optimiser = keras.optimizers.Adam(learning_rate=1e-4,
                                      beta_1=0.9,
                                      beta_2=0.999,
                                      amsgrad=False)
    model.compile(loss='mse',
                  optimizer=optimiser,
                  metrics=['acc', 'mse', 'mae'])
    model.summary()

    timer = TimeHistory()
    callbacks = [keras.callbacks.EarlyStopping(monitor='loss',
                                               min_delta=1e-12,
                                               patience=15,
                                               verbose=1,
                                               restore_best_weights=True),
                 timer]
    history = model.fit(x=inp,
                        y=out,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=np.float64(validate_per)/100.0,
                        shuffle=True,
                        callbacks=callbacks)
    score = model.evaluate(inp, out, verbose=1)
    print('Train+test loss:     ', score[0])
    print('Train+test accuracy: ', score[1])
    print('Train+test MSE:      ', score[2])
    print('Train+test MAE:      ', score[3])

    model.save(os.path.join(save_path, 'model'))
    # Causes annoying warnings, probably not harmful:
    # W tensorflow/python/util/util.cc:329] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
    # calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    # Stil better than storing in h5, I guess.
    save_history(save_path, timer, history.history)
    plot_history(history.history, timer, save_path)

    # Store final results
    if os.path.isdir(save_folder):
        shutil.rmtree(save_folder, ignore_errors=True)
    elif os.path.isfile(save_folder):
        os.remove(save_folder)
    os.rename(save_path, save_folder)
    return model

def plot_label_distribution(labels, save_dir, metabolite_names=None):
    plt.figure(figsize=(19.2, 10.8), dpi=300)
    plt.suptitle('Concentration distribution, N=%i' % (labels.shape[0]))
    metabolite_names = convert_molecule_names(metabolite_names, shorten=True)
    for ii in range(labels.shape[1]):
        plt.subplot(1, labels.shape[1], ii + 1)
        plt.hist(labels[:, ii], bins=50)
        if metabolite_names:
            plt.title('%s' % metabolite_names[ii])
        plt.xlim([0, 1])
    plt.savefig(os.path.join(save_dir, 'concentrations.png'))
    plt.close()

def save_history(save_path, timer, history):
    history['time (ms)'] = timer.times
    keys = sorted(history.keys())
    with open(os.path.join(save_path, 'history.csv'), "w") as out_file:
        writer = csv.writer(out_file, delimiter=",")
        writer.writerow(keys)
        writer.writerows(zip(*[history[key] for key in keys]))

def plot_history(history, timer, filepath='', show_plot=False):
    history_keys = history.keys()

    plt.figure(figsize=(19.2, 10.8), dpi=300)
    plt.suptitle('Error Metrics')

    plt.subplot(4, 1, 1)
    for key in history_keys:
        if 'acc' in key:
            plt.semilogy(history[key], label=key)
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')

    plt.subplot(4, 1, 2) # MSE is loss
    for key in history_keys:
        if 'mse' in key:
            plt.semilogy(history[key], label=key)
    plt.ylabel('MSE')
    plt.legend(loc='lower left')

    plt.subplot(4, 1, 3)
    for key in history_keys:
        if 'mae' in key:
            plt.semilogy(history[key], label=key)
    plt.ylabel('MAE')
    plt.legend(loc='lower left')

    plt.subplot(4, 1, 4)
    plt.plot(timer.times, label='Time per Epoch')
    plt.ylabel('Time (ms)')
    plt.legend(loc='upper right')

    plt.savefig(os.path.join(filepath, 'history.png'))
    if show_plot:
        plt.show()
    else:
        plt.close()
