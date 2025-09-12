# mrsnet/cnn.py - MRSNet - CNN models
#
# SPDX-FileCopyrightText: Copyright (C) 2019 Max Chandler, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2020-2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Convolutional Neural Network models for MRSNet.

This module provides CNN implementations for concentration prediction
from MRS spectra, including various architectures and training utilities.
"""

import csv
import io
import json
import os
import sys
from time import time_ns

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, InputLayer, MaxPool2D, ReLU
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import plot_model

from mrsnet.cfg import Cfg
from mrsnet.train import calculate_flops, reshape_spectra_data


class CNN:
  """Convolutional Neural Network for concentration prediction.

  This class implements CNN models for predicting metabolite concentrations
  from MRS spectra. It supports various architectures and training strategies.

  Attributes
  ----------
      model (str): Model architecture identifier
      metabolites (list): List of metabolite names to predict
      pulse_sequence (str): Pulse sequence type
      acquisitions (list): List of acquisition types
      datatype (list): List of data types (e.g., ['magnitude', 'phase'])
      norm (str): Normalization method
      output (str): Output type (always "concentrations")
      low_ppm (float): Lower PPM bound for input data
      high_ppm (float): Upper PPM bound for input data
      fft_samples (int): Number of FFT samples
      train_dataset_name (str): Name of training dataset
      cnn_arch (keras.Model): The actual CNN model
  """

  def __init__(self, model, metabolites, pulse_sequence, acquisitions, datatype, norm):
    """Initialize a CNN model.

    Parameters
    ----------
        model (str): Model architecture identifier (e.g., 'cnn_small_softmax')
        metabolites (list): List of metabolite names to predict
        pulse_sequence (str): Pulse sequence type (e.g., 'megapress')
        acquisitions (list): List of acquisition types (e.g., ['edit_off', 'difference'])
        datatype (list): List of data types (e.g., ['magnitude', 'phase'])
        norm (str): Normalization method (e.g., 'sum', 'max')
    """
    self.model = model
    self.metabolites = metabolites
    self.pulse_sequence = pulse_sequence
    self.acquisitions = acquisitions
    self.datatype = datatype
    self.norm = norm
    self.output = "concentrations"

    # Input spectra data (constant!)
    self.low_ppm = -1.0
    self.high_ppm = -4.5
    self.fft_samples = 2048

    self.train_dataset_name = None
    self.cnn_arch = None

  def __str__(self):
    """Get string representation of the model path.

    Returns
    -------
        str: Model path string combining all parameters
    """
    n = os.path.join(self.model, "-".join(self.metabolites),
                     self.pulse_sequence, "-".join(self.acquisitions),
                     "-".join(self.datatype), self.norm)
    return n

  def reset(self):
    """Reset the CNN architecture and training dataset name.

    Clears the current model architecture and training dataset information.
    """
    del self.cnn_arch
    self.cnn_arch = None
    self.train_dataset_name = None

  def _freq_conv_layer(self, filter, c, s, dropout):
    """Add a frequency convolution layer to the CNN.

    Parameters
    ----------
        filter (int): Number of filters
        c (int): Convolution kernel size
        s (int): Stride size
        dropout (float): Dropout rate
    """
    if s <= 0:
      self.cnn_arch.add(Conv2D(filter, c))
    else:
      self.cnn_arch.add(Conv2D(filter, c, strides=(1,s)))
    if dropout == 0.0:
      self.cnn_arch.add(BatchNormalization())
    self.cnn_arch.add(ReLU())
    if dropout > 0.0:
      self.cnn_arch.add(Dropout(dropout))
    if s < 0:
      self.cnn_arch.add(MaxPool2D((1,-s)))

  def _construct(self, input_shape, output_shape):
    """Construct the CNN architecture.

    Parameters
    ----------
        input_shape (tuple): Input tensor shape
        output_shape (tuple): Output tensor shape
    """
    self.cnn_arch = Sequential(name=self.model)
    vals = self.model.split("_")
    if vals[0] != 'cnn':
      raise RuntimeError(f"Unknown model {vals[0]}")
    if vals[1] == 'small' or vals[1] == 'medium' or vals[1] == 'large':
      # cnn_[small,medium,large]_[softmax,sigmoid][_pool]
      if vals[1] == 'small':
        freq_convolution1 = 7
        freq_convolution2 = 5
        freq_convolution3 = 3
      elif vals[1] == 'medium':
        freq_convolution1 = 9
        freq_convolution2 = 7
        freq_convolution3 = 5
      elif vals[1] == 'large':
        freq_convolution1 = 11
        freq_convolution2 = 9
        freq_convolution3 = 7
      freq_convolution4 = 3
      dropout1 = 0.0
      dropout2 = 0.3
      output_act = vals[2]
      if vals[-1] == 'pool':
        strides1 = -2
        strides2 = -3
      else:
        strides1 = 2
        strides2 = 3
      filter1 = 256
      filter2 = 512
      dense = 1024
    else:
      # cnn_[S1]_[S2]_[C1]_[C2]_[C3]_[C4]_[O1]_[O2]_[F1]_[F2]_[D]_[softmax,sigmoid]
      strides1 = int(vals[1])
      strides2 = int(vals[2])
      freq_convolution1 = int(vals[3])
      freq_convolution2 = int(vals[4])
      freq_convolution3 = int(vals[5])
      freq_convolution4 = int(vals[6])
      dropout1 = float(vals[7])
      dropout2 = float(vals[8])
      filter1 = int(vals[9])
      filter2 = int(vals[10])
      dense = int(vals[11])
      output_act = vals[12]

    self.cnn_arch.add(InputLayer(shape=input_shape))
    self._freq_conv_layer(filter1, (1,freq_convolution1), strides1, dropout1)
    self._freq_conv_layer(filter1, (1,freq_convolution2), strides1, dropout1)

    while self.cnn_arch.layers[-1].output.shape[1] != 1:
      self._freq_conv_layer(filter1,
                            (min(self.cnn_arch.layers[-1].output.shape[1],3),freq_convolution3),
                            1, dropout1)

    for n_filters in [filter1, filter2]:
      for _l in range(2):
        self._freq_conv_layer(n_filters, (1, freq_convolution4), 1,        dropout1)
        self._freq_conv_layer(n_filters, (1, freq_convolution4), strides2, dropout1)

    self.cnn_arch.add(Flatten())
    self.cnn_arch.add(Dense(dense,activation="sigmoid"))
    if dropout2 > 0.0:
      self.cnn_arch.add(Dropout(dropout2))
    self.cnn_arch.add(Dense(output_shape[-1], activation=output_act))

  def train(self, d_data, v_data, epochs, batch_size,
            folder, verbose=0, image_dpi=[300], screen_dpi=96, train_dataset_name=""):
    """Train the CNN model on provided data.

    Parameters
    ----------
        d_data (list): Training data [spectra, concentrations]
        v_data (list, optional): Validation data [spectra, concentrations]. Defaults to None
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        folder (str): Output folder for saving results
        verbose (int, optional): Verbosity level. Defaults to 0
        image_dpi (list, optional): Image DPI settings. Defaults to [300]
        screen_dpi (int, optional): Screen DPI setting. Defaults to 96
        train_dataset_name (str, optional): Name of training dataset. Defaults to ""

    Returns
    -------
        tuple: (training_results, validation_results) dictionaries with MSE and MAE scores

    Raises
    ------
        RuntimeError: If data format is incorrect
    """
    devices = tf.config.list_logical_devices("GPU")
    if len(devices) < 1:
      print("**WARNING, we do not have a GPU for Tensorflow!**")
      devices = []
    else:
      devices = [devices[l].name for l in range(0,len(devices))]
      if verbose > 0:
        print(f"GPU Devices: {devices}")
    if len(d_data) != 2:
      raise RuntimeError("d_data argument must be a list [spectra,conc]")
    if v_data is not None and len(v_data) != 2:
      raise RuntimeError("v_data argument must be a list [spectra,conc]")

    if len(train_dataset_name) > 0:
      self.train_dataset_name = train_dataset_name

    if not os.path.isdir(folder):
      os.makedirs(folder)

    if verbose > 0:
      print(f"# Train CNN {self!s}")

    d_inp = tf.convert_to_tensor(d_data[0], dtype=tf.float32)
    d_out = tf.convert_to_tensor(d_data[1], dtype=tf.float32)
    d_inp = reshape_spectra_data(d_inp, add_channel_dim=True)

    train_data = tf.data.Dataset.from_tensor_slices((d_inp, d_out))

    if v_data is not None:
      v_inp = tf.convert_to_tensor(v_data[0], dtype=tf.float32)
      v_out = tf.convert_to_tensor(v_data[1], dtype=tf.float32)
      v_inp = reshape_spectra_data(v_inp, add_channel_dim=True)
      validation_data = tf.data.Dataset.from_tensor_slices((v_inp, v_out))
    else:
      validation_data = None

    if verbose > 0:
      print("  Input:",d_inp.shape,"[spectrum, acquisition x datatype, frequency]")
      print("  Output:",d_out.shape,"[spectrum, metabolite_concentration]")

    learning_rate = Cfg.val['base_learning_rate'] * batch_size / 16.0

    if len(devices) > 1:
      # Multi-GPU training
      dev_multiplier = len(devices)
      mirrored_strategy = tf.distribute.MirroredStrategy(devices=devices)
      with mirrored_strategy.scope():
        self._construct(d_inp.shape[1:],d_out.shape[1:])
        optimiser = keras.optimizers.Adam(learning_rate=learning_rate * dev_multiplier,
                                          beta_1=Cfg.val['beta1'],
                                          beta_2=Cfg.val['beta2'],
                                          epsilon=Cfg.val['epsilon'])
        self.cnn_arch.compile(loss='mse',
                              optimizer=optimiser,
                              metrics=['mae'])
    else:
      # Single GPU / CPU training
      dev_multiplier = 1
      self._construct(d_inp.shape[1:],d_out.shape[1:])
      optimiser = keras.optimizers.Adam(learning_rate=learning_rate,
                                        beta_1=Cfg.val['beta1'],
                                        beta_2=Cfg.val['beta2'],
                                        epsilon=Cfg.val['epsilon'])
      self.cnn_arch.compile(loss='mse',
                            optimizer=optimiser,
                              metrics=['mae'])

    # Calculate FLOPs
    self.flops = calculate_flops(self.cnn_arch, d_inp.shape[1:])

    for dpi in image_dpi:
      plot_model(self.cnn_arch,
                 to_file=os.path.join(folder,'architecture@'+str(dpi)+'.png'),
                 show_shapes=True,
                 show_dtype=True,
                 show_layer_names=True,
                 rankdir='TB',
                 expand_nested=True,
                 dpi=dpi)
    if verbose > 0:
      self.cnn_arch.summary()

    timer = TimeHistory(epochs)
    callbacks = [keras.callbacks.EarlyStopping(monitor='loss',
                                               min_delta=1e-8,
                                               patience=25,
                                               mode='min',
                                               verbose=(verbose > 0),
                                               restore_best_weights=True),
                 timer]

    # Dataset options
    # Robust against TF AutoShardPolicy changes on single-machine multi-GPU
    # Set tf.data sharding to OFF and apply options to both train and validation datasets. This avoids recent AutoShardPolicy regressions on single-machine multi-GPU while keeping behavior stable on CPU/single-GPU.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_data = train_data.batch(batch_size * dev_multiplier).with_options(options)
    if validation_data is not None:
      validation_data = validation_data.batch(batch_size * dev_multiplier).with_options(options)

    # Train
    history = self.cnn_arch.fit(train_data,
                                validation_data=validation_data,
                                epochs=epochs,
                                verbose=(verbose > 0)*2,
                                shuffle=True,
                                callbacks=callbacks)
    le = len(history.history['loss'])
    history.history['time (ms)'] = np.add(timer.times[:le,1],-timer.times[:le,0]) // 1000000

    if verbose > 0:
      print("# Evaluating")
    d_score = self.cnn_arch.evaluate(d_inp, d_out, verbose=(verbose > 0)*2)
    if v_data is not None:
      v_score = self.cnn_arch.evaluate(v_inp, v_out, verbose=(verbose > 0)*2)
    else:
      v_score = np.array([np.nan,np.nan])
    if verbose > 0:
      print("      Train          Validation")
      print(f"MSE:  {d_score[0]:.12f} {v_score[0]:.12f}")
      print(f"MAE:  {d_score[1]:.12f} {v_score[1]:.12f}")
    self._save_results(folder, history.history, d_score, v_score, image_dpi, screen_dpi, verbose)

    d_res={"MSE":d_score[0],"MAE":d_score[1]}
    v_res={"MSE":v_score[0],"MAE":v_score[1]}
    return d_res, v_res

  def predict(self, d_inp, reshape=True, verbose=0):
    """Make predictions on input data.

    Parameters
    ----------
        d_inp (numpy.ndarray): Input data tensor
        reshape (bool, optional): Whether to reshape input data. Defaults to True
        verbose (int, optional): Verbosity level. Defaults to 0

    Returns
    -------
        numpy.ndarray: Predicted concentrations
    """
    if reshape:
      d_inp = tf.convert_to_tensor(d_inp, dtype=tf.float32)
      d_inp = reshape_spectra_data(d_inp, add_channel_dim=True)

    # Dataset options
    # Robust against TF AutoShardPolicy changes on single-machine multi-GPU
    # Set tf.data sharding to OFF and apply options to both train and validation datasets. This avoids recent AutoShardPolicy regressions on single-machine multi-GPU while keeping behavior stable on CPU/single-GPU.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    data = tf.data.Dataset.from_tensor_slices(d_inp).batch(32).with_options(options)
    return np.array(self.cnn_arch.predict(data,verbose=(verbose>0)*2),dtype=np.float64)

  def save(self, folder):
    """Save the trained CNN model to disk.

    Parameters
    ----------
        folder (str): Directory to save the model
    """
    os.makedirs(folder, exist_ok=True)
    self.cnn_arch.save(os.path.join(folder, "model.keras"))

    # Calculate model metrics from summary
    # Capture model summary output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    self.cnn_arch.summary()
    sys.stdout = old_stdout
    summary_output = buffer.getvalue()

    # Parse parameter counts from summary
    trainable_params = 0
    non_trainable_params = 0
    for line in summary_output.split('\n'):
      if 'Total params:' in line:
        # Extract total params
        total_match = line.split('Total params:')[1].split()[0].replace(',', '')
        total_params = int(total_match)
      elif 'Trainable params:' in line:
        # Extract trainable params
        trainable_match = line.split('Trainable params:')[1].split()[0].replace(',', '')
        trainable_params = int(trainable_match)
      elif 'Non-trainable params:' in line:
        # Extract non-trainable params
        non_trainable_match = line.split('Non-trainable params:')[1].split()[0].replace(',', '')
        non_trainable_params = int(non_trainable_match)
    # If we couldn't parse trainable/non-trainable separately, use total
    if trainable_params == 0 and non_trainable_params == 0:
      trainable_params = total_params
      non_trainable_params = 0

    # Calculate FLOPs if possible
    if hasattr(self, 'flops'):
      flops = self.flops
    else:
      flops = 0

    with open(os.path.join(folder, "mrsnet.json"), 'w') as f:
      print(json.dumps({
          'model': self.model,
          'metabolites': self.metabolites,
          'pulse_sequence': self.pulse_sequence,
          'acquisitions': self.acquisitions,
          'datatype': self.datatype,
          'norm': self.norm,
          'train_dataset_name': self.train_dataset_name,
          'trainable_params': trainable_params,
          'non_trainable_params': non_trainable_params,
          'total_params': trainable_params + non_trainable_params,
          'flops': flops
        }, indent=2, sort_keys=True), file=f)

  @staticmethod
  def load(path):
    """Load a trained CNN model from disk.

    Parameters
    ----------
        path (str): Directory containing the saved model

    Returns
    -------
        CNN: Loaded CNN model instance
    """
    with open(os.path.join(path, "mrsnet.json")) as f:
      data = json.load(f)
    model = CNN(data['model'], data['metabolites'], data['pulse_sequence'], data['acquisitions'],
                data['datatype'], data['norm'])
    model.train_dataset_name = data['train_dataset_name']
    model.cnn_arch = load_model(os.path.join(path, "model.keras"))
    return model

  def _save_results(self, folder, history, d_score, v_score, image_dpi, screen_dpi, verbose):
    """Save training results to files.

    Parameters
    ----------
        folder (str): Output folder for results
        history (dict): Training history
        d_score (list): Training scores
        v_score (list): Validation scores
        image_dpi (list): Image DPI settings
        screen_dpi (int): Screen DPI setting
        verbose (int): Verbosity level
    """
    keys = sorted(history.keys())
    # History data
    with open(os.path.join(folder, 'history.csv'), "w") as out_file:
      writer = csv.writer(out_file, delimiter=",")
      writer.writerows([[self.model+" Training Results"],
                        [""],
                        ["",     "Train",    "Validation"],
                        ["MSE",  d_score[0], v_score[0]],
                        ["MAE",  d_score[1], v_score[1]],
                        [""],
                        ["History"]])
      writer.writerow(keys)
      writer.writerows(zip(*[history[key] for key in keys], strict=False))
    # Plot
    fig, axes = plt.subplots(1, 3)
    fig.suptitle(f"{self.model} Training Results")
    for key in keys:
      if 'mse' in key or 'loss' in key:
        axes[0].semilogy(history[key], label=key)
        axes[0].set_ylabel('MSE')
        axes[0].legend(loc='upper right')
      if 'mae' in key:
        axes[1].semilogy(history[key], label=key)
        axes[1].set_ylabel('MAE')
        axes[1].legend(loc='upper right')
      if 'time' in key:
        axes[2].plot(history[key], label=key)
        axes[2].set_ylabel('Time (ms)')
        axes[2].legend(loc='upper right')
    for dpi in image_dpi:
      plt.savefig(os.path.join(folder, 'history@'+str(dpi)+'.png'), dpi=dpi)
    if verbose > 1:
      fig.set_dpi(screen_dpi)
      plt.show(block=True)
    plt.close()

class TimeHistory(keras.callbacks.Callback):
  """Callback to track training time per epoch.

  Records the start and end time of each epoch for performance monitoring.
  """

  def __init__(self, epochs):
    """Initialize time history callback.

    Parameters
    ----------
        epochs (int): Total number of epochs to track
    """
    super().__init__()
    self.counter = 0
    self.times = np.zeros((epochs,2),dtype=np.int64)

  def on_train_begin(self, logs=None):
    """Call at the beginning of training.

    Parameters
    ----------
        logs (dict, optional): Training logs. Defaults to None
    """
    self.counter = 0

  def on_epoch_begin(self, batch, logs=None):
    """Call at the beginning of each epoch.

    Parameters
    ----------
        batch (int): Batch number
        logs (dict, optional): Training logs. Defaults to None
    """
    self.times[self.counter,0] = time_ns()

  def on_epoch_end(self, batch, logs=None):
    """Call at the end of each epoch.

    Parameters
    ----------
        batch (int): Batch number
        logs (dict, optional): Training logs. Defaults to None
    """
    self.times[self.counter,1] = time_ns()
    self.counter += 1
