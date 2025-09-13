# mrsnet/ae_quantifier.py - MRSNet - autoencoder-quantification model
#
# SPDX-FileCopyrightText: Copyright (C) 2022-2025 Zien Ma, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2022-2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Autoencoder-quantifier models for MRSNet.

This module provides combined autoencoder-quantifier models that can both
reconstruct spectra and predict metabolite concentrations simultaneously.
"""

import csv
import io
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model

from mrsnet.cfg import Cfg
from mrsnet.cnn import TimeHistory
from mrsnet.train import calculate_flops, reshape_spectra_data


# Helper to construct dense layer
def _dense_layer(m,units, activation, dropout, name=None):
  """Construct a dense layer using functional API.

  Parameters
  ----------
      m (tensor): Input tensor
      units (int): Number of units in the dense layer
      activation (str): Activation function name
      dropout (float): Dropout rate (>0), batch normalization (=0), or nothing (<0)
      name (str, optional): Layer name. Defaults to None

  Returns
  -------
      tensor: Output tensor
  """
  if activation == "None":
    x = Dense(units,name=name)(m)
  else:
    x = Dense(units, activation=activation, name=name)(m)
  if dropout == 0.0:
    x = BatchNormalization()(x)
  elif dropout > 0.0:
    x = Dropout(dropout)(x)
  else:
    x = x
  return x

#  Fully connected autoencoder via Model interface (using Sequential interface internally)
class FCAutoEncQuant(Model):
  """Fully connected autoencoder-quantifier model.

  This class implements a fully connected autoencoder with an integrated
  quantifier for both spectrum reconstruction and concentration prediction.
  """

  def __init__(self, n_specs, n_freqs, layers_enc, layers_dec, activation, activation_last, dropout, output_conc, unit, layers, act, act_last, dp, name='FCAutoEncQuant'):
    """Initialize fully connected autoencoder-quantifier.

    Parameters
    ----------
        n_specs (int): Number of spectra (acquisitions x datatype)
        n_freqs (int): Number of frequency bins in spectra
        layers_enc (int): Number of encoder layers
        layers_dec (int): Number of decoder layers
        activation (str): Activation function for encoder/decoder
        activation_last (str): Activation function for decoder output
        dropout (float): Dropout rate for encoder/decoder
        output_conc (int): Number of output concentrations
        unit (int): Number of units in first quantifier layer
        layers (int): Number of quantifier layers
        act (str): Activation function for quantifier
        act_last (str): Activation function for quantifier output
        dp (float): Dropout rate for quantifier
        name (str, optional): Model name. Defaults to 'FCAutoEncQuant'
    """
    # n_specs: number of spectra (acquisisions x datatype)
    # n_freqs: number of frequency bins in spectra
    # layers_enc: number of layers in encoder
    # layers_dec: number of layers in encoder
    # activation: activation function (relu, sigmoid, tanh)
    # dropout: Dropout if > 0.0; 0.0, BatchNormalisation; negative, nothing
    self.n_specs = n_specs
    super().__init__(name=name)

    # Encoder
    encoder_input = keras.Input(shape=(n_specs,n_freqs),  name = "spectra_input")
    units = n_freqs
    x = _dense_layer(encoder_input, units, activation, dropout)
    units //=2
    for _l in range(0,layers_enc-2):
      x = _dense_layer(x, units, activation, dropout)
      units //= 2
    self.units = units

    encoder_output = _dense_layer(x, units, activation, -1) # no regulariser at latent representation

    # Decoder
    units = n_freqs // (2 ** (layers_dec-1))
    x = _dense_layer(encoder_output, units, activation, -1)
    units = n_freqs // (2 ** (layers_dec-2))
    for _l in range(0, layers_dec-2):
      x = _dense_layer(x, units, activation, -1)
      units *= 2
    self.units = units
    decoder_output = _dense_layer(x, units, activation_last, -1, "decoder_output")

    #Quantifier
    x = Flatten()(encoder_output)
    for _l in range(0, layers-1):
      x = _dense_layer(x, unit, act, dp)
      unit //= 2
    # FIXME: Still struggling with the quantifier architecture design, minor problem but could imporve the efficiency
    quantifier_output = _dense_layer(x, output_conc, act_last, -1, "quantifier_output")

    self.model = keras.Model(inputs=encoder_input, outputs=[decoder_output,quantifier_output],name="AEQ")
    self.model.build((None, n_specs, n_freqs))
    #self.model.summary()

# Autoencoder model
class AutoencoderQuantifier:
  """Autoencoder-quantifier model for MRSNet.

  This class implements a combined autoencoder-quantifier model that can both
  reconstruct spectra and predict metabolite concentrations simultaneously.

  Attributes
  ----------
      model (str): Model architecture identifier
      metabolites (list): List of metabolite names
      pulse_sequence (str): Pulse sequence type
      acquisitions (list): List of acquisition types
      datatype (list): List of data types
      norm (str): Normalization method
      low_ppm (float): Lower PPM bound for input data
      high_ppm (float): Upper PPM bound for input data
      fft_samples (int): Number of FFT samples
      train_dataset_name (str): Name of training dataset
      aeq (keras.Model): The actual autoencoder-quantifier model
  """

  def __init__(self, model, metabolites, pulse_sequence, acquisitions, datatype,norm):
    """Initialize an autoencoder-quantifier model.

    Parameters
    ----------
        model (str): Model architecture identifier
        metabolites (list): List of metabolite names
        pulse_sequence (str): Pulse sequence type
        acquisitions (list): List of acquisition types
        datatype (list): List of data types
        norm (str): Normalization method
    """
    self.model = model
    self.metabolites = metabolites
    self.pulse_sequence = pulse_sequence
    self.acquisitions = acquisitions
    self.datatype = datatype
    self.norm = norm
    # Input spectra data (constant!)
    self.low_ppm = -1.0
    self.high_ppm = -4.5
    self.fft_samples = 2048

    self.train_dataset_name = None
    self.aeq = None

  def __str__(self):
    """Return the model architecture string.

    Returns
    -------
        str: Model architecture string
    """
    return os.path.join(self.model, "-".join(self.metabolites),
                        self.pulse_sequence, "-".join(self.acquisitions),
                        "-".join(self.datatype), self.norm)

  def reset(self):
    """Reset the model to initial state."""
    del self.aeq
    self.aeq = None
    self.train_dataset_name = None

  def _construct(self, ae_shape, output_conc=None):
    """Construct the autoencoder-quantifier model architecture.

    Parameters
    ----------
        ae_shape (tuple): Shape of input data (n_specs, n_freqs)
        output_conc (int, optional): Number of output concentrations. Defaults to None
    """
    # Autoencoder only; for quantifier use convert_to_quantifier
    n_specs = ae_shape[0] # number of spectras: acqusitions x datatype
    n_freqs = ae_shape[1] # number of frequency bins in spectras
    p = self.model.split("_")
    if p[0] == "caeq":
      # Construct actual encoder-quantifier architecture
      if p[1] == "fc":
        # Convert trained autoencoder to trainable quantifier
        # aeq_fc_UNITS_LAYERS_ACT[_ACT-LAST]
        # FIXME: more arguments - DO (dropout/BatchNorm/Noting), etc.
        lin = int(p[2])
        lout = int(p[3])
        activation = p[4]
        activation_last = p[5]
        dropout = float(p[6])
        unit = int(p[7])
        layers = int(p[8])
        act = p[9]
        act_last = p[10]
        dp = float(p[11])
        self.aeq = FCAutoEncQuant(n_specs,n_freqs,lin,lout,activation,activation_last,dropout,output_conc,unit,layers,act,act_last,dp).model
        #For example: FCAEQ = FCAutoEncQuant(2, 2048, 5, 7, "tanh", "tanh", 0.3, 5, 256, 4, "tanh", "tanh", 0.3)
      else:
        raise RuntimeError(f"Unknown encoder-quantifier architecture {self.model}")

  def train(self, d_data, v_data, epochs, batch_size, folder, verbose=0,
              image_dpi=[300], screen_dpi=96, train_dataset_name=""):
      """Train the autoencoder-quantifier model.

      Parameters
      ----------
          d_data (list): Training data [spectra_in, spectra_out, concentrations]
          v_data (list, optional): Validation data [spectra_in, spectra_out, concentrations]
          epochs (int): Number of training epochs
          batch_size (int): Batch size for training
          folder (str): Output folder for results
          verbose (int, optional): Verbosity level. Defaults to 0
          image_dpi (list, optional): Image DPI settings. Defaults to [300]
          screen_dpi (int, optional): Screen DPI setting. Defaults to 96
          train_dataset_name (str, optional): Name of training dataset. Defaults to ""

      Returns
      -------
          tuple: (train_results, validation_results)
      """
      devices = tf.config.list_logical_devices("GPU")
      if len(devices) < 1:
        print("**WARNING, we do not have a GPU for Tensorflow!**")
        devices = []
      else:
        devices = [devices[l].name for l in range(0, len(devices))]
        if verbose > 0:
          print(f"GPU Devices: {devices}")
      if len(d_data) != 3:
        raise RuntimeError("d_data argument must be a list [spectra_in,spectra_out|conc]")
      if v_data is not None and len(v_data) != 3:
        raise RuntimeError("v_data argument must be a list [spectra_in,spectra_out|conc]")

      if len(train_dataset_name) > 0:
        self.train_dataset_name = train_dataset_name

      if not os.path.isdir(folder):
        os.makedirs(folder)

      if self.model[0:4] == "caeq":
        return self._train_aeq(d_data, v_data, epochs, batch_size, folder, verbose,
                               image_dpi, screen_dpi, train_dataset_name, devices)
      raise RuntimeError(f"Unknown autoencoder model {self.model}")

  def _train_aeq(self, d_data, v_data, epochs, batch_size, folder, verbose,
                image_dpi, screen_dpi, train_dataset_name, devices):
    """Train the autoencoder-quantifier model.

    Parameters
    ----------
        d_data (list): Training data [spectra_in, spectra_out, concentrations]
        v_data (list, optional): Validation data [spectra_in, spectra_out, concentrations]
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        folder (str): Output folder for results
        verbose (int): Verbosity level
        image_dpi (list): Image DPI settings
        screen_dpi (int): Screen DPI setting
        train_dataset_name (str): Name of training dataset
        devices (list): List of GPU devices

    Returns
    -------
        tuple: (train_results, validation_results)
    """
    # Validate data format
    if len(d_data) != 3:
      raise RuntimeError(f"AutoencoderQuantifier requires 3 data elements [spectra_in, spectra_out, concentrations], got {len(d_data)}")

    if v_data is not None and len(v_data) != 3:
      raise RuntimeError(f"AutoencoderQuantifier validation data requires 3 elements [spectra_in, spectra_out, concentrations], got {len(v_data)}")

    # Setup training data
    if verbose > 0:
      print("# Prepare data")
    # We reshape the (batch, acquisition, datatype, frequency) spectra tensor to
    # the channel (so channel is not last, as often assumed). That means we treat
    # (batch, acquisition x datatype, frequency) where the 2nd index is effectively
    # echo acquisition-dataypte signal as separate with a separate 1D network
    # (for now we do not have any operations crossing the channels, I think).
    #
    # FIXME: Something to consider: (also train_ae)
    # Instead, we could also reshape it to (batch x acquisition x datatype, frequency)
    # meaning each signal, idependent of acquistion and datatype, is handled
    # the same with the same network. We only have a collection of 1D signals for
    # the autoencoder - for quantification then this would need some more complex
    # reshaping as there we will have to consider the signals across acquisitons and
    # datatypes.

    # Input spectra (for autoencoder and quantifier)
    d_spectra_in = tf.convert_to_tensor(d_data[0], dtype=tf.float32)
    d_spectra_in = reshape_spectra_data(d_spectra_in)

    # Output spectra for autoencoder
    d_spectra_out = tf.convert_to_tensor(d_data[1], dtype=tf.float32)
    d_spectra_out = reshape_spectra_data(d_spectra_out)
    # Output concentrations for quantifier
    d_conc = tf.convert_to_tensor(d_data[-1], dtype=tf.float32)

    train_data_spectra = tf.data.Dataset.from_tensor_slices(d_spectra_in)
    train_data_target = tf.data.Dataset.from_tensor_slices((d_spectra_out, d_conc))

    if v_data is not None:
        v_spectra_in = tf.convert_to_tensor(v_data[0], dtype=tf.float32)
        v_spectra_in = reshape_spectra_data(v_spectra_in)

        v_spectra_out = tf.convert_to_tensor(v_data[1], dtype=tf.float32)
        v_spectra_out = reshape_spectra_data(v_spectra_out)
        v_conc = tf.convert_to_tensor(v_data[-1], dtype=tf.float32)
        val_data_spectra = tf.data.Dataset.from_tensor_slices(v_spectra_in)
        val_data_target = tf.data.Dataset.from_tensor_slices((v_spectra_out, v_conc))

    else:
        val_data_spectra = None
        val_data_target = None
    if verbose > 0:
        print("  Spectra In:",d_spectra_in.shape,"[spectrum, acquisition x datatype, frequency]")
        print("  Spectra Out:", d_spectra_out.shape, "[spectrum, acquisition x datatype, frequency]")
        print("  Concentrations Out:",d_conc.shape,"[spectrum, metabolite_concentration]")

    # Autoencoder Quantification Network training
    if verbose > 0:
      print(f"# Train Autoencoder Quantification Network {self!s}")

    learning_rate = Cfg.val['base_learning_rate'] * batch_size / 16.0
    loss = keras.losses.Huber(name='huber_loss')

    if len(devices) > 1:
      # Multi-GPU training
      dev_multiplier = len(devices)
      mirrored_strategy = tf.distribute.MirroredStrategy(devices=devices)
      with mirrored_strategy.scope():
        self._construct(d_spectra_in.shape[1:], output_conc=d_conc.shape[1])
        self.output = "concentrations"
        optimiser = keras.optimizers.Adam(learning_rate=learning_rate * dev_multiplier,
                                          beta_1=Cfg.val['beta1'],
                                          beta_2=Cfg.val['beta2'],
                                          epsilon=Cfg.val['epsilon'])
        self.aeq.compile(loss=[loss,loss],
                        optimizer=optimiser,
                        metrics=['mae'])
    else:
      # Single GPU / CPU training
      dev_multiplier = 1
      self._construct(d_spectra_in.shape[1:], output_conc=d_conc.shape[1])
      self.output = "concentrations"
      optimiser = keras.optimizers.Adam(learning_rate=learning_rate,
                                        beta_1=Cfg.val['beta1'],
                                        beta_2=Cfg.val['beta2'],
                                        epsilon=Cfg.val['epsilon'])
      self.aeq.compile(loss=[loss,loss],
                      optimizer=optimiser,
                      metrics=['mae'])

    # Calculate FLOPs
    self.flops = calculate_flops(self.aeq, d_spectra_in.shape[1:])

    for dpi in image_dpi:
        plot_model(self.aeq,
                  to_file=os.path.join(folder,'architecture-quantification-network@'+str(dpi)+'.png'),
                  show_shapes=True,
                  show_dtype=True,
                  show_layer_names=True,
                  rankdir='TB',
                  expand_nested=True,
                  dpi=dpi)

    if verbose > 0:
        self.aeq.summary()


    timer = TimeHistory(epochs)
    callbacks = [keras.callbacks.EarlyStopping(monitor='loss',
                                                 min_delta=1e-8,
                                                 patience=25,
                                                 mode='min',
                                                 verbose=(verbose > 0),
                                                 restore_best_weights=True),
                   timer]

    # Dataset options
    train_data=tf.data.Dataset.zip((train_data_spectra,train_data_target))
    if v_data is not None:
        val_data=tf.data.Dataset.zip((val_data_spectra, val_data_target))
    else:
        val_data=None

    # Dataset options
    # Robust against TF AutoShardPolicy changes on single-machine multi-GPU
    # Set tf.data sharding to OFF and apply options to both train and validation datasets. This avoids recent AutoShardPolicy regressions on single-machine multi-GPU while keeping behavior stable on CPU/single-GPU.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_data = train_data.batch(batch_size * dev_multiplier).with_options(options)
    if val_data is not None:
        val_data = val_data.batch(batch_size * dev_multiplier).with_options(options)

    # Train
    history = self.aeq.fit(train_data,
                          validation_data=val_data,
                          epochs=epochs,
                          verbose=(verbose > 0)*2,
                          shuffle=True,
                          callbacks=callbacks)
    le = len(history.history['loss'])
    history.history['time (ms)'] = np.add(timer.times[:le,1],-timer.times[:le,0]) // 1000000

    if verbose > 0:
      print("# Evaluating Quantifier")
    d_score = self.aeq.evaluate(d_spectra_in, (d_spectra_out,d_conc), verbose=(verbose > 0)*2)
    if v_data is not None:
      v_score = self.aeq.evaluate(v_spectra_in, (v_spectra_out,v_conc), verbose=(verbose > 0)*2)
    else:
      v_score = np.array([np.nan,np.nan,np.nan,np.nan,np.nan])
    if verbose > 0:
      print( "             Train          Validation")
      print(f"TOTAL_HUBER_LOSS:  {d_score[0]:.12f} {v_score[0]:.12f}")
      print(f"AE_LOSS       :  {d_score[1]:.12f} {v_score[1]:.12f}")
      print(f"Q_LOSS       :  {d_score[2]:.12f} {v_score[2]:.12f}")
      print(f"AE_MAE       :  {d_score[3]:.12f} {v_score[3]:.12f}")
      print(f"Q_MAE       :  {d_score[4]:.12f} {v_score[4]:.12f}")
    self._save_results(folder, "cae_quantifier", history.history, d_score, v_score, loss, image_dpi, screen_dpi, verbose)

    d_res = {loss:d_score[2],"MAE":d_score[4]}
    v_res = {loss:v_score[2],"MAE":v_score[4]}

    return d_res, v_res

  def predict(self, spec_in, reshape=True, verbose=0):
    """Predict concentrations or spectra from input spectra.

    Parameters
    ----------
        spec_in (array-like): Input spectra data
        reshape (bool, optional): Whether to reshape input data. Defaults to True
        verbose (int, optional): Verbosity level. Defaults to 0

    Returns
    -------
        numpy.ndarray: Predicted concentrations or spectra
    """
    out_shape = spec_in.shape # Preserve shape of input spectra to reshape output (spectra only) accordingly
    if reshape:
      spec_in = tf.convert_to_tensor(spec_in, dtype=tf.float32)
      spec_in = tf.reshape(spec_in,(spec_in.shape[0],spec_in.shape[1]*spec_in.shape[2],spec_in.shape[3]))
    # Dataset options
    # Robust against TF AutoShardPolicy changes on single-machine multi-GPU
    # Set tf.data sharding to OFF and apply options to both train and validation datasets. This avoids recent AutoShardPolicy regressions on single-machine multi-GPU while keeping behavior stable on CPU/single-GPU.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    data = tf.data.Dataset.from_tensor_slices(spec_in).batch(32).with_options(options)
    if self.output == "concentrations":
        return np.array(self.aeq.predict(data, verbose=(verbose > 0) * 2)[1], dtype=np.float64)
    if self.output == "spectra":
        return np.array(tf.reshape(self.aeq.predict(data, verbose=(verbose > 0) * 2)[0], out_shape), dtype=np.float64)
    raise RuntimeError(f"Unknown output {self.output}")

  def save(self, folder):
    """Save the trained model to disk.

    Parameters
    ----------
        folder (str): Directory to save the model
    """
    os.makedirs(folder, exist_ok=True)
    self.aeq.save(os.path.join(folder, "model.keras"))

    # Calculate model metrics from summary
    # Capture model summary output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    self.aeq.summary()
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
          'output': self.output,
          'trainable_params': trainable_params,
          'non_trainable_params': non_trainable_params,
          'total_params': trainable_params + non_trainable_params,
          'flops': flops
        }, indent=2, sort_keys=True), file=f)

  @staticmethod
  def load(path):
      """Load a saved model from disk.

      Parameters
      ----------
          path (str): Directory containing the saved model

      Returns
      -------
          AutoencoderQuantifier: Loaded model instance
      """
      with open(os.path.join(path, "mrsnet.json")) as f:
          data = json.load(f)
      model = AutoencoderQuantifier(data['model'], data['metabolites'], data['pulse_sequence'], data['acquisitions'],
                          data['datatype'], data['norm'])
      model.output = data['output']
      model.train_dataset_name = data['train_dataset_name']
      model.aeq = load_model(os.path.join(path, "model.keras"))
      return model

  def _save_results(self, folder, prefix, history, d_score, v_score, loss, image_dpi, screen_dpi, verbose):
      """Save training results to files.

      Parameters
      ----------
          folder (str): Output directory
          prefix (str): File prefix for saved files
          history (dict): Training history
          d_score (list): Training scores
          v_score (list): Validation scores
          loss (str): Loss function name
          image_dpi (list): Image DPI settings
          screen_dpi (int): Screen DPI setting
          verbose (int): Verbosity level
      """
      keys = sorted(history.keys())
      # History data
      with open(os.path.join(folder, prefix + '_history.csv'), "w") as out_file:
          writer = csv.writer(out_file, delimiter=",")
          if self.model[0:4] == "caeq":
              writer.writerows([[self.model + " " + prefix.upper() + " Training Results", "", "", "", ""],
                                [""],
                                ["", "Train", "Validation"],
                                ["TOTAL_LOSS", d_score[0], v_score[0]],
                                ["AE_LOSS",d_score[1], v_score[1]],
                                ["AEQ_LOSS", d_score[2], v_score[2]],
                                ["AE_MAE",d_score[3], v_score[3]],
                                ["AEQ_MAE", d_score[4], v_score[4]],
                                [""],
                                ["History"]])
          else:
              writer.writerows(
                  [[self.model + " " + prefix.upper() + " Training Results"],
                   [""],
                   ["", "Train", "Validation"],
                   [loss.name.upper(), d_score[0], v_score[0]],
                   ["MAE", d_score[1], v_score[1]],
                   [""],
                   ["History"]])
          writer.writerow(keys)
          writer.writerows(zip(*[history[key] for key in keys], strict=False))
      # Plot
      fig, axes = plt.subplots(1, 3)
      fig.suptitle(f"{self.model} {prefix.upper()} Training Results")
      for key in keys:
          if loss in key or 'loss' in key:
              axes[0].semilogy(history[key], label=key)
              axes[0].set_ylabel(loss.name.upper())
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
          plt.savefig(os.path.join(folder, prefix + '_history@' + str(dpi) + '.png'), dpi=dpi)
      if verbose > 1:
          fig.set_dpi(screen_dpi)
          plt.show(block=True)
      plt.close()
