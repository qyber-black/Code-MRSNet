# mrsnet/autoencoder.py - MRSNet - autoencoder model
#
# SPDX-FileCopyrightText: Copyright (C) 2022-2025 Zien Ma, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2022-2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Autoencoder models for MRSNet.

This module provides various autoencoder architectures for MRS spectra,
including fully connected autoencoders and encoder-quantifier models
for concentration prediction.
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
from tensorflow.keras.layers import (
  Activation,
  BatchNormalization,
  Dense,
  Dropout,
  Flatten,
  LeakyReLU,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model

try:
  from keras.saving import register_keras_serializable
except Exception:
  from tensorflow.keras.utils import register_keras_serializable

from mrsnet.cfg import Cfg
from mrsnet.cnn import TimeHistory
from mrsnet.train import calculate_flops

# Global load context for subclassed model deserialization
_FCAE_LOAD_CTX = None

def _dense_layer(x, units, activation, dropout, name=None):
  """Construct a dense layer with optional batch normalization and dropout using functional API.

  Parameters
  ----------
      x (tensor): Input tensor
      units (int): Number of units in the dense layer
      activation (str): Activation function name
      dropout (float): Dropout rate (>0), batch normalization (=0), or nothing (<0)
      name (str, optional): Layer name. Defaults to None

  Returns
  -------
      tensor: Output tensor
  """
  # Dense layer over last index (freq. bins) using given activation
  # if dropout > 0: dropout rate after activation;
  #            ==0: batch normalisation before activation;
  #            < 0: nothing
  x = Dense(units, name=name)(x)
  if dropout == 0.0:
    x = BatchNormalization()(x)
  if activation.startswith("leakyrelu") and len(activation) > 9:
    try:
      alpha = float(activation[9:])
      x = LeakyReLU(alpha=alpha)(x)
    except ValueError as err:
      raise ValueError(f"Invalid LeakyReLU alpha value: {activation[9:]}") from err
  elif activation != "None":
    # If None, create layer without activation function
    x = Activation(activation)(x)
  if dropout > 0.0:
    x = Dropout(dropout)(x)
  return x

@register_keras_serializable(package="mrsnet", name="FCAutoEnc")
class FCAutoEnc(Model):
  """Fully connected autoencoder model for MRS spectra.

  This class implements a fully connected autoencoder using dense layers
  for MRS spectra reconstruction.

  Attributes
  ----------
      encoder (keras.Model): Encoder network
      decoder (keras.Model): Decoder network
  """

  def __init__(self, n_specs, n_freqs, layers_enc, layers_dec, activation, activation_last, dropout, name='FCAutoEnc', **kwargs):
    """Initialize a fully connected autoencoder.

    Parameters
    ----------
        n_specs (int): Number of spectra (acquisitions x datatype)
        n_freqs (int): Number of frequency bins in spectra
        layers_enc (list): List of encoder layer sizes
        layers_dec (list): List of decoder layer sizes
        activation (str): Activation function for hidden layers
        activation_last (str): Activation function for output layer
        dropout (float): Dropout rate
        name (str, optional): Model name. Defaults to 'FCAutoEnc'
    """
    self.n_specs = n_specs
    super().__init__(name=name, **kwargs)

    # Input
    encoder_input = keras.Input(shape=(n_specs, n_freqs), name="spectra_input")

    # Encoder
    units = n_freqs
    x = encoder_input
    for _l in range(0,layers_enc-1):
      x = _dense_layer(x, units, activation, dropout)
      units //= 2
    self.units = units
    encoder_output = _dense_layer(x, units, activation, -1, name="latent") # no regulariser at latent representation

    # Decoder
    units = n_freqs//(2**(layers_dec-1))
    x = encoder_output
    for _l in range(0,layers_dec-1):
      x = _dense_layer(x, units, activation, -1) # no regularisers in decoder
      units *= 2
    decoder_output = _dense_layer(x, units, activation_last, -1, name="decoder_output")

    # Create models
    self.encoder = keras.Model(inputs=encoder_input, outputs=encoder_output, name='Encoder')
    self.decoder = keras.Model(inputs=encoder_output, outputs=decoder_output, name='Decoder')

    # Build the full model
    self.build((None, n_specs, n_freqs)) # Build encoder and decoder

  @classmethod
  def from_config(cls, config):
    """Create FCAutoEnc instance from configuration during model deserialization.

    This method is called by Keras during model loading to reconstruct the
    FCAutoEnc instance from saved configuration. It relies on a global load
    context (_FCAE_LOAD_CTX) that provides the necessary hyperparameters.

    Parameters
    ----------
        config (dict): Configuration dictionary containing model metadata

    Returns
    -------
        FCAutoEnc: New FCAutoEnc instance with parameters from load context

    Raises
    ------
        TypeError: If no load context is available during deserialization
    """
    # Use hyperparameters provided by loader context
    global _FCAE_LOAD_CTX
    ctx = _FCAE_LOAD_CTX
    if ctx is None:
      raise TypeError("FCAutoEnc cannot be deserialized without load context")
    return cls(
      ctx['n_specs'], ctx['n_freqs'],
      ctx['layers_enc'], ctx['layers_dec'],
      ctx['activation'], ctx['activation_last'], ctx['dropout'],
      name=config.get('name','FCAutoEnc')
    )

  def call(self,x):
    """Forward pass through the autoencoder.

    Parameters
    ----------
        x (tensor): Input spectra tensor

    Returns
    -------
        tensor: Reconstructed spectra tensor
    """
    x = self.encoder(x)
    x = self.decoder(x)
    return x

@register_keras_serializable(package="mrsnet", name="EncQuant")
class EncQuant(Model):
  """Encoder-quantifier model for concentration prediction.

  This class combines a pre-trained encoder with a quantifier network
  for predicting metabolite concentrations from MRS spectra.
  """

  def __init__(self, encoder, n_specs, n_freqs, output_conc, units, layers, act, act_last, dp, name='EncQuant', **kwargs):
    """Initialize encoder-quantifier model.

    Parameters
    ----------
        encoder: Pre-trained encoder model
        n_specs (int): Number of spectra (acquisitions x datatype)
        n_freqs (int): Number of frequency bins in spectra
        output_conc (int): Number of output concentrations
        units (int): Number of units in first dense quantifier layer
        layers (int): Number of dense layers in quantifier
        act (str): Activation function for internal layers
        act_last (str): Activation function for output layer
        dp (float): Dropout rate
        name (str, optional): Model name. Defaults to 'EncQuant'
    """
    self.n_specs = n_specs
    super().__init__(name=name, **kwargs)

    # Encoder
    self.encoder = encoder
    self.encoder.trainable = False # FIXME: we cold continue training the encoder when the quantifier is trained

    # Quantifier
    q_in = keras.Input(shape=encoder.output_shape[1:], name="q_in")
    x = Flatten()(q_in)
    for _l in range(0, layers-1):
      x = _dense_layer(x, units, act, dp)
      units //= 2
    q_out = _dense_layer(x, output_conc, act_last, -1, name="q_out")

    self.quantifier = keras.Model(inputs=q_in, outputs=q_out, name='Quantifier')

    self.build((None, n_specs, n_freqs))

  def call(self,x):
    """Forward pass through the encoder-quantifier.

    Parameters
    ----------
        x (tensor): Input spectra tensor

    Returns
    -------
        tensor: Predicted concentrations tensor
    """
    x = self.encoder(x)
    x = self.quantifier(x)
    return x

# Autoencoder model
class Autoencoder:
  """Main autoencoder interface for MRSNet.

  This class provides a unified interface for both autoencoder and encoder-quantifier
  models, handling model construction, training, and prediction.

  Attributes
  ----------
      model (str): Model architecture identifier
      metabolites (list): List of metabolite names
      pulse_sequence (str): Pulse sequence type
      acquisitions (list): List of acquisition types
      datatype (list): List of data types
      norm (str): Normalization method
      output (str): Output type ("spectra" or "concentrations")
      encoder (keras.Model, optional): Pre-trained encoder for quantifier mode
      autoencoder_model (str, optional): Autoencoder model name for quantifier mode
      autoencoder_train_dataset_name (str, optional): Training dataset name for encoder
      low_ppm (float): Lower PPM bound for input data
      high_ppm (float): Upper PPM bound for input data
      fft_samples (int): Number of FFT samples
      train_dataset_name (str): Name of training dataset
      autoencoder_arch (keras.Model): The actual autoencoder model
  """

  def __init__(self, model, metabolites, pulse_sequence, acquisitions, datatype, norm,
               encoder=None, encoder_model=None, encoder_train_dataset_name=None):
    """Initialize an autoencoder model.

    Parameters
    ----------
        model (str): Model architecture identifier
        metabolites (list): List of metabolite names
        pulse_sequence (str): Pulse sequence type
        acquisitions (list): List of acquisition types
        datatype (list): List of data types
        norm (str): Normalization method
        encoder (keras.Model, optional): Pre-trained encoder for quantifier mode
        encoder_model (str, optional): Autoencoder model name for quantifier mode
        encoder_train_dataset_name (str, optional): Training dataset name for encoder
    """
    self.model = model
    self.metabolites = metabolites
    self.pulse_sequence = pulse_sequence
    self.acquisitions = acquisitions
    self.datatype = datatype
    self.norm = norm
    if encoder is not None:
      # Quantifier
      self.encoder = encoder
      self.autoencoder_model = encoder_model
      self.autoencoder_train_dataset_name = encoder_train_dataset_name
      self.output = "concentrations"
    else:
      # Autoencoder
      self.output = "spectra"

    # Input spectra data (constant!)
    self.low_ppm = -1.0
    self.high_ppm = -4.5
    self.fft_samples = 2048

    self.train_dataset_name = None
    self.ae = None
    self.ae_path = None

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
    self.ae = None
    self.train_dataset_name = None

  def _construct(self, ae_shape, output_conc=None):
    """Construct the autoencoder model architecture.

    Parameters
    ----------
        ae_shape (tuple): Shape of input data (n_specs, n_freqs)
        output_conc (int, optional): Number of output concentrations. Defaults to None
    """
    n_specs = ae_shape[0] # number of spectras: acqusitions x datatype
    n_freqs = ae_shape[1] # number of frequency bins in spectras
    p = self.model.split("_")
    if p[0] == "aeq":
      # Construct actual encoder-quantifier architecture
      if p[1] == "fc":
        # Consrtuct quantifier from trained (auto)encoder
        # aeq_fc_UNITS_LAYERS_ACT_ACT-LAST_DO
        units = int(p[2])
        layers = int(p[3])
        act = p[4]
        act_last = p[5]
        dropout = float(p[6])
        self.ae = EncQuant(self.encoder,n_specs,n_freqs,output_conc,units,layers,act,act_last,dropout)
      else:
        raise RuntimeError(f"Unknown encoder-quantifier architecture {self.model}")
    elif p[0] == "ae":
      if p[1] == "fc":
        # ae_fc_[LIN]_[LOUT]_[ACT]_[ACT-LAST]_[DO]
        #    LIN: dense layers in encoder
        #    LOUT: dense layers in decoder
        #    ACT: activation function (relu, sigmoid, tanh, ...)
        #    ACT-LAST: last activation function (relu, sigmoid, tanh, ...)
        #    DO: Dropout if > 0.0; 0.0, BatchNormalisation; negative, no regulariser
        lin = int(p[2])
        lout = int(p[3])
        act = p[4]
        act_last = p[5]
        dropout = float(p[6])
        self.ae = FCAutoEnc(n_specs,n_freqs,lin,lout,act,act_last,dropout)
      else:
        raise RuntimeError(f"Unknown autoencoder variant: {p[1]}")
    else:
      raise RuntimeError(f"Not an autoencoder: {p[0]}")

  def train(self, d_data, v_data, epochs, batch_size, folder, verbose=0,
            image_dpi=[300], screen_dpi=96, train_dataset_name=""):
    """Train the autoencoder model.

    Parameters
    ----------
        d_data (list): Training data [spectra_in, spectra_out]
        v_data (list, optional): Validation data [spectra_in, spectra_out]
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
      devices = [devices[l].name for l in range(0,len(devices))]
      if verbose > 0:
        print(f"GPU Devices: {devices}")
    if len(d_data) != 2:
      raise RuntimeError("d_data argument must be a list [spectra_in,spectra_out|conc]")
    if v_data is not None and len(v_data) != 2:
      raise RuntimeError("v_data argument must be a list [spectra_in,spectra_out|conc]")

    if len(train_dataset_name) > 0:
      self.train_dataset_name = train_dataset_name

    if not os.path.isdir(folder):
      os.makedirs(folder)

    if self.model[0:3] == "ae_":
      return self._train_ae(d_data, v_data, epochs, batch_size, folder, verbose,
                            image_dpi, screen_dpi, train_dataset_name, devices)
    if self.model[0:4] == "aeq_":
      return self._train_aeq(d_data, v_data, epochs, batch_size, folder, verbose,
                             image_dpi, screen_dpi, train_dataset_name, devices)
    raise RuntimeError(f"Unknown autoencoder model {self.model}")

  def _train_ae(self, d_data, v_data, epochs, batch_size, folder, verbose,
                image_dpi, screen_dpi, train_dataset_name, devices):
    """Train the autoencoder model.

    Parameters
    ----------
        d_data (list): Training data [spectra_in, spectra_out]
        v_data (list, optional): Validation data [spectra_in, spectra_out]
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
    # Setup training data
    if verbose > 0:
      print("# Prepare data")
    # We reshape the (batch, acquisition, datatype, frequency) spectra tensor to
    # the channel (so channel is not last, as often assumed). That means we treat
    # (batch, acquisition x datatype, frequency) where the 2nd index is effectively
    # echo acquisition-dataypte signal as separate with a separate 1D network
    # (for now we do not have any operations crossing the channels, I think).

    # Input spectra (for autoencoder and quantifier)
    d_spectra_in = tf.convert_to_tensor(d_data[0], dtype=tf.float32)
    d_spectra_in = tf.reshape(d_spectra_in,
                              (d_spectra_in.shape[0],
                               d_spectra_in.shape[1]*d_spectra_in.shape[2],d_spectra_in.shape[3]))
    if v_data is not None:
      v_spectra_in = tf.convert_to_tensor(v_data[0], dtype=tf.float32)
      v_spectra_in = tf.reshape(v_spectra_in,
                                (v_spectra_in.shape[0],
                                 v_spectra_in.shape[1]*v_spectra_in.shape[2],v_spectra_in.shape[3]))

    # Output spectra for autoencoder
    d_spectra_out = tf.convert_to_tensor(d_data[1], dtype=tf.float32)
    d_spectra_out = tf.reshape(d_spectra_out,
                               (d_spectra_out.shape[0],
                                d_spectra_out.shape[1]*d_spectra_out.shape[2],d_spectra_out.shape[3]))
    train_data = tf.data.Dataset.from_tensor_slices((d_spectra_in, d_spectra_out))
    if v_data is not None:
      v_spectra_out = tf.convert_to_tensor(v_data[1], dtype=tf.float32)
      v_spectra_out = tf.reshape(v_spectra_out,
                                 (v_spectra_out.shape[0],
                                  v_spectra_out.shape[1]*v_spectra_out.shape[2],v_spectra_out.shape[3]))
      val_data = tf.data.Dataset.from_tensor_slices((v_spectra_in, v_spectra_out))
    else:
      val_data = None
    if verbose > 0:
      print("  Spectra In: ",d_spectra_in.shape,"[spectrum, acquisition x datatype, frequency]")
      print("  Spectra Out:",d_spectra_out.shape,"[spectrum, acquisition x datatype, frequency]")

    # Autoencoder training
    if verbose > 0:
      print(f"# Train Autoencoder {self!s}")

    learning_rate = Cfg.val['base_learning_rate'] * batch_size / 16.0
    loss = keras.losses.Huber(name='huber_loss')

    if len(devices) > 1:
      # Multi-GPU training
      dev_multiplier = len(devices)
      mirrored_strategy = tf.distribute.MirroredStrategy(devices=devices)
      with mirrored_strategy.scope():
        self._construct(d_spectra_in.shape[1:])
        print("Set the self.output to spectra")
        optimiser = keras.optimizers.Adam(learning_rate=learning_rate * dev_multiplier,
                                          beta_1=Cfg.val['beta1'],
                                          beta_2=Cfg.val['beta2'],
                                          epsilon=Cfg.val['epsilon'])
        self.ae.compile(loss=loss,
                        optimizer=optimiser,
                        metrics=['mae'])
    else:
      # Single GPU / CPU training
      dev_multiplier = 1
      self._construct(d_spectra_in.shape[1:])
      # Store input shape for FLOPs calculation during save
      self._last_input_shape = d_spectra_in.shape[1:]
      print("Set the self.output to spectra")
      optimiser = keras.optimizers.Adam(learning_rate=learning_rate,
                                        beta_1=Cfg.val['beta1'],
                                        beta_2=Cfg.val['beta2'],
                                        epsilon=Cfg.val['epsilon'])
      self.ae.compile(loss=loss,
                      optimizer=optimiser,
                      metrics=['mae'])

    # Calculate FLOPs
    self.flops = calculate_flops(self.ae, d_spectra_in.shape[1:])

    for dpi in image_dpi:
      try:
        plot_model(self.ae.encoder,
                   to_file=os.path.join(folder,'architecture-encoder@'+str(dpi)+'.png'),
                   show_shapes=True,
                   show_dtype=True,
                   show_layer_names=True,
                   rankdir='TB',
                   expand_nested=True,
                   dpi=dpi)
        plot_model(self.ae.decoder,
                   to_file=os.path.join(folder,'architecture-decoder@'+str(dpi)+'.png'),
                   show_shapes=True,
                   show_dtype=True,
                   show_layer_names=True,
                   rankdir='TB',
                   expand_nested=True,
                   dpi=dpi)
      except Exception as e:
        try:
          # Fallback to SVG if PNG generation via dot fails
          plot_model(self.ae.encoder,
                     to_file=os.path.join(folder,'architecture-encoder@'+str(dpi)+'.svg'),
                     show_shapes=True,
                     show_dtype=True,
                     show_layer_names=True,
                     rankdir='TB',
                     expand_nested=True,
                     dpi=dpi)
          plot_model(self.ae.decoder,
                     to_file=os.path.join(folder,'architecture-decoder@'+str(dpi)+'.svg'),
                     show_shapes=True,
                     show_dtype=True,
                     show_layer_names=True,
                     rankdir='TB',
                     expand_nested=True,
                     dpi=dpi)
          if verbose > 0:
            print("# WARNING: Graphviz PNG plot failed; wrote SVG instead:", e)
        except Exception as e2:
          if verbose > 0:
            print("# WARNING: Skipping model architecture plots (Graphviz error):", e2)
    if verbose > 0:
      self.ae.summary()
      self.ae.encoder.summary()
      self.ae.decoder.summary()

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
    if val_data is not None:
      val_data = val_data.batch(batch_size * dev_multiplier).with_options(options)

    # Train
    history = self.ae.fit(train_data,
                          validation_data=val_data,
                          epochs=epochs,
                          verbose=(verbose > 0)*2,
                          shuffle=True,
                          callbacks=callbacks)
    le = len(history.history['loss'])
    history.history['time (ms)'] = np.add(timer.times[:le,1],-timer.times[:le,0]) // 1000000

    if verbose > 0:
      print("# Evaluating Autoencoder")
    d_score = self.ae.evaluate(d_spectra_in, d_spectra_out, verbose=(verbose > 0)*2)
    if v_data is not None:
       v_score = self.ae.evaluate(v_spectra_in, v_spectra_out, verbose=(verbose > 0)*2)
    else:
       v_score = np.array([np.nan,np.nan])
    if verbose > 0:
       print( "             Train          Validation")
       print(f"{loss.name.upper():10s}:  {d_score[0]:.12f} {v_score[0]:.12f}")
       print(f"MAE       :  {d_score[1]:.12f} {v_score[1]:.12f}")
    self._save_results(folder, "ae", history.history, d_score, v_score, loss.name, image_dpi, screen_dpi, verbose)

    d_res={loss.name.upper():d_score[0],"MAE":d_score[1]}
    v_res={loss.name.upper():v_score[0],"MAE":v_score[1]}
    return d_res, v_res

  def _train_aeq(self, d_data, v_data, epochs, batch_size, folder, verbose,
                image_dpi, screen_dpi, train_dataset_name, devices):
    """Train the encoder-quantifier model.

    Parameters
    ----------
        d_data (list): Training data [spectra_in, concentrations]
        v_data (list, optional): Validation data [spectra_in, concentrations]
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
    # Setup training data
    if verbose > 0:
      print("# Prepare data")
    # We reshape the (batch, acquisition, datatype, frequency) spectra tensor to
    # the channel (so channel is not last, as often assumed). That means we treat
    # (batch, acquisition x datatype, frequency) where the 2nd index is effectively
    # echo acquisition-dataypte signal as separate with a separate 1D network
    # (for now we do not have any operations crossing the channels, I think).

    # Input spectra (for autoencoder and quantifier)
    d_spectra_in = tf.convert_to_tensor(d_data[0], dtype=tf.float32)
    d_spectra_in = tf.reshape(d_spectra_in,
                              (d_spectra_in.shape[0],
                               d_spectra_in.shape[1]*d_spectra_in.shape[2],d_spectra_in.shape[3]))
    # Output concentrations for quantifier
    d_conc = tf.convert_to_tensor(d_data[1], dtype=tf.float32)
    train_data = tf.data.Dataset.from_tensor_slices((d_spectra_in, d_conc))
    if v_data is not None:
      # Validation data
      v_spectra_in = tf.convert_to_tensor(v_data[0], dtype=tf.float32)
      v_spectra_in = tf.reshape(v_spectra_in,
                                (v_spectra_in.shape[0],
                                 v_spectra_in.shape[1]*v_spectra_in.shape[2],v_spectra_in.shape[3]))
      v_conc = tf.convert_to_tensor(v_data[1], dtype=tf.float32)
      val_data = tf.data.Dataset.from_tensor_slices((v_spectra_in, v_conc))
    else:
      val_data = None
    if verbose > 0:
      print("  Spectra In:        ",d_spectra_in.shape,"[spectrum, acquisition x datatype, frequency]")
      print("  Concentrations Out:",d_conc.shape,"[spectrum, metabolite_concentration]")

    # Quantifier training
    if verbose > 0:
      print(f"# Train Quantifier {self!s}")

    learning_rate = Cfg.val['base_learning_rate'] * batch_size / 16.0
    loss = keras.losses.Huber(name='huber_loss')

    if len(devices) > 1:
      # Multi-GPU training
      dev_multiplier = len(devices)
      mirrored_strategy = tf.distribute.MirroredStrategy(devices=devices)
      with mirrored_strategy.scope():
        self._construct(d_spectra_in.shape[1:], output_conc=d_conc.shape[1])
        optimiser = keras.optimizers.Adam(learning_rate=learning_rate * dev_multiplier,
                                          beta_1=Cfg.val['beta1'],
                                          beta_2=Cfg.val['beta2'],
                                          epsilon=Cfg.val['epsilon'])
        self.ae.compile(loss=loss,
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
      self.ae.compile(loss=loss,
                      optimizer=optimiser,
                      metrics=['mae'])

    # Calculate FLOPs
    self.flops = calculate_flops(self.ae, d_spectra_in.shape[1:])

    for dpi in image_dpi:
      try:
        plot_model(self.ae.encoder,
                   to_file=os.path.join(folder,'architecture-encoder-frozen@'+str(dpi)+'.png'),
                   show_shapes=True,
                   show_dtype=True,
                   show_layer_names=True,
                   rankdir='TB',
                   expand_nested=True,
                   dpi=dpi)
        plot_model(self.ae.quantifier,
                   to_file=os.path.join(folder,'architecture-quantifier@'+str(dpi)+'.png'),
                   show_shapes=True,
                   show_dtype=True,
                   show_layer_names=True,
                   rankdir='TB',
                   expand_nested=True,
                   dpi=dpi)
      except Exception as e:
        try:
          plot_model(self.ae.encoder,
                     to_file=os.path.join(folder,'architecture-encoder-frozen@'+str(dpi)+'.svg'),
                     show_shapes=True,
                     show_dtype=True,
                     show_layer_names=True,
                     rankdir='TB',
                     expand_nested=True,
                     dpi=dpi)
          plot_model(self.ae.quantifier,
                     to_file=os.path.join(folder,'architecture-quantifier@'+str(dpi)+'.svg'),
                     show_shapes=True,
                     show_dtype=True,
                     show_layer_names=True,
                     rankdir='TB',
                     expand_nested=True,
                     dpi=dpi)
          if verbose > 0:
            print("# WARNING: Graphviz PNG plot failed; wrote SVG instead:", e)
        except Exception as e2:
          if verbose > 0:
            print("# WARNING: Skipping model architecture plots (Graphviz error):", e2)
    if verbose > 0:
      self.ae.summary()
      self.ae.encoder.summary()
      self.ae.quantifier.summary()

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
    if val_data is not None:
      val_data = val_data.batch(batch_size * dev_multiplier).with_options(options)

    # Train
    history = self.ae.fit(train_data,
                          validation_data=val_data,
                          epochs=epochs,
                          verbose=(verbose > 0)*2,
                          shuffle=True,
                          callbacks=callbacks)
    le = len(history.history['loss'])
    history.history['time (ms)'] = np.add(timer.times[:le,1],-timer.times[:le,0]) // 1000000

    if verbose > 0:
      print("# Evaluating Quantifier")
    d_score = self.ae.evaluate(d_spectra_in, d_conc, verbose=(verbose > 0)*2)
    if v_data is not None:
      v_score = self.ae.evaluate(v_spectra_in, v_conc, verbose=(verbose > 0)*2)
    else:
      v_score = np.array([np.nan,np.nan])
    if verbose > 0:
      print( "             Train          Validation")
      print(f"{loss.name.upper():10s}:  {d_score[0]:.12f} {v_score[0]:.12f}")
      print(f"MAE       :  {d_score[1]:.12f} {v_score[1]:.12f}")
    self._save_results(folder, "ae_quantifier", history.history, d_score, v_score, loss.name, image_dpi, screen_dpi, verbose)

    d_res = {loss.name.upper():d_score[0],"MAE":d_score[1]}
    v_res = {loss.name.upper():v_score[0],"MAE":v_score[1]}

    return d_res, v_res

  def predict(self, spec_in, reshape=True, verbose=0):
    """Predict spectra or concentrations from input spectra.

    Parameters
    ----------
        spec_in (array-like): Input spectra data
        reshape (bool, optional): Whether to reshape input data. Defaults to True
        verbose (int, optional): Verbosity level. Defaults to 0

    Returns
    -------
        numpy.ndarray: Predicted spectra or concentrations
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
    if self.output == "spectra":
      return np.array(tf.reshape(self.ae.predict(data,verbose=(verbose>0)*2),out_shape),dtype=np.float64)
    if self.output == "concentrations":
      return np.array(self.ae.predict(data,verbose=(verbose>0)*2), dtype=np.float64)
    raise RuntimeError(f"Unknown output {self.output}")

  def save(self, folder):
    """Save the trained model to disk.

    Parameters
    ----------
        folder (str): Directory to save the model
    """
    os.makedirs(folder, exist_ok=True)
    self.ae.save(os.path.join(folder, "model.keras"))

    # Calculate model metrics from summary
    # Capture model summary output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    self.ae.summary()
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
            'autoencoder_model': self.autoencoder_model if hasattr(self,'autoencoder_model') else None,
            'autoencoder_train_dataset_name': self.autoencoder_train_dataset_name if hasattr(self,'autoencoder_train_dataset_name')
                                              else None,
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
        Autoencoder: Loaded model instance
    """
    with open(os.path.join(path, "mrsnet.json")) as f:
      data = json.load(f)
    model = Autoencoder(data['model'], data['metabolites'], data['pulse_sequence'], data['acquisitions'],
                        data['datatype'], data['norm'])
    model.output = data['output']
    model.train_dataset_name = data['train_dataset_name']
    # Provide deserialization context for subclassed models saved via SavedModel
    global _FCAE_LOAD_CTX
    # Infer AE architecture hyperparameters from model string and input build shape
    p = model.model.split("_")
    if p[0] == "ae" and p[1] == "fc":
      # We do not know n_specs/n_freqs at this point; they are present in SavedModel build_config
      # Keras will call from_config first; we will set n_specs/n_freqs from build_config via fallback below.
      # For now, set placeholders; we will patch after loading once we see input shape.
      # However, we can infer n_specs from acquisitions x datatype and n_freqs from model.fft_samples
      n_specs = len(model.acquisitions) * len(model.datatype)
      n_freqs = model.fft_samples
      _FCAE_LOAD_CTX = {
        'n_specs': n_specs,
        'n_freqs': n_freqs,
        'layers_enc': int(p[2]),
        'layers_dec': int(p[3]),
        'activation': p[4],
        'activation_last': p[5],
        'dropout': float(p[6])
      }
    try:
      model.ae = load_model(os.path.join(path, "model.keras"),
                            custom_objects={
                               "FCAutoEnc": FCAutoEnc,
                               "EncQuant": EncQuant
                             },
                            safe_mode=False)
    finally:
      _FCAE_LOAD_CTX = None
    return model

  def _save_results(self, folder, prefix, history, d_score, v_score, loss_name, image_dpi, screen_dpi, verbose):
    """Save training results to files.

    Parameters
    ----------
        folder (str): Output directory
        prefix (str): File prefix for saved files
        history (dict): Training history
        d_score (list): Training scores
        v_score (list): Validation scores
        loss_name (str): Loss function name
        image_dpi (list): Image DPI settings
        screen_dpi (int): Screen DPI setting
        verbose (int): Verbosity level
    """
    keys = sorted(history.keys())
    # History data
    with open(os.path.join(folder, prefix+'_history.csv'), "w") as out_file:
      writer = csv.writer(out_file, delimiter=",")
      if self.model[0:3] =="aeq":
        writer.writerows([[self.model+" "+prefix.upper()+" Training Results", "", "", "", "", "Loaded AE: " + self.ae_path],
                        [""],
                        ["",     "Train",    "Validation"],
                        [loss_name.upper(),  d_score[0], v_score[0]],
                        ["MAE",  d_score[1], v_score[1]],
                        [""],
                        ["History"]])
      else:
        writer.writerows(
            [[self.model + " " + prefix.upper() + " Training Results"],
             [""],
             ["", "Train", "Validation"],
             [loss_name.upper(), d_score[0], v_score[0]],
             ["MAE", d_score[1], v_score[1]],
             [""],
             ["History"]])
      writer.writerow(keys)
      writer.writerows(zip(*[history[key] for key in keys], strict=False))
    # Plot
    fig, axes = plt.subplots(1, 3)
    fig.suptitle(f"{self.model} {prefix.upper()} Training Results")
    for key in keys:
      if loss_name in key or 'loss' in key:
        axes[0].semilogy(history[key], label=key)
        axes[0].set_ylabel(loss_name.upper())
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
      plt.savefig(os.path.join(folder, prefix+'_history@'+str(dpi)+'.png'), dpi=dpi)
    if verbose > 1:
      fig.set_dpi(screen_dpi)
      plt.show(block=True)
    plt.close()
