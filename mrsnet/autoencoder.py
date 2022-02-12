# mrsnet/autoencoder.py - MRSNet - autoencoder model
#
# SPDX-FileCopyrightText: Copyright (C) 2022 Zien Ma, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2022 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from time import time_ns

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

from mrsnet.cfg import Cfg
from .cnn import TimeHistory

# Helper to construct convolutional encoder layer
def _enc_conv_layer(m, filter, c, s, pooling, dropout):
  # Conv1D layer over last index (freq. bins) with convolution kernel size c and relu activation
  # if pooling == True:  use s max pooling;
  #            == False: use s strides
  # if dropout > 0: dropout rate after activation;
  #            ==0: batch normalisation before activation;
  #            < 0: nothing
  if pooling:
    m.add(Conv1D(filter, c, strides=1, padding="same", data_format="channels_first"))
  else:
    m.add(Conv1D(filter, c, strides=s, padding="same", data_format="channels_first"))
  if dropout == 0.0:
    m.add(BatchNormalization())
  m.add(Activation('relu'))
  if dropout > 0.0:
    m.add(Dropout(dropout))
  if pooling:
    m.add(MaxPool1D(s))

# Helper to construct convolutional decoder layer
def _dec_convt_layer(m, filter, c, s, pooling, dropout):
  # Conv1DT layer over last index (freq. filter bins) with convolution kernel size c and relu activation
  # if pooling == True:  use s max pooling;
  #            == False: use s strides
  # if dropout > 0: dropout rate after activation;
  #            ==0: batch normalisation before activation;
  #            < 0: nothing
  if pooling:
    m.add(Conv1DTranspose(filter, c, strides=1, padding="same", data_format="channels_first"))
  else:
    m.add(Conv1DTranspose(filter, c, strides=s, padding="same", data_format="channels_first"))
  if dropout == 0.0:
    m.add(BatchNormalization())
  m.add(Activation('relu'))
  if dropout > 0.0:
    m.add(Dropout(dropout))
  if pooling:
    m.add(UpSampling1D(s))

# Convolutional autoencoder via Model interface (using Sequential interface internally)
class ConvAutoEnc(Model):

  def __init__(self, n_specs, n_freqs, filter, latent, pooling, dropout, name='ConvAutoEnc'):
    # n_specs: number of spectra (acquisisions x datatype)
    # n_freqs: number of frequency bins in spectra
    # filter: numbner of filters on input conv layer (others computed from this)
    # latent: size of latent representation
    # pooling: Pooling or strides for up/downsampling?
    # dropout: Dropout if > 0.0; 0.0, BatchNormalisation; negative, nothing
    # FIXME: could parameterise kernel size(s) and strides and also depth of network
    super(ConvAutoEnc, self).__init__(name=name)

    # Encoder
    self.encoder = tf.keras.Sequential(name='Encoder')
    # Encoder Layers: filter, convolution_kernel, strides, pooling, dropout
    _enc_conv_layer(self.encoder, filter,   7, 2, pooling, dropout)
    _enc_conv_layer(self.encoder, filter,   7, 2, pooling, dropout)
    _enc_conv_layer(self.encoder, 2*filter, 5, 2, pooling, dropout)
    _enc_conv_layer(self.encoder, 2*filter, 5, 2, pooling, dropout)
    _enc_conv_layer(self.encoder, 2*filter, 3, 1, pooling, dropout)
    _enc_conv_layer(self.encoder, 2*filter, 3, 1, pooling, dropout)
    self.encoder.add(Flatten())
    self.encoder.add(Dense(latent)) # Latent representation size

    # Decoder
    self.decoder = tf.keras.Sequential(name='Decoder')
    n_channels = 2*filter
    n_signal = n_freqs//(2*2*2*2) # Divide by strides in encoder
    self.decoder.add(Dense(n_channels * n_signal))
    self.decoder.add(Reshape(target_shape=(n_channels, n_signal)))
    # Decoder Layers: filter, convolution_kernel, strides, pooling, dropout
    _dec_convt_layer(self.decoder, 2*filter, 3, 1, pooling, dropout)
    _dec_convt_layer(self.decoder, 2*filter, 3, 1, pooling, dropout)
    _dec_convt_layer(self.decoder, 2*filter, 5, 2, pooling, dropout)
    _dec_convt_layer(self.decoder, 2*filter, 5, 2, pooling, dropout)
    _dec_convt_layer(self.decoder, filter,   7, 2, pooling, dropout)
    _dec_convt_layer(self.decoder, filter,   7, 2, pooling, dropout)
    # Final, no activation
    self.decoder.add(Conv1D(n_specs, 7, activation=None, padding='same', data_format="channels_first"))

    self.build((None,n_specs,n_freqs))

  def call(self,x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

# Helper to construct dense layer
def _dense_layer(m, units, activation, dropout):
  # Dense layer over last index (freq. bins) using given activation
  # if dropout > 0: dropout rate after activation;
  #            ==0: batch normalisation before activation;
  #            < 0: nothing
  m.add(Dense(units))
  if dropout == 0.0:
    m.add(BatchNormalization())
  m.add(Activation(activation))
  if dropout > 0.0:
    m.add(Dropout(dropout))

#  Fully connected autoencoder via Model interface (using Sequential interface internally)
class FCAutoEnc(Model):

  def __init__(self, n_specs, n_freqs, layers_enc, layers_dec, activation, dropout, name='FCAutoEnc'):
    # n_specs: number of spectra (acquisisions x datatype)
    # n_freqs: number of frequency bins in spectra
    # layers_enc: number of layers in encoder
    # layers_dec: number of layers in encoder
    # activation: activation function (relu, sigmoid, tanh)
    # dropout: Dropout if > 0.0; 0.0, BatchNormalisation; negative, nothing
    super(FCAutoEnc, self).__init__(name=name)

    # Encoder
    self.encoder = tf.keras.Sequential(name='Encoder')
    units = n_freqs
    for l in range(0,layers_enc-1):
      _dense_layer(self.encoder, units, activation, dropout)
      units //= 2
    _dense_layer(self.encoder, units, activation, -1) # no regulariser at latent representation

    # Decoder
    self.decoder = tf.keras.Sequential(name='Decoder')
    units = n_freqs//(2**(layers_dec-1))
    for l in range(0,layers_dec):
      _dense_layer(self.decoder, units, activation, -1) # no regularisers in decoder
      units *= 2

    self.build((None,n_specs,n_freqs))

  def call(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

# Autoencoder model
class Autoencoder:

  def __init__(self, model, metabolites, pulse_sequence, acquisitions, datatype, norm):
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
    self.ae = None

  def __str__(self):
    n = os.path.join(self.model, "-".join(self.metabolites),
                     self.pulse_sequence, "-".join(self.acquisitions),
                     "-".join(self.datatype), self.norm)
    return n

  def reset(self):
    del self.ae
    self.ae = None
    self.train_dataset_name = None

  def _construct(self, ae_shape):
    n_specs = ae_shape[0] # number of spectras: acqusitions x datatype
    n_freqs = ae_shape[1] # number of frequency bins in spectras
    p = self.model.split("_")
    if p[0] != "ae":
      raise Exception("Not an auto-encoder")
    if p[1] == "cnn":
      # ae_cnn_[FILTER]_[LATENT]_[pool|stride]_[DO]
      #    FILTER: size of initial filter on conv input; other filter sizes computed from it
      #    LATENT: size of latent representation
      #    pool|stride: use max-pooling/up-sampling or strides for reduction/increase
      #    DO: Dropout if > 0.0; 0.0, BatchNormalisation; negative, no regulariser
      filter = int(p[2])
      latent = int(p[3])
      if p[4] == "pool":
        pooling = True
      elif p[4] == "stride":
        pooling = False
      else:
        raise Exception("2nd arg must be pool|stride")
      dropout = float(p[5])
      self.ae = ConvAutoEnc(n_specs,n_freqs,filter,latent,pooling,dropout)
    elif p[1] == "fc":
      # ae_fc_[LIN]_[LOUT]_[ACT]_[DO]
      #    LIN: dense layers in encoder
      #    LOUT: dense layers in decoder
      #    ACT: activation function (relu, sigmoid, tanh, ...)
      #    DO: Dropout if > 0.0; 0.0, BatchNormalisation; negative, no regulariser
      lin = int(p[2])
      lout = int(p[3])
      act = p[4]
      dropout = float(p[5])
      self.ae = FCAutoEnc(n_specs,n_freqs,lin,lout,act,dropout)
    else:
      raise Exception("Unknown autoencoder variant")

  def train(self, d_data, v_data, epochs, batch_size,
            folder, verbose=0, image_dpi=[300], screen_dpi=96, train_dataset_name=""):
    devices = tf.config.list_logical_devices("GPU")
    if len(devices) < 1:
      print("**WARNING, we do not have a GPU for Tensorflow!**")
      devices = []
    else:
      devices = [devices[l].name for l in range(0,len(devices))]
      if verbose > 0:
        print(f"GPU Devices: {devices}")
    if len(d_data) != 3:
      raise Exception("d_data argument must be a list [spectra_in,spectra_out,conc]")
    if v_data != None and len(v_data) != 3:
      raise Exception("v_data argument must be a list [spectra_in,spectra_out,conc]")

    if len(train_dataset_name) > 0:
      self.train_dataset_name = train_dataset_name

    if not os.path.isdir(folder):
      os.makedirs(folder)

    # Setup training data
    if verbose > 0:
      print("# Prepare data")

    # We reshape the (batch, acquisition, datatype, frequency) spectra tensor to
    # the channel (so channel is not last, as often assumed). That means we treat
    # (batch, acquisition x datatype, frequency) where the 2nd index is effectively
    # echo acquisition-dataypte signal as separate with a separate 1D network
    # (for now we do not have any operations crossing the channels, I think).
    #
    # FIXME: Something to consider:
    # Instead, we could also reshape it to (batch x acquisition x datatype, frequency)
    # meaning each signal, idependent of acquistion and datatype, is handled
    # the same with the same network. We only have a collection of 1D signals for
    # the autoencoder - for quantification then this would need some more complex
    # reshaping as there we will have to consider the signals across acquisitons and
    # datatypes.

    d_spectra_in = tf.convert_to_tensor(d_data[0], dtype=tf.float32)
    d_spectra_out = tf.convert_to_tensor(d_data[1], dtype=tf.float32)
    d_conc = tf.convert_to_tensor(d_data[2], dtype=tf.float32)
    d_spectra_in = tf.reshape(d_spectra_in,
                              (d_spectra_in.shape[0],
                               d_spectra_in.shape[1]*d_spectra_in.shape[2],d_spectra_in.shape[3]))
    d_spectra_out = tf.reshape(d_spectra_out,
                               (d_spectra_out.shape[0],
                                d_spectra_out.shape[1]*d_spectra_out.shape[2],d_spectra_out.shape[3]))
    ae_train_data = tf.data.Dataset.from_tensor_slices((d_spectra_in, d_spectra_out))
    conc_train_data = tf.data.Dataset.from_tensor_slices((d_spectra_in, d_conc))

    if v_data != None:
      v_spectra_in = tf.convert_to_tensor(v_data[0], dtype=tf.float32)
      v_spectra_out = tf.convert_to_tensor(v_data[1], dtype=tf.float32)
      v_conc = tf.convert_to_tensor(v_data[2], dtype=tf.float32)
      v_spectra_in = tf.reshape(v_spectra_in,
                                (v_spectra_in.shape[0],
                                 v_spectra_in.shape[1]*v_spectra_in.shape[2],v_spectra_in.shape[3]))
      v_spectra_out = tf.reshape(v_spectra_out,
                                 (v_spectra_out.shape[0],
                                  v_spectra_out.shape[1]*v_spectra_out.shape[2],v_spectra_out.shape[3]))
      ae_val_data = tf.data.Dataset.from_tensor_slices((v_spectra_in, v_spectra_out))
      conc_val_data = tf.data.Dataset.from_tensor_slices((v_spectra_in, v_conc))
    else:
      ae_val_data = None
      conc_val_data = None

    if verbose > 0:
      print("  Spectra In:    ",d_spectra_in.shape,"[spectrum, acquisition x datatype, frequency]")
      print("  Spectra Out:   ",d_spectra_out.shape,"[spectrum, acquisition x datatype, frequency]")
      print("  Concentrations:",d_conc.shape,"[spectrum, metabolite_concentration]")

    # Autoencoder training
    if verbose > 0:
      print("# Train Autoencoder %s" % str(self))

    learning_rate = Cfg.val['base_learning_rate'] * batch_size / 16.0
    loss = "huber_loss"

    if len(devices) > 1:
      # Multi-GPU training
      dev_multiplier = len(devices)
      mirrored_strategy = tf.distribute.MirroredStrategy(devices=devices)
      with mirrored_strategy.scope():
        self._construct(d_spectra_in.shape[1:])
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
      optimiser = keras.optimizers.Adam(learning_rate=learning_rate,
                                        beta_1=Cfg.val['beta1'],
                                        beta_2=Cfg.val['beta2'],
                                        epsilon=Cfg.val['epsilon'])
      self.ae.compile(loss=loss,
                      optimizer=optimiser,
                      metrics=['mae'])

    for dpi in image_dpi:
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
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ae_train_data = ae_train_data.batch(batch_size * dev_multiplier).with_options(options)
    ae_val_data = ae_val_data.batch(batch_size * dev_multiplier).with_options(options)

    # Train
    history = self.ae.fit(ae_train_data,
                          validation_data=ae_val_data,
                          epochs=epochs,
                          verbose=(verbose > 0)*2,
                          shuffle=True,
                          callbacks=callbacks)
    le = len(history.history['loss'])
    history.history['time (ms)'] = np.add(timer.times[:le,1],-timer.times[:le,0]) // 1000000

    if verbose > 0:
      print("# Evaluating Autoencoder")
    d_score = self.ae.evaluate(d_spectra_in, d_spectra_out, verbose=(verbose > 0)*2)
    if v_data != None:
      v_score = self.ae.evaluate(v_spectra_in, v_spectra_out, verbose=(verbose > 0)*2)
    else:
      v_score = np.array([np.nan,np.nan])
    if verbose > 0:
      print(f"             Train          Validation")
      print(f"{loss.upper():10s}:  {d_score[0]:.12f} {v_score[0]:.12f}")
      print(f"MAE       :  {d_score[1]:.12f} {v_score[1]:.12f}")
    self._save_results(folder, "ae", history.history, d_score, v_score, loss, image_dpi, screen_dpi, verbose)

    d_res={loss.upper():d_score[0],"MAE":d_score[1]}
    v_res={loss.upper():v_score[0],"MAE":v_score[1]}
    return d_res, v_res

    # FIXME: regression/concentration fitting network after autoencoder is trained
    # The best way to do this is to add additional parameters to the model string that indicate
    # we wish to predict the concentrations and then paramters with that to indicate the structure
    # of the concentration prediction network part. Then we can select between autoencoder only
    # or autoencoder+concentration. This also needs to then be consider in the predict function.
    # The analyse function can determine if it is to analyse spectra error or concentration
    # errors from this as well.

  def predict(self, spec_in, reshape=True, verbose=0):
    # FIXME: set up to do the autoencoder spectra prediction (which should be ~noise filtering).
    # Once we have the concentration prediction, we need to switch between two modes here
    # depending on the model string.
    out_shape = spec_in.shape # Preserve shape of input spectra to reshape output accordingly
    if reshape:
      spec_in = tf.convert_to_tensor(spec_in, dtype=tf.float32)
      spec_in = tf.reshape(spec_in,(spec_in.shape[0],spec_in.shape[1]*spec_in.shape[2],spec_in.shape[3]))
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    data = tf.data.Dataset.from_tensor_slices((spec_in)).batch(32).with_options(options)
    return np.array(tf.reshape(self.ae.predict(data,verbose=(verbose>0)*2),out_shape),dtype=np.float64)

  def save(self, folder):
    path=os.path.join(folder, "tf_ae_model")
    self.ae.save(path)
    with open(os.path.join(path, "mrsnet.json"), 'w') as f:
      print(json.dumps({
          'model': self.model,
          'metabolites': self.metabolites,
          'pulse_sequence': self.pulse_sequence,
          'acquisitions': self.acquisitions,
          'datatype': self.datatype,
          'norm': self.norm,
          'train_dataset_name': self.train_dataset_name
        }, indent=2, sort_keys=True), file=f)

  @staticmethod
  def load(path):
    with open(os.path.join(path, "tf_ae_model", "mrsnet.json"), 'r') as f:
      data = json.load(f)
    model = Autoencoder(data['model'], data['metabolites'], data['pulse_sequence'], data['acquisitions'],
                        data['datatype'], data['norm'])
    model.train_dataset_name = data['train_dataset_name']
    model.ae = load_model(os.path.join(path,"tf_ae_model"))
    return model

  def _save_results(self, folder, prefix, history, d_score, v_score, loss, image_dpi, screen_dpi, verbose):
    keys = sorted(history.keys())
    # History data
    with open(os.path.join(folder, prefix+'_history.csv'), "w") as out_file:
      writer = csv.writer(out_file, delimiter=",")
      writer.writerows([[self.model+" "+prefix.upper()+" Training Results"],
                        [""],
                        ["",     "Train",    "Validation"],
                        [loss.upper(),  d_score[0], v_score[0]],
                        ["MAE",  d_score[1], v_score[1]],
                        [""],
                        ["History"]])
      writer.writerow(keys)
      writer.writerows(zip(*[history[key] for key in keys]))
    # Plot
    fig, axes = plt.subplots(1, 3)
    fig.suptitle(f"{self.model} {prefix.upper()} Training Results")
    for key in keys:
      if loss in key or 'loss' in key:
        axes[0].semilogy(history[key], label=key)
        axes[0].set_ylabel(loss.upper())
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
