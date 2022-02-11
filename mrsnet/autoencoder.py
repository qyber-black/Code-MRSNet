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

from .cnn import TimeHistory

# Helper to construct convolutional encoder layer
def _enc_conv_layer(m, filter, c, s, pooling, dropout):
  if pooling:
    m.add(Conv2D(filter, c, padding="same"))
  else:
    m.add(Conv2D(filter, c, padding="same", strides=s))
  if dropout == 0.0:
    m.add(BatchNormalization())
  m.add(Activation('relu'))
  if dropout > 0.0:
    m.add(Dropout(dropout))
  if pooling:
    m.add(MaxPool2D(s))

# Helper to construct convolutional decoder layer
def _dec_conv_layer(m, filter, c, s, pooling, dropout):
  if pooling:
    m.add(Conv2DTranspose(filter, c, padding="same"))
  else:
    m.add(Conv2DTranspose(filter, c, padding="same", strides=s))
  if dropout == 0.0:
    m.add(BatchNormalization())
  m.add(Activation('relu'))
  if dropout > 0.0:
    m.add(Dropout(dropout))
  if pooling:
    m.add(UpSampling2D(s))

# Helper to construct dense encoder layer
def _enc_dense_layer(m, dim, activation,):
  if activation == 're':
    m.add(Dense(dim,activation='relu'))
  elif activation == 'sig':
    m.add(Dense(dim,activation='sigmoid'))
  elif activation == 'ta':
    m.add(Dense(dim, activation='tanh'))

# Helper to construct dense decoder layer
def _dec_dense_layer(m,dim,activation,):
  if activation == 're':
    m.add(Dense(dim,activation='relu'))
  elif activation == 'sig':
    m.add(Dense(dim,activation='sigmoid'))
  elif activation == 'ta':
    m.add(Dense(dim, activation='tanh'))

# Dropout layer
def _drop_out_layer(m,r):
  m.add(Dropout(r))

# Plot difference
def plot_spectra_(input, contrast_input, title, data, location, num,datatype):
  l = []
  for i in range(0, 2048):
    l.append(-4.5+i*3/2048)

  plt.figure()
  plt.plot(l, input, label='Reconstructed Spectra', color='#DC143C')
  plt.plot(l, contrast_input, label=data, color='#4169E1')

  plt.xlim(-4.5, -1.5)
  plt.ylim()

  plt.xlabel('$Frequency$')
  plt.ylabel(datatype) # Magnitude Phase Imaginary Real

  plt.title(title+'_'+num)
  plt.legend(loc='best')

  if int(num)<0:
    print('The plot will not be shown.')
  else:
    plt.savefig(os.path.join(location,num+'_'+title+  '.png'))
  plt.show()

# Convolutional autoencoder via Model interface (using Sequential interface internally)
class ConvAutoEnc(Model):

  def __init__(self, ae_shape, name='ConvAutoEnc'):
    super(ConvAutoEnc, self).__init__(name=name)

    # Encoder
    self.encoder = tf.keras.Sequential(name='Encoder')
    re_r = ae_shape[0]//2             # reshape for decoder (divide by row strides in encoder)
    re_c = ae_shape[1]//(4*4*2*4*4)   # reshape for decoder (divide by column strides in encoder)
    re_f = 512                        # last filter size, reshape in decoder
    # Encoder Layers: filter, convolution_kernel, strides, pooling, dropout
    # pooling: Pooling or strides for up/downsampling?
    # dropout: Dropout if > 0.0; 0.0, BatchNormalisation; negative, no regulariser
    _enc_conv_layer(self.encoder, 256, (1,7), (1,4), False, 0.0)
    _enc_conv_layer(self.encoder, 256, (1,5), (1,4), False, -1.0)
    #
    _enc_conv_layer(self.encoder, 256, (2,3), (1,1), False, -1.0)
    _enc_conv_layer(self.encoder, 256, (2,3), (2,2), False, -1.0)
    _enc_conv_layer(self.encoder, 512, (2,3), (1,1), False, -1.0)
    _enc_conv_layer(self.encoder, 512, (2,3), (1,1), False, -1.0)
    #
    _enc_conv_layer(self.encoder, 512, (1,5), (1,4), False, -1.0)
    _enc_conv_layer(self.encoder, re_f, (1,5), (1,4), False, -1.0)
    self.encoder.add(Flatten())
    self.encoder.add(Dense(2048)) # Latent representation size
    # Decoder
    self.decoder = tf.keras.Sequential(name='Decoder')
    self.decoder.add(Dense(re_r * re_c * re_f))
    self.decoder.add(Reshape(target_shape=(re_r,re_c,re_f)))
    # Decoder Layers: filter, convolution_kernel, strides, pooling, dropout
    _dec_convt_layer(self.decoder, re_f, (1,5), (1,4), False, -1.0)
    _dec_convt_layer(self.decoder, 512, (1,5), (1,4), False, -1.0)
    #
    _dec_convt_layer(self.decoder, 512, (2,3), (1,1), False, -1.0)
    _dec_convt_layer(self.decoder, 512, (2,3), (1,1), False, -1.0)
    _dec_convt_layer(self.decoder, 256, (2,3), (2,2), False, -1.0)
    _dec_convt_layer(self.decoder, 256, (2,3), (1,1), False, -1.0)
    #
    _dec_convt_layer(self.decoder, 256, (1,5), (1,4), False, -1.0)
    _dec_convt_layer(self.decoder, 256, (1,7), (1,4), False, -1.0)
    # Final, no activation
    self.decoder.add(Conv2D(1, kernel_size=(1, 7), activation=None, padding='same'))

    self.build((None,*ae_shape))

  def call(self,x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

#  Fully connected autoencoder via Model interface (using Sequential interface internally) For magnitude
class DenseAutoEnc_mag(Model):

  def __init__(self, ae_shape, name='DenseAutoEnc_mag'):
    super(DenseAutoEnc_mag, self).__init__(name=name)
    # Encoder
    self.encoder = tf.keras.Sequential(name='Encoder')
    self.encoder.add(Flatten())
    _enc_dense_layer(self.encoder, 2048, "ta")
    _drop_out_layer(self.encoder,0.3)
    _enc_dense_layer(self.encoder, 1024, "ta")
    _drop_out_layer(self.encoder, 0.3)
    _enc_dense_layer(self.encoder, 512, "ta")
    _drop_out_layer(self.encoder, 0.3)
    _enc_dense_layer(self.encoder, 256, "ta")
    _drop_out_layer(self.encoder, 0.3)
    _enc_dense_layer(self.encoder, 128, "ta")
    _drop_out_layer(self.encoder, 0.3)
    _enc_dense_layer(self.encoder, 64, "ta")
    _drop_out_layer(self.encoder, 0.3)
    _enc_dense_layer(self.encoder, 32, "ta")
    #_enc_dense_layer(self.encoder, 32, "sig")
    #_drop_out_layer(self.encoder, 0.3)
    #_enc_dense_layer(self.encoder, 16, "ta")
    #_drop_out_layer(self.encoder, 0.3)
    #_enc_dense_layer(self.encoder, 8, "ta")
    #_drop_out_layer(self.encoder, 0.3)
    #_enc_dense_layer(self.encoder, 8, "sig")

    # Decoder
    self.decoder = tf.keras.Sequential(name='Decoder')
    #_dec_dense_layer(self.decoder, 8, "ta")
    #_dec_dense_layer(self.decoder, 16, "ta")
    _dec_dense_layer(self.decoder, 32, "ta")
    _dec_dense_layer(self.decoder, 64, "ta")
    _dec_dense_layer(self.decoder, 128, "ta")
    _dec_dense_layer(self.decoder, 256, "ta")
    _dec_dense_layer(self.decoder, 512, "ta")
    _dec_dense_layer(self.decoder, 1024, "ta")
    _dec_dense_layer(self.decoder, 2048, "ta")
    self.decoder.add(Reshape((1, 2048)))

    self.build((None,*ae_shape))

  def call(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

# Fully connected autoencoder via Model interface (using Sequential interface internally)
# For Real:r and Imaginary:i
class DenseAutoEnc_r_i(Model):

  def __init__(self, ae_shape, name='DenseAutoEnc_r_i'):
    super(DenseAutoEnc_r_i, self).__init__(name=name)
    # Encoder
    self.encoder = tf.keras.Sequential(name='Encoder')
    self.encoder.add(Flatten())
    _enc_dense_layer(self.encoder, 2048, "ta")
    _drop_out_layer(self.encoder,0.3)
    _enc_dense_layer(self.encoder, 1024, "ta")
    _drop_out_layer(self.encoder, 0.3)
    _enc_dense_layer(self.encoder, 512, "ta")
    _drop_out_layer(self.encoder, 0.3)
    _enc_dense_layer(self.encoder, 256, "ta")
    _drop_out_layer(self.encoder, 0.3)
    _enc_dense_layer(self.encoder, 128, "ta")
    _drop_out_layer(self.encoder, 0.3)
    _enc_dense_layer(self.encoder, 64, "ta")
    #_drop_out_layer(self.encoder, 0.3)
    #_enc_dense_layer(self.encoder, 32, "ta")
    #_enc_dense_layer(self.encoder, 32, "sig")
    #_drop_out_layer(self.encoder, 0.3)
    #_enc_dense_layer(self.encoder, 16, "ta")
    #_drop_out_layer(self.encoder, 0.3)
    #_enc_dense_layer(self.encoder, 8, "ta")
    #_drop_out_layer(self.encoder, 0.3)
    #_enc_dense_layer(self.encoder, 8, "sig")

    # Decoder
    self.decoder = tf.keras.Sequential(name='Decoder')
    _dec_dense_layer(self.decoder, 8, "ta")
    _dec_dense_layer(self.decoder, 16, "ta")
    _dec_dense_layer(self.decoder, 32, "ta")
    _dec_dense_layer(self.decoder, 64, "ta")
    _dec_dense_layer(self.decoder, 128, "ta")
    _dec_dense_layer(self.decoder, 256, "ta")
    _dec_dense_layer(self.decoder, 512, "ta")
    _dec_dense_layer(self.decoder, 1024, "ta")
    _dec_dense_layer(self.decoder, 2048, "ta")
    self.decoder.add(Reshape((1, 2048)))

    self.build((None,*ae_shape))

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
    if self.model != "ae_cnn": # FIXME: eventually we'll have more models, parameterised/selected via self.model
      raise Exception("Unknown autoencoder model")
    # Define the autoencoder via the functional interface
    #self.ae = ConvAutoEnc(ae_shape=ae_shape)
    if self.datatype[0] == "magnitude":
      self.ae = DenseAutoEnc_mag(ae_shape=ae_shape)
    elif self.datatype[0] == "real":
      self.ae = DenseAutoEnc_r_i(ae_shape=ae_shape)
    elif self.datatype[0]== "imaginary":
      self.ae = DenseAutoEnc_r_i(ae_shape=ae_shape)

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
    if v_data != None and len(v_data) != 2:
      raise Exception("v_data argument must be a list [spectra_in,spectra_out,conc]")

    if len(train_dataset_name) > 0:
      self.train_dataset_name = train_dataset_name

    if not os.path.isdir(folder):
      os.makedirs(folder)

    # Setup training data - FIXME: maybe 3D OK here, no flattening?
    if verbose > 0:
      print("# Prepare data")

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

    if verbose > 1:
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
        self.cnn.compile(loss=loss,
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
      self.cnn.compile(loss=loss,
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
    history = self.cnn.fit(ae_train_data,
                           validation_data=ae_val_data,
                           epochs=epochs,
                           verbose=(verbose > 0)*2,
                           shuffle=True,
                           callbacks=callbacks)
    le = len(history.history['loss'])
    history.history['time (ms)'] = np.add(timer.times[:le,1],-timer.times[:le,0]) // 1000000


    # FIXME: comparison of input/output

    # Noisy spectra dataset to test the performance of the reconstruction, It's An isolated variable to prevent if I mess up with the future d_inp variable, to keep the consistency of the d_inp
    x_test_n = tf.reshape(d_inp, (d_inp.shape[0], 1, 2048))
    # Clean spectra dataset to test the performance of the reconstruction
    x_test = tf.reshape(d_out, (d_out.shape[0], 1, 2048))

    # These are the cheap job I did for see the comparison, I have to manually changed the number on directory address every time
    if self.datatype[0] == "magnitude":
      os.makedirs(folder + "/Comparison_ta_ta_32_32_5000-1n_dpAll_0.3/")
      save_plot = folder + "/Comparison_ta_ta_32_32_5000-1n_dpAll_0.3/"
    elif self.datatype[0] == "real":
      os.makedirs(folder + "/Comparison_ta_ta_64_8_5000-1n_dpAll_0.3/")
      save_plot = folder + "/Comparison_ta_ta_64_8_5000-1n_dpAll_0.3/"
    elif self.datatype[0] == "imaginary":
      os.makedirs(folder + "/Comparison_ta_ta_64_8_5000-1n_dpAll_0.3/")
      save_plot = folder + "/Comparison_ta_ta_64_8_5000-1n_dpAll_0.3/"

    # Print out the plots of first 5 noisy spectra that being put into the autoencoder and its comparison:"Clean spectra"
    for i in range(5):
      encoder_imgs = self.ae.encoder(x_test_n[i]).numpy()
      decoder_imgs = self.ae.decoder(encoder_imgs).numpy()

      plot_spectra_(decoder_imgs[0, 0], x_test_n[i, 0], 'Reconstructed spectra vs Noisy spectra', 'Noisy spectra', save_plot,str(i),self.datatype[0])
      plot_spectra_(decoder_imgs[0, 0], x_test[i, 0], 'Reconstructed spectra vs clean spectra', 'Clean spectra', save_plot,str(i),self.datatype[0])

    ######################

    if verbose > 0:
      print("# Evaluating Autoencoder")
    d_score = self.ae.evaluate(d_spectra_in, d_spectra_out, verbose=(verbose > 0)*2)
    if v_data != None:
      v_score = self.ae.evaluate(v_spectra_in, v_spectra_out, verbose=(verbose > 0)*2)
    else:
      v_score = np.array([np.nan,np.nan])
    if verbose > 0:
      print(f"{' '*len(loss)}   Train          Validation")
      print("%s:  %.12f %.12f" % (loss.upper(), d_score[0], v_score[0]))
      print("%s:  %.12f %.12f" % (loss.upper(), d_score[1], v_score[1]))
    self._save_results(folder, "ae", history.history, d_score, v_score, loss, image_dpi, screen_dpi, verbose)

    d_res={loss.upper():d_score[0],"MAE":d_score[1]}
    v_res={loss.upper():v_score[0],"MAE":v_score[1]}
    return d_res, v_res

    # FIXME: regression/concentration fitting network after autoencoder is trained

  # FIXME: if this is not defined, analyse_model will not analyse concentration predictions and quantify does not work
  #        this is OK as long as we only have the autoencoder part.
  ##def predict(self, d_inp, reshape=True, verbose=0):
  ##  if reshape:
  ##    d_inp = tf.convert_to_tensor(d_inp, dtype=tf.float32)
  ##    d_inp = tf.reshape(d_inp,(d_inp.shape[0],d_inp.shape[1]*d_inp.shape[2],d_inp.shape[3],1))
  ##  options = tf.data.Options()
  ##  options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
  ##  data = tf.data.Dataset.from_tensor_slices((d_inp)).batch(32).with_options(options)
  ##  return np.array(self.reg.predict(data,verbose=(verbose>0)*2),dtype=np.float64)

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

  def _save_results(self, folder, prefix, history, d_score, v_score, loss, no_show, image_dpi, screen_dpi):
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
