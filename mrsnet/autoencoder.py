# mrsnet/autoencoder.py - MRSNet - autoencoder model
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2019, Max Chandler, PhD student at Cardiff University
# Copyright (C) 2020-2021, Frank C Langbein <frank@langbein.org>, Cardiff University

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
def _dec_convt_layer(m, filter, c, s, pooling, dropout):
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
def _enc_dense_layer(m,dim,activation,):
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

# plot diff
def plot_spectra_(input, contrast, datatype):
      l = []
      for i in range(0, 2048):
          l.append(-4.5+i*3/2048)

      plt.figure()
      plt.plot(l, input, label='Reconstructed Spectra', color='#DC143C')
      plt.plot(l, contrast, color='#4169E1')

      plt.xlim(-4.5, -1.5)
      plt.ylim()

      plt.xlabel('$Frequency$')
      plt.ylabel('$Magnitude$')

      plt.title(datatype)
      plt.legend(loc='best')
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

#  Fully connected autoencoder via Model interface (using Sequential interface internally)
class DenseAutoEnc(Model):

  def __init__(self, ae_shape, name='DenseAutoEnc'):
    super(DenseAutoEnc, self).__init__(name=name)
    # Encoder
    self.encoder = tf.keras.Sequential(name='Encoder')
    self.encoder.add(Flatten())
    _enc_dense_layer(self.encoder, 1024, "re")

    _enc_dense_layer(self.encoder, 512, "re")

    _enc_dense_layer(self.encoder, 256, "re")

    _enc_dense_layer(self.encoder, 128, "re")

    _enc_dense_layer(self.encoder, 64, "re")

    _enc_dense_layer(self.encoder, 32, "sig")
    # Decoder
    self.decoder = tf.keras.Sequential(name='Decoder')
    _dec_dense_layer(self.decoder, 32, "re")
    _dec_dense_layer(self.decoder, 64, "re")
    _dec_dense_layer(self.decoder, 128, "re")
    _dec_dense_layer(self.decoder, 256, "re")
    _dec_dense_layer(self.decoder, 512, "re")
    _dec_dense_layer(self.decoder, 1024, "re")
    _dec_dense_layer(self.decoder, 2048, "sig")
    self.decoder.add(Reshape((1, 2048)))

    self.build((None,*ae_shape))

  def call(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Autoencoder:

  def __init__(self,
               model, metabolites, pulse_sequence,
               acquisitions, datatype, norm):
    self.model = model
    self.metabolites = metabolites
    self.pulse_sequence = pulse_sequence
    self.acquisitions = acquisitions
    self.datatype = datatype
    self.norm = norm

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
    self.ae = DenseAutoEnc(ae_shape=ae_shape)

  def train(self, d_inp, d_out, v_inp, v_out, epochs, batch_size,
            folder, verbose=0,no_show=False, image_dpi=[300], screen_dpi=96, train_dataset_name=""):
    if tf.test.gpu_device_name():
      print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
      print("WARNING, we do not have a default GPU for Tensorflow!")

    if len(train_dataset_name) > 0:
      self.train_dataset_name = train_dataset_name

    if not os.path.isdir(folder):
      os.makedirs(folder)

    if verbose > 0:
      print("# Train Autoencoder %s" % str(self))

    # Autoencoder - flatten tensor as for CNN - FIXME: maybe 3D OK here, no flatting?
    d_inp = tf.convert_to_tensor(d_inp, dtype=tf.float32)
    d_out = tf.convert_to_tensor(d_out, dtype=tf.float32)
    #d_inp = tf.reshape(d_inp,(d_inp.shape[0],d_inp.shape[1]*d_inp.shape[2],d_inp.shape[3],1))
    #d_out = tf.reshape(d_out,(d_out.shape[0],d_out.shape[1]*d_out.shape[2],d_out.shape[3],1))
    d_inp = tf.reshape(d_inp, (d_inp.shape[0], d_inp.shape[1] * d_inp.shape[2], d_inp.shape[3]))
    d_out = tf.reshape(d_out, (d_out.shape[0], d_out.shape[1] * d_out.shape[2], d_out.shape[3]))

    if len(v_inp) > 0:
      v_inp = tf.convert_to_tensor(v_inp, dtype=tf.float32)
      v_out = tf.convert_to_tensor(v_out, dtype=tf.float32)
      #v_inp = tf.reshape(v_inp,(v_inp.shape[0],v_inp.shape[1]*v_inp.shape[2],v_inp.shape[3],1))
      #v_out = tf.reshape(v_out,(v_out.shape[0],v_out.shape[1]*v_out.shape[2],v_out.shape[3],1))
      v_inp = tf.reshape(v_inp, (v_inp.shape[0], v_inp.shape[1] * v_inp.shape[2], v_inp.shape[3]))
      v_out = tf.reshape(v_out, (v_out.shape[0], v_out.shape[1] * v_out.shape[2], v_out.shape[3]))

      validation_data = (v_inp,v_out)
    else:
      validation_data = None

    if verbose > 1:
      print("  Input:",d_inp.shape,"[spectrum, acquisition x datatype, frequency]")
      print("  Output:",d_out.shape,"[spectrum, acquisition x datatype, frequency]")

    self._construct(d_inp.shape[1:])# It's ae.shape argument

    optimiser = keras.optimizers.Adam(learning_rate=1e-4,
                                      beta_1=0.9,
                                      beta_2=0.999)
    self.ae.compile(loss='mse',
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
    history = self.ae.fit(x=d_inp,
                          y=d_out,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=(verbose > 0)*2,
                          validation_data=validation_data,
                          shuffle=True,
                          callbacks=callbacks)

    le = len(history.history['loss'])
    history.history['time (ms)'] = np.add(timer.times[:le,1],-timer.times[:le,0]) // 1000000


    x_test_n = tf.convert_to_tensor(d_out, dtype=tf.float32)
    x_test_n = tf.reshape(x_test_n, (3500, 1, 2048))
    x_test = tf.convert_to_tensor(d_inp, dtype=tf.float32)
    x_test = tf.reshape(x_test, (3500, 1, 2048))
    encoder_imgs = self.ae.encoder(x_test_n[0]).numpy()
    decoder_imgs = self.ae.decoder(encoder_imgs).numpy()
    plot_spectra_(decoder_imgs[0, 0], x_test_n[0, 0], 'Contrast with Noise_spectra')
    plot_spectra_(decoder_imgs[0, 0], x_test[0, 0], 'Contrast with Clean_spectra')


    if verbose > 0:
      print("# Evaluating")
    d_score = self.ae.evaluate(d_inp, d_out, verbose=(verbose > 0)*2)
    if len(v_inp) > 0:
      v_score = self.ae.evaluate(v_inp, v_out, verbose=(verbose > 0)*2)
    else:
      v_score = np.array([np.nan,np.nan])
    if verbose > 0:
      print("      Train          Validation")
      print('MSE:  %.12f %.12f' % (d_score[0], v_score[0]))
      print('MAE:  %.12f %.12f' % (d_score[1], v_score[1]))
    self._save_results(folder, history.history, d_score, v_score, no_show, image_dpi, screen_dpi)

    d_res={"MSE":d_score[0],"MAE":d_score[1]}
    v_res={"MSE":v_score[0],"MAE":v_score[1]}
    return d_res, v_res

  # FIXME: if this is not defined, analyse_model will not analyse concentration predictions and quantify does not work
  #        this is OK as long as we only have the autoencoder part.
  #def predict(self, d_inp, reshape=True, verbose=0):
  #  if reshape:
  #    d_inp = tf.convert_to_tensor(d_inp, dtype=tf.float32)
  #    d_inp = tf.reshape(d_inp,(d_inp.shape[0],d_inp.shape[1]*d_inp.shape[2],d_inp.shape[3],1))
  #  return np.array(self.ae.predict(x=d_inp,verbose=(verbose>0)*2,batch_size=32),dtype=np.float64)

  def save(self, folder):
    path=os.path.join(folder, "tf_model")
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
    with open(os.path.join(path, "tf_model", "mrsnet.json"), 'r') as f:
      data = json.load(f)
    model = Autoencoder(data['model'], data['metabolites'], data['pulse_sequence'], data['acquisitions'],
                        data['datatype'], data['norm'])
    model.train_dataset_name = data['train_dataset_name']
    model.ae = load_model(os.path.join(path,"tf_model"))
    return model

  def _save_results(self, folder, history, d_score, v_score, no_show, image_dpi, screen_dpi):
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
      writer.writerows(zip(*[history[key] for key in keys]))
    # Plot
    fig, axes = plt.subplots(1, 3)
    fig.suptitle("%s Training Results" % self.model)
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
    if not no_show:
      fig.set_dpi(screen_dpi)
      plt.show(block=True)
    plt.close()