# mrsnet/cnn.py - MRSNet - CNN models
#
# SPDX-FileCopyrightText: Copyright (C) 2019 Max Chandler, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2020-2021 Frank C Langbein <frank@langbein.org>, Cardiff University
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

class CNN:
  def __init__(self, model, metabolites, pulse_sequence, acquisitions, datatype, norm):
    self.model = model
    self.metabolites = metabolites
    self.pulse_sequence = pulse_sequence
    self.acquisitions = acquisitions
    self.datatype = datatype
    self.norm = norm

    # Input data (constant!)
    self.low_ppm = -1.0
    self.high_ppm = -4.5
    self.fft_samples = 2048

    self.train_dataset_name = None
    self.cnn = None

  def __str__(self):
    n = os.path.join(self.model, "-".join(self.metabolites),
                     self.pulse_sequence, "-".join(self.acquisitions),
                     "-".join(self.datatype), self.norm)
    return n

  def reset(self):
    del self.cnn
    self.cnn = None
    self.train_dataset_name = None

  def _freq_conv_layer(self, filter, c, s, dropout):
    if s <= 0:
      self.cnn.add(Conv2D(filter, c))
    else:
      self.cnn.add(Conv2D(filter, c, strides=(1,s)))
    if dropout == 0.0:
      self.cnn.add(BatchNormalization())
    self.cnn.add(ReLU())
    if dropout > 0.0:
      self.cnn.add(Dropout(dropout))
    if s < 0:
      self.cnn.add(MaxPool2D((1,-s)))

  def _construct(self, input_shape, output_shape):
    self.cnn = Sequential(name=self.model)
    vals = self.model.split("_")
    if vals[0] != 'cnn':
      raise Exception(f"Unknown model {vals[0]}")
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

    self.cnn.add(InputLayer(input_shape=input_shape))
    self._freq_conv_layer(filter1, (1,freq_convolution1), strides1, dropout1)
    self._freq_conv_layer(filter1, (1,freq_convolution2), strides1, dropout1)

    while self.cnn.layers[-1].output.shape[1] != 1:
      self._freq_conv_layer(filter1,
                            (min(self.cnn.layers[-1].output.shape[1],3),freq_convolution3),
                            1, dropout1)

    for n_filters in [filter1, filter2]:
      for ii in range(2):
        self._freq_conv_layer(n_filters, (1, freq_convolution4), 1,        dropout1)
        self._freq_conv_layer(n_filters, (1, freq_convolution4), strides2, dropout1)

    self.cnn.add(Flatten())
    self.cnn.add(Dense(dense,activation="sigmoid"))
    if dropout2 > 0.0:
      self.cnn.add(Dropout(dropout2))
    self.cnn.add(Dense(output_shape[-1], activation=output_act))

  def train(self, d_inp, d_out, v_inp, v_out, epochs, batch_size,
            folder, verbose=0, image_dpi=[300], screen_dpi=96, train_dataset_name=""):
    if tf.test.gpu_device_name():
      print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
      print("WARNING, we do not have a default GPU for Tensorflow!")

    if len(train_dataset_name) > 0:
      self.train_dataset_name = train_dataset_name

    if not os.path.isdir(folder):
      os.makedirs(folder)

    if verbose > 0:
      print(f"# Train CNN {str(self)}")

    d_inp = tf.convert_to_tensor(d_inp, dtype=tf.float32)
    d_out = tf.convert_to_tensor(d_out, dtype=tf.float32)
    d_inp = tf.reshape(d_inp,(d_inp.shape[0],d_inp.shape[1]*d_inp.shape[2],d_inp.shape[3],1))

    if len(v_inp) > 0:
      v_inp = tf.convert_to_tensor(v_inp, dtype=tf.float32)
      v_out = tf.convert_to_tensor(v_out, dtype=tf.float32)
      v_inp = tf.reshape(v_inp,(v_inp.shape[0],v_inp.shape[1]*v_inp.shape[2],v_inp.shape[3],1))
      validation_data = (v_inp,v_out)
    else:
      validation_data = None

    if verbose > 1:
      print("  Input:",d_inp.shape,"[spectrum, acquisition x datatype, frequency]")
      print("  Output:",d_out.shape,"[spectrum, metabolite_concentration]")
    self._construct(d_inp.shape[1:],d_out.shape[1:])
    optimiser = keras.optimizers.Adam(learning_rate=1e-4,
                                      beta_1=0.9,
                                      beta_2=0.999)
    self.cnn.compile(loss='mse',
                     optimizer=optimiser,
                     metrics=['mae'])

    for dpi in image_dpi:
      plot_model(self.cnn,
                 to_file=os.path.join(folder,'architecture@'+str(dpi)+'.png'),
                 show_shapes=True,
                 show_dtype=True,
                 show_layer_names=True,
                 rankdir='TB',
                 expand_nested=True,
                 dpi=dpi)
    if verbose > 0:
      self.cnn.summary()

    timer = TimeHistory(epochs)
    callbacks = [keras.callbacks.EarlyStopping(monitor='loss',
                                               min_delta=1e-8,
                                               patience=25,
                                               mode='min',
                                               verbose=(verbose > 0),
                                               restore_best_weights=True),
                 timer]
    history = self.cnn.fit(x=d_inp,
                           y=d_out,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=(verbose > 0)*2,
                           validation_data=validation_data,
                           shuffle=True,
                           callbacks=callbacks)
    le = len(history.history['loss'])
    history.history['time (ms)'] = np.add(timer.times[:le,1],-timer.times[:le,0]) // 1000000

    if verbose > 0:
      print("# Evaluating")
    d_score = self.cnn.evaluate(d_inp, d_out, verbose=(verbose > 0)*2)
    if len(v_inp) > 0:
      v_score = self.cnn.evaluate(v_inp, v_out, verbose=(verbose > 0)*2)
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
    if reshape:
      d_inp = tf.convert_to_tensor(d_inp, dtype=tf.float32)
      d_inp = tf.reshape(d_inp,(d_inp.shape[0],d_inp.shape[1]*d_inp.shape[2],d_inp.shape[3],1))
    return np.array(self.cnn.predict(x=d_inp,verbose=(verbose>0)*2,batch_size=32),dtype=np.float64)

  def save(self, folder):
    path=os.path.join(folder, "tf_model")
    self.cnn.save(path)
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
    model = CNN(data['model'], data['metabolites'], data['pulse_sequence'], data['acquisitions'],
                data['datatype'], data['norm'])
    model.train_dataset_name = data['train_dataset_name']
    model.cnn = load_model(os.path.join(path,"tf_model"))
    return model

  def _save_results(self, folder, history, d_score, v_score, image_dpi, screen_dpi, verbose):
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
  def __init__(self, epochs):
    super(TimeHistory, self).__init__()
    self.counter = 0
    self.times = np.zeros((epochs,2),dtype=np.int64)
  def on_train_begin(self, logs=None):
    self.counter = 0
  def on_epoch_begin(self, batch, logs=None):
    self.times[self.counter,0] = time_ns()
  def on_epoch_end(self, batch, logs=None):
    self.times[self.counter,1] = time_ns()
    self.counter += 1
