# mrsnet/selection.py - MRSNet - selection
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2019, Max Chandler, PhD student at Cardiff University
# Copyright (C) 2020-2021, Frank C Langbein <frank@langbein.org>, Cardiff University

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from .analyse import analyse_model
from .getfolder import get_folder
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras import Sequential
import mrsnet.dataset as dataset
class AETrain:

  '''def __init__(self,
               model, metabolites, pulse_sequence,
               acquisitions, datatype, norm,
               validate,
               d_inp, d_out, epochs, batch_size,
               path_mode, dataset_name,
               image_dpi, screen_dpi,
               no_show, verbose):
    self.model = model
    self.metabolites = metabolites
    self.pulse_sequence = pulse_sequence
    self.acquisitions = acquisitions
    self.datatype = datatype
    self.norm = norm
    self.validate = validate
    self.d_inp = d_inp
    self.d_out = d_out
    self.epochs = epochs
    self.batch_size = batch_size
    self.path_mode = path_mode
    self.dataset_name = dataset_name
    self.image_dpi = image_dpi
    self.screen_dpi = screen_dpi
    self.no_show = no_show
    self.verbose = verbose'''

  def __init__(self):
    self.AE = None

  def Encoder(self):
    # self.encoder = Sequential(name = self.model)
    # self.encoder.add(Input(shape=(1, 2048, 2)))
    self.en = tf.keras.Sequential([
      layers.Input(shape=(2, 2048, 1)),#1, 2048, 2
      layers.Conv2D(1024, (1, 9), activation='relu', padding='same'), #, strides=(1, 2)
      layers.BatchNormalization(),
      layers.MaxPooling2D(pool_size = (1, 4), padding='same'),
      layers.Dropout(0.3),
      layers.Conv2D(512, (1, 7), activation='relu', padding='same'),
      layers.MaxPooling2D(pool_size=(1, 2), padding='same'),
      layers.Dropout(0.3),
      layers.Conv2D(256, (1, 5), activation='relu', padding='same'),
      layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
      layers.Dropout(0.25),
      #layers.Conv2D(128, (1, 3), activation='relu', padding='same'),
      #layers.MaxPooling2D(pool_size=(1, 2), padding='same'),
      #layers.Dropout(0.25),
      #layers.Conv2D(64, (1, 2), activation='relu', padding='same'),
      #layers.MaxPooling2D(pool_size=(1, 2), padding='same'),
      #layers.Dropout(0.2),
      #layers.Conv2D(32, (1, 2), activation='relu', padding='same'),
      #layers.MaxPooling2D(pool_size=(1, 2), padding='same'),
      #layers.Dropout(0.2),
      #layers.Conv2D(16, (1, 2), activation='relu', padding='same', strides=(1, 2)),
      #layers.Dropout(0.2),
      #layers.Conv2D(8, (1, 2), activation='relu', padding='same')
       ],name='Encoder')

  def encoder(self, x):
    encoder = self.en(x)
    return encoder

  def Decoder(self):
    self.de = tf.keras.Sequential([
      #layers.Conv2D(16, kernel_size=(1, 2),activation='relu', padding='same'),
      #layers.UpSampling2D((1, 2)),
      #layers.Conv2D(32, kernel_size=(1, 2), activation='relu', padding='same'),
      #layers.UpSampling2D((1, 2)),
      #layers.Conv2D(64, kernel_size=(1, 2), activation='relu', padding='same'),
      #layers.UpSampling2D((1, 2)),
      #layers.Conv2D(128, kernel_size=(1, 3), activation='relu', padding='same'),
      #layers.UpSampling2D((2, 2)),
      layers.Conv2D(256, kernel_size=(1, 5), activation='relu', padding='same'),
      layers.UpSampling2D((2, 2)),
      layers.Conv2D(512, kernel_size=(1, 7), activation='relu', padding='same'),
      layers.UpSampling2D((1, 2)),
      layers.Conv2D(1024, kernel_size=(1, 9), activation='relu', padding='same'),
      layers.UpSampling2D((1, 4)),
      layers.Conv2D(1, kernel_size=(1, 9), activation='sigmoid', padding='same')],name='Decoder')

  def decoder(self, x):
    decoder = self.de(x)
    return decoder

  def Autoencoder(self):

    self.ae = Sequential()
    self.ae.add(self.en)
    self.ae.add(self.de)

  def train(self):
    #optimiser = tf.keras.optimizers.Adam(learning_rate=1e-4)

    train = self.ae.compile(optimizer='adam', loss=losses.MeanSquaredError())
    return train

  def Fit(self,train,test,train_n,test_n,z,batch_size):

    fit = self.ae.fit(train_n, train,
                      epochs=z,batch_size=batch_size,
                      shuffle=True,
                      validation_data=(test_n, test))
    return fit

  def summary(self):
    summary_en = self.en.summary()
    summary_de = self.de.summary()
    return summary_en, summary_de

  def train_(self):
    # FIXME: implement

    raise(Exception("Not implemented"))

  def plot_spectra(self,input,basis,starter,title,noise_p,noise_mu,noise_sigma,datatype):
    l = []
    for i in range(1, 2049):
      l.append(i)

    plt.figure()
    plt.plot(l, input,label='Spectra')
    plt.xlim(-1, 2048)
    plt.ylim()

    plt.xlabel('$x$')
    plt.ylabel('$y$')

    plt.title(starter+' '+title+' '+str(noise_p)+'_'+str(noise_mu)+'_'+str(noise_sigma)+' '+datatype)
    plt.legend(loc='best')
    plt.savefig('/home/zien/Desktop/Fig in MRSNet/Fig/' +basis+'/'+ datatype+ '_'+ title+'.png')
    #plt.show()

autoencoder = AETrain()

#autoencoder.Encoder()
#autoencoder.Decoder()
#autoencoder.Autoencoder()
#autoencoder.train()
#autoencoder.Fit()
#autoencoder.summary()


