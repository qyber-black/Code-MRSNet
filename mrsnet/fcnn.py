# mrsnet/fcnn.py - MRSNet - FoundationalCNN model
#
# SPDX-FileCopyrightText: Copyright (C) 2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""FoundationalCNN (fCNN) model for MRSNet.

This module implements the FoundationalCNN architecture from the paper:
"Quantification of Magnetic Resonance Spectroscopy Data Using Deep Learning"
Hatami, N., Sdika, M., Ratiney, H., 2018.
https://arxiv.org/abs/1806.07237

The FoundationalCNN is a 7-layer CNN model designed for MRS metabolite
quantification using CReLU activation and convolutional layers.

MODIFICATIONS FOR MRSNET:
* FoundationalCNN-specific 7-layer CNN architecture
* Custom CReLU activation layer for concatenated ReLU
* Maintained original 7-layer structure with CReLU activation
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
from tensorflow.keras.layers import (
    Activation, BatchNormalization, Conv1D, Dense, Dropout,
    Flatten, Input, MaxPooling1D, ReLU, Concatenate
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model

try:
    from keras.saving import register_keras_serializable
except Exception:
    from tensorflow.keras.utils import register_keras_serializable

from mrsnet.cfg import Cfg
from mrsnet.cnn import TimeHistory
from mrsnet.train import calculate_flops, reshape_spectra_data


@register_keras_serializable(package="mrsnet", name="CReLU")
class CReLU(keras.layers.Layer):
    """Concatenated ReLU activation as used in FoundationalCNN.

    CReLU concatenates ReLU(x) and ReLU(-x) to avoid learning redundant
    filters of both positive and negative phase information.
    """

    def __init__(self, name=None, **kwargs):
        """Initialize CReLU layer."""
        super().__init__(name=name, **kwargs)
        self.relu = ReLU()

    def call(self, inputs):
        """Forward pass through CReLU."""
        pos = self.relu(inputs)
        neg = self.relu(-inputs)
        return Concatenate()([pos, neg])

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        return config


@register_keras_serializable(package="mrsnet", name="FoundationalCNNModel")
class FoundationalCNNModel(Model):
    """FoundationalCNN model for MRS metabolite quantification.

    This implements the 7-layer CNN architecture from the MICCAI 2018 paper
    with CReLU activation and convolutional layers for metabolite quantification.
    """

    def __init__(self, n_freqs, n_metabolites, name='FoundationalCNNModel', **kwargs):
        """Initialize FoundationalCNN model.

        Parameters
        ----------
            n_freqs (int): Number of frequency points in spectrum
            n_metabolites (int): Number of metabolites to quantify
            name (str, optional): Model name
        """
        super().__init__(name=name, **kwargs)
        self.n_freqs = n_freqs
        self.n_metabolites = n_metabolites

        # Input layer - 2 channels (real and imaginary parts)
        self.input_layer = Input(shape=(n_freqs, 2), name='spectrum_input')

        # Layer 1: Conv1D + CReLU + MaxPool
        self.conv1 = Conv1D(32, kernel_size=3, padding='same', name='conv1')
        self.crelu1 = CReLU(name='crelu1')
        self.pool1 = MaxPooling1D(pool_size=2, name='pool1')

        # Layer 2: Conv1D + CReLU + MaxPool
        self.conv2 = Conv1D(64, kernel_size=3, padding='same', name='conv2')
        self.crelu2 = CReLU(name='crelu2')
        self.pool2 = MaxPooling1D(pool_size=2, name='pool2')

        # Layer 3: Conv1D + CReLU + MaxPool
        self.conv3 = Conv1D(128, kernel_size=3, padding='same', name='conv3')
        self.crelu3 = CReLU(name='crelu3')
        self.pool3 = MaxPooling1D(pool_size=2, name='pool3')

        # Layer 4: Conv1D + CReLU + MaxPool
        self.conv4 = Conv1D(256, kernel_size=3, padding='same', name='conv4')
        self.crelu4 = CReLU(name='crelu4')
        self.pool4 = MaxPooling1D(pool_size=2, name='pool4')

        # Layer 5: Conv1D + CReLU + MaxPool
        self.conv5 = Conv1D(512, kernel_size=3, padding='same', name='conv5')
        self.crelu5 = CReLU(name='crelu5')
        self.pool5 = MaxPooling1D(pool_size=2, name='pool5')

        # Flatten for fully connected layers
        self.flatten = Flatten(name='flatten')

        # Layer 6: Fully Connected
        self.fc1 = Dense(1024, activation='relu', name='fc1')
        self.dropout1 = Dropout(0.5, name='dropout1')

        # Layer 7: Fully Connected (output layer)
        # Output: n_metabolites (no macromolecule scaling factor for now)
        self.fc2 = Dense(n_metabolites, activation='linear', name='fc2')

        # Build the model
        self.build((None, n_freqs, 2))

    def call(self, inputs):
        """Forward pass through FoundationalCNN."""
        # Layer 1
        x = self.conv1(inputs)
        x = self.crelu1(x)
        x = self.pool1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.crelu2(x)
        x = self.pool2(x)

        # Layer 3
        x = self.conv3(x)
        x = self.crelu3(x)
        x = self.pool3(x)

        # Layer 4
        x = self.conv4(x)
        x = self.crelu4(x)
        x = self.pool4(x)

        # Layer 5
        x = self.conv5(x)
        x = self.crelu5(x)
        x = self.pool5(x)

        # Flatten and fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)

        return x

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'n_freqs': self.n_freqs,
            'n_metabolites': self.n_metabolites
        })
        return config


class FoundationalCNN:
    """FoundationalCNN for MRS metabolite quantification.

    This class implements the FoundationalCNN architecture from the MICCAI 2018
    paper for metabolite concentration prediction from MRS spectra.

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
        fcnn_arch (keras.Model): The actual FoundationalCNN model
    """

    def __init__(self, model, metabolites, pulse_sequence, acquisitions, datatype, norm):
        """Initialize a FoundationalCNN model.

        Parameters
        ----------
            model (str): Model architecture identifier (e.g., 'fcnn_default')
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
        self.fcnn_arch = None

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
        """Reset the FoundationalCNN architecture and training dataset name.

        Clears the current model architecture and training dataset information.
        """
        self.fcnn_arch = None
        self.train_dataset_name = None

    def _construct(self, input_shape, output_shape):
        """Construct the FoundationalCNN architecture using functional API.

        Parameters
        ----------
            input_shape (tuple): Input tensor shape
            output_shape (tuple): Output tensor shape
        """
        vals = self.model.split("_")
        if vals[0] != 'fcnn':
            raise RuntimeError(f"Unknown model {vals[0]}")

        # Parse model parameters
        if len(vals) == 1:
            # Default FoundationalCNN configuration
            n_freqs = input_shape[-2]  # Frequency dimension
            n_metabolites = len(self.metabolites)
        elif len(vals) == 2 and vals[1] == 'default':
            # fcnn_default configuration
            n_freqs = input_shape[-2]  # Frequency dimension
            n_metabolites = len(self.metabolites)
        else:
            # Custom FoundationalCNN configuration: fcnn_[FREQS]_[METABOLITES]
            n_freqs = int(vals[1]) if len(vals) > 1 else input_shape[-2]
            n_metabolites = int(vals[2]) if len(vals) > 2 else len(self.metabolites)

        # Create FoundationalCNN model
        self.fcnn_arch = FoundationalCNNModel(n_freqs, n_metabolites, name=self.model)

    def train(self, d_data, v_data, epochs, batch_size,
              folder, verbose=0, image_dpi=[300], screen_dpi=96, train_dataset_name=""):
        """Train the FoundationalCNN model on provided data.

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
            print(f"# Train FoundationalCNN {self!s}")

        # Prepare data
        d_inp = tf.convert_to_tensor(d_data[0], dtype=tf.float32)
        d_out = tf.convert_to_tensor(d_data[1], dtype=tf.float32)
        d_inp = reshape_spectra_data(d_inp, add_channel_dim=False)

        # Convert to 2-channel input (real and imaginary parts)
        # Input shape: (batch, acquisitions*datatypes, freqs) -> (batch, freqs, 2)
        d_inp = self._convert_to_2channel(d_inp)

        train_data = tf.data.Dataset.from_tensor_slices((d_inp, d_out))

        if v_data is not None:
            v_inp = tf.convert_to_tensor(v_data[0], dtype=tf.float32)
            v_out = tf.convert_to_tensor(v_data[1], dtype=tf.float32)
            v_inp = reshape_spectra_data(v_inp, add_channel_dim=False)
            v_inp = self._convert_to_2channel(v_inp)
            validation_data = tf.data.Dataset.from_tensor_slices((v_inp, v_out))
        else:
            validation_data = None

        if verbose > 0:
            print("  Input:", d_inp.shape, "[spectrum, frequency, channels]")
            print("  Output:", d_out.shape, "[spectrum, metabolite_concentration]")

        learning_rate = Cfg.val['base_learning_rate'] * batch_size / 16.0

        if len(devices) > 1:
            # Multi-GPU training
            dev_multiplier = len(devices)
            mirrored_strategy = tf.distribute.MirroredStrategy(devices=devices)
            with mirrored_strategy.scope():
                self._construct(d_inp.shape[1:], d_out.shape[1:])
                optimiser = keras.optimizers.Adam(learning_rate=learning_rate * dev_multiplier,
                                                  beta_1=Cfg.val['beta1'],
                                                  beta_2=Cfg.val['beta2'],
                                                  epsilon=Cfg.val['epsilon'])
                self.fcnn_arch.compile(loss='mse',
                                      optimizer=optimiser,
                                      metrics=['mae'])
        else:
            # Single GPU / CPU training
            dev_multiplier = 1
            self._construct(d_inp.shape[1:], d_out.shape[1:])
            optimiser = keras.optimizers.Adam(learning_rate=learning_rate,
                                              beta_1=Cfg.val['beta1'],
                                              beta_2=Cfg.val['beta2'],
                                              epsilon=Cfg.val['epsilon'])
            self.fcnn_arch.compile(loss='mse',
                                  optimizer=optimiser,
                                  metrics=['mae'])

        # Calculate FLOPs
        try:
            self.flops = calculate_flops(self.fcnn_arch, d_inp.shape[1:])
        except Exception as e:
            if verbose > 0:
                print(f"Error calculating FLOPs: {e}")
            self.flops = 0

        # Plot model architecture
        for dpi in image_dpi:
            try:
                plot_model(self.fcnn_arch,
                           to_file=os.path.join(folder,'architecture@'+str(dpi)+'.png'),
                           show_shapes=True,
                           show_dtype=True,
                           show_layer_names=True,
                           rankdir='TB',
                           expand_nested=True,
                           dpi=dpi)
            except Exception as e:
                try:
                    plot_model(self.fcnn_arch,
                               to_file=os.path.join(folder,'architecture@'+str(dpi)+'.svg'),
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
                        print("# WARNING: Skipping model architecture plot (Graphviz error):", e2)

        if verbose > 0:
            self.fcnn_arch.summary()

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
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        train_data = train_data.batch(batch_size * dev_multiplier).with_options(options)
        if validation_data is not None:
            validation_data = validation_data.batch(batch_size * dev_multiplier).with_options(options)

        # Train
        history = self.fcnn_arch.fit(train_data,
                                    validation_data=validation_data,
                                    epochs=epochs,
                                    verbose=(verbose > 0)*2,
                                    shuffle=True,
                                    callbacks=callbacks)

        le = len(history.history['loss'])
        history.history['time (ms)'] = np.add(timer.times[:le,1],-timer.times[:le,0]) // 1000000

        if verbose > 0:
            print("# Evaluating")
        d_score = self.fcnn_arch.evaluate(d_inp, d_out, verbose=(verbose > 0)*2)
        if v_data is not None:
            v_score = self.fcnn_arch.evaluate(v_inp, v_out, verbose=(verbose > 0)*2)
        else:
            v_score = np.array([np.nan,np.nan])
        if verbose > 0:
            print("      Train          Validation")
            print(f"MSE:  {d_score[0]:.12f} {v_score[0]:.12f}")
            print(f"MAE:  {d_score[1]:.12f} {v_score[1]:.12f}")

        self._save_results(folder, history.history, d_score, v_score, image_dpi, screen_dpi, verbose)

        d_res = {"MSE": d_score[0], "MAE": d_score[1]}
        v_res = {"MSE": v_score[0], "MAE": v_score[1]}
        return d_res, v_res

    def _convert_to_2channel(self, spectra):
        """Convert spectra to 2-channel format (real and imaginary).

        Parameters
        ----------
            spectra (tensor): Input spectra tensor

        Returns
        -------
            tensor: 2-channel spectra tensor
        """
        # Input shape: (batch, acquisitions*datatypes, freqs)
        # Output shape: (batch, freqs, 2)

        # For now, use magnitude and phase as the two channels
        # In practice, you might want to use real and imaginary parts
        batch_size = tf.shape(spectra)[0]
        n_channels = tf.shape(spectra)[1]  # acquisitions*datatypes
        n_freqs = tf.shape(spectra)[2]

        # Reshape to (batch, freqs, channels) and take first 2 channels
        spectra_reshaped = tf.transpose(spectra, [0, 2, 1])  # (batch, freqs, channels)

        # Take first 2 channels or pad if needed
        if n_channels >= 2:
            spectra_2channel = spectra_reshaped[:, :, :2]
        else:
            # Pad with zeros if we have fewer than 2 channels
            padding = tf.zeros((batch_size, n_freqs, 2 - n_channels))
            spectra_2channel = tf.concat([spectra_reshaped, padding], axis=-1)

        return spectra_2channel

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
            d_inp = reshape_spectra_data(d_inp, add_channel_dim=False)
            d_inp = self._convert_to_2channel(d_inp)

        # Dataset options
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        data = tf.data.Dataset.from_tensor_slices(d_inp).batch(32).with_options(options)

        # Get predictions
        concentrations = self.fcnn_arch.predict(data, verbose=(verbose>0)*2)

        return np.array(concentrations, dtype=np.float64)

    def save(self, folder):
        """Save the trained FoundationalCNN model to disk.

        Parameters
        ----------
            folder (str): Directory to save the model
        """
        os.makedirs(folder, exist_ok=True)
        self.fcnn_arch.save(os.path.join(folder, "model.keras"))

        # Calculate model metrics from summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.fcnn_arch.summary()
        sys.stdout = old_stdout
        summary_output = buffer.getvalue()

        # Parse parameter counts from summary with robust fallbacks
        total_params = None
        trainable_params = None
        non_trainable_params = None
        for line in summary_output.split('\n'):
            if 'Total params:' in line:
                total_match = line.split('Total params:')[1].split()[0].replace(',', '')
                try:
                    total_params = int(total_match)
                except Exception:
                    total_params = None
            elif 'Trainable params:' in line:
                trainable_match = line.split('Trainable params:')[1].split()[0].replace(',', '')
                try:
                    trainable_params = int(trainable_match)
                except Exception:
                    trainable_params = None
            elif 'Non-trainable params:' in line:
                non_trainable_match = line.split('Non-trainable params:')[1].split()[0].replace(',', '')
                try:
                    non_trainable_params = int(non_trainable_match)
                except Exception:
                    non_trainable_params = None

        # Fallbacks using model APIs
        if total_params is None:
            try:
                total_params = int(self.fcnn_arch.count_params())
            except Exception:
                total_params = 0
        if trainable_params is None:
            try:
                trainable_params = int(np.sum([tf.size(v).numpy() for v in self.fcnn_arch.trainable_variables]))
            except Exception:
                trainable_params = 0
        if non_trainable_params is None:
            try:
                non_trainable_params = int(np.sum([tf.size(v).numpy() for v in self.fcnn_arch.non_trainable_variables]))
            except Exception:
                non_trainable_params = max(0, total_params - trainable_params)

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
        """Load a trained FoundationalCNN model from disk.

        Parameters
        ----------
            path (str): Directory containing the saved model

        Returns
        -------
            FoundationalCNN: Loaded FoundationalCNN model instance
        """
        with open(os.path.join(path, "mrsnet.json")) as f:
            data = json.load(f)
        model = FoundationalCNN(data['model'], data['metabolites'], data['pulse_sequence'], data['acquisitions'],
                                data['datatype'], data['norm'])
        model.train_dataset_name = data['train_dataset_name']
        model.fcnn_arch = load_model(os.path.join(path, "model.keras"),
                                     custom_objects={
                                         "FoundationalCNNModel": FoundationalCNNModel,
                                         "CReLU": CReLU
                                     },
                                     safe_mode=False)
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
