# mrsnet/qmrs.py - MRSNet - QMRS model
#
# SPDX-FileCopyrightText: Copyright (C) 2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""QMRS (Quantitative Magnetic Resonance Spectroscopy) model for MRSNet.

This module implements the QMRS architecture from the paper:
"Q-MRS: A Deep Learning Framework for Quantitative Magnetic Resonance Spectra Analysis"

QMRS is a CNN-LSTM model with multi-headed MLP that uses transfer learning
and parameter constraints for MRS metabolite quantification.
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
    Flatten, Input, MaxPooling1D, ReLU, LSTM, Bidirectional,
    Concatenate, Lambda
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


@register_keras_serializable(package="mrsnet", name="InceptionModule")
class InceptionModule(keras.layers.Layer):
    """Inception module for QMRS architecture.

    This implements a simplified inception module with multiple
    convolutional branches of different kernel sizes.
    """

    def __init__(self, n_filters, name=None, **kwargs):
        """Initialize Inception module.

        Parameters
        ----------
            n_filters (int): Number of filters for each branch
            name (str, optional): Layer name
        """
        super().__init__(name=name, **kwargs)
        self.n_filters = n_filters

        # Branch 1: 1x1 convolution
        self.branch1 = Conv1D(n_filters, kernel_size=1, padding='same', activation='relu')

        # Branch 2: 3x1 convolution
        self.branch2 = Conv1D(n_filters, kernel_size=3, padding='same', activation='relu')

        # Branch 3: 5x1 convolution
        self.branch3 = Conv1D(n_filters, kernel_size=5, padding='same', activation='relu')

        # Concatenation layer
        self.concat = Concatenate(axis=-1)

    def call(self, inputs):
        """Forward pass through Inception module."""
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)

        return self.concat([branch1, branch2, branch3])

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({'n_filters': self.n_filters})
        return config


@register_keras_serializable(package="mrsnet", name="MultiHeadMLP")
class MultiHeadMLP(keras.layers.Layer):
    """Multi-headed MLP for QMRS parameter prediction.

    This implements separate heads for different parameter types:
    - Metabolite amplitudes
    - Global parameters (line-broadening, phases)
    - Individual line-broadening
    - Individual frequency shifts
    - Baseline coefficients
    """

    def __init__(self, n_metabolites, n_baseline_coeffs=6, name=None, **kwargs):
        """Initialize MultiHeadMLP.

        Parameters
        ----------
            n_metabolites (int): Number of metabolites
            n_baseline_coeffs (int): Number of baseline coefficients
            name (str, optional): Layer name
        """
        super().__init__(name=name, **kwargs)
        self.n_metabolites = n_metabolites
        self.n_baseline_coeffs = n_baseline_coeffs

        # Shared hidden layers
        self.fc1 = Dense(512, activation='relu', name='shared_fc1')
        self.fc2 = Dense(256, activation='relu', name='shared_fc2')
        self.dropout = Dropout(0.5, name='shared_dropout')

        # Head 1: Metabolite amplitudes
        self.amplitude_head = Dense(n_metabolites, activation='linear', name='amplitude_head')

        # Head 2: Global parameters (Gaussian line-broadening, zeroth-order phase, first-order phase)
        self.global_head = Dense(3, activation='linear', name='global_head')

        # Head 3: Individual Lorentzian line-broadening
        self.lorentzian_head = Dense(n_metabolites, activation='linear', name='lorentzian_head')

        # Head 4: Individual frequency shifts
        self.frequency_head = Dense(n_metabolites, activation='linear', name='frequency_head')

        # Head 5: Baseline coefficients (2 spectra * n_baseline_coeffs)
        self.baseline_head = Dense(2 * n_baseline_coeffs, activation='linear', name='baseline_head')

    def call(self, inputs):
        """Forward pass through MultiHeadMLP."""
        # Shared layers
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.dropout(x)

        # Individual heads
        amplitudes = self.amplitude_head(x)
        global_params = self.global_head(x)
        lorentzian_params = self.lorentzian_head(x)
        frequency_params = self.frequency_head(x)
        baseline_params = self.baseline_head(x)

        return {
            'amplitudes': amplitudes,
            'global_params': global_params,
            'lorentzian_params': lorentzian_params,
            'frequency_params': frequency_params,
            'baseline_params': baseline_params
        }

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'n_metabolites': self.n_metabolites,
            'n_baseline_coeffs': self.n_baseline_coeffs
        })
        return config


@register_keras_serializable(package="mrsnet", name="QMRSModel")
class QMRSModel(Model):
    """QMRS model for MRS metabolite quantification.

    This implements the CNN-LSTM architecture with multi-headed MLP
    from the QMRS paper for metabolite parameter prediction.
    """

    def __init__(self, n_freqs, n_metabolites, n_baseline_coeffs=6, name='QMRSModel', **kwargs):
        """Initialize QMRS model.

        Parameters
        ----------
            n_freqs (int): Number of frequency points in spectrum
            n_metabolites (int): Number of metabolites to quantify
            n_baseline_coeffs (int): Number of baseline coefficients
            name (str, optional): Model name
        """
        super().__init__(name=name, **kwargs)
        self.n_freqs = n_freqs
        self.n_metabolites = n_metabolites
        self.n_baseline_coeffs = n_baseline_coeffs

        # Input layer - 2 channels (edit-OFF and DIFF spectra)
        self.input_layer = Input(shape=(n_freqs, 2), name='spectrum_input')

        # Initial convolution and pooling
        self.conv1 = Conv1D(32, kernel_size=3, padding='same', activation='relu', name='conv1')
        self.pool1 = MaxPooling1D(pool_size=2, name='pool1')

        # Three inception modules with increasing filters
        self.inception1 = InceptionModule(32, name='inception1')
        self.inception2 = InceptionModule(33, name='inception2')  # n+1 filters
        self.inception3 = InceptionModule(34, name='inception3')  # n+2 filters

        # Bidirectional LSTM
        self.lstm = Bidirectional(LSTM(128, return_sequences=False),
                                merge_mode='concat', name='bidirectional_lstm')

        # Multi-headed MLP
        self.multi_head_mlp = MultiHeadMLP(n_metabolites, n_baseline_coeffs, name='multi_head_mlp')

        # Build the model
        self.build((None, n_freqs, 2))

    def call(self, inputs):
        """Forward pass through QMRS."""
        # Initial convolution and pooling
        x = self.conv1(inputs)
        x = self.pool1(x)

        # Inception modules
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)

        # Bidirectional LSTM
        x = self.lstm(x)

        # Multi-headed MLP
        outputs = self.multi_head_mlp(x)

        return outputs

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'n_freqs': self.n_freqs,
            'n_metabolites': self.n_metabolites,
            'n_baseline_coeffs': self.n_baseline_coeffs
        })
        return config


class QMRS:
    """QMRS for MRS metabolite quantification.

    This class implements the QMRS architecture with transfer learning
    and parameter constraints for metabolite concentration prediction
    from MRS spectra.

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
        qmrs_arch (keras.Model): The actual QMRS model
    """

    def __init__(self, model, metabolites, pulse_sequence, acquisitions, datatype, norm):
        """Initialize a QMRS model.

        Parameters
        ----------
            model (str): Model architecture identifier (e.g., 'qmrs_default')
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
        self.qmrs_arch = None

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
        """Reset the QMRS architecture and training dataset name.

        Clears the current model architecture and training dataset information.
        """
        self.qmrs_arch = None
        self.train_dataset_name = None

    def _create_training_model(self, input_shape, output_shape):
        """Create a training model that wraps QMRSModel for Keras training.

        Parameters
        ----------
            input_shape (tuple): Input tensor shape
            output_shape (tuple): Output tensor shape
        """
        # Create the QMRS model
        n_freqs = input_shape[-2]  # Frequency dimension
        n_metabolites = len(self.metabolites)

        self.qmrs_arch = QMRSModel(n_freqs, n_metabolites, name=self.model)

        # Create a training model that outputs only metabolite amplitudes
        input_layer = Input(shape=input_shape, name='training_input')
        qmrs_outputs = self.qmrs_arch(input_layer)

        # Extract only the amplitudes for training (ignore other parameters)
        if isinstance(qmrs_outputs, dict):
            amplitude_output = qmrs_outputs['amplitudes']
        else:
            amplitude_output = qmrs_outputs

        # Create training model
        self.training_model = Model(inputs=input_layer, outputs=amplitude_output, name=f"{self.model}_training")

        return self.training_model

    def _construct(self, input_shape, output_shape):
        """Construct the QMRS architecture using functional API.

        Parameters
        ----------
            input_shape (tuple): Input tensor shape
            output_shape (tuple): Output tensor shape
        """
        vals = self.model.split("_")
        if vals[0] != 'qmrs':
            raise RuntimeError(f"Unknown model {vals[0]}")

        # Parse model parameters
        if len(vals) == 1:
            # Default QMRS configuration
            n_freqs = input_shape[-2]  # Frequency dimension
            n_metabolites = len(self.metabolites)
        elif len(vals) == 2 and vals[1] == 'default':
            # qmrs_default configuration
            n_freqs = input_shape[-2]  # Frequency dimension
            n_metabolites = len(self.metabolites)
        else:
            # Custom QMRS configuration: qmrs_[FREQS]_[METABOLITES]
            n_freqs = int(vals[1]) if len(vals) > 1 else input_shape[-2]
            n_metabolites = int(vals[2]) if len(vals) > 2 else len(self.metabolites)

        # Create QMRS model
        self.qmrs_arch = QMRSModel(n_freqs, n_metabolites, name=self.model)

    def train(self, d_data, v_data, epochs, batch_size,
              folder, verbose=0, image_dpi=[300], screen_dpi=96, train_dataset_name=""):
        """Train the QMRS model on provided data.

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
            print(f"# Train QMRS {self!s}")

        # Prepare data
        d_inp = tf.convert_to_tensor(d_data[0], dtype=tf.float32)
        d_out = tf.convert_to_tensor(d_data[1], dtype=tf.float32)
        d_inp = reshape_spectra_data(d_inp, add_channel_dim=False)

        # Convert to 2-channel input (edit-OFF and DIFF spectra)
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
                self._create_training_model(d_inp.shape[1:], d_out.shape[1:])
                optimiser = keras.optimizers.Adam(learning_rate=learning_rate * dev_multiplier,
                                                  beta_1=Cfg.val['beta1'],
                                                  beta_2=Cfg.val['beta2'],
                                                  epsilon=Cfg.val['epsilon'])
                self.training_model.compile(loss='mse',
                                          optimizer=optimiser,
                                          metrics=['mae'])
        else:
            # Single GPU / CPU training
            dev_multiplier = 1
            self._create_training_model(d_inp.shape[1:], d_out.shape[1:])
            optimiser = keras.optimizers.Adam(learning_rate=learning_rate,
                                              beta_1=Cfg.val['beta1'],
                                              beta_2=Cfg.val['beta2'],
                                              epsilon=Cfg.val['epsilon'])
            self.training_model.compile(loss='mse',
                                      optimizer=optimiser,
                                      metrics=['mae'])

        # Calculate FLOPs
        try:
            self.flops = calculate_flops(self.training_model, d_inp.shape[1:])
        except Exception as e:
            if verbose > 0:
                print(f"Error calculating FLOPs: {e}")
            self.flops = 0

        # Plot model architecture
        for dpi in image_dpi:
            try:
                plot_model(self.training_model,
                           to_file=os.path.join(folder,'architecture@'+str(dpi)+'.png'),
                           show_shapes=True,
                           show_dtype=True,
                           show_layer_names=True,
                           rankdir='TB',
                           expand_nested=True,
                           dpi=dpi)
            except Exception as e:
                try:
                    plot_model(self.training_model,
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
            self.training_model.summary()

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
        history = self.training_model.fit(train_data,
                                         validation_data=validation_data,
                                         epochs=epochs,
                                         verbose=(verbose > 0)*2,
                                         shuffle=True,
                                         callbacks=callbacks)

        le = len(history.history['loss'])
        history.history['time (ms)'] = np.add(timer.times[:le,1],-timer.times[:le,0]) // 1000000

        if verbose > 0:
            print("# Evaluating")
        d_score = self.training_model.evaluate(d_inp, d_out, verbose=(verbose > 0)*2)
        if v_data is not None:
            v_score = self.training_model.evaluate(v_inp, v_out, verbose=(verbose > 0)*2)
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
        """Convert spectra to 2-channel format (edit-OFF and DIFF).

        Parameters
        ----------
            spectra (tensor): Input spectra tensor

        Returns
        -------
            tensor: 2-channel spectra tensor
        """
        # Input shape: (batch, acquisitions*datatypes, freqs)
        # Output shape: (batch, freqs, 2)

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
        predictions = self.qmrs_arch.predict(data, verbose=(verbose>0)*2)

        # Extract metabolite amplitudes from the multi-head output
        if isinstance(predictions, dict):
            concentrations = predictions['amplitudes']
        else:
            concentrations = predictions

        return np.array(concentrations, dtype=np.float64)

    def save(self, folder):
        """Save the trained QMRS model to disk.

        Parameters
        ----------
            folder (str): Directory to save the model
        """
        os.makedirs(folder, exist_ok=True)
        self.qmrs_arch.save(os.path.join(folder, "model.keras"))

        # Calculate model metrics from summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.training_model.summary()
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
                total_params = int(self.training_model.count_params())
            except Exception:
                total_params = 0
        if trainable_params is None:
            try:
                trainable_params = int(np.sum([tf.size(v).numpy() for v in self.training_model.trainable_variables]))
            except Exception:
                trainable_params = 0
        if non_trainable_params is None:
            try:
                non_trainable_params = int(np.sum([tf.size(v).numpy() for v in self.training_model.non_trainable_variables]))
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
        """Load a trained QMRS model from disk.

        Parameters
        ----------
            path (str): Directory containing the saved model

        Returns
        -------
            QMRS: Loaded QMRS model instance
        """
        with open(os.path.join(path, "mrsnet.json")) as f:
            data = json.load(f)
        model = QMRS(data['model'], data['metabolites'], data['pulse_sequence'], data['acquisitions'],
                     data['datatype'], data['norm'])
        model.train_dataset_name = data['train_dataset_name']
        model.qmrs_arch = load_model(os.path.join(path, "model.keras"),
                                     custom_objects={
                                         "QMRSModel": QMRSModel,
                                         "InceptionModule": InceptionModule,
                                         "MultiHeadMLP": MultiHeadMLP
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
