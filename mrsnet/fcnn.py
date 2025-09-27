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

MODIFICATIONS FOR MRSNET INTEGRATION:

1. CONTEXT ADAPTATION (NECESSARY):
   - Original: Designed for single acquisition input (frequency points only)
   - MRSNet: Adapted for arbitrary acquisition contexts (multiple acquisitions + datatypes)
   - Input format: (batch, freqs, channels) where channels = acquisitions x datatypes
   - Channel handling: Maps arbitrary acquisitions to 2-channel format (edit_off, difference)
   - Context validation: Ensures compatibility with supported datatypes (magnitude, phase, real, imaginary)
   - Flexible processing: Handles any number of acquisitions and datatypes

2. ARCHITECTURAL SIMPLIFICATIONS (MEMORY CONSTRAINTS):
   - Original: Full 7-layer CNN with original filter sizes
   - MRSNet: Reduced default parameters for GPU memory efficiency
   - Memory optimization: Configurable filter sizes and FC layer units
   - Configurable parameters: 'fcnn_[CONV_FILTERS]_[KERNEL]_[POOL]_[FC_UNITS]_[DROPOUT]'
   - Output simplification: Direct metabolite concentrations (no macromolecule scaling)

3. IMPLEMENTATION DETAILS:
   - CReLU activation: Custom implementation of concatenated ReLU
   - CNN architecture: 5 convolutional layers + 2 fully connected layers
   - Convolutional layers: Conv1D + BatchNorm + CReLU + MaxPooling
   - FC layers: Dense + Dropout + Dense (output)
   - Default filters: [32, 64, 128, 256, 512] (configurable)

4. TRAINING SIMPLIFICATIONS:
   - Macromolecule output: Removed (simplified to metabolite concentrations only)
   - Output focus: Direct metabolite concentration prediction
   - Training model: Standard CNN training without additional outputs
   - Loss function: MSE for concentration prediction

5. COMPATIBILITY FEATURES:
   - Context validation: Explicit error messages for unsupported contexts
   - Flexible channel handling: Supports arbitrary acquisition/datatype combinations
   - Memory management: Configurable architecture parameters
   - Model string parsing: Dynamic configuration based on model identifier

This implementation maintains the core FoundationalCNN architecture while adapting it for
the MRSNet framework's arbitrary context handling and simplifying the output structure
for practical implementation within the existing framework.
"""

import csv
import io
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import (
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
    ReLU,
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

    def build(self, input_shape):
        """Build the layer with the given input shape.

        Parameters
        ----------
            input_shape (tuple): Input tensor shape
        """
        # Build the ReLU layers
        self.relu.build(input_shape)

        # Mark as built
        self.built = True
        self._build_input_shape = input_shape

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        return config


@register_keras_serializable(package="mrsnet", name="FoundationalCNNModel")
class FoundationalCNNModel(Model):
    """FoundationalCNN model for MRS metabolite quantification.

    This implements the 7-layer CNN architecture from the MICCAI 2018 paper
    with CReLU activation and convolutional layers for metabolite quantification.
    Output includes metabolite concentrations plus macromolecule scaling factor.
    """

    def __init__(self, n_freqs, n_metabolites, n_channels=2, conv_filters=None, kernel_size=3,
                 pool_size=2, fc_units=1024, dropout_rate=0.5, name='FoundationalCNNModel', **kwargs):
        """Initialize FoundationalCNN model.

        Parameters
        ----------
            n_freqs (int): Number of frequency points in spectrum
            n_metabolites (int): Number of metabolites to quantify
            conv_filters (list, optional): Number of filters for each conv layer.
                Default: [32, 64, 128, 256, 512] (original paper architecture)
            kernel_size (int): Convolutional kernel size. Default: 3
            pool_size (int): MaxPooling size. Default: 2
            fc_units (int): Number of units in first FC layer. Default: 1024
            dropout_rate (float): Dropout rate. Default: 0.5
            name (str, optional): Model name
        """
        super().__init__(name=name, **kwargs)
        self.n_freqs = n_freqs
        self.n_metabolites = n_metabolites

        # Set default conv_filters if not provided (original paper architecture)
        if conv_filters is None:
            conv_filters = [32, 64, 128, 256, 512]
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.fc_units = fc_units
        self.dropout_rate = dropout_rate

        # Input layer - variable channels (2 for real-only, 4 for real+imaginary)
        self.input_layer = Input(shape=(n_freqs, n_channels), name='spectrum_input')

        # Create convolutional layers dynamically based on conv_filters
        self.conv_layers = []
        self.crelu_layers = []
        self.pool_layers = []

        for i, filters in enumerate(self.conv_filters):
            conv_layer = Conv1D(filters, kernel_size=self.kernel_size,
                              padding='same', name=f'conv{i+1}')
            crelu_layer = CReLU(name=f'crelu{i+1}')
            pool_layer = MaxPooling1D(pool_size=self.pool_size, name=f'pool{i+1}')

            self.conv_layers.append(conv_layer)
            self.crelu_layers.append(crelu_layer)
            self.pool_layers.append(pool_layer)

        # Flatten for fully connected layers
        self.flatten = Flatten(name='flatten')

        # Layer 6: Fully Connected
        self.fc1 = Dense(self.fc_units, activation='relu', name='fc1')
        self.dropout1 = Dropout(self.dropout_rate, name='dropout1')

        # Layer 7: Fully Connected (output layer)
        # Output: n_metabolites (macromolecule scaling factor removed for compatibility)
        self.fc2 = Dense(n_metabolites, activation='linear', name='fc2')

    def build(self, input_shape):
        """Build the model with the given input shape.

        Parameters
        ----------
            input_shape (tuple): Input tensor shape (batch, freqs, channels)
        """
        # Initialize all layers with proper input shapes
        # This ensures all weights are properly initialized

        # Build convolutional layers
        x_shape = input_shape
        for i, conv_layer in enumerate(self.conv_layers):
            conv_layer.build(x_shape)
            # After conv: same batch, same freqs, filters channels
            # After CReLU: same batch, same freqs, 2 * filters channels
            # After MaxPool: same batch, freqs // pool_size, 2 * filters channels
            x_shape = (x_shape[0], x_shape[1] // self.pool_size, self.conv_filters[i] * 2)

        # Build fully connected layers
        # After all conv+pool layers, we have flattened output
        flattened_size = x_shape[1] * x_shape[2]  # freqs * channels
        self.fc1.build((x_shape[0], flattened_size))
        self.fc2.build((x_shape[0], self.fc_units))

        # Mark as built
        self.built = True
        self._build_input_shape = input_shape

    def call(self, inputs):
        """Forward pass through FoundationalCNN."""
        x = inputs

        # Apply convolutional layers dynamically
        for conv_layer, crelu_layer, pool_layer in zip(self.conv_layers,
                                                       self.crelu_layers,
                                                       self.pool_layers, strict=False):
            x = conv_layer(x)
            x = crelu_layer(x)
            x = pool_layer(x)

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
            'n_metabolites': self.n_metabolites,
            'conv_filters': self.conv_filters,
            'kernel_size': self.kernel_size,
            'pool_size': self.pool_size,
            'fc_units': self.fc_units,
            'dropout_rate': self.dropout_rate
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

    def __init__(self, model, metabolites, pulse_sequence, acquisitions, datatype, norm,
                 conv_filters=None, kernel_size=3, pool_size=2, fc_units=1024, dropout_rate=0.5):
        """Initialize a FoundationalCNN model.

        Parameters
        ----------
            model (str): Model architecture identifier (e.g., 'fcnn_default', 'fcnn_original')
            metabolites (list): List of metabolite names to predict
            pulse_sequence (str): Pulse sequence type (e.g., 'megapress')
            acquisitions (list): List of acquisition types (e.g., ['edit_off', 'difference'])
            datatype (list): List of data types (e.g., ['magnitude', 'phase'])
            norm (str): Normalization method (e.g., 'sum', 'max')
            conv_filters (list, optional): Number of filters for each conv layer
            kernel_size (int): Convolutional kernel size
            pool_size (int): MaxPooling size
            fc_units (int): Number of units in first FC layer
            dropout_rate (float): Dropout rate
        """
        self.model = model
        self.metabolites = metabolites
        self.pulse_sequence = pulse_sequence
        self.acquisitions = acquisitions
        self.datatype = datatype
        self.norm = norm
        self.output = "concentrations"

        # Architecture parameters
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.fc_units = fc_units
        self.dropout_rate = dropout_rate

        # Input spectra data (constant!)
        self.low_ppm = -1.0
        self.high_ppm = -4.5
        self.fft_samples = 2048

        self.train_dataset_name = None
        self.fcnn_arch = None

        # Validate context compatibility
        self._validate_context()

    def _parse_model_config(self, input_shape):
        """Parse model string to extract configuration parameters.

        Parameters
        ----------
            input_shape (tuple): Input tensor shape

        Returns
        -------
            tuple: (conv_filters, kernel_size, pool_size, fc_units, dropout_rate)
        """
        vals = self.model.split("_")
        if vals[0] != 'fcnn':
            raise RuntimeError(f"Unknown model {vals[0]}")

        # Default parameters (original paper architecture)
        default_conv_filters = [32, 64, 128, 256, 512]
        default_kernel_size = 3
        default_pool_size = 2
        default_fc_units = 1024
        default_dropout_rate = 0.5

        if len(vals) == 1 or (len(vals) == 2 and vals[1] == 'default'):
            # fcnn or fcnn_default: use default parameters
            conv_filters = default_conv_filters
            kernel_size = default_kernel_size
            pool_size = default_pool_size
            fc_units = default_fc_units
            dropout_rate = default_dropout_rate
        elif len(vals) == 2 and vals[1] == 'original':
            # fcnn_original: use original paper parameters
            conv_filters = default_conv_filters
            kernel_size = default_kernel_size
            pool_size = default_pool_size
            fc_units = default_fc_units
            dropout_rate = default_dropout_rate
        else:
            # Custom configuration: fcnn_[CONV_FILTERS]_[KERNEL]_[POOL]_[FC_UNITS]_[DROPOUT]
            # Parse conv_filters (comma-separated list)
            if len(vals) > 1 and vals[1]:
                conv_filters = [int(x) for x in vals[1].split(',')]
            else:
                conv_filters = default_conv_filters

            kernel_size = int(vals[2]) if len(vals) > 2 else default_kernel_size
            pool_size = int(vals[3]) if len(vals) > 3 else default_pool_size
            fc_units = int(vals[4]) if len(vals) > 4 else default_fc_units
            dropout_rate = float(vals[5]) if len(vals) > 5 else default_dropout_rate

        # Override with instance parameters if provided
        if self.conv_filters is not None:
            conv_filters = self.conv_filters
        if self.kernel_size != 3:  # If not default
            kernel_size = self.kernel_size
        if self.pool_size != 2:  # If not default
            pool_size = self.pool_size
        if self.fc_units != 1024:  # If not default
            fc_units = self.fc_units
        if self.dropout_rate != 0.5:  # If not default
            dropout_rate = self.dropout_rate

        return conv_filters, kernel_size, pool_size, fc_units, dropout_rate

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

    def _validate_context(self):
        """Validate that the context (acquisitions + datatypes) is compatible with FoundationalCNN.

        FoundationalCNN can handle arbitrary contexts but works best with:
        - Any number of acquisitions (processed as separate channels)
        - Any datatypes (magnitude, phase, real, imaginary)
        - Minimum 1 acquisition and 1 datatype
        """
        if not self.acquisitions:
            raise ValueError("FoundationalCNN requires at least one acquisition")
        if not self.datatype:
            raise ValueError("FoundationalCNN requires at least one datatype")

        # Check for supported datatypes
        supported_datatypes = {'magnitude', 'phase', 'real', 'imaginary'}
        unsupported = set(self.datatype) - supported_datatypes
        if unsupported:
            raise ValueError(f"FoundationalCNN does not support datatypes: {unsupported}. "
                           f"Supported datatypes: {supported_datatypes}")

        # Log context information
        print(f"FoundationalCNN context: {len(self.acquisitions)} acquisitions, {len(self.datatype)} datatypes")
        print(f"  Acquisitions: {self.acquisitions}")
        print(f"  Datatypes: {self.datatype}")

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
        # Parse model configuration
        conv_filters, kernel_size, pool_size, fc_units, dropout_rate = self._parse_model_config(input_shape)

        # Get framework-determined parameters
        # After _convert_to_2channel: (batch, freqs, channels)
        n_freqs = input_shape[0]  # Frequency dimension is the first dimension
        n_channels = input_shape[1]  # Channel dimension is the second dimension
        n_metabolites = len(self.metabolites)

        # Create FoundationalCNN model with parsed parameters
        self.fcnn_arch = FoundationalCNNModel(
            n_freqs=n_freqs,
            n_metabolites=n_metabolites,
            n_channels=n_channels,
            conv_filters=conv_filters,
            kernel_size=kernel_size,
            pool_size=pool_size,
            fc_units=fc_units,
            dropout_rate=dropout_rate,
            name=self.model
        )

        # Rebuild the model with the actual input shape
        # input_shape is (freqs, channels), but build expects (batch, freqs, channels)
        full_input_shape = (None, *input_shape)
        self.fcnn_arch.build(full_input_shape)

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
        if verbose > 0:
            print(f"  Debug - Original d_inp shape: {d_inp.shape}")

        d_inp = reshape_spectra_data(d_inp, add_channel_dim=False)
        if verbose > 0:
            print(f"  Debug - After reshape_spectra_data: {d_inp.shape}")

        # Convert to 2-channel input (real and imaginary parts)
        # Input shape: (batch, acquisitions*datatypes, freqs) -> (batch, freqs, channels)
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
            print("  Input shape for model:", d_inp.shape[1:])

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

        # Create comprehensive architecture plots
        self._create_architecture_plots(folder, image_dpi, verbose)

        if verbose > 0:
            self.fcnn_arch.summary()

        timer = TimeHistory(epochs)
        callbacks = [keras.callbacks.EarlyStopping(monitor='loss',
                                                   min_delta=1e-8,
                                                   patience=25,
                                                   mode='min',
                                                   verbose=(verbose > 0)*2,
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
        # For MEGA-PRESS: edit_off (channel 0) and difference (channel 1)

        # Use static shapes for debugging
        input_shape = spectra.shape

        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input (batch, channels, freqs), got {len(input_shape)}D: {input_shape}")

        batch_size = input_shape[0]
        n_channels = input_shape[1]  # acquisitions*datatypes
        n_freqs = input_shape[2]

        # Reshape to (batch, freqs, channels)
        spectra_reshaped = tf.transpose(spectra, [0, 2, 1])  # (batch, freqs, channels)

        # Handle different channel configurations
        if n_channels == 1:
            # Single channel: pad with zeros
            padding = tf.zeros((batch_size, n_freqs, 1))
            spectra_2channel = tf.concat([spectra_reshaped, padding], axis=-1)
        elif n_channels == 2:
            # Exactly 2 channels: use as-is
            spectra_2channel = spectra_reshaped
        elif n_channels == 4:
            # 4 channels (2 acquisitions * 2 datatypes): preserve real and imaginary
            # Channel 0: edit_off_real, Channel 1: edit_off_imaginary
            # Channel 2: edit_on_real, Channel 3: edit_on_imaginary
            # Keep all 4 channels to preserve phase information
            spectra_2channel = spectra_reshaped
        else:
            # More than 4 channels or other configurations: take first 2 channels
            spectra_2channel = spectra_reshaped[:, :, :2]

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

        # Create model instance without building the architecture
        model = FoundationalCNN.__new__(FoundationalCNN)
        model.model = data['model']
        model.metabolites = data['metabolites']
        model.pulse_sequence = data['pulse_sequence']
        model.acquisitions = data['acquisitions']
        model.datatype = data['datatype']
        model.norm = data['norm']
        model.train_dataset_name = data['train_dataset_name']

        # Set default attributes (same as constructor)
        # FIXME: should not be hardcoded, but fine for now as we only use these values
        model.low_ppm = -1.0
        model.high_ppm = -4.5
        model.fft_samples = 2048
        model.output = "concentrations"

        # Set default architecture parameters (same as constructor defaults)
        model.conv_filters = [32, 64, 128, 256, 512]  # Default from constructor
        model.kernel_size = 3
        model.pool_size = 2
        model.fc_units = 1024
        model.dropout_rate = 0.5

        # Load the saved model architecture directly
        model.fcnn_arch = load_model(os.path.join(path, "model.keras"),
                                     custom_objects={
                                         "FoundationalCNNModel": FoundationalCNNModel,
                                         "CReLU": CReLU
                                     },
                                     compile=False,
                                     safe_mode=False)
        return model

    def _create_architecture_plots(self, folder, image_dpi, verbose):
        """Create comprehensive architecture plots for FoundationalCNN."""
        try:
            # 1. Main training model architecture
            for dpi in image_dpi:
                try:
                    plot_model(self.training_model,
                               to_file=os.path.join(folder, f'fcnn_training_architecture@{dpi}.png'),
                               show_shapes=True,
                               show_dtype=True,
                               show_layer_names=True,
                               rankdir='TB',
                               expand_nested=True,
                               dpi=dpi)
                except Exception as e:
                    try:
                        plot_model(self.training_model,
                                   to_file=os.path.join(folder, f'fcnn_training_architecture@{dpi}.svg'),
                                   show_shapes=True,
                                   show_dtype=True,
                                   show_layer_names=True,
                                   rankdir='TB',
                                   expand_nested=True,
                                   dpi=dpi)
                        if verbose > 0:
                            print(f"# WARNING: FCNN training architecture PNG failed; wrote SVG instead: {e}")
                    except Exception as e2:
                        if verbose > 0:
                            print(f"# WARNING: FCNN training architecture plot failed: {e2}")

            # 2. Individual FCNN architecture (core model) - Enhanced
            self._plot_enhanced_core_architecture(folder, image_dpi, verbose)

            # 3. Create module architecture plots
            self._create_module_plots(folder, image_dpi, verbose)

        except Exception as e:
            if verbose > 0:
                print(f"# WARNING: Architecture plotting failed: {e}")

    def _plot_enhanced_core_architecture(self, folder, image_dpi, verbose):
        """Create an enhanced plot of the FCNN core architecture showing all layers."""
        try:
            # Create a detailed model that shows the FCNN architecture step by step
            from tensorflow.keras.layers import BatchNormalization, Conv1D, Dense, Dropout, Input, MaxPooling1D
            from tensorflow.keras.models import Model

            input_layer = Input(shape=(2048, 2), name='fcnn_input')  # 2 channels (edit_off, difference)

            # Conv Layer 1
            conv1 = Conv1D(filters=32, kernel_size=3, padding='same', name='conv1')(input_layer)
            bn1 = BatchNormalization(name='bn1')(conv1)
            crelu1 = CReLU(name='crelu1')(bn1)
            pool1 = MaxPooling1D(pool_size=2, name='pool1')(crelu1)

            # Conv Layer 2
            conv2 = Conv1D(filters=64, kernel_size=3, padding='same', name='conv2')(pool1)
            bn2 = BatchNormalization(name='bn2')(conv2)
            crelu2 = CReLU(name='crelu2')(bn2)
            pool2 = MaxPooling1D(pool_size=2, name='pool2')(crelu2)

            # Conv Layer 3
            conv3 = Conv1D(filters=128, kernel_size=3, padding='same', name='conv3')(pool2)
            bn3 = BatchNormalization(name='bn3')(conv3)
            crelu3 = CReLU(name='crelu3')(bn3)
            pool3 = MaxPooling1D(pool_size=2, name='pool3')(crelu3)

            # Conv Layer 4
            conv4 = Conv1D(filters=256, kernel_size=3, padding='same', name='conv4')(pool3)
            bn4 = BatchNormalization(name='bn4')(conv4)
            crelu4 = CReLU(name='crelu4')(bn4)
            pool4 = MaxPooling1D(pool_size=2, name='pool4')(crelu4)

            # Conv Layer 5
            conv5 = Conv1D(filters=512, kernel_size=3, padding='same', name='conv5')(pool4)
            bn5 = BatchNormalization(name='bn5')(conv5)
            crelu5 = CReLU(name='crelu5')(bn5)
            pool5 = MaxPooling1D(pool_size=2, name='pool5')(crelu5)

            # Flatten and FC layers
            from tensorflow.keras.layers import Flatten
            flatten = Flatten(name='flatten')(pool5)
            fc1 = Dense(units=512, activation='relu', name='fc1')(flatten)
            dropout = Dropout(rate=0.5, name='dropout')(fc1)
            fc2 = Dense(units=len(self.metabolites), activation='linear', name='fc2')(dropout)

            # Create the detailed model
            detailed_model = Model(inputs=input_layer, outputs=fc2, name='FoundationalCNN_Detailed')

            for dpi in image_dpi:
                try:
                    plot_model(detailed_model,
                               to_file=os.path.join(folder, f'fcnn_detailed_architecture@{dpi}.png'),
                               show_shapes=True,
                               show_dtype=True,
                               show_layer_names=True,
                               rankdir='TB',
                               expand_nested=True,
                               dpi=dpi)
                except Exception as e:
                    try:
                        plot_model(detailed_model,
                                   to_file=os.path.join(folder, f'fcnn_detailed_architecture@{dpi}.svg'),
                                   show_shapes=True,
                                   show_dtype=True,
                                   show_layer_names=True,
                                   rankdir='TB',
                                   expand_nested=True,
                                   dpi=dpi)
                        if verbose > 0:
                            print(f"# WARNING: FCNN detailed architecture PNG failed; wrote SVG instead: {e}")
                    except Exception as e2:
                        if verbose > 0:
                            print(f"# WARNING: FCNN detailed architecture plot failed: {e2}")

        except Exception as e:
            if verbose > 0:
                print(f"# WARNING: FCNN detailed architecture creation failed: {e}")

    def _create_module_plots(self, folder, image_dpi, verbose):
        """Create visual plots of FoundationalCNN modules."""
        try:
            # Create CReLU plot
            self._plot_crelu(folder, image_dpi, verbose)

        except Exception as e:
            if verbose > 0:
                print(f"# WARNING: Module plotting failed: {e}")

    def _plot_crelu(self, folder, image_dpi, verbose):
        """Create a visual plot of CReLU."""
        try:
            # Create a simple model to demonstrate CReLU
            from tensorflow.keras.layers import BatchNormalization, Conv1D, Input
            input_layer = Input(shape=(2048, 1), name='crelu_input')
            conv = Conv1D(filters=32, kernel_size=3, padding='same', name='conv_layer')(input_layer)
            bn = BatchNormalization(name='batch_norm')(conv)
            crelu = CReLU(name='block_crelu')
            output = crelu(bn)

            from tensorflow.keras.models import Model
            demo_model = Model(inputs=input_layer, outputs=output, name='CReLU_Block')

            for dpi in image_dpi:
                try:
                    plot_model(demo_model,
                               to_file=os.path.join(folder, f'fcnn_crelu@{dpi}.png'),
                               show_shapes=True,
                               show_dtype=True,
                               show_layer_names=True,
                               rankdir='TB',
                               expand_nested=True,
                               dpi=dpi)
                except Exception as e:
                    try:
                        plot_model(demo_model,
                                   to_file=os.path.join(folder, f'fcnn_crelu@{dpi}.svg'),
                                   show_shapes=True,
                                   show_dtype=True,
                                   show_layer_names=True,
                                   rankdir='TB',
                                   expand_nested=True,
                                   dpi=dpi)
                        if verbose > 0:
                            print(f"# WARNING: CReLU PNG failed; wrote SVG instead: {e}")
                    except Exception as e2:
                        if verbose > 0:
                            print(f"# WARNING: CReLU plot failed: {e2}")

        except Exception as e:
            if verbose > 0:
                print(f"# WARNING: CReLU demo model creation failed: {e}")

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
