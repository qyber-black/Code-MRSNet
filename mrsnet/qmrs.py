# mrsnet/qmrs.py - MRSNet - QMRS model
#
# SPDX-FileCopyrightText: Copyright (C) 2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""QMRS (Quantitative Magnetic Resonance Spectroscopy) model for MRSNet.

This module implements the QMRS architecture from the paper:
"Q-MRS: A Deep Learning Framework for Quantitative Magnetic Resonance Spectra Analysis"
by Wu, C.J, et al, 2024.
https://arxiv.org/abs/2408.15999

QMRS is a CNN-LSTM model with multi-headed MLP that uses transfer learning
and parameter constraints for MRS metabolite quantification.

MODIFICATIONS FOR MRSNET INTEGRATION:

1. CONTEXT ADAPTATION (NECESSARY):
   - Original: Designed for single acquisition input (281 frequency points)
   - MRSNet: Adapted for arbitrary acquisition contexts (multiple acquisitions + datatypes)
   - Input format: (batch, freqs, channels) where channels = acquisitions x datatypes
   - Channel handling: Maps arbitrary acquisitions to 2-channel format (edit_off, difference)
   - Context validation: Ensures compatibility with supported datatypes (magnitude, phase, real, imaginary)
   - Flexible processing: Handles any number of acquisitions and datatypes

2. ARCHITECTURAL SIMPLIFICATIONS (MEMORY CONSTRAINTS):
   - Original: Full transfer learning pipeline with pre-trained weights
   - MRSNet: Simplified implementation without transfer learning infrastructure
   - Memory optimization: Reduced default parameters for GPU compatibility
   - Configurable parameters: 'qmrs_[INITIAL_FILTERS]_[INCEPTION_FILTERS]_[LSTM_UNITS]_[MLP_UNITS]_[DROPOUT]'
   - Baseline coefficients: Fixed to 6 coefficients (not configurable)

3. IMPLEMENTATION DETAILS:
   - InceptionModule: Custom implementation with multi-scale convolutions (1x1, 3x3, 5x5)
   - CNN backbone: Initial conv layer + 2 inception modules + max pooling
   - LSTM layer: Bidirectional LSTM with concatenated merge mode
   - MultiHeadMLP: Separate MLPs for different parameter types
   - Output heads: Metabolite amplitudes, global parameters, line-broadening, frequency shifts, baseline

4. TRAINING SIMPLIFICATIONS:
   - Transfer learning: Not implemented (would require pre-trained model infrastructure)
   - Parameter constraints: Simplified to basic MLP outputs
   - Multi-head outputs: Maintained but simplified parameter prediction
   - Training model: Direct concentration output for MRSNet compatibility

5. COMPATIBILITY FEATURES:
   - Context validation: Explicit error messages for unsupported contexts
   - Flexible channel handling: Supports arbitrary acquisition/datatype combinations
   - Memory management: Configurable architecture parameters
   - Model string parsing: Dynamic configuration based on model identifier

This implementation maintains the core QMRS architecture while adapting it for
the MRSNet framework's arbitrary context handling and simplifying the transfer
learning components for practical implementation within the existing framework.
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
    LSTM,
    Bidirectional,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model

try:
    from keras.saving import register_keras_serializable
except Exception:
    from tensorflow.keras.utils import register_keras_serializable

from mrsnet.cfg import Cfg
from mrsnet.cnn import TimeHistory
from mrsnet.train import (
    LrHistory,
    calculate_flops,
    enable_deterministic_ops_if_configured,
    reshape_spectra_data,
    set_auto_mixed_precision_policy_if_enabled,
)


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

    def build(self, input_shape):
        """Build the layer with the given input shape.

        Parameters
        ----------
            input_shape (tuple): Input tensor shape
        """
        # Create a dummy input to build all layers
        dummy_input = tf.keras.Input(shape=input_shape[1:], name='dummy_input')

        # Forward pass to build all layers
        branch1 = self.branch1(dummy_input)
        branch2 = self.branch2(dummy_input)
        branch3 = self.branch3(dummy_input)
        output = self.concat([branch1, branch2, branch3]) # noqa: F841

        # Mark as built
        self.built = True
        self._build_input_shape = input_shape

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({'n_filters': self.n_filters})
        return config


@register_keras_serializable(package="mrsnet", name="MultiHeadMLP")
class MultiHeadMLP(keras.layers.Layer):
    """Multi-headed MLP for QMRS parameter prediction.

    This implements separate heads for different parameter types. In the MRSNet
    integration the amplitude head is active; other heads are disabled and
    returned as zero tensors for compatibility with the paper's interface.
    """

    def __init__(self, n_metabolites, n_baseline_coeffs=6, mlp_hidden_units=None,
                 dropout_rate=0.5, name=None, **kwargs):
        """Initialize MultiHeadMLP.

        Parameters
        ----------
            n_metabolites (int): Number of metabolites
            n_baseline_coeffs (int): Number of baseline coefficients
            mlp_hidden_units (list): Hidden units in MLP layers
            dropout_rate (float): Dropout rate
            name (str, optional): Layer name
        """
        super().__init__(name=name, **kwargs)
        self.n_metabolites = n_metabolites
        self.n_baseline_coeffs = n_baseline_coeffs

        # Set default MLP hidden units if not provided
        if mlp_hidden_units is None:
            mlp_hidden_units = [512, 256]
        self.mlp_hidden_units = mlp_hidden_units
        self.dropout_rate = dropout_rate

        # Shared hidden layers
        self.fc1 = Dense(self.mlp_hidden_units[0], activation='relu', name='shared_fc1')
        self.fc2 = Dense(self.mlp_hidden_units[1], activation='relu', name='shared_fc2')
        self.dropout = Dropout(self.dropout_rate, name='shared_dropout')

        # Head 1: Metabolite amplitudes
        self.amplitude_head = Dense(n_metabolites, activation='linear', name='amplitude_head')

        # Head 2: Global parameters (Gaussian line-broadening, zeroth-order phase, first-order phase) - DISABLED
        # self.global_head = Dense(3, activation='linear', name='global_head')

        # Head 3: Individual Lorentzian line-broadening - DISABLED
        # self.lorentzian_head = Dense(n_metabolites, activation='linear', name='lorentzian_head')

        # Head 4: Individual frequency shifts - DISABLED
        # self.frequency_head = Dense(n_metabolites, activation='linear', name='frequency_head')

        # Head 5: Baseline coefficients (2 spectra * n_baseline_coeffs) - DISABLED
        # self.baseline_head = Dense(2 * n_baseline_coeffs, activation='linear', name='baseline_head')

    def call(self, inputs):
        """Forward pass through MultiHeadMLP."""
        # Shared layers
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.dropout(x)

        # Individual heads - DISABLED unused heads to avoid gradient warnings
        # Only amplitudes are used in training, so we skip execution of other heads
        # to eliminate "gradients do not exist" warnings while preserving architecture
        amplitudes = self.amplitude_head(x)

        # Unused heads - disabled to avoid gradient warnings
        # global_params = self.global_head(x)
        # lorentzian_params = self.lorentzian_head(x)
        # frequency_params = self.frequency_head(x)
        # baseline_params = self.baseline_head(x)

        # Optimized: Use tf.fill for better performance than tf.zeros
        batch_size = tf.shape(amplitudes)[0]
        # Global parameters placeholder: [G, phi0, phi1]
        global_params = tf.fill([batch_size, 3], 0.0)
        lorentzian_params = tf.fill([batch_size, self.n_metabolites], 0.0)
        frequency_params = tf.fill([batch_size, self.n_metabolites], 0.0)
        baseline_params = tf.fill([batch_size, self.n_baseline_coeffs], 0.0)
        return {
            'amplitudes': amplitudes,
            'global_params': global_params,
            'lorentzian_params': lorentzian_params,
            'frequency_params': frequency_params,
            'baseline_params': baseline_params
        }

    def build(self, input_shape):
        """Build the layer with the given input shape.

        Parameters
        ----------
            input_shape (tuple): Input tensor shape
        """
        # Create a dummy input to build all layers
        dummy_input = tf.keras.Input(shape=input_shape[1:], name='dummy_input')

        # Forward pass to build all layers
        x = self.fc1(dummy_input)
        x = self.fc2(x)
        x = self.dropout(x)

        # Build amplitude head
        amplitudes = self.amplitude_head(x) # noqa: F841

        # Mark as built
        self.built = True
        self._build_input_shape = input_shape

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'n_metabolites': self.n_metabolites,
            'n_baseline_coeffs': self.n_baseline_coeffs,
            'mlp_hidden_units': self.mlp_hidden_units,
            'dropout_rate': self.dropout_rate
        })
        return config


@register_keras_serializable(package="mrsnet", name="QMRSModel")
class QMRSModel(Model):
    """QMRS model for MRS metabolite quantification.

    This implements the CNN-LSTM architecture with multi-headed MLP
    from the QMRS paper for metabolite parameter prediction.
    """

    def __init__(self, n_freqs, n_metabolites, n_baseline_coeffs=6,
                 initial_filters=32, inception_filters=32, lstm_units=128,
                 mlp_hidden_units=None, dropout_rate=0.5, name='QMRSModel', **kwargs):
        """Initialize QMRS model.

        Parameters
        ----------
            n_freqs (int): Number of frequency points in spectrum
            n_metabolites (int): Number of metabolites to quantify
            n_baseline_coeffs (int): Number of baseline coefficients
            initial_filters (int): Number of filters in initial conv layer
            inception_filters (int): Base number of filters for inception modules
            lstm_units (int): Number of LSTM units
            mlp_hidden_units (list): Hidden units in MLP layers
            dropout_rate (float): Dropout rate
            name (str, optional): Model name
        """
        super().__init__(name=name, **kwargs)
        self.n_freqs = n_freqs
        self.n_metabolites = n_metabolites
        self.n_baseline_coeffs = n_baseline_coeffs
        self.initial_filters = initial_filters
        self.inception_filters = inception_filters
        self.lstm_units = lstm_units

        # Set default MLP hidden units if not provided
        if mlp_hidden_units is None:
            mlp_hidden_units = [512, 256]
        self.mlp_hidden_units = mlp_hidden_units
        self.dropout_rate = dropout_rate

        # Input layer - variable channels handled at build time
        # Use a placeholder channel dimension; real value set in build()
        self.input_layer = None

        # Initial convolution and pooling - will be built with actual input shape
        self.conv1 = None  # Will be created in build() method
        self.pool1 = MaxPooling1D(pool_size=2, name='pool1')

        # Three inception modules with increasing filters (n, n+1, n+2)
        self.inception1 = InceptionModule(self.inception_filters, name='inception1')
        self.inception2 = InceptionModule(self.inception_filters + 1, name='inception2')
        self.inception3 = InceptionModule(self.inception_filters + 2, name='inception3')

        # Bidirectional LSTM
        self.lstm = Bidirectional(LSTM(self.lstm_units, return_sequences=False),
                                  merge_mode='concat', name='bidirectional_lstm')

        # Multi-headed MLP
        self.multi_head_mlp = MultiHeadMLP(n_metabolites, n_baseline_coeffs,
                                           mlp_hidden_units=self.mlp_hidden_units,
                                           dropout_rate=self.dropout_rate,
                                           name='multi_head_mlp')

        # Don't build automatically - will be built with correct input shape later

    def build(self, input_shape):
        """Build the model with the given input shape.

        Parameters
        ----------
            input_shape (tuple): Input tensor shape
        """
        # Create input placeholder if needed
        if self.input_layer is None:
            self.input_layer = Input(shape=input_shape, name='spectrum_input')

        # Create the Conv1D layer with the correct input channels
        if self.conv1 is None:
            self.conv1 = Conv1D(self.initial_filters, kernel_size=3, padding='same',
                                activation='relu', name='conv1')

        # Create a dummy input to build all layers
        # input_shape should be (freqs, channels), so we need to add batch dimension
        dummy_input = tf.keras.Input(shape=input_shape, name='dummy_input')

        # Forward pass to build all layers
        # Initial convolution and pooling
        x = self.conv1(dummy_input)
        x = self.pool1(x)

        # Inception modules
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)

        # Bidirectional LSTM
        x = self.lstm(x)

        # Multi-head MLP
        x = self.multi_head_mlp(x)

        # Mark as built
        self.built = True
        self._build_input_shape = input_shape

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
            'n_baseline_coeffs': self.n_baseline_coeffs,
            'initial_filters': self.initial_filters,
            'inception_filters': self.inception_filters,
            'lstm_units': self.lstm_units,
            'mlp_hidden_units': self.mlp_hidden_units,
            'dropout_rate': self.dropout_rate
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

    def __init__(self, model, metabolites, pulse_sequence, acquisitions, datatype, norm,
                 initial_filters=32, inception_filters=32, lstm_units=128,
                 mlp_hidden_units=None, dropout_rate=0.5, baseline_coeffs=6):
        """Initialize a QMRS model.

        Parameters
        ----------
            model (str): Model architecture identifier (e.g., 'qmrs_default')
            metabolites (list): List of metabolite names to predict
            pulse_sequence (str): Pulse sequence type (e.g., 'megapress')
            acquisitions (list): List of acquisition types (e.g., ['edit_off', 'difference'])
            datatype (list): List of data types (e.g., ['magnitude', 'phase'])
            norm (str): Normalization method (e.g., 'sum', 'max')
            initial_filters (int): Number of filters in initial conv layer
            inception_filters (int): Base number of filters for inception modules
            lstm_units (int): Number of LSTM units
            mlp_hidden_units (list): Hidden units in MLP layers
            dropout_rate (float): Dropout rate
            baseline_coeffs (int): Number of baseline coefficients (default: 6)
        """
        self.model = model
        self.metabolites = metabolites
        self.pulse_sequence = pulse_sequence
        self.acquisitions = acquisitions
        self.datatype = datatype
        self.norm = norm
        self.output = "concentrations"

        # Architecture parameters
        self.initial_filters = initial_filters
        self.inception_filters = inception_filters
        self.lstm_units = lstm_units
        self.mlp_hidden_units = mlp_hidden_units
        self.dropout_rate = dropout_rate
        self.baseline_coeffs = baseline_coeffs

        # Input spectra data (constant!)
        self.low_ppm = -1.0
        self.high_ppm = -4.5
        self.fft_samples = 2048

        self.train_dataset_name = None
        self.qmrs_arch = None

        # Validate context compatibility
        self._validate_context()

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
        """Validate that the context (acquisitions + datatypes) is compatible with QMRS.

        QMRS can handle arbitrary contexts but works best with:
        - Any number of acquisitions (processed as separate channels)
        - Any datatypes (magnitude, phase, real, imaginary)
        - Minimum 1 acquisition and 1 datatype
        """
        if not self.acquisitions:
            raise ValueError("QMRS requires at least one acquisition")
        if not self.datatype:
            raise ValueError("QMRS requires at least one datatype")

        # Check for supported datatypes
        supported_datatypes = {'magnitude', 'phase', 'real', 'imaginary'}
        unsupported = set(self.datatype) - supported_datatypes
        if unsupported:
            raise ValueError(f"QMRS does not support datatypes: {unsupported}. "
                           f"Supported datatypes: {supported_datatypes}")

        # Log context information
        print(f"QMRS context: {len(self.acquisitions)} acquisitions, {len(self.datatype)} datatypes")
        print(f"  Acquisitions: {self.acquisitions}")
        print(f"  Datatypes: {self.datatype}")

    def reset(self):
        """Reset the QMRS architecture and training dataset name.

        Clears the current model architecture and training dataset information.
        """
        self.qmrs_arch = None
        self.train_dataset_name = None

    def _parse_model_config(self, input_shape):
        """Parse model string to extract configuration parameters.

        Parameters
        ----------
            input_shape (tuple): Input tensor shape

        Returns
        -------
            tuple: (initial_filters, inception_filters, lstm_units, mlp_hidden_units, dropout_rate, baseline_coeffs)
        """
        vals = self.model.split("_")
        if vals[0] != 'qmrs':
            raise RuntimeError(f"Unknown model {vals[0]}")

        # Default parameters (original paper architecture)
        default_initial_filters = 32
        default_inception_filters = 32
        default_lstm_units = 128
        default_mlp_hidden_units = [512, 256]
        default_dropout_rate = 0.5
        default_baseline_coeffs = 6

        if len(vals) == 1 or (len(vals) == 2 and vals[1] == 'default'):
            # qmrs or qmrs_default: use default parameters
            initial_filters = default_initial_filters
            inception_filters = default_inception_filters
            lstm_units = default_lstm_units
            mlp_hidden_units = default_mlp_hidden_units
            dropout_rate = default_dropout_rate
            baseline_coeffs = default_baseline_coeffs
        else:
            # Custom configuration: qmrs_[INITIAL_FILTERS]_[INCEPTION_FILTERS]_[LSTM_UNITS]_[MLP_UNITS]_[DROPOUT]_[BASELINE_COEFFS]
            initial_filters = int(vals[1]) if len(vals) > 1 else default_initial_filters
            inception_filters = int(vals[2]) if len(vals) > 2 else default_inception_filters
            lstm_units = int(vals[3]) if len(vals) > 3 else default_lstm_units

            # Parse MLP hidden units (comma-separated list)
            if len(vals) > 4 and vals[4]:
                mlp_hidden_units = [int(x) for x in vals[4].split(',')]
            else:
                mlp_hidden_units = default_mlp_hidden_units

            dropout_rate = float(vals[5]) if len(vals) > 5 else default_dropout_rate
            baseline_coeffs = int(vals[6]) if len(vals) > 6 else default_baseline_coeffs

        # Override with instance parameters if provided
        if self.initial_filters != 32:  # If not default
            initial_filters = self.initial_filters
        if self.inception_filters != 32:  # If not default
            inception_filters = self.inception_filters
        if self.lstm_units != 128:  # If not default
            lstm_units = self.lstm_units
        if self.mlp_hidden_units is not None:  # If provided
            mlp_hidden_units = self.mlp_hidden_units
        if self.dropout_rate != 0.5:  # If not default
            dropout_rate = self.dropout_rate
        if self.baseline_coeffs != 6:  # If not default
            baseline_coeffs = self.baseline_coeffs

        return initial_filters, inception_filters, lstm_units, mlp_hidden_units, dropout_rate, baseline_coeffs

    def _create_training_model(self, input_shape, output_shape):
        """Create a training model that wraps QMRSModel for Keras training.

        Parameters
        ----------
            input_shape (tuple): Input tensor shape
            output_shape (tuple): Output tensor shape
        """
        # Parse model configuration
        initial_filters, inception_filters, lstm_units, mlp_hidden_units, dropout_rate, baseline_coeffs = self._parse_model_config(input_shape)

        # Get framework-determined parameters
        n_freqs = input_shape[-2]
        n_metabolites = len(self.metabolites)
        n_baseline_coeffs = baseline_coeffs  # Use configurable baseline coefficients

        # Create the QMRS model with parsed parameters
        self.qmrs_arch = QMRSModel(
            n_freqs=n_freqs,
            n_metabolites=n_metabolites,
            n_baseline_coeffs=n_baseline_coeffs,
            initial_filters=initial_filters,
            inception_filters=inception_filters,
            lstm_units=lstm_units,
            mlp_hidden_units=mlp_hidden_units,
            dropout_rate=dropout_rate,
            name=self.model
        )

        # Rebuild the model with the actual input shape
        self.qmrs_arch.build(input_shape)

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

        # Rebuild the model with the actual input shape
        self.qmrs_arch.build(input_shape)

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

        # Deterministic ops and AMP policy (auto) per config
        enable_deterministic_ops_if_configured()
        set_auto_mixed_precision_policy_if_enabled(verbose)

        if len(devices) > 1:
            # Multi-GPU training
            dev_multiplier = len(devices)
            mirrored_strategy = tf.distribute.MirroredStrategy(devices=devices)
            with mirrored_strategy.scope():
                self._create_training_model(d_inp.shape[1:], d_out.shape[1:])
                optimiser = keras.optimizers.Adam(learning_rate=learning_rate * dev_multiplier,
                                                  beta_1=Cfg.val['beta1'],
                                                  beta_2=Cfg.val['beta2'],
                                                  epsilon=Cfg.val['epsilon'],
                                                  clipnorm=(Cfg.val.get('optimizer_clipnorm', 0.0) or None),
                                                  clipvalue=(Cfg.val.get('optimizer_clipvalue', 0.0) or None))
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
                                              epsilon=Cfg.val['epsilon'],
                                              clipnorm=(Cfg.val.get('optimizer_clipnorm', 0.0) or None),
                                              clipvalue=(Cfg.val.get('optimizer_clipvalue', 0.0) or None))
            self.training_model.compile(loss='mse',
                                        optimizer=optimiser,
                                        metrics=['mae'])

        # Calculate FLOPs
        try:
            self.flops = calculate_flops(self.training_model, d_inp.shape[1:])
        except Exception as e:
            if verbose > 0:
                print(f"Error calculating FLOPs: {e}")
            self.flops = None

        # Create comprehensive architecture plots
        self._create_architecture_plots(folder, image_dpi, verbose)

        if verbose > 0:
            self.training_model.summary()

        timer = TimeHistory(epochs)
        # Prefer quantification MAE when available; fallback to loss
        if validation_data is not None:
            monitor_metric = f"val_{Cfg.val.get('monitor_metric_quant','mae')}"
        else:
            monitor_metric = Cfg.val.get('monitor_metric_quant','mae') if 'mae' in self.training_model.metrics_names else 'loss'
        callbacks = [
            keras.callbacks.EarlyStopping(monitor=monitor_metric,
                                          min_delta=1e-8,
                                          patience=Cfg.val.get('early_stopping_patience', 25),
                                          mode='min',
                                          verbose=(verbose > 0)*2,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor=monitor_metric,
                                              factor=0.5,
                                              patience=10,
                                              min_lr=Cfg.val.get('reduce_lr_min_lr', 1e-7),
                                              mode='min',
                                              verbose=(verbose > 0)*2),
            LrHistory(),
            timer
        ]

        # Dataset options
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        if Cfg.val.get('cache_datasets', False):
            train_data = train_data.cache()
            if validation_data is not None:
                validation_data = validation_data.cache()
        shuffle_buffer = max(1024, batch_size * dev_multiplier * 8)
        shuffle_seed = getattr(self, 'shuffle_seed', getattr(self, 'seed', None))
        train_data = (train_data
                        .shuffle(shuffle_buffer, seed=shuffle_seed, reshuffle_each_iteration=True)
                        .batch(batch_size * dev_multiplier)
                        .with_options(options)
                        .prefetch(tf.data.AUTOTUNE))
        if validation_data is not None:
            validation_data = (validation_data
                                 .batch(batch_size * dev_multiplier)
                                 .with_options(options)
                                 .prefetch(tf.data.AUTOTUNE))

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
        # For MEGA-PRESS: edit_off (channel 0) and difference (channel 1)

        batch_size = tf.shape(spectra)[0]
        n_channels = tf.shape(spectra)[1]  # acquisitions*datatypes
        n_freqs = tf.shape(spectra)[2]

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
        bsz = Cfg.val.get('predict_batch_size', 32)
        data = tf.data.Dataset.from_tensor_slices(d_inp).batch(bsz).with_options(options).prefetch(tf.data.AUTOTUNE)

        # Get predictions
        predictions = self.qmrs_arch.predict(data, verbose=(verbose>0)*2)

        # Extract metabolite amplitudes from the multi-head output
        # Keras predict returns numpy arrays; if dict-like is returned by a tf model, adapt gracefully
        try:
            if isinstance(predictions, dict):
                concentrations = predictions['amplitudes']
            else:
                # If model outputs a tuple/list/dict, attempt key access via attribute
                concentrations = predictions
        except Exception:
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

        # Calculate model metrics from summary (use the model we saved)
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.qmrs_arch.summary()
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
            flops = None

        with open(os.path.join(folder, "mrsnet.json"), 'w') as f:
            print(json.dumps({
                'model': self.model,
                'metabolites': self.metabolites,
                'pulse_sequence': self.pulse_sequence,
                'acquisitions': self.acquisitions,
                'datatype': self.datatype,
                'norm': self.norm,
                'train_dataset_name': self.train_dataset_name,
                'seed': getattr(self, 'seed', None),
                'shuffle_seed': getattr(self, 'shuffle_seed', None),
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

    def _create_architecture_plots(self, folder, image_dpi, verbose):
        """Create comprehensive architecture plots for QMRS."""
        try:
            # 1. Main training model architecture
            for dpi in image_dpi:
                try:
                    plot_model(self.training_model,
                               to_file=os.path.join(folder, f'qmrs_training_architecture@{dpi}.png'),
                               show_shapes=True,
                               show_dtype=True,
                               show_layer_names=True,
                               rankdir='TB',
                               expand_nested=True,
                               dpi=dpi)
                except Exception as e:
                    try:
                        plot_model(self.training_model,
                                   to_file=os.path.join(folder, f'qmrs_training_architecture@{dpi}.svg'),
                                   show_shapes=True,
                                   show_dtype=True,
                                   show_layer_names=True,
                                   rankdir='TB',
                                   expand_nested=True,
                                   dpi=dpi)
                        if verbose > 0:
                            print(f"# WARNING: QMRS training architecture PNG failed; wrote SVG instead: {e}")
                    except Exception as e2:
                        if verbose > 0:
                            print(f"# WARNING: QMRS training architecture plot failed: {e2}")

            # 2. Individual QMRS architecture (core model) - Enhanced
            self._plot_enhanced_core_architecture(folder, image_dpi, verbose)

            # 3. Create module architecture plots
            self._create_module_plots(folder, image_dpi, verbose)

        except Exception as e:
            if verbose > 0:
                print(f"# WARNING: Architecture plotting failed: {e}")

    def _plot_enhanced_core_architecture(self, folder, image_dpi, verbose):
        """Create an enhanced plot of the QMRS core architecture showing all layers."""
        try:
            # Create a detailed model that shows the QMRS architecture step by step
            from tensorflow.keras.layers import LSTM, Bidirectional, Conv1D, Input, MaxPooling1D
            from tensorflow.keras.models import Model

            input_layer = Input(shape=(2048, 2), name='qmrs_input')  # 2 channels (edit_off, difference)

            # CNN backbone: initial conv + max pooling
            conv1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', name='conv1')(input_layer)
            pool1 = MaxPooling1D(pool_size=2, name='pool1')(conv1)

            # Inception modules (n, n+1, n+2)
            inception1 = InceptionModule(n_filters=32, name='inception1')(pool1)
            inception2 = InceptionModule(n_filters=33, name='inception2')(inception1)
            inception3 = InceptionModule(n_filters=34, name='inception3')(inception2)

            # LSTM layer
            lstm = Bidirectional(LSTM(units=128, return_sequences=False, name='lstm'), merge_mode='concat', name='bidirectional_lstm')(inception3)

            # Multi-head MLP
            multihead_mlp = MultiHeadMLP(n_metabolites=len(self.metabolites), n_baseline_coeffs=6, name='multi_head_mlp')(lstm)

            # Create the detailed model
            detailed_model = Model(inputs=input_layer, outputs=multihead_mlp, name='QMRS_Detailed')

            for dpi in image_dpi:
                try:
                    plot_model(detailed_model,
                               to_file=os.path.join(folder, f'qmrs_detailed_architecture@{dpi}.png'),
                               show_shapes=True,
                               show_dtype=True,
                               show_layer_names=True,
                               rankdir='TB',
                               expand_nested=True,
                               dpi=dpi)
                except Exception as e:
                    try:
                        plot_model(detailed_model,
                                   to_file=os.path.join(folder, f'qmrs_detailed_architecture@{dpi}.svg'),
                                   show_shapes=True,
                                   show_dtype=True,
                                   show_layer_names=True,
                                   rankdir='TB',
                                   expand_nested=True,
                                   dpi=dpi)
                        if verbose > 0:
                            print(f"# WARNING: QMRS detailed architecture PNG failed; wrote SVG instead: {e}")
                    except Exception as e2:
                        if verbose > 0:
                            print(f"# WARNING: QMRS detailed architecture plot failed: {e2}")

        except Exception as e:
            if verbose > 0:
                print(f"# WARNING: QMRS detailed architecture creation failed: {e}")

    def _create_module_plots(self, folder, image_dpi, verbose):
        """Create visual plots of QMRS modules."""
        try:
            # Create InceptionModule plot
            self._plot_inception_module(folder, image_dpi, verbose)

            # Create MultiHeadMLP plot
            self._plot_multihead_mlp(folder, image_dpi, verbose)

        except Exception as e:
            if verbose > 0:
                print(f"# WARNING: Module plotting failed: {e}")

    def _plot_inception_module(self, folder, image_dpi, verbose):
        """Create a visual plot of InceptionModule."""
        try:
            # Create a simple model to demonstrate InceptionModule
            from tensorflow.keras.layers import Input
            input_layer = Input(shape=(2048, 1), name='inception_input')
            inception = InceptionModule(n_filters=32, name='block_inception')
            output = inception(input_layer)

            from tensorflow.keras.models import Model
            demo_model = Model(inputs=input_layer, outputs=output, name='InceptionModule_Block')

            for dpi in image_dpi:
                try:
                    plot_model(demo_model,
                               to_file=os.path.join(folder, f'qmrs_inception_module@{dpi}.png'),
                               show_shapes=True,
                               show_dtype=True,
                               show_layer_names=True,
                               rankdir='TB',
                               expand_nested=True,
                               dpi=dpi)
                except Exception as e:
                    try:
                        plot_model(demo_model,
                                   to_file=os.path.join(folder, f'qmrs_inception_module@{dpi}.svg'),
                                   show_shapes=True,
                                   show_dtype=True,
                                   show_layer_names=True,
                                   rankdir='TB',
                                   expand_nested=True,
                                   dpi=dpi)
                        if verbose > 0:
                            print(f"# WARNING: InceptionModule PNG failed; wrote SVG instead: {e}")
                    except Exception as e2:
                        if verbose > 0:
                            print(f"# WARNING: InceptionModule plot failed: {e2}")

        except Exception as e:
            if verbose > 0:
                print(f"# WARNING: InceptionModule demo model creation failed: {e}")

    def _plot_multihead_mlp(self, folder, image_dpi, verbose):
        """Create a visual plot of MultiHeadMLP."""
        try:
            # Create a simple model to demonstrate MultiHeadMLP
            from tensorflow.keras.layers import Input
            input_layer = Input(shape=(128,), name='multihead_input')  # LSTM output features
            multihead_mlp = MultiHeadMLP(n_metabolites=5, n_baseline_coeffs=6, name='block_multihead_mlp')
            outputs = multihead_mlp(input_layer)

            from tensorflow.keras.models import Model
            demo_model = Model(inputs=input_layer, outputs=outputs, name='MultiHeadMLP_Block')

            for dpi in image_dpi:
                try:
                    plot_model(demo_model,
                               to_file=os.path.join(folder, f'qmrs_multihead_mlp@{dpi}.png'),
                               show_shapes=True,
                               show_dtype=True,
                               show_layer_names=True,
                               rankdir='TB',
                               expand_nested=True,
                               dpi=dpi)
                except Exception as e:
                    try:
                        plot_model(demo_model,
                                   to_file=os.path.join(folder, f'qmrs_multihead_mlp@{dpi}.svg'),
                                   show_shapes=True,
                                   show_dtype=True,
                                   show_layer_names=True,
                                   rankdir='TB',
                                   expand_nested=True,
                                   dpi=dpi)
                        if verbose > 0:
                            print(f"# WARNING: MultiHeadMLP PNG failed; wrote SVG instead: {e}")
                    except Exception as e2:
                        if verbose > 0:
                            print(f"# WARNING: MultiHeadMLP plot failed: {e2}")

        except Exception as e:
            if verbose > 0:
                print(f"# WARNING: MultiHeadMLP demo model creation failed: {e}")

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
