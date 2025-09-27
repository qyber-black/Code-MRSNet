# mrsnet/encdec.py - MRSNet - EncDec model
#
# SPDX-FileCopyrightText: Copyright (C) 2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""EncDec (Encoder-Decoder) model for MRSNet.

This module implements the EncDec architecture from the paper:
"Quantification of Spatially Localized Magnetic Resonance Spectroscopy by a Novel Deep Learning Approach without Spectral Fitting"
by Zhang, Y., Shen, J., Mag Reson Med, 2023, May 15;90(4):1282-1296.
doi: 10.1002/mrm.29711
https://pmc.ncbi.nlm.nih.gov/articles/PMC10524908/

EncDec is an encoder-decoder style neural network with WaveNet blocks and GRU
for arbitrary MRS data processing and metabolite quantification.

MODIFICATIONS FOR MRSNET INTEGRATION:

1. CONTEXT ADAPTATION (NECESSARY):
   - Original: Designed for JPRESS data with multiple echoes (32 echoes x 512 frequency points)
   - MRSNet: Adapted for arbitrary acquisition contexts (acquisitions + datatypes)
   - Input format: (batch, acquisitions, freqs, datatypes) instead of (batch, echoes, freqs, channels)
   - Context validation: Ensures compatibility with supported datatypes (magnitude, phase, real, imaginary)
   - Flexible processing: Handles any number of acquisitions and datatypes

2. ARCHITECTURAL SIMPLIFICATIONS (MEMORY CONSTRAINTS):
   - Original: 128 filters in WaveNet blocks, 32 echoes
   - MRSNet: Reduced to 16 filters (default) and 2 acquisitions for GPU memory efficiency
   - Memory-optimized defaults: 'encdec_default' uses minimal parameters
   - Original parameters available: 'encdec_original' preserves paper architecture
   - Configurable: 'encdec_[ACQUISITIONS]_[FILTERS]' for custom configurations

3. IMPLEMENTATION DETAILS:
   - Custom WaveNetBlock: Dilated convolutions with residual connections (dilation rates: 1, 2, 4)
   - AttentionGRU: Bidirectional GRU with Bahdanau attention mechanism
   - ContextConverter: Converts MRSNet input format to EncDec-compatible format
   - Multi-head output: Concentrations (primary), FIDs, and phase parameters
   - Attention mechanism: Softmax over acquisitions dimension (corrected from original implementation)

4. TRAINING SIMPLIFICATIONS:
   - Output focus: Primary output is metabolite concentrations for MRSNet compatibility
   - FID reconstruction: Simplified for memory efficiency
   - Phase prediction: Basic phase shift and frequency offset parameters

5. COMPATIBILITY FEATURES:
   - Context validation: Explicit error messages for unsupported contexts
   - Flexible input handling: Supports both 3D and 4D input shapes
   - Memory management: Configurable parameters for different GPU capacities
   - Model string parsing: Dynamic configuration based on model identifier

This implementation maintains the core EncDec architecture while adapting it for
the MRSNet framework's arbitrary context handling and memory constraints.
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
    GRU,
    Add,
    Bidirectional,
    Concatenate,
    Conv1D,
    Dense,
    GlobalAveragePooling1D,
    Input,
    Multiply,
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


@register_keras_serializable(package="mrsnet", name="WaveNetBlock")
class WaveNetBlock(keras.layers.Layer):
    """WaveNet block for EncDec architecture.

    This implements a WaveNet-style dilated convolution block with
    residual connections and gated activation.
    """

    def __init__(self, filters, kernel_size=5, dilation_rate=1, name=None, **kwargs):
        """Initialize WaveNet block.

        Parameters
        ----------
            filters (int): Number of filters
            kernel_size (int): Kernel size for convolution
            dilation_rate (int): Dilation rate for convolution
            name (str, optional): Layer name
        """
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

        # Dilated convolution layers
        self.conv_filter = Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                                  padding='same', activation='tanh', name=f'{name}_filter')
        self.conv_gate = Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                               padding='same', activation='sigmoid', name=f'{name}_gate')

        # 1x1 convolution for residual connection
        self.conv_residual = Conv1D(filters, 1, padding='same', name=f'{name}_residual')

        # Add layer for residual connection
        self.add = Add(name=f'{name}_add')

    def call(self, inputs):
        """Forward pass through WaveNet block."""
        # Gated activation
        filter_out = self.conv_filter(inputs)
        gate_out = self.conv_gate(inputs)
        gated = Multiply()([filter_out, gate_out])

        # Residual connection
        residual = self.conv_residual(inputs)
        output = self.add([gated, residual])

        return output

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate
        })
        return config


@register_keras_serializable(package="mrsnet", name="ContextConverter")
class ContextConverter(keras.layers.Layer):
    """Convert spectra to context-aware format (acquisitions, freqs, channels)."""

    def __init__(self, n_acquisitions=2, n_datatypes=1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_acquisitions = n_acquisitions
        self.n_datatypes = n_datatypes

    def call(self, inputs):
        """Convert input spectra to context-aware format."""
        # Check if input is already in context format
        if len(inputs.shape) == 4:  # (batch, acquisitions, freqs, channels)
            return inputs

        # Input shape: (batch, acquisitions*datatypes, freqs)
        # Output shape: (batch, acquisitions, freqs, datatypes)

        batch_size = tf.shape(inputs)[0]
        n_channels = tf.shape(inputs)[1]  # acquisitions*datatypes
        n_freqs = tf.shape(inputs)[2]

        # Reshape to (batch, acquisitions, freqs, datatypes)
        spectra_context = tf.reshape(inputs, [batch_size, self.n_acquisitions, n_freqs, self.n_datatypes])

        return spectra_context

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_acquisitions': self.n_acquisitions,
            'n_datatypes': self.n_datatypes
        })
        return config


@register_keras_serializable(package="mrsnet", name="AttentionGRU")
class AttentionGRU(keras.layers.Layer):
    """Attention-based GRU for EncDec architecture.

    This implements a bidirectional GRU with attention mechanism
    that applies softmax attention across the echoes dimension
    to integrate individual echo representations.
    """

    def __init__(self, units, name=None, **kwargs):
        """Initialize AttentionGRU.

        Parameters
        ----------
            units (int): Number of GRU units
            name (str, optional): Layer name
        """
        super().__init__(name=name, **kwargs)
        self.units = units

        # Bidirectional GRU
        self.gru = Bidirectional(GRU(units, return_sequences=True),
                                merge_mode='concat', name=f'{name}_gru')

        # Attention mechanism
        self.attention_dense = Dense(1, activation='linear', name=f'{name}_attention')

        # Global average pooling
        self.global_pool = GlobalAveragePooling1D(name=f'{name}_pool')

    def call(self, inputs):
        """Forward pass through AttentionGRU."""
        # Apply bidirectional GRU
        gru_out = self.gru(inputs)

        # Apply attention across echoes dimension
        attention_scores = self.attention_dense(gru_out)  # (batch, echoes, 1)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)  # Softmax over echoes
        attended = Multiply()([gru_out, attention_weights])

        # Global pooling
        output = self.global_pool(attended)

        return output

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({'units': self.units})
        return config


@register_keras_serializable(package="mrsnet", name="EncDecModel")
class EncDecModel(Model):
    """EncDec model for MRS metabolite quantification.

    This implements the encoder-decoder architecture with WaveNet blocks
    and GRU for MEGA-PRESS data processing.
    """

    def __init__(self, n_acquisitions, n_freqs, n_metabolites, n_datatypes=1, filters=64, name='EncDecModel', **kwargs):
        """Initialize EncDec model.

        Parameters
        ----------
            n_acquisitions (int): Number of acquisitions in the data
            n_freqs (int): Number of frequency points per acquisition
            n_metabolites (int): Number of metabolites to quantify
            n_datatypes (int): Number of datatypes per acquisition
            filters (int): Number of filters for WaveNet blocks (default: 64)
            name (str, optional): Model name
        """
        super().__init__(name=name, **kwargs)
        self.n_acquisitions = n_acquisitions
        self.n_freqs = n_freqs
        self.n_metabolites = n_metabolites
        self.n_datatypes = n_datatypes
        self.filters = filters

        # Input layer - (batch, acquisitions, freqs, datatypes)
        self.input_layer = Input(shape=(n_acquisitions, n_freqs, n_datatypes), name='context_input')

        # Encoder: 3 WaveNet blocks with increasing dilation
        self.wavenet1 = WaveNetBlock(filters, dilation_rate=1, name='wavenet1')
        self.wavenet2 = WaveNetBlock(filters, dilation_rate=2, name='wavenet2')
        self.wavenet3 = WaveNetBlock(filters, dilation_rate=4, name='wavenet3')

        # Global pooling for each echo
        self.echo_pool = GlobalAveragePooling1D(name='echo_pool')

        # Attention GRU for echo integration
        self.attention_gru = AttentionGRU(filters, name='attention_gru')

        # Skip connections for decoder
        self.skip_concat = Concatenate(axis=-1, name='skip_concat')

        # Decoder: 2 WaveNet blocks
        self.decoder_wavenet1 = WaveNetBlock(filters, dilation_rate=1, name='decoder_wavenet1')
        self.decoder_wavenet2 = WaveNetBlock(filters, dilation_rate=1, name='decoder_wavenet2')

        # Output layers
        # Metabolite concentrations
        self.concentration_dense = Dense(n_metabolites, activation='linear', name='concentrations')

        # Individual metabolite FIDs (for all acquisitions) - original size
        # Calculate the actual decoder output size: batch_size * n_freqs * filters
        decoder_output_size = n_freqs * filters  # filters from WaveNet decoder
        # FID projection with enhanced reconstruction (tanh for better signal bounds)
        self.fid_projection = Dense(n_acquisitions * n_freqs * 2, activation='tanh', name='fids')

        # Spectral parameters
        self.phase_dense = Dense(2, activation='linear', name='phase')  # phase shift and frequency offset

    def build(self, input_shape):
        """Build the model with proper input shape."""
        super().build(input_shape)
        # The layers will be built automatically when called

    def call(self, inputs):
        """Forward pass through EncDec."""
        # Encoder: Process each acquisition through WaveNet blocks
        # Input is already in context format: (batch, acquisitions, freqs, datatypes)
        batch_size = tf.shape(inputs)[0]
        x = tf.reshape(inputs, [batch_size * self.n_acquisitions, self.n_freqs, self.n_datatypes])

        # WaveNet blocks
        x1 = self.wavenet1(x)
        x2 = self.wavenet2(x1)
        x3 = self.wavenet3(x2)

        # Pool each acquisition
        pooled = self.echo_pool(x3)  # (batch * acquisitions, filters)
        pooled = tf.reshape(pooled, [batch_size, self.n_acquisitions, self.filters])

        # Attention GRU integration
        unified_repr = self.attention_gru(pooled)  # (batch, 2*filters)

        # Decoder: Reconstruct FIDs
        # Expand unified representation for decoder
        decoder_input = tf.tile(tf.expand_dims(unified_repr, 1), [1, self.n_freqs, 1])

        # Skip connections
        skip_features = self.skip_concat([x1, x2, x3])
        skip_features = tf.reshape(skip_features, [batch_size, self.n_freqs, -1])

        # Decoder WaveNet blocks
        decoder_out1 = self.decoder_wavenet1(decoder_input)
        decoder_out2 = self.decoder_wavenet2(decoder_out1)

        # Outputs
        concentrations = self.concentration_dense(unified_repr)
        # Reshape decoder output properly: (batch_size, n_freqs, filters) -> (batch_size, n_freqs * filters)
        decoder_flat = tf.reshape(decoder_out2, [batch_size, self.n_freqs * self.filters])
        fids = self.fid_projection(decoder_flat)
        phase_params = self.phase_dense(unified_repr)

        return {
            'concentrations': concentrations,
            'fids': fids,
            'phase': phase_params
        }

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'n_acquisitions': self.n_acquisitions,
            'n_freqs': self.n_freqs,
            'n_metabolites': self.n_metabolites,
            'n_datatypes': self.n_datatypes,
            'filters': self.filters
        })
        return config


class EncDec:
    """EncDec for MRS metabolite quantification.

    This class implements the encoder-decoder architecture with WaveNet blocks
    and GRU for MEGA-PRESS data processing and metabolite concentration prediction.

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
        encdec_arch (keras.Model): The actual EncDec model
    """

    def __init__(self, model, metabolites, pulse_sequence, acquisitions, datatype, norm, filters=64):
        """Initialize an EncDec model.

        Parameters
        ----------
            model (str): Model architecture identifier (e.g., 'encdec_default')
            metabolites (list): List of metabolite names to predict
            pulse_sequence (str): Pulse sequence type (e.g., 'megapress')
            acquisitions (list): List of acquisition types (e.g., ['edit_off', 'edit_on', 'difference'])
            datatype (list): List of data types (e.g., ['magnitude', 'phase'])
            norm (str): Normalization method (e.g., 'sum', 'max')
            filters (int): Number of filters for WaveNet blocks (default: 64)
        """
        self.model = model
        self.metabolites = metabolites
        self.pulse_sequence = pulse_sequence
        self.acquisitions = acquisitions
        self.datatype = datatype
        self.norm = norm
        self.filters = filters
        self.output = "concentrations"

        # Input spectra data (constant!)
        self.low_ppm = -1.0
        self.high_ppm = -4.5
        self.fft_samples = 2048

        self.train_dataset_name = None
        self.encdec_arch = None

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
        """Validate that the context (acquisitions + datatypes) is compatible with EncDec.

        EncDec can handle arbitrary contexts but works best with:
        - Any number of acquisitions (processed as separate channels)
        - Any datatypes (magnitude, phase, real, imaginary)
        - Minimum 1 acquisition and 1 datatype
        """
        if not self.acquisitions:
            raise ValueError("EncDec requires at least one acquisition")
        if not self.datatype:
            raise ValueError("EncDec requires at least one datatype")

        # Check for supported datatypes
        supported_datatypes = {'magnitude', 'phase', 'real', 'imaginary'}
        unsupported = set(self.datatype) - supported_datatypes
        if unsupported:
            raise ValueError(f"EncDec does not support datatypes: {unsupported}. "
                           f"Supported datatypes: {supported_datatypes}")

        # Log context information
        print(f"EncDec context: {len(self.acquisitions)} acquisitions, {len(self.datatype)} datatypes")
        print(f"  Acquisitions: {self.acquisitions}")
        print(f"  Datatypes: {self.datatype}")

    def reset(self):
        """Reset the EncDec architecture and training dataset name.

        Clears the current model architecture and training dataset information.
        """
        self.encdec_arch = None
        self.train_dataset_name = None

    def _parse_model_config(self, input_shape):
        """Parse model configuration from model string.

        Parameters
        ----------
            input_shape (tuple): Input tensor shape

        Returns
        -------
            tuple: (n_acquisitions, filters)
        """
        vals = self.model.split("_")
        if vals[0] != 'encdec':
            raise RuntimeError(f"Unknown model {vals[0]}")

        # Parse model parameters
        if len(vals) == 1:
            # Default EncDec configuration - memory-friendly defaults
            n_acquisitions = 2   # MEGA-PRESS has edit_off and difference
            filters = 16  # Minimal filters for memory efficiency
        elif len(vals) == 2 and vals[1] == 'default':
            # encdec_default configuration - memory-friendly defaults
            n_acquisitions = 2   # MEGA-PRESS has edit_off and difference
            filters = 16  # Minimal filters for memory efficiency
        elif len(vals) == 2 and vals[1] == 'original':
            # encdec_original configuration - original paper parameters
            n_acquisitions = 2   # MEGA-PRESS has edit_off and difference
            filters = 128  # Original paper filters
        else:
            # Custom EncDec configuration: encdec_[ACQUISITIONS]_[FILTERS]
            n_acquisitions = int(vals[1]) if len(vals) > 1 else 2
            filters = int(vals[2]) if len(vals) > 2 else 16

        # Override with instance filters if provided
        if hasattr(self, 'filters') and self.filters != 16:
            filters = self.filters

        return n_acquisitions, filters

    def _create_training_model(self, input_shape, output_shape):
        """Create a training model that wraps EncDecModel for Keras training.

        Parameters
        ----------
            input_shape (tuple): Input tensor shape
            output_shape (tuple): Output tensor shape
        """
        # Parse model configuration
        n_acquisitions, filters = self._parse_model_config(input_shape)

        # Get framework-determined parameters
        n_freqs = input_shape[1]  # Frequency dimension (second dimension)
        n_metabolites = len(self.metabolites)
        n_datatypes = len(self.datatype)

        # Create EncDec model with concrete input shape
        concrete_input_shape = (None, n_acquisitions, n_freqs, n_datatypes)
        self.encdec_arch = EncDecModel(n_acquisitions, n_freqs, n_metabolites, n_datatypes=n_datatypes, filters=filters, name=self.model)

        # Create a training model that outputs only metabolite concentrations
        input_layer = Input(shape=input_shape, name='training_input')

        # Convert input to context format using custom layer
        context_converter = ContextConverter(n_acquisitions=n_acquisitions, n_datatypes=n_datatypes, name='context_converter')
        context_input = context_converter(input_layer)

        # Get EncDec outputs
        encdec_outputs = self.encdec_arch(context_input)

        # Extract only the concentrations for training (ignore FIDs and phase)
        if isinstance(encdec_outputs, dict):
            concentration_output = encdec_outputs['concentrations']
        else:
            concentration_output = encdec_outputs

        # Create training model
        self.training_model = Model(inputs=input_layer, outputs=concentration_output, name=f"{self.model}_training")

        return self.training_model

    def _construct(self, input_shape, output_shape):
        """Construct the EncDec architecture using functional API.

        Parameters
        ----------
            input_shape (tuple): Input tensor shape
            output_shape (tuple): Output tensor shape
        """
        # Parse model configuration
        n_acquisitions, filters = self._parse_model_config(input_shape)

        # Get framework-determined parameters
        n_freqs = input_shape[1]  # Frequency dimension (second dimension)
        n_metabolites = len(self.metabolites)

        # Create EncDec model
        self.encdec_arch = EncDecModel(n_acquisitions, n_freqs, n_metabolites, filters=filters, name=self.model)

    def train(self, d_data, v_data, epochs, batch_size,
              folder, verbose=0, image_dpi=[300], screen_dpi=96, train_dataset_name=""):
        """Train the EncDec model on provided data.

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
            print(f"# Train EncDec {self!s}")

        # Prepare data
        d_inp = tf.convert_to_tensor(d_data[0], dtype=tf.float32)
        d_out = tf.convert_to_tensor(d_data[1], dtype=tf.float32)
        d_inp = reshape_spectra_data(d_inp, add_channel_dim=False)

        # Convert to context format (acquisitions, freqs, datatypes)
        # Input shape: (batch, acquisitions*datatypes, freqs) -> (batch, acquisitions, freqs, datatypes)
        d_inp = self._convert_to_context_format(d_inp)

        train_data = tf.data.Dataset.from_tensor_slices((d_inp, d_out))

        if v_data is not None:
            v_inp = tf.convert_to_tensor(v_data[0], dtype=tf.float32)
            v_out = tf.convert_to_tensor(v_data[1], dtype=tf.float32)
            v_inp = reshape_spectra_data(v_inp, add_channel_dim=False)
            v_inp = self._convert_to_context_format(v_inp)
            validation_data = tf.data.Dataset.from_tensor_slices((v_inp, v_out))
        else:
            validation_data = None

        if verbose > 0:
            print("  Input:", d_inp.shape, "[batch, acquisitions, frequency, channels]")
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

    def _convert_to_context_format(self, spectra):
        """Convert spectra to context-aware format (acquisitions, freqs, datatypes).

        Parameters
        ----------
            spectra (tensor): Input spectra tensor

        Returns
        -------
            tensor: Context-aware format spectra tensor
        """
        # Handle both 3D and 4D input shapes
        if len(spectra.shape) == 4:
            # Input shape: (batch, acquisitions, datatypes, freqs)
            batch_size = tf.shape(spectra)[0]
            n_acquisitions = tf.shape(spectra)[1]
            n_datatypes = tf.shape(spectra)[2]
            n_freqs = tf.shape(spectra)[3]

            # Reshape to (batch, acquisitions, freqs, datatypes)
            spectra_context = tf.transpose(spectra, [0, 1, 3, 2])  # (batch, acquisitions, freqs, datatypes)

        else:
            # Input shape: (batch, acquisitions*datatypes, freqs)
            batch_size = tf.shape(spectra)[0]
            n_channels = tf.shape(spectra)[1]  # acquisitions*datatypes
            n_freqs = tf.shape(spectra)[2]

            # Reshape to (batch, acquisitions, freqs, datatypes)
            n_acquisitions = len(self.acquisitions)
            n_datatypes = len(self.datatype)
            spectra_context = tf.reshape(spectra, [batch_size, n_acquisitions, n_freqs, n_datatypes])

        return spectra_context

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
            d_inp = self._convert_to_context_format(d_inp)

        # Dataset options
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        data = tf.data.Dataset.from_tensor_slices(d_inp).batch(32).with_options(options)

        # Get predictions
        predictions = self.encdec_arch.predict(data, verbose=(verbose>0)*2)

        # Extract metabolite concentrations from the multi-head output
        if isinstance(predictions, dict):
            concentrations = predictions['concentrations']
        else:
            concentrations = predictions

        return np.array(concentrations, dtype=np.float64)

    def save(self, folder):
        """Save the trained EncDec model to disk.

        Parameters
        ----------
            folder (str): Directory to save the model
        """
        os.makedirs(folder, exist_ok=True)
        self.encdec_arch.save(os.path.join(folder, "model.keras"))

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
        """Load a trained EncDec model from disk.

        Parameters
        ----------
            path (str): Directory containing the saved model

        Returns
        -------
            EncDec: Loaded EncDec model instance
        """
        with open(os.path.join(path, "mrsnet.json")) as f:
            data = json.load(f)
        model = EncDec(data['model'], data['metabolites'], data['pulse_sequence'], data['acquisitions'],
                      data['datatype'], data['norm'])
        model.train_dataset_name = data['train_dataset_name']
        model.encdec_arch = load_model(os.path.join(path, "model.keras"),
                                      custom_objects={
                                          "EncDecModel": EncDecModel,
                                          "WaveNetBlock": WaveNetBlock,
                                          "AttentionGRU": AttentionGRU
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
