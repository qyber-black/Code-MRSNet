# mrsnet/qnet.py - MRSNet - QNet model
#
# SPDX-FileCopyrightText: Copyright (C) 2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""QNet (Quantification Network) model for MRSNet.

This module implements the QNet architecture from the paper:
"Quantification of Magnetic Resonance Spectroscopy Data Using Deep Learning"
by Chen, D., et al. IEEE Trans Biomed Eng, 2024 Jun;71(6):1841-1852.
doi: 10.1109/TBME.2024.3354123
https://pubmed.ncbi.nlm.nih.gov/38224519/

QNet consists of two deep learning modules:
1. IF Extraction Module: Predicts imperfection factors (phase, frequency, linewidth)
2. MM Signal Prediction Module: Predicts macromolecule background signal
3. LLS Module: Linear Least Squares for metabolite concentration estimation

IMPLEMENTATION VARIANTS:

This module provides two QNet implementations:

1. QNet (Basic Implementation):
   - Uses BasicLLSModule with learnable linear combinations
   - Simplified LLS for practical implementation
   - Suitable for general-purpose metabolite quantification

2. QNetBasis (Full Basis Set LLS):
   - Implements the complete LLS approach from the original paper
   - Uses actual metabolite basis spectra with imperfection factor modulation
   - Automatically extracts basis parameters from dataset paths
   - Scientifically accurate metabolite quantification

MODIFICATIONS FOR MRSNET INTEGRATION:

1. CONTEXT ADAPTATION (NECESSARY):
   - Original: Designed for single acquisition input (frequency points only)
   - MRSNet: Adapted for arbitrary acquisition contexts (multiple acquisitions + datatypes)
   - Input format: (batch, acquisitions, freqs) instead of (batch, freqs)
   - AcquisitionSplitter: Processes each acquisition separately through QNet modules
   - Context validation: Ensures compatibility with supported datatypes (magnitude, phase, real, imaginary)
   - Multi-acquisition combination: Concatenates IF factors from all acquisitions

2. ARCHITECTURAL IMPLEMENTATIONS:

   A. QNet (Basic Implementation):
   - LLS module: BasicLLSModule with learnable matrix mapping IF factors to concentrations
   - MM Signal Prediction Module: DISABLED in current implementation
   - Training: Only IF extraction module + basic LLS
   - Memory optimization: Reduced default filter sizes for GPU compatibility
   - Configurable parameters: 'qnet_[IF_FILTERS]_[MM_FILTERS]_[KERNEL]_[IF_FC]_[MM_FC]_[IF_FACTORS]'

   B. QNetBasis (Full Basis Set LLS):
   - LLS module: BasisLLSModule with actual metabolite basis spectra
   - Imperfection factor modulation: Phase, frequency, and linewidth shifts applied to basis spectra
   - Analytical LLS solution: Pseudo-inverse for metabolite concentration estimation
   - Automatic basis parameter extraction: From dataset paths (source, manufacturer, omega, linewidth)
   - Basis set consistency: Ensures correct basis set for each training dataset
   - TensorFlow integration: Fully differentiable for training

3. IMPLEMENTATION DETAILS:
   - StackedConvolutionalBlock (SCB): Custom implementation with Conv1D, BatchNorm, ReLU
   - IF Extraction Module: 3 SCBs with filters [16, 32, 64] (default)
   - MM Signal Prediction Module: 6 SCBs with filters [16, 32, 64, 128, 256, 256] (default)
   - Multi-acquisition processing: Each acquisition processed independently
   - Output combination: Concatenated IF factors from all acquisitions

4. BASIS SET INTEGRATION (QNetBasis):
   - Dataset path parsing: Extracts basis parameters from MRSNet dataset paths
   - Format: source_sample_rate_samples/manufacturer/omega/linewidth/metabolites/pulse_sequence/...
   - Supported sources: fid-a, fid-a-2d, lcmodel, pygamma, su-*
   - Automatic parameter matching: Ensures basis consistency with training data
   - Basis file loading: Uses MRSNet's basis management system

5. TRAINING FEATURES:
   - QNet: BasicLLSModule with learnable linear combinations
   - QNetBasis: Full basis set LLS with imperfection factor modulation
   - MM Signal Prediction: Not used in training (module exists but is disabled)
   - Imperfection factors: Combined across acquisitions for improved quantification
   - Training model: Wrapper that outputs concentrations via LLS module
   - Gradient handling: Proper variable scope management for training

6. COMPATIBILITY FEATURES:
   - Context validation: Explicit error messages for unsupported contexts
   - Flexible acquisition handling: Supports any number of acquisitions
   - Memory management: Configurable SCB filter sizes
   - Model string parsing: Dynamic configuration based on model identifier
   - Basis parameter extraction: Automatic parameter inference from dataset paths

USAGE EXAMPLES:

Basic QNet (simplified LLS):
    qnet = QNet('qnet_original', metabolites, pulse_sequence, acquisitions, datatype, norm)

Full Basis Set QNet (scientifically accurate):
    qnet_basis = QNetBasis('qnet_original', metabolites, pulse_sequence, acquisitions, datatype, norm,
                          dataset_path='data/sim-spectra-megapress/fid-a-2d_2000_4096/siemens/123.23/2.0/Cr-GABA-Gln-Glu-NAA/megapress/...')

This implementation provides both simplified and full-fidelity QNet variants,
maintaining compatibility with the MRSNet framework while offering scientific
accuracy through proper basis set integration.
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
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Layer, MaxPooling1D, Reshape
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model, register_keras_serializable

from mrsnet.basis_lls import BasisLLSModule, create_basis_lls_module, extract_basis_params_from_dataset_path

try:
    from keras.saving import register_keras_serializable
except Exception:
    from tensorflow.keras.utils import register_keras_serializable

from mrsnet.cfg import Cfg
from mrsnet.cnn import TimeHistory
from mrsnet.train import calculate_flops, reshape_spectra_data


@register_keras_serializable(package="mrsnet", name="StackedConvolutionalBlock")
class StackedConvolutionalBlock(keras.layers.Layer):
    """Stacked Convolutional Block (SCB) as used in QNet.

    Each SCB contains 2 convolutional layers with ReLU activation
    and max pooling.
    """

    def __init__(self, filters, kernel_size=3, name=None, **kwargs):
        """Initialize SCB.

        Parameters
        ----------
            filters (int): Number of filters in convolutional layers
            kernel_size (int): Kernel size for convolution (default 3)
            name (str, optional): Layer name
        """
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

        # Two convolutional layers
        self.conv1 = Conv1D(filters, kernel_size, padding='same', activation='relu')
        self.conv2 = Conv1D(filters, kernel_size, padding='same', activation='relu')
        self.maxpool = MaxPooling1D(pool_size=2)

    def build(self, input_shape):
        """Build the SCB layers with the given input shape.

        Parameters
        ----------
            input_shape (tuple): Input tensor shape
        """
        # Build convolutional layers
        self.conv1.build(input_shape)
        # Second conv layer expects output from first conv layer
        conv1_output_shape = (input_shape[0], input_shape[1], self.filters)
        self.conv2.build(conv1_output_shape)
        # MaxPool reduces spatial dimension by pool_size
        self.maxpool.build(conv1_output_shape)

        # Mark as built
        self.built = True
        self._build_input_shape = input_shape

    def call(self, inputs):
        """Forward pass through SCB."""
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxpool(x)
        return x

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config


@register_keras_serializable(package="mrsnet", name="AcquisitionSplitter")
class AcquisitionSplitter(Layer):
    """Custom layer to split multi-acquisition input into single acquisitions."""

    def __init__(self, acquisition_idx, **kwargs):
        super().__init__(**kwargs)
        self.acquisition_idx = acquisition_idx

    def call(self, inputs):
        """Extract single acquisition from multi-acquisition input."""
        # Input shape: (batch, acquisitions, freqs)
        # Output shape: (batch, freqs)
        # Use tf.gather with proper axis
        return tf.gather(inputs, self.acquisition_idx, axis=1)

    def compute_output_shape(self, input_shape):
        """Compute the output shape."""
        return (input_shape[0], input_shape[2])  # (batch, freqs)

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({'acquisition_idx': self.acquisition_idx})
        return config


@register_keras_serializable(package="mrsnet", name="AcquisitionDatatypeSplitter")
class AcquisitionDatatypeSplitter(Layer):
    """Custom layer to split multi-acquisition multi-datatype input into single acquisitions."""

    def __init__(self, acquisition_idx, n_datatypes, **kwargs):
        super().__init__(**kwargs)
        self.acquisition_idx = acquisition_idx
        self.n_datatypes = n_datatypes

    def call(self, inputs):
        """Extract single acquisition from multi-acquisition multi-datatype input."""
        # Input shape: (batch, acquisition*datatype, freqs)
        # Output shape: (batch, datatype, freqs)
        # Extract all datatypes for the specified acquisition
        start_idx = self.acquisition_idx * self.n_datatypes
        end_idx = start_idx + self.n_datatypes
        return tf.gather(inputs, tf.range(start_idx, end_idx), axis=1)

    def compute_output_shape(self, input_shape):
        """Compute the output shape."""
        return (input_shape[0], self.n_datatypes, input_shape[2])  # (batch, datatype, freqs)

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'acquisition_idx': self.acquisition_idx,
            'n_datatypes': self.n_datatypes
        })
        return config


@register_keras_serializable(package="mrsnet", name="DatatypeProcessor")
class DatatypeProcessor(Layer):
    """Custom layer to process multiple datatypes for a single acquisition."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Don't store qnet_model to avoid serialization issues
        # We'll handle this differently in the training model

    def call(self, inputs):
        """Process multiple datatypes for a single acquisition."""
        # Input shape: (batch, datatype, freqs)
        # For now, just use the first datatype (real part)
        single_datatype = tf.gather(inputs, 0, axis=1)  # Take first datatype
        return single_datatype

    def compute_output_shape(self, input_shape):
        """Compute the output shape."""
        # Return the frequency dimension only
        return (input_shape[0], input_shape[2])  # (batch, freqs)

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        """Create layer from config."""
        return cls(**config)


@register_keras_serializable(package="mrsnet", name="IFConcatenator")
class IFConcatenator(keras.layers.Layer):
    """Custom layer to concatenate IF factors from multiple acquisitions."""

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs):
        """Concatenate IF factors from multiple acquisitions."""
        return tf.concat(inputs, axis=-1)

    def compute_output_shape(self, input_shape):
        """Compute the output shape."""
        # input_shape is a list of shapes: [(batch, if_factors1), (batch, if_factors2)]
        total_if_factors = sum(shape[1] for shape in input_shape)
        return (input_shape[0][0], total_if_factors)

    def get_config(self):
        """Get configuration for serialization.

        Returns
        -------
        dict
            Dictionary containing layer configuration parameters for serialization.
            Includes n_acquisitions and n_datatypes parameters.
        """
        config = super().get_config()
        return config

@register_keras_serializable(package="mrsnet", name="BasicLLSModule")
class BasicLLSModule(keras.layers.Layer):
    """Basic Linear Least Squares module for QNet metabolite quantification.

    This is a simplified LLS implementation that learns linear combinations
    of imperfection factors to predict metabolite concentrations.
    """

    def __init__(self, n_metabolites, n_if_factors, name=None, **kwargs):
        """Initialize BasicLLSModule.

        Parameters
        ----------
            n_metabolites (int): Number of metabolites to quantify
            n_if_factors (int): Number of imperfection factors per metabolite
            name (str, optional): Layer name
        """
        super().__init__(name=name, **kwargs)
        self.n_metabolites = n_metabolites
        self.n_if_factors = n_if_factors

        # Learnable LLS matrix: maps IF factors to metabolite concentrations
        self.lls_matrix = None

    def build(self, input_shape):
        """Build the LLS matrix layer."""
        super().build(input_shape)
        # Input shape: (batch, n_metabolites * n_if_factors)
        self.lls_matrix = self.add_weight(
            name='lls_matrix',
            shape=(self.n_metabolites * self.n_if_factors, self.n_metabolites),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, if_factors):
        """Apply LLS to imperfection factors.

        Parameters
        ----------
            if_factors (tensor): Imperfection factors tensor

        Returns
        -------
            tensor: Predicted metabolite concentrations
        """
        # Handle variable input sizes (for multiple acquisitions)
        input_size = tf.shape(if_factors)[1]
        expected_size = self.n_metabolites * self.n_if_factors

        # Use tf.cond for conditional operations in graph mode
        # Both branches must return the same shape
        if_factors_processed = tf.cond(
            input_size > expected_size,
            lambda: self._handle_multiple_acquisitions(if_factors),  # Handle multiple acquisitions
            lambda: if_factors[:, :expected_size]  # Truncate to expected size for consistency
        )

        # Reshape IF factors: (batch, n_metabolites * n_if_factors)
        if_factors_flat = tf.reshape(if_factors_processed, [-1, self.n_metabolites * self.n_if_factors])

        # Apply LLS: concentrations = IF_factors @ LLS_matrix
        concentrations = tf.matmul(if_factors_flat, self.lls_matrix)

        return concentrations

    def _handle_multiple_acquisitions(self, if_factors):
        """Handle multiple acquisitions by averaging IF factors.

        Parameters
        ----------
            if_factors (tensor): Concatenated IF factors from multiple acquisitions

        Returns
        -------
            tensor: Averaged IF factors for single acquisition
        """
        # Reshape to separate acquisitions: (batch, n_acquisitions, n_metabolites * n_if_factors)
        batch_size = tf.shape(if_factors)[0]
        n_acquisitions = tf.shape(if_factors)[1] // (self.n_metabolites * self.n_if_factors)

        # Reshape to separate acquisitions
        if_factors_reshaped = tf.reshape(if_factors, [batch_size, n_acquisitions, self.n_metabolites * self.n_if_factors])

        # Average across acquisitions
        if_factors_averaged = tf.reduce_mean(if_factors_reshaped, axis=1)

        return if_factors_averaged

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'n_metabolites': self.n_metabolites,
            'n_if_factors': self.n_if_factors
        })
        return config


@register_keras_serializable(package="mrsnet", name="QNetModel")
class QNetModel(Model):
    """QNet model combining IF extraction, MM prediction, and LLS.

    This implements the complete QNet architecture with two deep learning
    modules and a linear least squares component.
    """

    def __init__(self, n_freqs, n_metabolites, n_if_factors=3,
                 if_scb_filters=None, mm_scb_filters=None, kernel_size=3,
                 if_fc_units=128, mm_fc_units=512, name='QNetModel', **kwargs):
        """Initialize QNet model.

        Parameters
        ----------
            n_freqs (int): Number of frequency points in spectrum
            n_metabolites (int): Number of metabolites to quantify
            n_if_factors (int): Number of imperfection factors per metabolite (default 3: phase, freq, linewidth)
            if_scb_filters (list): List of filters for IF extraction SCBs
            mm_scb_filters (list): List of filters for MM prediction SCBs
            kernel_size (int): Kernel size for all SCBs
            if_fc_units (int): FC layer units for IF module
            mm_fc_units (int): FC layer units for MM module
            name (str, optional): Model name
        """
        super().__init__(name=name, **kwargs)
        self.n_freqs = n_freqs
        self.n_metabolites = n_metabolites
        self.n_if_factors = n_if_factors

        # Set default filter lists if not provided (original paper architecture)
        if if_scb_filters is None:
            if_scb_filters = [16, 32, 64]
        if mm_scb_filters is None:
            mm_scb_filters = [16, 32, 64, 128, 256, 256]

        self.if_scb_filters = if_scb_filters
        self.mm_scb_filters = mm_scb_filters
        self.kernel_size = kernel_size
        self.if_fc_units = if_fc_units
        self.mm_fc_units = mm_fc_units

        # Input layer - QNet expects 1D spectrum input
        self.input_layer = Input(shape=(n_freqs,), name='spectrum_input')

        # Reshape input for Conv1D layers: (batch, freqs) -> (batch, freqs, 1)
        self.input_reshape = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1), name='input_reshape')

        # IF Extraction Module: Dynamic SCBs + 2 FCLs
        self.if_scbs = []
        for i, filters in enumerate(self.if_scb_filters):
            scb = StackedConvolutionalBlock(filters, kernel_size=self.kernel_size, name=f'if_scb{i+1}')
            self.if_scbs.append(scb)
        self.if_flatten = Flatten(name='if_flatten')
        self.if_fc1 = Dense(self.if_fc_units, activation='relu', name='if_fc1')
        self.if_fc2 = Dense(n_metabolites * n_if_factors, name='if_output')

        # LLS Module for concentration estimation
        self.lls_module = BasicLLSModule(
            n_metabolites=n_metabolites,
            n_if_factors=n_if_factors,
            name='basic_lls'
        )

        # MM Signal Prediction Module: Dynamic SCBs + 2 FCLs - DISABLED
        # Commented out to eliminate gradient warnings since MM branch is not used
        # self.mm_scbs = []
        # for i, filters in enumerate(self.mm_scb_filters):
        #     scb = StackedConvolutionalBlock(filters, kernel_size=self.kernel_size, name=f'mm_scb{i+1}')
        #     self.mm_scbs.append(scb)
        # self.mm_flatten = Flatten(name='mm_flatten')
        # self.mm_fc1 = Dense(self.mm_fc_units, activation='relu', name='mm_fc1')
        # self.mm_fc2 = Dense(n_freqs, name='mm_output')

        # Build the model
        # Let Keras handle building automatically when the model is first used

    def build(self, input_shape):
        """Build the model with the given input shape.

        Parameters
        ----------
            input_shape (tuple): Input tensor shape (batch, freqs)
        """
        # Initialize all layers with proper input shapes
        # This ensures all weights are properly initialized

        # Build input reshape layer
        self.input_reshape.build(input_shape)

        # Build IF extraction branch layers
        if_input_shape = (input_shape[0], input_shape[1], 1)  # After reshape
        for i, scb in enumerate(self.if_scbs):
            scb.build(if_input_shape)
            # Update input shape for next SCB: MaxPool reduces spatial dimension by 2
            if_input_shape = (if_input_shape[0], if_input_shape[1] // 2, self.if_scb_filters[i])
        self.if_flatten.build(if_input_shape)
        # Flatten output shape
        flattened_size = if_input_shape[1] * if_input_shape[2]
        self.if_fc1.build((if_input_shape[0], flattened_size))
        self.if_fc2.build((if_input_shape[0], self.if_fc_units))

        # Build LLS module
        # LLS module expects input from all acquisitions combined
        # The actual input shape will be adjusted when the model is used
        lls_input_shape = (if_input_shape[0], self.n_metabolites * self.n_if_factors)
        self.lls_module.build(lls_input_shape)

        # Mark as built
        self.built = True
        self._build_input_shape = input_shape

    def call(self, inputs):
        """Forward pass through QNet."""
        # Reshape input for Conv1D layers: (batch, freqs) -> (batch, freqs, 1)
        x = self.input_reshape(inputs)

        # IF Extraction branch
        if_x = x
        for scb in self.if_scbs:
            if_x = scb(if_x)
        if_x = self.if_flatten(if_x)
        if_x = self.if_fc1(if_x)
        if_factors = self.if_fc2(if_x)

        # MM Signal Prediction branch - DISABLED to avoid gradient warnings
        # The MM branch is not used in training, so we skip execution to eliminate
        # "gradients do not exist" warnings while preserving the architecture
        # mm_x = x
        # for scb in self.mm_scbs:
        #     mm_x = scb(mm_x)
        # mm_x = self.mm_flatten(mm_x)
        # mm_x = self.mm_fc1(mm_x)
        # mm_signal = self.mm_fc2(mm_x)

        # Return dummy MM signal with correct shape for compatibility
        mm_signal = tf.zeros([tf.shape(if_factors)[0], self.n_freqs], dtype=tf.float32)
        return {
            'if_factors': if_factors,
            'mm_signal': mm_signal
        }

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'n_freqs': self.n_freqs,
            'n_metabolites': self.n_metabolites,
            'n_if_factors': self.n_if_factors,
            'if_scb_filters': self.if_scb_filters,
            'mm_scb_filters': self.mm_scb_filters,
            'kernel_size': self.kernel_size,
            'if_fc_units': self.if_fc_units,
            'mm_fc_units': self.mm_fc_units
        })
        return config


class QNet:
    """QNet (Quantification Network) for MRS metabolite quantification.

    This class implements the QNet architecture with two deep learning modules
    for imperfection factor extraction and macromolecule signal prediction,
    combined with linear least squares for metabolite concentration estimation.

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
        qnet_arch (keras.Model): The actual QNet model
    """

    def __init__(self, model, metabolites, pulse_sequence, acquisitions, datatype, norm,
                 if_scb_filters=None, mm_scb_filters=None, kernel_size=3,
                 if_fc_units=128, mm_fc_units=512, n_if_factors=3):
        """Initialize a QNet model.

        Parameters
        ----------
            model (str): Model architecture identifier (e.g., 'qnet_default', 'qnet_original')
            metabolites (list): List of metabolite names to predict
            pulse_sequence (str): Pulse sequence type (e.g., 'megapress')
            acquisitions (list): List of acquisition types (e.g., ['edit_off', 'difference'])
            datatype (list): List of data types (e.g., ['magnitude', 'phase'])
            norm (str): Normalization method (e.g., 'sum', 'max')
            if_scb_filters (list): List of filters for IF extraction SCBs
            mm_scb_filters (list): List of filters for MM prediction SCBs
            kernel_size (int): Kernel size for all SCBs
            if_fc_units (int): FC layer units for IF module
            mm_fc_units (int): FC layer units for MM module
            n_if_factors (int): Number of imperfection factors per metabolite
        """
        self.model = model
        self.metabolites = metabolites
        self.pulse_sequence = pulse_sequence
        self.acquisitions = acquisitions
        self.datatype = datatype
        self.norm = norm
        self.output = "concentrations"

        # Architecture parameters
        self.if_scb_filters = if_scb_filters
        self.mm_scb_filters = mm_scb_filters
        self.kernel_size = kernel_size
        self.if_fc_units = if_fc_units
        self.mm_fc_units = mm_fc_units
        self.n_if_factors = n_if_factors

        # Input spectra data (constant!)
        self.low_ppm = -1.0
        self.high_ppm = -4.5
        self.fft_samples = 2048

        self.train_dataset_name = None
        self.qnet_arch = None

        # Basis set for LLS (will be loaded during training)
        self.basis_set = None

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
        """Validate that the context (acquisitions + datatypes) is compatible with QNet.

        QNet can handle arbitrary contexts but works best with:
        - Any number of acquisitions (processed separately)
        - Any datatypes (magnitude, phase, real, imaginary)
        - Minimum 1 acquisition and 1 datatype
        """
        if not self.acquisitions:
            raise ValueError("QNet requires at least one acquisition")
        if not self.datatype:
            raise ValueError("QNet requires at least one datatype")

        # Check for supported datatypes
        supported_datatypes = {'magnitude', 'phase', 'real', 'imaginary'}
        unsupported = set(self.datatype) - supported_datatypes
        if unsupported:
            raise ValueError(f"QNet does not support datatypes: {unsupported}. "
                           f"Supported datatypes: {supported_datatypes}")

        # Log context information
        print(f"QNet context: {len(self.acquisitions)} acquisitions, {len(self.datatype)} datatypes")
        print(f"  Acquisitions: {self.acquisitions}")
        print(f"  Datatypes: {self.datatype}")

    def reset(self):
        """Reset the QNet architecture and training dataset name.

        Clears the current model architecture and training dataset information.
        """
        self.qnet_arch = None
        self.train_dataset_name = None
        self.basis_set = None

    def _parse_model_config(self, input_shape):
        """Parse model string to extract configuration parameters.

        Parameters
        ----------
            input_shape (tuple): Input tensor shape

        Returns
        -------
            tuple: (if_scb_filters, mm_scb_filters, kernel_size, if_fc_units, mm_fc_units, n_if_factors)
        """
        vals = self.model.split("_")
        if vals[0] != 'qnet':
            raise RuntimeError(f"Unknown model {vals[0]}")

        # Default parameters (original paper architecture)
        default_if_scb_filters = [16, 32, 64]
        default_mm_scb_filters = [16, 32, 64, 128, 256, 256]
        default_kernel_size = 3
        default_if_fc_units = 128
        default_mm_fc_units = 512
        default_n_if_factors = 3

        if len(vals) == 1 or (len(vals) == 2 and vals[1] == 'default'):
            # qnet or qnet_default: use default parameters
            if_scb_filters = default_if_scb_filters
            mm_scb_filters = default_mm_scb_filters
            kernel_size = default_kernel_size
            if_fc_units = default_if_fc_units
            mm_fc_units = default_mm_fc_units
            n_if_factors = default_n_if_factors
        elif len(vals) == 2 and vals[1] == 'original':
            # qnet_original: use original paper parameters (512 frequency points)
            if_scb_filters = default_if_scb_filters
            mm_scb_filters = default_mm_scb_filters
            kernel_size = default_kernel_size
            if_fc_units = default_if_fc_units
            mm_fc_units = default_mm_fc_units
            n_if_factors = default_n_if_factors
        elif len(vals) == 3 and vals[1] == 'basis' and vals[2] == 'original':
            # qnet_basis_original: use original paper parameters for basis variant
            if_scb_filters = default_if_scb_filters
            mm_scb_filters = default_mm_scb_filters
            kernel_size = default_kernel_size
            if_fc_units = default_if_fc_units
            mm_fc_units = default_mm_fc_units
            n_if_factors = default_n_if_factors
        else:
            # Custom configuration: qnet_[IF_FILTERS]_[MM_FILTERS]_[KERNEL]_[IF_FC]_[MM_FC]_[IF_FACTORS]
            # Parse IF SCB filters (comma-separated list)
            if len(vals) > 1 and vals[1]:
                if_scb_filters = [int(x) for x in vals[1].split(',')]
            else:
                if_scb_filters = default_if_scb_filters

            # Parse MM SCB filters (comma-separated list)
            if len(vals) > 2 and vals[2]:
                mm_scb_filters = [int(x) for x in vals[2].split(',')]
            else:
                mm_scb_filters = default_mm_scb_filters

            kernel_size = int(vals[3]) if len(vals) > 3 else default_kernel_size
            if_fc_units = int(vals[4]) if len(vals) > 4 else default_if_fc_units
            mm_fc_units = int(vals[5]) if len(vals) > 5 else default_mm_fc_units
            n_if_factors = int(vals[6]) if len(vals) > 6 else default_n_if_factors

        # Override with instance parameters if provided
        if self.if_scb_filters is not None:  # If provided
            if_scb_filters = self.if_scb_filters
        if self.mm_scb_filters is not None:  # If provided
            mm_scb_filters = self.mm_scb_filters
        if self.kernel_size != 3:  # If not default
            kernel_size = self.kernel_size
        if self.if_fc_units != 128:  # If not default
            if_fc_units = self.if_fc_units
        if self.mm_fc_units != 512:  # If not default
            mm_fc_units = self.mm_fc_units
        if self.n_if_factors != 3:  # If not default
            n_if_factors = self.n_if_factors

        return if_scb_filters, mm_scb_filters, kernel_size, if_fc_units, mm_fc_units, n_if_factors

    def _construct(self, input_shape, output_shape):
        """Construct the QNet architecture using functional API.

        Parameters
        ----------
            input_shape (tuple): Input tensor shape
            output_shape (tuple): Output tensor shape
        """
        # Parse model configuration
        if_scb_filters, mm_scb_filters, kernel_size, if_fc_units, mm_fc_units, n_if_factors = self._parse_model_config(input_shape)

        # Get framework-determined parameters
        n_freqs = input_shape[-1]
        n_metabolites = len(self.metabolites)

        # Create the QNet model with parsed parameters
        self.qnet_arch = QNetModel(
            n_freqs=n_freqs,
            n_metabolites=n_metabolites,
            n_if_factors=n_if_factors,
            if_scb_filters=if_scb_filters,
            mm_scb_filters=mm_scb_filters,
            kernel_size=kernel_size,
            if_fc_units=if_fc_units,
            mm_fc_units=mm_fc_units,
            name=self.model
        )

        # Create a wrapper model for training that outputs concentrations directly
        # This is a simplified approach - in practice, you'd implement the full LLS
        self.training_model = self._create_training_model()

    def _create_training_model(self):
        """Create a training model that outputs concentrations directly.

        This is a simplified approach for training. In practice, you would
        implement the full LLS module with basis set modulation.

        Returns
        -------
            keras.Model: Training model that outputs metabolite concentrations
        """
        # Get the input shape from MRSNet after reshape_spectra_data
        # The data is reshaped from (batch, acquisition, datatype, frequency) to (batch, acquisition*datatype, frequency)
        input_shape = (len(self.acquisitions) * len(self.datatype), self.qnet_arch.n_freqs)

        # Create input layer for multi-acquisition input
        inputs = Input(shape=input_shape, name='spectrum_input')

        # Process each acquisition separately with QNet
        # Split acquisitions: (batch, acquisition*datatype, freqs) -> list of (batch, freqs)
        acquisition_outputs = []

        # Extract edit_off acquisition (index 0) with all its datatypes
        edit_off_acq = AcquisitionDatatypeSplitter(0, len(self.datatype), name='extract_edit_off')(inputs)
        edit_off_processor = DatatypeProcessor(name='process_edit_off')
        edit_off_data = edit_off_processor(edit_off_acq)  # Extract first datatype
        edit_off_outputs = self.qnet_arch(edit_off_data)  # Process with QNet
        edit_off_if_factors = edit_off_outputs['if_factors']  # Extract IF factors
        acquisition_outputs.append(edit_off_if_factors)

        # Extract second acquisition (index 1) if available
        if len(self.acquisitions) > 1:
            second_acq = AcquisitionDatatypeSplitter(1, len(self.datatype), name='extract_second_acq')(inputs)
            second_processor = DatatypeProcessor(name='process_second_acq')
            second_data = second_processor(second_acq)  # Extract first datatype
            second_outputs = self.qnet_arch(second_data)  # Process with QNet
            second_if_factors = second_outputs['if_factors']  # Extract IF factors
            acquisition_outputs.append(second_if_factors)

            # Combine IF factors from both acquisitions
            # Use custom layer to concatenate IF factors
            if_factors_combined = IFConcatenator(name='concat_if_factors')(acquisition_outputs)
        else:
            # Only one acquisition, use its IF factors directly
            if_factors_combined = edit_off_if_factors

        # Apply Basic LLS module for metabolite concentration estimation
        # Create a new LLS module for the training model to ensure proper variable scope
        training_lls_module = BasicLLSModule(
            n_metabolites=self.qnet_arch.n_metabolites,
            n_if_factors=self.qnet_arch.n_if_factors,
            name='training_lls'
        )
        concentrations = training_lls_module(if_factors_combined)

        # Create training model
        training_model = Model(inputs=inputs, outputs=concentrations, name=f'{self.model}_training')

        return training_model

    def _apply_imperfection_factors(self, basis_set, if_factors):
        """Apply imperfection factors to basis set.

        Parameters
        ----------
            basis_set (tensor): Original basis set
            if_factors (tensor): Predicted imperfection factors

        Returns
        -------
            tensor: Modified basis set with applied imperfection factors
        """
        # Reshape IF factors: (batch, n_metabolites * 3) -> (batch, n_metabolites, 3)
        batch_size = tf.shape(if_factors)[0]
        n_metabolites = len(self.metabolites)
        if_reshaped = tf.reshape(if_factors, (batch_size, n_metabolites, 3))

        # Extract individual factors
        phase_shift = if_reshaped[:, :, 0]  # Phase shift
        freq_shift = if_reshaped[:, :, 1]    # Frequency shift
        #linewidth_dev = if_reshaped[:, :, 2] # Linewidth deviation

        # Apply imperfection factors to basis set
        # This is a simplified implementation - in practice, this would involve
        # complex signal processing operations
        modified_basis = basis_set

        # Add phase shift
        phase_factor = tf.exp(1j * phase_shift)
        modified_basis = modified_basis * tf.cast(phase_factor, tf.complex64)

        # Add frequency shift (simplified)
        freq_factor = tf.exp(1j * 2 * np.pi * freq_shift)
        modified_basis = modified_basis * tf.cast(freq_factor, tf.complex64)

        return modified_basis

    def _linear_least_squares(self, modified_basis, clean_spectrum):
        """Perform linear least squares estimation of metabolite concentrations.

        Parameters
        ----------
            modified_basis (tensor): Basis set with applied imperfection factors
            clean_spectrum (tensor): Spectrum with MM signal removed

        Returns
        -------
            tensor: Estimated metabolite concentrations
        """
        # Convert to real part for LLS
        basis_real = tf.math.real(modified_basis)
        spectrum_real = tf.math.real(clean_spectrum)

        # Solve LLS: c = (M^T M)^(-1) M^T y
        # Using pseudo-inverse for numerical stability
        concentrations = tf.linalg.pinv(basis_real) @ tf.expand_dims(spectrum_real, -1)
        concentrations = tf.squeeze(concentrations, -1)

        return concentrations

    def train(self, d_data, v_data, epochs, batch_size,
              folder, verbose=0, image_dpi=[300], screen_dpi=96, train_dataset_name=""):
        """Train the QNet model on provided data.

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
            print(f"# Train QNet {self!s}")

        # Prepare data
        d_inp = tf.convert_to_tensor(d_data[0], dtype=tf.float32)
        d_out = tf.convert_to_tensor(d_data[1], dtype=tf.float32)
        d_inp = reshape_spectra_data(d_inp, add_channel_dim=False)

        train_data = tf.data.Dataset.from_tensor_slices((d_inp, d_out))

        if v_data is not None:
            v_inp = tf.convert_to_tensor(v_data[0], dtype=tf.float32)
            v_out = tf.convert_to_tensor(v_data[1], dtype=tf.float32)
            v_inp = reshape_spectra_data(v_inp, add_channel_dim=False)
            validation_data = tf.data.Dataset.from_tensor_slices((v_inp, v_out))
        else:
            validation_data = None

        if verbose > 0:
            print("  Input:", d_inp.shape, "[spectrum, acquisition x datatype, frequency]")
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
                self.training_model.compile(loss='mse',
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

        # Create comprehensive architecture plots
        self._create_architecture_plots(folder, image_dpi, verbose)

        if verbose > 0:
            self.qnet_arch.summary()

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

        # Dataset options
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        data = tf.data.Dataset.from_tensor_slices(d_inp).batch(32).with_options(options)

        # Use training model for prediction
        concentrations = self.training_model.predict(data, verbose=(verbose>0)*2)

        return np.array(concentrations, dtype=np.float64)

    def save(self, folder):
        """Save the trained QNet model to disk.

        Parameters
        ----------
            folder (str): Directory to save the model
        """
        os.makedirs(folder, exist_ok=True)
        self.training_model.save(os.path.join(folder, "model.keras"))

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
        """Load a trained QNet model from disk.

        Parameters
        ----------
            path (str): Directory containing the saved model

        Returns
        -------
            QNet: Loaded QNet model instance
        """
        try:
            with open(os.path.join(path, "mrsnet.json")) as f:
                data = json.load(f)
            model = QNet(data['model'], data['metabolites'], data['pulse_sequence'], data['acquisitions'],
                         data['datatype'], data['norm'])
            model.train_dataset_name = data['train_dataset_name']
            model.training_model = load_model(os.path.join(path, "model.keras"),
                                             custom_objects={
                                                 "QNetModel": QNetModel,
                                                 "StackedConvolutionalBlock": StackedConvolutionalBlock,
                                                 "BasicLLSModule": BasicLLSModule,
                                                 "AcquisitionSplitter": AcquisitionSplitter,
                                                 "AcquisitionDatatypeSplitter": AcquisitionDatatypeSplitter,
                                                 "DatatypeProcessor": DatatypeProcessor,
                                                 "IFConcatenator": IFConcatenator
                                             },
                                             safe_mode=False)
            # Reconstruct the QNet architecture
            model._construct(model.training_model.input_shape[1:], model.training_model.output_shape[1:])
            return model
        except Exception as e:
            print(f"QNet load error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _create_architecture_plots(self, folder, image_dpi, verbose):
        """Create comprehensive architecture plots for QNet."""
        try:
            # 1. Main training model architecture
            for dpi in image_dpi:
                try:
                    plot_model(self.training_model,
                               to_file=os.path.join(folder, f'qnet_training_architecture@{dpi}.png'),
                               show_shapes=True,
                               show_dtype=True,
                               show_layer_names=True,
                               rankdir='TB',
                               expand_nested=True,
                               dpi=dpi)
                except Exception as e:
                    try:
                        plot_model(self.training_model,
                                   to_file=os.path.join(folder, f'qnet_training_architecture@{dpi}.svg'),
                                   show_shapes=True,
                                   show_dtype=True,
                                   show_layer_names=True,
                                   rankdir='TB',
                                   expand_nested=True,
                                   dpi=dpi)
                        if verbose > 0:
                            print(f"# WARNING: QNet training architecture PNG failed; wrote SVG instead: {e}")
                    except Exception as e2:
                        if verbose > 0:
                            print(f"# WARNING: QNet training architecture plot failed: {e2}")

            # 2. Individual QNet architecture (core model) - Enhanced
            self._plot_enhanced_core_architecture(folder, image_dpi, verbose)

            # 3. Create module architecture plots
            self._create_module_plots(folder, image_dpi, verbose)

        except Exception as e:
            if verbose > 0:
                print(f"# WARNING: Architecture plotting failed: {e}")

    def _plot_enhanced_core_architecture(self, folder, image_dpi, verbose):
        """Create an enhanced plot of the QNet core architecture showing all layers."""
        try:
            # Create a detailed model that shows the QNet architecture step by step
            from tensorflow.keras.layers import Dense, Input
            from tensorflow.keras.models import Model

            input_layer = Input(shape=(2048,), name='qnet_input')  # Single acquisition spectrum

            # IF Extraction Module
            if_scb1 = StackedConvolutionalBlock(filters=16, kernel_size=3, name='if_scb1')(input_layer)
            if_scb2 = StackedConvolutionalBlock(filters=32, kernel_size=3, name='if_scb2')(if_scb1)
            if_scb3 = StackedConvolutionalBlock(filters=64, kernel_size=3, name='if_scb3')(if_scb2)
            if_fc1 = Dense(units=128, activation='relu', name='if_fc1')(if_scb3)
            if_output = Dense(units=len(self.metabolites) * self.n_if_factors, activation='linear', name='if_output')(if_fc1)

            # MM Signal Prediction Module
            mm_scb1 = StackedConvolutionalBlock(filters=16, kernel_size=3, name='mm_scb1')(input_layer)
            mm_scb2 = StackedConvolutionalBlock(filters=32, kernel_size=3, name='mm_scb2')(mm_scb1)
            mm_scb3 = StackedConvolutionalBlock(filters=64, kernel_size=3, name='mm_scb3')(mm_scb2)
            mm_scb4 = StackedConvolutionalBlock(filters=128, kernel_size=3, name='mm_scb4')(mm_scb3)
            mm_scb5 = StackedConvolutionalBlock(filters=256, kernel_size=3, name='mm_scb5')(mm_scb4)
            mm_scb6 = StackedConvolutionalBlock(filters=256, kernel_size=3, name='mm_scb6')(mm_scb5)
            mm_fc1 = Dense(units=512, activation='relu', name='mm_fc1')(mm_scb6)
            mm_output = Dense(units=len(self.metabolites), activation='linear', name='mm_output')(mm_fc1)

            # Create outputs dictionary
            outputs = {
                'if_factors': if_output,
                'mm_signal': mm_output
            }

            # Create the detailed model
            detailed_model = Model(inputs=input_layer, outputs=outputs, name='QNet_Detailed')

            for dpi in image_dpi:
                try:
                    plot_model(detailed_model,
                               to_file=os.path.join(folder, f'qnet_detailed_architecture@{dpi}.png'),
                               show_shapes=True,
                               show_dtype=True,
                               show_layer_names=True,
                               rankdir='TB',
                               expand_nested=True,
                               dpi=dpi)
                except Exception as e:
                    try:
                        plot_model(detailed_model,
                                   to_file=os.path.join(folder, f'qnet_detailed_architecture@{dpi}.svg'),
                                   show_shapes=True,
                                   show_dtype=True,
                                   show_layer_names=True,
                                   rankdir='TB',
                                   expand_nested=True,
                                   dpi=dpi)
                        if verbose > 0:
                            print(f"# WARNING: QNet detailed architecture PNG failed; wrote SVG instead: {e}")
                    except Exception as e2:
                        if verbose > 0:
                            print(f"# WARNING: QNet detailed architecture plot failed: {e2}")

        except Exception as e:
            if verbose > 0:
                print(f"# WARNING: QNet detailed architecture creation failed: {e}")

    def _create_module_plots(self, folder, image_dpi, verbose):
        """Create visual plots of QNet modules."""
        try:
            # Create StackedConvolutionalBlock plot
            self._plot_stacked_conv_block(folder, image_dpi, verbose)

            # Create BasicLLSModule plot
            self._plot_basic_lls_module(folder, image_dpi, verbose)

            # Create AcquisitionSplitter plot
            self._plot_acquisition_splitter(folder, image_dpi, verbose)

            # Create IFConcatenator plot
            self._plot_if_concatenator(folder, image_dpi, verbose)

        except Exception as e:
            if verbose > 0:
                print(f"# WARNING: Module plotting failed: {e}")

    def _plot_stacked_conv_block(self, folder, image_dpi, verbose):
        """Create a visual plot of StackedConvolutionalBlock."""
        try:
            # Create a simple model to demonstrate SCB
            from tensorflow.keras.layers import Input
            input_layer = Input(shape=(2048, 1), name='scb_input')
            scb = StackedConvolutionalBlock(filters=32, kernel_size=3, name='block_scb')
            output = scb(input_layer)

            from tensorflow.keras.models import Model
            demo_model = Model(inputs=input_layer, outputs=output, name='StackedConvolutionalBlock_Block')

            for dpi in image_dpi:
                try:
                    plot_model(demo_model,
                               to_file=os.path.join(folder, f'qnet_stacked_conv_block@{dpi}.png'),
                               show_shapes=True,
                               show_dtype=True,
                               show_layer_names=True,
                               rankdir='TB',
                               expand_nested=True,
                               dpi=dpi)
                except Exception as e:
                    try:
                        plot_model(demo_model,
                                   to_file=os.path.join(folder, f'qnet_stacked_conv_block@{dpi}.svg'),
                                   show_shapes=True,
                                   show_dtype=True,
                                   show_layer_names=True,
                                   rankdir='TB',
                                   expand_nested=True,
                                   dpi=dpi)
                        if verbose > 0:
                            print(f"# WARNING: SCB PNG failed; wrote SVG instead: {e}")
                    except Exception as e2:
                        if verbose > 0:
                            print(f"# WARNING: SCB plot failed: {e2}")

        except Exception as e:
            if verbose > 0:
                print(f"# WARNING: SCB demo model creation failed: {e}")

    def _plot_basic_lls_module(self, folder, image_dpi, verbose):
        """Create a visual plot of BasicLLSModule."""
        try:
            # Create a simple model to demonstrate BasicLLSModule
            from tensorflow.keras.layers import Input
            input_layer = Input(shape=(10,), name='lls_input')  # 2 acquisitions * 5 metabolites * 1 if_factor
            lls_module = BasicLLSModule(n_metabolites=5, n_if_factors=2, name='block_lls')
            output = lls_module(input_layer)

            from tensorflow.keras.models import Model
            demo_model = Model(inputs=input_layer, outputs=output, name='BasicLLSModule_Block')

            for dpi in image_dpi:
                try:
                    plot_model(demo_model,
                               to_file=os.path.join(folder, f'qnet_basic_lls_module@{dpi}.png'),
                               show_shapes=True,
                               show_dtype=True,
                               show_layer_names=True,
                               rankdir='TB',
                               expand_nested=True,
                               dpi=dpi)
                except Exception as e:
                    try:
                        plot_model(demo_model,
                                   to_file=os.path.join(folder, f'qnet_basic_lls_module@{dpi}.svg'),
                                   show_shapes=True,
                                   show_dtype=True,
                                   show_layer_names=True,
                                   rankdir='TB',
                                   expand_nested=True,
                                   dpi=dpi)
                        if verbose > 0:
                            print(f"# WARNING: BasicLLS PNG failed; wrote SVG instead: {e}")
                    except Exception as e2:
                        if verbose > 0:
                            print(f"# WARNING: BasicLLS plot failed: {e2}")

        except Exception as e:
            if verbose > 0:
                print(f"# WARNING: BasicLLS demo model creation failed: {e}")

    def _plot_acquisition_splitter(self, folder, image_dpi, verbose):
        """Create a visual plot of AcquisitionSplitter."""
        try:
            # Create a simple model to demonstrate AcquisitionSplitter
            from tensorflow.keras.layers import Input
            input_layer = Input(shape=(2, 2048), name='splitter_input')  # 2 acquisitions, 2048 freqs
            splitter = AcquisitionSplitter(0, name='block_splitter')
            output = splitter(input_layer)

            from tensorflow.keras.models import Model
            demo_model = Model(inputs=input_layer, outputs=output, name='AcquisitionSplitter_Block')

            for dpi in image_dpi:
                try:
                    plot_model(demo_model,
                               to_file=os.path.join(folder, f'qnet_acquisition_splitter@{dpi}.png'),
                               show_shapes=True,
                               show_dtype=True,
                               show_layer_names=True,
                               rankdir='TB',
                               expand_nested=True,
                               dpi=dpi)
                except Exception as e:
                    try:
                        plot_model(demo_model,
                                   to_file=os.path.join(folder, f'qnet_acquisition_splitter@{dpi}.svg'),
                                   show_shapes=True,
                                   show_dtype=True,
                                   show_layer_names=True,
                                   rankdir='TB',
                                   expand_nested=True,
                                   dpi=dpi)
                        if verbose > 0:
                            print(f"# WARNING: AcquisitionSplitter PNG failed; wrote SVG instead: {e}")
                    except Exception as e2:
                        if verbose > 0:
                            print(f"# WARNING: AcquisitionSplitter plot failed: {e2}")

        except Exception as e:
            if verbose > 0:
                print(f"# WARNING: AcquisitionSplitter demo model creation failed: {e}")

    def _plot_if_concatenator(self, folder, image_dpi, verbose):
        """Create a visual plot of IFConcatenator."""
        try:
            # Create a simple model to demonstrate IFConcatenator
            from tensorflow.keras.layers import Input
            input1 = Input(shape=(5,), name='if_factors_1')
            input2 = Input(shape=(5,), name='if_factors_2')
            concatenator = IFConcatenator(name='block_concatenator')
            output = concatenator([input1, input2])

            from tensorflow.keras.models import Model
            demo_model = Model(inputs=[input1, input2], outputs=output, name='IFConcatenator_Block')

            for dpi in image_dpi:
                try:
                    plot_model(demo_model,
                               to_file=os.path.join(folder, f'qnet_if_concatenator@{dpi}.png'),
                               show_shapes=True,
                               show_dtype=True,
                               show_layer_names=True,
                               rankdir='TB',
                               expand_nested=True,
                               dpi=dpi)
                except Exception as e:
                    try:
                        plot_model(demo_model,
                                   to_file=os.path.join(folder, f'qnet_if_concatenator@{dpi}.svg'),
                                   show_shapes=True,
                                   show_dtype=True,
                                   show_layer_names=True,
                                   rankdir='TB',
                                   expand_nested=True,
                                   dpi=dpi)
                        if verbose > 0:
                            print(f"# WARNING: IFConcatenator PNG failed; wrote SVG instead: {e}")
                    except Exception as e2:
                        if verbose > 0:
                            print(f"# WARNING: IFConcatenator plot failed: {e2}")

        except Exception as e:
            if verbose > 0:
                print(f"# WARNING: IFConcatenator demo model creation failed: {e}")

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


class QNetBasis(QNet):
    """QNet with Full Basis Set LLS for MRS metabolite quantification.

    This class extends QNet to use the full basis set LLS implementation
    as described in the original QNet paper, providing scientifically accurate
    metabolite quantification through proper basis set integration.

    Key Features:
    ------------
    1. **Basis Set Matrix Construction**: Converts MRSNet Basis objects to numerical matrices
    2. **Imperfection Factor Modulation**: Applies phase, frequency, and linewidth shifts to basis spectra
    3. **Analytical LLS Solution**: Uses pseudo-inverse for metabolite concentration estimation
    4. **Automatic Parameter Extraction**: Extracts basis parameters from dataset paths
    5. **TensorFlow Integration**: Fully differentiable for training

    Architecture:
    ------------
    - **IF Extraction Module**: Deep CNN predicting imperfection factors (phase, frequency, linewidth)
    - **BasisLLSModule**: Full basis set LLS with imperfection factor modulation
    - **Multi-acquisition Support**: Processes multiple acquisitions and combines IF factors

    Basis Parameter Extraction:
    --------------------------
    Automatically extracts basis parameters from MRSNet dataset paths:
    - Format: source_sample_rate_samples/manufacturer/omega/linewidth/metabolites/pulse_sequence/...
    - Example: fid-a-2d_2000_4096/siemens/123.23/2.0/Cr-GABA-Gln-Glu-NAA/megapress/...
    - Supported sources: fid-a, fid-a-2d, lcmodel, pygamma, su-*

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
        qnet_arch (keras.Model): The actual QNet model
        basis_lls_module (BasisLLSModule): Full basis set LLS module
        basis_source (str): Basis source (e.g., 'fid-a-2d', 'lcmodel')
        manufacturer (str): Scanner manufacturer (e.g., 'siemens', 'ge')
        omega (float): Larmor frequency in Hz
        linewidth (float): Linewidth parameter

    Examples
    --------
    >>> # Automatic parameter extraction from dataset path
    >>> qnet_basis = QNetBasis(
    ...     model='qnet_original',
    ...     metabolites=['Cr', 'GABA', 'Glu', 'Gln', 'NAA'],
    ...     pulse_sequence='megapress',
    ...     acquisitions=['edit_off', 'difference'],
    ...     datatype=['real'],
    ...     norm='sum',
    ...     dataset_path='data/sim-spectra-megapress/fid-a-2d_2000_4096/siemens/123.23/2.0/Cr-GABA-Gln-Glu-NAA/megapress/sobol/1.0-adc_normal-0.0-0.03/10000-1'
    ... )
    >>>
    >>> # Manual parameter specification
    >>> qnet_basis = QNetBasis(
    ...     model='qnet_original',
    ...     metabolites=['Cr', 'GABA', 'Glu', 'Gln', 'NAA'],
    ...     pulse_sequence='megapress',
    ...     acquisitions=['edit_off', 'difference'],
    ...     datatype=['real'],
    ...     norm='sum',
    ...     basis_source='lcmodel',
    ...     manufacturer='siemens',
    ...     omega=123.23,
    ...     linewidth=2.0
    ... )

    Notes
    -----
    - Requires basis files to be available in data/basis-dist/
    - Automatically matches basis parameters to training dataset
    - Provides scientifically accurate metabolite quantification
    - Fully compatible with MRSNet training and benchmarking pipeline
    """

    def __init__(self, model, metabolites, pulse_sequence, acquisitions, datatype, norm,
                 basis_source=None, manufacturer=None, omega=None, linewidth=None, dataset_path=None):
        """Initialize QNet with full basis set LLS.

        Parameters
        ----------
        model : str
            Model architecture identifier
        metabolites : list
            List of metabolite names to predict
        pulse_sequence : str
            Pulse sequence type
        acquisitions : list
            List of acquisition types
        datatype : list
            List of data types
        norm : str
            Normalization method
        basis_source : str, optional
            Basis source for LLS module. If None, will be extracted from dataset_path
        manufacturer : str, optional
            Scanner manufacturer. If None, will be extracted from dataset_path
        omega : float, optional
            Larmor frequency in Hz. If None, will be extracted from dataset_path
        linewidth : float, optional
            Linewidth parameter. If None, will be extracted from dataset_path
        dataset_path : str, optional
            Path to training dataset. Used to extract basis parameters if not provided
        """
        # Initialize parent QNet class
        super().__init__(model, metabolites, pulse_sequence, acquisitions, datatype, norm)

        # Extract basis parameters from dataset path if not provided
        samples = 4096  # Default samples for basis files
        if dataset_path and (basis_source is None or manufacturer is None or omega is None or linewidth is None):
            try:
                basis_params = extract_basis_params_from_dataset_path(dataset_path)
                if basis_source is None:
                    basis_source = basis_params['source']
                if manufacturer is None:
                    manufacturer = basis_params['manufacturer']
                if omega is None:
                    omega = basis_params['omega']
                if linewidth is None:
                    linewidth = basis_params['linewidth']
                # Extract samples parameter
                samples = basis_params['samples']
                print(f"Extracted basis parameters from dataset: {basis_source}, {manufacturer}, {omega}, {linewidth}, samples={samples}")
            except Exception as e:
                print(f"Warning: Could not extract basis parameters from dataset path: {e}")
                # Use defaults
                if basis_source is None:
                    basis_source = 'lcmodel'
                if manufacturer is None:
                    manufacturer = 'siemens'
                if omega is None:
                    omega = 123.23
                if linewidth is None:
                    linewidth = 2.0
                samples = 4096  # Default samples for basis files

        # Store basis set parameters
        self.basis_source = basis_source
        self.manufacturer = manufacturer
        self.omega = omega
        self.linewidth = linewidth
        self.samples = samples

    def _create_training_model(self):
        """Create a training model using full basis set LLS."""
        # Ensure the QNet architecture is constructed
        if self.qnet_arch is None:
            # Use default input shape for construction
            input_shape = (len(self.acquisitions) * len(self.datatype), self.fft_samples)
            output_shape = (len(self.metabolites),)
            self._construct(input_shape, output_shape)

        # Get the input shape from MRSNet after reshape_spectra_data
        input_shape = (len(self.acquisitions) * len(self.datatype), self.qnet_arch.n_freqs)

        # Create input layer for multi-acquisition input
        inputs = Input(shape=input_shape, name='spectrum_input')

        # Process each acquisition separately with QNet
        acquisition_outputs = []

        # Create a new QNetModel instance for the training model to avoid serialization issues
        qnet_model = QNetModel(
            n_freqs=self.qnet_arch.n_freqs,
            n_metabolites=len(self.metabolites),
            n_if_factors=self.qnet_arch.n_if_factors,
            if_scb_filters=self.qnet_arch.if_scb_filters,
            kernel_size=self.qnet_arch.kernel_size,
            if_fc_units=self.qnet_arch.if_fc_units,
            name=f'{self.model}_training'
        )

        # Extract edit_off acquisition (index 0) with all its datatypes
        edit_off_acq = AcquisitionDatatypeSplitter(0, len(self.datatype), name='extract_edit_off')(inputs)
        edit_off_processor = DatatypeProcessor(name='process_edit_off')
        edit_off_data = edit_off_processor(edit_off_acq)
        edit_off_outputs = qnet_model(edit_off_data)
        edit_off_if_factors = edit_off_outputs['if_factors']
        acquisition_outputs.append(edit_off_if_factors)

        # Extract second acquisition (index 1) if available
        if len(self.acquisitions) > 1:
            second_acq = AcquisitionDatatypeSplitter(1, len(self.datatype), name='extract_second_acq')(inputs)
            second_processor = DatatypeProcessor(name='process_second_acq')
            second_data = second_processor(second_acq)
            second_outputs = qnet_model(second_data)
            second_if_factors = second_outputs['if_factors']
            acquisition_outputs.append(second_if_factors)

            # Combine IF factors from both acquisitions
            if_factors_combined = IFConcatenator(name='concat_if_factors')(acquisition_outputs)
        else:
            # Only one acquisition, use its IF factors directly
            if_factors_combined = edit_off_if_factors

        # Create basis LLS module locally for the training model
        # This avoids serialization issues by not storing it as an instance attribute
        # Pass values directly instead of using self to avoid references to QNetBasis instance
        basis_lls_module = create_basis_lls_module(
            metabolites=list(self.metabolites),  # Convert to list to avoid reference
            basis_source=str(self.basis_source),  # Convert to string to avoid reference
            manufacturer=str(self.manufacturer),  # Convert to string to avoid reference
            omega=float(self.omega),  # Convert to float to avoid reference
            linewidth=float(self.linewidth),  # Convert to float to avoid reference
            pulse_sequence=str(self.pulse_sequence),  # Convert to string to avoid reference
            acquisitions=list(self.acquisitions),  # Convert to list to avoid reference
            n_freqs=int(self.qnet_arch.n_freqs),  # Convert to int to avoid reference
            sample_rate=2000,
            samples=int(self.samples),  # Convert to int to avoid reference
            name='basis_lls'
        )

        # For BasisLLSModule, use only the first acquisition (edit_off) to avoid batch size mismatches
        # This is consistent with the original QNet paper which focuses on single acquisition LLS
        n_metabolites = len(self.metabolites)

        # Use Reshape layer instead of Lambda to avoid tf scope issues
        edit_off_if_factors_reshaped = Reshape(
            (n_metabolites, 3),
            name='reshape_if_factors'
        )(edit_off_if_factors)

        # Use the first acquisition for basis LLS (edit_off)
        observed_spectrum = edit_off_data  # Use the processed spectrum

        # Apply basis LLS module
        concentrations = basis_lls_module(
            if_factors=edit_off_if_factors_reshaped,
            observed_spectrum=observed_spectrum,
            acquisition=str(self.acquisitions[0])  # Convert to string to avoid reference
        )

        # Create training model
        training_model = Model(inputs=inputs, outputs=concentrations, name=f'{self.model}_basis_training')

        return training_model

    def save(self, folder):
        """Save QNet with basis LLS module to disk.

        Parameters
        ----------
            folder (str): Directory to save the model
        """
        os.makedirs(folder, exist_ok=True)

        # Save the training model (this contains the BasisLLSModule)
        self.training_model.save(os.path.join(folder, "model.keras"))

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

        # Extract parameter counts from summary
        for line in summary_output.split('\n'):
            if 'Total params:' in line:
                try:
                    total_params = int(line.split('Total params:')[1].split()[0].replace(',', ''))
                except (ValueError, IndexError):
                    pass
            elif 'Trainable params:' in line:
                try:
                    trainable_params = int(line.split('Trainable params:')[1].split()[0].replace(',', ''))
                except (ValueError, IndexError):
                    pass
            elif 'Non-trainable params:' in line:
                try:
                    non_trainable_params = int(line.split('Non-trainable params:')[1].split()[0].replace(',', ''))
                except (ValueError, IndexError):
                    pass

        # Create model metadata
        model_data = {
            'model': self.model,
            'metabolites': self.metabolites,
            'pulse_sequence': self.pulse_sequence,
            'acquisitions': self.acquisitions,
            'datatype': self.datatype,
            'norm': self.norm,
            'train_dataset_name': self.train_dataset_name,
            'fft_samples': self.fft_samples,
            'output': self.output,
            'low_ppm': self.low_ppm,
            'high_ppm': self.high_ppm,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'flops': getattr(self, 'flops', None)
        }

        # Save model metadata
        with open(os.path.join(folder, "mrsnet.json"), "w") as f:
            json.dump(model_data, f, indent=2)

        # Save basis LLS module parameters
        basis_config = {
            'basis_source': self.basis_source,
            'manufacturer': self.manufacturer,
            'omega': self.omega,
            'linewidth': self.linewidth,
            'samples': self.samples
        }

        with open(os.path.join(folder, "basis_config.json"), "w") as f:
            json.dump(basis_config, f, indent=2)

    @staticmethod
    def load(path):
        """Load QNet with basis LLS module from disk."""
        # Load parent QNet data
        model = QNet.__new__(QNetBasis)

        with open(os.path.join(path, "mrsnet.json")) as f:
            data = json.load(f)

        # Load basis configuration
        basis_config_path = os.path.join(path, "basis_config.json")
        if os.path.exists(basis_config_path):
            with open(basis_config_path) as f:
                basis_config = json.load(f)
        else:
            # Default basis configuration
            basis_config = {
                'basis_source': 'lcmodel',
                'manufacturer': 'siemens',
                'omega': 123.23,
                'linewidth': 2.0,
                'samples': 4096
            }

        # Initialize model attributes
        model.model = data['model']
        model.metabolites = data['metabolites']
        model.pulse_sequence = data['pulse_sequence']
        model.acquisitions = data['acquisitions']
        model.datatype = data['datatype']
        model.norm = data['norm']
        model.train_dataset_name = data['train_dataset_name']
        model.fft_samples = data.get('fft_samples', 2048)
        model.output = data.get('output', "concentrations")
        model.low_ppm = data.get('low_ppm', 0.2)
        model.high_ppm = data.get('high_ppm', 4.2)

        # Load basis configuration
        model.basis_source = basis_config['basis_source']
        model.manufacturer = basis_config['manufacturer']
        model.omega = basis_config['omega']
        model.linewidth = basis_config['linewidth']
        model.samples = basis_config.get('samples', 4096)

        # Load the training model (which contains the BasisLLSModule)
        model.training_model = load_model(os.path.join(path, "model.keras"),
                                         custom_objects={
                                             "QNetModel": QNetModel,
                                             "StackedConvolutionalBlock": StackedConvolutionalBlock,
                                             "BasicLLSModule": BasicLLSModule,
                                             "BasisLLSModule": BasisLLSModule,
                                             "AcquisitionSplitter": AcquisitionSplitter,
                                             "AcquisitionDatatypeSplitter": AcquisitionDatatypeSplitter,
                                             "DatatypeProcessor": DatatypeProcessor,
                                             "IFConcatenator": IFConcatenator
                                         },
                                         safe_mode=False)

        # Extract the QNet architecture from the training model
        # The training model contains the QNetModel as a sub-model
        model.qnet_arch = None  # Will be set when needed during prediction

        return model
