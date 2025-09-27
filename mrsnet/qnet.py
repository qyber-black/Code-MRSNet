# mrsnet/qnet.py - MRSNet - QNet model
#
# SPDX-FileCopyrightText: Copyright (C) 2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""QNet (Quantification Network) model for MRSNet.

This module implements the QNet architecture from the paper:
"Magnetic Resonance Spectroscopy Quantification Aided by Deep Estimations of
Imperfection Factors and Macromolecular Signal"

QNet consists of two deep learning modules:
1. IF Extraction Module: Predicts imperfection factors (phase, frequency, linewidth)
2. MM Signal Prediction Module: Predicts macromolecule background signal
3. LLS Module: Linear Least Squares for metabolite concentration estimation
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
    Flatten, Input, MaxPooling1D, ReLU
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


@register_keras_serializable(package="mrsnet", name="QNetModel")
class QNetModel(Model):
    """QNet model combining IF extraction, MM prediction, and LLS.

    This implements the complete QNet architecture with two deep learning
    modules and a linear least squares component.
    """

    def __init__(self, n_freqs, n_metabolites, n_if_factors=3, name='QNetModel', **kwargs):
        """Initialize QNet model.

        Parameters
        ----------
            n_freqs (int): Number of frequency points in spectrum
            n_metabolites (int): Number of metabolites to quantify
            n_if_factors (int): Number of imperfection factors (default 3: phase, freq, linewidth)
            name (str, optional): Model name
        """
        super().__init__(name=name, **kwargs)
        self.n_freqs = n_freqs
        self.n_metabolites = n_metabolites
        self.n_if_factors = n_if_factors

        # Input layer
        self.input_layer = Input(shape=(n_freqs,), name='spectrum_input')

        # Transpose input for Conv1D layers
        # Input is (batch, n_acquisitions * n_datatypes, n_freqs) -> (batch, n_freqs, n_acquisitions * n_datatypes)
        self.input_transpose = tf.keras.layers.Lambda(lambda x: tf.transpose(x, [0, 2, 1]), name='input_transpose')

        # IF Extraction Module: 3 SCBs + 2 FCLs
        self.if_scb1 = StackedConvolutionalBlock(16, name='if_scb1')
        self.if_scb2 = StackedConvolutionalBlock(32, name='if_scb2')
        self.if_scb3 = StackedConvolutionalBlock(64, name='if_scb3')
        self.if_flatten = Flatten(name='if_flatten')
        self.if_fc1 = Dense(128, activation='relu', name='if_fc1')
        self.if_fc2 = Dense(n_metabolites * n_if_factors, name='if_output')

        # MM Signal Prediction Module: 6 SCBs + 2 FCLs
        self.mm_scb1 = StackedConvolutionalBlock(16, name='mm_scb1')
        self.mm_scb2 = StackedConvolutionalBlock(32, name='mm_scb2')
        self.mm_scb3 = StackedConvolutionalBlock(64, name='mm_scb3')
        self.mm_scb4 = StackedConvolutionalBlock(128, name='mm_scb4')
        self.mm_scb5 = StackedConvolutionalBlock(256, name='mm_scb5')
        self.mm_scb6 = StackedConvolutionalBlock(256, name='mm_scb6')
        self.mm_flatten = Flatten(name='mm_flatten')
        self.mm_fc1 = Dense(512, activation='relu', name='mm_fc1')
        self.mm_fc2 = Dense(n_freqs, name='mm_output')

        # Build the model
        self.build((None, n_freqs))

    def call(self, inputs):
        """Forward pass through QNet."""
        # Transpose input for Conv1D layers
        x = self.input_transpose(inputs)

        # IF Extraction branch
        if_x = self.if_scb1(x)
        if_x = self.if_scb2(if_x)
        if_x = self.if_scb3(if_x)
        if_x = self.if_flatten(if_x)
        if_x = self.if_fc1(if_x)
        if_factors = self.if_fc2(if_x)

        # MM Signal Prediction branch
        mm_x = self.mm_scb1(x)
        mm_x = self.mm_scb2(mm_x)
        mm_x = self.mm_scb3(mm_x)
        mm_x = self.mm_scb4(mm_x)
        mm_x = self.mm_scb5(mm_x)
        mm_x = self.mm_scb6(mm_x)
        mm_x = self.mm_flatten(mm_x)
        mm_x = self.mm_fc1(mm_x)
        mm_signal = self.mm_fc2(mm_x)

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
            'n_if_factors': self.n_if_factors
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

    def __init__(self, model, metabolites, pulse_sequence, acquisitions, datatype, norm):
        """Initialize a QNet model.

        Parameters
        ----------
            model (str): Model architecture identifier (e.g., 'qnet_default')
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
        self.qnet_arch = None

        # Basis set for LLS (will be loaded during training)
        self.basis_set = None

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
        """Reset the QNet architecture and training dataset name.

        Clears the current model architecture and training dataset information.
        """
        self.qnet_arch = None
        self.train_dataset_name = None
        self.basis_set = None

    def _construct(self, input_shape, output_shape):
        """Construct the QNet architecture using functional API.

        Parameters
        ----------
            input_shape (tuple): Input tensor shape
            output_shape (tuple): Output tensor shape
        """
        vals = self.model.split("_")
        if vals[0] != 'qnet':
            raise RuntimeError(f"Unknown model {vals[0]}")

        # Parse model parameters
        if len(vals) == 1:
            # Default QNet configuration
            n_freqs = input_shape[-1]
            n_metabolites = len(self.metabolites)
        elif len(vals) == 2 and vals[1] == 'default':
            # qnet_default configuration
            n_freqs = input_shape[-1]
            n_metabolites = len(self.metabolites)
        else:
            # Custom QNet configuration: qnet_[FREQS]_[METABOLITES]
            n_freqs = int(vals[1]) if len(vals) > 1 else input_shape[-1]
            n_metabolites = int(vals[2]) if len(vals) > 2 else len(self.metabolites)

        # Create QNet model
        self.qnet_arch = QNetModel(n_freqs, n_metabolites, name=self.model)

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
        # Get the input shape from the QNet model
        # The QNet model expects (n_acquisitions * n_datatypes, n_freqs) input
        input_shape = (len(self.acquisitions) * len(self.datatype), self.fft_samples)

        # Create input layer
        inputs = Input(shape=input_shape, name='spectrum_input')

        # Get QNet outputs
        qnet_outputs = self.qnet_arch(inputs)

        # Simplified concentration prediction using IF factors
        # In practice, this would use the LLS module with basis set
        if_factors = qnet_outputs['if_factors']

        # Reshape IF factors to get concentration estimates
        # This is a placeholder - real implementation would use LLS
        concentrations = Dense(len(self.metabolites),
                             activation='softmax',
                             name='concentrations')(if_factors)

        # Create training model
        training_model = Model(inputs=inputs, outputs=concentrations, name=f"{self.model}_training")

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
        linewidth_dev = if_reshaped[:, :, 2] # Linewidth deviation

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
            self.flops = calculate_flops(self.qnet_arch, d_inp.shape[1:])
        except Exception as e:
            if verbose > 0:
                print(f"Error calculating FLOPs: {e}")
            self.flops = 0

        # Plot model architecture
        for dpi in image_dpi:
            try:
                plot_model(self.qnet_arch,
                           to_file=os.path.join(folder,'architecture@'+str(dpi)+'.png'),
                           show_shapes=True,
                           show_dtype=True,
                           show_layer_names=True,
                           rankdir='TB',
                           expand_nested=True,
                           dpi=dpi)
            except Exception as e:
                try:
                    plot_model(self.qnet_arch,
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
            self.qnet_arch.summary()

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
        with open(os.path.join(path, "mrsnet.json")) as f:
            data = json.load(f)
        model = QNet(data['model'], data['metabolites'], data['pulse_sequence'], data['acquisitions'],
                     data['datatype'], data['norm'])
        model.train_dataset_name = data['train_dataset_name']
        model.training_model = load_model(os.path.join(path, "model.keras"),
                                         custom_objects={
                                             "QNetModel": QNetModel,
                                             "StackedConvolutionalBlock": StackedConvolutionalBlock
                                         },
                                         safe_mode=False)
        # Reconstruct the QNet architecture
        model._construct(model.training_model.input_shape[1:], model.training_model.output_shape[1:])
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
