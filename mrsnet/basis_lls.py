# mrsnet/basis_lls.py - MRSNet - Basis Set LLS Module
#
# SPDX-FileCopyrightText: Copyright (C) 2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Basis Set Linear Least Squares (LLS) Module for QNet.

This module implements the full basis set LLS as described in the original QNet paper:
"Quantification of Magnetic Resonance Spectroscopy Data Using Deep Learning"
by Chen, D., et al. IEEE Trans Biomed Eng, 2024 Jun;71(6):1841-1852.

The BasisLLSModule provides:
1. Basis set matrix construction from MRSNet Basis objects
2. Imperfection factor modulation of basis spectra
3. Analytical LLS solution for metabolite concentration estimation
4. TensorFlow integration for differentiable training
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import register_keras_serializable

import mrsnet.basis as basis
from mrsnet.cfg import Cfg


@register_keras_serializable(package="mrsnet", name="BasisLLSModule")
class BasisLLSModule(keras.layers.Layer):
    """
    Full basis set LLS module for QNet metabolite quantification.

    This module implements the complete LLS approach from the original QNet paper,
    including basis set matrix construction, imperfection factor modulation,
    and analytical concentration estimation.

    Parameters
    ----------
    metabolites : list
        List of metabolite names
    basis_source : str
        Basis source (e.g., 'lcmodel', 'fid-a', 'pygamma')
    manufacturer : str
        Scanner manufacturer
    omega : float
        Larmor frequency in Hz
    linewidth : float
        Linewidth parameter
    pulse_sequence : str
        Pulse sequence type
    acquisitions : list
        List of acquisition types
    n_freqs : int
        Number of frequency points
    sample_rate : float
        Sample rate in Hz
    name : str, optional
        Layer name
    """

    def __init__(self, metabolites, basis_source, manufacturer, omega, linewidth,
                 pulse_sequence, acquisitions, n_freqs, sample_rate=2000,
                 samples=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.metabolites = sorted(metabolites)
        self.basis_source = basis_source
        self.manufacturer = manufacturer
        self.omega = omega
        self.linewidth = linewidth
        self.pulse_sequence = pulse_sequence
        self.acquisitions = acquisitions
        self.n_freqs = n_freqs
        self.sample_rate = sample_rate
        self.samples = samples

        # Basis set matrices (one per acquisition)
        self.basis_matrices = {}

        # Frequency axis for imperfection factor modulation
        self.freq_axis = None

    def build(self, input_shape):
        """Build the basis set matrices and frequency axis."""
        super().build(input_shape)

        # Load basis set from MRSNet
        try:
            # Try to load basis set with original samples (from basis files)
            # We'll resize the matrices to match n_freqs later
            basis_obj = basis.Basis(
                metabolites=self.metabolites,
                source=self.basis_source,
                manufacturer=self.manufacturer,
                omega=self.omega,
                linewidth=self.linewidth,
                pulse_sequence=self.pulse_sequence,
                sample_rate=self.sample_rate,
                samples=self.samples  # Use samples from dataset path
            ).setup(Cfg.val['path_basis'], search_basis=Cfg.val['search_basis'])

            # Construct basis matrices for each acquisition
            for acq in self.acquisitions:
                basis_matrix = self._construct_basis_matrix(basis_obj, acq)
                self.basis_matrices[acq] = tf.constant(basis_matrix, dtype=tf.complex64)

        except Exception as e:
            print(f"Warning: Could not load basis set ({e}), using dummy basis matrices")
            # Create dummy basis matrices for testing
            for acq in self.acquisitions:
                real_part = np.random.normal(0, 0.1, (self.n_freqs, len(self.metabolites)))
                imag_part = np.random.normal(0, 0.1, (self.n_freqs, len(self.metabolites)))
                dummy_matrix = real_part.astype(np.complex64) + imag_part.astype(np.complex64) * 1j
                self.basis_matrices[acq] = tf.constant(dummy_matrix, dtype=tf.complex64)

        # Create frequency axis for imperfection factor modulation
        self.freq_axis = tf.constant(
            np.linspace(-self.sample_rate/2, self.sample_rate/2, self.n_freqs),
            dtype=tf.float32
        )

    def _construct_basis_matrix(self, basis_obj, acquisition):
        """
        Construct basis set matrix for a specific acquisition.

        Parameters
        ----------
        basis_obj : Basis
            MRSNet Basis object
        acquisition : str
            Acquisition type (e.g., 'edit_off', 'difference')

        Returns
        -------
        numpy.ndarray
            Complex basis matrix of shape (n_freqs, n_metabolites)
        """
        basis_matrix = np.zeros((self.n_freqs, len(self.metabolites)), dtype=complex)

        for i, metabolite in enumerate(self.metabolites):
            if metabolite in basis_obj.spectra and acquisition in basis_obj.spectra[metabolite]:
                spectrum = basis_obj.spectra[metabolite][acquisition]
                fft_data, _ = spectrum.get_f()

                # Resize to match training data size (n_freqs)
                if len(fft_data) != self.n_freqs:
                    if len(fft_data) > self.n_freqs:
                        # Downsample: take the first n_freqs points (basis files are typically 4096, training data is 2048)
                        fft_data = fft_data[:self.n_freqs]
                    else:
                        # Upsample: pad with zeros
                        padded = np.zeros(self.n_freqs, dtype=complex)
                        padded[:len(fft_data)] = fft_data
                        fft_data = padded

                basis_matrix[:, i] = fft_data
            else:
                # Zero spectrum if metabolite/acquisition not found
                basis_matrix[:, i] = np.zeros(self.n_freqs, dtype=complex)

        return basis_matrix

    def _apply_imperfection_factors(self, basis_matrix, if_factors):
        """
        Apply imperfection factors to basis set matrix.

        Parameters
        ----------
        basis_matrix : tensor
            Basis set matrix of shape (n_freqs, n_metabolites)
        if_factors : tensor
            Imperfection factors of shape (batch, n_metabolites, 3)
            where last dimension is [phase_shift, freq_shift, linewidth_dev]

        Returns
        -------
        tensor
            Modulated basis matrix of shape (batch, n_freqs, n_metabolites)
        """
        batch_size = tf.shape(if_factors)[0]
        n_metabolites = tf.shape(if_factors)[1]

        # Extract imperfection factors
        phase_shifts = if_factors[:, :, 0]  # (batch, n_metabolites)
        freq_shifts = if_factors[:, :, 1]  # (batch, n_metabolites)
        linewidth_devs = if_factors[:, :, 2]  # (batch, n_metabolites)

        # Expand dimensions for broadcasting
        freq_axis = tf.expand_dims(tf.expand_dims(self.freq_axis, 0), 2)  # (1, n_freqs, 1)
        phase_shifts = tf.expand_dims(phase_shifts, 1)  # (batch, 1, n_metabolites)
        freq_shifts = tf.expand_dims(freq_shifts, 1)  # (batch, 1, n_metabolites)
        linewidth_devs = tf.expand_dims(linewidth_devs, 1)  # (batch, 1, n_metabolites)

        # Apply phase shift: multiply by exp(i * phase_shift)
        phase_modulation = tf.exp(tf.complex(0.0, phase_shifts))

        # Apply frequency shift: multiply by exp(i * 2π * freq_shift * t)
        # Convert frequency shift from ppm to Hz
        freq_shift_hz = freq_shifts * self.omega  # ppm to Hz
        freq_modulation = tf.exp(tf.complex(0.0, 2 * np.pi * freq_shift_hz * freq_axis / self.sample_rate))

        # Apply linewidth deviation: multiply by exp(-linewidth_dev * |freq|)
        # This is a simplified model - in practice, linewidth affects the time domain
        linewidth_modulation = tf.exp(tf.complex(-tf.abs(linewidth_devs) * tf.abs(freq_axis) / self.sample_rate, 0.0))

        # Combine all modulations
        total_modulation = phase_modulation * freq_modulation * linewidth_modulation

        # Apply modulation to basis matrix
        # basis_matrix: (n_freqs, n_metabolites)
        # total_modulation: (batch, n_freqs, n_metabolites)
        basis_matrix_expanded = tf.expand_dims(basis_matrix, 0)  # (1, n_freqs, n_metabolites)
        modulated_basis = basis_matrix_expanded * total_modulation

        return modulated_basis

    def call(self, if_factors, observed_spectrum, acquisition):
        """
        Apply LLS to estimate metabolite concentrations.

        Parameters
        ----------
        if_factors : tensor
            Imperfection factors tensor of shape (batch, n_metabolites, 3)
        observed_spectrum : tensor
            Observed spectrum tensor of shape (batch, n_freqs)
        acquisition : str
            Acquisition type string

        Returns
        -------
        tensor
            Estimated metabolite concentrations of shape (batch, n_metabolites)
        """
        # Get basis matrix for this acquisition
        basis_matrix = self.basis_matrices[acquisition]

        # Apply imperfection factors
        modulated_basis = self._apply_imperfection_factors(basis_matrix, if_factors)

        # Convert to real-valued problem for LLS
        # Take real part of both basis and observed spectrum
        real_basis = tf.math.real(modulated_basis)  # (batch, n_freqs, n_metabolites)
        real_observed = tf.math.real(observed_spectrum)  # (batch, n_freqs)

        # Solve LLS: min ||M'c - y||²
        # Solution: c = (M'^T M')^(-1) M'^T y

        # Compute M'^T M' for each batch element
        MtM = tf.matmul(real_basis, real_basis, transpose_a=True)  # (batch, n_metabolites, n_metabolites)

        # Tikhonov regularization to ensure numerical stability
        epsilon = tf.constant(1e-8, dtype=real_basis.dtype)
        batch_size = tf.shape(MtM)[0]
        n_met = tf.shape(MtM)[-1]
        identity = tf.eye(n_met, batch_shape=[batch_size], dtype=MtM.dtype)
        MtM_reg = MtM + epsilon * identity

        # Compute M'^T y for each batch element
        Mty = tf.matmul(real_basis, tf.expand_dims(real_observed, -1), transpose_a=True)  # (batch, n_metabolites, 1)

        # Solve linear system: (MtM + eps I) * c = Mty
        concentrations = tf.linalg.solve(MtM_reg, Mty)  # (batch, n_metabolites, 1)

        # Squeeze the last dimension to get (batch, n_metabolites)
        concentrations = tf.squeeze(concentrations, -1)

        return concentrations

    def compute_output_shape(self, input_shape):
        """Compute the output shape for the layer."""
        # Input: if_factors (batch, n_metabolites, 3), observed_spectrum (batch, n_freqs)
        # Output: concentrations (batch, n_metabolites)
        if input_shape is None:
            return (None, len(self.metabolites))

        # Handle both single input and multiple inputs
        if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 2:
            # Multiple inputs: if_factors and observed_spectrum
            batch_size = input_shape[0][0] if input_shape[0] is not None else None
        else:
            # Single input case
            batch_size = input_shape[0] if input_shape is not None else None

        return (batch_size, len(self.metabolites))

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'metabolites': self.metabolites,
            'basis_source': self.basis_source,
            'manufacturer': self.manufacturer,
            'omega': self.omega,
            'linewidth': self.linewidth,
            'pulse_sequence': self.pulse_sequence,
            'acquisitions': self.acquisitions,
            'n_freqs': self.n_freqs,
            'sample_rate': self.sample_rate,
            'samples': self.samples
        })
        return config


def extract_basis_params_from_dataset_path(dataset_path):
    """
    Extract basis parameters from MRSNet dataset path.

    Dataset path format: source_sample_rate_samples/manufacturer/omega/linewidth/metabolites/pulse_sequence/...
    Example: fid-a-2d_2000_4096/siemens/123.23/2.0/Cr-GABA-Gln-Glu-NAA/megapress/...

    Parameters
    ----------
    dataset_path : str
        Path to the dataset

    Returns
    -------
    dict
        Dictionary containing basis parameters:
        - source: Basis source (e.g., 'fid-a-2d')
        - manufacturer: Scanner manufacturer (e.g., 'siemens')
        - omega: Larmor frequency in Hz (e.g., 123.23)
        - linewidth: Linewidth parameter (e.g., 2.0)
        - metabolites: List of metabolite names
        - pulse_sequence: Pulse sequence type (e.g., 'megapress')
        - sample_rate: Sample rate in Hz
        - samples: Number of samples
    """
    import os

    # Split the path into components
    path_parts = dataset_path.split(os.sep)

    # Find the dataset name part (contains source, sample_rate, samples)
    dataset_name = None
    for part in path_parts:
        if '_' in part and any(char.isdigit() for char in part):
            dataset_name = part
            break

    if dataset_name is None:
        raise ValueError(f"Could not extract dataset name from path: {dataset_path}")

    # Parse dataset name: source_sample_rate_samples
    name_parts = dataset_name.split('_')
    if len(name_parts) < 3:
        raise ValueError(f"Invalid dataset name format: {dataset_name}")

    source = '_'.join(name_parts[:-2])  # Everything except last two parts
    sample_rate = int(name_parts[-2])
    samples = int(name_parts[-1])

    # Extract other parameters from path
    manufacturer = None
    omega = None
    linewidth = None
    metabolites = None
    pulse_sequence = None

    for i, part in enumerate(path_parts):
        if part in ['siemens', 'ge', 'philips']:
            manufacturer = part
        elif part.replace('.', '').isdigit() and manufacturer is not None:
            if omega is None:
                omega = float(part)
            elif linewidth is None:
                linewidth = float(part)
        elif '-' in part and any(char.isupper() for char in part):
            metabolites = part.split('-')
        elif part in ['megapress', 'press', 'steam']:
            pulse_sequence = part

    # Set defaults if not found
    if manufacturer is None:
        manufacturer = 'siemens'
    if omega is None:
        omega = 123.23
    if linewidth is None:
        linewidth = 2.0
    if metabolites is None:
        metabolites = ['Cr', 'GABA', 'Glu', 'Gln', 'NAA']
    if pulse_sequence is None:
        pulse_sequence = 'megapress'

    return {
        'source': source,
        'manufacturer': manufacturer,
        'omega': omega,
        'linewidth': linewidth,
        'metabolites': metabolites,
        'pulse_sequence': pulse_sequence,
        'sample_rate': sample_rate,
        'samples': samples
    }


def create_basis_lls_module(metabolites, basis_source='lcmodel', manufacturer='siemens',
                           omega=123.23, linewidth=2.0, pulse_sequence='megapress',
                           acquisitions=['edit_off', 'difference'], n_freqs=2048,
                           sample_rate=2000, samples=None, name='basis_lls'):
    """
    Create a BasisLLSModule with default parameters.

    Parameters
    ----------
    metabolites : list
        List of metabolite names
    basis_source : str, optional
        Basis source. Defaults to 'lcmodel'
    manufacturer : str, optional
        Scanner manufacturer. Defaults to 'siemens'
    omega : float, optional
        Larmor frequency in Hz. Defaults to 123.23
    linewidth : float, optional
        Linewidth parameter. Defaults to 2.0
    pulse_sequence : str, optional
        Pulse sequence type. Defaults to 'megapress'
    acquisitions : list, optional
        List of acquisition types. Defaults to ['edit_off', 'difference']
    n_freqs : int, optional
        Number of frequency points. Defaults to 2048
    sample_rate : float, optional
        Sample rate in Hz. Defaults to 2000
    samples : int, optional
        Number of samples in basis files. Defaults to None
    name : str, optional
        Module name. Defaults to 'basis_lls'

    Returns
    -------
    BasisLLSModule
        Configured basis set LLS module
    """
    return BasisLLSModule(
        metabolites=metabolites,
        basis_source=basis_source,
        manufacturer=manufacturer,
        omega=omega,
        linewidth=linewidth,
        pulse_sequence=pulse_sequence,
        acquisitions=acquisitions,
        n_freqs=n_freqs,
        sample_rate=sample_rate,
        samples=samples,
        name=name
    )
