# mrsnet/basis_lls.py - MRSNet - Basis Set LLS Module
#
# SPDX-FileCopyrightText: Copyright (C) 2025-2026 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Basis Set Linear Least Squares (LLS) Module for QNet.

This module provides an adapted physical-basis readout for QNet, motivated by:
"Quantification of Magnetic Resonance Spectroscopy Data Using Deep Learning"
by Chen, D., et al. IEEE Trans Biomed Eng, 2024 Jun;71(6):1841-1852.

The BasisLLSModule provides:
1. Basis set matrix construction from MRSNet Basis objects
2. Imperfection factor modulation of basis spectra
3. Analytical LLS solution for metabolite concentration estimation
4. TensorFlow integration for differentiable training
"""

import mrsnet.basis as basis
import numpy as np
import tensorflow as tf
from mrsnet.cfg import Cfg
from tensorflow import keras
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="mrsnet", name="BasisLLSModule")
class BasisLLSModule(keras.layers.Layer):
    """Fit real spectra with physically modulated, frequency-aligned bases.

    Phase is in radians, frequency shift is in signed MRSNet ppm, and the
    absolute third factor is additional Lorentzian FWHM in Hz. Model target
    normalisation is separate from downstream evaluation normalisation.
    """

    def __init__(
        self,
        metabolites,
        basis_source,
        manufacturer,
        omega,
        linewidth,
        pulse_sequence,
        acquisitions,
        n_freqs,
        sample_rate=2000,
        samples=None,
        name=None,
        solver_version=None,
        high_ppm=-4.5,
        low_ppm=-1.0,
        target_norm="none",
        **kwargs,
    ):
        if solver_version != 2:
            raise ValueError(
                "QNet-basis solver changed: old checkpoints require fresh training "
                "and cannot be loaded as corrected models"
            )
        super().__init__(name=name, **kwargs)
        self.metabolites = sorted(metabolites)
        self.basis_source = basis_source
        self.manufacturer = manufacturer
        self.omega = float(np.asarray(omega).item())
        self.linewidth = linewidth
        self.pulse_sequence = pulse_sequence
        self.acquisitions = list(acquisitions)
        self.n_freqs = int(n_freqs)
        self.sample_rate = float(np.asarray(sample_rate).item())
        self.samples = samples
        self.high_ppm = float(high_ppm)
        self.low_ppm = float(low_ppm)
        self.target_norm = "none" if target_norm is None else target_norm
        self.solver_version = solver_version
        if self.target_norm not in {"none", "sum", "max"}:
            raise ValueError("Unknown model target normalisation")
        if (
            not self.metabolites
            or not self.acquisitions
            or self.n_freqs < 2
            or not np.isfinite([self.omega, self.sample_rate, self.high_ppm, self.low_ppm]).all()
            or self.omega <= 0
            or self.sample_rate <= 0
            or self.high_ppm >= self.low_ppm
        ):
            raise ValueError("Invalid basis spectrum configuration")
        self.basis_matrices = {}
        self.basis_fids = {}
        self.crop_indices = None
        self.ppm_grid = None
        self.time_axis = None
        self.basis_is_real = False

    def build(self, input_shape):
        """Require a real basis and match the input export's FFT crop."""
        try:
            basis_obj = basis.Basis(
                metabolites=self.metabolites,
                source=self.basis_source,
                manufacturer=self.manufacturer,
                omega=self.omega,
                linewidth=self.linewidth,
                pulse_sequence=self.pulse_sequence,
                sample_rate=self.sample_rate,
                samples=self.samples,
            ).setup(Cfg.val["path_basis"], search_basis=Cfg.val["search_basis"])
        except Exception as error:
            if Cfg.val.get("basis_lls_require_real", False):
                raise RuntimeError("Real basis loading failed; dummy fallback is forbidden") from error
            # Preserve the generic diagnostic fallback, forbidden by paper protocols.
            self.crop_indices = tf.range(self.n_freqs, dtype=tf.int32)
            self.time_axis = tf.constant(np.arange(self.n_freqs) / self.sample_rate, dtype=tf.float32)
            self.ppm_grid = np.linspace(self.high_ppm, self.low_ppm, self.n_freqs)
            for acquisition in self.acquisitions:
                shape = (self.n_freqs, len(self.metabolites))
                matrix = np.random.normal(0, 0.1, shape) + 1j * np.random.normal(0, 0.1, shape)
                self.basis_matrices[acquisition] = tf.constant(matrix, dtype=tf.complex64)
                fid = np.fft.ifft(np.fft.ifftshift(matrix.T, axes=-1), axis=-1)
                self.basis_fids[acquisition] = tf.constant(fid, dtype=tf.complex64)
            super().build(input_shape)
            return
        for acquisition in self.acquisitions:
            fids = []
            for metabolite in self.metabolites:
                spectrum = basis_obj.spectra[metabolite][acquisition]
                fid, grid, indices = self._resampled_fid(spectrum)
                if self.ppm_grid is None:
                    self.ppm_grid = grid
                    self.crop_indices = tf.constant(indices, dtype=tf.int32)
                    self.time_axis = tf.constant(np.arange(len(fid)) / self.sample_rate, dtype=tf.float32)
                elif len(fid) != len(self.time_axis) or not np.array_equal(grid, self.ppm_grid):
                    raise ValueError("Basis spectra do not share one exported ppm grid")
                fids.append(fid)
            fids = np.asarray(fids, dtype=np.complex64)
            self.basis_fids[acquisition] = tf.constant(fids)
            fft = np.fft.fftshift(np.fft.fft(fids, axis=-1), axes=-1)
            self.basis_matrices[acquisition] = tf.constant(fft[:, indices].T, dtype=tf.complex64)
        self.basis_is_real = True
        super().build(input_shape)

    def _resampled_fid(self, spectrum):
        """Keep basis scaling while reproducing Spectrum.rescale_fft bins."""
        sample_rate = float(np.asarray(spectrum.sample_rate).item())
        omega = float(np.asarray(spectrum.omega).item())
        if not (
            np.isclose(sample_rate, self.sample_rate, rtol=1e-12, atol=0)
            and np.isclose(omega, self.omega, rtol=1e-12, atol=0)
        ):
            raise ValueError("Basis sample rate or Larmor frequency differs from the model")
        fft, native_grid = spectrum.get_f()
        fft = np.asarray(fft)
        if (
            fft.ndim != 1
            or not np.isfinite(fft).all()
            or not np.isfinite(native_grid).all()
            or (self.samples is not None and len(fft) != int(self.samples))
            or np.min(native_grid) > self.high_ppm
            or np.max(native_grid) < self.low_ppm
        ):
            raise ValueError("Basis does not cover the requested frequency grid")
        bandwidth = sample_rate / (2.0 * omega)
        length = int(2.0 * bandwidth / ((self.low_ppm - self.high_ppm) / self.n_freqs))
        centre = float(np.asarray(spectrum.center_ppm + spectrum.b0_shift_ppm).item())
        for _ in range(Cfg.val["spectrum_rescale_fft_max_repeats"] + 1):
            full_grid = np.linspace(-1, 1, length) * bandwidth + centre
            indices = np.flatnonzero((full_grid >= self.high_ppm) & (full_grid <= self.low_ppm))
            if len(indices) == self.n_freqs:
                break
            length += 1
        else:
            raise ValueError("Cannot construct the requested basis frequency grid")
        fid = np.fft.ifft(np.fft.ifftshift(fft))
        fid = np.pad(fid, (0, max(0, length - len(fid))))[:length]
        return fid, full_grid[indices], indices

    def _modulated_basis(self, if_factors, acquisition):
        """Apply physical imperfection factors before the FFT and crop."""
        factors = tf.cast(if_factors, tf.float32)
        factors = tf.ensure_shape(factors, (None, len(self.metabolites), 3))
        tf.debugging.assert_all_finite(factors, "Non-finite imperfection factors")
        time = self.time_axis[None, None, :]
        phase = factors[:, :, 0, None]
        shift_hz = factors[:, :, 1, None] * self.omega
        width_hz = tf.abs(factors[:, :, 2, None])
        exponent = tf.complex(-np.pi * width_hz * time, phase + 2.0 * np.pi * shift_hz * time)
        fid = self.basis_fids[acquisition][None, :, :] * tf.exp(exponent)
        fft = tf.signal.fftshift(tf.signal.fft(fid), axes=-1)
        return tf.transpose(tf.gather(fft, self.crop_indices, axis=-1), (0, 2, 1))

    def call(self, if_factors, observed_spectrum, acquisition):
        """Solve a real-valued fit and apply the configured target scale."""
        observed = tf.cast(tf.math.real(observed_spectrum), tf.float32)
        observed = tf.ensure_shape(observed, (None, self.n_freqs))
        tf.debugging.assert_all_finite(observed, "Non-finite observed spectrum")
        tf.debugging.assert_equal(tf.shape(observed)[0], tf.shape(if_factors)[0])
        matrix = tf.math.real(self._modulated_basis(if_factors, acquisition))
        normal = tf.matmul(matrix, matrix, transpose_a=True)
        rhs = tf.matmul(matrix, observed[:, :, None], transpose_a=True)
        ridge = tf.eye(len(self.metabolites), dtype=tf.float32) * 1e-8
        concentrations = tf.squeeze(tf.linalg.solve(normal + ridge, rhs), -1)
        concentrations = tf.debugging.check_numerics(concentrations, "Non-finite basis fit")
        if self.target_norm != "none":
            denominator = (
                tf.reduce_sum(concentrations, axis=-1, keepdims=True)
                if self.target_norm == "sum"
                else tf.reduce_max(concentrations, axis=-1, keepdims=True)
            )
            denominator = tf.debugging.check_numerics(denominator, "Non-finite model target denominator")
            tf.debugging.assert_positive(
                denominator, message="Model target normalisation requires a positive denominator"
            )
            concentrations = concentrations / denominator
        # Finite negative coefficients are retained; no prediction clipping.
        return concentrations

    def compute_output_shape(self, input_shape):
        """Return one coefficient per metabolite and input sample."""
        return (input_shape[0], len(self.metabolites))

    def get_config(self):
        """Record solver semantics so old checkpoints cannot be reinterpreted."""
        config = super().get_config()
        config.update(
            {
                "metabolites": self.metabolites,
                "basis_source": self.basis_source,
                "manufacturer": self.manufacturer,
                "omega": self.omega,
                "linewidth": self.linewidth,
                "pulse_sequence": self.pulse_sequence,
                "acquisitions": self.acquisitions,
                "n_freqs": self.n_freqs,
                "sample_rate": self.sample_rate,
                "samples": self.samples,
                "solver_version": self.solver_version,
                "high_ppm": self.high_ppm,
                "low_ppm": self.low_ppm,
                "target_norm": self.target_norm,
            }
        )
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
        if "_" in part and any(char.isdigit() for char in part):
            dataset_name = part
            break

    if dataset_name is None:
        raise ValueError(f"Could not extract dataset name from path: {dataset_path}")

    # Parse dataset name: source_sample_rate_samples
    name_parts = dataset_name.split("_")
    if len(name_parts) < 3:
        raise ValueError(f"Invalid dataset name format: {dataset_name}")

    source = "_".join(name_parts[:-2])  # Everything except last two parts
    sample_rate = int(name_parts[-2])
    samples = int(name_parts[-1])

    # Extract other parameters from path
    manufacturer = None
    omega = None
    linewidth = None
    metabolites = None
    pulse_sequence = None

    for i, part in enumerate(path_parts):
        if part in ["siemens", "ge", "philips"]:
            manufacturer = part
        elif part.replace(".", "").isdigit() and manufacturer is not None:
            if omega is None:
                omega = float(part)
            elif linewidth is None:
                linewidth = float(part)
        elif "-" in part and any(char.isupper() for char in part):
            metabolites = part.split("-")
        elif part in ["megapress", "press", "steam"]:
            pulse_sequence = part

    # Set defaults if not found
    if manufacturer is None:
        manufacturer = "siemens"
    if omega is None:
        omega = 123.23
    if linewidth is None:
        linewidth = 2.0
    if metabolites is None:
        metabolites = ["Cr", "GABA", "Glu", "Gln", "NAA"]
    if pulse_sequence is None:
        pulse_sequence = "megapress"

    return {
        "source": source,
        "manufacturer": manufacturer,
        "omega": omega,
        "linewidth": linewidth,
        "metabolites": metabolites,
        "pulse_sequence": pulse_sequence,
        "sample_rate": sample_rate,
        "samples": samples,
    }


def create_basis_lls_module(
    metabolites,
    basis_source="lcmodel",
    manufacturer="siemens",
    omega=123.23,
    linewidth=2.0,
    pulse_sequence="megapress",
    acquisitions=["edit_off", "difference"],
    n_freqs=2048,
    sample_rate=2000,
    samples=None,
    name="basis_lls",
    solver_version=2,
    high_ppm=-4.5,
    low_ppm=-1.0,
    target_norm="none",
):
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
        Larmor frequency in MHz. Defaults to 123.23
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
        name=name,
        solver_version=solver_version,
        high_ppm=high_ppm,
        low_ppm=low_ppm,
        target_norm=target_norm,
    )
