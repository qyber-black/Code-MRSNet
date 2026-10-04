# SPDX-FileCopyrightText: Copyright (C) 2026 Frank C Langbein
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fit an effective simulator linewidth to a fixed-composition magnitude spectrum."""

import numpy as np
from scipy.optimize import least_squares


def fit_basis_linewidth(
    fid,
    observed,
    indices,
    sample_rate,
    base_linewidth,
    linewidth_bounds,
    shift_bounds=(-5.0, 5.0),
    mask=None,
    grid_points=(20, 11),
):
    """Profile amplitude scale and fit shared exponential broadening and shift.

    Parameters
    ----------
    fid : array_like
        Fixed-composition complex FID, already padded to the export FFT length.
    observed : array_like
        Non-negative observed magnitude at the selected exported FFT bins.
    indices : array_like
        Strictly increasing integer indices selecting those bins after fftshift.
    sample_rate : float
        FID sample rate in Hz.
    base_linewidth : float
        Lorentzian broadening already applied to the supplied FID, in Hz.
    linewidth_bounds : pair of float
        Total effective linewidth bounds in Hz; neither may narrow the basis.
    shift_bounds : pair of float
        Frequency-offset bounds in Hz on the increasing signed FFT axis.
    mask : array_like, optional
        Boolean fit-bin mask, excluding the water-attenuation region explicitly.
    grid_points : pair of int
        Coarse total-linewidth and offset grid sizes, including endpoints.

    Returns
    -------
    dict
        Fit parameters, acceptance/boundary flags and coarse objective profile.
        These are effective matching parameters, not peak FWHM or fit CIs.

    Raises
    ------
    ValueError
        If inputs, bounds or bin selections are invalid or have zero signal.
    """
    fid = np.asarray(fid, dtype=complex)
    observed = np.asarray(observed, dtype=float)
    indices = np.asarray(indices)
    mask = np.ones(observed.shape, dtype=bool) if mask is None else np.asarray(mask)
    bounds = np.asarray([linewidth_bounds, shift_bounds], dtype=float)
    counts = np.asarray(grid_points)
    if (
        fid.ndim != 1
        or len(fid) < 4
        or observed.ndim != 1
        or indices.shape != observed.shape
        or indices.dtype.kind not in "iu"
        or mask.shape != observed.shape
        or mask.dtype.kind != "b"
        or np.count_nonzero(mask) < 4
        or bounds.shape != (2, 2)
        or counts.shape != (2,)
        or counts.dtype.kind not in "iu"
        or np.any(counts < 3)
        or not np.isfinite(fid).all()
        or not np.isfinite(observed).all()
        or np.any(observed < 0)
        or not np.isfinite([sample_rate, base_linewidth]).all()
        or sample_rate <= 0
        or base_linewidth < 0
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] >= bounds[:, 1])
        or bounds[0, 0] < base_linewidth
        or bounds[0, 0] <= 0
        or len(indices) < 4
        or np.any(np.diff(indices.astype(np.int64)) <= 0)
        or indices[0] < 0
        or indices[-1] >= len(fid)
    ):
        raise ValueError("Invalid fixed-composition linewidth fit inputs")
    target = observed[mask]
    reference = float(np.max(target))
    if reference <= 0 or np.max(np.abs(fid)) <= 0:
        raise ValueError("Linewidth fit requires non-zero basis and observed signal")
    target = target / reference
    time = np.arange(len(fid), dtype=float) / sample_rate

    def residual(parameters):
        width, shift = parameters
        exponent = -np.pi * (width - base_linewidth) * time + 2j * np.pi * shift * time
        magnitude = np.abs(np.fft.fftshift(np.fft.fft(fid * np.exp(exponent))))[indices][mask]
        denominator = float(np.dot(magnitude, magnitude))
        if not np.isfinite(magnitude).all() or denominator <= 0:
            raise ValueError("Modulated basis has invalid or zero fit-bin magnitude")
        scale = max(0.0, float(np.dot(magnitude, target)) / denominator)
        return scale * magnitude - target, scale * reference

    profile = []
    starts = []
    for width in np.linspace(*bounds[0], int(counts[0])):
        candidates = []
        for shift in np.linspace(*bounds[1], int(counts[1])):
            error, scale = residual((width, shift))
            candidates.append((float(np.mean(error * error)), float(shift), scale))
        mse, shift, scale = min(candidates)
        profile.append(dict(linewidth_hz=float(width), shift_hz=shift, scale=scale, relative_mse=mse))
        starts.append((mse, float(width), shift))
    solutions = []
    # Multiple starts reduce sensitivity to local minima without claiming a global optimum.
    for _, width, shift in sorted(starts)[:3]:
        result = least_squares(
            lambda parameters: residual(parameters)[0],
            [width, shift],
            bounds=(bounds[:, 0], bounds[:, 1]),
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=200,
        )
        error, scale = residual(result.x)
        solutions.append((float(np.mean(error * error)), result, scale))
    mse, result, scale = min(solutions, key=lambda entry: entry[0])
    identifiable = bool(np.linalg.matrix_rank(result.jac) == 2)
    accepted = bool(result.success and identifiable and scale > 0)
    width, shift = result.x
    boundary = np.any(np.isclose(result.x[:, None], bounds, rtol=0, atol=1e-5), axis=1)
    return dict(
        linewidth_hz=float(width),
        added_linewidth_hz=float(width - base_linewidth),
        base_linewidth_hz=float(base_linewidth),
        shift_hz=float(shift),
        scale=float(scale),
        relative_mse=mse,
        accepted=accepted,
        identifiable=identifiable,
        optimiser_success=bool(result.success),
        linewidth_at_boundary=bool(boundary[0]),
        shift_at_boundary=bool(boundary[1]),
        fit_bins=int(np.count_nonzero(mask)),
        profile=profile,
        uncertainty="none; objective profile is not a confidence interval",
    )
