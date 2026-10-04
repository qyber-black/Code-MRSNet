# SPDX-FileCopyrightText: Copyright (C) 2026 Frank C Langbein
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure paired spectral-magnitude residuals without phase assumptions."""

import numpy as np


def magnitude_descriptors(phantom, reference, ppm, windows, exclusions=(), tau=0.02, min_bins=50):
    """Measure already scaled MEGAPRESS magnitude arrays.

    Parameters
    ----------
    phantom, reference : array_like
        Finite non-negative arrays [sample, acquisition, frequency], ordered
        difference, OFF, ON. Each OFF maximum must be one; acquisitions must
        share that spectrum's OFF scaling factor. No rescaling is performed.
    ppm : array_like
        Signed ppm coordinate of every frequency bin, in either direction.
    windows : mapping
        Descriptor names mapped to (centre_ppm, half_width_ppm, acquisition).
    exclusions : iterable of pairs
        Centre and half-width of regions excluded from the OFF baseline mask.
    tau : float
        Baseline bins have reference OFF magnitude below tau times its maximum.
    min_bins : int
        Required minimum number of baseline bins per sample.

    Returns
    -------
    list of dict
        Acquisition/overall MAE, baseline magnitude-residual power and window
        RMS magnitude residuals. Overall weights acquisitions and bins equally.

    Raises
    ------
    ValueError
        If paired arrays, coordinates, windows or baseline selections are invalid.
    """
    d, r = np.asarray(phantom), np.asarray(reference)
    nu = np.asarray(ppm)
    if (
        nu.ndim != 1
        or d.shape != r.shape
        or d.ndim != 3
        or d.shape[1:] != (3, len(nu))
        or not len(d)
        or not len(nu)
        or not np.isfinite(nu).all()
        or not np.isfinite(d).all()
        or not np.isfinite(r).all()
        or np.any(d < 0)
        or np.any(r < 0)
        or not np.isfinite(tau)
        or not 0 < tau <= 1
        or not isinstance(min_bins, int)
        or min_bins < 1
    ):
        raise ValueError("Invalid paired three-acquisition magnitude arrays")
    if not np.allclose(d[:, 1].max(axis=1), 1, rtol=0, atol=2e-7) or not np.allclose(
        r[:, 1].max(axis=1), 1, rtol=0, atol=2e-7
    ):
        raise ValueError("OFF magnitude must define the shared spectral scale")
    regions = list(exclusions)
    selections = {}
    for name, (center, width, acq) in windows.items():
        if not np.isfinite([center, width]).all() or width <= 0 or not isinstance(acq, int) or acq not in (0, 1, 2):
            raise ValueError("Invalid magnitude-residual window")
        win = np.abs(nu - center) <= width
        if not np.any(win):
            raise ValueError("Magnitude-residual window contains no frequency bins")
        selections[name] = (win, acq)
    for center, width in regions:
        if not np.isfinite([center, width]).all() or width <= 0:
            raise ValueError("Invalid baseline exclusion window")
    delta = d - r
    output = []
    for i in range(len(d)):
        mask = r[i, 1] < tau * r[i, 1].max()
        for center, width in regions:
            mask &= np.abs(nu - center) > width
        if mask.sum() < min_bins:
            raise ValueError("Too few signal-free OFF bins")
        row = dict(
            difference_mag_mae=float(np.abs(delta[i, 0]).mean()),
            off_mag_mae=float(np.abs(delta[i, 1]).mean()),
            on_mag_mae=float(np.abs(delta[i, 2]).mean()),
            overall_mag_mae=float(np.abs(delta[i]).mean()),
            baseline_mag_power=float(np.square(delta[i, 1, mask]).mean()),
            baseline_bins=int(mask.sum()),
        )
        for name, (win, acq) in selections.items():
            row["resid_mag_" + name] = float(np.sqrt(np.square(delta[i, acq, win]).mean()))
        output.append(row)
    return output
