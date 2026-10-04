# SPDX-FileCopyrightText: Copyright (C) 2026 Frank C Langbein
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Robust fixed-composition linewidth fitting with a restrained magnitude baseline."""

import numpy as np
from scipy.optimize import least_squares

from mrsnet.linewidth import fit_basis_linewidth


def fit_robust_basis_linewidth(
    fid,
    observed,
    indices,
    sample_rate,
    base_linewidth,
    linewidth_bounds,
    shift_bounds=(-5.0, 5.0),
    mask=None,
    grid_points=(20, 11),
    noise_mask=None,
    noise_scale=None,
    baseline_limit=None,
):
    """Fit common broadening/offset/scale and a bounded constant baseline.

    Parameters
    ----------
    fid, observed, indices, sample_rate, base_linewidth, linewidth_bounds,
    shift_bounds, mask, grid_points
        As in fit_basis_linewidth; concentration ratios remain fixed.
    noise_mask : array_like of bool, optional
        Explicit low-signal bins inside the fit mask. Otherwise use bins below
        2 percent of the supplied basis magnitude maximum. Require 20 adjacent
        noise-bin pairs when estimating the noise proxy automatically. Low-signal
        coverage is optional when both noise_scale and baseline_limit are supplied.
    noise_scale : float, optional
        Positive magnitude-residual scale in observed units. Default is the
        adjacent-difference MAD proxy, with a 1e-6 relative numerical floor.
        This proxy is not a calibrated complex-noise standard deviation.
    baseline_limit : float, optional
        Positive upper limit in observed units. Default is the 95th percentile
        of low-signal magnitudes, with a 1e-6 relative numerical floor.

    Returns
    -------
    dict
        Parameters, robust objective, continuously refined offset profiles and
        explicit numerical/boundary/model-mismatch QC. Profile support is a
        diagnostic cost tolerance, not a confidence interval or a physical
        validation certificate. Legacy unrobust fitting remains available.

    Raises
    ------
    ValueError
        For invalid data or insufficient low-signal bins.
    """
    # Reuse the existing strict input checks and retain its start as one control.
    initial = fit_basis_linewidth(
        fid,
        observed,
        indices,
        sample_rate,
        base_linewidth,
        linewidth_bounds,
        shift_bounds=shift_bounds,
        mask=mask,
        grid_points=grid_points,
    )
    fid, observed, indices = np.asarray(fid, complex), np.asarray(observed, float), np.asarray(indices)
    mask = np.ones(observed.shape, bool) if mask is None else np.asarray(mask)
    reference = float(observed[mask].max())
    target = observed[mask] / reference
    time = np.arange(len(fid)) / sample_rate
    nominal = np.abs(np.fft.fftshift(np.fft.fft(fid)))[indices]
    basis_scale = float(nominal[mask].max())
    if basis_scale <= 0:
        raise ValueError("Non-zero basis required in the fit window")
    quiet = mask & (nominal <= 0.02 * basis_scale) if noise_mask is None else np.asarray(noise_mask)
    if quiet.shape != mask.shape or quiet.dtype.kind != "b" or np.any(quiet & ~mask):
        raise ValueError("Noise mask must select only fit-window bins")
    enough_quiet = np.count_nonzero(quiet) >= 20
    if not enough_quiet and (noise_scale is None or baseline_limit is None):
        raise ValueError("At least 20 low-signal bins are required")
    if noise_scale is None:
        pairs = quiet[1:] & quiet[:-1]
        if pairs.sum() < 20:
            raise ValueError("At least 20 adjacent noise-bin pairs are required")
        differences = np.diff(observed)[pairs]
        noise_scale = np.median(np.abs(differences - np.median(differences))) / (0.67448975 * np.sqrt(2))
        noise_scale = max(float(noise_scale), reference * 1e-6)
    if baseline_limit is None:
        baseline_limit = max(float(np.quantile(observed[quiet], 0.95)), reference * 1e-6)
    if not np.isfinite([noise_scale, baseline_limit]).all() or min(noise_scale, baseline_limit) <= 0:
        raise ValueError("Positive finite noise scale and baseline limit required")
    noise, cap = noise_scale / reference, baseline_limit / reference
    lower = np.array([linewidth_bounds[0], shift_bounds[0], 0.0, 0.0])
    upper = np.array([linewidth_bounds[1], shift_bounds[1], np.inf, cap])

    def residual(parameters):
        width, shift, amplitude, baseline = parameters
        exponent = -np.pi * (width - base_linewidth) * time + 2j * np.pi * shift * time
        magnitude = np.abs(np.fft.fftshift(np.fft.fft(fid * np.exp(exponent))))[indices][mask] / basis_scale
        return amplitude * magnitude + baseline - target

    def optimise(start, fixed_width=None):
        if fixed_width is None:
            function, lo, hi, x = residual, lower, upper, start
        else:
            function = lambda values: residual(np.r_[fixed_width, values])
            lo, hi, x = lower[1:], upper[1:], start[1:]
        return least_squares(
            function,
            np.clip(x, lo, hi),
            bounds=(lo, hi),
            loss="soft_l1",
            f_scale=noise,
            x_scale="jac",
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
            max_nfev=400,
        )

    def objective(result):
        return float(2 * result.cost / len(target))

    baseline = min(float(np.median(observed[quiet]) / reference), cap / 2) if enough_quiet else 0.0
    amplitude = max(initial["scale"] * basis_scale / reference, 1e-6)
    seed = np.array([initial["linewidth_hz"], initial["shift_hz"], amplitude, baseline])
    solutions = [optimise(seed)]
    # Distinct width basins, not the three adjacent best coarse-grid widths.
    for width in np.linspace(*linewidth_bounds, 5):
        start = seed.copy()
        start[0] = width
        solutions.append(optimise(start))
    best = min(solutions, key=objective)
    profile = []
    for width in np.linspace(*linewidth_bounds, int(grid_points[0])):
        candidates = []
        for shift in (shift_bounds[0], best.x[1], shift_bounds[1]):
            start = best.x.copy()
            start[1] = shift
            candidates.append(optimise(start, fixed_width=width))
        result = min(candidates, key=objective)
        profile.append(
            dict(
                linewidth_hz=float(width),
                shift_hz=float(result.x[0]),
                robust_objective=objective(result),
                optimiser_success=bool(result.success),
            )
        )
        solutions.append(optimise(np.r_[width, result.x]))
    best = min(solutions, key=objective)
    error = residual(best.x)
    boundary = np.any(
        np.isclose(best.x[:2, None], np.array([linewidth_bounds, shift_bounds]), rtol=0, atol=1e-5), axis=1
    )
    identifiable = bool(np.linalg.matrix_rank(best.jac) == 4)
    profile_min = min([objective(best), *[p["robust_objective"] for p in profile]])
    # One noise-scale squared is a declared cost-support diagnostic, not a CI.
    supported = [p["linewidth_hz"] for p in profile if p["robust_objective"] <= profile_min + noise**2]
    unresolved = bool(
        len(supported) > 1 and (supported[0] == linewidth_bounds[0] or supported[-1] == linewidth_bounds[1])
    )
    outlier_fraction = float(np.mean(np.abs(error) > 3 * noise))
    reasons = []
    for flag, reason in (
        (not best.success, "optimiser_failure"),
        (not identifiable, "rank_deficient"),
        (boundary[0], "linewidth_boundary"),
        (boundary[1], "shift_boundary"),
        (best.x[3] >= cap * (1 - 1e-5), "baseline_limit"),
        (unresolved, "profile_unresolved"),
        (outlier_fraction > 0.25, "model_mismatch"),
    ):
        if flag:
            reasons.append(reason)
    return dict(
        linewidth_hz=float(best.x[0]),
        added_linewidth_hz=float(best.x[0] - base_linewidth),
        base_linewidth_hz=float(base_linewidth),
        shift_hz=float(best.x[1]),
        scale=float(best.x[2] * reference / basis_scale),
        baseline=float(best.x[3] * reference),
        baseline_limit=float(baseline_limit),
        noise_scale=float(noise_scale),
        robust_objective=objective(best),
        relative_mse=float(np.mean(error**2)),
        accepted=not reasons,
        identifiable=identifiable,
        optimiser_success=bool(best.success),
        linewidth_at_boundary=bool(boundary[0]),
        shift_at_boundary=bool(boundary[1]),
        profile_unresolved=unresolved,
        residual_outlier_fraction=outlier_fraction,
        rejection_reasons=reasons,
        fit_bins=len(target),
        noise_bins=int(quiet.sum()),
        profile=profile,
        profile_cost_tolerance=float(noise**2),
        loss="soft_l1",
        estimator_version=2,
        uncertainty="diagnostic objective support only; not a confidence interval",
    )
