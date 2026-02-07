# mrsnet/compare.py - MRSNet - compare and analyse spectra
#
# SPDX-FileCopyrightText: Copyright (C) 2021-2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2022-2025 Zien Ma, PhD student at Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Spectrum comparison utilities for MRSNet.

This module provides functions for comparing datasets with basis spectra
and analyzing differences between experimental and simulated data.
"""

import copy
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

import mrsnet.molecules as molecules
from mrsnet.dataset import Dataset


def compare_basis(ds, basis, high_ppm=-4.5, low_ppm=-1, n_fft_pts=2048, verbose=0, screen_dpi=96,
                 out_dir=None, save_prefix=None,
                 noise_mc_trials=0, noise_sigma=0.03, noise_mu=0.0,
                 individual_linewidths=None, overall_linewidth=None,
                 extra_error_trials=None):
  """Compare dataset spectra to spectra generated from basis.

  Generates reference spectra from a basis set using the dataset's concentrations
  and compares them with the actual dataset spectra.

  Parameters
  ----------
      ds (Dataset): Dataset to compare
      basis (Basis): Basis set for generating reference spectra
      high_ppm (float, optional): Upper PPM bound. Defaults to -4.5
      low_ppm (float, optional): Lower PPM bound. Defaults to -1
      n_fft_pts (int, optional): Number of FFT points. Defaults to 2048
      verbose (int, optional): Verbosity level. Defaults to 0
      screen_dpi (int, optional): DPI for screen display. Defaults to 96
      out_dir (str, optional): Directory to save plots/metrics. Defaults to None
      save_prefix (str, optional): Filename prefix for saved artifacts. Defaults to basis parameters
  """
  # Compare dataset to spectra generated from basis
  # Setup basis - handle both single basis and list of bases
  if verbose > 0:
    print("# Preparing reference spectra from basis")
  ref_spectra = Dataset("Basis Spectra")
  diff = 0.0

  # Check if basis is a list (individual bases) or single basis
  if isinstance(basis, list):
    # Individual bases for each spectrum
    if len(basis) != len(ds.concentrations):
      raise ValueError(f"Number of individual bases ({len(basis)}) must match number of spectra ({len(ds.concentrations)})")

    for l in range(len(ds.concentrations)):
      if verbose > 3:
        basis[l].plot('magnitude','fft')
        plt.show(block=True)
        plt.close()
      s,c = basis[l].combine(ds.concentrations[l],"ref_"+ds.spectra[l]["edit_off"].id)
      ref_spectra.spectra.append(s)
      ref_spectra.concentrations.append(c)
      for m in ds.metabolites:
        diff += np.abs(c[m] - ds.concentrations[l][m])
  else:
    # Single basis for all spectra
    if verbose > 3:
      basis.plot('magnitude','fft')
      plt.show(block=True)
      plt.close()
    for l in range(len(ds.concentrations)):
      s,c = basis.combine(ds.concentrations[l],"ref_"+ds.spectra[l]["edit_off"].id)
      ref_spectra.spectra.append(s)
      ref_spectra.concentrations.append(c)
      for m in ds.metabolites:
        diff += np.abs(c[m] - ds.concentrations[l][m])

  if diff > 1e-12:
    print(f"Warning, difference in concentrations between dataset and reference: {diff}")

  if verbose > 0:
    print("# Converting reference and dataset spectra")
  r_inp, r_out = ref_spectra.export(metabolites=ds.metabolites, norm='max',
                                    acquisitions=['difference','edit_off','edit_on'],
                                    datatype=['magnitude','phase','real','imaginary'],
                                    high_ppm=high_ppm, low_ppm=low_ppm, n_fft_pts=n_fft_pts,
                                    verbose=verbose)
  d_inp, _ = ds.export(metabolites=ds.metabolites, norm='max',
                       acquisitions=['difference','edit_off','edit_on'],
                       datatype=['magnitude','phase','real','imaginary'],
                       high_ppm=high_ppm, low_ppm=low_ppm, n_fft_pts=n_fft_pts,
                       export_concentrations=False, verbose=verbose)
  nu = np.linspace(high_ppm, low_ppm, r_inp.shape[-1])
  # Full signed error tensor [samples, acq, dtype, freq]
  err_full = r_inp - d_inp

  all_diff = np.ndarray((r_inp.shape[1],r_inp.shape[2],r_inp.shape[3]*r_inp.shape[0]))
  all_ref = np.ndarray((r_inp.shape[1],r_inp.shape[2],r_inp.shape[3]*r_inp.shape[0]))
  all_dat = np.ndarray((r_inp.shape[1],r_inp.shape[2],r_inp.shape[3]*r_inp.shape[0]))
  for l in range(len(ds.spectra)):
    diff = err_full[l,:,:,:]
    all_diff[:,:,l*r_inp.shape[3]:(l+1)*r_inp.shape[3]] = diff
    all_ref[:,:,l*r_inp.shape[3]:(l+1)*r_inp.shape[3]] = r_inp[l,:,:,:]
    all_dat[:,:,l*r_inp.shape[3]:(l+1)*r_inp.shape[3]] = d_inp[l,:,:,:]
    # Add linewidth info to header if available
    lw_info = ""
    if individual_linewidths is not None and l < len(individual_linewidths) and individual_linewidths[l] is not None:
      lw_info = f" (LW: {individual_linewidths[l]:.1f} Hz)"
    print(f"## Spectra differences {l}: {ds.spectra[l][next(iter(ds.spectra[l].keys()))].id}{lw_info}")
    dd = np.sum(np.abs(diff),axis=2) / diff.shape[2]
    m = np.mean(diff, axis=2)
    s = np.std(diff, axis=2)
    print("                    %12s %12s %12s" % ("MAE", "Mean", "Std"))  # noqa: UP031
    print(f"Diff     Magnitude: {dd[0,0]:12f} {m[0,0]:12f} {s[0,0]:12f}")
    print(f"             Phase: {dd[0,1]:12f} {m[0,1]:12f} {s[0,1]:12f}")
    print(f"              Real: {dd[0,2]:12f} {m[0,2]:12f} {s[0,2]:12f}")
    print(f"         Imaginary: {dd[0,3]:12f} {m[0,3]:12f} {s[0,3]:12f}")
    print(f"Edit_Off Magnitude: {dd[1,0]:12f} {m[1,0]:12f} {s[1,0]:12f}")
    print(f"             Phase: {dd[1,1]:12f} {m[1,1]:12f} {s[1,1]:12f}")
    print(f"              Real: {dd[1,2]:12f} {m[1,2]:12f} {s[1,2]:12f}")
    print(f"         Imaginary: {dd[1,3]:12f} {m[1,3]:12f} {s[1,3]:12f}")
    print(f"Edit_On  Magnitude: {dd[2,0]:12f} {m[2,0]:12f} {s[2,0]:12f}")
    print(f"             Phase: {dd[2,1]:12f} {m[2,1]:12f} {s[2,1]:12f}")
    print(f"              Real: {dd[2,2]:12f} {m[2,2]:12f} {s[2,2]:12f}")
    print(f"         Imaginary: {dd[2,3]:12f} {m[2,3]:12f} {s[2,3]:12f}")
    if verbose >= 2:
      # Determine source string per spectrum when basis is a list
      src_str = basis.source if not isinstance(basis, list) else basis[l].source
      plot_diff_spectra(r_inp[l,:,:,:],d_inp[l,:,:,:],r_out[l,:],nu,
                        ds.metabolites,src_str,screen_dpi)
      plt.show(block=True)
      plt.close()

  # Add overall linewidth info to header if available
  lw_info = ""
  if overall_linewidth is not None:
    lw_info = f" (Mean LW: {overall_linewidth:.1f} Hz)"
  print(f"# Differences over all spectra (max normalised to 1){lw_info}")
  dd = np.sum(np.abs(all_diff),axis=2) / all_diff.shape[2]
  m = np.mean(all_diff, axis=2)
  s = np.std(all_diff, axis=2)
  rm = np.sqrt(np.mean(all_diff**2, axis=2))
  corr = np.zeros_like(dd)
  for k in range(all_ref.shape[0]):
    for d in range(all_ref.shape[1]):
      x = all_ref[k,d,:]
      y = all_dat[k,d,:]
      sx = np.std(x)
      sy = np.std(y)
      if sx > 0.0 and sy > 0.0:
        corr[k,d] = np.corrcoef(x, y)[0,1]
      else:
        corr[k,d] = np.nan
  # Cosine similarity per acquisition/datatype
  cos = np.zeros_like(dd)
  for k in range(all_ref.shape[0]):
    for d in range(all_ref.shape[1]):
      x = all_ref[k,d,:]
      y = all_dat[k,d,:]
      nx = np.linalg.norm(x)
      ny = np.linalg.norm(y)
      if nx > 0.0 and ny > 0.0:
        cos[k,d] = float(np.dot(x, y) / (nx * ny))
      else:
        cos[k,d] = np.nan
  print("                    %12s %12s %12s %12s %12s" % ("MAE", "RMSE", "Mean", "Std", "Corr"))  # noqa: UP031
  print(f"Diff     Magnitude: {dd[0,0]:12f} {rm[0,0]:12f} {m[0,0]:12f} {s[0,0]:12f} {corr[0,0]:12.6f}")
  print(f"             Phase: {dd[0,1]:12f} {rm[0,1]:12f} {m[0,1]:12f} {s[0,1]:12f} {corr[0,1]:12.6f}")
  print(f"              Real: {dd[0,2]:12f} {rm[0,2]:12f} {m[0,2]:12f} {s[0,2]:12f} {corr[0,2]:12.6f}")
  print(f"         Imaginary: {dd[0,3]:12f} {rm[0,3]:12f} {m[0,3]:12f} {s[0,3]:12f} {corr[0,3]:12.6f}")
  print(f"Edit_Off Magnitude: {dd[1,0]:12f} {rm[1,0]:12f} {m[1,0]:12f} {s[1,0]:12f} {corr[1,0]:12.6f}")
  print(f"             Phase: {dd[1,1]:12f} {rm[1,1]:12f} {m[1,1]:12f} {s[1,1]:12f} {corr[1,1]:12.6f}")
  print(f"              Real: {dd[1,2]:12f} {rm[1,2]:12f} {m[1,2]:12f} {s[1,2]:12f} {corr[1,2]:12.6f}")
  print(f"         Imaginary: {dd[1,3]:12f} {rm[1,3]:12f} {m[1,3]:12f} {s[1,3]:12f} {corr[1,3]:12.6f}")
  print(f"Edit_On  Magnitude: {dd[2,0]:12f} {rm[2,0]:12f} {m[2,0]:12f} {s[2,0]:12f} {corr[2,0]:12.6f}")
  print(f"             Phase: {dd[2,1]:12f} {rm[2,1]:12f} {m[2,1]:12f} {s[2,1]:12f} {corr[2,1]:12.6f}")
  print(f"              Real: {dd[2,2]:12f} {rm[2,2]:12f} {m[2,2]:12f} {s[2,2]:12f} {corr[2,2]:12.6f}")
  print(f"         Imaginary: {dd[2,3]:12f} {rm[2,3]:12f} {m[2,3]:12f} {s[2,3]:12f} {corr[2,3]:12.6f}")

  # Signed error averaged across samples for plotting and return
  signed_spectrum = np.mean((r_inp - d_inp), axis=0)

  # Always save summary plots and metrics
  if out_dir is not None:
    os.makedirs(out_dir, exist_ok=True)
    if save_prefix is None:
      rep_basis = basis[0] if isinstance(basis, list) and len(basis) > 0 else basis
      save_prefix = f"{rep_basis.source}_{rep_basis.manufacturer}_{rep_basis.omega}_{rep_basis.linewidth}_{rep_basis.pulse_sequence}_{rep_basis.sample_rate}_{rep_basis.samples}"
    # Combined signed error + histogram plots per representation (magnitude/real/imaginary)
    # Plot for Magnitude (dtype 0)
    fig_mag = plot_mae_hist_combo(signed_spectrum, nu, all_diff, 0, screen_dpi,
                                  noise_trials=None, dtype_name="Magnitude")
    fig_mag.savefig(os.path.join(out_dir, f"{save_prefix}_error_mag.png"), dpi=300)
    plt.close(fig_mag)
    # Plot for Phase (dtype 1)
    fig_phase = plot_mae_hist_combo(signed_spectrum, nu, all_diff, 1, screen_dpi,
                                    noise_trials=None, dtype_name="Phase")
    fig_phase.savefig(os.path.join(out_dir, f"{save_prefix}_error_phase.png"), dpi=300)
    plt.close(fig_phase)
    # Plot for Real (dtype 2)
    fig_real = plot_mae_hist_combo(signed_spectrum, nu, all_diff, 2, screen_dpi,
                                   noise_trials=None, dtype_name="Real")
    fig_real.savefig(os.path.join(out_dir, f"{save_prefix}_error_real.png"), dpi=300)
    plt.close(fig_real)
    # Plot for Imaginary (dtype 3)
    fig_imag = plot_mae_hist_combo(signed_spectrum, nu, all_diff, 3, screen_dpi,
                                   noise_trials=None, dtype_name="Imaginary")
    fig_imag.savefig(os.path.join(out_dir, f"{save_prefix}_error_imag.png"), dpi=300)
    plt.close(fig_imag)
    # Peak stats on data spectra (edit_off magnitude)
    peak_stats = compute_peak_stats(d_inp, nu)

    # Optional Monte Carlo noise analysis on reference spectra
    noise_info = None
    band_trials = None
    if isinstance(noise_mc_trials, int) and noise_mc_trials > 0 and (noise_sigma > 0.0 or noise_mu > 0.0):
      # Accumulators over trials
      dd_trials = []
      rm_trials = []
      mean_trials = []
      std_trials = []
      corr_trials = []
      cos_trials = []
      # MAE spectrum uncertainty per trial for all acq/dtypes
      err_spec_trials_all = []

      for _t in range(noise_mc_trials):
        # Create noisy copy of reference spectra
        mc_specs = []
        # Draw per-sample noise parameters
        mu_vec = np.random.uniform(0.0, noise_mu, len(ref_spectra.spectra))
        sg_vec = np.random.uniform(0.0, noise_sigma, len(ref_spectra.spectra))
        idx_sample = 0
        for sdict in ref_spectra.spectra:
          sc = {a: copy.deepcopy(sdict[a]) for a in sdict}
          # Add ADC normal noise to edit_off and edit_on, then recompute difference
          sc['edit_off'].add_noise_adc_normal(mu=mu_vec[idx_sample], sigma=sg_vec[idx_sample])
          sc['edit_on'].add_noise_adc_normal(mu=mu_vec[idx_sample], sigma=sg_vec[idx_sample])
          sc['difference'] = sc['edit_on'].__class__.comb(1.0, sc['edit_on'], -1.0, sc['edit_off'],
                                                         sc['edit_on'].id+"_+_"+sc['edit_off'].id,
                                                         "difference")
          sc['edit_off'].__class__.correct_b0_multi(sc)
          mc_specs.append(sc)
          idx_sample += 1
        mc_ds = Dataset("Basis Spectra MC")
        mc_ds.spectra = mc_specs
        mc_r_inp, _ = mc_ds.export(metabolites=ds.metabolites, norm='max',
                                   acquisitions=['difference','edit_off','edit_on'],
                                   datatype=['magnitude','phase','real','imaginary'],
                                   high_ppm=high_ppm, low_ppm=low_ppm, n_fft_pts=n_fft_pts,
                                   export_concentrations=False, verbose=0)
        diff_t = mc_r_inp - d_inp   # [samples, acq, dtype, freq]
        # Aggregate to [acq, dtype]
        dd_t = np.mean(np.sum(np.abs(diff_t), axis=3) / diff_t.shape[3], axis=0)
        rm_t = np.sqrt(np.mean(np.mean(diff_t**2, axis=3), axis=0))
        m_t = np.mean(np.mean(diff_t, axis=3), axis=0)
        s_t = np.std(diff_t, axis=(0,3))
        # Corr and cosine
        corr_t = np.zeros((mc_r_inp.shape[1], mc_r_inp.shape[2]))
        cos_t = np.zeros((mc_r_inp.shape[1], mc_r_inp.shape[2]))
        for k in range(mc_r_inp.shape[1]):
          for dch in range(mc_r_inp.shape[2]):
            x = mc_r_inp[:,k,dch,:].reshape(-1)
            y = d_inp[:,k,dch,:].reshape(-1)
            sx = np.std(x)
            sy = np.std(y)
            if sx > 0.0 and sy > 0.0:
              corr_t[k,dch] = np.corrcoef(x, y)[0,1]
            else:
              corr_t[k,dch] = np.nan
            nx = np.linalg.norm(x)
            ny = np.linalg.norm(y)
            if nx > 0.0 and ny > 0.0:
              cos_t[k,dch] = float(np.dot(x, y) / (nx * ny))
            else:
              cos_t[k,dch] = np.nan
        dd_trials.append(dd_t)
        rm_trials.append(rm_t)
        mean_trials.append(m_t)
        std_trials.append(s_t)
        corr_trials.append(corr_t)
        cos_trials.append(cos_t)
        # Store per-trial MAE spectra [acq, dtype, freq]
        err_spec_trials_all.append(np.mean((mc_r_inp - d_inp), axis=0))

      dd_trials = np.array(dd_trials)
      rm_trials = np.array(rm_trials)
      mean_trials = np.array(mean_trials)
      std_trials = np.array(std_trials)
      corr_trials = np.array(corr_trials)
      cos_trials = np.array(cos_trials)
      err_spec_trials_all = np.array(err_spec_trials_all)  # [trials, acq, dtype, freq]

      noise_info = {
        "trials": int(noise_mc_trials),
        "sigma_max": float(noise_sigma),
        "mu_max": float(noise_mu),
        "mae_mean": np.nanmean(dd_trials, axis=0).tolist(),
        "mae_std": np.nanstd(dd_trials, axis=0).tolist(),
        "rmse_mean": np.nanmean(rm_trials, axis=0).tolist(),
        "rmse_std": np.nanstd(rm_trials, axis=0).tolist(),
        "mean_mean": np.nanmean(mean_trials, axis=0).tolist(),
        "mean_std": np.nanstd(mean_trials, axis=0).tolist(),
        "std_mean": np.nanmean(std_trials, axis=0).tolist(),
        "std_std": np.nanstd(std_trials, axis=0).tolist(),
        "corr_mean": np.nanmean(corr_trials, axis=0).tolist(),
        "corr_std": np.nanstd(corr_trials, axis=0).tolist(),
        "cosine_mean": np.nanmean(cos_trials, axis=0).tolist(),
        "cosine_std": np.nanstd(cos_trials, axis=0).tolist()
      }
      band_trials = err_spec_trials_all

    # If external error trials are provided (e.g., linewidth MC), render bands too
    if extra_error_trials is not None:
      try:
        import numpy as _np
        if band_trials is None:
          band_trials = _np.asarray(extra_error_trials)
        else:
          band_trials = _np.concatenate([_np.asarray(band_trials), _np.asarray(extra_error_trials)], axis=0)
      except Exception:
        pass

    if band_trials is not None:
      fig_mag = plot_mae_hist_combo(signed_spectrum, nu, all_diff, 0, screen_dpi,
                                    noise_trials=band_trials, dtype_name="Magnitude")
      fig_mag.savefig(os.path.join(out_dir, f"{save_prefix}_error_mag.png"), dpi=300)
      plt.close(fig_mag)
      fig_phase = plot_mae_hist_combo(signed_spectrum, nu, all_diff, 1, screen_dpi,
                                      noise_trials=band_trials, dtype_name="Phase")
      fig_phase.savefig(os.path.join(out_dir, f"{save_prefix}_error_phase.png"), dpi=300)
      plt.close(fig_phase)
      fig_real = plot_mae_hist_combo(signed_spectrum, nu, all_diff, 2, screen_dpi,
                                     noise_trials=band_trials, dtype_name="Real")
      fig_real.savefig(os.path.join(out_dir, f"{save_prefix}_error_real.png"), dpi=300)
      plt.close(fig_real)
      fig_imag = plot_mae_hist_combo(signed_spectrum, nu, all_diff, 3, screen_dpi,
                                     noise_trials=band_trials, dtype_name="Imaginary")
      fig_imag.savefig(os.path.join(out_dir, f"{save_prefix}_error_imag.png"), dpi=300)
      plt.close(fig_imag)

    # Save metrics JSON
    acq_names = ["difference","edit_off","edit_on"]
    dt_names = ["magnitude","phase","real","imaginary"]
    metrics = {}
    for ai, an in enumerate(acq_names):
      metrics[an] = {}
      for di, dn in enumerate(dt_names):
        metrics[an][dn] = {
          "mae": float(dd[ai,di]),
          "rmse": float(rm[ai,di]),
          "mean": float(m[ai,di]),
          "std": float(s[ai,di]),
          "corr": None if np.isnan(corr[ai,di]) else float(corr[ai,di]),
          "cosine": None if np.isnan(cos[ai,di]) else float(cos[ai,di])
        }
    # Build summary block
    summary = {
      "mae_overall": float(np.nanmean(dd)),
      "rmse_overall": float(np.nanmean(rm)),
      "mean_overall": float(np.nanmean(m)),
      "std_overall": float(np.nanmean(s)),
      "corr_overall": float(np.nanmean(corr)),
      "cosine_overall": float(np.nanmean(cos))
    }

    with open(os.path.join(out_dir, f"{save_prefix}_metrics.json"), "w") as f:
      rep_basis = basis[0] if isinstance(basis, list) and len(basis) > 0 else basis
      json.dump({
        "basis": {
          "source": rep_basis.source,
          "manufacturer": rep_basis.manufacturer,
          "omega": rep_basis.omega,
          "linewidth": rep_basis.linewidth,
          "pulse_sequence": rep_basis.pulse_sequence,
          "sample_rate": rep_basis.sample_rate,
          "samples": rep_basis.samples
        },
        "linewidth": {
          "individual": None if individual_linewidths is None else [None if v is None else float(v) for v in individual_linewidths],
          "overall": None if overall_linewidth is None else float(overall_linewidth)
        },
        "n_samples": int(r_inp.shape[0]),
        "n_fft_pts": int(r_inp.shape[3]),
        "summary": summary,
        "metrics": metrics,
        "peak_stats": peak_stats,
        "noise": noise_info
      }, f, indent=2)

  # Return metrics for optional aggregation by callers
  return {
    "n_samples": int(r_inp.shape[0]),
    "n_fft_pts": int(r_inp.shape[3]),
    "sum_abs_diff": np.sum(np.abs(all_diff), axis=2),
    "sum_sq_diff": np.sum(all_diff**2, axis=2),
    "sum_diff": np.sum(all_diff, axis=2),
    "sum_ref": np.sum(all_ref, axis=2),
    "sum_dat": np.sum(all_dat, axis=2),
    "sum_ref_sq": np.sum(all_ref**2, axis=2),
    "sum_dat_sq": np.sum(all_dat**2, axis=2),
    "sum_ref_dat": np.sum(all_ref*all_dat, axis=2),
    "mae": dd,
    "rmse": rm,
    "mean": m,
    "std": s,
    "corr": corr,
    "cosine": cos,
    "mae_spectrum": np.mean(np.abs(r_inp - d_inp), axis=0),
    "signed_spectrum": signed_spectrum,
    "nu": nu,
    "peak_stats": compute_peak_stats(d_inp, nu),
    "summary": {
      "mae_overall": float(np.nanmean(dd)),
      "rmse_overall": float(np.nanmean(rm)),
      "mean_overall": float(np.nanmean(m)),
      "std_overall": float(np.nanmean(s)),
      "corr_overall": float(np.nanmean(corr)),
      "cosine_overall": float(np.nanmean(cos))
    }
  }

def plot_diff_spectra(r, d, c, nu, metabolites, source, screen_dpi):
    """Plot difference between dataset and basis reference spectrum.

    Parameters
    ----------
        r: Reference spectrum data
        d: Dataset spectrum data
        c: Concentration data
        nu: Frequency data
        metabolites (list): List of metabolite names
        source (str): Data source identifier
        screen_dpi (int): Screen DPI for plotting
    """
    # Plot difference between dataset and basis reference spectrum
    figure, axes = plt.subplots(4, 4, sharex=True, dpi=screen_dpi)
    axes[0,3].remove()
    axes[1,3].remove()
    axes[2,3].remove()
    axes[3,3].remove()

    plt.suptitle(f"Difference between Data (blue) and {source.upper()} Reference (orange)")

    axes[0,0].set_title("Difference")
    axes[0,0].plot(nu,d[0,0,:])
    axes[0,0].plot(nu,r[0,0,:],linewidth=.75)
    axes[0,0].set_ylabel("Magnitude")
    axes[1,0].plot(nu,d[0,1,:])
    axes[1,0].plot(nu,r[0,1,:],linewidth=.75)
    axes[1,0].set_ylabel("Phase")
    axes[2,0].plot(nu,d[0,2,:])
    axes[2,0].plot(nu,r[0,2,:],linewidth=.75)
    axes[2,0].set_ylabel("Real")
    axes[3,0].plot(nu,d[0,3,:])
    axes[3,0].plot(nu,r[0,3,:],linewidth=.75)
    axes[3,0].set_ylabel("Imaginary")
    axes[3,0].set_xlabel("Frequency (ppm)")
    axes[3,0].xaxis.set_major_formatter(FuncFormatter(lambda x_val, tick_pos: f"{np.abs(x_val):.8g}"))

    axes[0,1].set_title("Edit Off")
    axes[0,1].plot(nu,d[1,0,:])
    axes[0,1].plot(nu,r[1,0,:],linewidth=.75)
    axes[0,0].get_shared_y_axes().join(axes[0,0], axes[0,1])
    axes[1,1].plot(nu,d[1,1,:])
    axes[1,1].plot(nu,r[1,1,:],linewidth=.75)
    axes[1,0].get_shared_y_axes().join(axes[1,0], axes[1,1])
    axes[2,1].plot(nu,d[1,2,:])
    axes[2,1].plot(nu,r[1,2,:],linewidth=.75)
    axes[2,0].get_shared_y_axes().join(axes[2,0], axes[2,1])
    axes[3,1].plot(nu,d[1,3,:])
    axes[3,1].plot(nu,r[1,3,:],linewidth=.75)
    axes[3,0].get_shared_y_axes().join(axes[3,0], axes[3,1])
    axes[3,1].set_xlabel("Frequency (ppm)")
    axes[3,1].xaxis.set_major_formatter(FuncFormatter(lambda x_val, tick_pos: f"{np.abs(x_val):.8g}"))

    axes[0,2].set_title("Edit On")
    axes[0,2].plot(nu,d[2,0,:])
    axes[0,2].plot(nu,r[2,0,:],linewidth=.75)
    axes[0,0].get_shared_y_axes().join(axes[0,0], axes[0,2])
    axes[1,2].plot(nu,d[2,1,:])
    axes[1,2].plot(nu,r[2,1,:],linewidth=.75)
    axes[1,0].get_shared_y_axes().join(axes[1,0], axes[1,2])
    axes[2,2].plot(nu,d[2,2,:])
    axes[2,2].plot(nu,r[2,2,:],linewidth=.75)
    axes[2,0].get_shared_y_axes().join(axes[2,0], axes[2,2])
    axes[3,2].plot(nu,d[2,3,:])
    axes[3,2].plot(nu,r[2,3,:],linewidth=.75)
    axes[3,0].get_shared_y_axes().join(axes[3,0], axes[3,2])
    axes[3,2].set_xlabel("Frequency (ppm)")
    axes[3,2].xaxis.set_major_formatter(FuncFormatter(lambda x_val, tick_pos: f"{np.abs(x_val):.8g}"))

    ax = plt.subplot(1, 4, 4)
    plt.title('Concentrations')
    ax.bar(np.linspace(0, c.shape[0] - 1, c.shape[0]), c)
    ax.set_xticks(np.arange(len(metabolites)))
    ax.set_xticklabels(molecules.short_name(metabolites))

    return figure

def plot_mae_spectrum(mae_spectrum, nu, screen_dpi):
    """Plot mean absolute error spectrum across all spectra.

    Parameters
    ----------
        mae_spectrum: numpy.ndarray of shape [acquisition, datatype, frequency]
        nu: numpy.ndarray ppm axis
        screen_dpi: int

    Returns
    -------
        matplotlib.figure.Figure
    """
    figure, axes = plt.subplots(4, 3, sharex=True, dpi=screen_dpi)
    plt.suptitle("Mean Absolute Error across Spectra")

    labels_dtype = ["Magnitude", "Phase", "Real", "Imaginary"]
    labels_acq = ["Difference", "Edit Off", "Edit On"]

    for acq in range(0,3):
      for dt in range(0,4):
        axes[dt, acq].plot(nu, mae_spectrum[acq, dt, :])
        if acq == 0:
          axes[dt, acq].set_ylabel(labels_dtype[dt])
        if dt == 3:
          axes[dt, acq].set_xlabel("Frequency (ppm)")
        axes[dt, acq].set_ylabel("MAE (normalized)") if acq == 0 else None
        axes[dt, acq].xaxis.set_major_formatter(FuncFormatter(lambda x_val, tick_pos: f"{np.abs(x_val):.8g}"))
        if dt == 0:
          axes[dt, acq].set_title(labels_acq[acq])
    return figure

def compute_peak_stats(d_inp, nu):
  """Compute basic peak stats (FWHM, drift, SNR) for NAA and Cr on edit_off magnitude.

  Parameters
  ----------
      d_inp: numpy array [samples, acquisition, datatype, freq]
      nu: ppm axis

  Returns
  -------
      dict with keys 'NAA', 'Cr' each containing fwhm_mean, fwhm_std,
      drift_mean_ppm, drift_std_ppm, snr_mean, snr_std, and n (count).
  """
  def series_stats(target_ppm, window_ppm=0.2, excl_ppm=0.1):
    sig = d_inp[:,1,0,:]  # edit_off, magnitude
    fwhms = []
    drifts = []
    snrs = []
    # Build noise mask excluding peak neighborhoods of NAA and Cr
    mask_noise = np.ones_like(nu, dtype=bool)
    for loc in [-2.01, -3.015]:
      mask_noise &= np.abs(nu - loc) > excl_ppm
    noise_std = np.std(sig[:,mask_noise], axis=1)
    for l in range(sig.shape[0]):
      vals = sig[l,:]
      # Find peak within window
      mask = np.abs(nu - target_ppm) <= window_ppm
      if not np.any(mask):
        continue
      idx_local = np.argmax(vals[mask])
      nu_local = nu[mask]
      val_local = vals[mask]
      peak_ppm = nu_local[idx_local]
      peak_val = val_local[idx_local]
      # Drift
      drifts.append(float(peak_ppm - target_ppm))
      # FWHM
      half = peak_val / 2.0
      # Left crossing
      li = idx_local
      while li > 0 and val_local[li] > half:
        li -= 1
      if li == idx_local:
        left_ppm = peak_ppm
      else:
        x0, y0 = nu_local[li], val_local[li]
        x1, y1 = nu_local[li+1], val_local[li+1]
        if y1 != y0:
          left_ppm = x0 + (half - y0) * (x1 - x0) / (y1 - y0)
        else:
          left_ppm = x0
      # Right crossing
      ri = idx_local
      while ri < len(val_local)-1 and val_local[ri] > half:
        ri += 1
      if ri == idx_local:
        right_ppm = peak_ppm
      else:
        x0, y0 = nu_local[ri-1], val_local[ri-1]
        x1, y1 = nu_local[ri], val_local[ri]
        if y1 != y0:
          right_ppm = x0 + (half - y0) * (x1 - x0) / (y1 - y0)
        else:
          right_ppm = x1
      fwhms.append(float(abs(right_ppm - left_ppm)))
      # SNR
      if noise_std[l] > 0:
        snrs.append(float(peak_val / noise_std[l]))
    def agg(vs):
      if len(vs) == 0:
        return (None, None)
      arr = np.array(vs, dtype=float)
      return (float(np.nanmean(arr)), float(np.nanstd(arr)))
    f_m, f_s = agg(fwhms)
    d_m, d_s = agg(drifts)
    s_m, s_s = agg(snrs)
    return {"fwhm_mean": f_m, "fwhm_std": f_s,
            "drift_mean_ppm": d_m, "drift_std_ppm": d_s,
            "snr_mean": s_m, "snr_std": s_s,
            "n": len(drifts)}

  return {
    "NAA": series_stats(-2.01),
    "Cr": series_stats(-3.015)
  }

def plot_mae_hist_combo(signed_spectrum, nu, all_diff, dtype_idx, screen_dpi, noise_trials=None, dtype_name="Magnitude"):
  """Plot signed error per frequency and histogram per acquisition for a given datatype.

  Parameters
  ----------
      signed_spectrum: ndarray [acq, dtype, freq]
      nu: ppm axis
      all_diff: ndarray flattened signed diffs [acq, dtype, samples*freq]
      dtype_idx: 0 Magnitude, 2 Real, 3 Imaginary
      screen_dpi: int
      noise_trials: optional ndarray [trials, acq, dtype, freq] for signed errors
      dtype_name: str label
  """
  fig, axs = plt.subplots(3, 3, dpi=screen_dpi)
  acq_titles = ["Difference", "Edit Off", "Edit On"]
  # Left column: MAE vs frequency (with noise bands if available)
  # Middle column: histogram of signed error
  # Right column: 2D correlation heatmap (frequency vs signed error)
  for acq in range(3):
    ax_l = axs[acq, 0]
    ax_m = axs[acq, 1]
    ax_r = axs[acq, 2]
    y = signed_spectrum[acq, dtype_idx, :]
    ax_l.plot(nu, y, label="Signed error")
    if noise_trials is not None:
      # Center uncertainty around central line: use std of deviations from y
      dev = noise_trials[:, acq, dtype_idx, :] - y[np.newaxis, :]
      std_band = np.nanstd(dev, axis=0)
      ax_l.fill_between(nu, y - std_band, y + std_band,
                        color="#1f77b4", alpha=0.3, label="±1 sd")
    ax_l.xaxis.set_major_formatter(FuncFormatter(lambda x_val, tick_pos: f"{np.abs(x_val):.8g}"))
    if acq == 2:
      ax_l.set_xlabel("Frequency (ppm)")
    ax_l.set_ylabel(f"{acq_titles[acq]}\nSigned error (normalized)")
    if acq == 0:
      ax_l.legend(loc="best")
    # Histogram (signed)
    vals = all_diff[acq, dtype_idx, :]
    ax_m.hist(vals, bins=50, color="#1f77b4", alpha=0.8)
    ax_m.set_xlabel("Signed error (normalized)")
    ax_m.set_ylabel("Count")
    # 2D correlation heatmap (frequency vs signed error)
    n_points = len(vals)
    n_freq = len(nu)
    n_samp = max(1, n_points // n_freq)
    X = np.tile(nu, n_samp) # noqa: N806
    # y-values already flattened as vals
    # Compute Pearson r
    if np.std(X) > 0 and np.std(vals) > 0:
      r = float(np.corrcoef(X, vals)[0,1])
    else:
      r = float('nan')
    # 2D histogram
    xbins = np.linspace(nu.min(), nu.max(), 80)
    y_low, y_high = np.percentile(vals, [0.5, 99.5])
    if not np.isfinite(y_low) or not np.isfinite(y_high) or y_low == y_high:
      y_low, y_high = (vals.min() - 1e-6, vals.max() + 1e-6)
    ybins = np.linspace(y_low, y_high, 80)
    H, xe, ye = np.histogram2d(X, vals, bins=[xbins, ybins]) # noqa: N806
    # Normalize to density per bin (optional)
    # H = H / np.sum(H)
    ax_r.pcolormesh(xe, ye, H.T, shading='auto', cmap='viridis')
    ax_r.set_xlabel("Frequency (ppm)")
    ax_r.xaxis.set_major_formatter(FuncFormatter(lambda x_val, tick_pos: f"{np.abs(x_val):.8g}"))
    if acq == 0:
      ax_r.set_title("Freq-error heatmap")
    ax_r.text(0.01, 0.95, f"r={r:.2f}", transform=ax_r.transAxes,
              ha='left', va='top', fontsize=9,
              bbox={'facecolor': 'white', 'alpha': 0.6, 'edgecolor': 'none', 'pad': 2})
  if acq == 0:
      ax_l.set_title("Per-frequency signed error")
      ax_m.set_title("Signed error distribution")
  fig.suptitle(f"{dtype_name} - Signed Error, Distribution, and Freq-Error Heatmap")
  fig.tight_layout()
  return fig
