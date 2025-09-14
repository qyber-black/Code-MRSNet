# mrsnet/compare.py - MRSNet - compare and analyse spectra
#
# SPDX-FileCopyrightText: Copyright (C) 2021-2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2022-2025 Zien Ma, PhD student at Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Spectrum comparison utilities for MRSNet.

This module provides functions for comparing datasets with basis spectra
and analyzing differences between experimental and simulated data.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter

import mrsnet.molecules as molecules
from mrsnet.dataset import Dataset


def compare_basis(ds, basis, high_ppm=-4.5, low_ppm=-1, n_fft_pts=2048, verbose=0, screen_dpi=96, out_dir=None, save_prefix=None):
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
  """
  # Compare dataset to spectra generated from basis
  # Setup basis
  if verbose > 0:
    print("# Preparing reference spectra from basis")
  ref_spectra = Dataset("Basis Spectra")
  diff = 0.0
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

  all_diff = np.ndarray((r_inp.shape[1],r_inp.shape[2],r_inp.shape[3]*r_inp.shape[0]))
  all_ref = np.ndarray((r_inp.shape[1],r_inp.shape[2],r_inp.shape[3]*r_inp.shape[0]))
  all_dat = np.ndarray((r_inp.shape[1],r_inp.shape[2],r_inp.shape[3]*r_inp.shape[0]))
  for l in range(len(ds.spectra)):
    diff = r_inp[l,:,:,:] - d_inp[l,:,:,:]
    all_diff[:,:,l*r_inp.shape[3]:(l+1)*r_inp.shape[3]] = diff
    all_ref[:,:,l*r_inp.shape[3]:(l+1)*r_inp.shape[3]] = r_inp[l,:,:,:]
    all_dat[:,:,l*r_inp.shape[3]:(l+1)*r_inp.shape[3]] = d_inp[l,:,:,:]
    print(f"## Spectra differences {l}: {ds.spectra[l][next(iter(ds.spectra[l].keys()))].id}")
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
    if verbose > 1:
      plot_diff_spectra(r_inp[l,:,:,:],d_inp[l,:,:,:],r_out[l,:],nu,
                        ds.metabolites,basis.source,screen_dpi)
      plt.show(block=True)
      plt.close()

  print("# Differences over all spectra (max normalised to 1)")
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
  # Always save summary plots and metrics
  if out_dir is not None:
    try:
      os.makedirs(out_dir, exist_ok=True)
    except Exception:
      pass
    if save_prefix is None:
      save_prefix = f"{basis.source}_{basis.manufacturer}_{basis.omega}_{basis.linewidth}_{basis.pulse_sequence}_{basis.sample_rate}_{basis.samples}"
    # MAE spectrum plot
    mae_spectrum = np.mean(np.abs(r_inp - d_inp), axis=0)
    fig_mae = plot_mae_spectrum(mae_spectrum, nu, screen_dpi)
    fig_mae.savefig(os.path.join(out_dir, f"compare_mae_spectrum_{save_prefix}.png"), dpi=300)
    plt.close(fig_mae)
    # Histogram summary plot
    fig, axs = plt.subplots(4,3,sharey=True, dpi=screen_dpi)
    fig.suptitle("Differences over all Spectra")
    for l in range(3):
      for k in range(4):
        sns.histplot(all_diff[l,k,:], kde=True, ax=axs[k,l])
    axs[0,0].set_ylabel("Magnitude - Count")
    axs[1,0].set_ylabel("Phase - Count")
    axs[2,0].set_ylabel("Real - Count")
    axs[3,0].set_ylabel("Imaginary - Count")
    axs[3,0].set_xlabel("Difference - Error")
    axs[3,1].set_xlabel("Edit Off - Error")
    fig.savefig(os.path.join(out_dir, f"compare_error_hist_{save_prefix}.png"), dpi=300)
    plt.close(fig)
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
          "corr": None if np.isnan(corr[ai,di]) else float(corr[ai,di])
        }
    with open(os.path.join(out_dir, f"compare_metrics_{save_prefix}.json"), "w") as f:
      json.dump({
        "basis": {
          "source": basis.source,
          "manufacturer": basis.manufacturer,
          "omega": basis.omega,
          "linewidth": basis.linewidth,
          "pulse_sequence": basis.pulse_sequence,
          "sample_rate": basis.sample_rate,
          "samples": basis.samples
        },
        "metrics": metrics
      }, f, indent=2)

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
        axes[dt, acq].xaxis.set_major_formatter(FuncFormatter(lambda x_val, tick_pos: f"{np.abs(x_val):.8g}"))
        if dt == 0:
          axes[dt, acq].set_title(labels_acq[acq])
    return figure
