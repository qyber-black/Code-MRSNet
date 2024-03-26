# mrsnet/comapre.py - MRSNet - compare and analyse spectra
#
# SPDX-FileCopyrightText: Copyright (C) 2021-2024 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

import mrsnet.molecules as molecules
from mrsnet.dataset import Dataset

def compare_basis(ds, basis, high_ppm=-4.5, low_ppm=-1, n_fft_pts=2048, verbose=0, screen_dpi=96):
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
    print("Warning, difference in concentrations between dataset and reference: {diff}")

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
                       verbose=verbose)
  ppm_step = (low_ppm - high_ppm)/(r_inp.shape[-1]-1)
  nu = np.arange(high_ppm,low_ppm+ppm_step,ppm_step)

  all_diff = np.ndarray((r_inp.shape[1],r_inp.shape[2],r_inp.shape[3]*r_inp.shape[0]))
  for l in range(len(ds.spectra)):
    diff = r_inp[l,:,:,:] - d_inp[l,:,:,:]
    all_diff[:,:,l*r_inp.shape[3]:(l+1)*r_inp.shape[3]] = diff
    print(f"## Spectra differences {l}: {ds.spectra[l][list(ds.spectra[l].keys())[0]].id}")
    dd = np.sum(np.abs(diff),axis=2) / diff.shape[2]
    m = np.mean(diff, axis=2)
    s = np.std(diff, axis=2)
    print("                    %12s %12s %12s" % ("MAE", "Mean", "Std"))
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
  print("                    %12s %12s %12s" % ("MAE", "Mean", "Std"))
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
  if verbose > 3:
    fig, axs = plt.subplots(4,3,sharey=True)
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
    axs[3,2].set_xlabel("Edif On - Error")
    plt.show(block=True)
    plt.close()

def plot_diff_spectra(r, d, c, nu, metabolites, source, screen_dpi):
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
    axes[3,0].xaxis.set_major_formatter(FuncFormatter(lambda x_val, tick_pos: "{:.8g}".format(np.abs(x_val))))

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
    axes[3,1].xaxis.set_major_formatter(FuncFormatter(lambda x_val, tick_pos: "{:.8g}".format(np.abs(x_val))))

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
    axes[3,2].xaxis.set_major_formatter(FuncFormatter(lambda x_val, tick_pos: "{:.8g}".format(np.abs(x_val))))

    ax = plt.subplot(1, 4, 4)
    plt.title('Concentrations')
    ax.bar(np.linspace(0, c.shape[0] - 1, c.shape[0]), c)
    ax.set_xticks(np.arange(len(metabolites)))
    ax.set_xticklabels(molecules.short_name(metabolites))

    return figure
