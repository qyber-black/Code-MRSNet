# mrsnet/comapre.py - MRSNet - compare and analyse spectra
#
# SPDX-FileCopyrightText: Copyright (C) 2021 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .dataset import Dataset
from . import molecules

def compare_basis(ds, basis, verbose=0, image_dpi=[300], screen_dpi=96):
  # Compare dataset to spectra generated from basis
  # Setup basis
  if verbose > 0:
    print("# Preparing reference spectra from basis")
  ref_spectra = Dataset("Basis Spectra")
  diff = 0.0
  if verbose > 1:
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
                                    verbose=verbose)
  d_inp, _ = ds.export(metabolites=ds.metabolites, norm='max',
                       acquisitions=['difference','edit_off','edit_on'],
                       datatype=['magnitude','phase','real','imaginary'],
                       verbose=verbose)
  ppm_step = (ds.low_ppm - ds.high_ppm)/(r_inp.shape[-1]-1)
  nu = np.arange(ds.high_ppm,ds.low_ppm+ppm_step,ppm_step)

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
      fig = plot_diff_spectra(r_inp[l,:,:,:],d_inp[l,:,:,:],r_out[l,:],nu,
                              ds.metabolites,basis.source,image_dpi,screen_dpi)
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
  if verbose > 0:
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

  print("# Time Domain Differences over all Spectra (max amplitude normalised to 1)")
  adc_l = ds.spectra[0]['edit_off'].adc_len(pad=False)
  spc_l = len(ds.spectra)
  all_diff = np.ndarray((3,2,spc_l*adc_l))
  for s in range(len(ref_spectra.spectra)):
    for l,a in enumerate(["difference","edit_off","edit_on"]):
      r_data = ref_spectra.spectra[s][a].adc(pad=False)
      r_data /= np.max(np.abs(r_data))
      d_data = ds.spectra[s][a].adc(pad=False)
      d_data /= np.max(np.abs(d_data))
      r_data = np.interp(np.arange(0,len(d_data),1)*ds.spectra[s][a].dt,
                         np.arange(0,len(r_data),1)*ref_spectra.spectra[s][a].dt,
                         r_data)
      all_diff[l,0,s*adc_l:(s+1)*adc_l] = np.abs(r_data) - np.abs(d_data)
      all_diff[l,1,s*adc_l:(s+1)*adc_l] = np.angle(r_data) - np.angle(d_data)
  dd = np.sum(np.abs(all_diff),axis=2) / all_diff.shape[2]
  m = np.mean(all_diff, axis=2)
  s = np.std(all_diff, axis=2)
  print("                    %12s %12s %12s" % ("MAE", "Mean", "Std"))
  print(f"Diff     Magnitude: {dd[0,0]:12f} {m[0,0]:12f} {s[0,0]:12f}")
  print(f"             Phase: {dd[0,1]:12f} {m[0,1]:12f} {s[0,1]:12f}")
  print(f"Edit_Off Magnitude: {dd[1,0]:12f} {m[1,0]:12f} {s[1,0]:12f}")
  print(f"             Phase: {dd[1,1]:12f} {m[1,1]:12f} {s[1,1]:12f}")
  print(f"Edit_On  Magnitude: {dd[2,0]:12f} {m[2,0]:12f} {s[2,0]:12f}")
  print(f"             Phase: {dd[2,1]:12f} {m[2,1]:12f} {s[2,1]:12f}")
  if verbose > 0:
    fig, axs = plt.subplots(2,3,sharey=True)
    fig.suptitle("Time Domain Differences over all Spectra (max amplitude normalised to 1)")
    for l in range(3):
      for k in range(2):
        sns.histplot(all_diff[l,k,:], kde=True, ax=axs[k,l])
    axs[0,0].set_ylabel("Magnitude - Count")
    axs[1,0].set_ylabel("Phase - Count")
    axs[1,0].set_xlabel("Difference - Error")
    axs[1,1].set_xlabel("Edit Off - Error")
    axs[1,2].set_xlabel("Edif On - Error")
    plt.show(block=True)
    plt.close()

def plot_diff_spectra(r, d, c, nu, metabolites, source, image_dpi, screen_dpi):
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

    ax = plt.subplot(1, 4, 4)
    plt.title('Concentrations')
    ax.bar(np.linspace(0, c.shape[0] - 1, c.shape[0]), c)
    ax.set_xticks(np.arange(len(metabolites)))
    ax.set_xticklabels(molecules.short_name(metabolites))

    return figure
