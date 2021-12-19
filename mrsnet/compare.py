# mrsnet/comapre.py - MRSNet - compare and analyse spectra
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2021, Frank C Langbein <frank@langbein.org>, Cardiff University

import os
import numpy as np
import matplotlib.pyplot as plt

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
    # FIXME: plot can change dimensions of spectra and so combine does not work if plot before combine
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

  for l in range(len(ds.spectra)):
    if verbose > 1:
      print(f"## Spectra differences {l}: {ds.spectra[l][list(ds.spectra[l].keys())[0]].id}")
    dd = np.sum(np.abs(r_inp[l,:,:,:] - d_inp[l,:,:,:]),axis=2)
    print(f"Diff     Magnitude: {dd[0,0]:12f}")
    print(f"             Phase: {dd[0,1]:12f}")
    print(f"              Real: {dd[0,2]:12f}")
    print(f"         Imaginary: {dd[0,3]:12f}")
    print(f"Edit_Off Magnitude: {dd[1,0]:12f}")
    print(f"             Phase: {dd[1,1]:12f}")
    print(f"              Real: {dd[1,2]:12f}")
    print(f"         Imaginary: {dd[1,3]:12f}")
    print(f"Edit_On  Magnitude: {dd[2,0]:12f}")
    print(f"             Phase: {dd[2,1]:12f}")
    print(f"              Real: {dd[2,2]:12f}")
    print(f"         Imaginary: {dd[2,3]:12f}")
    # FIXME: statistics on difference distribution
    fig = plot_diff_spectra(r_inp[l,:,:,:],d_inp[l,:,:,:],r_out[l,:],
                            ref_spectra.spectra[l], ds.spectra[l], ref_spectra, ds,
                            ds.metabolites,basis.source,image_dpi,screen_dpi)
    plt.show(block=True)
    plt.close()

def plot_diff_spectra(r, d, c, rs, ds, rds, dds, metabolites, source, image_dpi, screen_dpi):
    # Plot difference between dataset and basis reference spectrum
    figure, axes = plt.subplots(4, 4, sharex=True, dpi=screen_dpi)
    axes[0,3].remove()
    axes[1,3].remove()
    axes[2,3].remove()
    axes[3,3].remove()

    plt.suptitle(f"Difference between Data (blue) and {source.upper()} Reference (orange)")

    axes[0,0].set_title("Difference")
    _, dx = ds['difference'].rescale_fft(high_ppm=dds.high_ppm, low_ppm=dds.low_ppm, npts=dds.n_fft_pts)
    _, rx = rs['difference'].rescale_fft(high_ppm=rds.high_ppm, low_ppm=rds.low_ppm, npts=rds.n_fft_pts)
    axes[0,0].plot(dx,d[0,0,:])
    axes[0,0].plot(rx,r[0,0,:],linewidth=.75)
    axes[0,0].set_ylabel("Magnitude")
    axes[1,0].plot(dx,d[0,1,:])
    axes[1,0].plot(rx,r[0,1,:],linewidth=.75)
    axes[1,0].set_ylabel("Phase")
    axes[2,0].plot(dx,d[0,2,:])
    axes[2,0].plot(rx,r[0,2,:],linewidth=.75)
    axes[2,0].set_ylabel("Real")
    axes[3,0].plot(dx,d[0,3,:])
    axes[3,0].plot(rx,r[0,3,:],linewidth=.75)
    axes[3,0].set_ylabel("Imaginary")
    axes[3,0].set_xlabel("Frequency (ppm)")

    axes[0,1].set_title("Edit Off")
    _, dx = ds['edit_off'].rescale_fft(high_ppm=dds.high_ppm, low_ppm=dds.low_ppm, npts=dds.n_fft_pts)
    _, rx = rs['edit_off'].rescale_fft(high_ppm=rds.high_ppm, low_ppm=rds.low_ppm, npts=rds.n_fft_pts)
    axes[0,1].plot(dx,d[1,0,:])
    axes[0,1].plot(rx,r[1,0,:],linewidth=.75)
    axes[0,0].get_shared_y_axes().join(axes[0,0], axes[0,1])
    axes[1,1].plot(dx,d[1,1,:])
    axes[1,1].plot(rx,r[1,1,:],linewidth=.75)
    axes[1,0].get_shared_y_axes().join(axes[1,0], axes[1,1])
    axes[2,1].plot(dx,d[1,2,:])
    axes[2,1].plot(rx,r[1,2,:],linewidth=.75)
    axes[2,0].get_shared_y_axes().join(axes[2,0], axes[2,1])
    axes[3,1].plot(dx,d[1,3,:])
    axes[3,1].plot(rx,r[1,3,:],linewidth=.75)
    axes[3,0].get_shared_y_axes().join(axes[3,0], axes[3,1])
    axes[3,1].set_xlabel("Frequency (ppm)")

    axes[0,2].set_title("Edit On")
    _, dx = ds['edit_on'].rescale_fft(high_ppm=dds.high_ppm, low_ppm=dds.low_ppm, npts=dds.n_fft_pts)
    _, rx = rs['edit_on'].rescale_fft(high_ppm=rds.high_ppm, low_ppm=rds.low_ppm, npts=rds.n_fft_pts)
    axes[0,2].plot(dx,d[2,0,:])
    axes[0,2].plot(rx,r[2,0,:],linewidth=.75)
    axes[0,0].get_shared_y_axes().join(axes[0,0], axes[0,2])
    axes[1,2].plot(dx,d[2,1,:])
    axes[1,2].plot(rx,r[2,1,:],linewidth=.75)
    axes[1,0].get_shared_y_axes().join(axes[1,0], axes[1,2])
    axes[2,2].plot(dx,d[2,2,:])
    axes[2,2].plot(rx,r[2,2,:],linewidth=.75)
    axes[2,0].get_shared_y_axes().join(axes[2,0], axes[2,2])
    axes[3,2].plot(dx,d[2,3,:])
    axes[3,2].plot(rx,r[2,3,:],linewidth=.75)
    axes[3,0].get_shared_y_axes().join(axes[3,0], axes[3,2])
    axes[3,2].set_xlabel("Frequency (ppm)")

    ax = plt.subplot(1, 4, 4)
    plt.title('Concentrations')
    ax.bar(np.linspace(0, c.shape[0] - 1, c.shape[0]), c)
    ax.set_xticks(np.arange(len(metabolites)))
    ax.set_xticklabels(molecules.short_name(metabolites))

    return figure
