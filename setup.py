#!/usr/bin/env python3
#
# setup.py - MRSNet - setup MRSNet code
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2019, Max Chandler, PhD student at Cardiff University
# Copyright (C) 2020, Frank C Langbein <frank@langbein.org>, Cardiff University

import os
import zipfile
import requests
from tqdm import tqdm
from urllib.request import urlopen
import compileall

def main():
    # Download LCModel basis sets from http://purcell.healthsciences.purdue.edu/mrslab/basis_sets.html
    print('Downloading LCModel MEGA-PRESS basis sets from Purdue for Siemens, Phillips & GE.')
    save_directory = os.path.join('data', 'basis', 'lcmodel')
    os.makedirs(save_directory, exist_ok=True)
    urls = ["http://purcell.healthsciences.purdue.edu/mrslab/files/3t_IU_MP_te68_748_ppm_inv.basis",
            "http://purcell.healthsciences.purdue.edu/mrslab/files/3t_IU_MP_te68_diff_yesNAAG_noLac_Kaiser.basis",
            "http://purcell.healthsciences.purdue.edu/mrslab/files/3t_IU_MP_te68_diff_yesNAAG_noLac_c.basis",
            "http://purcell.healthsciences.purdue.edu/mrslab/files/3t_GE_MEGAPRESS_Kaiser_oct2011_75_ppm_inv.basis",
            "http://purcell.healthsciences.purdue.edu/mrslab/files/3t_GE_MEGAPRESS_Kaiser_oct2011_1975_diff.basis",
            "http://purcell.healthsciences.purdue.edu/mrslab/files/3t_GE_MEGAPRESS_june2011_diff.basis",
            "http://purcell.healthsciences.purdue.edu/mrslab/files/3t_philips_MEGAPRESS_Kaiser_oct2011_75_ppm_inv.basis",
            "http://purcell.healthsciences.purdue.edu/mrslab/files/3t_philips_MEGAPRESS_Kaiser_oct2011_1975_diff.basis",
            "http://purcell.healthsciences.purdue.edu/mrslab/files/3t_philips_MEGAPRESS_may2010_diff.basis"]
    filenames = ["MEGAPRESS_edit_off_Siemens_3T.basis",
                 "MEGAPRESS_difference_Siemens_3T_kasier.basis",
                 "MEGAPRESS_difference_Siemens_3T_govindaraju.basis",
                 "MEGAPRESS_edit_off_GE_3T.basis",
                 "MEGAPRESS_difference_GE_3T_kasier.basis",
                 "MEGAPRESS_difference_GE_3T_govindaraju.basis",
                 "MEGAPRESS_edit_off_Phillips_3T.basis",
                 "MEGAPRESS_difference_Phillips_3T_kasier.basis",
                 "MEGAPRESS_difference_Phillips_3T_govindaraju.basis"]
    for url, filename in tqdm(zip(urls, filenames), total=len(urls), desc='Downloading LCModel basis sets'):
        response = urlopen(url)
        with open(os.path.join(save_directory, filename), 'wb') as output:
            output.write(response.read())

    # Download fid-a basis sets
    print('Downloading FID-A basis set')
    dir = os.path.join('data', 'basis')
    os.makedirs(dir, exist_ok=True)
    zip_filename = download_large_file('https://qyber.black/data/MRIS/phantoms/fida_20200716.zip')
    print('Extracting FIDA-basis to '+dir)
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(dir)
    print('Deleting downloaded zip file')
    os.remove(zip_filename)

    # Download pygamma basis sets
    print('Downloading PyGamma basis set')
    dir = os.path.join('data', 'basis')
    os.makedirs(dir, exist_ok=True)
    zip_filename = download_large_file('https://qyber.black/data/MRIS/phantoms/pygamma_20200716.zip')
    print('Extracting FIDA-basis to '+dir)
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(dir)
    print('Deleting downloaded zip file')
    os.remove(zip_filename)

    # Download benchmark dataset
    print('Downloading benchmark dataset: WARNING, this file is large ~3.3GiB.')
    dir = os.path.join('data', 'benchmark')
    os.makedirs(dir, exist_ok=True)
    zip_filename = download_large_file('https://qyber.black/data/MRIS/phantoms/GABAPhantoms_20190815.zip')
    print('Extracting benchmark datasets to '+dir)
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(dir)
    print('Deleting downloaded zip file')
    os.remove(zip_filename)

    # Forcing to compile all python files
    compileall.compile_dir('.', force=True)

def download_large_file(url):
    # https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        filesize = int(requests.get(url, stream=True).headers['Content-length'])
        chunk_size = 8192
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=chunk_size), total=filesize/chunk_size, desc='Downloading experimental benchmark datasets'):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    # f.flush()
    return local_filename

if __name__ == '__main__':
    main()
