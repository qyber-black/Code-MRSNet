# MRSNet

> SPDX-FileCopyrightText: Copyright (C) 2019 Max Chandler, PhD student at Cardiff University  
> SPDX-FileCopyrightText: Copyright (C) 2020-2024 Frank C Langbein <frank@langbein.org>, Cardiff University  
> SPDX-FileCopyrightText: Copyright (C) 2021-2022 S Shermer <lw1660@gmail.com>, Swansea University
> SPDX-License-Identifier: AGPL-3.0-or-later  

MRSNet is aimed at MR spectral quantification using artificial neural
networks. It is aimed at MEGAPRESS spectra. It also provides methods to
generate datasets from loaded LCModel ".BASIS" files or simulated by
[FID-A](https://github.com/CIC-methods/FID-A) or
[PyGamma](https://vespa-mrs.github.io/vespa.io/other_packages/dev_gamma).

More information can be found in the associated paper:

M Chandler, C Jenkins, SM Shermer, FC Langbein. **MRSNet: Metabolite
Quantification from Edited Magnetic Resonance Spectra With Convolutional Neural
Network**. Preprint, 2019. [arXiv:1909.03836](https://arxiv.org/abs/1909.03836)
https://langbein.org/mrsnet-paper/

## Getting Started

### Prerequisites

* Tested on Linux and may not work on any other platform without some adjustments.
  Standard packages for Linux are:
  * Git and git-lfs for git with submodules and LFS support.
  * Python 3.11 (more recent versions may not work).
  * Install these using your package manager with root privileges. E.g. Debian
    based distributions:
    `sudo apt update && sudo apt install git git-lfs python3.11 python3.11-venv`
* For all standard python packages used, see `requirements.txt`. These will be 
  installed with the commands below, but here are some extra notes on potential
  issues.
  * [Tensorflow](https://www.tensorflow.org/) as machine learning library. In particular 
    for training, but also for quantification, a GPU (with tensorflow support) is strongly
    recommended, with [cudnn](https://developer.nvidia.com/cudnn) or [OneAPI](https://www.intel.com/content/www/us/en/developer/articles/guide/optimization-for-tensorflow-installation-guide.html). Version 2.15 should
    work, but more recent versions will likely fail.
  * For scipy/numpy you may need to install lapack and blas libraries for your
    system. By default we use numpy's fft, but you can also use fftw3 for the
    Fourier transform functions via pyfftw (see the `npfft_module` config 
    variable and configuration files below), for which you should install `libfftw3`.
  * [PyGamma](https://github.com/pygamma-mrs/gamma),
    a MRS simulation toolbox. You only need this if you wish to use the pygamma basis
    spectra simulation. It is currently commented out in `requirements.txt` as not 
    supported in python 3.11. If needed you can still try to install it manually or use a 
    supported python version. See https://pygamma-mrs.github.io/gamma.io/release/GammaBuildingLibrary.html
    for installation instructions.
  * GPyOpt is no longer maintained, but usable, and depends on gpy. It can be safely
    commented out from `requirements.txt` if model selection is not used.
  * Any missing libraries may cause the pip3 install command below to fail.
* [FID-A](https://github.com/CIC-methods/FID-A), a MRS simulation toolbox. This is provided
  via a git submodule and integrated during the installation process below.
  * [MATLAB](https://mathworks.com/) - Only required if you plan to simulate new FID-A spectra (the basis
    sets we used in the paper are in the git data/basis-dist submodule).

### Install Instructions (Linux)

1. Clone the repository:
   ```
   git clone https://qyber.black/mrs/code-mrsnet.git mrsnet
   ```
   Check the clone url, as it may be different if you use a different
   repository, e.g. from a mirror or alternative versions for development, etc.
2. Navigate to the directory:
   ```
   cd mrsnet
   ```
   Make sure to select a branch or tag with `git checkout BRANCH_OR_TAG` for a
   specific version instead of the main branch.
3. Update submodules:
   ```
   git submodule update --init --recursive
   ```
4. Install the requirements:
   ```
   pip3 install -r requirements.txt
   ```
   Of course, you can and probably shoudl install these in a virtual environment to avoid
   conflicts. Note that the requirements may need additional libraries, etc. to be
   installed on you system that pip does not add (see note above). Potentially
   you may have to set this up in a virtual environment or use the
   `--break-system-packages` options (on your own risk of breaking something else).
   Optionally you may want to install pygamma manually (see prerequisites above). In 
   general dependency issues of python packages failing to installed can be addressed 
   by commenting them out of `requirements.txt`, but it may mean that certain MRSNet 
   functionality may not work.

To update to the latest version (of your selected branch), run `git pull` and
step 3 and 4 above in the project folder. To switch to another version or branch
run `git checkout BRANCH_OR_TAG` first.

Call `mrsnet.py --help` to get further information about all its sub-commands
and `mrsnet.py COMMAND --help` for details for each sub-command. The
sub-commands available are:

* basis:              Generate basis, if it does not exist.
* simulate:           Generate simulated spectra dataset.
* generate_datasets:  Generate standard simulated spectra datasets.
* compare:            Compare spectra with basis.
* train:              Train model on dataset.
* select:             Model selection on dataset.
* quantify:           Quantify spectra in dicoms.
* benchmark:          Run benchmark on model.

Generally it is best to run `mrsnet.py` from the base-folder of the git
repository. The folder locations in data are determined by the real location of
the `mrsnet.py` file (not symbolic links). These and other configuration values
can be overwritten by providing a `~/.config/mrsnet.json` file (see `Cfg` class
in `mrsnet/cfg.py` for details; there is also a `cfg.json` file in the project
folder, generated by this class, with the default values that can also be
changed there). If you change the location of the folders in data, you do have
to make sure the submodule data is available in the new location. MRSNet has 
search paths for basis, model and simulation datasets defined as `search_*`
variables in the configuration files. It stores any newly generated data
under the data folder in `basis`, `sim-spectra`, or `model` as default paths
that are always added by `cfg.py`. MRSNet also stores other configuration 
values in `cfg.json` in the project folder or alternatively `mrsnet.json` in
the config folder. This overwrites the defaults from `cfg.py` (`mrsnet.json`
overwrites `cfg.json`).

### Folders and Git Submodules

The benchmark dataset is in `data/benchmark`. Newly generated basis sets are stored in
`data/basis`. The default basis set is in a separate git repository as submodule in
`data/basis-dist`. Newly generated artificial neural network models are stored under
`data/model`. Our best models we distribute are stored in `data/model-dist` as a separate
git submodule. Newly simulated spectra are stored in `data/sim-spectra`. The submodules
with this data are automatically installed with the above git submodule command. The 
`*-dist` paths are automatically added to the search paths in the configuration.

The additional git submodules containing the data are

* [Data - MRS - MEGAPRESS Spectra](https://qyber.black/mrs/data-mrs-megapress-spectra) -
  Swansea benchmark phantom datasets collected at Swansea University's 3T Siemens scanner (in `data/benchmark`);
* [Data - MRSNet - Models - Dist](https://qyber.black/mrs/data-mrsnet-model-dist) -
  Best performing trained models for MRSNet (in `data/models-dist`);
* [Data - MRSNet - Basis Spectra - Dist](https://qyber.black/mrs/data-mrsnet-basis) -
  Standard basis sets used ofr MEGAPRESS simulation (in `data/basis-dist`).
* [Code - QDicom Utilities](https://qyber.black/ca/code-qdicom-utilities) -
  Library to read dicoms.

There are further git repositories on qyber.black with more data, generated for the
publications, etc. that you can also use for your own analysis:

* [Data - MRSNet - Models CNN](https://qyber.black/mrs/data-mrsnet-model-cnn): contains 
  a large amount of CNN models that you could clone into `data/model-cnn` and then add that
  path to the model search path in `cfg.json` or `mrsnet.json`. Note that this is a very
  large repository. It contains the complete analysis data for the CNN models.
* [Data - MRSNet - Simulated Spectra - MEGAPRESS](https://qyber.black/mrs/data-mrsnet-simulated-spectra-megapress):
  contains a range of simulated MEGAPRESS spectra with our simulators using the basis datasets in
  `data/basis-dist`. These datases have been used in the papers for training and testing the
  models. You may use these to train your own models, etc. You can clone this into 
  `data/sim-spectra-megapress`. Note, this is a very large repository.

## Simulating Spectra

To generate a simulated spectra dataset with the standard set of metabolites use
```
./mrsnet.py simulate --source lcmodel --sample random --noise_sigma 0.1 -n 10 -vv
```
This uses the lcmodel basis set (see basis subcommand for other basis sets and
how to generate them, if needed) to generate 10 spectra, sampling the
concentrations randomly, adding normal distributed noise with a standard
deviation of 0.1 to the time domain signal. The spectra are stored in a joblib
datafile under `data/sim-spectra` according to the parameters that were used to
generate them. The above would be stored in
`data/sim-spectra/lcmodel/siemens/123.23/1.0/Cr-GABA-Gln-Glu-NAA/megapress/random/1.0-0.0-0.1/10-1`
where the folder `10-1` indicates that this is the 1st set of 10 spectra generated.

## Training a Network

To train a model run, e.g.,
```
./mrsnet.py train -d TRAIN-DATA-PATH -e 100 --validate 5 -m cnn_small_softmax -vv
```
This trains a model based on the simulated spectra in the TRAIN-DATA-PATH (see
previous section of how to generate these and what these paths are) for 100
epochs using 5-fold cross validating on the cnn_small_softmax model with some
verbosity.

MRSNet can run model selection approaches over a set of model parameters
(currently hardcoded in `mrsnet/selection.py`) and also run the training
on a remote system using a separate script - see `scheduler/run_scw.sh` for
an example running on Supercomputing Wales. For example, run
```
./mrsnet.py select -d DATASET_PATH -e 100 --validate 0.8 --method grid cnn-simple-all --remote ./scheduler/run_scw.sh:USERNAME:10:15 -vv
```

## Running the Benchmark

To run the benchmark dataset on a model run
```
./mrsnet.py benchmark --model MODEL -vv
```
where MODEL is the path to the trained tensorflow model in the `data/model-dist`
or `data/model` folders (the path indicates the parameters used for the model
architecture and the training/testing data). Results are stored in the model
folder.

## Quantifying your own MEGA-PRESS Spectra

Quantifying your own spectra in dicom files or spectra joblib files (from
simulate) is done via
```
./mrsnet.py quantify -d DATASET -m MODEL -vv
```
DATASET is either a joblib file or a folder with dicom spectra. The MODEL is the
folder with the trained tensorflow model. Results are stored in the data folder
specified, as csv file. If there is a `concentrations.json` file at the top-level
in the data folder, this is assumed to contain the ground truth and quantification
results are compared to it.

The code will attempt to analyse all of the spectra contained in the provided
directory. There are a couple of caveats to enable this to work correctly:

1. All three acquisitions for each MEGA-PRESS scan must be present (edit on,
   edit off, difference).
2. Spectra that belong to the same scan must have a unique ID of your choice
   added to their filename (e.g. SCAN_001 or be in separate folders where the
   folder becomes the ID).
3. Spectra of the different acquisition types must be labelled, by adding either
   "EDIT_OFF", "EDIT_ON" or "DIFF" to anywhere after the unique ID from 2 in
   their filename.

An example for two MEGA-PRESS scan would be six files:
```
SCAN_000_EDIT_OFF.ima
SCAN_000_EDIT_ON.ima
SCAN_000_DIFF.ima
SCAN_001_EDIT_OFF.ima
SCAN_001_EDIT_ON.ima
SCAN_001_DIFF.ima
```
Also see the folders in the benchmark dataset (`data/benchmark`), which you
can use as an example structure where folders separate the spectra (e.g.
`data/benchmark/E1/MEGA_Combi_WS_ON`; note that the `concentrations.json`
file is not at the top-level for each of the spectra collections, so would
not be used if you run quantify on it; it is found separately by the benchmark
sub-command only).

Note, loading of non-Siemens DICOM files has not been tested.

## Issues

* If GPyOpt for gpo selection fails with "not positive definite, even with jitter.",
  see https://github.com/SheffieldML/GPy/issues/660 for a solution. Changing
  ```
  L = linalg.cholesky(A + np.eye(A.shape[0]) * jitter, lower=True)
  ```
  to
  ```
  L = np.linalg.cholesky(A + np.eye(A.shape[0]) * jitter)
  ```
  in `GPy/util/linalg.py` (GPy is a dependency of GPyOpt) seems to fix this.

## Versioning

Released versions:
* v1.0 - first release, tensorflow 1 and python2.
* v2.0 - update to python3 and tensorflow 2; code, api and ui cleanups; updates to
  spectra processing; extended dataset generation, model training, model selection, 
  and quantification.

## Locations

The code is developed and maintained on [qyber\\black](https://qyber.black)
at https://qyber.black/mrs/code-mrsnet

This code is mirrored at
* https://github.com/qyber-black/code-mrsnet

The mirrors are only for convenience, accessibility and backup.

## People

* [Max Chandler](https://qyber.black/max), [School of Computer Science and Informatics](https://www.cardiff.ac.uk/computer-science), [Cardiff University](https://www.cardiff.ac.uk/)
* [Frank C Langbein](https://qyber.black/xis10z), [School of Computer Science and Informatics](https://www.cardiff.ac.uk/computer-science), [Cardiff University](https://www.cardiff.ac.uk/); [langbein.org](https://langbein.org/)
* [Sophie M Shermer](https://qyber.black/lw1660), [Physics](https://www.swansea.ac.uk/physics), [Swansea University](https://www.swansea.ac.uk/)
* [Christopher W Jenkins](https://qyber.black/chris), [Physics](https://www.swansea.ac.uk/physics) and [Centre for Nanohealth](https://www.swansea.ac.uk/nanohealth/facilities/) and [Clinical Imaging Unit](https://www.swansea.ac.uk/medicine/research/researchfacilities/jointclinicalresearchfacility/clinicalimagingfacility/), [Swansea University](https://www.swansea.ac.uk/); [Cardiff University Brain Research Imaging Centre (CUBRIC)](https://www.cardiff.ac.uk/cardiff-university-brain-research-imaging-centre)

## Acknowledgments

* Brian Soher (VeSPA/PyGamma) for help locating the PyGamma pulse sequence code for MEGA-PRESS, PRESS and STEAM.

## Contact

For any general enquiries relating to this project, [send an e-mail](mailto:gitlab+mrs-code-mrsnet-38-issue-@qyber.black).

## Citation

M Chandler, SM Shermer, FC Langbein. **Code - MRSNet**. Version 2.0. Software, 2024.
[[DEV:https://qyber.black/mrs/code-mrsnet]](https://qyber.black/mrs/code-mrsnet)
[[MIRROR:https://github.com/MaxChandler/MRSNet]](https://github.com/qyber-black/code-mrsnet)
