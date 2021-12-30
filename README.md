# MRSNet

> SPDX-FileCopyrightText: Copyright (C) 2019 Max Chandler, PhD student at Cardiff University  
> SPDX-FileCopyrightText: Copyright (C) 2020-2021 Frank C Langbein <frank@langbein.org>, Cardiff University  
> SPDX-License-Identifier: AGPL-3.0-or-later  

MRSNet is aimed at MR spectral quantification using convolutional neural
networks. It is mainly aimed at MEGAPRESS spectra. It also provides methods to
generate datasets from loaded LCModel ".BASIS" files or simulated by
[FID-A](https://github.com/CIC-methods/FID-A) or
[PyGamma](https://scion.duhs.duke.edu/vespa/gamma/wiki/PyGamma).

More information can be found in the associated paper:

M Chandler, C Jenkins, SM Shermer, FC Langbein. **MRSNet: Metabolite
Quantification from Edited Magnetic Resonance Spectra With Convolutional Neural
Network**. Preprint, 2019. [arXiv:1909.03836](https://arxiv.org/abs/1909.03836)
https://langbein.org/mrsnet-paper/

## Built With

### Software

* [Keras](https://keras.io/) - The Deep Learning framework used.
* [Tensorflow](https://www.tensorflow.org/) - Underlying Machine Learning
  library.
* [FID-A](https://github.com/CIC-methods/FID-A) - MRS simulation toolbox.
* [PyGamma](https://scion.duhs.duke.edu/vespa/gamma/wiki/PyGamma) - Another MRS
  simulation toolbox.

### Data Sources

* [Swansea Benchmark Dataset](https://langbein.org/gabaphantoms_20190815) -
  Benchmark phantom datasets collected at Swansea University's 3T Siemens
  scanner.
* [Purdue LCModel basis sets](http://purcell.healthsciences.purdue.edu/mrslab/basis_sets.html) -
  Data source for the LCModel basis sets

## Getting Started

### Prerequisites

* Tested mostly on Linux and may not work on any other platform without some
  adjustments.
* In particular for training, but also for quantification, a GPU (with
  tensorflow support) is strongly recommended.
* Standard packages
  * Git and git-lfs for git with submodules and LFS support.
  * Python 3.9 with pip (recent python3 versions should be OK).
  * Install these using your package manager with root privileges. E.g. Debian
    based distributions:
    `sudo apt update && sudo apt install git git-lfs python3.9 pip`.
  * For scipy/numpy you may need to install lapack and blas libraries for your
    system. We use fftw3 for the Fourier transform functions via pyfftw by
    default (see the `npfft_module` config variable and configuration files
    below), so you may also have to install `libfftw3`. Any such missing
    packages may cause the pip3 install command below to fail.
* MATLAB - Only required if you plan to simulate new FID-A spectra (the basis
  sets we used in the paper are in the git data/basis submodule).

### Install Instructions (Linux)

1. Clone the repository: `git clone git@qyber.black:mrs/code-mrsnet.git mrsnet`
   (check the clone url, as this  may be different if you use a different
   repository, e.g. from a mirror or alternative versions for development, etc).
2. Navigate to the directory: `cd mrsnet`
   (make sure to select a branch or tag with `git checkout BRANCH_OR_TAG` for a
   specific version instead of the master branch).
3. Update submodules: `git submodule update --init --recursive`.
4. Install the requirements: `pip3 install -r requirements.txt`.

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

#### Folders and Git Submodules

The benchmark dataset is in `data/bechmark` and the basis sets in `data/basis`
and the best models for distribution in `data/model-dist`. All these folders are
git submodules and installed with the procedure above. They are generally
required to run MRSNet.

By default, networks are stored in `data/models` along with some basic
analytics and the simulated spectra are stored in `data/sim-spectra`. Usually
these folders are empty on installation. There are two git repositories on
qyber.black with some data for these folders, generated for the publications,
etc.:

* [Data - MRSNet Models](https://qyber.black/mrs/data-mrsnet-model)
* [Data - MRSNet Simulated Spectra](https://qyber.black/mrs/data-mrsnet-simulated-spectra)

These can be cloned into those folders, if you wish to explore this data and use
it for your own analysis.

Generally it is best to run `mrsnet.py` from the base-folder of the git
repository. The folder locations in data are determined by the real location of
the `mrsnet.py` file (not symbolic links). These and other configuration values
can be overwritten by providing a `~/.config/mrsnet.json` file (see `Cfg` class
in `mrsnet/cfg.py` for details). If you change the location of the folders in
data, you do have to make sure the submodule data is available in the new
location. MRSNet does not search multiple paths. MRSNet also stored configuration
values in `cfg.json` in the root of the project folder. This overwrites the
defaults from `cfg.py`, but the above `mrsnet.json` overwrites these settings.

MRSNet uses a MRSNET_DEV environment variable for activating test and development
code. It's values are explained in `mrsnet/cfg.py`, but this is only relevant
for development and not the general operation. Note, operation of this may change
at any time.

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

## Versioning

Released versions:
* v1.0 - first release, tensorflow 1 and python2.
* v1.1-dev1 - update to python3 and tensorflow 2; code, api and ui cleanups.
* v1.1-dev2 - updates to spectra processing; extended dataset generation and model selection.

## Locations

The code is developed and maintained on [qyber\\black](https://qyber.black)
at https://qyber.black/mrs/code-mrsnet

This code is mirrored at
* https://github.com/MaxChandler/MRSNet

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

M Chandler, C Jenkins, SM Shermer, FC Langbein. **Code - MRSNet**. Version 1.0. _FigShare_, Software, 16th August 2019.
[[10.6084/m9.figshare.9824417.v1]](https://doi.org/10.6084/m9.figshare.9824417.v1)
[[DEV:https://qyber.black/mrs/code-mrsnet]](https://qyber.black/mrs/code-mrsnet)
[[MIRROR:https://github.com/MaxChandler/MRSNet]](https://github.com/MaxChandler/MRSNet)
