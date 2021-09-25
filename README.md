# MRSNet

MRSNet is aimed at MR spectral quantification using convolutional neural networks. It is main aimed at MEGAPRESS spectra. It also provides methods to generate datasets from loaded LCModel ".BASIS" files or simulated by [FID-A](https://github.com/CIC-methods/FID-A) or
[PyGamma](https://scion.duhs.duke.edu/vespa/gamma/wiki/PyGamma).

More information can be found in the associated paper

M. Chandler, C. Jenkins, S. M. Shermer, F. C. Langbein. MRSNet: Metabolite Quantification from Edited Magnetic Resonance Spectra With Convolutional Neural Network. Preprint, 2019. [arxiv:1909.03836] https://langbein.org/mrsnet-paper/

## Built With

### Software

* [Keras](https://keras.io/) - The Deep Learning framework used
* [Tensorflow](https://www.tensorflow.org/) - Underlying Machine Learning library
* [FID-A](https://github.com/CIC-methods/FID-A) - MRS simulation toolbox
* [PyGamma](https://scion.duhs.duke.edu/vespa/gamma/wiki/PyGamma) - Another MRS simulation toolbox
* [VeSPA](https://scion.duhs.duke.edu/vespa/project) - Versatile Simulation, Pulses and Analysis

### Data sources

* [Swansea Benchmark Dataset](https://langbein.org/gabaphantoms_20190815) - Benchmark phantom datasets collected at Swansea University's 3T Siemens scanner.
* [Purdue LCModel basis sets](http://purcell.healthsciences.purdue.edu/mrslab/basis_sets.html) - Data source for the LCModel basis sets

## Getting Started

### Prerequisites

* Python 3.9 for main MRSNet.
* MATLAB - Only required if you plan to simulate new FID-A spectra. (basis sets we used are in the git data/basis submodule)
* Linux system packages:
    * Git-lfs for git submodule support: `git-lfs`
    * Install these using your package manager with root privileges. E.g. Debian based distributions: `sudo apt update && sudo apt install git-lfs`.

### Install instructions (Linux)

1. Clone the repository: `git clone https://qyber.black/MRIS/mrsnet.git`
2. Navigate to the directory: `cd mrsnet`
3. Update submodules: `git submodule update --init --recursive`.
6. Install the requirements (CPU or GPU): `pip3 install -r requirements.txt` (GPU support requires [CUDA](https://developer.nvidia.com/cuda-zone): There's a good guide available [here](https://www.tensorflow.org/install/gpu))
7. Download the additional required data: `python3 setup.py` (currenlty not needed as basis is in the data/basis git submodule!)

Call `mrsnet.py --help` to get further information about all its sub-commands and `mrsnet.py COMMAND --help` for details for each sub-command.

## Training a network

To train a model run
```
python3 mrsnet.py train
```

An example of a more complex training call:
```
python3 mrsnet.py train -n 10000 -e 200 -b 16 --basis_source fida --linewidths 0.75 1 1.25 --omega 900 --model_name mrsnet_small_kernel
```
This simulates spectra and trains a model with:
* 200 epochs, with a mini-batch size of 16
* Spectra are sourced from FID-A, with a scanner B0 field of 900MHz
* 10,000 Spectra are evenly split (3,333) over the linewidths (0.75, 1, 1.25)
* For the network architecture called "mrsnet_small_kernel" (found in [cnns.py](cnns.py))

#### Folders

By default, networks are stored in `data/models/` along with some basic analytics. The benchmark dataset is in `data/bechmark/` and the basis sets in `data/basis`.

## Quantifying Spectra

To quantify spectra run:
```
python3 mrsnet.py quantify
```
Defaults are to use the E1 MEGA-PRESS benchmark spectra, with the best model from MRSNet. The default behaviour of quantify is to use the best network from the MRSNet paper to quantify the bundled E1 dataset.

Quantifying spectra and specifying the model and spectra directory:
```
python2 mrsnet.py quantify --model data/model/some_model_dir --spectra some/spectra/directory
```

### Quantifying your own MEGA-PRESS spectra

The code will attempt to analyse all of the spectra contained in the provided directory. There are a couple of caveats to enable this to work correctly:

1. All three acquisitions for each MEGA-PRESS scan must be present (edit on, edit off, difference).
2. Spectra that belong to the same scan must have a unique ID of your choice added to their filename (e.g. SCAN_001).
	1. If you know the ground truth of the scan, it should be added to the dictionary located in utilities/constants.py. This enables the code to analyse the performance of the quantification.
3. Spectra of the different acquisition types must be labelled, by adding either "EDIT_OFF", "EDIT_ON" or "DIFF" to anywhere after the unique ID from 2 in their filename.

An example for two MEGA-PRESS scan would be six files:
```
SCAN_000_EDIT_OFF.ima
SCAN_000_EDIT_ON.ima
SCAN_000_DIFF.ima
SCAN_001_EDIT_OFF.ima
SCAN_001_EDIT_ON.ima
SCAN_001_DIFF.ima
```

### Non-Siemens DICOM files

Loading of non-Siemens DICOM files has not been tested.

### Training and testing

The `run_test.py` script is setup to generate various datasets for training and testing MRSNet and produce the data for the associated papers. Use it as

```
python3 run_test.py DATASET REPEATS
```

to generate the DATASET with REPEATS repetition for each parameter set. Use `-help` for more information and a list of all datasets.

## Versioning

We use [SemVer](http://semver.org/) for versioning.

Released versions:
* v1.0 - first release, tensorflow 1 and python2.
* v1.1 - update to python3 and tensorflow 2, some code and interface cleanups.

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

## License

Copyright (C) 2019, M Chandler.
Copyright (C) 2020-2021, FC Langbein.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
