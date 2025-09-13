# MRSNet API Documentation

> SPDX-FileCopyrightText: Copyright (C) 2019 Max Chandler, PhD student at Cardiff University
> SPDX-FileCopyrightText: Copyright (C) 2022-2025 Zien Ma, PhD student at Cardiff University
> SPDX-FileCopyrightText: Copyright (C) 2020-2025 Frank C Langbein <frank@langbein.org>, Cardiff University
> SPDX-License-Identifier: AGPL-3.0-or-later

## Overview

MRSNet is a Python package for Magnetic Resonance Spectra (MRS) quantification using artificial neural networks, specifically designed for MEGAPRESS spectra. The package provides methods to generate datasets from LCModel basis files or simulated spectra using FID-A or PyGamma.

## Package Structure

The main package is located in `mrsnet/` and contains the following modules:

- `__init__.py` - Package initialization
- `cfg.py` - Configuration management
- `molecules.py` - Metabolite definitions and utilities
- `basis.py` - Spectral basis management
- `spectrum.py` - Individual spectrum handling
- `dataset.py` - Dataset management and operations
- `cnn.py` - Convolutional Neural Network models
- `autoencoder.py` - Autoencoder models
- `ae_quantifier.py` - Autoencoder-quantifier models
- `train.py` - Training utilities and validation
- `analyse.py` - Model analysis and evaluation
- `compare.py` - Spectrum comparison utilities
- `grid.py` - Grid search utilities
- `selection.py` - Model selection algorithms
- `getfolder.py` - File system utilities
- `qdicom/` - DICOM file handling utilities
- `simulators/` - Simulation tools (FID-A, PyGamma)

## Main Entry Point

### `mrsnet.py`

The main command-line interface providing the following subcommands:

- `basis` - Generate basis spectra
- `simulate` - Generate simulated spectra datasets
- `generate_datasets` - Generate standard datasets
- `compare` - Compare spectra with basis
- `train` - Train models on datasets
- `select` - Model selection on datasets
- `quantify` - Quantify spectra in DICOM files
- `benchmark` - Run benchmark on models

## Core Classes and Functions

### Configuration (`mrsnet.cfg`)

#### `Cfg` Class
Configuration management for MRSNet.

**Static Methods:**
- `init(bin_path)` - Initialize configuration with root path
- `get_su_bases(reload=False)` - Get list of SU-* basis sets
- `dev(flag)` - Check if development flag is set
- `_screen_dpi()` - Get screen DPI for plotting

**Configuration Variables:**
- `path_root` - MRSNet root path
- `path_basis` - Save path for basis spectra
- `path_simulation` - Save path for simulation data
- `path_model` - Save path for trained models
- `path_benchmark` - Save path for benchmark data
- `search_basis` - Search paths for basis spectra
- `search_simulation` - Search paths for simulation data
- `search_model` - Search paths for models
- `figsize` - Default figure size for plots
- `fft_peak_location_estimator` - FFT peak location estimation method
- `b0_correct_ppm_range` - B0 correction PPM range
- `water_peak_ppm_range` - Water peak PPM range
- `phase_correct` - Phase correction algorithm
- `base_learning_rate` - Base learning rate for training
- `beta1`, `beta2`, `epsilon` - Adam optimizer parameters
- `disable_gpu` - Disable GPU usage flag

### Molecules (`mrsnet.molecules`)

#### Constants
- `NAMES` - Dictionary mapping long metabolite names to short names
- `B0_CORRECTION` - List of metabolites for B0 correction with reference peaks
- `WATER_REFERENCE` - Water peak reference PPM
- `GYROMAGNETIC_RATIO` - 1H gyromagnetic ratio in MHz/T

#### Functions
- `convert_names(molecules, shorten=False)` - Convert molecule names to standard form
- `short_name(names)` - Get short molecule name
- `long_name(names)` - Get long molecule name

### Basis Management (`mrsnet.basis`)

#### `BasisCollection` Class
Manages collections of basis spectra.

**Methods:**
- `__init__()` - Initialize empty collection
- `add(metabolites, source, manufacturer, omega, linewidth, pulse_sequence, sample_rate, samples, path_basis, search_basis)` - Add basis to collection
- `get(metabolites, source, manufacturer, omega, linewidth, pulse_sequence)` - Get basis from collection
- `describe()` - Get description of all bases in collection

#### `Basis` Class
Individual basis spectrum management.

**Methods:**
- `__init__(metabolites, source, manufacturer, omega, linewidth, pulse_sequence, sample_rate, samples)` - Initialize basis
- `setup(path_basis, search_basis)` - Load and setup basis spectra
- `name()` - Get basis name string
- `combine(concentrations, id)` - Combine basis spectra with concentrations
- `plot(data, type)` - Plot basis spectra

**Private Methods:**
- `_load(path_basis, search_basis)` - Load basis from files
- `_load_fida(path_basis, search_basis, source, second_call)` - Load FID-A basis
- `_load_su3t(path_basis, search_basis, source)` - Load SU-3TSkyra basis
- `_load_pygamma(path_basis, search_basis)` - Load PyGamma basis
- `_load_lcm(path_basis, search_basis)` - Load LCModel basis
- `_add_missing_spectra()` - Add missing MEGAPRESS spectra
- `_normalise()` - Normalize all spectra
- `_correct_b0()` - Apply B0 correction

### Spectrum Handling (`mrsnet.spectrum`)

#### `Spectrum` Class
Individual spectrum representation and manipulation.

**Methods:**
- `__init__(id, pulse_sequence, acquisition, omega, source, metabolites, linewidth)` - Initialize spectrum
- `set_f(fft, sample_rate, center_ppm, b0_shift_ppm, scale, remove_water_peak, phase_correct, force_phase_correct)` - Set frequency domain data
- `set_t(adc, sample_rate, center_ppm, b0_shift_ppm, scale, remove_water_peak, phase_correct, force_phase_correct)` - Set time domain data
- `get_f()` - Get frequency domain data
- `get_t()` - Get time domain data
- `correct_b0(shift)` - Apply B0 correction
- `rescale_fft(high_ppm, low_ppm, npts)` - Rescale FFT to specified range
- `add_noise_adc_normal(mu, sigma)` - Add ADC noise
- `plot(axes, type, mode)` - Plot spectrum
- `plot_spectrum(concentrations, screen_dpi, type)` - Plot full spectrum
- `save_json(filename, version)` - Save spectrum to JSON

**Static Methods:**
- `plot_full_spectrum(spectra, concentrations, screen_dpi, type)` - Plot multiple spectra
- `comb(f1, s1, f2, s2, id, acq)` - Combine two spectra
- `combs(fs, ss, id, acq)` - Combine multiple spectra
- `correct_b0_multi(spectra)` - Apply B0 correction to multiple spectra
- `load_fida(fida_file, id, source, su)` - Load FID-A spectrum
- `load_pygamma(pygamma_dir, search_path, metabolite, pulse_sequence, omega, linewidth, npts, dt, simulate, force_phase_correct)` - Load PyGamma spectrum
- `load_lcm(basis_file, acquisition, req_omega, req_metabolites)` - Load LCModel spectrum
- `load_dicom(file, concentrations, metabolites, verbose)` - Load DICOM spectrum
- `load_csv(file, concentrations, metabolites, verbose)` - Load CSV spectrum

**Private Methods:**
- `_set(sample_rate, center_ppm, b0_shift_ppm, scale, remove_water_peak, phase_correct, force_phase_correct)` - Internal data setting
- `_fft_peak_location(location, ppm_range)` - Find FFT peak location
- `_phase_correct_ernst()` - Apply Ernst phase correction
- `_phase_correct_acme()` - Apply ACME phase correction
- `_fft_remove_water_peak()` - Remove water peak from FFT

### Dataset Management (`mrsnet.dataset`)

#### `Dataset` Class
Manages collections of spectra and concentrations.

**Methods:**
- `__init__(name)` - Initialize dataset
- `load_dicoms(folder, concentrations, metabolites, verbose)` - Load DICOM data
- `generate_spectra(basis, num, samplers, verbose)` - Generate spectra from basis
- `add_noise(noise_p, noise_type, noise_mu, noise_sigma, verbose)` - Add noise to spectra
- `save(path, folder, spectra_only)` - Save dataset
- `plot_concentrations(norm)` - Plot concentration distributions
- `export(metabolites, high_ppm, low_ppm, n_fft_pts, norm, acquisitions, datatype, normalise, export_concentrations, verbose)` - Export data for training

**Static Methods:**
- `load(folder, force_clean, info_only)` - Load dataset from file
- `_export_spectra(s, acquisitions, datatypes, high_ppm, low_ppm, n_fft_pts, normalise)` - Export single spectrum
- `_export_concentrations(c, metabolites, norm)` - Export concentrations

**Private Methods:**
- `_check_export(d_inp, d_out, metabolites, high_ppm, low_ppm, n_fft_pts, norm, acquisitions, datatype, normalise, verbose)` - Check export correctness

### Neural Network Models

#### CNN Models (`mrsnet.cnn`)

#### `CNN` Class
Convolutional Neural Network for concentration prediction.

**Methods:**
- `__init__(model, metabolites, pulse_sequence, acquisitions, datatype, norm)` - Initialize CNN
- `reset()` - Reset model state
- `train(d_data, v_data, epochs, batch_size, folder, verbose, image_dpi, screen_dpi, train_dataset_name)` - Train model
- `predict(d_inp, reshape, verbose)` - Predict concentrations
- `save(folder)` - Save trained model

**Static Methods:**
- `load(path)` - Load trained model

**Private Methods:**
- `_freq_conv_layer(filter, c, s, dropout)` - Add frequency convolution layer
- `_construct(input_shape, output_shape)` - Construct CNN architecture
- `_save_results(folder, history, d_score, v_score, image_dpi, screen_dpi, verbose)` - Save training results

#### `TimeHistory` Class
Callback for tracking training time.

**Methods:**
- `__init__(epochs)` - Initialize timer
- `on_train_begin(logs)` - Called at training start
- `on_epoch_begin(batch, logs)` - Called at epoch start
- `on_epoch_end(batch, logs)` - Called at epoch end

#### Autoencoder Models (`mrsnet.autoencoder`)

#### `FCAutoEnc` Class
Fully connected autoencoder model.

**Methods:**
- `__init__(n_specs, n_freqs, layers_enc, layers_dec, activation, activation_last, dropout, name)` - Initialize autoencoder
- `call(x)` - Forward pass

#### `EncQuant` Class
Encoder-quantifier model.

**Methods:**
- `__init__(encoder, n_specs, n_freqs, output_conc, units, layers, act, act_last, dp, name)` - Initialize quantifier
- `call(x)` - Forward pass

#### `Autoencoder` Class
Main autoencoder interface.

**Methods:**
- `__init__(model, metabolites, pulse_sequence, acquisitions, datatype, norm, encoder, encoder_model, encoder_train_dataset_name)` - Initialize autoencoder
- `reset()` - Reset model state
- `train(d_data, v_data, epochs, batch_size, folder, verbose, image_dpi, screen_dpi, train_dataset_name)` - Train model
- `predict(spec_in, reshape, verbose)` - Predict output
- `save(folder)` - Save model

**Static Methods:**
- `load(path)` - Load trained model

**Private Methods:**
- `_construct(ae_shape, output_conc)` - Construct autoencoder architecture
- `_train_ae(d_data, v_data, epochs, batch_size, folder, verbose, image_dpi, screen_dpi, train_dataset_name, devices)` - Train autoencoder
- `_train_aeq(d_data, v_data, epochs, batch_size, folder, verbose, image_dpi, screen_dpi, train_dataset_name, devices)` - Train quantifier
- `_save_results(folder, prefix, history, d_score, v_score, loss, image_dpi, screen_dpi, verbose)` - Save training results

#### Autoencoder-Quantifier Models (`mrsnet.ae_quantifier`)

#### `FCAutoEncQuant` Class
Fully connected autoencoder-quantifier model.

**Methods:**
- `__init__(n_specs, n_freqs, layers_enc, layers_dec, activation, activation_last, dropout, output_conc, unit, layers, act, act_last, dp, name)` - Initialize model
- `call(x)` - Forward pass

#### `AutoencoderQuantifier` Class
Main autoencoder-quantifier interface.

**Methods:**
- `__init__(model, metabolites, pulse_sequence, acquisitions, datatype, norm)` - Initialize model
- `reset()` - Reset model state
- `train(d_data, v_data, epochs, batch_size, folder, verbose, image_dpi, screen_dpi, train_dataset_name)` - Train model
- `predict(spec_in, reshape, verbose)` - Predict output
- `save(folder)` - Save model

**Static Methods:**
- `load(path)` - Load trained model

**Private Methods:**
- `_construct(ae_shape, output_conc)` - Construct model architecture
- `_train_aeq(d_data, v_data, epochs, batch_size, folder, verbose, image_dpi, screen_dpi, train_dataset_name, devices)` - Train model
- `_save_results(folder, prefix, history, d_score, v_score, loss, image_dpi, screen_dpi, verbose)` - Save training results

### Training Utilities (`mrsnet.train`)

#### `Train` Class
Base class for training strategies.

**Methods:**
- `__init__(k)` - Initialize trainer
- `_plot_distributions(d_out, folder, image_dpi, screen_dpi, verbose)` - Plot output distributions
- `_cross_validate(model, epochs, batch_size, data, folder, train_dataset_name, verbose, image_dpi, screen_dpi)` - Perform cross-validation
- `_plot_cross_validate(model, train_res, val_res, has_error, folder, verbose, image_dpi, screen_dpi)` - Plot cross-validation results

#### `NoValidation` Class
Training without validation.

**Methods:**
- `__init__()` - Initialize no-validation trainer
- `train(model, data, epochs, batch_size, path_model, train_dataset_name, image_dpi, screen_dpi, shuffle, verbose)` - Train without validation

#### `Split` Class
Train/validation split.

**Methods:**
- `__init__(p)` - Initialize split trainer
- `train(model, data, epochs, batch_size, path_model, train_dataset_name, image_dpi, screen_dpi, shuffle, verbose)` - Train with split

#### `DuplexSplit` Class
Duplex train/validation split.

**Methods:**
- `__init__(p)` - Initialize duplex split trainer
- `train(model, data, epochs, batch_size, path_model, train_dataset_name, image_dpi, screen_dpi, shuffle, verbose)` - Train with duplex split

#### `KFold` Class
K-fold cross-validation.

**Methods:**
- `__init__(k)` - Initialize K-fold trainer
- `train(model, data, epochs, batch_size, path_model, train_dataset_name, image_dpi, screen_dpi, shuffle, verbose)` - Train with K-fold CV

#### `DuplexKFold` Class
Duplex K-fold cross-validation.

**Methods:**
- `__init__(k)` - Initialize duplex K-fold trainer
- `train(model, data, epochs, batch_size, path_model, train_dataset_name, image_dpi, screen_dpi, shuffle, verbose)` - Train with duplex K-fold CV

### Analysis and Evaluation (`mrsnet.analyse`)

#### Functions
- `analyse_model(model, inp, out, folder, prefix, id, save_conc, show_conc, norm, verbose, image_dpi, screen_dpi)` - Analyze model performance
- `_analyse_model_error(model, pre, out, folder, prefix, verbose, image_dpi, screen_dpi)` - Analyze concentration prediction errors
- `_analyse_spectra_error(model, pre, inp, out, folder, prefix, id, verbose, image_dpi, screen_dpi)` - Analyze spectra prediction errors
- `_plot_predicted_spectra(model, prefix, s, inp, pre, out)` - Plot predicted spectra

### Comparison Utilities (`mrsnet.compare`)

#### Functions
- `compare_basis(ds, basis, high_ppm, low_ppm, n_fft_pts, verbose, screen_dpi)` - Compare dataset to basis spectra
- `plot_diff_spectra(r, d, c, nu, metabolites, source, screen_dpi)` - Plot difference between spectra

### Grid Search (`mrsnet.grid`)

#### `Grid` Class
Key-value grid for parameter search.

**Methods:**
- `__init__(values)` - Initialize grid
- `__iter__()` - Iterate over grid combinations

**Static Methods:**
- `load(filename)` - Load grid from JSON file
- `all_combinations_sort(lst)` - Generate all combinations of list

#### `GridIterator` Class
Iterator for grid combinations.

**Methods:**
- `__init__(grid)` - Initialize iterator
- `__next__()` - Get next combination

### Model Selection (`mrsnet.selection`)

#### `Select` Class
Base class for model selection.

**Methods:**
- `__init__(remote, metabolites, dataset, epochs, validate, screen_dpi, image_dpi, verbose)` - Initialize selector
- `_add_task(key_vals, path_model)` - Add training task
- `_run_tasks(load_only)` - Run training tasks
- `_run(args)` - Run single training job
- `_run_remote(id, all)` - Run remote training job
- `_load_performance(model_path, fold)` - Load performance metrics
- `_save_performance(collection_name, var_keys, fix_keys)` - Save performance results

#### `SelectGrid` Class
Grid search model selection.

**Methods:**
- `__init__(metabolites, dataset, epochs, validate, remote, screen_dpi, image_dpi, verbose)` - Initialize grid selector
- `optimise(collection_name, models, path_model)` - Run grid search optimization

#### `SelectQMC` Class
Quasi-Monte Carlo model selection.

**Methods:**
- `__init__(metabolites, dataset, epochs, validate, repeats, remote, screen_dpi, image_dpi, verbose)` - Initialize QMC selector
- `optimise(collection_name, models, path_model)` - Run QMC optimization

#### `SelectGPO` Class
Gaussian Process Optimization model selection.

**Methods:**
- `__init__(metabolites, dataset, epochs, validate, repeats, remote, screen_dpi, image_dpi, verbose)` - Initialize GPO selector
- `optimise(collection_name, models, path_model)` - Run GPO optimization

#### `SelectGA` Class
Genetic Algorithm model selection.

**Methods:**
- `__init__(metabolites, dataset, epochs, validate, repeats, remote, screen_dpi, image_dpi, verbose)` - Initialize GA selector
- `optimise(collection_name, models, path_model)` - Run GA optimization

#### Helper Functions
- `_ga_fitness_func(solution, solution_idx)` - GA fitness function
- `_ga_on_generation(ga_instance)` - GA generation callback
- `_get_std_name(name)` - Get standard name from path

### File System Utilities (`mrsnet.getfolder`)

#### Functions
- `get_folder(folder, subfolder_pattern, timeout, delay)` - Get unique subfolder for data storage

## API Usage Patterns

The MRSNet API follows a consistent pattern across all modules:

1. **Configuration**: Initialize the configuration system with `Cfg.init()`
2. **Data Loading**: Use `Dataset.load()` or `Spectrum.load_*()` methods
3. **Model Creation**: Instantiate models (CNN, Autoencoder, etc.) with parameters
4. **Training**: Use training classes from `mrsnet.train` module
5. **Prediction**: Call `predict()` method on trained models
6. **Analysis**: Use `analyse.py` functions for evaluation

### Key Design Principles

- **Modular Architecture**: Each component (basis, spectrum, dataset, model) is self-contained
- **Consistent Interfaces**: All models implement similar `train()`, `predict()`, and `save()` methods
- **Flexible Configuration**: Configuration can be overridden via JSON files
- **Multiple Data Sources**: Support for LCModel, FID-A, PyGamma, and DICOM data
- **Extensible Models**: Easy to add new model architectures

## Usage Examples

### Basic Usage

```python
import mrsnet

# Initialize configuration
mrsnet.cfg.Cfg.init('/path/to/mrsnet')

# Load dataset
from mrsnet.dataset import Dataset
ds = Dataset.load('/path/to/dataset')

# Train CNN model
from mrsnet.cnn import CNN
model = CNN('cnn_small_softmax', ['Cr', 'GABA', 'Glu', 'Gln', 'NAA'],
            'megapress', ['edit_off', 'difference'], ['magnitude', 'phase'], 'sum')

# Train with K-fold cross-validation
from mrsnet.train import KFold
trainer = KFold(k=5)
trainer.train(model, data, epochs=100, batch_size=16,
              path_model='/path/to/models', verbose=1)

# Quantify spectra
predictions = model.predict(spectra_data)
```

### Command Line Usage

```bash
# Generate basis spectra
python mrsnet.py basis --source lcmodel --metabolites Cr GABA Glu Gln NAA

# Simulate dataset
python mrsnet.py simulate --source lcmodel --sample random --num 1000

# Train model
python mrsnet.py train -d /path/to/dataset -e 100 --validate 5 -m cnn_small_softmax

# Quantify DICOM spectra
python mrsnet.py quantify -d /path/to/dicoms -m /path/to/model
```

## Dependencies

- Python 3.11+ (tested up to Python 3.13)
- TensorFlow 2.20
- NumPy 2.3.3
- SciPy 1.16.1
- Matplotlib 3.10
- Seaborn 0.13.2
- scikit-learn 1.7.2
- joblib 1.4.2
- tqdm 4.66.4
- sobol_seq 0.2.0
- PyTorch 2.8.0
- Optuna 4.5.0
- pydicom 2.4.4
- pygad 3.3.1
- lmfit 1.3.2
- pyfftw 0.15.0
- GPyOpt (optional, commented out in requirements.txt)
- PyGamma (optional, commented out in requirements.txt)

## License

AGPL-3.0-or-later

## Authors

- Max Chandler (Cardiff University)
- Frank C Langbein (Cardiff University)
- Sophie M Shermer (Swansea University)
- Christopher W Jenkins (Swansea University)
- Zien Ma (Cardiff University)
