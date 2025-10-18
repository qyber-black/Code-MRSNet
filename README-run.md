# run.py - Generic MRSNet Execution Script

> SPDX-FileCopyrightText: Copyright (C) 2025 Frank C Langbein <frank@langbein.org>, Cardiff University\
> SPDX-License-Identifier: AGPL-3.0-or-later


The `run.py` script provides a flexible way to execute MRSNet commands using JSON configuration files. It automatically checks for existing results to avoid re-execution and supports dependency management between runs.

## Features

- **JSON Configuration**: Define multiple MRSNet operations in a single JSON file
- **Result Checking**: Automatically detects existing models and benchmark results
- **Dependency Management**: Run operations in the correct order based on dependencies
- **Dry Run Mode**: Preview commands without executing them
- **Force Mode**: Override result checking and re-run all operations
- **Comprehensive Logging**: Detailed output showing progress and results

## Usage

```bash
# Basic usage
python run.py config.json

# Preview commands without executing
python run.py config.json --dry-run

# Force re-execution of all commands
python run.py config.json --force

# Show help
python run.py --help
```

## JSON Configuration Format

The JSON configuration file has two main sections:

### `common` Section
Contains arguments that apply to all runs, including the command:

```json
{
  "common": {
    "command": "train",
    "metabolites": ["Cr", "GABA", "Gln", "Glu", "NAA"],
    "dataset": "data/sim-spectra-megapress/fid-a-2d_2000_4096/siemens/123.23/2.0/Cr-GABA-Gln-Glu-NAA/megapress/sobol/1.0-adc_normal-0.0-0.03/10000-1",
    "epochs": 1000,
    "batchsize": 16,
    "norm": "sum",
    "verbose": 1
  }
}
```

**Note**: The `command` can be specified in either the `common` section (applies to all runs) or in individual run configurations (overrides the common command).

### `runs` Section

Contains an array of individual operations to execute:

```json
{
  "runs": [
    {
      "name": "cnn_training",
      "command": "train",
      "args": {
        "model": "cnn_default",
        "acquisitions": ["edit_off", "edit_on"],
        "datatype": ["real"],
        "validate": 0.8
      }
    },
    {
      "name": "cnn_benchmark",
      "command": "benchmark",
      "args": {
        "model": "data/model/cnn_default/...",
        "norm": "max"
      },
      "depends_on": "cnn_training"
    }
  ]
}
```

### Run Configuration Fields

- **`name`** (required): Unique identifier for the run
- **`command`** (optional): MRSNet command to execute (`train`, `benchmark`, `simulate`, `basis`, etc.). If not specified, uses the command from the `common` section
- **`args`** (optional): Command-specific arguments that override or supplement common arguments
- **`depends_on`** (optional): Name of another run that must complete successfully first

## Supported Commands

The script supports all MRSNet commands:

- **`train`**: Train models on datasets
- **`benchmark`**: Run benchmarks on trained models
- **`simulate`**: Generate simulated datasets
- **`basis`**: Generate basis sets
- **`compare`**: Compare spectra
- **`quantify`**: Quantify spectra in DICOMs
- **`select`**: Model selection and hyperparameter optimization

## Result Checking

The script automatically checks for existing results to avoid re-execution:

### Training Results

For `train` commands, the script checks for existing model files in:
- `data/model/`
- `data/model-dist/`
- `data/model-ae/`
- `data/model-cae/`
- `data/model-cnn/`

It looks for `model.keras` files in the expected directory structure based on the training parameters.

### Benchmark Results

For `benchmark` commands, completion is determined from the benchmark sequence definitions used by `mrsnet.py`:
- Reads `benchmark_sequences.json` from the configured benchmark root (`path_benchmark` in `cfg.json`, fallback: `data/benchmark`).
- For each sequence and variant in that file, expects an analyse output file in the model folder matching:
  - `<b_id>_<variant>_<norm>_concentration_errors.json` (or any norm if `norm` is `default`).
- Also expects the aggregated file:
  - `benchmark_all_<norm>_concentration_errors.json` (or any norm if `norm` is `default`).

This mirrors the outputs written by `analyse_model` during `mrsnet.py benchmark`.

## Example Configurations

### Common Command with Multiple Runs

```json
{
  "common": {
    "command": "train",
    "metabolites": ["Cr", "GABA", "Gln", "Glu", "NAA"],
    "dataset": "data/sim-spectra-megapress/fid-a-2d_2000_4096/siemens/123.23/2.0/Cr-GABA-Gln-Glu-NAA/megapress/sobol/1.0-adc_normal-0.0-0.03/10000-1",
    "epochs": 1000,
    "batchsize": 16,
    "norm": "sum",
    "validate": 0.8,
    "verbose": 1
  },
  "runs": [
    {
      "name": "cnn_real",
      "args": {
        "model": "cnn_default",
        "acquisitions": ["edit_off", "edit_on"],
        "datatype": ["real"]
      }
    },
    {
      "name": "cnn_imaginary",
      "args": {
        "model": "cnn_default",
        "acquisitions": ["edit_off", "edit_on"],
        "datatype": ["real", "imaginary"]
      }
    },
    {
      "name": "fcnn_real",
      "args": {
        "model": "fcnn_default",
        "acquisitions": ["edit_off", "edit_on"],
        "datatype": ["real"]
      }
    }
  ]
}
```

### Mixed Commands (Common + Override)

```json
{
  "common": {
    "command": "train",
    "metabolites": ["Cr", "GABA", "Gln", "Glu", "NAA"],
    "dataset": "data/sim-spectra-megapress/fid-a-2d_2000_4096/siemens/123.23/2.0/Cr-GABA-Gln-Glu-NAA/megapress/sobol/1.0-adc_normal-0.0-0.03/10000-1",
    "epochs": 1000,
    "batchsize": 16,
    "norm": "sum",
    "validate": 0.8,
    "verbose": 1
  },
  "runs": [
    {
      "name": "cnn_training",
      "args": {
        "model": "cnn_default",
        "acquisitions": ["edit_off", "edit_on"],
        "datatype": ["real"]
      }
    },
    {
      "name": "cnn_benchmark",
      "command": "benchmark",
      "args": {
        "model": "data/model/cnn_default/...",
        "norm": "max"
      },
      "depends_on": "cnn_training"
    }
  ]
}
```

### Basic Training and Benchmarking

```json
{
  "common": {
    "metabolites": ["Cr", "GABA", "Gln", "Glu", "NAA"],
    "dataset": "data/sim-spectra-megapress/fid-a-2d_2000_4096/siemens/123.23/2.0/Cr-GABA-Gln-Glu-NAA/megapress/sobol/1.0-adc_normal-0.0-0.03/10000-1",
    "epochs": 1000,
    "batchsize": 16,
    "norm": "sum",
    "verbose": 1
  },
  "runs": [
    {
      "name": "cnn_training",
      "command": "train",
      "args": {
        "model": "cnn_default",
        "acquisitions": ["edit_off", "edit_on"],
        "datatype": ["real"],
        "validate": 0.8
      }
    },
    {
      "name": "cnn_benchmark",
      "command": "benchmark",
      "args": {
        "norm": "max",
        "model": "data/model/cnn_default/Cr-GABA-Gln-Glu-NAA/megapress/edit_off-edit_on/real/sum/16/1000/fid-a-2d_2000_4096_siemens_123.23_2.0_Cr-GABA-Gln-Glu-NAA_megapress_sobol_1.0-adc_normal-0.0-0.03_10000-1/Split_0.8-1"
      },
      "depends_on": "cnn_training"
    }
  ]
}
```

### Multiple Model Training

```json
{
  "common": {
    "metabolites": ["Cr", "GABA", "Gln", "Glu", "NAA"],
    "dataset": "data/sim-spectra-megapress/fid-a-2d_2000_4096/siemens/123.23/2.0/Cr-GABA-Gln-Glu-NAA/megapress/sobol/1.0-adc_normal-0.0-0.03/10000-1",
    "epochs": 2000,
    "batchsize": 16,
    "norm": "sum",
    "validate": 0,
    "verbose": 1
  },
  "runs": [
    {
      "name": "fcnn_training",
      "command": "train",
      "args": {
        "model": "fcnn_default",
        "acquisitions": ["edit_off", "edit_on"],
        "datatype": ["real"]
      }
    },
    {
      "name": "qmrs_training",
      "command": "train",
      "args": {
        "model": "qmrs_default",
        "acquisitions": ["edit_off", "edit_on"],
        "datatype": ["real"]
      }
    },
    {
      "name": "fcnn_benchmark",
      "command": "benchmark",
      "args": {
        "norm": "max"
      },
      "depends_on": "fcnn_training"
    },
    {
      "name": "qmrs_benchmark",
      "command": "benchmark",
      "args": {
        "norm": "max"
      },
      "depends_on": "qmrs_training"
    }
  ]
}
```

### Simulation Pipeline

```json
{
  "common": {
    "metabolites": ["Cr", "GABA", "Gln", "Glu", "NAA"],
    "source": "fid-a-2d",
    "manufacturer": "siemens",
    "omega": 123.23,
    "linewidth": 2.0,
    "pulse_sequence": "megapress",
    "sample_rate": 2000,
    "samples": 4096,
    "verbose": 1
  },
  "runs": [
    {
      "name": "generate_basis",
      "command": "basis",
      "args": {
        "metabolites": ["Cr", "GABA", "Gln", "Glu", "NAA"]
      }
    },
    {
      "name": "simulate_dataset",
      "command": "simulate",
      "args": {
        "metabolites": ["Cr", "GABA", "Gln", "Glu", "NAA"],
        "n_samples": 10000,
        "noise_level": 0.03,
        "adc_noise": 0.0,
        "method": "sobol"
      },
      "depends_on": "generate_basis"
    }
  ]
}
```

## Output

The script provides detailed output including:

- Configuration summary
- Individual run progress
- Command execution details
- Success/failure status
- Final summary with statistics

Example output:
```
🚀 Starting MRSNet execution
============================================================
Configuration: config.json
Dry run: False
Force: False
Total runs: 4
============================================================

====================================================================================================
RUN 1/4: CNN_TRAINING
Command: train
====================================================================================================
✅ Found existing model: data/model/cnn_default/.../model.keras
⏭️  Skipping cnn_training - results already exist

====================================================================================================
EXECUTION SUMMARY
====================================================================================================
Successful: 4/4
Failed: 0/4

Detailed Results:
  cnn_training: ✅ SUCCESS
  cnn_benchmark: ✅ SUCCESS

====================================================================================================
🎉 ALL RUNS COMPLETED SUCCESSFULLY!
```

## Error Handling

The script handles various error conditions:

- **Configuration errors**: Invalid JSON, missing required fields
- **Dependency failures**: Runs that depend on failed runs are skipped
- **Command execution errors**: Individual command failures don't stop the entire process
- **File system errors**: Missing datasets, permission issues, etc.
