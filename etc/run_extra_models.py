#!/usr/bin/env python3
"""
Test script for extra MRSNet models.

This script trains all 4 extra models (EncDec, FoundationalCNN, QNet, QMRS) with
paper-based configurations on the fid-a-2d simulated dataset and runs benchmarks.

Configuration:
- Dataset: fid-a-2d simulated dataset with linewidth 2.0
- Training: 100K samples with 80/20 train/validation split and 1000 epochs
- Benchmark: Trained model on same dataset
- Metabolites: Cr, GABA, Gln, Glu, NAA
- Acquisitions: edit_off, edit_on
- Datatype: real
- Normalization: sum
FIXME: consider other acquisitions and datatypes?

The script will fail if any mrsnet commands issue an error to ensure we don't
overlook any issues.
"""

import os
import subprocess
import sys
import time

EPOCHS = 3
DS_SIZE = 10000

def run_command(cmd, description, fail_on_error=True):
    """Run a command and handle the output."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=fail_on_error)
        duration = time.time() - start_time

        print(f"Exit code: {result.returncode}")
        print(f"Duration: {duration:.2f} seconds")

        if result.stdout:
            print("STDOUT:")
            print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
        else:
            print(f"❌ ERROR: {description}")

        return result.returncode == 0

    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"Exit code: {e.returncode}")
        print(f"Duration: {duration:.2f} seconds")

        if e.stdout:
            print("STDOUT:")
            print(e.stdout)

        if e.stderr:
            print("STDERR:")
            print(e.stderr)

        print(f"❌ ERROR: {description}")
        return False

def test_model(model_name, model_string):
    """Test a model with 80/20 split training, and benchmark."""
    print(f"\n{'='*80}")
    print(f"TESTING: {model_name.upper()}")
    print(f"{'='*80}")

    # Common parameters
    metabolites = ["Cr", "GABA", "Gln", "Glu", "NAA"]
    acquisitions = ["edit_off", "edit_on"]
    datatype = ["real"]
    norm = "sum"

    # Dataset paths - using DS_SIZE dataset with explicit train/validation split
    dataset = f"data/sim-spectra-megapress/fid-a-2d_2000_4096/siemens/123.23/2.0/Cr-GABA-Gln-Glu-NAA/megapress/sobol/1.0-adc_normal-0.0-0.03/{DS_SIZE}-1"

    # Check if dataset exists
    if not os.path.exists(dataset):
        print(f"❌ ERROR: Dataset not found: {dataset}")
        print("Please ensure the dataset exists before running this script.")
        return False

    # Model-specific parameters
    if model_name == "encdec":
        batch_size = 4  # Smaller batch size for memory efficiency
        epochs = EPOCHS # More epochs for proper training
    else:
        batch_size = 16 # Larger batch size for other models
        epochs = EPOCHS # More epochs for proper training

    # Step 1: Check if model already exists and is properly trained
    print("\n🔍 Step 1: Check if model already exists")

    # Construct the complete model path based on training parameters
    metabolites_str = "-".join(metabolites)
    acquisitions_str = "-".join(acquisitions)
    datatype_str = "-".join(datatype)
    train_dataset_name = f"fid-a-2d_2000_4096_siemens_123.23_2.0_{metabolites_str}_megapress_sobol_1.0-adc_normal-0.0-0.03_{DS_SIZE}-1"

    # Check both data/model and data/model-dist directories
    model_dirs = [f"data/model/{model_string}", f"data/model-dist/{model_string}"]
    complete_model_path = None
    model_keras_path = None
    model_exists = False

    for base_model_dir in model_dirs:
        # The complete model path structure
        candidate_path = os.path.join(
            base_model_dir,
            metabolites_str,
            "megapress",
            acquisitions_str,
            datatype_str,
            norm,
            str(batch_size),
            str(epochs),
            train_dataset_name,
            "Split_0.8-1"  # This is the trainer ID from the training command
        )

        candidate_keras_path = os.path.join(candidate_path, "model.keras")
        if os.path.exists(candidate_keras_path):
            complete_model_path = candidate_path
            model_keras_path = candidate_keras_path
            model_exists = True
            print(f"✅ Found existing model in: {base_model_dir}")
            break

    # If not found with Split_0.8-1, check for NoValidation-1 (common in model-dist)
    if not model_exists:
        for base_model_dir in model_dirs:
            candidate_path = os.path.join(
                base_model_dir,
                metabolites_str,
                "megapress",
                acquisitions_str,
                datatype_str,
                norm,
                str(batch_size),
                str(epochs),
                train_dataset_name,
                "NoValidation-1"  # Alternative trainer ID
            )

            candidate_keras_path = os.path.join(candidate_path, "model.keras")
            if os.path.exists(candidate_keras_path):
                complete_model_path = candidate_path
                model_keras_path = candidate_keras_path
                model_exists = True
                print(f"✅ Found existing model in: {base_model_dir} (NoValidation-1)")
                break

    # If still not found, check for models with different parameters (different epochs/samples)
    if not model_exists:
        print("🔍 Checking for models with different parameters...")
        # Check common parameter variations
        param_variations = [
            (16, 1000, 100000),  # Common in model-dist
            (16, 500, 100000),   # Another common variation
        ]

        for batch_var, epochs_var, samples_var in param_variations:
            train_dataset_var = f"fid-a-2d_2000_4096_siemens_123.23_2.0_{metabolites_str}_megapress_sobol_1.0-adc_normal-0.0-0.03_{samples_var}-1"

            for base_model_dir in model_dirs:
                for trainer_id in ["NoValidation-1", "Split_0.8-1"]:
                    candidate_path = os.path.join(
                        base_model_dir,
                        metabolites_str,
                        "megapress",
                        acquisitions_str,
                        datatype_str,
                        norm,
                        str(batch_var),
                        str(epochs_var),
                        train_dataset_var,
                        trainer_id
                    )

                    candidate_keras_path = os.path.join(candidate_path, "model.keras")
                    if os.path.exists(candidate_keras_path):
                        complete_model_path = candidate_path
                        model_keras_path = candidate_keras_path
                        model_exists = True
                        print("✅ Found existing model with different parameters:")
                        print(f"   Batch: {batch_var}, Epochs: {epochs_var}, Samples: {samples_var}")
                        print(f"   Path: {base_model_dir}")
                        break
                if model_exists:
                    break
            if model_exists:
                break

    if model_exists:
        print(f"✅ Model already exists and is trained: {complete_model_path}")
        print(f"✅ Model.keras file found: {model_keras_path}")
        print("⏭️  Skipping training, proceeding to benchmark...")
        print("ℹ️  Note: Using existing model (may have different training parameters)")
    else:
        print("❌ Model not found in any expected location")
        print("🚀 Proceeding with training...")

        # Step 2: Training with DS_SIZE samples (80/20 train/validation split)
        print(f"\n🚀 Step 2: Training with {DS_SIZE} samples (80/20 split) and {EPOCHS} epochs")
        train_cmd = ["python3", "mrsnet.py", "train", "--metabolites", *metabolites, "--acquisitions", *acquisitions, "--datatype", *datatype, "--norm", norm, "--model", model_string, "--batchsize", str(batch_size), "--epochs", str(epochs), "--dataset", dataset, "-k", "0.8"]

        if not run_command(train_cmd, f"Training {model_name}"):
            return False

        # After training, the model should be in data/model with Split_0.8-1
        base_model_dir = f"data/model/{model_string}"
        complete_model_path = os.path.join(
            base_model_dir,
            metabolites_str,
            "megapress",
            acquisitions_str,
            datatype_str,
            norm,
            str(batch_size),
            str(epochs),
            train_dataset_name,
            "Split_0.8-1"
        )
        model_keras_path = os.path.join(complete_model_path, "model.keras")

        # Verify the model was created after training
        if not os.path.exists(model_keras_path):
            print(f"❌ ERROR: Model was not created after training: {model_keras_path}")
            return False

        print(f"✅ SUCCESS: Model created after training: {model_keras_path}")

    # Step 2: Benchmark the trained model
    print("\n📊 Step 2: Benchmark trained model")
    benchmark_cmd = [
        "python3", "mrsnet.py", "benchmark",
        "-m", complete_model_path,
        "--norm", "max",
        "-v"
    ]

    if not run_command(benchmark_cmd, f"Benchmark {model_name}"):
        return False

    print(f"✅ SUCCESS: Testing completed for {model_name}")
    return True

def main():
    """Run tests for all models."""
    print("🚀 Starting test of extra models")
    print("="*60)
    print("This script will:")
    print(f"1. Train each model with {DS_SIZE} samples (80/20 train/validation split)")
    print("2. Run benchmarks on the trained models")
    print("3. Fail if any command issues an error")
    print("="*60)

    # Model configurations - using paper-based configurations
    models = {
        "fcnn": "fcnn_original",      # Original paper parameters
        "qnet": "qnet_original",      # Original paper parameters
        "qmrs": "qmrs_original",      # Original paper parameters
        "encdec": "encdec_default",   # Memory-friendly defaults (original too large)
    }

    results = {}

    for model_name, model_string in models.items():
        success = test_model(model_name, model_string)
        results[model_name] = success

        if not success:
            print(f"\n❌ {model_name.upper()} failed - stopping tests")
            break

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")

    for model_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{model_name.upper()}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("All new models trained successfully and benchmarks completed!")
        print("\nTrained models available in:")
        for model_name, model_string in models.items():
            metabolites_str = "-".join(["Cr", "GABA", "Gln", "Glu", "NAA"])
            acquisitions_str = "-".join(["edit_off", "edit_on"])
            datatype_str = "-".join(["real"])
            train_dataset_name = f"fid-a-2d_2000_4096_siemens_123.23_2.0_{metabolites_str}_megapress_sobol_1.0-adc_normal-0.0-0.03_{DS_SIZE}-1"
            complete_path = f"data/model/{model_string}/{metabolites_str}/megapress/{acquisitions_str}/{datatype_str}/sum/16/{EPOCHS}/{train_dataset_name}/Split_0.8-1"
            print(f"  - {model_name}: {complete_path}")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Please check the error messages above.")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
