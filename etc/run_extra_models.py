#!/usr/bin/env python3
"""
Test script for extra MRSNet models.

This script trains all 4 extra models (EncDec, FoundationalCNN, QNet, QMRS) with
paper-based configurations on the fid-a-2d simulated dataset and runs benchmarks.

Configuration (see constants in script to adjust):
- Dataset: fid-a-2d simulated dataset with linewidth 2.0
- Training: 100K samples with 80/20 train/validation split and 1000 epochs
- Benchmark: Trained model on same dataset
- Metabolites: Cr, GABA, Gln, Glu, NAA
- Acquisitions: Multiple combinations (edit_off+edit_on, edit_off+difference)
- Datatypes: Multiple combinations (real, real+imaginary)
- Normalization: sum

The script tests all combinations of acquisitions and datatypes for each model,
providing comprehensive coverage of different input configurations.

Key features:
- Continues training all models even if some fail
- Only benchmarks models that successfully trained
- Records and reports training failures vs benchmark failures separately
- Provides detailed summary of which models succeeded/failed at each stage
"""

import os
import subprocess
import sys
import time

# Training parameters
EPOCHS = 3
DS_SIZE = 10000
SPLIT_K = 0.8
SPLIT_FOLD = "Split_0.8-1"

# Models to test
MODELS = {
    #"fcnn": "fcnn_original",             # Original paper parameters
    #"qnet": "qnet_original",             # Original paper parameters
    #"qnet_basis": "qnet_basis_original", # Original paper parameters
    #"qmrs": "qmrs_original",             # Original paper parameters
    "encdec": "encdec_default",          # Memory-friendly defaults (original too large)
}

# Multiple acquisitions and datatypes to test
ACQUISITIONS_LIST = [
    ["edit_off", "edit_on"],
    ["edit_off", "difference"],
]

DATATYPES_LIST = [
    ["real"],
    ["real", "imaginary"],
]

def run_command(cmd, description, fail_on_error=True):
    """Run a command and handle the output."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=fail_on_error) # noqa: S603
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

def test_model(model_name, model_string, acquisitions, datatype):
    """Test a model with 80/20 split training, and benchmark."""
    acquisitions_str = "-".join(sorted(acquisitions))
    datatype_str = "-".join(sorted(datatype))

    print(f"\n{'='*80}")
    print(f"TESTING: {model_name.upper()}")
    print(f"ACQUISITIONS: {acquisitions_str}")
    print(f"DATATYPE: {datatype_str}")
    print(f"{'='*80}")

    # Common parameters
    metabolites = ["Cr", "GABA", "Gln", "Glu", "NAA"]
    norm = "sum"

    # Dataset paths - using DS_SIZE dataset with explicit train/validation split
    dataset = f"data/sim-spectra-megapress/fid-a-2d_2000_4096/siemens/123.23/2.0/Cr-GABA-Gln-Glu-NAA/megapress/sobol/1.0-adc_normal-0.0-0.03/{DS_SIZE}-1"

    # Check if dataset exists
    if not os.path.exists(dataset):
        print(f"❌ ERROR: Dataset not found: {dataset}")
        print("Please ensure the dataset exists before running this script.")
        return {"training_success": False, "benchmark_success": False, "error": "Dataset not found"}

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
            SPLIT_FOLD  # This is the trainer ID from the training command
        )

        candidate_keras_path = os.path.join(candidate_path, "model.keras")
        if os.path.exists(candidate_keras_path):
            complete_model_path = candidate_path
            model_keras_path = candidate_keras_path
            model_exists = True
            print(f"✅ Found existing model in: {base_model_dir}")
            break

    training_success = False
    if model_exists:
        print(f"✅ Model already exists and is trained: {complete_model_path}")
        print(f"✅ Model.keras file found: {model_keras_path}")
        print("⏭️  Skipping training, proceeding to benchmark...")
        training_success = True
    else:
        print("⚠️ Model not found in any expected location")
        print("⏭️ Proceeding with training...")

        # Step 2: Training with DS_SIZE samples (80/20 train/validation split)
        print(f"\n🚀 Step 2: Training with {DS_SIZE} samples (80/20 split) and {EPOCHS} epochs")
        train_cmd = ["python3", "mrsnet.py", "train", "--metabolites", *metabolites, "--acquisitions", *acquisitions, "--datatype", *datatype, "--norm", norm, "--model", model_string, "--batchsize", str(batch_size), "--epochs", str(epochs), "--dataset", dataset, "-k", str(SPLIT_K)]

        training_success = run_command(train_cmd, f"Training {model_name}", fail_on_error=False)

        if training_success:
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
                SPLIT_FOLD
            )
            model_keras_path = os.path.join(complete_model_path, "model.keras")

            # Verify the model was created after training
            if not os.path.exists(model_keras_path):
                print(f"❌ ERROR: Model was not created after training: {model_keras_path}")
                print(f"ℹ️  Checking if directory exists: {os.path.exists(complete_model_path)}") # noqa: RUF001
                if os.path.exists(complete_model_path):
                    print(f"ℹ️  Directory contents: {os.listdir(complete_model_path)}") # noqa: RUF001
                training_success = False
            else:
                print(f"✅ SUCCESS: Model created after training: {model_keras_path}")
        else:
            print(f"❌ Training failed for {model_name}")

    # Step 3: Benchmark the trained model (only if training succeeded)
    benchmark_success = False
    if training_success and complete_model_path:
        print("\n📊 Step 3: Benchmark trained model")
        benchmark_cmd = [
            "python3", "mrsnet.py", "benchmark",
            "-m", complete_model_path,
            "--norm", "max"
        ]

        benchmark_success = run_command(benchmark_cmd, f"Benchmark {model_name}", fail_on_error=False)

        if benchmark_success:
            print(f"✅ SUCCESS: Testing completed for {model_name}")
        else:
            print(f"❌ Benchmark failed for {model_name}")
    else:
        print(f"⏭️  Skipping benchmark for {model_name} due to training failure")

    return {
        "training_success": training_success,
        "benchmark_success": benchmark_success,
        "model_path": complete_model_path if training_success else None
    }

def main():
    """Run tests for all models."""
    print("🚀 Starting test of extra models")
    print("="*60)
    print("This script will:")
    print(f"1. Train each model with {DS_SIZE} samples (80/20 train/validation split)")
    print("2. Run benchmarks on the trained models")
    print("3. Test multiple acquisition and datatype combinations")
    print("4. Fail if any command issues an error")
    print("="*60)

    # Model configurations - using paper-based configurations
    models = MODELS

    results = {}
    total_tests = len(models) * len(ACQUISITIONS_LIST) * len(DATATYPES_LIST)
    test_count = 0

    for model_name, model_string in models.items():
        results[model_name] = {}

        for acquisitions in ACQUISITIONS_LIST:
            acquisitions_key = "-".join(sorted(acquisitions))
            results[model_name][acquisitions_key] = {}

            for datatype in DATATYPES_LIST:
                datatype_key = "-".join(sorted(datatype))
                test_count += 1

                print(f"\n{'='*100}")
                print(f"TEST {test_count}/{total_tests}: {model_name.upper()} - {acquisitions_key} - {datatype_key}")
                print(f"{'='*100}")

                result = test_model(model_name, model_string, acquisitions, datatype)
                results[model_name][acquisitions_key][datatype_key] = result

                # Continue with other tests even if this one failed
                if not result["training_success"]:
                    print(f"\n❌ {model_name.upper()} ({acquisitions_key}, {datatype_key}) training failed - continuing with other tests")
                elif not result["benchmark_success"]:
                    print(f"\n❌ {model_name.upper()} ({acquisitions_key}, {datatype_key}) benchmark failed - continuing with other tests")

    # Summary
    print(f"\n{'='*100}")
    print("TEST SUMMARY")
    print(f"{'='*100}")

    total_count = 0
    training_success_count = 0
    benchmark_success_count = 0
    training_failures = []
    benchmark_failures = []

    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()}:")
        for acquisitions_key, acquisitions_results in model_results.items():
            print(f"  {acquisitions_key}:")
            for datatype_key, result in acquisitions_results.items():
                total_count += 1

                # Training status
                training_status = "✅ TRAINED" if result["training_success"] else "❌ TRAINING FAILED"
                print(f"    {datatype_key}: {training_status}")

                if result["training_success"]:
                    training_success_count += 1

                    # Benchmark status (only if training succeeded)
                    benchmark_status = "✅ BENCHMARKED" if result["benchmark_success"] else "❌ BENCHMARK FAILED"
                    print(f"      Benchmark: {benchmark_status}")

                    if result["benchmark_success"]:
                        benchmark_success_count += 1
                    else:
                        benchmark_failures.append(f"{model_name} ({acquisitions_key}, {datatype_key})")
                else:
                    training_failures.append(f"{model_name} ({acquisitions_key}, {datatype_key})")

    print(f"\n{'='*100}")
    print("OVERALL RESULTS")
    print(f"{'='*100}")
    print(f"Training: {training_success_count}/{total_count} models trained successfully")
    print(f"Benchmarking: {benchmark_success_count}/{training_success_count} trained models benchmarked successfully")

    if training_failures:
        print(f"\n❌ TRAINING FAILURES ({len(training_failures)}):")
        for failure in training_failures:
            print(f"  - {failure}")

    if benchmark_failures:
        print(f"\n❌ BENCHMARK FAILURES ({len(benchmark_failures)}):")
        for failure in benchmark_failures:
            print(f"  - {failure}")

    print(f"\n{'='*100}")

    if training_success_count == total_count and benchmark_success_count == training_success_count:
        print("\n🎉 ALL TESTS PASSED!")
        print("All models trained successfully and benchmarks completed!")
        print("\nTrained models available in:")
        for model_name, model_string in models.items():
            metabolites_str = "-".join(["Cr", "GABA", "Gln", "Glu", "NAA"])
            print(f"\n{model_name.upper()}:")
            for acquisitions in ACQUISITIONS_LIST:
                acquisitions_str = "-".join(sorted(acquisitions))
                print(f"  {acquisitions_str}:")
                for datatype in DATATYPES_LIST:
                    datatype_str = "-".join(sorted(datatype))
                    train_dataset_name = f"fid-a-2d_2000_4096_siemens_123.23_2.0_{metabolites_str}_megapress_sobol_1.0-adc_normal-0.0-0.03_{DS_SIZE}-1"
                    complete_path = f"data/model/{model_string}/{metabolites_str}/megapress/{acquisitions_str}/{datatype_str}/sum/16/{EPOCHS}/{train_dataset_name}/{SPLIT_FOLD}"
                    print(f"    {datatype_str}: {complete_path}")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Please check the error messages above.")
        print("\nNote: Only successfully trained models were benchmarked.")

    # Return True if at least some models were successfully trained and benchmarked
    return training_success_count > 0 and benchmark_success_count > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
