#!/usr/bin/env python3
#
# run.py - Generic MRSNet execution script with JSON configuration
#
# SPDX-FileCopyrightText: Copyright (C) 2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This script executes MRSNet commands based on JSON configuration files.
# It supports checking for existing results to avoid re-execution and
# provides a flexible way to run multiple MRSNet operations with common
# and specific arguments.

"""
Generic MRSNet execution script with JSON configuration.

This script allows you to define MRSNet operations in JSON format and execute
them with automatic result checking to avoid re-running completed operations.

JSON Format:
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
        "model": "cnn_default",
        "norm": "max"
      },
      "depends_on": "cnn_training"
    }
  ]
}

Usage:
    python run.py config.json
    python run.py config.json --dry-run
    python run.py config.json --force
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any


class MRSNetRunner:
    """MRSNet execution runner with result checking."""

    def __init__(self, config_file: str, dry_run: bool = False, force: bool = False):
        """Initialize the runner.

        Args:
            config_file: Path to JSON configuration file
            dry_run: If True, only print commands without executing
            force: If True, ignore existing results and re-run
        """
        self.config_file = config_file
        self.dry_run = dry_run
        self.force = force
        self.config = self._load_config()
        self.results = {}  # Track execution results (bool)
        self.artifacts = {}  # Track discovered artifacts per run (e.g., model_path)

    def _load_config(self) -> dict[str, Any]:
        """Load and validate configuration file."""
        try:
            with open(self.config_file) as f:
                config = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}") from e

        # Validate required sections
        if 'runs' not in config:
            raise ValueError("Configuration must contain 'runs' section")
        if not isinstance(config['runs'], list):
            raise ValueError("'runs' must be a list")

        # Set default common section
        if 'common' not in config:
            config['common'] = {}

        return config

    def _merge_args(self, run_config: dict[str, Any]) -> dict[str, Any]:
        """Merge common and run-specific arguments.

        If run_config contains the flag '_only_args_no_common', only the run-specific
        'args' are returned without merging values from 'common'.
        """
        if run_config.get('_only_args_no_common'):
            return dict(run_config.get('args', {}))
        args = self.config['common'].copy()
        if 'args' in run_config:
            args.update(run_config['args'])
        return args

    def _build_command(self, run_config: dict[str, Any]) -> list[str]:
        """Build mrsnet.py command from run configuration."""
        # Get command from run config, fallback to common config
        command = run_config.get('command')
        if command is None:
            command = self.config['common'].get('command')
            if command is None:
                raise ValueError("Command must be specified in either run config or common config")

        args = self._merge_args(run_config)

        cmd = ['python3', 'mrsnet.py', command]

        # Filter arguments for commands that only accept a subset of common args
        # benchmark accepts only: --model, --norm and -v/--verbose
        if command == 'benchmark':
            allowed_keys = {'model', 'norm', 'verbose'}
            args = {k: v for k, v in args.items() if k in allowed_keys}

        # Add arguments to command
        for key, value in args.items():
            if value is None:
                continue

            # Skip 'command' as it's only used for internal logic
            if key == 'command':
                continue

            # Special handling: verbose is a flag without an argument in mrsnet.py
            if key == 'verbose':
                # If numeric, repeat -v that many times; if truthy, add once
                if isinstance(value, int):
                    for _ in range(max(0, value)):
                        cmd.append('-v')
                elif value:
                    cmd.append('-v')
                continue

            # Positional-only arguments for specific commands
            # mrsnet.py select expects 'collection' as a positional argument
            if key in ('collection',):
                if isinstance(value, list):
                    for v in value:
                        cmd.append(str(v))
                else:
                    cmd.append(str(value))
                continue

            # Handle list arguments (e.g., metabolites, acquisitions, datatype)
            if isinstance(value, list):
                if value:  # Only add if list is not empty
                    cmd.extend([f'--{key}'] + [str(v) for v in value])
            else:
                # Handle boolean flags
                if isinstance(value, bool):
                    if value:
                        cmd.append(f'--{key}')
                else:
                    cmd.extend([f'--{key}', str(value)])

        return cmd

    def _check_model_exists(self, run_config: dict[str, Any]) -> bool:
        """Check if a model already exists for training commands."""
        # Get command from run config, fallback to common config
        command = run_config.get('command')
        if command is None:
            command = self.config['common'].get('command')
            if command is None:
                return False

        if command != 'train':
            return False

        args = self._merge_args(run_config)

        # Extract key parameters
        model = args.get('model')
        metabolites = args.get('metabolites', [])
        dataset = args.get('dataset', '')
        epochs = args.get('epochs', 1000)  # noqa: F841
        batchsize = args.get('batchsize', 16)  # noqa: F841
        norm = args.get('norm', 'sum')  # noqa: F841
        validate = args.get('validate', None)  # noqa: F841

        if not model or not metabolites or not dataset:
            return False

        # Use helper to locate latest model folder containing model.keras
        latest_path = self._find_latest_model_path(args)
        if latest_path is not None:
            print(f"✅ Found existing model: {os.path.join(latest_path, 'model.keras')}")
            return True
        return False

    def _validate_kfold_completeness(self, trainer_path: str, expected_k: int) -> tuple[bool, list[int], list[int]]:
        """Validate that all expected KFold directories exist.

        Parameters
        ----------
            trainer_path (str): Path to the trained model directory
            expected_k (int): Expected number of folds

        Returns
        -------
            tuple: (is_complete, missing_folds, found_folds)
        """
        missing_folds = []
        found_folds = []

        for fold_idx in range(expected_k):
            fold_path = os.path.join(trainer_path, f"fold-{fold_idx}")
            if os.path.exists(fold_path):
                found_folds.append(fold_idx)
            else:
                missing_folds.append(fold_idx)

        is_complete = len(missing_folds) == 0
        return is_complete, missing_folds, found_folds

    def _extract_k_from_folder_name(self, folder_name: str) -> int | None:
        """Extract expected number of folds from folder name.

        Parameters
        ----------
            folder_name (str): Folder name (e.g., "KFold_5-1", "DuplexKFold_3-2")

        Returns
        -------
            int | None: Expected number of folds or None if not found
        """
        try:
            if "KFold_" in folder_name:
                return int(folder_name.split("KFold_")[1].split("-")[0])
            elif "DuplexKFold_" in folder_name:
                return int(folder_name.split("DuplexKFold_")[1].split("-")[0])
        except (ValueError, IndexError):
            pass
        return None

    def _find_latest_model_path(self, args: dict[str, Any]) -> str | None:
        """Return the path to the latest trainer folder containing model.keras.

        If KFold-like, returns the fold-* directory containing model.keras; otherwise
        returns the trainer directory itself. Returns None if not found.
        """
        model = args.get('model')
        metabolites = args.get('metabolites', [])
        dataset = args.get('dataset', '')
        epochs = args.get('epochs', 1000)
        batchsize = args.get('batchsize', 16)
        norm = args.get('norm', 'sum')
        validate = args.get('validate', 0.8)
        if not model or not metabolites or not dataset:
            return None

        metabolites_str = '-'.join(sorted(metabolites))
        dataset_parts = dataset.split('/')
        if len(dataset_parts) >= 2 and dataset_parts[0] == 'data':
            start_idx = 1
            if len(dataset_parts) >= 3 and dataset_parts[1].startswith('sim-'):
                start_idx = 2
            dataset_name = '_'.join(dataset_parts[start_idx:])
        else:
            dataset_name = '_'.join(dataset_parts)
        acquisitions = args.get('acquisitions', ['edit_off', 'edit_on'])
        datatype = args.get('datatype', ['real'])
        acquisitions_str = '-'.join(sorted(acquisitions))
        datatype_str = '-'.join(sorted(datatype))
        pulse_sequence = 'megapress'
        kfold_like = False
        if validate is None:
            trainer_base = "NoValidation"
        elif validate > 1.0:
            trainer_base = f"KFold_{int(validate)}"
            kfold_like = True
        elif validate < -1.0:
            trainer_base = f"DuplexKFold_{int(-validate)}"
            kfold_like = True
        elif validate > 0.0:
            trainer_base = f"Split_{validate}"
        elif validate < 0.0:
            trainer_base = f"DuplexSplit_{-validate}"
        elif validate == 0.0:
            trainer_base = "NoValidation"
        else:
            trainer_base = "NoValidation"

        model_bases: list[str] = [
            'data/model',
            'data/model-dist',
            'data/model-ae',
            'data/model-cae',
            'data/model-cnn'
        ]
        try:
            root_dir = os.path.dirname(os.path.abspath(__file__))
            cfg_path = os.path.join(root_dir, 'cfg.json')
            if os.path.isfile(cfg_path):
                with open(cfg_path) as f:
                    cfg_vals = json.load(f)
                path_model = cfg_vals.get('path_model')
                if path_model:
                    model_bases.append(path_model)
                for p in cfg_vals.get('search_model', []) or []:
                    model_bases.append(p)
        except Exception as e:
            print(f"# Warning: failed to read cfg.json for model search paths: {e}")

        seen = set()
        dedup_bases: list[str] = []
        for b in model_bases:
            if b not in seen:
                dedup_bases.append(b)
                seen.add(b)

        found_candidates: list[tuple[int, str]] = []
        for base in dedup_bases:
            base_path = os.path.join(
                base, model, metabolites_str, pulse_sequence,
                acquisitions_str, datatype_str, norm, str(batchsize),
                str(epochs), dataset_name
            )
            if not os.path.isdir(base_path):
                continue
            try:
                trainer_dirs = [d for d in os.listdir(base_path)
                                if os.path.isdir(os.path.join(base_path, d)) and d.startswith(trainer_base + '-')]
            except Exception:
                trainer_dirs = []
            best_trainer_dir = None
            best_idx = -1
            for d in trainer_dirs:
                try:
                    idx = int(d.split('-')[-1])
                except Exception:
                    idx = -1
                if idx > best_idx:
                    best_idx = idx
                    best_trainer_dir = d
            if best_trainer_dir is None:
                continue
            trainer_path = os.path.join(base_path, best_trainer_dir)
            if kfold_like:
                try:
                    folds = [d for d in os.listdir(trainer_path)
                             if os.path.isdir(os.path.join(trainer_path, d)) and d.startswith('fold-')]
                except Exception:
                    folds = []
                for fold_dir in sorted(folds, reverse=True):
                    model_file = os.path.join(trainer_path, fold_dir, 'model.keras')
                    if os.path.isfile(model_file):
                        found_candidates.append((best_idx, os.path.join(trainer_path, fold_dir)))
                        break
            else:
                model_file = os.path.join(trainer_path, 'model.keras')
                if os.path.isfile(model_file):
                    found_candidates.append((best_idx, trainer_path))

        # Return the global most recent path across bases by highest trainer index
        if not found_candidates:
            return None
        found_candidates.sort(key=lambda x: x[0], reverse=True)
        best_path = found_candidates[0][1]

        # For KFold-like models, validate completeness
        if kfold_like:
            # Extract trainer directory path (parent of fold directory)
            trainer_path = os.path.dirname(best_path)
            trainer_name = os.path.basename(trainer_path)
            expected_k = self._extract_k_from_folder_name(trainer_name)

            if expected_k is not None:
                is_complete, missing_folds, found_folds = self._validate_kfold_completeness(trainer_path, expected_k)
                if not is_complete:
                    print(f"⚠️  WARNING: KFold validation incomplete for {trainer_path}")
                    print(f"Expected {expected_k} folds, found {len(found_folds)} folds")
                    print(f"Missing folds: {missing_folds}")
                    print(f"Found folds: {found_folds}")
                    print("This indicates a partially trained model or training failure")
                    # Still return the path but mark it as incomplete
                    return best_path

        return best_path

    def _check_benchmark_exists(self, run_config: dict[str, Any]) -> bool:
        """Check if benchmark results already exist."""
        # Get command from run config, fallback to common config
        command = run_config.get('command')
        if command is None:
            command = self.config['common'].get('command')
            if command is None:
                return False

        if command != 'benchmark':
            return False

        args = self._merge_args(run_config)

        # Determine model destination folder for benchmark artifacts
        model_path = args.get('model')

        # If no explicit model path provided, try to derive latest trainer path (same as train lookup)
        if not model_path:
            # Reuse train existence logic to discover latest trainer folder
            # Build with same parameters and pick highest trainer index as in _check_model_exists
            tmp_config = dict(run_config)
            tmp_config['command'] = 'train'
            if not self._check_model_exists(tmp_config):
                return False
            # _check_model_exists only returns bool; to avoid deep refactor, require explicit model path for now
            return False

        if not os.path.isdir(model_path):
            return False

        # Load benchmark sequences from cfg path_benchmark (fallback to data/benchmark)
        bench_root = 'data/benchmark'
        try:
            root_dir = os.path.dirname(os.path.abspath(__file__))
            cfg_path = os.path.join(root_dir, 'cfg.json')
            if os.path.isfile(cfg_path):
                with open(cfg_path) as f:
                    cfg_vals = json.load(f)
                bench_root = cfg_vals.get('path_benchmark', bench_root)
        except Exception as e:
            print(f"# Warning: failed to read cfg.json for benchmark path: {e}")

        # Read benchmark_sequences.json
        seq_file = os.path.join(bench_root, 'benchmark_sequences.json')
        try:
            with open(seq_file) as f:
                benchmark_seqs = json.load(f)
        except Exception as e:
            print(f"# Warning: failed to read {seq_file}: {e}")
            return False

        # Determine norm requested; if 'default', accept any norm by globbing
        norm = args.get('norm', 'default')

        # All per-sequence artifacts: prefix = b_id + '_' + variant + '_' + norm
        # analyse.py writes <prefix>_concentration_errors.json (always)
        # Accept any norm if norm == 'default'
        import glob as _glob

        def has_seq_files(prefix_base: str) -> bool:
            pattern = f"{prefix_base}_{norm}_concentration_errors.json" if norm != 'default' else f"{prefix_base}_*_concentration_errors.json"
            matches = _glob.glob(os.path.join(model_path, pattern))
            return len(matches) > 0

        # Check all sequence variants
        for b_id, variants in benchmark_seqs.items():
            for variant in variants:
                prefix_base = f"{b_id}_{variant}"
                if not has_seq_files(prefix_base):
                    return False

        # Check aggregated benchmark_all
        agg_base = "benchmark_all"
        if not has_seq_files(agg_base):
            return False

        print("✅ Found existing benchmark artifacts for all sequences and aggregate")
        return True

    def _check_result_exists(self, run_config: dict[str, Any]) -> bool:
        """Check if results already exist for this run."""
        if self.force:
            return False

        # Get command from run config, fallback to common config
        command = run_config.get('command')
        if command is None:
            command = self.config['common'].get('command')
            if command is None:
                return False

        if command == 'train':
            return self._check_model_exists(run_config)
        elif command == 'benchmark':
            return self._check_benchmark_exists(run_config)
        # Add more result checking logic for other commands as needed

        return False

    def _run_command(self, cmd: list[str], description: str) -> bool:
        """Execute a command and return success status."""
        print(f"\n{'=' * 60}")
        print(f"RUNNING: {description}")
        print(f"COMMAND: {' '.join(cmd)}")
        print(f"{'=' * 60}")

        if self.dry_run:
            print("🔍 DRY RUN - Command not executed")
            return True

        start_time = time.time()

        try:
            # Ensure child Python processes do not buffer stdout/stderr
            env = os.environ.copy()
            env.setdefault('PYTHONUNBUFFERED', '1')

            # Inherit parent's stdout/stderr to stream output live to terminal
            result = subprocess.run(cmd, check=True, env=env)  # noqa: S603
            duration = time.time() - start_time

            print(f"Exit code: {result.returncode}")
            print(f"Duration: {duration:.2f} seconds")

            print(f"✅ SUCCESS: {description}")
            return True

        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            print(f"Exit code: {e.returncode}")
            print(f"Duration: {duration:.2f} seconds")

            print(f"❌ ERROR: {description}")
            return False

    def _resolve_dependencies(self, run_config: dict[str, Any]) -> bool:
        """Check if dependencies are satisfied."""
        if 'depends_on' not in run_config:
            return True

        dependency = run_config['depends_on']
        if dependency not in self.results:
            print(f"❌ Dependency '{dependency}' not found in results")
            return False

        if not self.results[dependency]:
            print(f"❌ Dependency '{dependency}' failed")
            return False

        return True

    def run(self) -> bool:
        """Execute all runs in the configuration."""
        print("🚀 Starting MRSNet execution")
        print("=" * 60)
        print(f"Configuration: {self.config_file}")
        print(f"Dry run: {self.dry_run}")
        print(f"Force: {self.force}")
        print(f"Total runs: {len(self.config['runs'])}")
        print("=" * 60)

        success_count = 0
        total_count = len(self.config['runs'])

        for i, run_config in enumerate(self.config['runs'], 1):
            name = run_config.get('name', f"run_{i}")
            # Get command from run config, fallback to common config
            command = run_config.get('command')
            if command is None:
                command = self.config['common'].get('command', 'unknown')

            print(f"\n{'=' * 100}")
            print(f"RUN {i}/{total_count}: {name.upper()}")
            print(f"Command: {command}")
            print(f"{'=' * 100}")

            # Check dependencies
            if not self._resolve_dependencies(run_config):
                print(f"❌ Skipping {name} due to failed dependencies")
                self.results[name] = False
                continue

            # If this is a benchmark and depends on a training run, auto-fill model path from dependency if missing
            if command == 'benchmark':
                dep = run_config.get('depends_on')
                args = self._merge_args(run_config)
                if dep and not args.get('model'):
                    # Try to get model path from artifacts of dependency
                    dep_art = self.artifacts.get(dep, {})
                    dep_model_path = dep_art.get('model_path')
                    if not dep_model_path and self.results.get(dep):
                        # Attempt discovery from dependency's args
                        dep_args = self._merge_args(next((rc for rc in self.config['runs'] if rc.get('name') == dep), {}))
                        dep_model_path = self._find_latest_model_path(dep_args)
                    if dep_model_path:
                        # Inject model into run_config args for this execution, preserving existing args (e.g., norm)
                        existing_args = dict(run_config.get('args', {}))
                        existing_args['model'] = dep_model_path
                        run_config['args'] = existing_args
                        # Ensure we merge with common args (do not restrict to only args)
                        run_config.pop('_only_args_no_common', None)
                        # Update merged args variable for subsequent checks
                        args = self._merge_args(run_config)
                        # Persist artifact for this run
                        self.artifacts.setdefault(name, {})['model_path'] = dep_model_path

            # Check if results already exist
            if self._check_result_exists(run_config):
                print(f"⏭️  Skipping {name} - results already exist")
                self.results[name] = True
                success_count += 1
                continue

            # Build and execute command
            try:
                cmd = self._build_command(run_config)
                success = self._run_command(cmd, f"{name} ({command})")
                self.results[name] = success

                # Record artifacts for train runs: model_path discovered after success
                if success and command == 'train':
                    latest_path = self._find_latest_model_path(self._merge_args(run_config))
                    if latest_path:
                        self.artifacts.setdefault(name, {})['model_path'] = latest_path

                        # Additional validation for KFold training completeness
                        args = self._merge_args(run_config)
                        validate = args.get('validate', 0.8)
                        if validate is not None and (validate > 1.0 or validate < -1.0):
                            # This is a KFold or DuplexKFold training
                            if validate > 1.0:
                                expected_k = int(validate)
                            else:  # validate < -1.0
                                expected_k = int(-validate)

                            trainer_path = os.path.dirname(latest_path)
                            is_complete, missing_folds, found_folds = self._validate_kfold_completeness(trainer_path, expected_k)
                            if not is_complete:
                                print("⚠️  WARNING: Training completed but KFold validation incomplete!")
                                print(f"Expected {expected_k} folds, found {len(found_folds)} folds")
                                print(f"Missing folds: {missing_folds}")
                                print(f"Found folds: {found_folds}")
                                print("This indicates a partially trained model or training failure")
                                # Mark as failed due to incomplete KFold
                                success = False
                                self.results[name] = False

                if success:
                    success_count += 1
                else:
                    print(f"❌ {name} failed")

            except Exception as e:
                print(f"❌ Error building command for {name}: {e}")
                self.results[name] = False

        # Summary
        print(f"\n{'=' * 100}")
        print("EXECUTION SUMMARY")
        print(f"{'=' * 100}")
        print(f"Successful: {success_count}/{total_count}")
        print(f"Failed: {total_count - success_count}/{total_count}")

        print("\nDetailed Results:")
        for name, success in self.results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"  {name}: {status}")

        print(f"\n{'=' * 100}")

        if success_count == total_count:
            print("🎉 ALL RUNS COMPLETED SUCCESSFULLY!")
            return True
        else:
            print("⚠️  SOME RUNS FAILED")
            return False


def main():
    """Execute MRSNet commands from JSON configuration."""
    parser = argparse.ArgumentParser(
        description='Execute MRSNet commands from JSON configuration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'config',
        help='JSON configuration file'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing them'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Ignore existing results and re-run all commands'
    )

    args = parser.parse_args()

    try:
        runner = MRSNetRunner(args.config, args.dry_run, args.force)
        success = runner.run()
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
