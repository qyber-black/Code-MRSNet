#!/usr/bin/env python3
"""
Fix old concentration error analysis JSON files by normalizing to max normalization.

This script finds concentration error JSON files that are missing the 'norm' field,
normalizes all data to max normalization, and recalculates all error statistics
to ensure consistency across all files.

Usage:
    python fix_concentration_errors_json.py <folder_path> [--dry-run] [--verbose]

Arguments:
    folder_path: Path to search for concentration error JSON files (searches recursively)
    --dry-run: Show what would be changed without actually modifying files
    --verbose: Show detailed information about each file processed
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import linregress

def normalize_to_max(true_values: np.ndarray, predicted_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize both true and predicted values to max normalization.

    Parameters
    ----------
    true_values : np.ndarray
        Array of true concentration values
    predicted_values : np.ndarray
        Array of predicted concentration values

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (normalized_true, normalized_predicted) arrays
    """
    # Create copies to avoid modifying original data
    norm_true = true_values.copy()
    norm_pred = predicted_values.copy()

    # Normalize each sample to its maximum value
    for i in range(true_values.shape[0]):
        true_max = np.max(true_values[i, :])
        pred_max = np.max(predicted_values[i, :])

        if true_max != 0:
            norm_true[i, :] = true_values[i, :] / true_max
        if pred_max != 0:
            norm_pred[i, :] = predicted_values[i, :] / pred_max

    return norm_true, norm_pred

def recalculate_error_statistics(true_values: np.ndarray, predicted_values: np.ndarray) -> dict[str, Any]:
    """
    Recalculate all error statistics from true and predicted values.

    Parameters
    ----------
    true_values : np.ndarray
        Array of true concentration values
    predicted_values : np.ndarray
        Array of predicted concentration values

    Returns
    -------
    dict[str, Any]
        Dictionary containing all recalculated error statistics
    """
    error = predicted_values - true_values
    error_mean = np.mean(error, axis=0)
    error_std = np.std(error, axis=0)
    error_min = np.min(error, axis=0)
    error_max = np.max(error, axis=0)

    abserror = np.abs(error)
    abserror_mean = np.mean(abserror, axis=0)
    abserror_std = np.std(abserror, axis=0)
    abserror_min = np.min(abserror, axis=0)
    abserror_max = np.max(abserror, axis=0)

    # Per-metabolite statistics
    metabolites = ['Cr', 'GABA', 'Gln', 'Glu', 'NAA']  # Common metabolites
    metabolite_stats = {}

    for i, metabolite in enumerate(metabolites):
        if i < true_values.shape[1]:  # Only process if metabolite exists
            try:
                slope, intercept, r_value, p_value, std_err = linregress(true_values[:, i], predicted_values[:, i])
            except Exception:
                slope, intercept, r_value, p_value, std_err = np.nan, np.nan, np.nan, np.nan, np.nan

            metabolite_stats[metabolite] = {
                'error': {
                    'mean': float(error_mean[i]),
                    'std': float(error_std[i]),
                    'min': float(error_min[i]),
                    'max': float(error_max[i]),
                },
                'abserror': {
                    'mean': float(abserror_mean[i]),
                    'std': float(abserror_std[i]),
                    'min': float(abserror_min[i]),
                    'max': float(abserror_max[i]),
                },
                'linreg': {
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r_value': float(r_value),
                    'p_value': float(p_value),
                    'std_err': float(std_err)
                }
            }

    # Total statistics
    error_flat = error.flatten()
    abserror_flat = abserror.flatten()
    true_flat = true_values.flatten()
    pred_flat = predicted_values.flatten()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        slope, intercept, r_value, p_value, std_err = linregress(true_flat, pred_flat)

    total_stats = {
        'error': {
            'mean': float(np.mean(error_flat)),
            'std': float(np.std(error_flat)),
            'min': float(np.min(error_flat)),
            'max': float(np.max(error_flat)),
        },
        'abserror': {
            'mean': float(np.mean(abserror_flat)),
            'std': float(np.std(abserror_flat)),
            'min': float(np.min(abserror_flat)),
            'max': float(np.max(abserror_flat)),
        },
        'linreg': {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_value': float(r_value),
            'p_value': float(p_value),
            'std_err': float(std_err)
        }
    }

    return {
        'metabolite_stats': metabolite_stats,
        'total_stats': total_stats,
        'error': error.tolist(),
        'true': true_values.tolist(),
        'predicted': predicted_values.tolist()
    }

def fix_json_file(file_path: Path, dry_run: bool = False, verbose: bool = False) -> bool:
    """
    Fix a single concentration error JSON file by adding normalization info and recalculating stats.

    Parameters
    ----------
    file_path : Path
        Path to the JSON file to fix
    dry_run : bool
        If True, don't actually modify the file
    verbose : bool
        If True, print detailed information

    Returns
    -------
    bool
        True if file was successfully processed, False otherwise
    """
    try:
        if verbose:
            print(f"Processing: {file_path}")

        # Load the JSON file
        with open(file_path) as f:
            data = json.load(f)

        # Check if norm field already exists
        if 'norm' in data:
            if verbose:
                print(f"  Skipping: Already has norm field ({data['norm']})")
            return True

        # Check if required fields exist
        if 'true' not in data or 'predicted' not in data:
            print(f"  Error: Missing 'true' or 'predicted' fields in {file_path}")
            return False

        # Convert to numpy arrays
        true_values = np.array(data['true'])
        predicted_values = np.array(data['predicted'])

        if true_values.shape != predicted_values.shape:
            print(f"  Error: Shape mismatch between true and predicted values in {file_path}")
            return False

        # Normalize to max normalization
        if verbose:
            print("  Normalizing to max normalization...")

        norm_true, norm_pred = normalize_to_max(true_values, predicted_values)

        # Recalculate error statistics with normalized data
        if verbose:
            print("  Recalculating error statistics...")

        new_stats = recalculate_error_statistics(norm_true, norm_pred)

        # Update the data
        data['norm'] = 'max'

        # Update metabolite statistics
        for metabolite, stats in new_stats['metabolite_stats'].items():
            if metabolite in data:
                data[metabolite] = stats

        # Update total statistics
        data['total'] = new_stats['total_stats']

        # Update arrays with normalized data
        data['true'] = norm_true.tolist()
        data['predicted'] = norm_pred.tolist()
        data['error'] = new_stats['error']

        if not dry_run:
            # Create backup
            backup_path = file_path.with_suffix('.json.backup')
            if not backup_path.exists():
                file_path.rename(backup_path)

            # Write updated file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, sort_keys=True)

            if verbose:
                print(f"  Updated: {file_path}")
        else:
            if verbose:
                print(f"  Would update: {file_path}")

        return True

    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
        return False

def find_concentration_error_files(folder_path: Path) -> list[Path]:
    """
    Find all concentration error JSON files in the given folder and subfolders.

    Parameters
    ----------
    folder_path : Path
        Root folder to search

    Returns
    -------
    list[Path]
        List of paths to concentration error JSON files
    """
    pattern = "*_concentration_errors.json"
    return list(folder_path.rglob(pattern))

def main():
    """Run the concentration error JSON fixer."""
    parser = argparse.ArgumentParser(description="Fix old concentration error JSON files")
    parser.add_argument("folder_path", help="Path to search for concentration error JSON files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without modifying files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed information")

    args = parser.parse_args()

    folder_path = Path(args.folder_path)
    if not folder_path.exists():
        print(f"Error: Folder {folder_path} does not exist")
        sys.exit(1)

    if not folder_path.is_dir():
        print(f"Error: {folder_path} is not a directory")
        sys.exit(1)

    # Find all concentration error JSON files
    print(f"Searching for concentration error JSON files in {folder_path}...")
    json_files = find_concentration_error_files(folder_path)

    if not json_files:
        print("No concentration error JSON files found")
        return

    print(f"Found {len(json_files)} concentration error JSON files")

    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")

    # Process each file
    success_count = 0
    error_count = 0

    for json_file in json_files:
        if fix_json_file(json_file, dry_run=args.dry_run, verbose=args.verbose):
            success_count += 1
        else:
            error_count += 1

    # Summary
    print("\nSummary:")
    print(f"  Successfully processed: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total files: {len(json_files)}")

    if args.dry_run:
        print("\nRun without --dry-run to actually modify the files")

if __name__ == "__main__":
    main()
