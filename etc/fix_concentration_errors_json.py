#!/usr/bin/env python3
"""Fix old concentration error analysis JSON files by regenerating analysis and plots.

This script finds concentration error JSON files that are missing the 'norm' field,
regenerates the complete analysis with max normalization using the original
analyse_model_error function, ensuring perfect consistency with the original analysis.

Usage:
    python fix_concentration_errors_json.py <folder_path> [--verbose]

Arguments:
    folder_path: Path to search for concentration error JSON files (searches recursively)
    --verbose: Show detailed information about each file processed
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mrsnet.analyse import analyse_model
from mrsnet.cfg import Cfg


class DummyModel:
  """Dummy model wrapper for plot regeneration with precomputed predictions."""

  def __init__(self, metabolites, precomputed_predictions):
    self.metabolites = metabolites
    self.output = "concentrations"
    self.precomputed_predictions = precomputed_predictions

  def predict(self, inp, verbose=0):
    """Return precomputed predictions instead of computing them."""
    return self.precomputed_predictions


def fix_json_file(file_path: Path, verbose: bool = False) -> bool:
  """Fix a single concentration error JSON file by regenerating analysis and plots.

  Parameters
  ----------
  file_path : Path
      Path to the JSON file to fix
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
    if "norm" in data:
      if verbose:
        print(f"  JSON already has norm field ({data['norm']})")
      return True

    # Check if required fields exist
    if "true" not in data or "predicted" not in data:
      print(f"  Error: Missing 'true' or 'predicted' fields in {file_path}")
      return False

    # Convert to numpy arrays
    true_values = np.array(data["true"])
    predicted_values = np.array(data["predicted"])

    if true_values.shape != predicted_values.shape:
      print(f"  Error: Shape mismatch between true and predicted values in {file_path}")
      return False

    # Get metabolites from data
    metabolites = []
    for key in data.keys():
      if key not in ["true", "predicted", "error", "total", "norm", "prefix"]:
        if isinstance(data[key], dict) and "error" in data[key]:
          metabolites.append(key)
    if not metabolites:
      print(f"  Error: Could not determine metabolites for {file_path}")
      return False

    # Create dummy model and regenerate analysis
    if verbose:
      print("  Regenerating analysis with max normalization...")

    model = DummyModel(metabolites, predicted_values)
    folder = str(file_path.parent)
    prefix = file_path.stem.replace("_concentration_errors", "")
    image_dpi = Cfg.val["image_dpi"]
    screen_dpi = Cfg.val["screen_dpi"]

    # Use analyse_model to regenerate everything
    dummy_input = np.zeros_like(predicted_values)
    pre, info, error = analyse_model(
        model,
        dummy_input,
        true_values,
        folder,
        prefix,
        save_conc=False,
        show_conc=False,
        norm="max",
        verbose=verbose,
        image_dpi=image_dpi,
        screen_dpi=screen_dpi,
      )

    # Clean up memory
    del model
    del dummy_input
    del true_values
    del predicted_values
    del pre
    del info
    del error

    # Aggressive matplotlib cleanup
    plt.close('all')
    plt.clf()
    plt.cla()

    return True

  except Exception as e:
    print(f"  Error processing {file_path}: {e}")
    return False


def find_concentration_error_files(folder_path: Path) -> list[Path]:
  """Find all concentration error JSON files in the given folder and subfolders.

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
  bin_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "mrsnet.py")
  if not os.path.isfile(bin_path):
    raise RuntimeError("Cannot find location of mrsnet.py root folder")
  Cfg.init(bin_path)

  parser = argparse.ArgumentParser(description="Fix old concentration error JSON files")
  parser.add_argument("folder_path", help="Path to search for concentration error JSON files")
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

  # Process each file
  success_count = 0
  error_count = 0

  for i, json_file in enumerate(json_files):
    if args.verbose:
      print(f"Processing file {i+1}/{len(json_files)}: {json_file.name}")

    if fix_json_file(json_file, verbose=args.verbose):
      success_count += 1
    else:
      error_count += 1

    # Periodic memory cleanup every 5 files
    if (i + 1) % 5 == 0:
      plt.close('all')
      plt.clf()
      plt.cla()
      gc.collect()

  # Summary
  print("\nSummary:")
  print(f"  Successfully processed: {success_count}")
  print(f"  Errors: {error_count}")
  print(f"  Total files: {len(json_files)}")


if __name__ == "__main__":
  main()
