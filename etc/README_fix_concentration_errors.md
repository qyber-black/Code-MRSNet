# Fix Concentration Error JSON Files

This directory contains scripts to fix old concentration error analysis JSON files that are missing the `norm` field.

## Problem

In older versions of MRSNet, concentration error analysis JSON files were saved without indicating which normalization method was used. This makes it difficult to interpret the results correctly, as different normalization methods can significantly affect the error statistics.

## Solution

The `fix_concentration_errors_json.py` script:

1. **Finds** all `*_concentration_errors.json` files recursively in a given directory
2. **Regenerates** the complete analysis using the original `analyse_model` function with max normalization
3. **Overwrites** the JSON files with:
   - The `norm` field set to `'max'`
   - Completely recalculated error statistics using the original analysis algorithms
   - Normalized `true` and `predicted` arrays
   - All error metrics (mean, std, min, max) and linear regression statistics
4. **Uses** a dummy model wrapper to leverage the existing analysis infrastructure

## Usage

### Basic Usage

```bash
# Fix files with verbose output
python etc/fix_concentration_errors_json.py data/model-cnn --verbose

# Fix files silently
python etc/fix_concentration_errors_json.py data/model-cnn
```

### Arguments

- `folder_path`: Path to search for concentration error JSON files (searches recursively)
- `--verbose` or `-v`: Show detailed information about each file processed

### Examples

```bash
# Fix all files in the model-cnn directory
python etc/fix_concentration_errors_json.py data/model-cnn

# Fix files in a specific subdirectory with verbose output
python etc/fix_concentration_errors_json.py data/model-cnn/cnn_medium_sigmoid_pool --verbose
```

## Safety Features

- **Error Handling**: Continues processing other files if one fails
- **Validation**: Checks for required fields before processing
- **Memory Management**: Aggressive cleanup of matplotlib resources and garbage collection
- **Skip Existing**: Files that already have a `norm` field are skipped

## File Structure

The script expects JSON files with this structure:
```json
{
  "prefix": "train",
  "Cr": {
    "error": { "mean": 0.001, "std": 0.002, ... },
    "abserror": { "mean": 0.001, "std": 0.002, ... },
    "linreg": { "slope": 1.0, "r_value": 0.99, ... }
  },
  "true": [[0.3, 0.2, 0.5], ...],
  "predicted": [[0.31, 0.19, 0.5], ...],
  "error": [[0.01, -0.01, 0.0], ...],
  "total": { ... }
}
```

After processing, the files will have an additional `norm` field:
```json
{
  "prefix": "train",
  "norm": "max",
  ...
}
```

## Technical Approach

This approach is robust because:

1. **Complete Regeneration**: Uses the original `analyse_model` function to ensure perfect consistency
2. **Dummy Model Wrapper**: Creates a mock model that returns precomputed predictions
3. **Max Normalization**: All data is consistently normalized using max normalization
4. **Original Algorithms**: All error statistics are calculated using the exact same algorithms as the original analysis
5. **Memory Efficient**: Aggressive cleanup prevents memory leaks during batch processing

## Implementation Details

- **DummyModel Class**: Wraps precomputed predictions to work with the existing analysis infrastructure
- **No Plot Generation**: Uses `create_plots=False` to avoid regenerating plots
- **Automatic Metabolite Detection**: Extracts metabolite names from existing JSON structure
- **Batch Processing**: Processes multiple files with periodic memory cleanup
- **Error Recovery**: Continues processing even if individual files fail

## Notes

- The script overwrites files in place (no backup creation)
- Files that already have a `norm` field are skipped
- The script automatically detects metabolite names from the JSON structure
- All error statistics are recalculated using the original analysis algorithms
- Memory cleanup occurs every 5 files to prevent accumulation
