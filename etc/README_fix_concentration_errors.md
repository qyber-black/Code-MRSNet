# Fix Concentration Error JSON Files

This directory contains scripts to fix old concentration error analysis JSON files that are missing the `norm` field.

## Problem

In older versions of MRSNet, concentration error analysis JSON files were saved without indicating which normalization method was used. This makes it difficult to interpret the results correctly, as different normalization methods can significantly affect the error statistics.

## Solution

The `fix_concentration_errors_json.py` script:

1. **Finds** all `*_concentration_errors.json` files recursively in a given directory
2. **Normalizes** all data to max normalization (divides each sample by its maximum value)
3. **Recalculates** all error statistics from the normalized `true` and `predicted` data
4. **Updates** the JSON files with:
   - The `norm` field set to `'max'`
   - Recalculated error statistics (ensuring consistency)
   - Normalized `true` and `predicted` arrays
   - Creates backup files before modifying

## Usage

### Basic Usage

```bash
# Dry run to see what would be changed
python etc/fix_concentration_errors_json.py data/model-cnn --dry-run --verbose

# Actually fix the files
python etc/fix_concentration_errors_json.py data/model-cnn --verbose
```

### Arguments

- `folder_path`: Path to search for concentration error JSON files (searches recursively)
- `--dry-run`: Show what would be changed without actually modifying files
- `--verbose`: Show detailed information about each file processed

### Examples

```bash
# Fix all files in the model-cnn directory
python etc/fix_concentration_errors_json.py data/model-cnn

# Fix files in a specific subdirectory
python etc/fix_concentration_errors_json.py data/model-cnn/cnn_medium_sigmoid_pool

# Dry run with verbose output
python etc/fix_concentration_errors_json.py data/model-cnn --dry-run --verbose
```

## Safety Features

- **Backup Creation**: Original files are renamed to `.json.backup` before modification
- **Dry Run Mode**: Test what would be changed without modifying files
- **Error Handling**: Continues processing other files if one fails
- **Validation**: Checks for required fields before processing

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
  "norm": "sum",
  ...
}
```

## Approach Validity

This approach is valid because:

1. **Raw Data Available**: The JSON files contain the original `true` and `predicted` arrays
2. **Consistent Normalization**: All data is normalized to the same method (max normalization)
3. **Reversible Process**: All error statistics are calculated from the normalized values
4. **Consistency**: Recalculating ensures all statistics are consistent with max normalization
5. **Simplicity**: No need to detect different normalization methods - everything becomes comparable

## Notes

- The script processes files in place, creating backups
- Files that already have a `norm` field are skipped
- The script handles common metabolite names: Cr, GABA, Gln, Glu, NAA
- Error statistics are recalculated using the same algorithms as the original analysis code
