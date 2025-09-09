# TensorFlow 2.20 Migration Guide

This document outlines the changes made to upgrade MRSNet from TensorFlow 2.15 to TensorFlow 2.20.

## Changes Made

### 1. Updated Dependencies

**File: `requirements.txt`**
- Updated `tensorflow` from `2.18` to `2.20`
- Updated `tensorboard` from `2.18` to `2.20`

### 2. Fixed Import Compatibility

**Files Modified:**
- `mrsnet/autoencoder.py`
- `mrsnet/ae_quantifier.py`
- `mrsnet/cnn.py`

**Changes:**
- Replaced standalone `keras` imports with `tensorflow.keras` imports
- Updated all `from keras.*` statements to `from tensorflow.keras.*`
- Fixed deprecated `tf.data.experimental_distribute.auto_shard_policy` API usage
- Updated to use `tf.data.Options().distribute_options.auto_shard_policy` instead

**Before:**
```python
from keras.layers import Dense, Activation
from keras.models import Model, load_model
from keras.utils import plot_model
```

**After:**
```python
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
```

### 3. Fixed Deprecated API Usage

**Files Modified:**
- `mrsnet/autoencoder.py`
- `mrsnet/ae_quantifier.py`
- `mrsnet/cnn.py`

**Changes:**
- Updated deprecated `tf.data.experimental_distribute.auto_shard_policy` API
- Replaced with `tf.data.Options().distribute_options.auto_shard_policy`

**Before:**
```python
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
```

**After:**
```python
options = tf.data.Options()
options.distribute_options.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
```

## Compatibility Notes

### Model Loading
- **Existing models trained with TensorFlow 2.15 should be loadable** in TensorFlow 2.20
- TensorFlow maintains backward compatibility for saved models across minor version updates
- However, it's recommended to test loading existing models to ensure compatibility

### Python Version
- TensorFlow 2.20 supports Python 3.9 through 3.13
- Current project uses Python 3.11, which is fully compatible

### Breaking Changes
- The main breaking change addressed was the deprecation of standalone `keras` package imports
- All code now uses `tensorflow.keras` imports for better compatibility

## Testing

A test script `test_tf2_20_compatibility.py` has been created to verify:
1. TensorFlow 2.20 imports work correctly
2. MRSNet modules can be imported with the updated TensorFlow
3. Basic TensorFlow functionality works as expected
4. Deprecated API fixes work correctly

To run the test:
```bash
python test_tf2_20_compatibility.py
```

## Installation

To install the updated dependencies:

```bash
pip install -r requirements.txt
```

Or install TensorFlow 2.20 specifically:
```bash
pip install tensorflow==2.20
```

## Next Steps

1. **Test existing model loading**: Try loading any existing trained models to ensure they work with TensorFlow 2.20
2. **Train new models**: Verify that new model training works correctly with the updated TensorFlow version
3. **Performance testing**: Run benchmarks to ensure performance hasn't regressed
4. **Remove test files**: Once everything is verified, you can remove `test_tf2_20_compatibility.py`

## Potential Issues

If you encounter issues:

1. **Import errors**: Make sure all `keras` imports are updated to `tensorflow.keras`
2. **Model loading errors**: Try retraining models if loading fails
3. **Performance issues**: Check if any deprecated APIs are being used
4. **Memory issues**: TensorFlow 2.20 may have different memory management - monitor GPU/CPU usage

## References

- [TensorFlow 2.20 Release Notes](https://github.com/tensorflow/tensorflow/releases/tag/v2.20.0)
- [TensorFlow Installation Guide](https://www.tensorflow.org/install)
- [Keras Migration Guide](https://www.tensorflow.org/guide/keras/migrating_to_keras_3)
