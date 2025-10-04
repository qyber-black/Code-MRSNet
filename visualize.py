#!/usr/bin/env python3
#
# Visualize aggregate results into PDF tables/plots
#
# SPDX-FileCopyrightText: Copyright (C) 2025
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Visualize aggregate results into PDF tables/plots."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

CONC_ARCHS = ("cnn", "fcnn", "qmrs", "qnet", "qnet_basis", "encdec", "caeq", "aeq")
SPEC_ARCHS = ("ae", "caeq")


def _arch_of_model(model: str) -> str:
  for prefix in ("qnet_basis_",):
    if model.startswith(prefix):
      return prefix[:-1]
  for arch in ("cnn", "fcnn", "qmrs", "qnet", "encdec", "caeq", "aeq", "ae"):
    if model.startswith(arch + "_") or model == arch:
      return arch
  return "other"


def _read_aggregate(root: Path) -> pd.DataFrame:
  agg_csv = root / "aggregate" / "all_results.csv"
  if not agg_csv.exists():
    raise SystemExit(f"Aggregate CSV not found: {agg_csv}")
  df = pd.read_csv(agg_csv)
  return df


def _shorten_acq_text(val: object) -> str:
  s = str(val)
  if not s:
    return s
  sl = s.lower()
  # Replace common acquisition tokens
  sl = sl.replace('edit_off', 'off')
  sl = sl.replace('edit_on', 'on')
  sl = sl.replace('edit-on', 'on')  # Keep both underscore and hyphen versions
  sl = sl.replace('difference', 'diff')
  return sl


def _collect_folds(row: pd.Series, prefix: str) -> list[float]:
  vals: list[float] = []
  i = 0
  while True:
    key = f"{prefix}_fold{i}"
    if key not in row.index:
      break
    v = row.get(key)
    if pd.notna(v):
      try:
        vals.append(float(v))
      except Exception: # noqa: S110
        pass
    i += 1
  return vals


def _statistical_significance_test(rows: pd.DataFrame, alpha: float = 0.05, method: str = 'consecutive', sensitivity: str = 'standard') -> list[tuple[int, int]]:
  """Perform statistical significance tests on Validation MAE distributions.

  Uses Mann-Whitney U test to compare Validation MAE distributions between models.
  Red dashed lines in the visualization separate models with significantly different distributions.

  Args:
    rows: DataFrame containing model results with fold data
    alpha: Base significance level (default 0.05)
    method: Testing method - 'consecutive' (recommended) or 'pairwise' (with Bonferroni correction)
    sensitivity: Sensitivity level controlling detection threshold:
      - 'standard': α=0.05 (traditional statistical significance)
      - 'high': α=0.10 (more sensitive, detects smaller differences)
      - 'very_high': α=0.20 (very sensitive, detects subtle differences)
      - 'effect_size': α=0.05 + Cohen's d>0.2 (statistically significant AND practically meaningful)

  Returns:
  -------
    List of tuples (i, j) indicating models i and j are significantly different

  """
  significant_pairs = []

  # Collect validation fold data for all models
  val_fold_data = []
  valid_indices = []

  for idx, row in rows.iterrows():
    folds = _collect_folds(row, 'val_mae')
    if len(folds) >= 2:  # Need at least 2 folds for statistical test
      val_fold_data.append(folds)
      valid_indices.append(idx)

  if len(val_fold_data) < 2:
    return significant_pairs

  if method == 'consecutive':
    # Test consecutive models in ranking (recommended approach)
    # This identifies where performance differences become non-significant
    for i in range(len(val_fold_data) - 1):
      try:
        statistic, p_value = stats.mannwhitneyu(val_fold_data[i], val_fold_data[i + 1],
                                              alternative='two-sided')

        # Apply sensitivity adjustments
        adjusted_alpha = alpha
        is_significant = False

        if sensitivity == 'high':
          # More sensitive: use higher alpha
          adjusted_alpha = alpha * 2  # 0.10
          is_significant = p_value < adjusted_alpha
        elif sensitivity == 'very_high':
          # Very sensitive: use much higher alpha
          adjusted_alpha = alpha * 4  # 0.20
          is_significant = p_value < adjusted_alpha
        elif sensitivity == 'effect_size':
          # Effect size based: check both significance and effect size
          is_significant = p_value < alpha
          if is_significant:
            # Calculate Cohen's d effect size
            mean1, mean2 = np.mean(val_fold_data[i]), np.mean(val_fold_data[i + 1])
            std1, std2 = np.std(val_fold_data[i]), np.std(val_fold_data[i + 1])
            pooled_std = np.sqrt((std1**2 + std2**2) / 2)
            cohens_d = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0

            # Only consider significant if effect size is meaningful (d > 0.2)
            is_significant = cohens_d > 0.2
        else:  # 'standard'
          is_significant = p_value < alpha

        if is_significant:
          # For consecutive testing, use sorted positions (i, i+1)
          significant_pairs.append((i, i + 1))
      except Exception:
        # Skip if test fails (e.g., identical distributions)
        continue

  elif method == 'pairwise':
    # Test all pairwise combinations with Bonferroni correction
    n_comparisons = len(val_fold_data) * (len(val_fold_data) - 1) // 2
    corrected_alpha = alpha / n_comparisons if n_comparisons > 0 else alpha

    for i in range(len(val_fold_data)):
      for j in range(i + 1, len(val_fold_data)):
        try:
          statistic, p_value = stats.mannwhitneyu(val_fold_data[i], val_fold_data[j],
                                                alternative='two-sided')

          if p_value < corrected_alpha:
            # Convert back to original DataFrame indices
            orig_i = rows.index[valid_indices[i]]
            orig_j = rows.index[valid_indices[j]]
            significant_pairs.append((orig_i, orig_j))
        except Exception:
          # Skip if test fails (e.g., identical distributions)
          continue

  # Additional sliding window analysis for top models
  if sensitivity in ['high', 'very_high'] and len(val_fold_data) >= 3:
    # Test models within a small window (e.g., within 3 positions) for top models
    window_size = 3
    top_models_limit = min(50, len(val_fold_data))  # Focus on top 50 models

    for i in range(top_models_limit):
      for j in range(i + 1, min(i + window_size + 1, len(val_fold_data))):
        try:
          statistic, p_value = stats.mannwhitneyu(val_fold_data[i], val_fold_data[j],
                                                alternative='two-sided')

          # Use more lenient alpha for sliding window
          window_alpha = alpha * 3 if sensitivity == 'high' else alpha * 5

          if p_value < window_alpha:
            # Avoid duplicates
            if (i, j) not in significant_pairs and (j, i) not in significant_pairs:
              significant_pairs.append((i, j))
        except Exception:
          continue

  return significant_pairs


def _select_topn_conc(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
  # Use validation first, fall back to training
  df = df.copy()
  df['score'] = df['val_mae_mean'].fillna(df['train_mae_mean'])
  out = df[pd.notna(df['score'])].sort_values('score', ascending=True).head(top_n)
  return out


def _select_topn_spec(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
  # Spectra models identified by arch in SPEC_ARCHS; use validation/training means
  df = df.copy()
  df['arch'] = df['model'].astype(str).map(_arch_of_model)
  df = df[df['arch'].isin(SPEC_ARCHS)]
  df['score'] = df['val_mae_mean'].fillna(df['train_mae_mean'])
  out = df[pd.notna(df['score'])].sort_values('score', ascending=True).head(top_n)
  return out


def _model_params_str(model_name: str) -> list[str]:
  # Heuristic: split after first 1-2 parts
  parts = model_name.split('_')
  if len(parts) <= 1:
    return [model_name]
  arch = parts[0]
  rest = '_'.join(parts[1:])
  # Split on '-' for readability
  return [arch, rest]


def _extract_cnn_parameters(model_name: str) -> dict[str, any]:
  """Extract CNN parameters from model name.

  For detailed CNN models: cnn_[S1]_[S2]_[C1]_[C2]_[C3]_[C4]_[O1]_[O2]_[F1]_[F2]_[D]_[activation]
  For simple CNN models: cnn_[small,medium,large]_[activation][_pool]
  """
  if not model_name.startswith('cnn_'):
    return {}

  parts = model_name.split('_')
  if len(parts) < 2:
    return {}

  params = {}

  if parts[1] in ['small', 'medium', 'large']:
    # Simple CNN model - map to full parameters for consistency
    size = parts[1]
    activation = parts[2] if len(parts) > 2 else 'sigmoid'
    pool = 'pool' in parts

    # Map simplified parameters to full parameter set
    if size == 'small':
      params['conv1'] = 7
      params['conv2'] = 5
      params['conv3'] = 3
    elif size == 'medium':
      params['conv1'] = 9
      params['conv2'] = 7
      params['conv3'] = 5
    elif size == 'large':
      params['conv1'] = 11
      params['conv2'] = 9
      params['conv3'] = 7

    params['conv4'] = 3
    params['dropout1'] = 0.0
    params['dropout2'] = 0.3
    params['filter1'] = 256
    params['filter2'] = 512
    params['dense'] = 1024
    params['activation'] = activation

    if pool:
      params['strides1'] = -2
      params['strides2'] = -3
    else:
      params['strides1'] = 2
      params['strides2'] = 3

    # Store simplified name for display
    params['simplified_name'] = f"{size}_{activation}{'_pool' if pool else ''}"

  elif len(parts) >= 13:
    # Detailed CNN model
    try:
      params['strides1'] = int(parts[1])
      params['strides2'] = int(parts[2])
      params['conv1'] = int(parts[3])
      params['conv2'] = int(parts[4])
      params['conv3'] = int(parts[5])
      params['conv4'] = int(parts[6])
      params['dropout1'] = float(parts[7])
      params['dropout2'] = float(parts[8])
      params['filter1'] = int(parts[9])
      params['filter2'] = int(parts[10])
      params['dense'] = int(parts[11])
      params['activation'] = parts[12]

      # Check if this maps to a simplified model
      simplified_name = _map_to_simplified_cnn(params)
      if simplified_name:
        params['simplified_name'] = simplified_name
    except (ValueError, IndexError):
      pass

  return params


def _map_to_simplified_cnn(params: dict[str, any]) -> str | None:
  """Map full CNN parameters back to simplified name if possible."""
  # Check if parameters match simplified model patterns
  if (params.get('conv4') == 3 and
      params.get('dropout1') == 0.0 and
      params.get('dropout2') == 0.3 and
      params.get('filter1') == 256 and
      params.get('filter2') == 512 and
      params.get('dense') == 1024):

    # Check for small model
    if (params.get('conv1') == 7 and
        params.get('conv2') == 5 and
        params.get('conv3') == 3):
      pool = params.get('strides1') == -2 and params.get('strides2') == -3
      return f"small_{params.get('activation', 'sigmoid')}{'_pool' if pool else ''}"

    # Check for medium model
    elif (params.get('conv1') == 9 and
          params.get('conv2') == 7 and
          params.get('conv3') == 5):
      pool = params.get('strides1') == -2 and params.get('strides2') == -3
      return f"medium_{params.get('activation', 'sigmoid')}{'_pool' if pool else ''}"

    # Check for large model
    elif (params.get('conv1') == 11 and
          params.get('conv2') == 9 and
          params.get('conv3') == 7):
      pool = params.get('strides1') == -2 and params.get('strides2') == -3
      return f"large_{params.get('activation', 'sigmoid')}{'_pool' if pool else ''}"

  return None


def _extract_fcnn_parameters(model_name: str) -> dict[str, any]:
  """Extract FCNN parameters from model name.

  FCNN models: fcnn_[FREQS]_[METABOLITES] or fcnn (default)
  """
  if not model_name.startswith('fcnn_'):
    return {}

  parts = model_name.split('_')
  params = {'type': 'fcnn'}

  if len(parts) > 1 and parts[1] != 'default':
    # Extract frequency and metabolite parameters if present
    try:
      if len(parts) >= 2:
        params['freqs'] = parts[1]
      if len(parts) >= 3:
        params['metabolites'] = parts[2]
    except (ValueError, IndexError):
      pass

  return params


def _extract_qnet_parameters(model_name: str) -> dict[str, any]:
  """Extract QNet parameters from model name.

  QNet models: qnet_[IF_FILTERS]_[MM_FILTERS]_[KERNEL]_[IF_FC]_[MM_FC]_[IF_FACTORS]
  or qnet_default
  """
  if not model_name.startswith('qnet_'):
    return {}

  parts = model_name.split('_')
  params = {'type': 'qnet'}

  if len(parts) > 1 and parts[1] != 'default':
    try:
      if len(parts) >= 2:
        params['if_filters'] = int(parts[1])
      if len(parts) >= 3:
        params['mm_filters'] = int(parts[2])
      if len(parts) >= 4:
        params['kernel_size'] = int(parts[3])
      if len(parts) >= 5:
        params['if_fc_units'] = int(parts[4])
      if len(parts) >= 6:
        params['mm_fc_units'] = int(parts[5])
      if len(parts) >= 7:
        params['if_factors'] = int(parts[6])
    except (ValueError, IndexError):
      pass

  return params


def _extract_qnet_basis_parameters(model_name: str) -> dict[str, any]:
  """Extract QNet_basis parameters from model name.

  QNet_basis models: qnet_basis_[IF_FILTERS]_[MM_FILTERS]_[KERNEL]_[IF_FC]_[MM_FC]_[IF_FACTORS]
  or qnet_basis_default
  """
  if not model_name.startswith('qnet_basis_'):
    return {}

  parts = model_name.split('_')
  params = {'type': 'qnet_basis'}

  if len(parts) > 2 and parts[2] != 'default':
    try:
      if len(parts) >= 3:
        params['if_filters'] = int(parts[2])
      if len(parts) >= 4:
        params['mm_filters'] = int(parts[3])
      if len(parts) >= 5:
        params['kernel_size'] = int(parts[4])
      if len(parts) >= 6:
        params['if_fc_units'] = int(parts[5])
      if len(parts) >= 7:
        params['mm_fc_units'] = int(parts[6])
      if len(parts) >= 8:
        params['if_factors'] = int(parts[7])
    except (ValueError, IndexError):
      pass

  return params


def _extract_qmrs_parameters(model_name: str) -> dict[str, any]:
  """Extract QMRS parameters from model name.

  QMRS models: qmrs_[INITIAL_FILTERS]_[INCEPTION_FILTERS]_[LSTM_UNITS]_[MLP_UNITS]_[DROPOUT]
  or qmrs_default
  """
  if not model_name.startswith('qmrs_'):
    return {}

  parts = model_name.split('_')
  params = {'type': 'qmrs'}

  if len(parts) > 1 and parts[1] != 'default':
    try:
      if len(parts) >= 2:
        params['initial_filters'] = int(parts[1])
      if len(parts) >= 3:
        params['inception_filters'] = int(parts[2])
      if len(parts) >= 4:
        params['lstm_units'] = int(parts[3])
      if len(parts) >= 5:
        params['mlp_units'] = int(parts[4])
      if len(parts) >= 6:
        params['dropout'] = float(parts[5])
    except (ValueError, IndexError):
      pass

  return params


def _extract_encdec_parameters(model_name: str) -> dict[str, any]:
  """Extract EncDec parameters from model name.

  EncDec models: encdec_[FILTERS] or encdec_default
  """
  if not model_name.startswith('encdec_'):
    return {}

  parts = model_name.split('_')
  params = {'type': 'encdec'}

  if len(parts) > 1 and parts[1] != 'default':
    try:
      params['filters'] = int(parts[1])
    except (ValueError, IndexError):
      pass

  return params


def _extract_caeq_parameters(model_name: str) -> dict[str, any]:
  """Extract CAEQ parameters from model name.

  CAEQ models: caeq_fc_[LIN]_[LOUT]_[ACT]_[ACT-LAST]_[DO]_[UNITS]_[LAYERS]_[ACTIVATION]_[ACTIVATION-LAST]_[DP]
  """
  if not model_name.startswith('caeq_'):
    return {}

  parts = model_name.split('_')
  params = {'type': 'caeq'}

  if len(parts) >= 7 and parts[1] == 'fc':
    try:
      params['lin'] = int(parts[2])
      params['lout'] = int(parts[3])
      params['act'] = parts[4]
      params['act_last'] = parts[5]
      params['dropout'] = float(parts[6])
      if len(parts) >= 8:
        params['units'] = int(parts[7])
      if len(parts) >= 9:
        params['layers'] = int(parts[8])
      if len(parts) >= 10:
        params['activation'] = parts[9]
      if len(parts) >= 11:
        params['activation_last'] = parts[10]
      if len(parts) >= 12:
        params['dp'] = float(parts[11])
    except (ValueError, IndexError):
      pass

  return params


def _extract_ae_parameters(model_name: str) -> dict[str, any]:
  """Extract AE parameters from model name.

  AE models: ae_fc_[LIN]_[LOUT]_[ACT]_[ACT-LAST]_[DO]
  """
  if not model_name.startswith('ae_'):
    return {}

  parts = model_name.split('_')
  params = {'type': 'ae'}

  if len(parts) >= 7 and parts[1] == 'fc':
    try:
      params['lin'] = int(parts[2])
      params['lout'] = int(parts[3])
      params['act'] = parts[4]
      params['act_last'] = parts[5]
      params['dropout'] = float(parts[6])
    except (ValueError, IndexError):
      pass

  return params


def _extract_aeq_parameters(model_name: str) -> dict[str, any]:
  """Extract AEQ parameters from model name.

  AEQ models: aeq_fc_[UNITS]_[LAYERS]_[ACT]_[ACT-LAST]_[DO]
  """
  if not model_name.startswith('aeq_'):
    return {}

  parts = model_name.split('_')
  params = {'type': 'aeq'}

  if len(parts) >= 7 and parts[1] == 'fc':
    try:
      params['units'] = int(parts[2])
      params['layers'] = int(parts[3])
      params['act'] = parts[4]
      params['act_last'] = parts[5]
      params['dropout'] = float(parts[6])
    except (ValueError, IndexError):
      pass

  return params


def _extract_architecture_parameters(model_name: str) -> dict[str, any]:
  """Extract architecture-specific parameters from model name."""
  if model_name.startswith('cnn_'):
    return _extract_cnn_parameters(model_name)
  elif model_name.startswith('fcnn_'):
    return _extract_fcnn_parameters(model_name)
  elif model_name.startswith('qnet_basis_'):
    return _extract_qnet_basis_parameters(model_name)
  elif model_name.startswith('qnet_'):
    return _extract_qnet_parameters(model_name)
  elif model_name.startswith('qmrs_'):
    return _extract_qmrs_parameters(model_name)
  elif model_name.startswith('encdec_'):
    return _extract_encdec_parameters(model_name)
  elif model_name.startswith('caeq_'):
    return _extract_caeq_parameters(model_name)
  elif model_name.startswith('ae_'):
    return _extract_ae_parameters(model_name)
  elif model_name.startswith('aeq_'):
    return _extract_aeq_parameters(model_name)
  else:
    return {}


def _render_architecture_analysis(df: pd.DataFrame, out_dir: Path, dpi: int, log_scale: bool) -> None:
  """Create architecture parameter analysis plots."""
  # Group by architecture
  df['arch'] = df['model'].astype(str).map(_arch_of_model)

  for arch in df['arch'].unique():
    if pd.isna(arch) or arch == 'other':
      continue

    arch_df = df[df['arch'] == arch]
    if len(arch_df) <= 1:
      continue  # Need multiple models for analysis

    # Extract parameters for all models in this architecture
    param_data = []
    for _, row in arch_df.iterrows():
      model_name = row['model']
      params = _extract_architecture_parameters(model_name)
      if params:
        params['model'] = model_name
        params['val_mae_mean'] = row.get('val_mae_mean')
        params['train_mae_mean'] = row.get('train_mae_mean')
        param_data.append(params)

    if not param_data:
      continue

    # Convert to DataFrame for analysis
    param_df = pd.DataFrame(param_data)

    # Create parameter coverage, correlation, performance correlation, value-performance, interaction, sensitivity, and coverage gaps plots
    _create_parameter_coverage_plot(param_df, arch, out_dir, dpi)
    _create_parameter_correlation_plot(param_df, arch, out_dir, dpi)
    _create_parameter_performance_correlation_plot(param_df, arch, out_dir, dpi)
    _create_parameter_value_performance_plot(param_df, arch, out_dir, dpi)
    _create_parameter_interaction_plot(param_df, arch, out_dir, dpi)
    _create_coverage_gaps_plot(param_df, arch, out_dir, dpi)


def _create_parameter_coverage_plot(param_df: pd.DataFrame, arch: str, out_dir: Path, dpi: int) -> None:
  """Create plot showing parameter value coverage across models."""
  # Find all parameters (numeric and string)
  all_params = []
  for col in param_df.columns:
    if col not in ['model', 'val_mae_mean', 'train_mae_mean', 'simplified_name']:
      all_params.append(col)

  if not all_params:
    return

  # Create subplots for each parameter
  n_params = len(all_params)
  n_cols = min(3, n_params)
  n_rows = (n_params + n_cols - 1) // n_cols

  fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
  if n_params == 1:
    axes = [axes]
  elif n_rows == 1 and n_cols == 1:
    axes = [axes]
  elif n_rows == 1 or n_cols == 1:
    axes = axes.flatten()
  else:
    axes = axes.flatten()

  for i, param in enumerate(all_params):
    ax = axes[i]
    values = param_df[param].dropna()

    if len(values) > 1:
      if param_df[param].dtype in ['int64', 'float64']:
        # Numeric parameter - use bar chart like string parameters for proper centering
        value_counts = values.value_counts().sort_index()
        bars = ax.bar(range(len(value_counts)), value_counts.values, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels([str(x) for x in value_counts.index])
        ax.set_xlabel(param)
        ax.set_ylabel('Count')

        # Add value labels on bars
        for bar, count in zip(bars, value_counts.values, strict=False):
          height = bar.get_height()
          ax.text(bar.get_x() + bar.get_width()/2., height/2,
                  str(count), ha='center', va='center', fontsize=9)
      else:
        # String parameter - use bar chart
        value_counts = values.value_counts()
        bars = ax.bar(range(len(value_counts)), value_counts.values, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        ax.set_xlabel(param)
        ax.set_ylabel('Count')

        # Add value labels on bars
        for bar, count in zip(bars, value_counts.values, strict=False):
          height = bar.get_height()
          ax.text(bar.get_x() + bar.get_width()/2., height/2,
                  str(count), ha='center', va='center', fontsize=9)

      ax.set_title(f'{arch.upper()}: {param}')
    else:
      ax.text(0.5, 0.5, f'Single value: {values.iloc[0] if len(values) > 0 else "N/A"}',
              ha='center', va='center', transform=ax.transAxes)
      ax.set_title(f'{arch.upper()}: {param}')

  # Hide unused subplots
  for i in range(n_params, len(axes)):
    axes[i].set_visible(False)

  plt.tight_layout()
  out_file = out_dir / f'architecture_coverage_{arch}.pdf'
  fig.savefig(out_file, dpi=dpi, bbox_inches='tight')
  plt.close(fig)


def _create_parameter_impact_plot(param_df: pd.DataFrame, arch: str, out_dir: Path, dpi: int, log_scale: bool) -> None:
  """Create plot showing parameter impact on MAE performance."""
  # Find all parameters (numeric and string)
  all_params = []
  for col in param_df.columns:
    if col not in ['model', 'val_mae_mean', 'train_mae_mean', 'simplified_name']:
      all_params.append(col)

  if not all_params:
    return

  # Create subplots for each parameter vs MAE
  n_params = len(all_params)
  n_cols = min(2, n_params)
  n_rows = (n_params + n_cols - 1) // n_cols

  fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
  if n_params == 1:
    axes = [axes]
  elif n_rows == 1 and n_cols == 1:
    axes = [axes]
  elif n_rows == 1 or n_cols == 1:
    axes = axes.flatten()
  else:
    axes = axes.flatten()

  for i, param in enumerate(all_params):
    ax = axes[i]

    # Plot validation MAE vs parameter
    val_data = param_df[['val_mae_mean', param]].dropna()
    if len(val_data) > 1:
      if param_df[param].dtype in ['int64', 'float64']:
        # Numeric parameter
        ax.scatter(val_data[param], val_data['val_mae_mean'], alpha=0.7, label='Validation MAE', color='steelblue')

        # Add trend line for numeric parameters
        if len(val_data) > 2:
          try:
            z = np.polyfit(val_data[param], val_data['val_mae_mean'], 1)
            p = np.poly1d(z)
            ax.plot(val_data[param], p(val_data[param]), "r--", alpha=0.8)
          except (np.linalg.LinAlgError, Exception):
            # Skip trend line if polyfit fails
            pass

        # Set x-axis to only show actual values
        unique_vals = sorted(val_data[param].unique())
        ax.set_xticks(unique_vals)

      else:
        # String parameter - group by parameter value and show mean ± std
        grouped = val_data.groupby(param)['val_mae_mean'].agg(['mean', 'std', 'count']).reset_index()

        # Create bar plot with error bars
        x_pos = range(len(grouped))
        bars = ax.bar(x_pos, grouped['mean'], yerr=grouped['std'].fillna(0),
                     alpha=0.7, color='steelblue', capsize=5, label='Validation MAE')

        # Add count labels on bars
        for bar, count in zip(bars, grouped['count'], strict=False):
          height = bar.get_height()
          ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                  f'n={count}', ha='center', va='bottom', fontsize=8)

        # Set x-axis labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(grouped[param], rotation=45, ha='right')

    # Plot training MAE vs parameter
    train_data = param_df[['train_mae', param]].dropna()
    if len(train_data) > 1:
      if param_df[param].dtype in ['int64', 'float64']:
        # Numeric parameter
        ax.scatter(train_data[param], train_data['train_mae'], alpha=0.7, label='Training MAE', color='salmon')
      else:
        # String parameter - group by parameter value and show mean ± std
        grouped_train = train_data.groupby(param)['train_mae'].agg(['mean', 'std', 'count']).reset_index()

        # Create bar plot with error bars
        x_pos = range(len(grouped_train))
        bars = ax.bar(x_pos, grouped_train['mean'], yerr=grouped_train['std'].fillna(0),
                     alpha=0.7, color='salmon', capsize=5, label='Training MAE')

        # Add count labels on bars
        for bar, count in zip(bars, grouped_train['count'], strict=False):
          height = bar.get_height()
          ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                  f'n={count}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel(param)
    ax.set_ylabel('MAE')
    ax.set_title(f'{arch.upper()}: {param} vs MAE')
    ax.legend()

    if log_scale:
      ax.set_yscale('log')

  # Hide unused subplots
  for i in range(n_params, len(axes)):
    axes[i].set_visible(False)

  plt.tight_layout()
  out_file = out_dir / f'architecture_impact_{arch}.pdf'
  fig.savefig(out_file, dpi=dpi, bbox_inches='tight')
  plt.close(fig)


def _create_parameter_correlation_plot(param_df: pd.DataFrame, arch: str, out_dir: Path, dpi: int) -> None:
  """Create correlation matrix plot showing relationships between parameters and performance."""
  # Find numeric parameters (exclude performance metrics)
  numeric_params = []
  for col in param_df.columns:
    if (col not in ['model', 'val_mae_mean', 'train_mae_mean', 'simplified_name'] and
        not col.startswith('val_mae_fold') and
        not col.startswith('train_mae_fold') and
        not col.endswith('_std') and
        param_df[col].dtype in ['int64', 'float64']):
      numeric_params.append(col)

  if not numeric_params:
    return

  # Create correlation matrix (only architecture parameters)
  corr_data = param_df[numeric_params].corr()

  # Create the plot
  fig, ax = plt.subplots(figsize=(10, 8))

  # Create heatmap
  im = ax.imshow(corr_data, cmap='RdBu_r', vmin=-1, vmax=1)

  # Set ticks and labels
  ax.set_xticks(range(len(corr_data.columns)))
  ax.set_yticks(range(len(corr_data.columns)))
  ax.set_xticklabels(corr_data.columns, rotation=45, ha='right')
  ax.set_yticklabels(corr_data.columns)

  # Add correlation values as text
  for i in range(len(corr_data.columns)):
    for j in range(len(corr_data.columns)):
      value = corr_data.iloc[i, j]
      if not pd.isna(value):
        text_color = 'white' if abs(value) > 0.5 else 'black'
        ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                color=text_color, fontsize=9, fontweight='bold')

  # Add colorbar
  cbar = plt.colorbar(im, ax=ax, shrink=0.8)
  cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)

  ax.set_title(f'{arch.upper()}: Parameter-Performance Correlations', fontsize=14, fontweight='bold')

  plt.tight_layout()
  out_file = out_dir / f'architecture_correlation_{arch}.pdf'
  fig.savefig(out_file, dpi=dpi, bbox_inches='tight')
  plt.close(fig)


def _create_parameter_performance_correlation_plot(param_df: pd.DataFrame, arch: str, out_dir: Path, dpi: int) -> None:
  """Create bar plot showing correlation of each parameter with validation MAE."""
  # Find all parameters (numeric and string)
  all_params = []
  for col in param_df.columns:
    if col not in ['model', 'val_mae_mean', 'train_mae_mean', 'simplified_name']:
      all_params.append(col)

  if not all_params or 'val_mae_mean' not in param_df.columns:
    return

  # Calculate correlations for each parameter
  correlations = []
  param_names = []

  for param in all_params:
    if param_df[param].dtype in ['int64', 'float64']:
      # Numeric parameter - use Pearson correlation
      corr = param_df[param].corr(param_df['val_mae_mean'])
      if not pd.isna(corr):
        correlations.append(corr)
        param_names.append(param)
    else:
      # String parameter - use one-hot encoding and correlation
      # Create dummy variables for each unique value
      dummies = pd.get_dummies(param_df[param], prefix=param)
      if len(dummies.columns) > 1:  # Only if more than one unique value
        # Calculate correlation for each dummy variable
        for col in dummies.columns:
          corr = dummies[col].corr(param_df['val_mae_mean'])
          if not pd.isna(corr):
            correlations.append(corr)
            param_names.append(col)
      else:
        # Single value - no correlation to calculate
        continue

  if not correlations:
    return

  # Create the plot
  fig, ax = plt.subplots(figsize=(max(8, len(param_names) * 0.3), 6))

  # Create bar plot
  colors = ['red' if c < 0 else 'blue' for c in correlations]
  bars = ax.bar(range(len(param_names)), correlations, color=colors, alpha=0.7)

  # Add correlation values on bars
  for bar, corr in zip(bars, correlations, strict=False):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
            f'{corr:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
            fontsize=9, fontweight='bold')

  # Customize plot
  ax.set_xticks(range(len(param_names)))
  ax.set_xticklabels(param_names, rotation=45, ha='right')
  ax.set_ylabel('Correlation with Validation MAE')
  ax.set_title(f'{arch.upper()}: Parameter-Validation MAE Correlations', fontsize=14, fontweight='bold')
  ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
  ax.grid(axis='y', linestyle='--', alpha=0.3)

  # Add legend
  from matplotlib.patches import Patch
  legend_elements = [
    Patch(facecolor='blue', alpha=0.7, label='Positive Correlation'),
    Patch(facecolor='red', alpha=0.7, label='Negative Correlation')
  ]
  ax.legend(handles=legend_elements, loc='upper right')

  plt.tight_layout()
  out_file = out_dir / f'architecture_performance_correlation_{arch}.pdf'
  fig.savefig(out_file, dpi=dpi, bbox_inches='tight')
  plt.close(fig)


def _create_parameter_value_performance_plot(param_df: pd.DataFrame, arch: str, out_dir: Path, dpi: int) -> None:
  """Create plot showing parameter values vs validation MAE performance."""
  # Find all parameters (numeric and string)
  all_params = []
  for col in param_df.columns:
    if col not in ['model', 'val_mae_mean', 'train_mae_mean', 'simplified_name']:
      all_params.append(col)

  if not all_params or 'val_mae_mean' not in param_df.columns:
    return

  # Create subplots for each parameter
  n_params = len(all_params)
  n_cols = min(3, n_params)
  n_rows = (n_params + n_cols - 1) // n_cols

  fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
  if n_params == 1:
    axes = [axes]
  elif n_rows == 1 and n_cols == 1:
    axes = [axes]
  elif n_rows == 1 or n_cols == 1:
    axes = axes.flatten()
  else:
    axes = axes.flatten()

  for i, param in enumerate(all_params):
    ax = axes[i]

    if param_df[param].dtype in ['int64', 'float64']:
      # Numeric parameter - scatter plot with trend line
      values = param_df[param].dropna()
      val_mae = param_df.loc[values.index, 'val_mae_mean']

      # Remove rows where either value is NaN
      valid_mask = pd.notna(values) & pd.notna(val_mae)
      values_clean = values[valid_mask]
      val_mae_clean = val_mae[valid_mask]

      if len(values_clean) > 1:
        # Scatter plot
        ax.scatter(values_clean, val_mae_clean, alpha=0.7, s=50, color='steelblue')

        # Add trend line
        if len(values_clean) > 2:
          try:
            z = np.polyfit(values_clean, val_mae_clean, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(values_clean.min(), values_clean.max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
          except (np.linalg.LinAlgError, Exception):
            # Skip trend line if polyfit fails
            pass

        # Calculate and display correlation
        corr = values_clean.corr(val_mae_clean)
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                fontsize=10, fontweight='bold',
                bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})

        ax.set_xlabel(param)
        ax.set_ylabel('Validation MAE')
        ax.set_title(f'{arch.upper()}: {param} vs Val MAE')
        ax.grid(True, alpha=0.3)

    else:
      # String parameter - scatter plot with jittered x-values
      values = param_df[param].dropna()
      val_mae = param_df.loc[values.index, 'val_mae_mean']

      # Remove rows where either value is NaN
      valid_mask = pd.notna(values) & pd.notna(val_mae)
      values_clean = values[valid_mask]
      val_mae_clean = val_mae[valid_mask]

      if len(values_clean) > 1:
        unique_vals = values_clean.unique()
        if len(unique_vals) > 1:
          # Create scatter plot with jittered x-values for string parameters
          x_positions = []
          for val in values_clean:
            # Find the index of this value in unique_vals
            val_idx = list(unique_vals).index(val)
            # Add small random jitter to avoid overlapping points
            jitter = np.random.normal(0, 0.1)
            x_positions.append(val_idx + jitter)

          # Scatter plot
          ax.scatter(x_positions, val_mae_clean, alpha=0.7, s=50, color='steelblue')

          # Set x-axis labels
          ax.set_xticks(range(len(unique_vals)))
          ax.set_xticklabels(unique_vals, rotation=45, ha='right')
          ax.set_xlabel(param)
          ax.set_ylabel('Validation MAE')
          ax.set_title(f'{arch.upper()}: {param} vs Val MAE')
          ax.grid(True, alpha=0.3)

          # Calculate and display correlation (using numeric encoding)
          numeric_values = [list(unique_vals).index(val) for val in values_clean]
          corr = np.corrcoef(numeric_values, val_mae_clean)[0, 1]
          ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                  fontsize=10, fontweight='bold',
                  bbox={ 'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
        else:
          # Single value
          ax.text(0.5, 0.5, f'Single value: {unique_vals[0]}',
                  ha='center', va='center', transform=ax.transAxes)
          ax.set_title(f'{arch.upper()}: {param} vs Val MAE')

  # Hide unused subplots
  for i in range(n_params, len(axes)):
    axes[i].set_visible(False)

  plt.tight_layout()
  out_file = out_dir / f'architecture_value_performance_{arch}.pdf'
  fig.savefig(out_file, dpi=dpi, bbox_inches='tight')
  plt.close(fig)


def _create_parameter_interaction_plot(param_df: pd.DataFrame, arch: str, out_dir: Path, dpi: int) -> None:
  """Create heatmap showing parameter interaction effects on performance."""
  # Find numeric parameters (exclude performance metrics)
  numeric_params = []
  for col in param_df.columns:
    if (col not in ['model', 'val_mae_mean', 'train_mae_mean', 'simplified_name'] and
        not col.startswith('val_mae_fold') and
        not col.startswith('train_mae_fold') and
        not col.endswith('_std') and
        param_df[col].dtype in ['int64', 'float64']):
      numeric_params.append(col)

  if len(numeric_params) < 2:
    return

  # Determine performance column to use
  if 'val_mae_mean' in param_df.columns and not param_df['val_mae_mean'].isna().all():
    perf_col = 'val_mae_mean'
  elif 'train_mae_mean' in param_df.columns and not param_df['train_mae_mean'].isna().all():
    perf_col = 'train_mae_mean'
  else:
    return  # No valid performance data

  # Create interaction matrix
  n_params = len(numeric_params)
  interaction_matrix = np.zeros((n_params, n_params))

  for i, param1 in enumerate(numeric_params):
    for j, param2 in enumerate(numeric_params):
      if i != j:
        # Calculate interaction effect (correlation between param1*param2 and performance)
        interaction_term = param_df[param1] * param_df[param2]
        interaction_matrix[i, j] = interaction_term.corr(param_df[perf_col])
      else:
        # Diagonal: individual parameter correlation with performance
        interaction_matrix[i, j] = param_df[param1].corr(param_df[perf_col])

  # Create the plot
  fig, ax = plt.subplots(figsize=(max(8, n_params * 0.8), max(6, n_params * 0.8)))

  # Create heatmap
  im = ax.imshow(interaction_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

  # Set ticks and labels
  ax.set_xticks(range(n_params))
  ax.set_yticks(range(n_params))
  ax.set_xticklabels(numeric_params, rotation=45, ha='right')
  ax.set_yticklabels(numeric_params)

  # Add values as text
  for i in range(n_params):
    for j in range(n_params):
      value = interaction_matrix[i, j]
      if not pd.isna(value):
        text_color = 'white' if abs(value) > 0.5 else 'black'
        ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                color=text_color, fontsize=9, fontweight='bold')

  # Add colorbar
  cbar = plt.colorbar(im, ax=ax, shrink=0.8)
  perf_label = 'Validation MAE' if perf_col == 'val_mae_mean' else 'Training MAE'
  cbar.set_label(f'Correlation with {perf_label}', rotation=270, labelpad=20)

  ax.set_title(f'{arch.upper()}: Parameter Interaction Effects', fontsize=14, fontweight='bold')

  plt.tight_layout()
  out_file = out_dir / f'architecture_interaction_{arch}.pdf'
  fig.savefig(out_file, dpi=dpi, bbox_inches='tight')
  plt.close(fig)


def _create_coverage_gaps_plot(param_df: pd.DataFrame, arch: str, out_dir: Path, dpi: int) -> None:
  """Create plot showing parameter space coverage gaps and completeness."""
  # Find numeric parameters (exclude performance metrics)
  numeric_params = []
  for col in param_df.columns:
    # Exclude performance metrics and fold data
    if (col not in ['model', 'val_mae_mean', 'train_mae_mean', 'train_mae', 'simplified_name'] and
        not col.startswith('val_mae_fold') and
        not col.startswith('train_mae_fold') and
        not col.endswith('_std') and
        param_df[col].dtype in ['int64', 'float64']):
      numeric_params.append(col)

  if len(numeric_params) < 2:
    return

  # Create 2D parameter space coverage plots for top parameter pairs
  # Find most important parameters (highest correlation with performance)
  # Use validation MAE if available, otherwise fall back to training MAE
  if 'val_mae_mean' in param_df.columns and not param_df['val_mae_mean'].isna().all():
    mae_col = 'val_mae_mean'
  elif 'train_mae_mean' in param_df.columns and not param_df['train_mae_mean'].isna().all():
    mae_col = 'train_mae_mean'
  else:
    mae_col = None

  if mae_col is not None:
    correlations = []
    for param in numeric_params:
      corr = abs(param_df[param].corr(param_df[mae_col]))
      if not pd.isna(corr):
        correlations.append((param, corr))

    # Sort by correlation strength
    correlations.sort(key=lambda x: x[1], reverse=True)
    top_params = [p[0] for p in correlations[:4]]  # Top 4 parameters
  else:
    # Fallback: use first 4 numeric parameters if no valid validation data
    top_params = numeric_params[:4]

  if len(top_params) < 2:
    return

  # Create subplots for parameter pairs
  n_pairs = min(6, len(top_params) * (len(top_params) - 1) // 2)
  n_cols = min(3, n_pairs)
  n_rows = (n_pairs + n_cols - 1) // n_cols

  fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
  if n_pairs == 1:
    axes = [axes]
  elif n_rows == 1 and n_cols == 1:
    axes = [axes]
  elif n_rows == 1 or n_cols == 1:
    axes = axes.flatten()
  else:
    axes = axes.flatten()

  pair_idx = 0
  for i, param1 in enumerate(top_params):
    for _j, param2 in enumerate(top_params[i+1:], i+1):
      if pair_idx >= n_pairs:
        break

      ax = axes[pair_idx]

      # Get data
      values1 = param_df[param1].dropna()
      values2 = param_df[param2].dropna()

      # Find common indices
      common_idx = values1.index.intersection(values2.index)
      if len(common_idx) > 0:
        v1_common = param_df.loc[common_idx, param1]
        v2_common = param_df.loc[common_idx, param2]

        # Create scatter plot
        # Use validation MAE if available, otherwise fall back to training MAE
        if 'val_mae_mean' in param_df.columns and not param_df['val_mae_mean'].isna().all():
          mae_col = 'val_mae_mean'
          mae_label = 'Validation MAE'
        elif 'train_mae_mean' in param_df.columns and not param_df['train_mae_mean'].isna().all():
          mae_col = 'train_mae_mean'
          mae_label = 'Training MAE'
        else:
          mae_col = None
          mae_label = None

        if mae_col is not None:
          mae_values = param_df.loc[common_idx, mae_col]

          # Only use non-NaN values for coloring
          valid_color_mask = pd.notna(mae_values)
          if valid_color_mask.any():
            x_vals = v1_common[valid_color_mask]
            y_vals = v2_common[valid_color_mask]
            c_vals = mae_values[valid_color_mask]

            # Create scatter plot
            scatter = ax.scatter(x_vals, y_vals, c=c_vals, cmap='viridis', alpha=0.7, s=50)


            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label(mae_label, rotation=270, labelpad=15)
          else:
            ax.scatter(v1_common, v2_common, alpha=0.7, s=50, color='steelblue')
        else:
          ax.scatter(v1_common, v2_common, alpha=0.7, s=50, color='steelblue')

        # Add grid to show coverage
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_title(f'{arch.upper()}: {param1} vs {param2}')

        # Add coverage statistics
        coverage_text = f'Models: {len(common_idx)}'
        ax.text(0.05, 0.95, coverage_text, transform=ax.transAxes,
                fontsize=9, bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
      else:
        ax.text(0.5, 0.5, 'No common data points', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{arch.upper()}: {param1} vs {param2}')

      pair_idx += 1

  # Hide unused subplots
  for i in range(n_pairs, len(axes)):
    axes[i].set_visible(False)

  plt.tight_layout()
  out_file = out_dir / f'architecture_coverage_gaps_{arch}.pdf'
  fig.savefig(out_file, dpi=dpi, bbox_inches='tight')
  plt.close(fig)


def _render_benchmark_plots(df: pd.DataFrame, out_dir: Path, dpi: int, log_scale: bool) -> None:
  """Create benchmark plots for each model that has benchmark results."""
  # Find all benchmark columns
  bench_cols = [col for col in df.columns if col.startswith('bench_') and col.endswith('_mae')]
  if not bench_cols:
    return

  # Extract benchmark names (remove 'bench_' prefix and '_mae' suffix)
  bench_names = [col[6:-4] for col in bench_cols]  # Remove 'bench_' and '_mae'

  # Create plots for each model that has benchmark data
  for idx, (_, row) in enumerate(df.iterrows()):
    model_name = row.get('model', f'Model_{idx}')
    if pd.isna(model_name):
      continue

    # Check if this model has any benchmark data
    has_benchmark = False
    for bench_name in bench_names:
      mae_col = f'bench_{bench_name}_mae'
      if mae_col in row and pd.notna(row[mae_col]):
        has_benchmark = True
        break

    if not has_benchmark:
      continue

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Collect data for this model
    bench_data = []
    bench_labels = []
    mae_values = []
    mae_mins = []
    mae_maxs = []

    for bench_name in bench_names:
      mae_col = f'bench_{bench_name}_mae'
      min_col = f'bench_{bench_name}_mae_min'
      max_col = f'bench_{bench_name}_mae_max'

      if mae_col in row and pd.notna(row[mae_col]):
        mae_val = float(row[mae_col])
        mae_min = float(row[min_col]) if min_col in row and pd.notna(row[min_col]) else mae_val
        mae_max = float(row[max_col]) if max_col in row and pd.notna(row[max_col]) else mae_val

        bench_data.append(bench_name)
        bench_labels.append(bench_name.replace('_', ' ').title())
        mae_values.append(mae_val)
        mae_mins.append(mae_min)
        mae_maxs.append(mae_max)

    if not mae_values:
      continue

    # Create bar plot
    x_pos = np.arange(len(bench_data))
    bars = ax.bar(x_pos, mae_values, alpha=0.7, color='steelblue', capsize=5)

    # Add error bars (min/max range)
    errors_lower = [val - min_val for val, min_val in zip(mae_values, mae_mins, strict=False)]
    errors_upper = [max_val - val for val, max_val in zip(mae_values, mae_maxs, strict=False)]
    ax.errorbar(x_pos, mae_values, yerr=[errors_lower, errors_upper],
                fmt='none', color='black', capsize=3, capthick=1)

    # Customize plot
    ax.set_xlabel('Benchmark Dataset')
    ax.set_ylabel('MAE')
    ax.set_title(f'Benchmark Results: {model_name}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bench_labels, rotation=45, ha='right')

    if log_scale:
      ax.set_yscale('log')

    # Add value labels on bars
    for bar, val in zip(bars, mae_values, strict=False):
      height = bar.get_height()
      ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
              f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save plot
    safe_model_name = str(model_name).replace('/', '_').replace(' ', '_')
    out_file = out_dir / f'benchmark_{safe_model_name}.pdf'
    fig.savefig(out_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def _render_top_table(df: pd.DataFrame, title: str, out_pdf: Path, dpi: int, log_scale: bool, top_n: int = 25, sensitivity: str = 'high') -> None:
  if df.empty:
    return
  os.makedirs(out_pdf.parent, exist_ok=True)
  # Improve readability: larger base font
  plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
  })

  with PdfPages(out_pdf) as pdf:
    n_models = min(len(df), top_n)
    rows = df.head(n_models)

    # Perform statistical significance testing with configurable sensitivity
    significant_pairs = _statistical_significance_test(rows, sensitivity=sensitivity)

    # Determine parameter columns and which vary across selected rows
    candidate_params = [
      'model', 'dataset', 'metabolites', 'pulse', 'acq', 'datatype', 'norm', 'batch', 'epochs'
    ]
    present_cols = [c for c in candidate_params if c in rows.columns]
    varying_cols: list[str] = []
    common_cols: list[str] = []
    for c in present_cols:
      nun = rows[c].astype(str).nunique(dropna=True)
      if nun > 1:
        varying_cols.append(c)
      else:
        common_cols.append(c)

    # Always display certain keys even if constant, to avoid empty-looking rows
    for must_show in ('model', 'datatype'):
      if must_show in present_cols and must_show not in varying_cols:
        varying_cols.insert(0, must_show)
        if must_show in common_cols:
          common_cols.remove(must_show)

    # Two-row header policy:
    # - Top row shows only 'model' (if present)
    # - Bottom row shows all other varying parameters
    has_model = 'model' in varying_cols or 'model' in present_cols
    header_top_params = ['model'] if has_model else []
    header_bottom_params = [p for p in varying_cols if p != 'model']
    num_param_cols = max(1, max(len(header_top_params), len(header_bottom_params)))

    # Layout: 2 header rows, footer row; 2 rows per model; columns = param cols + 2 (MAE, Plot)
    model_rows = n_models * 2
    header_rows = 2
    total_rows = header_rows + model_rows + 1
    first_model_row = header_rows
    footer_row = total_rows - 1
    total_cols = num_param_cols + 2
    # Width ratios: param columns slightly narrower, then mae, then wide plot
    width_ratios = [1.2] * num_param_cols + [1.3, 3.8]
    # Compact, high-density layout to reduce whitespace
    height_units = max(11.7, 0.34 * model_rows + 1.4)
    width_units = max(8.3, 2 * num_param_cols + 8.0)
    fig = plt.figure(figsize=(width_units, height_units))
    gs = fig.add_gridspec(total_rows, total_cols, width_ratios=width_ratios, wspace=0.25, hspace=0.02)
    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.04, right=0.99, hspace=0.02, wspace=0.25)

    # Header rows: parameter names per cell, then Val/Train/Plot labels in top header row
    for ci in range(num_param_cols):
      ax_ht = fig.add_subplot(gs[0, ci])
      ax_ht.axis('off')
      name_top = header_top_params[ci] if ci < len(header_top_params) else ''
      if name_top:
        ax_ht.text(0.01, 0.65, name_top.capitalize(), ha='left', va='center', fontsize=12, fontweight='bold')
      ax_hb = fig.add_subplot(gs[1, ci])
      ax_hb.axis('off')
      name_bot = header_bottom_params[ci] if ci < len(header_bottom_params) else ''
      if name_bot:
        ax_hb.text(0.01, 0.65, name_bot.capitalize(), ha='left', va='center', fontsize=12, fontweight='bold')
    # MAE header with two rows (Val/Train)
    ax_mae_h_top = fig.add_subplot(gs[0, num_param_cols])
    ax_mae_h_top.axis('off')
    ax_mae_h_top.text(0.5, 0.65, 'Val. MAE ± Std', ha='center', va='center', fontsize=12, fontweight='bold')
    ax_mae_h_bot = fig.add_subplot(gs[1, num_param_cols])
    ax_mae_h_bot.axis('off')
    ax_mae_h_bot.text(0.5, 0.65, 'Train MAE ± Std', ha='center', va='center', fontsize=12, fontweight='bold')

    # Plot header
    ax_plot_h = fig.add_subplot(gs[0, num_param_cols + 1])
    ax_plot_h.axis('off')
    ax_plot_h.text(0.5, 0.7, 'Plot', ha='center', va='center', fontsize=12, fontweight='bold')

    # Compute global x-limits for plot bars across selected rows
    def safe_float(x):
      try:
        return float(x)
      except Exception:
        return np.nan
    vals = []
    for _, rr in rows.iterrows():
      vm = safe_float(rr.get('val_mae_mean'))
      tm = safe_float(rr.get('train_mae_mean'))
      # Prefer min/max across folds if present, else fall back to mean±std
      vmin = safe_float(rr.get('val_mae_min'))
      vmax = safe_float(rr.get('val_mae_max'))
      tmin = safe_float(rr.get('train_mae_min'))
      tmax = safe_float(rr.get('train_mae_max'))
      vs = safe_float(rr.get('val_mae_std')) if pd.notna(rr.get('val_mae_std')) else np.nan
      ts = safe_float(rr.get('train_mae_std')) if pd.notna(rr.get('train_mae_std')) else np.nan
      if not np.isnan(vm):
        lo = vmin if not np.isnan(vmin) else (vm - vs if not np.isnan(vs) else vm)
        hi = vmax if not np.isnan(vmax) else (vm + vs if not np.isnan(vs) else vm)
        vals.append(max(0.0, lo))
        vals.append(hi)
      if not np.isnan(tm):
        lo = tmin if not np.isnan(tmin) else (tm - ts if not np.isnan(ts) else tm)
        hi = tmax if not np.isnan(tmax) else (tm + ts if not np.isnan(ts) else tm)
        vals.append(max(0.0, lo))
        vals.append(hi)
    if log_scale:
      # For log scale, ensure positive values and set reasonable bounds
      x_min = 1e-6 if not vals else max(1e-6, np.nanmin(vals) * 0.1)
      x_max = 1.0 if not vals else np.nanmax(vals) * 10.0
    else:
      x_min = 0.0 if not vals else max(0.0, np.nanmin(vals) * 0.9)
      x_max = 1.0 if not vals else np.nanmax(vals) * 1.1

    for idx, (_, r) in enumerate(rows.iterrows()):
      top = first_model_row + 2 * idx
      bot = top + 1
      # Parameter cells: values aligned with header parameter names
      # Top parameter row: show the model value spanning all parameter columns
      if num_param_cols > 0:
        ax_model_span = fig.add_subplot(gs[top, 0:num_param_cols])
        ax_model_span.axis('off')
        mval = r.get('model')
        # Use simplified name if available, otherwise use full model name
        if 'simplified_name' in r and pd.notna(r.get('simplified_name')):
          mtxt = str(r.get('simplified_name'))
        else:
          mtxt = str(mval) if (pd.notna(mval) and str(mval) not in ('', 'nan')) else '-'
        # Align left within the spanned area for readability
        ax_model_span.text(0.01, 0.55, mtxt, ha='left', va='center', fontsize=12, family='monospace')

      for ci in range(num_param_cols):
        # bottom row value-only cells for non-model params
        # create top per-cell axes but keep empty to avoid overlaying the spanned text
        # (skip creating top per-cell axes to prevent covering the spanning text)
        # bottom row value
        ax_pb = fig.add_subplot(gs[bot, ci])
        ax_pb.axis('off')
        if ci < len(header_bottom_params):
          pname = header_bottom_params[ci]
          val = r.get(pname)
          if pd.notna(val) and str(val) != 'nan' and str(val) != '':
            txt = _shorten_acq_text(val) if pname == 'acq' else str(val)
          else:
            txt = '-'
          ax_pb.text(0.01, 0.55, txt, ha='left', va='center', fontsize=12, family='monospace')

      # Collect metrics
      v_mean = r.get('val_mae_mean')
      t_mean = r.get('train_mae_mean')
      v_std = r.get('val_mae_std')
      t_std = r.get('train_mae_std')
      v_folds = _collect_folds(r, 'val_mae')
      t_folds = _collect_folds(r, 'train_mae')
      v_min = (np.nanmin(v_folds) if len(v_folds) > 0 else r.get('val_mae_min'))
      v_max = (np.nanmax(v_folds) if len(v_folds) > 0 else r.get('val_mae_max'))
      t_min = (np.nanmin(t_folds) if len(t_folds) > 0 else r.get('train_mae_min'))
      t_max = (np.nanmax(t_folds) if len(t_folds) > 0 else r.get('train_mae_max'))

      # Column 1: MAE text spanning two rows (validation on top, training on bottom)
      ax_mae = fig.add_subplot(gs[top:bot+1, num_param_cols])
      ax_mae.axis('off')

      # Validation MAE on top row
      if pd.notna(v_mean):
        txt = f"{float(v_mean):.5f}"
        if pd.notna(v_std):
          txt += f" ± {float(v_std):.5f}"
        ax_mae.text(0.5, 0.75, txt, ha='center', va='center', fontsize=11, color='#1f4e79')

      # Training MAE on bottom row
      if pd.notna(t_mean):
        txt = f"{float(t_mean):.5f}"
        if pd.notna(t_std):
          txt += f" ± {float(t_std):.5f}"
        ax_mae.text(0.5, 0.25, txt, ha='center', va='center', fontsize=11, color='#8b0000')

      # Column 2: Bars (two axes: top validation, bottom training)
      # Single plot axes spanning both sub-rows: horizontal bars for Val and Train
      ax_plot = fig.add_subplot(gs[top:bot+1, num_param_cols + 1])
      ax_plot.set_xlim(x_min, x_max)
      ax_plot.set_yticks([])
      if log_scale:
        ax_plot.set_xscale('log')
      # Validation bar
      if pd.notna(v_mean):
        vm = float(v_mean)
        # Compute asymmetric error from min/max across folds when available
        v_lower = None
        v_upper = None
        if pd.notna(v_min) and pd.notna(v_max):
          v_lower = max(0.0, vm - float(v_min))
          v_upper = max(0.0, float(v_max) - vm)
        else:
          vs = float(v_std) if pd.notna(v_std) else 0.0
          v_lower = vs
          v_upper = vs
        ax_plot.barh([0.5], [vm], height=0.1, xerr=[[v_lower], [v_upper]], capsize=2, color='steelblue', alpha=0.9, error_kw={'elinewidth': 0.8, 'capthick': 0.8})
        for vf in v_folds:
          ax_plot.plot(vf, 0.5, 'o', color='black', ms=2.5)
      # Training bar
      if pd.notna(t_mean):
        tm = float(t_mean)
        t_lower = None
        t_upper = None
        if pd.notna(t_min) and pd.notna(t_max):
          t_lower = max(0.0, tm - float(t_min))
          t_upper = max(0.0, float(t_max) - tm)
        else:
          ts = float(t_std) if pd.notna(t_std) else 0.0
          t_lower = ts
          t_upper = ts
        ax_plot.barh([0.4], [tm], height=0.1, xerr=[[t_lower], [t_upper]], capsize=2, color='salmon', alpha=0.9, error_kw={'elinewidth': 0.8, 'capthick': 0.8})
        for tf in t_folds:
          ax_plot.plot(tf, 0.4, 'o', color='black', ms=2.5)
      ax_plot.grid(axis='x', linestyle='--', alpha=0.35)
      # Show x-axis labels only on the bottom plot
      is_last = (idx == len(rows) - 1)
      if not is_last:
        ax_plot.set_xticklabels([])
      else:
        ax_plot.tick_params(axis='x', labelsize=9)
      for spine in ax_plot.spines.values():
        spine.set_visible(False)

      # Add legend only to the first plot
      if idx == 0:
        from matplotlib.patches import Patch
        legend_elements = [
          Patch(facecolor='steelblue', alpha=0.9, label='Validation'),
          Patch(facecolor='salmon', alpha=0.9, label='Training')
        ]
        ax_plot.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.8)

      # Add statistical significance indicators
      # Check if this model is part of any significant pair (using sorted positions)
      for pair_i, pair_j in significant_pairs:
        if idx == pair_i or idx == pair_j:
          # Draw a horizontal red line across the entire row (all columns)
          # Position the line at the bottom of the model row (after training result)
          ax_separator = fig.add_subplot(gs[top:bot+1, :])
          ax_separator.axis('off')
          ax_separator.axhline(y=0.0, color='red', linewidth=2, alpha=0.7, linestyle='--')
          break  # Only add one indicator per model

    # Footer row: list parameters that are the same across the selection
    ax_footer = fig.add_subplot(gs[footer_row, :])
    ax_footer.axis('off')
    common_items = []
    for c in common_cols:
      # Use the single unique value
      val = rows[c].iloc[0]
      if pd.isna(val):
        continue
      common_items.append(f"{c}={val}")

    footer_text_parts = []
    if common_items:
      footer_text_parts.append(", ".join(common_items))

    # Add statistical significance explanation
    if significant_pairs:
      sensitivity_desc = {
        'standard': 'α=0.05',
        'high': 'α=0.10',
        'very_high': 'α=0.20',
        'effect_size': 'α=0.05 + Cohen\'s d>0.2'
      }.get(sensitivity, 'enhanced')
      footer_text_parts.append(f"Red dashed lines separate models with statistically different Validation MAE distributions (Mann-Whitney U test, {sensitivity_desc})")

    if footer_text_parts:
      footer_text = " | ".join(footer_text_parts)
      ax_footer.text(0.01, -0.2, footer_text, ha='left', va='center', fontsize=12)

    pdf.savefig(fig, bbox_inches='tight', dpi=dpi)
    plt.close(fig)


def main() -> None:
  """Visualize aggregate CSV into PDF plots/tables."""
  parser = argparse.ArgumentParser(description='Visualize aggregate CSV into PDF plots/tables.')
  parser.add_argument('root', type=str, help='Model root folder')
  parser.add_argument('--top_n', type=int, default=25, help='Top N entries to show')
  parser.add_argument('--dpi', type=int, default=300, help='Output DPI for saved PDFs')
  parser.add_argument('--log_scale', action='store_true', help='Use logarithmic scale for x-axis of bar plots')
  parser.add_argument('--sensitivity', type=str, default='high', choices=['standard', 'high', 'very_high', 'effect_size'],
                     help='Statistical test sensitivity: standard (α=0.05), high (α=0.10), very_high (α=0.20), effect_size (α=0.05 + Cohen\'s d>0.2). Red lines separate models with significantly different Validation MAE distributions.')
  args = parser.parse_args()

  root = Path(args.root).resolve()
  df = _read_aggregate(root)
  out_dir = root / 'aggregate'

  # Concentration results: overall top-N
  top_conc = _select_topn_conc(df, args.top_n)
  _render_top_table(top_conc, f'Top {args.top_n} Concentration Models (Overall)', out_dir / 'top_concentration_overall.pdf', args.dpi, args.log_scale, args.top_n, args.sensitivity)

  # Concentration results per-architecture (only if multiple architectures exist)
  df['arch'] = df['model'].astype(str).map(_arch_of_model)
  conc_archs_present = df[df['arch'].isin(CONC_ARCHS)]['arch'].dropna().unique()
  if len(conc_archs_present) > 1:
    for arch in CONC_ARCHS:
      sub = df[df['arch'] == arch]
      if len(sub) > 1:
        top_arch = _select_topn_conc(sub, args.top_n)
        _render_top_table(top_arch, f'Top {args.top_n} {arch.upper()} Concentration Models', out_dir / f'top_concentration_{arch}.pdf', args.dpi, args.log_scale, args.top_n, args.sensitivity)

  # Spectra results: overall and per-arch (ae, caeq)
  top_spec = _select_topn_spec(df, args.top_n)
  _render_top_table(top_spec, f'Top {args.top_n} Spectra Models (Overall)', out_dir / 'top_spectra_overall.pdf', args.dpi, args.log_scale, args.top_n, args.sensitivity)
  spec_archs_present = df[df['arch'].isin(SPEC_ARCHS)]['arch'].dropna().unique()
  if len(spec_archs_present) > 1:
    for arch in SPEC_ARCHS:
      sub = df[df['arch'] == arch]
      if len(sub) > 1:
        top_arch = _select_topn_spec(sub, args.top_n)
        _render_top_table(top_arch, f'Top {args.top_n} {arch.upper()} Spectra Models', out_dir / f'top_spectra_{arch}.pdf', args.dpi, args.log_scale, args.top_n, args.sensitivity)

  # Benchmark plots for each model
  _render_benchmark_plots(df, out_dir, args.dpi, args.log_scale)

  # Architecture parameter analysis
  _render_architecture_analysis(df, out_dir, args.dpi, args.log_scale)

  print(f"Wrote visualizations to: {out_dir}")


if __name__ == '__main__':
  main()


