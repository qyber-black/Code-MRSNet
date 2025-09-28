#!/usr/bin/env python3
#
# collect_performance.py
#
# SPDX-FileCopyrightText: Copyright (C) 2025 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Enhanced Model Performance Data Collection Script

This script collects performance data from model training runs and benchmark evaluations,
generating comprehensive reports in HTML format with improved organization and highlighting.

Features:
- Collects training/validation performance metrics
- Processes individual benchmark series (E1, E2, E3, etc.)
- Generates metabolite-specific analysis tables
- Highlights best performing values
- Supports multiple model directories
- Configuration-based table organization
- Shortened benchmark names (E1, E2, etc.)
- Standard deviation reporting
- Linear regression results (slope, R²)
- Prominent table group separation
"""

import argparse
import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class EnhancedModelPerformanceCollector:
    """Enhanced collector for model performance data with improved reporting."""

    def __init__(self, model_dirs: Optional[List[str]] = None):
        """
        Initialize the collector.

        Args:
            model_dirs: List of model directory paths. If None, uses data/model directory.
        """
        if model_dirs is None:
            # Default to data/model directory
            self.model_dirs = ["data/model"]
        else:
            self.model_dirs = model_dirs

        self.results = []

    def collect_all_performance_data(self) -> List[Dict[str, Any]]:
        """Collect performance data from all specified model directories."""
        print("Collecting performance data from all models...")

        for model_dir in self.model_dirs:
            print(f"  Processing directory: {model_dir}")
            self._collect_from_directory(model_dir)

        print(f"Collected performance data from {len(self.results)} model runs")
        return self.results

    def _collect_from_directory(self, directory: str):
        """Collect performance data from a single directory."""
        dir_path = Path(directory)

        if not dir_path.exists():
            print(f"    Warning: Directory {directory} does not exist")
            return

        # Walk through all subdirectories
        for root, dirs, files in os.walk(dir_path):
            root_path = Path(root)

            # Look for model run directories (containing train_concentration_errors.json)
            if 'train_concentration_errors.json' in files:
                model_name = root_path.name
                parent_name = root_path.parent.name

                print(f"  Processing {model_name} from {parent_name}")

                # Extract configuration from path
                config = self._extract_config_from_path(root_path)

                # Use the actual model type from the path, not split
                model_type = config.get('model_type', model_name)

                # Collect training performance
                train_perf = self._extract_concentration_errors(
                    root_path / 'train_concentration_errors.json'
                )

                # Collect validation performance
                val_perf = self._extract_concentration_errors(
                    root_path / 'validation_concentration_errors.json'
                )

                # Collect benchmark performance
                benchmark_perf = self._extract_benchmark_performance(root_path)

                # Collect training history
                history_data = self._extract_training_history(root_path / 'history.csv')

                # Create result entry
                result = {
                    'parent_directory': parent_name,
                    'path': str(root_path),
                    'train_performance': train_perf,
                    'validation_performance': val_perf,
                    'benchmark_performance': benchmark_perf,
                    'training_history': history_data,
                    **config,
                    'model_type': model_type  # Override config model_type with split
                }

                self.results.append(result)

    def _extract_config_from_path(self, path: Path) -> Dict[str, str]:
        """Extract configuration information from the directory path."""
        config = {}

        # Parse path components
        parts = path.parts

        # Extract model type (first part after data/model or data/model-dist)
        model_type_idx = None
        for i, part in enumerate(parts):
            if part in ['model', 'model-dist']:
                model_type_idx = i + 1
                break

        if model_type_idx and model_type_idx < len(parts):
            config['model_type'] = parts[model_type_idx]

        # Extract metabolites (next part after model type)
        if model_type_idx and model_type_idx + 1 < len(parts):
            config['metabolites'] = parts[model_type_idx + 1]

        # Extract datatype (next part after metabolites)
        if model_type_idx and model_type_idx + 2 < len(parts):
            config['datatype'] = parts[model_type_idx + 2]

        # Extract edit type (next part after datatype)
        if model_type_idx and model_type_idx + 3 < len(parts):
            config['edit_type'] = parts[model_type_idx + 3]

        # Extract complex type (next part after edit type)
        if model_type_idx and model_type_idx + 4 < len(parts):
            config['complex_type'] = parts[model_type_idx + 4]

        # Extract batch size and epochs (look for numeric parts in sequence)
        # The structure is: .../sum/batch_size/epochs/...
        batch_size = None
        epochs = None
        for i, part in enumerate(parts):
            if part.isdigit():
                if batch_size is None:
                    batch_size = int(part)
                elif epochs is None:
                    epochs = int(part)
                    break

        config['batch_size'] = batch_size
        config['epochs'] = epochs

        # Extract split information
        for part in parts:
            if part.startswith('Split_'):
                config['split'] = part
                break

        # Extract acquisition from the long config string
        for part in parts:
            if 'siemens' in part.lower():
                config['acquisition'] = 'siemens'
                break
            elif 'philips' in part.lower():
                config['acquisition'] = 'philips'
                break
            elif 'ge' in part.lower():
                config['acquisition'] = 'ge'
                break

        # Set defaults if not found
        config.setdefault('model_type', 'unknown')
        config.setdefault('metabolites', 'unknown')
        config.setdefault('datatype', 'unknown')
        config.setdefault('edit_type', 'unknown')
        config.setdefault('complex_type', 'unknown')
        config.setdefault('batch_size', 'unknown')
        config.setdefault('epochs', 'unknown')
        config.setdefault('split', 'unknown')
        config.setdefault('acquisition', 'unknown')

        return config

    def _extract_concentration_errors(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract concentration error data from JSON file."""
        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Extract summary metrics
            summary = {}

            # Overall MAE - prioritize total.abserror.mean if available
            if 'total' in data and 'abserror' in data['total']:
                summary['overall_mae'] = data['total']['abserror'].get('mean')
                summary['overall_mae_std'] = data['total']['abserror'].get('std')
            else:
                # Fallback to per-metabolite average
                mae_values = []
                for metabolite, metrics in data.items():
                    if isinstance(metrics, dict) and 'abserror' in metrics:
                        mae_values.append(metrics['abserror'].get('mean'))

                summary['overall_mae'] = np.mean(mae_values) if mae_values else None
                summary['overall_mae_std'] = np.std(mae_values) if mae_values else None

            # Overall correlation (use r_value from linreg)
            if 'total' in data and 'linreg' in data['total'] and 'r_value' in data['total']['linreg']:
                summary['overall_correlation'] = data['total']['linreg']['r_value']
            else:
                # Fallback to per-metabolite average
                corr_values = []
                for metabolite, metrics in data.items():
                    if isinstance(metrics, dict) and 'linreg' in metrics and 'r_value' in metrics['linreg']:
                        corr_values.append(metrics['linreg']['r_value'])

                summary['overall_correlation'] = np.mean(corr_values) if corr_values else None

            # Per-metabolite data
            metabolites = {}
            for metabolite, metrics in data.items():
                if isinstance(metrics, dict):
                    metabolites[metabolite] = {
                        'mae': metrics.get('abserror', {}).get('mean'),
                        'mae_std': metrics.get('abserror', {}).get('std'),
                        'correlation': metrics.get('linreg', {}).get('r_value'),
                        'slope': metrics.get('linreg', {}).get('slope'),
                        'intercept': metrics.get('linreg', {}).get('intercept'),
                        'p_value': metrics.get('linreg', {}).get('p_value'),
                        'std_err': metrics.get('linreg', {}).get('std_err')
                    }

            return {
                'summary': summary,
                'metabolites': metabolites
            }

        except Exception as e:
            print(f"    Error reading {file_path}: {e}")
            return None

    def _normalize_benchmark_name(self, benchmark_name: str) -> str:
        """Normalize benchmark names to handle different naming conventions.

        Converts different naming patterns to a common format:
        - E1_MEGA_RAW_Combi_WS_ON_max -> E1_MEGA_RAW_Combi_WS_ON
        - E1:MEGA_RAW_Combi_WS_ON -> E1_MEGA_RAW_Combi_WS_ON
        - E1:MEGA_RAW_Combi_WS_ON_sum -> E1_MEGA_RAW_Combi_WS_ON
        """
        # Replace colons with underscores
        normalized = benchmark_name.replace(':', '_')

        # Remove common suffixes
        suffixes_to_remove = ['_max', '_sum']
        for suffix in suffixes_to_remove:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
                break

        return normalized

    def _extract_benchmark_performance(self, directory: Path) -> Dict[str, Dict[str, Any]]:
        """Extract benchmark performance data from concentration error files."""
        benchmark_perf = {}

        # Look for benchmark concentration error files
        for file_path in directory.glob('*_concentration_errors.json'):
            filename = file_path.name

            # Skip training and validation files
            if filename.startswith(('train_', 'validation_')):
                continue

            # Extract benchmark name (everything before _concentration_errors.json)
            benchmark_name = filename.replace('_concentration_errors.json', '')

            # Normalize the benchmark name
            normalized_name = self._normalize_benchmark_name(benchmark_name)

            # Extract performance data
            perf_data = self._extract_concentration_errors(file_path)
            if perf_data:
                benchmark_perf[normalized_name] = perf_data

        return benchmark_perf

    def _extract_training_history(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract training history data from CSV file."""
        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            if len(lines) < 4:
                return None

            # Parse the specific format used by MRSNet
            # Line 0: Model name
            # Line 1: Empty
            # Line 2: Headers (Metric, Train, Validation)
            # Line 3+: Data rows

            train_values = {}
            val_values = {}

            for line in lines[3:]:  # Skip header lines
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    metric = parts[0].strip()
                    try:
                        train_val = float(parts[1].strip()) if parts[1].strip() else None
                        val_val = float(parts[2].strip()) if parts[2].strip() else None

                        train_values[metric] = train_val
                        val_values[metric] = val_val
                    except ValueError:
                        continue

            # Extract specific metrics we're interested in
            summary = {
                'total_epochs': None,  # Not available in this format
                'final_train_loss': train_values.get('MSE'),
                'final_val_loss': val_values.get('MSE'),
                'final_train_mae': train_values.get('MAE'),
                'final_val_mae': val_values.get('MAE'),
                'min_train_loss': train_values.get('MSE'),  # Only final value available
                'min_val_loss': val_values.get('MSE'),
                'min_train_mae': train_values.get('MAE'),
                'min_val_mae': val_values.get('MAE')
            }

            return {
                'summary': summary,
                'history': []  # Raw history not available in this format
            }

        except Exception as e:
            print(f"    Error reading {file_path}: {e}")
            return None

    def generate_html_report(self) -> str:
        """Generate comprehensive HTML performance report."""
        html = self._get_html_template()

        # Add summary statistics
        html += self._create_summary_section()

        # Add comprehensive comparison tables
        html += self._create_comparison_tables()

        # Add metabolite-specific tables
        html += self._create_metabolite_benchmark_tables()

        html += "</body></html>"
        return html

    def _get_html_template(self) -> str:
        """Get the HTML template with enhanced styling."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Model Performance Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }
        h1, h2, h3 { color: #333; }
        h1 { text-align: center; color: #2c3e50; margin-bottom: 30px; }
        h2 {
            border-bottom: 4px solid #007acc;
            padding-bottom: 15px;
            margin-bottom: 25px;
            margin-top: 40px;
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
        }
        h3 {
            border-bottom: 3px solid #007acc;
            padding-bottom: 10px;
            margin-bottom: 20px;
            margin-top: 30px;
            background-color: #f0f8ff;
            padding: 10px;
            border-radius: 3px;
        }
        h4 {
            border-bottom: 2px solid #007acc;
            padding-bottom: 8px;
            margin-bottom: 15px;
            margin-top: 25px;
            background-color: #f8f9fa;
            padding: 8px;
            border-radius: 2px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            background-color: white;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }
        th {
            background-color: #007acc;
            color: white;
            font-weight: bold;
            text-align: center;
        }
        .best {
            background-color: #d4edda;
            font-weight: bold;
        }
        .overall-best {
            background-color: #d4edda;
            font-weight: bold;
            text-decoration: underline;
        }
        .good { background-color: #fff3cd; }
        .poor { background-color: #f8d7da; }
        .metric {
            font-family: monospace;
            text-align: right;
        }
        .metric.best {
            background-color: #d4edda;
            font-weight: bold;
        }
        .metric.overall-best {
            background-color: #d4edda;
            font-weight: bold;
            color: #155724;
        }
        .summary {
            background-color: #e9ecef;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 5px solid #007acc;
        }
        .section {
            margin: 40px 0;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .config-header {
            background-color: #2c3e50;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 1.1em;
        }
    </style>
</head>
<body>
    <h1>Enhanced Model Performance Report</h1>
"""

    def _create_summary_section(self) -> str:
        """Create summary statistics section."""
        if not self.results:
            return ""

        # Calculate summary statistics
        model_types = list(set(r['model_type'] for r in self.results))

        # Find best validation MAE
        best_val_mae = None
        best_model = None
        for result in self.results:
            if result['validation_performance'] and result['validation_performance']['summary']['overall_mae']:
                mae = result['validation_performance']['summary']['overall_mae']
                if best_val_mae is None or mae < best_val_mae:
                    best_val_mae = mae
                    best_model = result['model_type']

        return ""

    def _create_comparison_tables(self) -> str:
        """Create unified comparison table with all performance metrics."""
        if not self.results:
            return ""

        # Collect all benchmark names
        all_benchmarks = set()
        for result in self.results:
            if result["benchmark_performance"]:
                all_benchmarks.update(result["benchmark_performance"].keys())

        # Sort benchmarks to put benchmark_all_max first, then E* series, then others
        sorted_benchmarks = sorted(all_benchmarks, key=lambda x: (
            0 if 'benchmark_all_max' in x else
            1 if x.startswith('E') and x[1:].isdigit() else 2, x
        ))

        html = """
        <div class="section">
            <h2>Comprehensive Model Performance Comparison</h2>
        """

        # Calculate overall best values across all comprehensive tables (for underlining)
        all_comprehensive_results = [r for r in self.results if r["train_performance"]]
        comprehensive_group_best_values = self._calculate_overall_best_values(all_comprehensive_results, sorted_benchmarks)

        # Group results by training configuration (dataset, acquisition, datatype) - epochs/batch size are part of model comparison
        config_groups = {}
        for result in self.results:
            if result["train_performance"]:  # Only require training performance, not validation
                # Create configuration key for training setup (excluding epochs/batch size)
                config_key = f"{result['metabolites']} | {result['datatype']} | {result['edit_type']} | {result['complex_type']} | {result['acquisition']}"
                if config_key not in config_groups:
                    config_groups[config_key] = []
                config_groups[config_key].append(result)

        # Create a table for each configuration
        for config_key, config_results in config_groups.items():
            html += f"""
            <div class="config-header">{config_key}</div>
            <table>
                <thead>
                    <tr>
                        <th>Model Type</th>
                        <th>Epochs</th>
                        <th>Batch Size</th>
                        <th>Train MAE</th>
                        <th>Val MAE</th>
                        <th>Overfitting</th>
            """

            # Add benchmark columns
            for benchmark in sorted_benchmarks:
                # Shorten benchmark name to part before first underscore and remove :MEGA suffix
                short_name = benchmark.split('_')[0].replace(':MEGA', '')
                html += f'                        <th>{short_name} MAE</th>\n'

            html += """
                    </tr>
                </thead>
                <tbody>
            """

            # Calculate best values for this configuration group only (use for highlighting)
            config_results_with_data = [r for r in config_results if r["train_performance"]]
            overall_best_values = self._calculate_overall_best_values(config_results_with_data, sorted_benchmarks)

            # Sort results by validation MAE (best first)
            config_results.sort(
                key=lambda x: x["validation_performance"]["summary"]["overall_mae"] if x["validation_performance"] and x["validation_performance"]["summary"] else float('inf')
            )

            for result in config_results:
                train_perf = result["train_performance"]["summary"] if result["train_performance"] else None
                val_perf = result["validation_performance"]["summary"] if result["validation_performance"] else None

                # Calculate overfitting (positive = overfitting)
                overfitting = None
                if train_perf and val_perf and train_perf.get("overall_mae") and val_perf.get("overall_mae"):
                    overfitting = train_perf["overall_mae"] - val_perf["overall_mae"]

                html += f"""
                    <tr>
                        <td>{result['model_type']}</td>
                        <td>{result['epochs']}</td>
                        <td>{result['batch_size']}</td>
                """

                # Add training/validation metrics with highlighting
                html += self._format_metric_cell(train_perf.get('overall_mae') if train_perf else None, overall_best_values['train_mae'], 'mae', 24,
                                               self._is_best_value(train_perf.get('overall_mae') if train_perf else None, overall_best_values['train_mae']),
                                               self._is_best_value(train_perf.get('overall_mae') if train_perf else None, comprehensive_group_best_values['train_mae']))
                html += self._format_metric_cell(val_perf.get('overall_mae') if val_perf else None, overall_best_values['val_mae'], 'mae', 24,
                                               self._is_best_value(val_perf.get('overall_mae') if val_perf else None, overall_best_values['val_mae']),
                                               self._is_best_value(val_perf.get('overall_mae') if val_perf else None, comprehensive_group_best_values['val_mae']))
                html += self._format_metric_cell(overfitting, overall_best_values['overfitting'], 'overfitting', 24,
                                               self._is_best_value(overfitting, overall_best_values['overfitting']),
                                               self._is_best_value(overfitting, comprehensive_group_best_values['overfitting']))

                # Add benchmark metrics
                for benchmark in sorted_benchmarks:
                    if (result["benchmark_performance"] and
                        benchmark in result["benchmark_performance"]):
                        bench_perf = result["benchmark_performance"][benchmark]
                        bench_summary = bench_perf.get('summary', {})
                        html += self._format_mae_with_std_cell(bench_summary.get('overall_mae'), bench_summary.get('overall_mae_std'),
                                                             overall_best_values[f'{benchmark}_mae'], 24,
                                                             self._is_best_value(bench_summary.get('overall_mae'), overall_best_values[f'{benchmark}_mae']),
                                                             self._is_best_value(bench_summary.get('overall_mae'), comprehensive_group_best_values[f'{benchmark}_mae']))
                    else:
                        html += '                        <td class="metric">N/A</td>\n'

                html += "                    </tr>\n"

            html += """
                </tbody>
            </table>
            """

        html += "</div>"
        return html

    def _create_metabolite_benchmark_tables(self) -> str:
        """Create metabolite-specific benchmark comparison tables."""
        if not self.results:
            return ""

        # Collect all benchmark names
        all_benchmarks = set()
        for result in self.results:
            if result["benchmark_performance"]:
                all_benchmarks.update(result["benchmark_performance"].keys())

        # Sort benchmarks to put E* series first
        sorted_benchmarks = sorted(all_benchmarks, key=lambda x: (
            0 if x.startswith('E') and x[1:].isdigit() else 1, x
        ))

        # Collect all metabolites
        all_metabolites = set()
        for result in self.results:
            if result["train_performance"] and result["train_performance"]["metabolites"]:
                all_metabolites.update(result["train_performance"]["metabolites"].keys())

        if not all_metabolites or not sorted_benchmarks:
            return ""

        html = """
        <div class="section">
            <h2>Metabolite-Specific Benchmark Performance</h2>
        """

        # Create a table for each metabolite
        for metabolite in sorted(all_metabolites):
            html += f"""
            <h3>{metabolite} Performance Across Benchmarks</h3>
            """

            # Calculate overall best values for this metabolite across all tables (for underlining)
            metabolite_group_best_values = self._calculate_metabolite_best_values(sorted_benchmarks, metabolite)

            # Group results by configuration
            config_groups = {}
            for result in self.results:
                if (result["benchmark_performance"] and
                    any(benchmark in result["benchmark_performance"] for benchmark in sorted_benchmarks)):
                    # Check if this metabolite exists in any benchmark
                    has_metabolite = False
                    for benchmark in sorted_benchmarks:
                        if (benchmark in result["benchmark_performance"] and
                            metabolite in result["benchmark_performance"][benchmark]["metabolites"]):
                            has_metabolite = True
                            break
                    if has_metabolite:
                        # Create configuration key for training setup (excluding epochs/batch size)
                        config_key = f"{result['metabolites']} | {result['datatype']} | {result['edit_type']} | {result['complex_type']} | {result['acquisition']}"
                        if config_key not in config_groups:
                            config_groups[config_key] = []
                        config_groups[config_key].append(result)

            # Create a table for each configuration
            for config_key, config_results in config_groups.items():
                html += f"""
                <div class="config-header">{config_key}</div>
                <table>
                    <thead>
                        <tr>
                            <th>Model Type</th>
                            <th>Epochs</th>
                            <th>Batch Size</th>
                """

                # Add benchmark columns for this metabolite
                for benchmark in sorted_benchmarks:
                    # Shorten benchmark name to part before first underscore and remove :MEGA suffix
                    short_name = benchmark.split('_')[0].replace(':MEGA', '')
                    html += f'                            <th>{short_name} MAE</th>\n'
                    html += f'                            <th>{short_name} R²</th>\n'

                html += """
                        </tr>
                    </thead>
                    <tbody>
                """

                # Calculate best values for this metabolite in this configuration group only
                metabolite_best_values = self._calculate_metabolite_best_values_for_config(sorted_benchmarks, metabolite, config_results)

                # Sort by overall validation MAE
                config_results.sort(
                    key=lambda x: x["validation_performance"]["summary"]["overall_mae"] if x["validation_performance"] and x["validation_performance"]["summary"] else float('inf')
                )

                for result in config_results:
                    html += f"""
                        <tr>
                            <td>{result['model_type']}</td>
                            <td>{result['epochs']}</td>
                            <td>{result['batch_size']}</td>
                    """

                    # Add benchmark metrics for this metabolite
                    for benchmark in sorted_benchmarks:
                        if (result["benchmark_performance"] and
                            benchmark in result["benchmark_performance"] and
                            metabolite in result["benchmark_performance"][benchmark]["metabolites"]):

                            metab_data = result["benchmark_performance"][benchmark]["metabolites"][metabolite]

                            # MAE
                            html += self._format_mae_with_std_cell(
                                metab_data["mae"],
                                metab_data.get("mae_std"),
                                metabolite_best_values[f'{benchmark}_mae'],
                                28,
                                self._is_best_value(metab_data["mae"], metabolite_best_values[f'{benchmark}_mae']),
                                self._is_best_value(metab_data["mae"], metabolite_group_best_values[f'{benchmark}_mae'])
                            )

                            # R² (r_value squared)
                            r_value = metab_data.get("correlation")
                            r_squared = r_value ** 2 if r_value is not None else None
                            html += self._format_metric_cell(
                                r_squared,
                                metabolite_best_values[f'{benchmark}_r_squared'],
                                'r_squared',
                                28,
                                self._is_best_value(r_squared, metabolite_best_values[f'{benchmark}_r_squared']),
                                self._is_best_value(r_squared, metabolite_group_best_values[f'{benchmark}_r_squared'])
                            )
                        else:
                            html += '                            <td class="metric">N/A</td>\n'
                            html += '                            <td class="metric">N/A</td>\n'

                    html += "                        </tr>\n"

                html += """
                    </tbody>
                </table>
                """

        html += "</div>"
        return html

    def _calculate_best_values(self, results, benchmarks):
        """Calculate best values for a specific configuration."""
        best_values = {}

        # Training/validation metrics
        train_maes = [r["train_performance"]["summary"]["overall_mae"] for r in results
                      if r["train_performance"] and r["train_performance"]["summary"]["overall_mae"] is not None]
        val_maes = [r["validation_performance"]["summary"]["overall_mae"] for r in results
                    if r["validation_performance"] and r["validation_performance"]["summary"]["overall_mae"] is not None]

        # Overfitting (closest to 0 is best)
        overfittings = []
        for r in results:
            if r["train_performance"] and r["validation_performance"]:
                train_mae = r["train_performance"]["summary"]["overall_mae"]
                val_mae = r["validation_performance"]["summary"]["overall_mae"]
                if train_mae is not None and val_mae is not None:
                    overfitting = train_mae - val_mae
                    overfittings.append(overfitting)

        best_values['train_mae'] = min(train_maes) if train_maes else None
        best_values['val_mae'] = min(val_maes) if val_maes else None
        best_values['overfitting'] = min(overfittings, key=abs) if overfittings else None

        # Benchmark metrics
        for benchmark in benchmarks:
            benchmark_maes = []
            benchmark_corrs = []

            for r in results:
                if (r["benchmark_performance"] and
                    benchmark in r["benchmark_performance"]):
                    bench_perf = r["benchmark_performance"][benchmark]
                    bench_summary = bench_perf.get('summary', {})
                    if bench_summary.get('overall_mae') is not None:
                        benchmark_maes.append(bench_summary.get('overall_mae'))
                    if bench_summary.get('overall_correlation') is not None:
                        benchmark_corrs.append(bench_summary.get('overall_correlation'))

            best_values[f'{benchmark}_mae'] = min(benchmark_maes) if benchmark_maes else None
            best_values[f'{benchmark}_corr'] = max(benchmark_corrs) if benchmark_corrs else None

        return best_values

    def _calculate_overall_best_values(self, results, benchmarks):
        """Calculate overall best values across all configurations, excluding N/A values."""
        overall_best = {}

        # Training/validation metrics - exclude None and NaN values
        train_maes = []
        for r in results:
            if (r["train_performance"] and
                r["train_performance"]["summary"]["overall_mae"] is not None and
                not (isinstance(r["train_performance"]["summary"]["overall_mae"], float) and
                     np.isnan(r["train_performance"]["summary"]["overall_mae"]))):
                train_maes.append(r["train_performance"]["summary"]["overall_mae"])

        val_maes = []
        for r in results:
            if (r["validation_performance"] and
                r["validation_performance"]["summary"]["overall_mae"] is not None and
                not (isinstance(r["validation_performance"]["summary"]["overall_mae"], float) and
                     np.isnan(r["validation_performance"]["summary"]["overall_mae"]))):
                val_maes.append(r["validation_performance"]["summary"]["overall_mae"])

        # Overfitting (closest to 0 is best) - exclude None and NaN values
        overfittings = []
        for r in results:
            if r["train_performance"] and r["validation_performance"]:
                train_mae = r["train_performance"]["summary"]["overall_mae"]
                val_mae = r["validation_performance"]["summary"]["overall_mae"]
                if (train_mae is not None and val_mae is not None and
                    not (isinstance(train_mae, float) and np.isnan(train_mae)) and
                    not (isinstance(val_mae, float) and np.isnan(val_mae))):
                    overfitting = train_mae - val_mae
                    overfittings.append(overfitting)

        overall_best['train_mae'] = min(train_maes) if train_maes else None
        overall_best['val_mae'] = min(val_maes) if val_maes else None
        overall_best['overfitting'] = min(overfittings, key=abs) if overfittings else None

        # Benchmark metrics - exclude None and NaN values
        for benchmark in benchmarks:
            benchmark_maes = []

            for r in results:
                if (r["benchmark_performance"] and
                    benchmark in r["benchmark_performance"]):
                    bench_perf = r["benchmark_performance"][benchmark]
                    bench_summary = bench_perf.get('summary', {})
                    mae_value = bench_summary.get('overall_mae')
                    if (mae_value is not None and
                        not (isinstance(mae_value, float) and np.isnan(mae_value))):
                        benchmark_maes.append(mae_value)

            overall_best[f'{benchmark}_mae'] = min(benchmark_maes) if benchmark_maes else None

        return overall_best

    def _calculate_metabolite_best_values(self, benchmarks, metabolite):
        """Calculate best values for a specific metabolite across benchmarks, excluding N/A values."""
        best_values = {}

        for benchmark in benchmarks:
            benchmark_maes = []
            benchmark_r_squareds = []

            for result in self.results:
                if (result["benchmark_performance"] and
                    benchmark in result["benchmark_performance"] and
                    metabolite in result["benchmark_performance"][benchmark]["metabolites"]):

                    metab_data = result["benchmark_performance"][benchmark]["metabolites"][metabolite]

                    # MAE - exclude None and NaN values
                    mae_value = metab_data["mae"]
                    if (mae_value is not None and
                        not (isinstance(mae_value, float) and np.isnan(mae_value))):
                        benchmark_maes.append(mae_value)

                    # R² - exclude None and NaN values
                    r_value = metab_data["correlation"]
                    if (r_value is not None and
                        not (isinstance(r_value, float) and np.isnan(r_value))):
                        r_squared = r_value ** 2
                        benchmark_r_squareds.append(r_squared)

            best_values[f'{benchmark}_mae'] = min(benchmark_maes) if benchmark_maes else None
            best_values[f'{benchmark}_r_squared'] = max(benchmark_r_squareds) if benchmark_r_squareds else None

        return best_values

    def _calculate_metabolite_best_values_for_config(self, benchmarks, metabolite, config_results):
        """Calculate best values for a specific metabolite across benchmarks for a specific configuration group."""
        best_values = {}

        for benchmark in benchmarks:
            benchmark_maes = []
            benchmark_r_squareds = []

            for result in config_results:
                if (result["benchmark_performance"] and
                    benchmark in result["benchmark_performance"] and
                    metabolite in result["benchmark_performance"][benchmark]["metabolites"]):

                    metab_data = result["benchmark_performance"][benchmark]["metabolites"][metabolite]

                    # MAE - exclude None and NaN values
                    mae_value = metab_data["mae"]
                    if (mae_value is not None and
                        not (isinstance(mae_value, float) and np.isnan(mae_value))):
                        benchmark_maes.append(mae_value)

                    # R² - exclude None and NaN values
                    r_value = metab_data["correlation"]
                    if (r_value is not None and
                        not (isinstance(r_value, float) and np.isnan(r_value))):
                        r_squared = r_value ** 2
                        benchmark_r_squareds.append(r_squared)

            best_values[f'{benchmark}_mae'] = min(benchmark_maes) if benchmark_maes else None
            best_values[f'{benchmark}_r_squared'] = max(benchmark_r_squareds) if benchmark_r_squareds else None

        return best_values

    def _is_best_value(self, value, best_value, tolerance=1e-10):
        """Check if a value is the best value, handling floating point precision."""
        if value is None or best_value is None:
            return False
        if isinstance(value, float) and np.isnan(value):
            return False
        if isinstance(best_value, float) and np.isnan(best_value):
            return False

        # Use tolerance for floating point comparison
        return abs(value - best_value) < tolerance

    def _format_metric_cell(self, value, best_value, metric_type, indent_level=28, is_best=False, is_overall_best=False):
        """Format a metric cell with highlighting for best values, excluding N/A values."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return f'{" " * indent_level}<td class="metric">N/A</td>\n'

        # Format the value
        if metric_type == 'mae' or metric_type == 'overfitting':
            formatted_value = f"{value:.6f}"
        elif metric_type == 'slope':
            formatted_value = f"{value:.4f}"
        elif metric_type == 'r_squared':
            formatted_value = f"{value:.4f}"
        else:  # correlation
            formatted_value = f"{value:.4f}"

        # Add highlighting class
        if is_overall_best:
            cell_class = "metric overall-best"
        elif is_best:
            cell_class = "metric best"
        else:
            cell_class = "metric"

        return f'{" " * indent_level}<td class="{cell_class}">{formatted_value}</td>\n'

    def _format_mae_with_std_cell(self, mae_value, std_value, best_value, indent_level=24, is_best=False, is_overall_best=False):
        """Format a MAE cell with standard deviation combined, excluding N/A values from highlighting."""
        if mae_value is None or (isinstance(mae_value, float) and np.isnan(mae_value)):
            return f'{" " * indent_level}<td class="metric">N/A</td>\n'

        # Format the MAE value
        formatted_mae = f"{mae_value:.6f}"

        # Add standard deviation if available
        if std_value is not None and not (isinstance(std_value, float) and np.isnan(std_value)):
            formatted_std = f"{std_value:.6f}"
            formatted_value = f"{formatted_mae} ± {formatted_std}"
        else:
            formatted_value = formatted_mae

        # Add highlighting class
        if is_overall_best:
            cell_class = "metric overall-best"
        elif is_best:
            cell_class = "metric best"
        else:
            cell_class = "metric"

        return f'{" " * indent_level}<td class="{cell_class}">{formatted_value}</td>\n'


def main():
    """Main function to run the enhanced performance collection script."""
    parser = argparse.ArgumentParser(
        description="Collect and report model performance data with enhanced features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --output report.html
  %(prog)s --model-dirs data/model --output report.html
  %(prog)s --model-dirs data/model data/model-ae --output report.html
        """
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output file path for the performance report'
    )

    parser.add_argument(
        '--model-dirs',
        nargs='+',
        help='Model directory paths (can specify multiple). Default: data/model'
    )

    args = parser.parse_args()

    # Initialize collector
    collector = EnhancedModelPerformanceCollector(args.model_dirs)

    # Collect performance data
    results = collector.collect_all_performance_data()

    if not results:
        print("No performance data found!")
        return

    # Generate report
    print(f"Generating enhanced performance report...")

    report_content = collector.generate_html_report()

    # Write report to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(report_content)

    print(f"Enhanced performance report generated: {output_path}")

    # Print summary
    model_types = list(set(r['model_type'] for r in results))
    best_val_mae = None
    best_model = None
    for result in results:
        if result['validation_performance'] and result['validation_performance']['summary']['overall_mae']:
            mae = result['validation_performance']['summary']['overall_mae']
            if best_val_mae is None or mae < best_val_mae:
                best_val_mae = mae
                best_model = result['model_type']

    print(f"\nSummary:")
    print(f"  Total models: {len(results)}")
    print(f"  Model types: {', '.join(model_types)}")
    print(f"  Best validation MAE: {best_val_mae:.6f} ({best_model})")


if __name__ == "__main__":
    main()
