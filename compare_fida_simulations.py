#!/usr/bin/env python3

"""
FID-A Simulation Comparison Script.

This script compares different FID-A simulation runs to identify sources of differences.
Run this script to generate a comprehensive comparison report.

Usage:
    python3 compare_fida_simulations.py

The script will:
1. Run simulations with cached=true and cached=false
2. Compare results with original files
3. Test reproducibility across multiple runs
4. Generate a detailed comparison report
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as sio


class FIDASimulatorComparison:
    def __init__(self, base_dir="/srv/data/Prj/MRS/code-mrsnet"):
        self.base_dir = Path(base_dir)
        self.test_dir = self.base_dir / "local" / "fida-comparison-test"
        self.basis_dist_dir = self.base_dir / "data" / "basis-dist"
        self.basis_dir = self.base_dir / "data" / "basis"
        self.results = {}

    def setup_directories(self):
        """Create test directories."""
        self.test_dir.mkdir(parents=True, exist_ok=True)
        (self.test_dir / "cached").mkdir(exist_ok=True)
        (self.test_dir / "non_cached").mkdir(exist_ok=True)
        (self.test_dir / "reproducibility").mkdir(exist_ok=True)

    def check_existing_results(self, metab, omega, use_cached):
        """Check if simulation results already exist."""
        output_dir = self.test_dir / ("cached" if use_cached else "non_cached")
        expected_file = output_dir / f"FIDA2D_{metab}_MEGAPRESS_EDITOFF_2.00_2000_4096_{omega:.2f}.mat"
        return expected_file.exists()

    def find_existing_basis_files(self, metab, omega):
        """Find existing basis files in basis-dist and basis directories."""
        pattern = f"FIDA2D_{metab}_MEGAPRESS_EDITOFF_2.00_2000_4096_{omega:.2f}.mat"

        existing_files = []

        # Search in basis-dist
        if self.basis_dist_dir.exists():
            for root in self.basis_dist_dir.rglob(pattern):
                existing_files.append({
                    'path': root,
                    'source': 'basis-dist',
                    'relative_path': root.relative_to(self.basis_dist_dir)
                })

        # Search in basis
        if self.basis_dir.exists():
            for root in self.basis_dir.rglob(pattern):
                existing_files.append({
                    'path': root,
                    'source': 'basis',
                    'relative_path': root.relative_to(self.basis_dir)
                })

        return existing_files

    def get_simulation_info(self, metab, omega, use_cached):
        """Get information about a simulation run."""
        output_dir = self.test_dir / ("cached" if use_cached else "non_cached")
        expected_file = output_dir / f"FIDA2D_{metab}_MEGAPRESS_EDITOFF_2.00_2000_4096_{omega:.2f}.mat"

        if expected_file.exists():
            stat = expected_file.stat()
            return {
                'exists': True,
                'file': expected_file,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
        else:
            return {
                'exists': False,
                'file': expected_file,
                'size': 0,
                'modified': None
            }

    def run_matlab_simulation(self, metab, omega, use_cached, output_dir, force_rerun=False):
        """Run MATLAB simulation with specified parameters."""
        # Check if results already exist
        if not force_rerun and self.check_existing_results(metab, omega, use_cached):
            print(f"Skipping simulation: {metab}, ω={omega}, cached={use_cached} (results already exist)")
            sim_info = self.get_simulation_info(metab, omega, use_cached)
            return {
                'success': True,
                'duration': 0,
                'returncode': 0,
                'stdout': 'Skipped - using existing results',
                'stderr': '',
                'skipped': True,
                'file_info': sim_info
            }

        print(f"Running simulation: {metab}, ω={omega}, cached={use_cached}")

        # Build MATLAB command as a single line
        matlab_cmd = (f"addpath('{self.base_dir}/mrsnet/simulators/fida'); "
                     f"addpath(genpath('{self.base_dir}/mrsnet/simulators/fida/FID-A')); "
                     f"save_dir='{output_dir}'; "
                     f"cache_dir='{self.base_dir}/local/fida2d-cache'; "
                     f"use_cached_unbroadened={str(use_cached).lower()}; "
                     f"metabolites={{'{metab}'}}; "
                     f"npts=4096; "
                     f"sw=2000; "
                     f"linewidths=[2.0]; "
                     f"mrsnet_omega={omega}; "
                     f"run('{self.base_dir}/mrsnet/simulators/fida/run_custom_simMegaPress_2D.m'); "
                     f"exit")

        print(f"MATLAB command: {matlab_cmd}")

        start_time = time.time()
        try:
            print(f"Executing MATLAB command...")
            result = subprocess.run(
                ['matlab', '-nosplash', '-nodisplay', '-r', matlab_cmd],
                text=True
            )

            duration = time.time() - start_time
            success = result.returncode == 0

            sim_info = self.get_simulation_info(metab, omega, use_cached)
            return {
                'success': success,
                'duration': duration,
                'returncode': result.returncode,
                'stdout': '',
                'stderr': '',
                'skipped': False,
                'file_info': sim_info
            }

        except Exception as e:
            return {
                'success': False,
                'duration': 0,
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'skipped': False,
                'file_info': {'exists': False, 'file': None, 'size': 0, 'modified': None}
            }

    def compare_mat_files(self, file1, file2, label):
        """Compare two .mat files and return detailed comparison."""
        try:
            A = sio.loadmat(file1, squeeze_me=True)
            B = sio.loadmat(file2, squeeze_me=True)

            def arr(x):
                return np.asarray(x).reshape(-1)

            # Compare FID
            fid_a = arr(A['fid'])
            fid_b = arr(B['fid'])
            fid_diff = np.abs(fid_a - fid_b)
            fid_max_diff = np.max(fid_diff)
            fid_mean_diff = np.mean(fid_diff)
            fid_rel_diff = fid_max_diff / np.max(np.abs(fid_a)) * 100

            # Compare FFT
            fft_a = arr(A['fft'])
            fft_b = arr(B['fft'])
            fft_diff = np.abs(fft_a - fft_b)
            fft_max_diff = np.max(fft_diff)
            fft_mean_diff = np.mean(fft_diff)
            fft_rel_diff = fft_max_diff / np.max(np.abs(fft_a)) * 100

            # Check if files are essentially identical
            identical = fid_max_diff < 1e-15 and fft_max_diff < 1e-15
            very_close = fid_max_diff < 1e-10 and fft_max_diff < 1e-10
            close = fid_max_diff < 1e-6 and fft_max_diff < 1e-6

            return {
                'label': label,
                'file1': str(file1),
                'file2': str(file2),
                'fid_shape': fid_a.shape,
                'fft_shape': fft_a.shape,
                'fid_max_diff': fid_max_diff,
                'fid_mean_diff': fid_mean_diff,
                'fid_rel_diff': fid_rel_diff,
                'fft_max_diff': fft_max_diff,
                'fft_mean_diff': fft_mean_diff,
                'fft_rel_diff': fft_rel_diff,
                'identical': identical,
                'very_close': very_close,
                'close': close,
                'success': True
            }

        except Exception as e:
            return {
                'label': label,
                'file1': str(file1),
                'file2': str(file2),
                'error': str(e),
                'success': False
            }

    def run_comparison_tests(self, metab='GABA', omega=123.23, force_rerun=False):
        """Run all comparison tests."""
        print("FID-A Simulation Comparison Test")
        print("=" * 50)
        print(f"Metabolite: {metab}")
        print(f"Omega: {omega}")
        print(f"Test directory: {self.test_dir}")
        print(f"Force rerun: {force_rerun}")
        print()

        self.setup_directories()

        # Find existing basis files
        print("Searching for existing basis files...")
        existing_files = self.find_existing_basis_files(metab, omega)
        print(f"Found {len(existing_files)} existing basis files:")
        for f in existing_files:
            print(f"  {f['source']}: {f['relative_path']}")
        print()

        # Test 1: Cached vs Non-cached
        print("Test 1: Cached vs Non-cached simulation")
        print("-" * 40)

        cached_dir = self.test_dir / "cached"
        non_cached_dir = self.test_dir / "non_cached"

        print("Running cached simulation...")
        cached_result = self.run_matlab_simulation(metab, omega, True, cached_dir, force_rerun)
        print(f"Cached simulation: {'SUCCESS' if cached_result['success'] else 'FAILED'}")
        if cached_result['success']:
            if cached_result.get('skipped', False):
                print(f"  Skipped (using existing results)")
                print(f"  File size: {cached_result['file_info']['size']} bytes")
                print(f"  Modified: {cached_result['file_info']['modified']}")
            else:
                print(f"  Duration: {cached_result['duration']:.1f} seconds")
        else:
            print(f"  Error: {cached_result['stderr'][:200]}...")

        print("Running non-cached simulation...")
        non_cached_result = self.run_matlab_simulation(metab, omega, False, non_cached_dir, force_rerun)
        print(f"Non-cached simulation: {'SUCCESS' if non_cached_result['success'] else 'FAILED'}")
        if non_cached_result['success']:
            if non_cached_result.get('skipped', False):
                print(f"  Skipped (using existing results)")
                print(f"  File size: {non_cached_result['file_info']['size']} bytes")
                print(f"  Modified: {non_cached_result['file_info']['modified']}")
            else:
                print(f"  Duration: {non_cached_result['duration']:.1f} seconds")
        else:
            print(f"  Error: {non_cached_result['stderr'][:200]}...")

        # Test 2: Reproducibility test (only if force_rerun or no existing results)
        print("\nTest 2: Reproducibility test (3 runs)")
        print("-" * 40)

        repro_results = []
        if force_rerun or not self.check_existing_results(metab, omega, True):
            for i in range(3):
                repro_dir = self.test_dir / "reproducibility" / f"run_{i}"
                repro_dir.mkdir(parents=True, exist_ok=True)

                print(f"Running reproducibility test {i+1}/3...")
                result = self.run_matlab_simulation(metab, omega, True, repro_dir, force_rerun)
                repro_results.append(result)
                print(f"  Run {i+1}: {'SUCCESS' if result['success'] else 'FAILED'}")
        else:
            print("Skipping reproducibility test (using existing results)")

        # Test 3: Compare with existing basis files
        print("\nTest 3: Compare with existing basis files")
        print("-" * 40)

        comparisons = []

        # Compare cached vs non-cached
        if cached_result['success'] and non_cached_result['success']:
            cached_file = cached_dir / f"FIDA2D_{metab}_MEGAPRESS_EDITOFF_2.00_2000_4096_{omega:.2f}.mat"
            non_cached_file = non_cached_dir / f"FIDA2D_{metab}_MEGAPRESS_EDITOFF_2.00_2000_4096_{omega:.2f}.mat"

            if cached_file.exists() and non_cached_file.exists():
                comparison = self.compare_mat_files(cached_file, non_cached_file, "Cached vs Non-cached")
                comparisons.append(comparison)

        # Compare with all existing basis files
        for existing_file in existing_files:
            if cached_result['success']:
                cached_file = cached_dir / f"FIDA2D_{metab}_MEGAPRESS_EDITOFF_2.00_2000_4096_{omega:.2f}.mat"
                if cached_file.exists():
                    label = f"Existing ({existing_file['source']}) vs New Cached"
                    comparison = self.compare_mat_files(existing_file['path'], cached_file, label)
                    comparisons.append(comparison)

            if non_cached_result['success']:
                non_cached_file = non_cached_dir / f"FIDA2D_{metab}_MEGAPRESS_EDITOFF_2.00_2000_4096_{omega:.2f}.mat"
                if non_cached_file.exists():
                    label = f"Existing ({existing_file['source']}) vs New Non-cached"
                    comparison = self.compare_mat_files(existing_file['path'], non_cached_file, label)
                    comparisons.append(comparison)

        # Reproducibility comparisons
        repro_comparisons = []
        if len(repro_results) >= 2 and all(r['success'] for r in repro_results[:2]):
            file1 = self.test_dir / "reproducibility" / "run_0" / f"FIDA2D_{metab}_MEGAPRESS_EDITOFF_2.00_2000_4096_{omega:.2f}.mat"
            file2 = self.test_dir / "reproducibility" / "run_1" / f"FIDA2D_{metab}_MEGAPRESS_EDITOFF_2.00_2000_4096_{omega:.2f}.mat"

            if file1.exists() and file2.exists():
                comparison = self.compare_mat_files(file1, file2, "Reproducibility Test")
                repro_comparisons.append(comparison)

        # Store results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'metabolite': metab,
                'omega': omega,
                'test_dir': str(self.test_dir),
                'force_rerun': force_rerun
            },
            'existing_files': existing_files,
            'simulations': {
                'cached': cached_result,
                'non_cached': non_cached_result,
                'reproducibility': repro_results
            },
            'comparisons': comparisons,
            'reproducibility_comparisons': repro_comparisons
        }

        return self.results

    def print_summary(self):
        """Print a summary of the results."""
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)

        # Existing files summary
        print(f"\nExisting Basis Files Found: {len(self.results['existing_files'])}")
        for f in self.results['existing_files']:
            print(f"  {f['source']}: {f['relative_path']}")

        # Simulation results
        print("\nSimulation Results:")
        cached_success = self.results['simulations']['cached']['success']
        non_cached_success = self.results['simulations']['non_cached']['success']

        cached_skipped = self.results['simulations']['cached'].get('skipped', False)
        non_cached_skipped = self.results['simulations']['non_cached'].get('skipped', False)

        print(f"  Cached simulation: {'✓ SUCCESS' if cached_success else '✗ FAILED'}")
        if cached_success:
            if cached_skipped:
                print(f"    Skipped (using existing results)")
                print(f"    File size: {self.results['simulations']['cached']['file_info']['size']} bytes")
            else:
                print(f"    Duration: {self.results['simulations']['cached']['duration']:.1f}s")

        print(f"  Non-cached simulation: {'✓ SUCCESS' if non_cached_success else '✗ FAILED'}")
        if non_cached_success:
            if non_cached_skipped:
                print(f"    Skipped (using existing results)")
                print(f"    File size: {self.results['simulations']['non_cached']['file_info']['size']} bytes")
            else:
                print(f"    Duration: {self.results['simulations']['non_cached']['duration']:.1f}s")

        # Comparison results
        print("\nComparison Results:")
        for comp in self.results['comparisons']:
            if comp['success']:
                print(f"\n  {comp['label']}:")
                print(f"    FID max diff: {comp['fid_max_diff']:.2e} ({comp['fid_rel_diff']:.6f}%)")
                print(f"    FFT max diff: {comp['fft_max_diff']:.2e} ({comp['fft_rel_diff']:.6f}%)")

                if comp['identical']:
                    print("    Status: ✓ IDENTICAL")
                elif comp['very_close']:
                    print("    Status: ✓ VERY CLOSE (numerical precision)")
                elif comp['close']:
                    print("    Status: ⚠ CLOSE (small differences)")
                else:
                    print("    Status: ⚠ SIGNIFICANT DIFFERENCES")
            else:
                print(f"\n  {comp['label']}: ✗ ERROR - {comp['error']}")

        # Reproducibility results
        if self.results['reproducibility_comparisons']:
            print("\nReproducibility Results:")
            for comp in self.results['reproducibility_comparisons']:
                if comp['success']:
                    print(f"  {comp['label']}:")
                    print(f"    FID max diff: {comp['fid_max_diff']:.2e}")
                    print(f"    FFT max diff: {comp['fft_max_diff']:.2e}")

                    if comp['identical']:
                        print("    Status: ✓ PERFECTLY REPRODUCIBLE")
                    elif comp['very_close']:
                        print("    Status: ✓ HIGHLY REPRODUCIBLE")
                    else:
                        print("    Status: ⚠ NOT REPRODUCIBLE")

        # Conclusions
        print("\nConclusions:")
        if not cached_success or not non_cached_success:
            print("  ⚠ Some simulations failed - check MATLAB errors")
        else:
            # Check if cache is causing differences
            cache_comparison = next((c for c in self.results['comparisons'] if 'Cached vs Non-cached' in c['label']), None)
            if cache_comparison and cache_comparison['success']:
                if cache_comparison['identical']:
                    print("  ✓ Cache mechanism works correctly - no differences")
                else:
                    print("  ⚠ Cache mechanism may be causing differences")
                    print(f"    FID difference: {cache_comparison['fid_max_diff']:.2e}")
                    print(f"    FFT difference: {cache_comparison['fft_max_diff']:.2e}")

    def convert_to_json_serializable(self, obj):
        """Convert numpy types and other non-serializable objects to JSON-serializable types."""
        if isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj

    def save_results(self, filename=None):
        """Save detailed results to JSON file."""
        if filename is None:
            filename = self.test_dir / "comparison_results.json"

        # Convert numpy types to JSON-serializable types
        serializable_results = self.convert_to_json_serializable(self.results)

        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\nDetailed results saved to: {filename}")

def main():
    """Main function to run the comparison."""
    import argparse

    parser = argparse.ArgumentParser(description='Compare FID-A simulations')
    parser.add_argument('--metab', default='GABA', help='Metabolite to test (default: GABA)')
    parser.add_argument('--omega', type=float, default=123.23, help='Omega value (default: 123.23)')
    parser.add_argument('--force-rerun', action='store_true', help='Force rerun all simulations')
    parser.add_argument('--base-dir', default='/srv/data/Prj/MRS/code-mrsnet', help='Base directory path')

    args = parser.parse_args()

    comparator = FIDASimulatorComparison(args.base_dir)

    try:
        results = comparator.run_comparison_tests(
            metab=args.metab,
            omega=args.omega,
            force_rerun=args.force_rerun
        )
        comparator.print_summary()
        comparator.save_results()

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nError during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
