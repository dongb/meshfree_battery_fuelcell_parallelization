#!/usr/bin/env python3
"""
Script to compare numeric values in text files between res.std and res.out directories.
"""

import os
from common import np
from pathlib import Path
import sys


def read_numeric_file(filepath):
    """
    Read numeric values from a text file.
    Handles both single column and multi-column data.
    
    Returns:
        numpy array of numeric values
    """
    try:
        # Try to load as a simple array first
        data = np.loadtxt(filepath)
        return data
    except Exception as e:
        print(f"  Warning: Could not read {filepath}: {e}")
        return None


def compare_arrays(arr1, arr2, filename, tolerance=1e-10):
    """
    Compare two numpy arrays and return comparison statistics.
    
    Args:
        arr1: Reference array (from res.std)
        arr2: Test array (from res.out)
        filename: Name of the file being compared
        tolerance: Relative tolerance for comparison
        
    Returns:
        dict: Comparison statistics
    """
    stats = {
        'filename': filename,
        'match': False,
        'shape_match': False,
        'size_std': None,
        'size_out': None,
        'max_abs_diff': None,
        'max_rel_diff': None,
        'mean_abs_diff': None,
        'mean_rel_diff': None,
        'num_mismatches': 0,
        'within_tolerance': False
    }
    
    if arr1 is None or arr2 is None:
        return stats
    
    # Flatten arrays for comparison
    arr1_flat = arr1.flatten()
    arr2_flat = arr2.flatten()
    
    stats['size_std'] = arr1_flat.size
    stats['size_out'] = arr2_flat.size
    
    # Check if shapes match
    if arr1.shape != arr2.shape:
        stats['shape_match'] = False
        return stats
    
    stats['shape_match'] = True
    
    # Calculate absolute differences
    abs_diff = np.abs(arr1_flat - arr2_flat)
    stats['max_abs_diff'] = np.max(abs_diff)
    stats['mean_abs_diff'] = np.mean(abs_diff)
    
    # Calculate relative differences (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = np.abs((arr1_flat - arr2_flat) / np.where(arr1_flat != 0, arr1_flat, 1e-20))
        # Filter out inf and nan values
        rel_diff_valid = rel_diff[np.isfinite(rel_diff)]
        
        if len(rel_diff_valid) > 0:
            stats['max_rel_diff'] = np.max(rel_diff_valid)
            stats['mean_rel_diff'] = np.mean(rel_diff_valid)
        else:
            stats['max_rel_diff'] = 0.0
            stats['mean_rel_diff'] = 0.0
    
    # Check if arrays are close within tolerance
    stats['within_tolerance'] = np.allclose(arr1_flat, arr2_flat, rtol=tolerance, atol=tolerance)
    stats['match'] = np.array_equal(arr1_flat, arr2_flat)
    
    # Count mismatches
    stats['num_mismatches'] = np.sum(~np.isclose(arr1_flat, arr2_flat, rtol=tolerance, atol=tolerance))
    
    return stats


def compare_directories(std_dir, out_dir, tolerance=1e-10, verbose=True):
    """
    Compare all matching text files between two directories.
    
    Args:
        std_dir: Path to reference directory (res.std)
        out_dir: Path to output directory (res.out)
        tolerance: Relative tolerance for comparison
        verbose: Print detailed information
        
    Returns:
        list: List of comparison statistics for each file
    """
    std_path = Path(std_dir)
    out_path = Path(out_dir)
    
    if not std_path.exists():
        print(f"Error: Directory {std_dir} does not exist!")
        return []
    
    if not out_path.exists():
        print(f"Error: Directory {out_dir} does not exist!")
        return []
    
    # Get all txt files in std directory
    std_files = {f.name: f for f in std_path.glob("*.txt")}
    out_files = {f.name: f for f in out_path.glob("*.txt")}
    
    # Find common files
    common_files = set(std_files.keys()) & set(out_files.keys())
    only_in_std = set(std_files.keys()) - set(out_files.keys())
    only_in_out = set(out_files.keys()) - set(std_files.keys())
    
    print(f"\n{'='*80}")
    print(f"DIRECTORY COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"Standard directory (res.std): {std_path}")
    print(f"Output directory (res.out):   {out_path}")
    print(f"\nFiles in res.std: {len(std_files)}")
    print(f"Files in res.out: {len(out_files)}")
    print(f"Common files:     {len(common_files)}")
    print(f"Only in res.std:  {len(only_in_std)}")
    print(f"Only in res.out:  {len(only_in_out)}")
    
    if only_in_std and verbose:
        print(f"\nFiles only in res.std:")
        for fname in sorted(only_in_std):
            print(f"  - {fname}")
    
    if only_in_out and verbose:
        print(f"\nFiles only in res.out:")
        for fname in sorted(only_in_out):
            print(f"  - {fname}")
    
    # Compare common files
    print(f"\n{'='*80}")
    print(f"COMPARING {len(common_files)} COMMON FILES")
    print(f"{'='*80}")
    print(f"Tolerance: {tolerance}\n")
    
    results = []
    
    for filename in sorted(common_files):
        if verbose:
            print(f"\nComparing: {filename}")
        
        std_file = std_files[filename]
        out_file = out_files[filename]
        
        # Read files
        std_data = read_numeric_file(std_file)
        out_data = read_numeric_file(out_file)
        
        # Compare
        stats = compare_arrays(std_data, out_data, filename, tolerance)
        results.append(stats)
        
        if verbose:
            if not stats['shape_match']:
                print(f"  ❌ SHAPE MISMATCH!")
                print(f"     res.std size: {stats['size_std']}")
                print(f"     res.out size: {stats['size_out']}")
            elif stats['match']:
                print(f"  ✓ EXACT MATCH")
            elif stats['within_tolerance']:
                print(f"  ✓ MATCH (within tolerance)")
                print(f"     Max abs diff:  {stats['max_abs_diff']:.3e}")
                print(f"     Max rel diff:  {stats['max_rel_diff']:.3e}")
                print(f"     Mean abs diff: {stats['mean_abs_diff']:.3e}")
            else:
                print(f"  ❌ MISMATCH (exceeds tolerance)")
                print(f"     Max abs diff:  {stats['max_abs_diff']:.3e}")
                print(f"     Max rel diff:  {stats['max_rel_diff']:.3e}")
                print(f"     Mean abs diff: {stats['mean_abs_diff']:.3e}")
                print(f"     Mismatches:    {stats['num_mismatches']}/{stats['size_std']}")
    
    return results


def print_summary(results):
    """Print summary statistics of all comparisons."""
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*80}")
    
    total = len(results)
    exact_matches = sum(1 for r in results if r['match'])
    tolerance_matches = sum(1 for r in results if r['within_tolerance'] and not r['match'])
    mismatches = sum(1 for r in results if not r['within_tolerance'] and r['shape_match'])
    shape_mismatches = sum(1 for r in results if not r['shape_match'])
    
    print(f"\nTotal files compared: {total}")
    print(f"Exact matches:        {exact_matches} ({100*exact_matches/total:.1f}%)")
    print(f"Within tolerance:     {tolerance_matches} ({100*tolerance_matches/total:.1f}%)")
    print(f"Exceeds tolerance:    {mismatches} ({100*mismatches/total:.1f}%)")
    print(f"Shape mismatches:     {shape_mismatches} ({100*shape_mismatches/total:.1f}%)")
    
    # Files that don't match
    failed_files = [r for r in results if not r['within_tolerance']]
    if failed_files:
        print(f"\n{'='*80}")
        print(f"FILES WITH DIFFERENCES EXCEEDING TOLERANCE:")
        print(f"{'='*80}")
        for r in failed_files:
            print(f"\n{r['filename']}:")
            if not r['shape_match']:
                print(f"  Shape mismatch: std={r['size_std']}, out={r['size_out']}")
            else:
                print(f"  Max abs diff:  {r['max_abs_diff']:.6e}")
                print(f"  Max rel diff:  {r['max_rel_diff']:.6e}")
                print(f"  Mean abs diff: {r['mean_abs_diff']:.6e}")
                print(f"  Mismatches:    {r['num_mismatches']}/{r['size_std']}")
    
    return exact_matches + tolerance_matches == total


if __name__ == "__main__":
    # Default directories
    workspace = "/home/bod/workspace/meshfree_battery_fuelcell_parallelization"
    std_dir = os.path.join(workspace, "res.std")
    out_dir = os.path.join(workspace, "res.out")
    
    # Default tolerance
    tolerance = 1e-10
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        std_dir = sys.argv[1]
    if len(sys.argv) > 2:
        out_dir = sys.argv[2]
    if len(sys.argv) > 3:
        tolerance = float(sys.argv[3])
    
    # Run comparison
    results = compare_directories(std_dir, out_dir, tolerance=tolerance, verbose=True)
    
    # Print summary
    all_pass = print_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if all_pass else 1)

