#!/usr/bin/env python3
"""
Test script to validate the vectorization of define_buttler_volmer.py
Compares the vectorized version against the original implementation
"""

import numpy as np
import time
import sys
from pathlib import Path

# Import the vectorized version (fuel_cell_3D)
sys.path.insert(0, str(Path(__file__).parent))
import define_buttler_volmer as vectorized_version

# Import the original version (IM_RKPM_with_fuel_cell_3D_image_mechanical_inlcude_pores)
sys.path.insert(0, str(Path(__file__).parent.parent / "IM_RKPM_with_fuel_cell_3D_image_mechanical_inlcude_pores"))
import define_buttler_volmer as original_version


def test_Dn_complex():
    """Test Dn_complex function"""
    print("\n" + "="*60)
    print("Testing Dn_complex")
    print("="*60)
    
    # Test cases with different array sizes and value ranges
    test_cases = [
        # (x, D_damage, description)
        (np.array([0.1, 0.3, 0.5, 0.7]), np.array([0.1, 0.2, 0.3, 0.4]), "Small array"),
        (np.linspace(-0.1, 1.0, 100), np.linspace(0.0, 0.8, 100), "Medium array"),
        (np.linspace(0.0, 1.0, 1000), np.linspace(0.0, 1.2, 1000), "Large array with clamping"),
        (np.array([0.0, 0.1, 0.4, 0.6, 0.8]), np.array([0.95, 0.95, 0.95, 0.95, 0.95]), "All clamped"),
    ]
    
    for x, D_damage_orig, description in test_cases:
        print(f"\n  Test: {description}")
        print(f"    x shape: {x.shape}, D_damage shape: {D_damage_orig.shape}")
        
        # Need to copy D_damage because original modifies it in-place
        D_damage_vec = D_damage_orig.copy()
        D_damage_orig_copy = D_damage_orig.copy()
        
        # Run original version
        t0 = time.time()
        D_orig, dD_dx_orig = original_version.Dn_complex(x, D_damage_orig_copy)
        t_orig = time.time() - t0
        
        # Run vectorized version
        t0 = time.time()
        D_vec, dD_dx_vec = vectorized_version.Dn_complex(x, D_damage_vec)
        t_vec = time.time() - t0
        
        # Compare results
        D_match = np.allclose(D_orig, D_vec, rtol=1e-10, atol=1e-12)
        dD_dx_match = np.allclose(dD_dx_orig, dD_dx_vec, rtol=1e-10, atol=1e-12)
        
        if D_match and dD_dx_match:
            speedup = t_orig / t_vec if t_vec > 0 else float('inf')
            print(f"    ✓ Results match!")
            print(f"    Original time: {t_orig*1e6:.2f} µs")
            print(f"    Vectorized time: {t_vec*1e6:.2f} µs")
            print(f"    Speedup: {speedup:.2f}x")
        else:
            print(f"    ✗ Results DO NOT match!")
            if not D_match:
                max_diff = np.max(np.abs(D_orig - D_vec))
                print(f"    Max difference in D: {max_diff}")
            if not dD_dx_match:
                max_diff = np.max(np.abs(dD_dx_orig - dD_dx_vec))
                print(f"    Max difference in dD_dx: {max_diff}")
            return False
    
    return True


def test_ocp_complex():
    """Test ocp_complex function"""
    print("\n" + "="*60)
    print("Testing ocp_complex")
    print("="*60)
    
    # Test cases with different array sizes and value ranges
    test_cases = [
        (np.array([0.1, 0.3, 0.5, 0.7]), "Small array"),
        (np.linspace(-0.1, 1.0, 100), "Medium array"),
        (np.linspace(0.0, 1.0, 1000), "Large array"),
        (np.array([0.0, 0.025, 0.1, 0.5, 0.95, 0.975, 0.999, 1.0]), "Boundary values"),
    ]
    
    for x, description in test_cases:
        print(f"\n  Test: {description}")
        print(f"    x shape: {x.shape}")
        
        # Run original version
        t0 = time.time()
        E_eq_orig, dEeq_dx_orig = original_version.ocp_complex(x)
        t_orig = time.time() - t0
        
        # Run vectorized version
        t0 = time.time()
        E_eq_vec, dEeq_dx_vec = vectorized_version.ocp_complex(x)
        t_vec = time.time() - t0
        
        # Compare results
        E_eq_match = np.allclose(E_eq_orig, E_eq_vec, rtol=1e-10, atol=1e-12)
        dEeq_dx_match = np.allclose(dEeq_dx_orig, dEeq_dx_vec, rtol=1e-10, atol=1e-12)
        
        if E_eq_match and dEeq_dx_match:
            speedup = t_orig / t_vec if t_vec > 0 else float('inf')
            print(f"    ✓ Results match!")
            print(f"    Original time: {t_orig*1e6:.2f} µs")
            print(f"    Vectorized time: {t_vec*1e6:.2f} µs")
            print(f"    Speedup: {speedup:.2f}x")
        else:
            print(f"    ✗ Results DO NOT match!")
            if not E_eq_match:
                max_diff = np.max(np.abs(E_eq_orig - E_eq_vec))
                print(f"    Max difference in E_eq: {max_diff}")
            if not dEeq_dx_match:
                max_diff = np.max(np.abs(dEeq_dx_orig - dEeq_dx_vec))
                print(f"    Max difference in dEeq_dx: {max_diff}")
            return False
    
    return True


def benchmark_performance():
    """Benchmark performance on larger arrays"""
    print("\n" + "="*60)
    print("Performance Benchmark (Large Arrays)")
    print("="*60)
    
    sizes = [100, 1000, 10000, 100000]
    
    print("\n  Dn_complex:")
    for size in sizes:
        x = np.linspace(0.0, 1.0, size)
        D_damage_orig = np.linspace(0.0, 0.8, size)
        D_damage_vec = D_damage_orig.copy()
        
        # Warmup
        original_version.Dn_complex(x[:10], D_damage_orig[:10].copy())
        vectorized_version.Dn_complex(x[:10], D_damage_vec[:10].copy())
        
        # Benchmark original
        t0 = time.time()
        for _ in range(10):
            original_version.Dn_complex(x, D_damage_orig.copy())
        t_orig = (time.time() - t0) / 10
        
        # Benchmark vectorized
        t0 = time.time()
        for _ in range(10):
            vectorized_version.Dn_complex(x, D_damage_vec.copy())
        t_vec = (time.time() - t0) / 10
        
        speedup = t_orig / t_vec if t_vec > 0 else float('inf')
        print(f"    Size {size:6d}: Original {t_orig*1e3:8.3f} ms | Vectorized {t_vec*1e3:8.3f} ms | Speedup: {speedup:6.2f}x")
    
    print("\n  ocp_complex:")
    for size in sizes:
        x = np.linspace(0.0, 1.0, size)
        
        # Warmup
        original_version.ocp_complex(x[:10])
        vectorized_version.ocp_complex(x[:10])
        
        # Benchmark original
        t0 = time.time()
        for _ in range(10):
            original_version.ocp_complex(x)
        t_orig = (time.time() - t0) / 10
        
        # Benchmark vectorized
        t0 = time.time()
        for _ in range(10):
            vectorized_version.ocp_complex(x)
        t_vec = (time.time() - t0) / 10
        
        speedup = t_orig / t_vec if t_vec > 0 else float('inf')
        print(f"    Size {size:6d}: Original {t_orig*1e3:8.3f} ms | Vectorized {t_vec*1e3:8.3f} ms | Speedup: {speedup:6.2f}x")


def main():
    print("="*60)
    print("Vectorization Validation Test")
    print("define_buttler_volmer.py")
    print("="*60)
    
    all_passed = True
    
    # Test Dn_complex
    if not test_Dn_complex():
        all_passed = False
    
    # Test ocp_complex
    if not test_ocp_complex():
        all_passed = False
    
    # Benchmark performance
    benchmark_performance()
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe vectorized implementation produces identical results")
        print("to the original implementation.")
        return 0
    else:
        print("✗ SOME TESTS FAILED!")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

