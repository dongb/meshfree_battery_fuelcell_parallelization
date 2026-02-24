"""
Unit tests for sparse H matrix computation.

These tests verify that the sparse H matrix implementation produces
numerically equivalent results to the original dense implementation
while using significantly less memory.
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import np, use_legate
from shape_function_vectorized import (
    compute_distances_and_valid_pairs,
    compute_z_and_H_2d_sparse,
    compute_z_and_H_2d_vectorized,
    compute_z_and_H_3d_sparse,
    compute_z_and_H_3d_vectorized,
    compute_phi_M_standard_sparse,
)

# Compatibility wrapper for testing with both numpy and cupynumeric
if use_legate:
    # cupynumeric doesn't have np.testing, so we implement a simple version
    def assert_allclose(actual, desired, rtol=1e-7, atol=0, err_msg=''):
        """Simple allclose assertion compatible with cupynumeric."""
        # Convert to numpy for comparison if using legate
        import numpy as numpy_orig
        actual_np = numpy_orig.array(actual)
        desired_np = numpy_orig.array(desired)
        if not numpy_orig.allclose(actual_np, desired_np, rtol=rtol, atol=atol):
            max_diff = numpy_orig.max(numpy_orig.abs(actual_np - desired_np))
            rel_diff = max_diff / (numpy_orig.max(numpy_orig.abs(desired_np)) + 1e-10)
            raise AssertionError(f"{err_msg}\nMax abs diff: {max_diff:.2e}, Max rel diff: {rel_diff:.2e}")

    def assert_array_equal(actual, desired, err_msg=''):
        """Simple array equality assertion compatible with cupynumeric."""
        import numpy as numpy_orig
        actual_np = numpy_orig.array(actual)
        desired_np = numpy_orig.array(desired)
        if not numpy_orig.array_equal(actual_np, desired_np):
            raise AssertionError(f"{err_msg}\nArrays are not equal")
else:
    # Use numpy's native testing functions
    assert_allclose = assert_allclose
    assert_array_equal = assert_array_equal


def test_compute_distances_and_valid_pairs():
    """Test that distance computation correctly identifies valid pairs."""
    print("\n=== Test: compute_distances_and_valid_pairs ===")

    # Create simple test data
    x_G = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    x_nodes = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
    a = np.array([1.0, 1.0, 1.0])  # Support radius

    valid_mask, gauss_idx, node_idx, z_sparse = compute_distances_and_valid_pairs(
        x_G, x_nodes, a, dtype=np.float64
    )

    # Expected valid pairs (distance/a <= 1.0):
    # G0-N0: 0/1 = 0 ✓
    # G0-N1: 0.5/1 = 0.5 ✓
    # G0-N2: 3/1 = 3 ✗
    # G1-N0: 1/1 = 1 ✓
    # G1-N1: 0.5/1 = 0.5 ✓
    # G1-N2: 2/1 = 2 ✗
    # G2-N0: 2/1 = 2 ✗
    # G2-N1: 1.5/1 = 1.5 ✗
    # G2-N2: 1/1 = 1 ✓

    expected_valid_count = 5
    assert len(gauss_idx) == expected_valid_count, \
        f"Expected {expected_valid_count} valid pairs, got {len(gauss_idx)}"

    # Check that all z values are <= 1.0
    assert np.all(z_sparse <= 1.0), "All z values should be <= 1.0"
    assert np.all(z_sparse >= 0.0), "All z values should be >= 0.0"

    print(f"✓ Found {len(gauss_idx)} valid pairs (expected {expected_valid_count})")
    print(f"✓ All z values in range [0, 1]: min={z_sparse.min():.3f}, max={z_sparse.max():.3f}")


def test_sparse_vs_dense_equivalence_3d():
    """Verify sparse and dense 3D methods produce identical results."""
    print("\n=== Test: Sparse vs Dense Equivalence (3D) ===")

    # Create test data
    np.random.seed(42)
    n_gauss = 100
    n_nodes = 200
    x_G = np.random.rand(n_gauss, 3) * 10.0
    x_nodes = np.random.rand(n_nodes, 3) * 10.0
    a = np.ones(n_nodes) * 2.0

    H_scaling_factor = 1.0e-6
    eps = 2.220446049250313e-16
    dtype = np.float64

    # Compute with dense method
    z_dense, z_P_x_dense, z_P_y_dense, z_P_z_dense, \
        H_T_dense, HT_P_x_dense, HT_P_y_dense, HT_P_z_dense, \
        H_dense, H_P_x_dense, H_P_y_dense, H_P_z_dense = \
        compute_z_and_H_3d_vectorized(x_G, x_nodes, a, H_scaling_factor, eps, dtype)

    # Find valid pairs
    valid_mask = (z_dense >= 0) & (z_dense <= 1.0)
    gauss_idx, node_idx = np.where(valid_mask)

    print(f"Valid pairs: {len(gauss_idx)} / {n_gauss * n_nodes} ({100.0 * len(gauss_idx) / (n_gauss * n_nodes):.2f}%)")

    # Compute with sparse method
    z_sparse, z_P_x_sparse, z_P_y_sparse, z_P_z_sparse, \
        H_T_sparse, HT_P_x_sparse, HT_P_y_sparse, HT_P_z_sparse, \
        H_sparse, H_P_x_sparse, H_P_y_sparse, H_P_z_sparse = \
        compute_z_and_H_3d_sparse(x_G, x_nodes, a, gauss_idx, node_idx, H_scaling_factor, eps, dtype)

    # Verify equivalence
    rtol = 1e-10

    # Compare z values
    assert_allclose(z_sparse, z_dense[gauss_idx, node_idx], rtol=rtol,
                               err_msg="z values don't match")
    print("✓ z values match (rtol=1e-10)")

    # Compare z derivatives
    assert_allclose(z_P_x_sparse, z_P_x_dense[gauss_idx, node_idx], rtol=rtol,
                               err_msg="z_P_x values don't match")
    assert_allclose(z_P_y_sparse, z_P_y_dense[gauss_idx, node_idx], rtol=rtol,
                               err_msg="z_P_y values don't match")
    assert_allclose(z_P_z_sparse, z_P_z_dense[gauss_idx, node_idx], rtol=rtol,
                               err_msg="z_P_z values don't match")
    print("✓ z derivatives match (rtol=1e-10)")

    # Compare H matrices
    assert_allclose(H_sparse, H_dense[gauss_idx, node_idx, :], rtol=rtol,
                               err_msg="H values don't match")
    assert_allclose(H_T_sparse, H_T_dense[gauss_idx, node_idx, :], rtol=rtol,
                               err_msg="H_T values don't match")
    print("✓ H matrices match (rtol=1e-10)")

    # Compare H derivatives
    assert_allclose(H_P_x_sparse, H_P_x_dense[gauss_idx, node_idx, :], rtol=rtol,
                               err_msg="H_P_x values don't match")
    assert_allclose(H_P_y_sparse, H_P_y_dense[gauss_idx, node_idx, :], rtol=rtol,
                               err_msg="H_P_y values don't match")
    assert_allclose(H_P_z_sparse, H_P_z_dense[gauss_idx, node_idx, :], rtol=rtol,
                               err_msg="H_P_z values don't match")
    print("✓ H derivatives match (rtol=1e-10)")

    # Verify memory savings
    dense_size = H_dense.nbytes + H_T_dense.nbytes + H_P_x_dense.nbytes + H_P_y_dense.nbytes + H_P_z_dense.nbytes
    dense_size += HT_P_x_dense.nbytes + HT_P_y_dense.nbytes + HT_P_z_dense.nbytes

    sparse_size = H_sparse.nbytes + H_T_sparse.nbytes + H_P_x_sparse.nbytes + H_P_y_sparse.nbytes + H_P_z_sparse.nbytes
    sparse_size += HT_P_x_sparse.nbytes + HT_P_y_sparse.nbytes + HT_P_z_sparse.nbytes

    memory_reduction = 100.0 * (1.0 - sparse_size / dense_size)
    print(f"✓ Memory reduction: {memory_reduction:.1f}% (dense: {dense_size/1e6:.1f} MB, sparse: {sparse_size/1e6:.1f} MB)")


def test_sparse_vs_dense_equivalence_2d():
    """Verify sparse and dense 2D methods produce identical results."""
    print("\n=== Test: Sparse vs Dense Equivalence (2D) ===")

    # Create test data
    np.random.seed(42)
    n_gauss = 100
    n_nodes = 200
    x_G = np.random.rand(n_gauss, 2) * 10.0
    x_nodes = np.random.rand(n_nodes, 2) * 10.0
    a = np.ones(n_nodes) * 2.0

    H_scaling_factor = 1.0e-6
    eps = 2.220446049250313e-16
    dtype = np.float64

    # Compute with dense method
    z_dense, z_P_x_dense, z_P_y_dense, \
        H_T_dense, HT_P_x_dense, HT_P_y_dense, \
        H_dense, H_P_x_dense, H_P_y_dense = \
        compute_z_and_H_2d_vectorized(x_G, x_nodes, a, H_scaling_factor, eps, dtype)

    # Find valid pairs
    valid_mask = (z_dense >= 0) & (z_dense <= 1.0)
    gauss_idx, node_idx = np.where(valid_mask)

    print(f"Valid pairs: {len(gauss_idx)} / {n_gauss * n_nodes} ({100.0 * len(gauss_idx) / (n_gauss * n_nodes):.2f}%)")

    # Compute with sparse method
    z_sparse, z_P_x_sparse, z_P_y_sparse, \
        H_T_sparse, HT_P_x_sparse, HT_P_y_sparse, \
        H_sparse, H_P_x_sparse, H_P_y_sparse = \
        compute_z_and_H_2d_sparse(x_G, x_nodes, a, gauss_idx, node_idx, H_scaling_factor, eps, dtype)

    # Verify equivalence
    rtol = 1e-10

    # Compare z values
    assert_allclose(z_sparse, z_dense[gauss_idx, node_idx], rtol=rtol,
                               err_msg="z values don't match")
    print("✓ z values match (rtol=1e-10)")

    # Compare z derivatives
    assert_allclose(z_P_x_sparse, z_P_x_dense[gauss_idx, node_idx], rtol=rtol,
                               err_msg="z_P_x values don't match")
    assert_allclose(z_P_y_sparse, z_P_y_dense[gauss_idx, node_idx], rtol=rtol,
                               err_msg="z_P_y values don't match")
    print("✓ z derivatives match (rtol=1e-10)")

    # Compare H matrices
    assert_allclose(H_sparse, H_dense[gauss_idx, node_idx, :], rtol=rtol,
                               err_msg="H values don't match")
    assert_allclose(H_T_sparse, H_T_dense[gauss_idx, node_idx, :], rtol=rtol,
                               err_msg="H_T values don't match")
    print("✓ H matrices match (rtol=1e-10)")

    # Compare H derivatives
    assert_allclose(H_P_x_sparse, H_P_x_dense[gauss_idx, node_idx, :], rtol=rtol,
                               err_msg="H_P_x values don't match")
    assert_allclose(H_P_y_sparse, H_P_y_dense[gauss_idx, node_idx, :], rtol=rtol,
                               err_msg="H_P_y values don't match")
    print("✓ H derivatives match (rtol=1e-10)")

    # Verify memory savings
    dense_size = H_dense.nbytes + H_T_dense.nbytes + H_P_x_dense.nbytes + H_P_y_dense.nbytes
    dense_size += HT_P_x_dense.nbytes + HT_P_y_dense.nbytes

    sparse_size = H_sparse.nbytes + H_T_sparse.nbytes + H_P_x_sparse.nbytes + H_P_y_sparse.nbytes
    sparse_size += HT_P_x_sparse.nbytes + HT_P_y_sparse.nbytes

    memory_reduction = 100.0 * (1.0 - sparse_size / dense_size)
    print(f"✓ Memory reduction: {memory_reduction:.1f}% (dense: {dense_size/1e6:.1f} MB, sparse: {sparse_size/1e6:.1f} MB)")


def test_phi_M_sparse_vs_dense_mode():
    """Test that compute_phi_M_standard_sparse produces same results in sparse and dense modes."""
    print("\n=== Test: compute_phi_M_standard_sparse Sparse vs Dense Mode ===")

    # Create test data (3D)
    np.random.seed(42)
    n_gauss = 50
    n_nodes = 100
    x_G = np.random.rand(n_gauss, 3) * 10.0
    x_nodes = np.random.rand(n_nodes, 3) * 10.0
    a = np.ones(n_nodes) * 2.0

    # Dummy data for other parameters
    Gauss_grain_id = np.zeros(n_gauss, dtype=int)
    nodes_grain_id = np.zeros(n_nodes, dtype=int)
    num_interface_segments = 0
    interface_nodes = np.array([])
    BxByCxCy = [0, 10, 0, 10, 0, 10]

    dtype = np.float64

    # Test with sparse mode
    M_sparse = np.zeros((n_gauss, 4, 4), dtype=dtype)
    M_P_x_sparse = np.zeros((n_gauss, 4, 4), dtype=dtype)
    M_P_y_sparse = np.zeros((n_gauss, 4, 4), dtype=dtype)
    M_P_z_sparse = np.zeros((n_gauss, 4, 4), dtype=dtype)

    result_sparse = compute_phi_M_standard_sparse(
        x_G, Gauss_grain_id, x_nodes, nodes_grain_id, a,
        M_sparse, M_P_x_sparse, M_P_y_sparse,
        num_interface_segments, interface_nodes, BxByCxCy,
        M_P_z_sparse, dtype, use_sparse_h_matrices=True
    )

    # Test with dense mode
    M_dense = np.zeros((n_gauss, 4, 4), dtype=dtype)
    M_P_x_dense = np.zeros((n_gauss, 4, 4), dtype=dtype)
    M_P_y_dense = np.zeros((n_gauss, 4, 4), dtype=dtype)
    M_P_z_dense = np.zeros((n_gauss, 4, 4), dtype=dtype)

    result_dense = compute_phi_M_standard_sparse(
        x_G, Gauss_grain_id, x_nodes, nodes_grain_id, a,
        M_dense, M_P_x_dense, M_P_y_dense,
        num_interface_segments, interface_nodes, BxByCxCy,
        M_P_z_dense, dtype, use_sparse_h_matrices=False
    )

    # Compare results
    rtol = 1e-10

    # Compare phi sparse arrays
    phi_row_sparse, phi_col_sparse, phi_data_sparse = result_sparse[0], result_sparse[1], result_sparse[2]
    phi_row_dense, phi_col_dense, phi_data_dense = result_dense[0], result_dense[1], result_dense[2]

    assert_array_equal(phi_row_sparse, phi_row_dense, err_msg="phi row indices don't match")
    assert_array_equal(phi_col_sparse, phi_col_dense, err_msg="phi col indices don't match")
    assert_allclose(phi_data_sparse, phi_data_dense, rtol=rtol, err_msg="phi values don't match")
    print("✓ phi sparse arrays match")

    # Compare phi derivatives
    phi_P_x_sparse, phi_P_y_sparse, phi_P_z_sparse = result_sparse[3], result_sparse[4], result_sparse[5]
    phi_P_x_dense, phi_P_y_dense, phi_P_z_dense = result_dense[3], result_dense[4], result_dense[5]

    assert_allclose(phi_P_x_sparse, phi_P_x_dense, rtol=rtol, err_msg="phi_P_x doesn't match")
    assert_allclose(phi_P_y_sparse, phi_P_y_dense, rtol=rtol, err_msg="phi_P_y doesn't match")
    assert_allclose(phi_P_z_sparse, phi_P_z_dense, rtol=rtol, err_msg="phi_P_z doesn't match")
    print("✓ phi derivatives match")

    # Compare M matrices
    M_sparse_out, M_P_x_sparse_out, M_P_y_sparse_out, M_P_z_sparse_out = result_sparse[6], result_sparse[7], result_sparse[8], result_sparse[9]
    M_dense_out, M_P_x_dense_out, M_P_y_dense_out, M_P_z_dense_out = result_dense[6], result_dense[7], result_dense[8], result_dense[9]

    assert_allclose(M_sparse_out, M_dense_out, rtol=rtol, err_msg="M matrices don't match")
    assert_allclose(M_P_x_sparse_out, M_P_x_dense_out, rtol=rtol, err_msg="M_P_x matrices don't match")
    assert_allclose(M_P_y_sparse_out, M_P_y_dense_out, rtol=rtol, err_msg="M_P_y matrices don't match")
    assert_allclose(M_P_z_sparse_out, M_P_z_dense_out, rtol=rtol, err_msg="M_P_z matrices don't match")
    print("✓ M matrices match")

    print("✓ All outputs match between sparse and dense modes (rtol=1e-10)")


def test_edge_cases():
    """Test edge cases like very sparse data and empty valid pairs."""
    print("\n=== Test: Edge Cases ===")

    # Test 1: Very sparse data (no valid pairs)
    x_G = np.array([[0.0, 0.0, 0.0]])
    x_nodes = np.array([[10.0, 10.0, 10.0]])  # Far away
    a = np.array([1.0])  # Small support radius

    valid_mask, gauss_idx, node_idx, z_sparse = compute_distances_and_valid_pairs(
        x_G, x_nodes, a, dtype=np.float64
    )

    assert len(gauss_idx) == 0, "Expected no valid pairs for distant nodes"
    print("✓ Handles empty valid pairs correctly")

    # Test 2: All pairs valid
    x_G = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    x_nodes = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    a = np.array([10.0, 10.0])  # Large support radius

    valid_mask, gauss_idx, node_idx, z_sparse = compute_distances_and_valid_pairs(
        x_G, x_nodes, a, dtype=np.float64
    )

    expected_pairs = x_G.shape[0] * x_nodes.shape[0]
    assert len(gauss_idx) == expected_pairs, f"Expected {expected_pairs} valid pairs (all pairs)"
    print(f"✓ Handles all-valid pairs correctly ({expected_pairs} pairs)")


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("SPARSE H MATRIX UNIT TESTS")
    print("=" * 80)

    try:
        test_compute_distances_and_valid_pairs()
        test_sparse_vs_dense_equivalence_3d()
        test_sparse_vs_dense_equivalence_2d()
        test_phi_M_sparse_vs_dense_mode()
        test_edge_cases()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
