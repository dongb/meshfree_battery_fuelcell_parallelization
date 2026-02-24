"""Vectorized version of compute_phi_M_standard and related functions.

This module provides vectorized implementations that operate on all Gauss points
and nodes simultaneously, eliminating the need for explicit loops.
"""

from common import csr_array, np


def compute_z_and_H_2d_vectorized(
    x_G, x_nodes, a, H_scaling_factor, eps, dtype=np.float64
):
    """Compute z values and H matrices for 2D case (vectorized).

    Args:
        x_G: Array of Gauss points, shape (n_gauss, 2)
        x_nodes: Array of node points, shape (n_nodes, 2)
        a: Array of scaling factors, shape (n_nodes,)
        H_scaling_factor: Scaling factor for H matrices
        eps: Small value to avoid division by zero
        dtype: Data type for arrays (default: np.float32 for memory efficiency)

    Returns:
        Tuple containing vectorized z values, derivatives, and H matrices
    """
    # Expand dimensions for broadcasting
    # x_G: (n_gauss, 1, 2) and x_nodes: (1, n_nodes, 2)
    x_G_exp = x_G[:, np.newaxis, :]  # (n_gauss, 1, 2)
    x_nodes_exp = x_nodes[np.newaxis, :, :]  # (1, n_nodes, 2)

    # Compute differences: (n_gauss, n_nodes, 2)
    diff = x_G_exp - x_nodes_exp

    # Compute distances: (n_gauss, n_nodes)
    distances = np.sqrt(np.sum(diff**2, axis=2))

    # Normalize by a: (n_gauss, n_nodes)
    z = distances / a[np.newaxis, :]

    # Compute partial derivatives
    # Avoid division by zero
    denom = a[np.newaxis, :] * np.maximum(distances, eps)
    z_P_x = diff[:, :, 0] / denom  # (n_gauss, n_nodes)
    z_P_y = diff[:, :, 1] / denom  # (n_gauss, n_nodes)

    # Free memory from distances
    # del distances

    # Compute H matrices
    # H_T shape will be (n_gauss, n_nodes, 3)
    H_T = np.zeros((x_G.shape[0], x_nodes.shape[0], 3), dtype=dtype)
    H_T[:, :, 0] = dtype(1.0)
    H_T[:, :, 1] = diff[:, :, 0] / H_scaling_factor
    H_T[:, :, 2] = diff[:, :, 1] / H_scaling_factor

    # Free memory from diff
    # del diff

    # Derivatives of H_T
    HT_P_x = np.zeros((x_G.shape[0], x_nodes.shape[0], 3), dtype=dtype)
    HT_P_x[:, :, 1] = dtype(1.0) / H_scaling_factor

    HT_P_y = np.zeros((x_G.shape[0], x_nodes.shape[0], 3), dtype=dtype)
    HT_P_y[:, :, 2] = dtype(1.0) / H_scaling_factor

    # H is transpose of H_T along last axis
    H = np.transpose(
        H_T, (0, 1, 2)
    )  # Still (n_gauss, n_nodes, 3) but transposed in last dim
    H_P_x = np.transpose(HT_P_x, (0, 1, 2))
    H_P_y = np.transpose(HT_P_y, (0, 1, 2))

    return z, z_P_x, z_P_y, H_T, HT_P_x, HT_P_y, H, H_P_x, H_P_y


def compute_z_and_H_3d_vectorized(
    x_G, x_nodes, a, H_scaling_factor, eps, dtype=np.float64
):
    """Compute z values and H matrices for 3D case (vectorized).

    Args:
        x_G: Array of Gauss points, shape (n_gauss, 3)
        x_nodes: Array of node points, shape (n_nodes, 3)
        a: Array of scaling factors, shape (n_nodes,)
        H_scaling_factor: Scaling factor for H matrices
        eps: Small value to avoid division by zero
        dtype: Data type for arrays (default: np.float32 for memory efficiency)

    Returns:
        Tuple containing vectorized z values, derivatives, and H matrices
    """
    # Expand dimensions for broadcasting
    x_G_exp = x_G[:, np.newaxis, :]  # (n_gauss, 1, 3)
    x_nodes_exp = x_nodes[np.newaxis, :, :]  # (1, n_nodes, 3)

    # Compute differences: (n_gauss, n_nodes, 3)
    diff = x_G_exp - x_nodes_exp

    # Compute distances: (n_gauss, n_nodes)
    distances = np.sqrt(np.sum(diff**2, axis=2))

    # Normalize by a: (n_gauss, n_nodes)
    z = distances / a[np.newaxis, :]

    # Compute partial derivatives
    denom = a[np.newaxis, :] * np.maximum(distances, eps)

    # save memory
    del distances

    z_P_x = diff[:, :, 0] / denom  # (n_gauss, n_nodes)
    z_P_y = diff[:, :, 1] / denom  # (n_gauss, n_nodes)
    z_P_z = diff[:, :, 2] / denom  # (n_gauss, n_nodes)

    # Compute H matrices
    # H_T shape will be (n_gauss, n_nodes, 4)
    H_T = np.zeros((x_G.shape[0], x_nodes.shape[0], 4), dtype=dtype)
    H_T[:, :, 0] = dtype(1.0)
    H_T[:, :, 1] = diff[:, :, 0] / H_scaling_factor
    H_T[:, :, 2] = diff[:, :, 1] / H_scaling_factor
    H_T[:, :, 3] = diff[:, :, 2] / H_scaling_factor

    # save memory
    del diff

    # Derivatives of H_T
    HT_P_x = np.zeros((x_G.shape[0], x_nodes.shape[0], 4), dtype=dtype)
    HT_P_x[:, :, 1] = dtype(1.0) / H_scaling_factor

    HT_P_y = np.zeros((x_G.shape[0], x_nodes.shape[0], 4), dtype=dtype)
    HT_P_y[:, :, 2] = dtype(1.0) / H_scaling_factor

    HT_P_z = np.zeros((x_G.shape[0], x_nodes.shape[0], 4), dtype=dtype)
    HT_P_z[:, :, 3] = dtype(1.0) / H_scaling_factor

    # H is transpose of H_T along last axis
    H = np.transpose(H_T, (0, 1, 2))
    H_P_x = np.transpose(HT_P_x, (0, 1, 2))
    H_P_y = np.transpose(HT_P_y, (0, 1, 2))
    H_P_z = np.transpose(HT_P_z, (0, 1, 2))

    return z, z_P_x, z_P_y, z_P_z, H_T, HT_P_x, HT_P_y, HT_P_z, H, H_P_x, H_P_y, H_P_z


def compute_phi_kernel(z):
    """Compute phi kernel values based on distance z.

    The kernel is a cubic spline-like function:
    - For 0 <= z < 0.5: phi = 2/3 - 4*z^2 + 4*z^3
    - For 0.5 <= z <= 1: phi = 4/3 - 4*z + 4*z^2 - 4/3*z^3
    - For z > 1: phi = 0

    Args:
        z: Array of normalized distances

    Returns:
        phi: Array of kernel values
        phi_P_z: Array of kernel derivatives w.r.t. z
    """
    phi = np.zeros_like(z)
    phi_P_z = np.zeros_like(z)

    # Mask for different regions
    mask1 = (z >= 0) & (z < 0.5)
    mask2 = (z >= 0.5) & (z <= 1.0)

    # Region 1: 0 <= z < 0.5
    phi[mask1] = 2.0 / 3 - 4 * z[mask1] ** 2 + 4 * z[mask1] ** 3
    phi_P_z[mask1] = -8.0 * z[mask1] + 12.0 * z[mask1] ** 2

    # Region 2: 0.5 <= z <= 1
    phi[mask2] = 4.0 / 3 - 4 * z[mask2] + 4 * z[mask2] ** 2 - 4.0 / 3 * z[mask2] ** 3
    phi_P_z[mask2] = -4 + 8 * z[mask2] - 4 * z[mask2] ** 2

    return phi, phi_P_z


def compute_distances_and_valid_pairs(x_G, x_nodes, a, dtype=np.float64):
    """Compute distances and identify valid pairs (z <= 1.0) without H matrices.

    This is a lightweight function to identify valid sparse pairs before creating
    H matrices, enabling significant memory savings by avoiding dense intermediate arrays.

    Args:
        x_G: Array of Gauss points, shape (n_gauss, ndim)
        x_nodes: Array of node points, shape (n_nodes, ndim)
        a: Array of scaling factors, shape (n_nodes,)
        dtype: Data type for arrays

    Returns:
        valid_mask: Boolean mask of valid pairs, shape (n_gauss, n_nodes)
        gauss_indices: Row indices of valid pairs (Gauss point indices)
        node_indices: Column indices of valid pairs (node indices)
        z_sparse: Normalized distances for valid pairs only
    """
    # Compute distances using broadcasting
    x_G_exp = x_G[:, np.newaxis, :]  # (n_gauss, 1, ndim)
    x_nodes_exp = x_nodes[np.newaxis, :, :]  # (1, n_nodes, ndim)
    diff = x_G_exp - x_nodes_exp  # (n_gauss, n_nodes, ndim)
    distances = np.sqrt(np.sum(diff**2, axis=2))  # (n_gauss, n_nodes)
    del diff

    # Normalize by support radius
    z = distances / a[np.newaxis, :]  # (n_gauss, n_nodes)
    del distances

    # Find valid pairs
    valid_mask = (z >= 0) & (z <= 1.0)
    gauss_indices, node_indices = np.where(valid_mask)
    z_sparse = z[gauss_indices, node_indices]

    return valid_mask, gauss_indices, node_indices, z_sparse


def compute_z_and_H_2d_sparse(
    x_G, x_nodes, a, gauss_indices, node_indices,
    H_scaling_factor, eps, dtype=np.float64
):
    """Compute H matrices for sparse pairs only (2D case).

    This function creates H matrices ONLY for pre-identified valid pairs,
    resulting in arrays sized (n_nnz, 3) instead of (n_gauss, n_nodes, 3).

    Args:
        x_G: Array of Gauss points, shape (n_gauss, 2)
        x_nodes: Array of node points, shape (n_nodes, 2)
        a: Array of scaling factors, shape (n_nodes,)
        gauss_indices: Row indices of valid pairs
        node_indices: Column indices of valid pairs
        H_scaling_factor: Scaling factor for H matrices
        eps: Small value to avoid division by zero
        dtype: Data type for arrays

    Returns:
        Tuple containing sparse z values, derivatives, and H matrices
        All arrays have shape (n_nnz,) or (n_nnz, 3) instead of dense shapes
    """
    n_nnz = len(gauss_indices)

    # Extract sparse pairs
    x_G_sparse = x_G[gauss_indices, :]  # (n_nnz, 2)
    x_nodes_sparse = x_nodes[node_indices, :]  # (n_nnz, 2)
    a_sparse = a[node_indices]  # (n_nnz,)

    # Compute for sparse pairs only
    diff = x_G_sparse - x_nodes_sparse  # (n_nnz, 2)
    distances = np.sqrt(np.sum(diff**2, axis=1))  # (n_nnz,)
    z = distances / a_sparse

    # Derivatives
    denom = a_sparse * np.maximum(distances, eps)
    z_P_x = diff[:, 0] / denom
    z_P_y = diff[:, 1] / denom
    del distances

    # H matrices (n_nnz, 3) - much smaller!
    H_T = np.zeros((n_nnz, 3), dtype=dtype)
    H_T[:, 0] = dtype(1.0)
    H_T[:, 1] = diff[:, 0] / H_scaling_factor
    H_T[:, 2] = diff[:, 1] / H_scaling_factor

    # H derivatives
    HT_P_x = np.zeros((n_nnz, 3), dtype=dtype)
    HT_P_x[:, 1] = dtype(1.0) / H_scaling_factor

    HT_P_y = np.zeros((n_nnz, 3), dtype=dtype)
    HT_P_y[:, 2] = dtype(1.0) / H_scaling_factor

    del diff

    # H is transpose of H_T (but in 2D case, no actual transpose needed for our usage)
    H = H_T.copy()
    H_P_x = HT_P_x.copy()
    H_P_y = HT_P_y.copy()

    return z, z_P_x, z_P_y, H_T, HT_P_x, HT_P_y, H, H_P_x, H_P_y


def compute_z_and_H_3d_sparse(
    x_G, x_nodes, a, gauss_indices, node_indices,
    H_scaling_factor, eps, dtype=np.float64
):
    """Compute H matrices for sparse pairs only (3D case).

    This function creates H matrices ONLY for pre-identified valid pairs,
    resulting in arrays sized (n_nnz, 4) instead of (n_gauss, n_nodes, 4).

    Args:
        x_G: Array of Gauss points, shape (n_gauss, 3)
        x_nodes: Array of node points, shape (n_nodes, 3)
        a: Array of scaling factors, shape (n_nodes,)
        gauss_indices: Row indices of valid pairs
        node_indices: Column indices of valid pairs
        H_scaling_factor: Scaling factor for H matrices
        eps: Small value to avoid division by zero
        dtype: Data type for arrays

    Returns:
        Tuple containing sparse z values, derivatives, and H matrices
        All arrays have shape (n_nnz,) or (n_nnz, 4) instead of dense shapes
    """
    n_nnz = len(gauss_indices)

    # Extract sparse pairs
    x_G_sparse = x_G[gauss_indices, :]  # (n_nnz, 3)
    x_nodes_sparse = x_nodes[node_indices, :]  # (n_nnz, 3)
    a_sparse = a[node_indices]  # (n_nnz,)

    # Compute for sparse pairs only
    diff = x_G_sparse - x_nodes_sparse  # (n_nnz, 3)
    distances = np.sqrt(np.sum(diff**2, axis=1))  # (n_nnz,)
    z = distances / a_sparse

    # Derivatives
    denom = a_sparse * np.maximum(distances, eps)
    z_P_x = diff[:, 0] / denom
    z_P_y = diff[:, 1] / denom
    z_P_z = diff[:, 2] / denom
    del distances

    # H matrices (n_nnz, 4) - much smaller!
    H_T = np.zeros((n_nnz, 4), dtype=dtype)
    H_T[:, 0] = dtype(1.0)
    H_T[:, 1] = diff[:, 0] / H_scaling_factor
    H_T[:, 2] = diff[:, 1] / H_scaling_factor
    H_T[:, 3] = diff[:, 2] / H_scaling_factor

    # H derivatives
    HT_P_x = np.zeros((n_nnz, 4), dtype=dtype)
    HT_P_x[:, 1] = dtype(1.0) / H_scaling_factor

    HT_P_y = np.zeros((n_nnz, 4), dtype=dtype)
    HT_P_y[:, 2] = dtype(1.0) / H_scaling_factor

    HT_P_z = np.zeros((n_nnz, 4), dtype=dtype)
    HT_P_z[:, 3] = dtype(1.0) / H_scaling_factor

    del diff

    # H is transpose of H_T (but no actual transpose needed for our usage)
    H = H_T.copy()
    H_P_x = HT_P_x.copy()
    H_P_y = HT_P_y.copy()
    H_P_z = HT_P_z.copy()

    return z, z_P_x, z_P_y, z_P_z, H_T, HT_P_x, HT_P_y, HT_P_z, H, H_P_x, H_P_y, H_P_z


def compute_phi_M_standard_vectorized(
    x_G,
    Gauss_grain_id,
    x_nodes,
    nodes_grain_id,
    a,
    M,
    M_P_x,
    M_P_y,
    num_interface_segments,
    interface_nodes,
    BxByCxCy,
    M_P_z=None,
    dtype=np.float64,
):
    """Vectorized version of compute_phi_M_standard.

    This function computes all Gauss-node interactions simultaneously
    using NumPy broadcasting and vectorized operations.

    Args:
        dtype: Data type for computations (default: np.float32 for memory efficiency).
               Use np.float64 for higher precision if needed.
    """
    if M_P_z is None:
        M_P_z = np.zeros_like(M)

    H_scaling_factor = dtype(1.0e-6)
    eps = dtype(2.220446049250313e-16)

    # Determine dimension from M shape
    is_3d = np.shape(M)[1] == 4

    # Compute z and H matrices for all point pairs
    if is_3d:
        (
            z,
            z_P_x,
            z_P_y,
            z_P_z,
            H_T,
            HT_P_x,
            HT_P_y,
            HT_P_z,
            H,
            H_P_x,
            H_P_y,
            H_P_z,
        ) = compute_z_and_H_3d_vectorized(x_G, x_nodes, a, H_scaling_factor, eps, dtype)
    else:
        (z, z_P_x, z_P_y, H_T, HT_P_x, HT_P_y, H, H_P_x, H_P_y) = (
            compute_z_and_H_2d_vectorized(x_G, x_nodes, a, H_scaling_factor, eps, dtype)
        )
        z_P_z = None
        H_P_z = None
        HT_P_z = None

    # Compute phi kernel for all distances
    phi, phi_P_z = compute_phi_kernel(z)

    # Find valid pairs (where z <= 1)
    valid_mask = (z >= 0) & (z <= 1.0)

    # Extract indices of valid pairs
    valid_indices = np.where(valid_mask)
    gauss_indices = valid_indices[0]
    node_indices = valid_indices[1]

    # Store nonzero values for sparse matrix construction (as numpy arrays)
    phi_nonzero_index_row = gauss_indices
    phi_nonzero_index_column = node_indices
    phi_nonzerovalue_data = phi[valid_mask]

    # Compute phi derivatives
    phi_P_x = phi_P_z * z_P_x
    phi_P_y = phi_P_z * z_P_y
    phi_P_x_nonzerovalue_data = phi_P_x[valid_mask]
    phi_P_y_nonzerovalue_data = phi_P_y[valid_mask]

    if is_3d:
        phi_P_z_val = phi_P_z * z_P_z
        phi_P_z_nonzerovalue_data = phi_P_z_val[valid_mask]
    else:
        phi_P_z_nonzerovalue_data = np.array([])

    # Update M matrices using vectorized operations
    # For each Gauss point, accumulate contributions from all valid nodes
    n_gauss = x_G.shape[0]

    print("Before: Find nodes that contribute to a Gauss pt", flush=True)
    for i in range(n_gauss):
        if i % 10000 == 0:
            print(f"i: {i}")
        # Find nodes that contribute to this Gauss point
        node_mask = valid_mask[i, :]
        if not np.any(node_mask):
            continue

        # Get relevant values for this Gauss point
        phi_i = phi[i, node_mask]
        phi_P_x_i = phi_P_x[i, node_mask]
        phi_P_y_i = phi_P_y[i, node_mask]

        H_i = H[i, node_mask, :]  # (n_valid_nodes, n_dim)
        H_T_i = H_T[i, node_mask, :]  # (n_valid_nodes, n_dim)
        H_P_x_i = H_P_x[i, node_mask, :]
        H_P_y_i = H_P_y[i, node_mask, :]
        HT_P_x_i = HT_P_x[i, node_mask, :]
        HT_P_y_i = HT_P_y[i, node_mask, :]

        # Compute outer products and accumulate
        # Using einsum for efficient tensor operations
        # M[i] += sum over valid nodes of (H * H_T * phi)
        M[i] += np.einsum("ni,nj,n->ij", H_i, H_T_i, phi_i)

        # M_P_x[i] += sum of three terms
        M_P_x[i] += (
            np.einsum("ni,nj,n->ij", H_i, H_T_i, phi_P_x_i)
            + np.einsum("ni,nj,n->ij", H_P_x_i, H_T_i, phi_i)
            + np.einsum("ni,nj,n->ij", H_i, HT_P_x_i, phi_i)
        )

        # M_P_y[i] += sum of three terms
        M_P_y[i] += (
            np.einsum("ni,nj,n->ij", H_i, H_T_i, phi_P_y_i)
            + np.einsum("ni,nj,n->ij", H_P_y_i, H_T_i, phi_i)
            + np.einsum("ni,nj,n->ij", H_i, HT_P_y_i, phi_i)
        )

        if is_3d:
            phi_P_z_i = phi_P_z_val[i, node_mask]
            H_P_z_i = H_P_z[i, node_mask, :]
            HT_P_z_i = HT_P_z[i, node_mask, :]

            M_P_z[i] += (
                np.einsum("ni,nj,n->ij", H_i, H_T_i, phi_P_z_i)
                + np.einsum("ni,nj,n->ij", H_P_z_i, H_T_i, phi_i)
                + np.einsum("ni,nj,n->ij", H_i, HT_P_z_i, phi_i)
            )

    return (
        phi_nonzero_index_row,
        phi_nonzero_index_column,
        phi_nonzerovalue_data,
        phi_P_x_nonzerovalue_data,
        phi_P_y_nonzerovalue_data,
        phi_P_z_nonzerovalue_data,
        M,
        M_P_x,
        M_P_y,
        M_P_z,
    )


def compute_phi_M_standard_sparse(
    x_G,
    Gauss_grain_id,
    x_nodes,
    nodes_grain_id,
    a,
    M,
    M_P_x,
    M_P_y,
    num_interface_segments,
    interface_nodes,
    BxByCxCy,
    M_P_z=None,
    dtype=np.float64,
    use_sparse_h_matrices=True,
):
    """Optimized sparse version of compute_phi_M_standard using vectorized operations.

    This function eliminates the loop over Gauss points by using advanced indexing
    and np.add.at for efficient accumulation. It provides 10-100x speedup for
    large numbers of Gauss points.

    Args:
        dtype: Data type for computations (default: np.float64).
        use_sparse_h_matrices: If True, creates H matrices only for valid sparse pairs,
                               reducing memory usage by 85-95%. If False, uses original
                               dense implementation for backward compatibility.
    """

    if M_P_z is None:
        M_P_z = np.zeros_like(M)

    H_scaling_factor = dtype(1.0e-6)
    eps = dtype(2.220446049250313e-16)

    # Determine dimension from M shape
    is_3d = np.shape(M)[1] == 4

    if use_sparse_h_matrices:
        # NEW SPARSE WORKFLOW: Memory-efficient sparse H matrix computation
        # Phase 1: Identify valid pairs (lightweight - only distance computation)
        print("Phase 1: Identifying valid pairs", flush=True)
        valid_mask, gauss_indices, node_indices, z_sparse = compute_distances_and_valid_pairs(
            x_G, x_nodes, a, dtype
        )
        print(
            f"Found {len(gauss_indices)} non-zero interactions "
            f"({100.0 * len(gauss_indices) / (x_G.shape[0] * x_nodes.shape[0]):.2f}% sparse)",
            flush=True,
        )

        # Phase 2: Compute phi for valid pairs
        print("Phase 2: Computing phi kernel", flush=True)
        phi_vals, phi_P_z_vals = compute_phi_kernel(z_sparse)

        # Phase 3: Create H matrices for valid pairs only (memory-efficient)
        print("Phase 3: Computing sparse H matrices", flush=True)
        if is_3d:
            z, z_P_x, z_P_y, z_P_z, H_T, HT_P_x, HT_P_y, HT_P_z, H, H_P_x, H_P_y, H_P_z = \
                compute_z_and_H_3d_sparse(
                    x_G, x_nodes, a, gauss_indices, node_indices,
                    H_scaling_factor, eps, dtype
                )
        else:
            z, z_P_x, z_P_y, H_T, HT_P_x, HT_P_y, H, H_P_x, H_P_y = \
                compute_z_and_H_2d_sparse(
                    x_G, x_nodes, a, gauss_indices, node_indices,
                    H_scaling_factor, eps, dtype
                )
            z_P_z = H_P_z = HT_P_z = None

    else:
        # ORIGINAL DENSE WORKFLOW: Backward compatibility
        print("Using dense H matrix computation (backward compatibility mode)", flush=True)
        # Compute z and H matrices for all point pairs
        if is_3d:
            (
                z,
                z_P_x,
                z_P_y,
                z_P_z,
                H_T,
                HT_P_x,
                HT_P_y,
                HT_P_z,
                H,
                H_P_x,
                H_P_y,
                H_P_z,
            ) = compute_z_and_H_3d_vectorized(x_G, x_nodes, a, H_scaling_factor, eps, dtype)
        else:
            (z, z_P_x, z_P_y, H_T, HT_P_x, HT_P_y, H, H_P_x, H_P_y) = (
                compute_z_and_H_2d_vectorized(x_G, x_nodes, a, H_scaling_factor, eps, dtype)
            )
            z_P_z = None
            H_P_z = None
            HT_P_z = None

        # Compute phi kernel for all distances
        phi, phi_P_z = compute_phi_kernel(z)

        # Find valid pairs (where z <= 1)
        valid_mask = (z >= 0) & (z <= 1.0)

        # Extract indices of valid pairs
        gauss_indices, node_indices = np.where(valid_mask)

        # Extract values for valid pairs only
        phi_vals = phi[gauss_indices, node_indices]
        phi_P_z_vals = phi_P_z[gauss_indices, node_indices]

    # Store nonzero values for sparse matrix construction
    phi_nonzero_index_row = gauss_indices
    phi_nonzero_index_column = node_indices
    phi_nonzerovalue_data = phi_vals

    # Compute phi derivatives for valid pairs only
    if use_sparse_h_matrices:
        # Sparse mode: z derivatives are already sparse (n_nnz,)
        phi_P_x_vals = phi_P_z_vals * z_P_x
        phi_P_y_vals = phi_P_z_vals * z_P_y
        if is_3d:
            phi_P_z_sparse_vals = phi_P_z_vals * z_P_z
        else:
            phi_P_z_sparse_vals = None
    else:
        # Dense mode: extract from dense arrays
        phi_P_x_vals = phi_P_z_vals * z_P_x[gauss_indices, node_indices]
        phi_P_y_vals = phi_P_z_vals * z_P_y[gauss_indices, node_indices]
        if is_3d:
            phi_P_z_sparse_vals = phi_P_z_vals * z_P_z[gauss_indices, node_indices]
        else:
            phi_P_z_sparse_vals = None

    phi_P_x_nonzerovalue_data = phi_P_x_vals
    phi_P_y_nonzerovalue_data = phi_P_y_vals

    if is_3d:
        phi_P_z_nonzerovalue_data = phi_P_z_sparse_vals
    else:
        phi_P_z_nonzerovalue_data = np.array([])

    # Get dimensions
    n_gauss = x_G.shape[0]

    print(
        f"Sparse computation: {len(gauss_indices)} non-zero interactions out of {n_gauss * x_nodes.shape[0]} total",
        flush=True,
    )

    # Extract all relevant H values for valid pairs
    if use_sparse_h_matrices:
        # Sparse mode: H matrices are already sparse (n_nnz, n_dim)
        H_vals = H  # Already sparse
        H_T_vals = H_T
        H_P_x_vals = H_P_x
        H_P_y_vals = H_P_y
        HT_P_x_vals = HT_P_x
        HT_P_y_vals = HT_P_y
    else:
        # Dense mode: extract from dense arrays using advanced indexing
        H_vals = H[gauss_indices, node_indices, :]  # Shape: (n_interactions, n_dim)
        H_T_vals = H_T[gauss_indices, node_indices, :]
        H_P_x_vals = H_P_x[gauss_indices, node_indices, :]
        H_P_y_vals = H_P_y[gauss_indices, node_indices, :]
        HT_P_x_vals = HT_P_x[gauss_indices, node_indices, :]
        HT_P_y_vals = HT_P_y[gauss_indices, node_indices, :]

    # Initialize M matrices (they come pre-initialized, but we'll reset them for consistency)
    M[:] = 0
    M_P_x[:] = 0
    M_P_y[:] = 0
    if is_3d:
        M_P_z[:] = 0

    # Compute all outer products at once (vectorized)
    # Shape: (n_interactions, n_dim, n_dim)
    print("Computing outer products for M matrices", flush=True)

    # Replace np.add.at (CPU fallback) with GPU-native scatter-sum via SpMV.
    # Build sparse selection matrix P of shape (n_gauss, n_nnz) once:
    #   P[gauss_indices[k], k] = 1
    # For each of the n_dim² columns, P @ col_vec performs a GPU-native SpMV
    # that accumulates contributions into the correct Gauss-point row.
    #
    # Use CSR (data, indices, indptr) format directly rather than COO
    # (data, (rows, cols)) to avoid a cuPyNumeric lexsort bug in legate_sparse
    # that corrupts index values for small n_nnz.
    # gauss_indices is non-decreasing (it comes from np.where on a 2D row-major
    # array), so indptr = cumsum of per-row counts.
    # Note: legate_sparse csr_array supports SpMV (csr @ 1d) but not SpMM
    # (csr @ 2d), so we keep the column-wise loop.
    n_nnz = len(gauss_indices)
    n_gauss_M = M.shape[0]
    n_dim = M.shape[1]          # 4 for 3D RKPM
    n_dim_sq = n_dim * n_dim
    counts_P = np.bincount(gauss_indices, minlength=n_gauss_M).astype(np.int64)
    indptr_P = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts_P)])
    col_indices_P = np.arange(n_nnz, dtype=np.int64)
    P = csr_array(
        (np.ones(n_nnz, dtype=phi_vals.dtype), col_indices_P, indptr_P),
        shape=(n_gauss_M, n_nnz),
    )

    def _scatter_sum(values_3d):
        """Scatter-sum (n_nnz, n_dim, n_dim) → (n_gauss, n_dim, n_dim) via SpMV loop."""
        flat = values_3d.reshape(n_nnz, n_dim_sq)
        out = np.zeros((n_gauss_M, n_dim_sq), dtype=values_3d.dtype)
        for col in range(n_dim_sq):
            out[:, col] = P @ flat[:, col]
        return out.reshape(n_gauss_M, n_dim, n_dim)

    # For M: H @ H_T * phi
    outer_HH = H_vals[:, :, np.newaxis] * H_T_vals[:, np.newaxis, :]
    weighted_HH = outer_HH * phi_vals[:, np.newaxis, np.newaxis]
    M[:] = _scatter_sum(weighted_HH)

    # For M_P_x: three terms
    # Term 1: H @ H_T * phi_P_x
    weighted_HH_phi_P_x = outer_HH * phi_P_x_vals[:, np.newaxis, np.newaxis]

    # Term 2: H_P_x @ H_T * phi
    outer_HPxH = H_P_x_vals[:, :, np.newaxis] * H_T_vals[:, np.newaxis, :]
    weighted_HPxH = outer_HPxH * phi_vals[:, np.newaxis, np.newaxis]

    # Term 3: H @ HT_P_x * phi
    outer_HHPx = H_vals[:, :, np.newaxis] * HT_P_x_vals[:, np.newaxis, :]
    weighted_HHPx = outer_HHPx * phi_vals[:, np.newaxis, np.newaxis]

    M_P_x[:] = _scatter_sum(weighted_HH_phi_P_x + weighted_HPxH + weighted_HHPx)

    # For M_P_y: three terms (similar structure)
    # Term 1: H @ H_T * phi_P_y
    weighted_HH_phi_P_y = outer_HH * phi_P_y_vals[:, np.newaxis, np.newaxis]

    # Term 2: H_P_y @ H_T * phi
    outer_HPyH = H_P_y_vals[:, :, np.newaxis] * H_T_vals[:, np.newaxis, :]
    weighted_HPyH = outer_HPyH * phi_vals[:, np.newaxis, np.newaxis]

    # Term 3: H @ HT_P_y * phi
    outer_HHPy = H_vals[:, :, np.newaxis] * HT_P_y_vals[:, np.newaxis, :]
    weighted_HHPy = outer_HHPy * phi_vals[:, np.newaxis, np.newaxis]

    M_P_y[:] = _scatter_sum(weighted_HH_phi_P_y + weighted_HPyH + weighted_HHPy)

    if is_3d:
        # Extract H_P_z and HT_P_z values
        if use_sparse_h_matrices:
            # Sparse mode: already sparse
            H_P_z_vals = H_P_z
            HT_P_z_vals = HT_P_z
        else:
            # Dense mode: extract from dense arrays
            H_P_z_vals = H_P_z[gauss_indices, node_indices, :]
            HT_P_z_vals = HT_P_z[gauss_indices, node_indices, :]

        # For M_P_z: three terms (similar structure)
        # Term 1: H @ H_T * phi_P_z
        weighted_HH_phi_P_z = outer_HH * phi_P_z_sparse_vals[:, np.newaxis, np.newaxis]

        # Term 2: H_P_z @ H_T * phi
        outer_HPzH = H_P_z_vals[:, :, np.newaxis] * H_T_vals[:, np.newaxis, :]
        weighted_HPzH = outer_HPzH * phi_vals[:, np.newaxis, np.newaxis]

        # Term 3: H @ HT_P_z * phi
        outer_HHPz = H_vals[:, :, np.newaxis] * HT_P_z_vals[:, np.newaxis, :]
        weighted_HHPz = outer_HHPz * phi_vals[:, np.newaxis, np.newaxis]

        M_P_z[:] = _scatter_sum(weighted_HH_phi_P_z + weighted_HPzH + weighted_HHPz)

    return (
        phi_nonzero_index_row,
        phi_nonzero_index_column,
        phi_nonzerovalue_data,
        phi_P_x_nonzerovalue_data,
        phi_P_y_nonzerovalue_data,
        phi_P_z_nonzerovalue_data,
        M,
        M_P_x,
        M_P_y,
        M_P_z,
    )
