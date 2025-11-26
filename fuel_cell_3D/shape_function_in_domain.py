import time
start_time = time.time()
import numpy as np
from numpy import sign

import matplotlib.pyplot as plt

from tqdm import tqdm

# Removed numba import - using vectorized NumPy instead

from scipy.sparse import csc_matrix, csr_matrix, bmat
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigs

from numpy.linalg import norm, eig


# Remove @jit decorator to allow vectorized numpy operations
def compute_phi_M(x_G, Gauss_grain_id, x_nodes, nodes_grain_id, a, M, M_P_x, M_P_y, num_interface_segments, interface_nodes, BxByCxCy, IM_RKPM, single_grain, M_P_z=None):
    """
    Vectorized computation of shape functions and moment matrices.
    This version minimizes loops and uses NumPy broadcasting for better performance.
    """
    
    if M_P_z is None:
        # Initialize with zeros or correct shape
        M_P_z = np.zeros_like(M)  # Adjust shape accordingly

    # Convert inputs to numpy arrays
    x_G = np.array(x_G, dtype=np.float64)
    x_nodes = np.array(x_nodes, dtype=np.float64)
    a = np.array(a, dtype=np.float64)
    nodes_grain_id = np.array(nodes_grain_id, dtype=np.int32)
    Gauss_grain_id = np.array(Gauss_grain_id, dtype=np.int32)
    
    n_G = x_G.shape[0]
    n_N = x_nodes.shape[0]
    n_dim_spatial = x_G.shape[1]  # 2 for 2D, 3 for 3D
    is_3d = (n_dim_spatial == 3)
    n_dim_basis = 3 if not is_3d else 4
    
    phi_nonzero_index_row = []
    phi_nonzero_index_column = []
    phi_nonzerovalue_data = []
    phi_P_x_nonzerovalue_data = []
    phi_P_y_nonzerovalue_data = []
    phi_P_z_nonzerovalue_data = []
    z = []
    z_P_x = []
    z_P_y = []
    z_P_z = []
    phipz = []
    
    if single_grain == 'False' and IM_RKPM == 'True':
        print('get in')
        
        # VECTORIZED distance computation to interface segments
        interface_nodes = np.array(interface_nodes, dtype=np.float64)
        BxByCxCy = np.array(BxByCxCy, dtype=np.float64)
        B = BxByCxCy[:, :2]  # (num_interface_segments, 2)
        C = BxByCxCy[:, 2:4]  # (num_interface_segments, 2)
        BC = C - B
        CB = -BC
        
        # Vectorized interface node checking
        interface_tolerance = 1e-10
        node_on_interface = np.any(
            (np.abs(x_nodes[:, None, 0] - interface_nodes[None, :, 0]) < interface_tolerance) &
            (np.abs(x_nodes[:, None, 1] - interface_nodes[None, :, 1]) < interface_tolerance),
            axis=1
        )  # Shape: (n_N,)
        node_not_on_interface = ~node_on_interface
        
        # Loop over Gauss points (still needed for distance computation per point)
        for i in range(n_G):
            """
            check the distance between point and segments, exact distance
            """
            # Vectorized distance computation for all segments
            BA = x_G[i, :2] - B  # (num_interface_segments, 2)
            CA = x_G[i, :2] - C  # (num_interface_segments, 2)

            BA_dot_BC = np.sum(BA * BC, axis=1)  # (num_interface_segments,)
            CA_dot_CB = np.sum(CA * CB, axis=1)

            sign_extension = BA_dot_BC * CA_dot_CB

            positive_mask = sign_extension > 0
            negative_mask = sign_extension < 0
            zero_mask = sign_extension == 0

            BC_norm = np.sqrt(np.sum(BC ** 2, axis=1))
            BA_dot_unit_BC = BA_dot_BC / (BC_norm + 1e-16)
            BC_unit = BC / (BC_norm[:, None] + 1e-16)
            BA_dot_unit_BC_times_unit_BC = BC_unit * BA_dot_unit_BC[:, None]

            dx_distance = np.zeros(num_interface_segments)
            
            # Case 1: Positive sign extension (perpendicular distance)
            BA_pos = BA - BA_dot_unit_BC_times_unit_BC
            dx_distance[positive_mask] = np.sqrt(np.sum(BA_pos[positive_mask]**2, axis=1))
            
            # Case 2: Negative sign extension (endpoint distance)
            norm_CA = np.sqrt(np.sum(CA**2, axis=1))
            norm_BA = np.sqrt(np.sum(BA**2, axis=1))
            dx_distance[negative_mask] = np.minimum(norm_CA[negative_mask], norm_BA[negative_mask])
            
            # Case 3: Zero sign extension (endpoint distance)
            dx_distance[zero_mask] = np.minimum(norm_CA[zero_mask], norm_BA[zero_mask])
            
            min_distance = np.min(dx_distance)
            min_index = np.argmin(dx_distance)
            
            # Find closest point on segment
            if positive_mask[min_index]:
                x_coor_min_point = BA_dot_unit_BC_times_unit_BC[min_index, 0] + B[min_index, 0]
                y_coor_min_point = BA_dot_unit_BC_times_unit_BC[min_index, 1] + B[min_index, 1]
            else:  # negative_mask or zero_mask
                if norm_CA[min_index] < norm_BA[min_index]:
                    x_coor_min_point = C[min_index, 0]
                    y_coor_min_point = C[min_index, 1]
                else:
                    x_coor_min_point = B[min_index, 0]
                    y_coor_min_point = B[min_index, 1]

            d_distance_dx = (x_G[i, 0] - x_coor_min_point) / (min_distance + 1e-16)
            d_distance_dy = (x_G[i, 1] - y_coor_min_point) / (min_distance + 1e-16)

            # Heaviside function computation
            heaviside_scaling_factor = 4.0e-7
            min_distance_mod = min_distance + 1.0e-15
            heaviside = np.tanh(min_distance_mod / heaviside_scaling_factor)
            sech2 = (1.0 / np.cosh(min_distance_mod / heaviside_scaling_factor)) ** 2
            heaviside_P_x = d_distance_dx / heaviside_scaling_factor * sech2
            heaviside_P_y = d_distance_dy / heaviside_scaling_factor * sech2

            # VECTORIZED computation for all nodes for this Gauss point
            # Compute distances to all nodes
            if is_3d:
                dx_all = x_G[i, 0] - x_nodes[:, 0]  # (n_N,)
                dy_all = x_G[i, 1] - x_nodes[:, 1]
                dz_all = x_G[i, 2] - x_nodes[:, 2]
                dist_to_node = np.sqrt(dx_all**2 + dy_all**2 + dz_all**2)
            else:
                dx_all = x_G[i, 0] - x_nodes[:, 0]  # (n_N,)
                dy_all = x_G[i, 1] - x_nodes[:, 1]
                dist_to_node = np.sqrt(dx_all**2 + dy_all**2)
            
            z_ij_all = dist_to_node / (a + 1e-16)  # (n_N,)
            
            # Vectorized z derivatives
            z_ij_P_x_all = dx_all / (a * z_ij_all * a + 2.220446049250313e-16)
            z_ij_P_y_all = dy_all / (a * z_ij_all * a + 2.220446049250313e-16)
            if is_3d:
                z_ij_P_z_all = dz_all / (a * z_ij_all * a + 2.220446049250313e-16)

            # Vectorized H matrix computation
            H_scaling_factor = 1.0e-6
            if is_3d:
                H_T_all = np.zeros((n_N, 4), dtype=np.float64)
                H_T_all[:, 0] = 1.0
                H_T_all[:, 1] = dx_all / H_scaling_factor
                H_T_all[:, 2] = dy_all / H_scaling_factor
                H_T_all[:, 3] = dz_all / H_scaling_factor
                HT_P_x = np.array([0, 1.0/H_scaling_factor, 0, 0], dtype=np.float64)
                HT_P_y = np.array([0, 0, 1.0/H_scaling_factor, 0], dtype=np.float64)
                HT_P_z = np.array([0, 0, 0, 1.0/H_scaling_factor], dtype=np.float64)
            else:
                H_T_all = np.zeros((n_N, 3), dtype=np.float64)
                H_T_all[:, 0] = 1.0
                H_T_all[:, 1] = dx_all / H_scaling_factor
                H_T_all[:, 2] = dy_all / H_scaling_factor
                HT_P_x = np.array([0, 1.0/H_scaling_factor, 0], dtype=np.float64)
                HT_P_y = np.array([0, 0, 1.0/H_scaling_factor], dtype=np.float64)
            
            H_all = H_T_all  # (n_N, n_dim_basis)
            
            # Vectorized shape function evaluation
            mask_01 = (z_ij_all >= 0) & (z_ij_all < 0.5)
            mask_051 = (z_ij_all >= 0.5) & (z_ij_all <= 1.0)
            mask_in_support = (z_ij_all >= 0) & (z_ij_all <= 1.0)
            
            phi_ij_all = np.zeros(n_N)
            phi_P_z_all = np.zeros(n_N)
            
            phi_ij_all[mask_01] = 2.0/3 - 4*z_ij_all[mask_01]**2 + 4*z_ij_all[mask_01]**3
            phi_P_z_all[mask_01] = -8.0*z_ij_all[mask_01] + 12.0*z_ij_all[mask_01]**2
            
            phi_ij_all[mask_051] = 4.0/3 - 4*z_ij_all[mask_051] + 4*z_ij_all[mask_051]**2 - (4.0/3)*z_ij_all[mask_051]**3
            phi_P_z_all[mask_051] = -4.0 + 8.0*z_ij_all[mask_051] - 4.0*z_ij_all[mask_051]**2
            
            # Vectorized grain matching
            grainid_match = (nodes_grain_id == Gauss_grain_id[i])
            
            # Combined mask for valid nodes
            valid_mask = mask_in_support & node_not_on_interface & grainid_match
            
            # Get indices of valid nodes
            valid_indices = np.where(valid_mask)[0]
            
            # Process valid nodes
            for j in valid_indices:
                phi_nonzero_index_row.append(i)
                phi_nonzero_index_column.append(j)
                phi_nonzerovalue_data.append(phi_ij_all[j] * heaviside)

                phi_P_x_ij = phi_P_z_all[j] * z_ij_P_x_all[j]
                phi_P_y_ij = phi_P_z_all[j] * z_ij_P_y_all[j]
                phi_P_x_nonzerovalue_data.append(phi_P_x_ij * heaviside + phi_ij_all[j] * heaviside_P_x)
                phi_P_y_nonzerovalue_data.append(phi_P_y_ij * heaviside + phi_ij_all[j] * heaviside_P_y)

                z.append(z_ij_all[j])
                z_P_x.append(z_ij_P_x_all[j])
                z_P_y.append(z_ij_P_y_all[j])
                phipz.append(phi_P_z_all[j])
                
                if is_3d:
                    phi_P_z_ij = phi_P_z_all[j] * z_ij_P_z_all[j]
                    phi_P_z_nonzerovalue_data.append(phi_P_z_ij)
                    z_P_z.append(z_ij_P_z_all[j])
                
                # Vectorized moment matrix update
                H = H_all[j]  # (n_dim_basis,)
                H_T = H_T_all[j]  # (n_dim_basis,)
                
                # Outer product for M update
                M[i] += np.outer(H, H_T) * phi_ij_all[j] * heaviside
                
                # M_P_x update
                M_P_x[i] += (np.outer(H, H_T) * (phi_P_x_ij * heaviside + phi_ij_all[j] * heaviside_P_x) +
                             np.outer(HT_P_x, H_T) * phi_ij_all[j] * heaviside +
                             np.outer(H, HT_P_x) * phi_ij_all[j] * heaviside)
                
                # M_P_y update
                M_P_y[i] += (np.outer(H, H_T) * (phi_P_y_ij * heaviside + phi_ij_all[j] * heaviside_P_y) +
                             np.outer(HT_P_y, H_T) * phi_ij_all[j] * heaviside +
                             np.outer(H, HT_P_y) * phi_ij_all[j] * heaviside)
                
                if is_3d:
                    M_P_z[i] += (np.outer(H, H_T) * phi_P_z_ij +
                                 np.outer(HT_P_z, H_T) * phi_ij_all[j] +
                                 np.outer(H, HT_P_z) * phi_ij_all[j])
                        
    else:
        # VECTORIZED single grain case (no interface distance computation needed)
        # Compute all Gauss point to node distances at once
        # x_G: (n_G, n_dim_spatial), x_nodes: (n_N, n_dim_spatial)
        
        for i in range(n_G):
            # Vectorized computation for all nodes for this Gauss point
            if is_3d:
                dx_all = x_G[i, 0] - x_nodes[:, 0]  # (n_N,)
                dy_all = x_G[i, 1] - x_nodes[:, 1]
                dz_all = x_G[i, 2] - x_nodes[:, 2]
                dist_to_node = np.sqrt(dx_all**2 + dy_all**2 + dz_all**2)
            else:
                dx_all = x_G[i, 0] - x_nodes[:, 0]  # (n_N,)
                dy_all = x_G[i, 1] - x_nodes[:, 1]
                dist_to_node = np.sqrt(dx_all**2 + dy_all**2)
            
            z_ij_all = dist_to_node / (a + 1e-16)  # (n_N,)
            
            # Vectorized z derivatives
            z_ij_P_x_all = dx_all / (a * z_ij_all * a + 2.220446049250313e-16)
            z_ij_P_y_all = dy_all / (a * z_ij_all * a + 2.220446049250313e-16)
            if is_3d:
                z_ij_P_z_all = dz_all / (a * z_ij_all * a + 2.220446049250313e-16)

            # Vectorized H matrix computation
            H_scaling_factor = 1.0e-6
            if is_3d:
                H_T_all = np.zeros((n_N, 4), dtype=np.float64)
                H_T_all[:, 0] = 1.0
                H_T_all[:, 1] = dx_all / H_scaling_factor
                H_T_all[:, 2] = dy_all / H_scaling_factor
                H_T_all[:, 3] = dz_all / H_scaling_factor
                HT_P_x = np.array([0, 1.0/H_scaling_factor, 0, 0], dtype=np.float64)
                HT_P_y = np.array([0, 0, 1.0/H_scaling_factor, 0], dtype=np.float64)
                HT_P_z = np.array([0, 0, 0, 1.0/H_scaling_factor], dtype=np.float64)
            else:
                H_T_all = np.zeros((n_N, 3), dtype=np.float64)
                H_T_all[:, 0] = 1.0
                H_T_all[:, 1] = dx_all / H_scaling_factor
                H_T_all[:, 2] = dy_all / H_scaling_factor
                HT_P_x = np.array([0, 1.0/H_scaling_factor, 0], dtype=np.float64)
                HT_P_y = np.array([0, 0, 1.0/H_scaling_factor], dtype=np.float64)
            
            H_all = H_T_all  # (n_N, n_dim_basis)
            
            # Vectorized shape function evaluation
            mask_01 = (z_ij_all >= 0) & (z_ij_all < 0.5)
            mask_051 = (z_ij_all >= 0.5) & (z_ij_all <= 1.0)
            mask_in_support = (z_ij_all >= 0) & (z_ij_all <= 1.0)
            
            phi_ij_all = np.zeros(n_N)
            phi_P_z_all = np.zeros(n_N)
            
            phi_ij_all[mask_01] = 2.0/3 - 4*z_ij_all[mask_01]**2 + 4*z_ij_all[mask_01]**3
            phi_P_z_all[mask_01] = -8.0*z_ij_all[mask_01] + 12.0*z_ij_all[mask_01]**2
            
            phi_ij_all[mask_051] = 4.0/3 - 4*z_ij_all[mask_051] + 4*z_ij_all[mask_051]**2 - (4.0/3)*z_ij_all[mask_051]**3
            phi_P_z_all[mask_051] = -4.0 + 8.0*z_ij_all[mask_051] - 4.0*z_ij_all[mask_051]**2
            
            # Get indices of nodes in support
            valid_indices = np.where(mask_in_support)[0]
            
            # Process valid nodes
            for j in valid_indices:
                phi_nonzerovalue_data.append(phi_ij_all[j])
                phi_nonzero_index_row.append(i)
                phi_nonzero_index_column.append(j)
                
                phi_P_x_ij = phi_P_z_all[j] * z_ij_P_x_all[j]
                phi_P_y_ij = phi_P_z_all[j] * z_ij_P_y_all[j]
                
                phi_P_x_nonzerovalue_data.append(phi_P_x_ij)
                phi_P_y_nonzerovalue_data.append(phi_P_y_ij)
                z.append(z_ij_all[j])
                z_P_x.append(z_ij_P_x_all[j])
                z_P_y.append(z_ij_P_y_all[j])
                phipz.append(phi_P_z_all[j])
                
                if is_3d:
                    phi_P_z_ij = phi_P_z_all[j] * z_ij_P_z_all[j]
                    phi_P_z_nonzerovalue_data.append(phi_P_z_ij)
                    z_P_z.append(z_ij_P_z_all[j])
                
                # Vectorized moment matrix update
                H = H_all[j]  # (n_dim_basis,)
                H_T = H_T_all[j]  # (n_dim_basis,)
                
                # Outer product for M update
                M[i] += np.outer(H, H_T) * phi_ij_all[j]
                
                # M_P_x update
                M_P_x[i] += (np.outer(H, H_T) * phi_P_x_ij +
                             np.outer(HT_P_x, H_T) * phi_ij_all[j] +
                             np.outer(H, HT_P_x) * phi_ij_all[j])
                
                # M_P_y update
                M_P_y[i] += (np.outer(H, H_T) * phi_P_y_ij +
                             np.outer(HT_P_y, H_T) * phi_ij_all[j] +
                             np.outer(H, HT_P_y) * phi_ij_all[j])
                
                if is_3d:
                    M_P_z[i] += (np.outer(H, H_T) * phi_P_z_ij +
                                 np.outer(HT_P_z, H_T) * phi_ij_all[j] +
                                 np.outer(H, HT_P_z) * phi_ij_all[j])

        save_distance_function = np.array([1.0], dtype=np.float64)
        save_distance_function_dx = np.array([1.0], dtype=np.float64)
        save_distance_function_dy = np.array([1.0], dtype=np.float64)
        save_point_D_coor = np.array([1.0], dtype=np.float64)
    
    return phi_nonzero_index_row, phi_nonzero_index_column, phi_nonzerovalue_data,phi_P_x_nonzerovalue_data, phi_P_y_nonzerovalue_data, phi_P_z_nonzerovalue_data, M, M_P_x, M_P_y, M_P_z
    # return save_point_D_coor, save_distance_function,save_distance_function_dx,save_distance_function_dy, phi_nonzero_index_row, phi_nonzero_index_column, phi_nonzerovalue_data,phi_P_x_nonzerovalue_data, phi_P_y_nonzerovalue_data, M, M_P_x, M_P_y


# @jit  # this is taking so long time, we are vectorizing this part
def shape_grad_shape_func(x_G,x_nodes, num_non_zero_phi_a,HT0, M, M_P_x, M_P_y, differential_method, HT1, HT2, phi_nonzerovalue_data,phi_P_x_nonzerovalue_data,phi_P_y_nonzerovalue_data, phi_nonzero_index_row, phi_nonzero_index_column, det_J_time_weight, IM_RKPM, M_P_z=None, HT3 = None,phi_P_z_nonzerovalue_data=None):
    """
    Vectorized shape function and gradient computation.
    Eliminates the loop over num_non_zero_phi_a for better performance.
    """
    
    # Convert inputs to numpy arrays for vectorization
    phi_nonzero_index_row = np.array(phi_nonzero_index_row, dtype=np.int32)
    phi_nonzero_index_column = np.array(phi_nonzero_index_column, dtype=np.int32)
    phi_nonzerovalue_data = np.array(phi_nonzerovalue_data, dtype=np.float64)
    phi_P_x_nonzerovalue_data = np.array(phi_P_x_nonzerovalue_data, dtype=np.float64)
    phi_P_y_nonzerovalue_data = np.array(phi_P_y_nonzerovalue_data, dtype=np.float64)
    det_J_time_weight = np.array(det_J_time_weight, dtype=np.float64)
    HT0 = np.array(HT0, dtype=np.float64)
    x_G = np.array(x_G, dtype=np.float64)
    x_nodes = np.array(x_nodes, dtype=np.float64)
    
    # Get indices for all non-zero entries
    i_indices = phi_nonzero_index_row
    j_indices = phi_nonzero_index_column
    
    # Determine dimensionality (2D or 3D)
    H_scaling_factor = 1.0e-6
    n_dim = np.shape(M)[1]  # 3 for 2D, 4 for 3D
    is_3d = (n_dim == 4)
    
    # Vectorized computation of H matrices for all entries
    x_I = x_nodes[j_indices]  # Shape: (num_non_zero_phi_a, 2 or 3)
    x_G_selected = x_G[i_indices]  # Shape: (num_non_zero_phi_a, 2 or 3)
    
    # Compute H_T for all entries at once
    H_T_vectorized = np.zeros((num_non_zero_phi_a, n_dim), dtype=np.float64)
    H_T_vectorized[:, 0] = 1.0
    H_T_vectorized[:, 1] = (x_G_selected[:, 0] - x_I[:, 0]) / H_scaling_factor
    H_T_vectorized[:, 2] = (x_G_selected[:, 1] - x_I[:, 1]) / H_scaling_factor
    if is_3d:
        H_T_vectorized[:, 3] = (x_G_selected[:, 2] - x_I[:, 2]) / H_scaling_factor
    
    # H is transpose of H_T (for this use case, they're the same)
    H_vectorized = H_T_vectorized  # Shape: (num_non_zero_phi_a, n_dim)
    
    # Vectorized computation of H derivatives
    if is_3d:
        HT_P_x = np.array([0, 1.0/H_scaling_factor, 0, 0], dtype=np.float64)
        HT_P_y = np.array([0, 0, 1.0/H_scaling_factor, 0], dtype=np.float64)
        HT_P_z = np.array([0, 0, 0, 1.0/H_scaling_factor], dtype=np.float64)
        H_P_z = HT_P_z
        phi_P_z_nonzerovalue_data = np.array(phi_P_z_nonzerovalue_data, dtype=np.float64)
        HT3 = np.array(HT3, dtype=np.float64)
    else:
        HT_P_x = np.array([0, 1.0/H_scaling_factor, 0], dtype=np.float64)
        HT_P_y = np.array([0, 0, 1.0/H_scaling_factor], dtype=np.float64)
    
    H_P_x = HT_P_x
    H_P_y = HT_P_y
    
    # Get M matrices for all relevant Gauss points
    M_selected = M[i_indices]  # Shape: (num_non_zero_phi_a, n_dim, n_dim)
    M_inv_selected = np.linalg.inv(M_selected.astype(np.float64))  # Shape: (num_non_zero_phi_a, n_dim, n_dim)
    
    # Vectorized shape function computation: HT0 @ M_inv @ H for all entries
    # Using einsum for efficient computation
    HT0_M_inv = np.einsum('i,jik->jk', HT0, M_inv_selected)  # Shape: (num_non_zero_phi_a, n_dim)
    shape_func_values = np.einsum('ij,ij->i', HT0_M_inv, H_vectorized) * phi_nonzerovalue_data
    
    # Vectorized gradient computation
    if differential_method == 'implicite' and IM_RKPM == 'False':
        HT1 = np.array(HT1, dtype=np.float64)
        HT2 = np.array(HT2, dtype=np.float64)
        
        HT1_M_inv = np.einsum('i,jik->jk', HT1, M_inv_selected)
        HT2_M_inv = np.einsum('i,jik->jk', HT2, M_inv_selected)
        
        grad_shape_func_x_values = np.einsum('ij,ij->i', HT1_M_inv, H_vectorized) * phi_nonzerovalue_data
        grad_shape_func_y_values = np.einsum('ij,ij->i', HT2_M_inv, H_vectorized) * phi_nonzerovalue_data
        
        if is_3d:
            HT3_M_inv = np.einsum('i,jik->jk', HT3, M_inv_selected)
            grad_shape_func_z_values = np.einsum('ij,ij->i', HT3_M_inv, H_vectorized) * phi_nonzerovalue_data
        
    else:  # differential_method == 'direct' or IM_RKPM == 'True'
        M_P_x_selected = M_P_x[i_indices]  # Shape: (num_non_zero_phi_a, n_dim, n_dim)
        M_P_y_selected = M_P_y[i_indices]  # Shape: (num_non_zero_phi_a, n_dim, n_dim)
        
        # Vectorized computation of M_inv derivatives: M_inv_P_x = -M_inv @ M_P_x @ M_inv
        M_inv_M_P_x = np.matmul(M_inv_selected, M_P_x_selected.astype(np.float64))
        M_inv_P_x_selected = -np.matmul(M_inv_M_P_x, M_inv_selected)
        
        M_inv_M_P_y = np.matmul(M_inv_selected, M_P_y_selected.astype(np.float64))
        M_inv_P_y_selected = -np.matmul(M_inv_M_P_y, M_inv_selected)
        
        # Vectorized gradient computation with three terms
        # Term 1: HT0 @ M_inv @ H * phi_P_x
        term1_x = np.einsum('ij,ij->i', HT0_M_inv, H_vectorized) * phi_P_x_nonzerovalue_data
        term1_y = np.einsum('ij,ij->i', HT0_M_inv, H_vectorized) * phi_P_y_nonzerovalue_data
        
        # Term 2: HT0 @ M_inv_P_x @ H * phi
        HT0_M_inv_P_x = np.einsum('i,jik->jk', HT0, M_inv_P_x_selected)
        HT0_M_inv_P_y = np.einsum('i,jik->jk', HT0, M_inv_P_y_selected)
        term2_x = np.einsum('ij,ij->i', HT0_M_inv_P_x, H_vectorized) * phi_nonzerovalue_data
        term2_y = np.einsum('ij,ij->i', HT0_M_inv_P_y, H_vectorized) * phi_nonzerovalue_data
        
        # Term 3: HT0 @ M_inv @ H_P_x * phi
        term3_x = np.einsum('ij,j->i', HT0_M_inv, H_P_x) * phi_nonzerovalue_data
        term3_y = np.einsum('ij,j->i', HT0_M_inv, H_P_y) * phi_nonzerovalue_data
        
        grad_shape_func_x_values = term1_x + term2_x + term3_x
        grad_shape_func_y_values = term1_y + term2_y + term3_y
        
        if is_3d:
            M_P_z_selected = M_P_z[i_indices]
            M_inv_M_P_z = np.matmul(M_inv_selected, M_P_z_selected.astype(np.float64))
            M_inv_P_z_selected = -np.matmul(M_inv_M_P_z, M_inv_selected)
            
            term1_z = np.einsum('ij,ij->i', HT0_M_inv, H_vectorized) * phi_P_z_nonzerovalue_data
            HT0_M_inv_P_z = np.einsum('i,jik->jk', HT0, M_inv_P_z_selected)
            term2_z = np.einsum('ij,ij->i', HT0_M_inv_P_z, H_vectorized) * phi_nonzerovalue_data
            term3_z = np.einsum('ij,j->i', HT0_M_inv, H_P_z) * phi_nonzerovalue_data
            
            grad_shape_func_z_values = term1_z + term2_z + term3_z
    
    # Multiply by det_J_time_weight
    det_J_selected = det_J_time_weight[i_indices]
    shape_func_times_det_J_time_weight_values = shape_func_values * det_J_selected
    grad_shape_func_x_times_det_J_time_weight_values = grad_shape_func_x_values * det_J_selected
    grad_shape_func_y_times_det_J_time_weight_values = grad_shape_func_y_values * det_J_selected
    
    # Prepare output
    grad_shape_func_z_value = []
    grad_shape_func_z_times_det_J_time_weight_value = []
    
    if is_3d:
        grad_shape_func_z_times_det_J_time_weight_values = grad_shape_func_z_values * det_J_selected
        grad_shape_func_z_value = grad_shape_func_z_values.tolist()
        grad_shape_func_z_times_det_J_time_weight_value = grad_shape_func_z_times_det_J_time_weight_values.tolist()
    
    # Convert to lists to match original output format
    return (
        shape_func_values.tolist(),
        shape_func_times_det_J_time_weight_values.tolist(),
        grad_shape_func_x_values.tolist(),
        grad_shape_func_y_values.tolist(),
        grad_shape_func_z_value,
        grad_shape_func_x_times_det_J_time_weight_values.tolist(),
        grad_shape_func_y_times_det_J_time_weight_values.tolist(),
        grad_shape_func_z_times_det_J_time_weight_value
    )

def shape_func_n_nodes_by_n_nodes(x_G,x_nodes, num_non_zero_phi_a,HT0, M, phi_nonzerovalue_data,phi_nonzero_index_row, phi_nonzero_index_column):
    shape_func_value = []

    for ii in range(num_non_zero_phi_a):
        i = int(phi_nonzero_index_row[ii])
        j = int(phi_nonzero_index_column[ii])

        # compute the shape function and the gradient of shape function
        x_I = x_nodes[j]

        H_sacling_factor = 1.0e-6
        if np.shape(M)[1] == 3:
            H_T = np.array([1, (x_G[i][0]-x_I[0])/H_sacling_factor, (x_G[i][1]-x_I[1])/H_sacling_factor],dtype=np.float64)
        if np.shape(M)[1] == 4:
            H_T = np.array([1, (x_G[i][0]-x_I[0])/H_sacling_factor, (x_G[i][1]-x_I[1])/H_sacling_factor, (x_G[i][2]-x_I[2])/H_sacling_factor],dtype=np.float64)
        
        H = np.transpose(H_T)

        shape_func_ij = np.dot((np.dot((HT0).astype(np.float64), (np.linalg.inv(M[i])).astype(np.float64))).astype(np.float64), H.astype(np.float64))*phi_nonzerovalue_data[ii]

        shape_func_value.append(shape_func_ij)
       
    return shape_func_value


