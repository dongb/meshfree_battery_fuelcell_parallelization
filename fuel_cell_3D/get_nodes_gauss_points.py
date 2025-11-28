from common import np, use_cupy


def _to_hashable(arr):
    """Convert array to hashable tuple for set operations."""
    if use_cupy and hasattr(arr, 'get'):
        arr = arr.get()
    if isinstance(arr, tuple):
        return arr
    if hasattr(arr, 'tolist'):
        return tuple(arr.tolist())
    return tuple(arr)



def get_x_nodes_fuel_cell_3d_toy_image(x_min,x_max,y_min,y_max,z_min, z_max, num_pixels_xyz, img_):
    """
    Vectorized version of get_x_nodes_fuel_cell_3d_toy_image.
    
    Generates nodes, Gauss points, and interface information for 3D fuel cell simulation.
    
    Parameters:
    -----------
    x_min, x_max : float
        X-axis domain bounds
    y_min, y_max : float
        Y-axis domain bounds
    z_min, z_max : float
        Z-axis domain bounds
    num_pixels_xyz : list
        Number of pixels along each axis [nx, ny, nz]
    img_ : ndarray
        3D image array with phase labels (0: pore, 1: electrode, 2: electrolyte)
    
    Returns:
    --------
    Tuple of node lists, cell nodes, and interface information
    """
    
    num_pixels_x, num_pixels_y, num_pixels_z = num_pixels_xyz
    
    # Calculate pixel sizes
    dx = (x_max - x_min) / num_pixels_x
    dy = (y_max - y_min) / num_pixels_y
    dz = (z_max - z_min) / num_pixels_z
    
    # Pre-generate coordinate arrays
    x_coords = np.arange(num_pixels_x + 1) * dx + x_min
    y_coords = np.arange(num_pixels_y + 1) * dy + y_min
    z_coords = np.arange(num_pixels_z + 1) * dz + z_min
    
    # Create meshgrid for all possible node positions
    # We'll use sets to track unique nodes for each phase
    nodes_electrolyte_set = set()
    nodes_electrode_set = set()
    nodes_pore_set = set()
    nodes_mechanical_set = set()
    
    # Cell nodes storage
    cell_nodes_electrolyte_x = []
    cell_nodes_electrolyte_y = []
    cell_nodes_electrolyte_z = []
    cell_nodes_electrode_x = []
    cell_nodes_electrode_y = []
    cell_nodes_electrode_z = []
    cell_nodes_pore_x = []
    cell_nodes_pore_y = []
    cell_nodes_pore_z = []
    
    # Boundary cell nodes
    cell_nodes_left_electrolyte_x = []
    cell_nodes_left_electrolyte_z = []
    cell_nodes_right_electrode_x = []
    cell_nodes_right_electrode_z = []
    cell_nodes_right_pore_x = []
    cell_nodes_right_pore_z = []
    
    cell_nodes_fixed_x = []
    cell_nodes_fixed_z = []
    
    # Interface nodes
    cell_nodes_interface_electrode_electrolyte_x = []
    cell_nodes_interface_electrode_electrolyte_y = []
    cell_nodes_interface_electrode_electrolyte_z = []
    cell_nodes_interface_electrode_pore_x = []
    cell_nodes_interface_electrode_pore_y = []
    cell_nodes_interface_electrode_pore_z = []
    
    segments_source_set = set()  # Use set to avoid duplicates
    
    # Get indices of all pixels for each phase - vectorized
    electrolyte_pixels = np.argwhere(img_ == 2)
    electrode_pixels = np.argwhere(img_ == 1)
    pore_pixels = np.argwhere(img_ == 0)
    non_zero_pixels = np.argwhere(img_ != 0)
    
    # Process mechanical nodes (all non-zero pixels) - vectorized
    for idx in non_zero_pixels:
        i, j, k = idx
        # 8 corners of the voxel
        corners = [
            (i, j, k), (i+1, j, k), (i+1, j+1, k), (i, j+1, k),
            (i, j, k+1), (i+1, j, k+1), (i+1, j+1, k+1), (i, j+1, k+1)
        ]
        for corner in corners:
            ci, cj, ck = corner
            node = _to_hashable(np.array([x_coords[ci], y_coords[cj], z_coords[ck]]))
            nodes_mechanical_set.add(node)
    
    # Process fixed nodes (j==0 and k==0) - vectorized
    fixed_mask = (non_zero_pixels[:, 1] == 0) & (non_zero_pixels[:, 2] == 0)
    fixed_pixels = non_zero_pixels[fixed_mask]
    for idx in fixed_pixels:
        i = idx[0]
        cell_nodes_fixed_x.append([x_coords[i], x_coords[i+1], x_coords[i+1], x_coords[i]])
        cell_nodes_fixed_z.append([z_coords[0], z_coords[0], z_coords[1], z_coords[1]])
    
    # Edge adjacency patterns for triple junction detection (12 edges of a voxel)
    edge_patterns = [
        # Bottom face (z=k)
        (np.array([[0,-1,0],[0,-1,-1],[0,0,-1]]), [(0,0,0), (1,0,0)]),  # edge 1
        (np.array([[1,0,0],[1,0,-1],[0,0,-1]]), [(1,0,0), (1,1,0)]),     # edge 2
        (np.array([[0,1,0],[0,1,-1],[0,0,-1]]), [(0,1,0), (1,1,0)]),     # edge 3
        (np.array([[-1,0,0],[-1,0,-1],[0,0,-1]]), [(0,0,0), (0,1,0)]),   # edge 4
        # Top face (z=k+1)
        (np.array([[0,-1,0],[0,-1,1],[0,0,1]]), [(0,0,1), (1,0,1)]),     # edge 5
        (np.array([[1,0,0],[1,0,1],[0,0,1]]), [(1,0,1), (1,1,1)]),       # edge 6
        (np.array([[0,1,0],[0,1,1],[0,0,1]]), [(0,1,1), (1,1,1)]),       # edge 7
        (np.array([[-1,0,0],[-1,0,1],[0,0,1]]), [(0,0,1), (0,1,1)]),     # edge 8
        # Vertical edges
        (np.array([[1,-1,0],[0,-1,0],[1,0,0]]), [(1,0,0), (1,0,1)]),     # edge 9
        (np.array([[1,0,0],[0,1,0],[1,1,0]]), [(1,1,0), (1,1,1)]),       # edge 10
        (np.array([[0,1,0],[-1,0,0],[-1,1,0]]), [(0,1,0), (0,1,1)]),     # edge 11
        (np.array([[0,-1,0],[-1,0,0],[-1,-1,0]]), [(0,0,0), (0,0,1)])    # edge 12
    ]
    
    # Process electrolyte pixels
    for idx in electrolyte_pixels:
        i, j, k = idx
        
        # Add cell nodes
        x_cell = [x_coords[i], x_coords[i+1], x_coords[i+1], x_coords[i],
                  x_coords[i], x_coords[i+1], x_coords[i+1], x_coords[i]]
        y_cell = [y_coords[j], y_coords[j], y_coords[j+1], y_coords[j+1],
                  y_coords[j], y_coords[j], y_coords[j+1], y_coords[j+1]]
        z_cell = [z_coords[k], z_coords[k], z_coords[k], z_coords[k],
                  z_coords[k+1], z_coords[k+1], z_coords[k+1], z_coords[k+1]]
        
        cell_nodes_electrolyte_x.append(x_cell)
        cell_nodes_electrolyte_y.append(y_cell)
        cell_nodes_electrolyte_z.append(z_cell)
        
        # Boundary cells
        if j == 0:
            cell_nodes_left_electrolyte_x.append([x_coords[i], x_coords[i+1], x_coords[i+1], x_coords[i]])
            cell_nodes_left_electrolyte_z.append([z_coords[k], z_coords[k], z_coords[k+1], z_coords[k+1]])
        
        # Check triple junctions for each edge
        for edge_adj, edge_coords in edge_patterns:
            adjacent_pixel_index = np.array([[i, j, k]]) + edge_adj
            
            # Filter valid indices
            filter_mask = (
                np.all(adjacent_pixel_index >= 0, axis=1) &
                (adjacent_pixel_index[:, 0] < num_pixels_x) &
                (adjacent_pixel_index[:, 1] < num_pixels_y) &
                (adjacent_pixel_index[:, 2] < num_pixels_z)
            )
            
            if np.any(filter_mask):
                filtered_adjacent = adjacent_pixel_index[filter_mask]
                unique_ids = np.unique(img_[tuple(filtered_adjacent.T)])
                
                # Triple junction: electrolyte (2) with both pore (0) and electrode (1)
                if (0 in unique_ids and 1 in unique_ids):
                    # Calculate edge endpoints
                    c1, c2 = edge_coords
                    segment = _to_hashable(np.array([x_coords[i+c1[0]], y_coords[j+c1[1]], z_coords[k+c1[2]],
                                                      x_coords[i+c2[0]], y_coords[j+c2[1]], z_coords[k+c2[2]]]))
                    segments_source_set.add(segment)
        
        # Add nodes
        corners = [
            (i, j, k), (i+1, j, k), (i+1, j+1, k), (i, j+1, k),
            (i, j, k+1), (i+1, j, k+1), (i+1, j+1, k+1), (i, j+1, k+1)
        ]
        for corner in corners:
            ci, cj, ck = corner
            node = _to_hashable(np.array([x_coords[ci], y_coords[cj], z_coords[ck]]))
            nodes_electrolyte_set.add(node)
        
        # Check interface with electrode (6 faces)
        face_checks = [
            ((-1, 0, 0), [(i, j, k), (i, j+1, k), (i, j+1, k+1), (i, j, k+1)]),       # -x face
            ((1, 0, 0), [(i+1, j, k), (i+1, j+1, k), (i+1, j+1, k+1), (i+1, j, k+1)]), # +x face
            ((0, -1, 0), [(i, j, k), (i+1, j, k), (i+1, j, k+1), (i, j, k+1)]),       # -y face
            ((0, 1, 0), [(i, j+1, k), (i+1, j+1, k), (i+1, j+1, k+1), (i, j+1, k+1)]), # +y face
            ((0, 0, -1), [(i+1, j, k), (i+1, j+1, k), (i, j+1, k), (i, j, k)]),       # -z face
            ((0, 0, 1), [(i+1, j, k+1), (i+1, j+1, k+1), (i, j+1, k+1), (i, j, k+1)])  # +z face
        ]
        
        for offset, face_corners in face_checks:
            ni, nj, nk = i + offset[0], j + offset[1], k + offset[2]
            if 0 <= ni < num_pixels_x and 0 <= nj < num_pixels_y and 0 <= nk < num_pixels_z:
                if img_[ni, nj, nk] == 1:  # Adjacent to electrode
                    face_x = [x_coords[fc[0]] for fc in face_corners]
                    face_y = [y_coords[fc[1]] for fc in face_corners]
                    face_z = [z_coords[fc[2]] for fc in face_corners]
                    cell_nodes_interface_electrode_electrolyte_x.append(face_x)
                    cell_nodes_interface_electrode_electrolyte_y.append(face_y)
                    cell_nodes_interface_electrode_electrolyte_z.append(face_z)
    
    # Process electrode pixels
    for idx in electrode_pixels:
        i, j, k = idx
        
        # Add cell nodes
        x_cell = [x_coords[i], x_coords[i+1], x_coords[i+1], x_coords[i],
                  x_coords[i], x_coords[i+1], x_coords[i+1], x_coords[i]]
        y_cell = [y_coords[j], y_coords[j], y_coords[j+1], y_coords[j+1],
                  y_coords[j], y_coords[j], y_coords[j+1], y_coords[j+1]]
        z_cell = [z_coords[k], z_coords[k], z_coords[k], z_coords[k],
                  z_coords[k+1], z_coords[k+1], z_coords[k+1], z_coords[k+1]]
        
        cell_nodes_electrode_x.append(x_cell)
        cell_nodes_electrode_y.append(y_cell)
        cell_nodes_electrode_z.append(z_cell)
        
        # Right boundary cells
        if j + 1 == num_pixels_y:
            cell_nodes_right_electrode_x.append([x_coords[i], x_coords[i+1], x_coords[i+1], x_coords[i]])
            cell_nodes_right_electrode_z.append([z_coords[k], z_coords[k], z_coords[k+1], z_coords[k+1]])
        
        # Add nodes
        corners = [
            (i, j, k), (i+1, j, k), (i+1, j+1, k), (i, j+1, k),
            (i, j, k+1), (i+1, j, k+1), (i+1, j+1, k+1), (i, j+1, k+1)
        ]
        for corner in corners:
            ci, cj, ck = corner
            node = _to_hashable(np.array([x_coords[ci], y_coords[cj], z_coords[ck]]))
            nodes_electrode_set.add(node)
        
        # Check interface with pore (6 faces)
        for offset, face_corners in face_checks:
            ni, nj, nk = i + offset[0], j + offset[1], k + offset[2]
            if 0 <= ni < num_pixels_x and 0 <= nj < num_pixels_y and 0 <= nk < num_pixels_z:
                if img_[ni, nj, nk] == 0:  # Adjacent to pore
                    face_x = [x_coords[fc[0]] for fc in face_corners]
                    face_y = [y_coords[fc[1]] for fc in face_corners]
                    face_z = [z_coords[fc[2]] for fc in face_corners]
                    cell_nodes_interface_electrode_pore_x.append(face_x)
                    cell_nodes_interface_electrode_pore_y.append(face_y)
                    cell_nodes_interface_electrode_pore_z.append(face_z)
    
    # Process pore pixels
    for idx in pore_pixels:
        i, j, k = idx
        
        # Add cell nodes
        x_cell = [x_coords[i], x_coords[i+1], x_coords[i+1], x_coords[i],
                  x_coords[i], x_coords[i+1], x_coords[i+1], x_coords[i]]
        y_cell = [y_coords[j], y_coords[j], y_coords[j+1], y_coords[j+1],
                  y_coords[j], y_coords[j], y_coords[j+1], y_coords[j+1]]
        z_cell = [z_coords[k], z_coords[k], z_coords[k], z_coords[k],
                  z_coords[k+1], z_coords[k+1], z_coords[k+1], z_coords[k+1]]
        
        cell_nodes_pore_x.append(x_cell)
        cell_nodes_pore_y.append(y_cell)
        cell_nodes_pore_z.append(z_cell)
        
        # Right boundary cells
        if j + 1 == num_pixels_y:
            cell_nodes_right_pore_x.append([x_coords[i], x_coords[i+1], x_coords[i+1], x_coords[i]])
            cell_nodes_right_pore_z.append([z_coords[k], z_coords[k], z_coords[k+1], z_coords[k+1]])
        
        # Add nodes
        corners = [
            (i, j, k), (i+1, j, k), (i+1, j+1, k), (i, j+1, k),
            (i, j, k+1), (i+1, j, k+1), (i+1, j+1, k+1), (i, j+1, k+1)
        ]
        for corner in corners:
            ci, cj, ck = corner
            node = _to_hashable(np.array([x_coords[ci], y_coords[cj], z_coords[ck]]))
            nodes_pore_set.add(node)
    
    # Convert sets to sorted lists
    x_nodes_mechanical = sorted(list(nodes_mechanical_set))
    x_nodes_electrolyte = sorted(list(nodes_electrolyte_set))
    x_nodes_electrode = sorted(list(nodes_electrode_set))
    x_nodes_pore = sorted(list(nodes_pore_set))
    segments_source = np.array(sorted(list(segments_source_set)))
    
    # Generate boundary node IDs
    nodes_id_left_electrolyte = []
    for idx, node in enumerate(x_nodes_electrolyte):
        if node[1] == y_min:  # j == 0
            nodes_id_left_electrolyte.append(idx)
    
    nodes_id_right_electrode = []
    for idx, node in enumerate(x_nodes_electrode):
        if node[1] == y_max:  # j == num_pixels_y
            nodes_id_right_electrode.append(idx)
    
    nodes_id_right_pore = []
    for idx, node in enumerate(x_nodes_pore):
        if node[1] == y_max:  # j == num_pixels_y
            nodes_id_right_pore.append(idx)
    
    return (x_nodes_mechanical, x_nodes_electrolyte, x_nodes_electrode, x_nodes_pore, 
            segments_source, cell_nodes_fixed_x, cell_nodes_fixed_z, 
            nodes_id_left_electrolyte, nodes_id_right_electrode, nodes_id_right_pore, 
            cell_nodes_electrolyte_x, cell_nodes_electrolyte_y, cell_nodes_electrolyte_z, 
            cell_nodes_electrode_x, cell_nodes_electrode_y, cell_nodes_electrode_z,
            cell_nodes_pore_x, cell_nodes_pore_y, cell_nodes_pore_z, 
            cell_nodes_left_electrolyte_x, cell_nodes_left_electrolyte_z, 
            cell_nodes_right_electrode_x, cell_nodes_right_electrode_z, 
            cell_nodes_right_pore_x, cell_nodes_right_pore_z,
            cell_nodes_interface_electrode_electrolyte_x,
            cell_nodes_interface_electrode_electrolyte_y,
            cell_nodes_interface_electrode_electrolyte_z,
            cell_nodes_interface_electrode_pore_x, 
            cell_nodes_interface_electrode_pore_y, 
            cell_nodes_interface_electrode_pore_z)



def x_G_and_def_J_time_weight_3d_fuelcell_domain(cell_nodes_x, cell_nodes_y, cell_nodes_z, 
                                                   x_G_domain, weight_G_domain):
    """
    Vectorized version of 3D Gauss point and Jacobian computation.
    
    Calculates Gauss point coordinates and Jacobian determinants for 3D hexahedral elements.
    
    Parameters:
    -----------
    cell_nodes_x, cell_nodes_y, cell_nodes_z : ndarray
        Coordinates of cell vertices (n_cells, 8)
    x_G_domain : list
        Gauss point locations in reference coordinates
    weight_G_domain : list
        Gauss quadrature weights
    
    Returns:
    --------
    x_G : list
        Physical coordinates of Gauss points
    det_J_time_weight : list
        Jacobian determinants times weights
    """
    
    cell_nodes_x = np.atleast_2d(cell_nodes_x)
    cell_nodes_y = np.atleast_2d(cell_nodes_y)
    cell_nodes_z = np.atleast_2d(cell_nodes_z)
    
    n_cells = cell_nodes_x.shape[0]
    n_gauss = len(x_G_domain)
    
    x_G_list = []
    y_G_list = []
    z_G_list = []
    det_J_time_weight = []
    
    # Convert Gauss points to array for vectorization
    xi_eta_zeta = np.array(x_G_domain)  # (n_gauss, 3)
    weights = np.array(weight_G_domain)
    
    # Shape functions: N_i(xi, eta, zeta)
    # For hexahedral element with 8 nodes
    for i in range(n_cells):
        x_ver = cell_nodes_x[i, :]
        y_ver = cell_nodes_y[i, :]
        z_ver = cell_nodes_z[i, :]
        
        for k in range(n_gauss):
            xi, eta, zeta = xi_eta_zeta[k]
            
            # Shape functions (8 nodes)
            N = np.array([
                (1 + xi) * (1 - eta) * (1 - zeta),
                (1 + xi) * (1 + eta) * (1 - zeta),
                (1 - xi) * (1 + eta) * (1 - zeta),
                (1 - xi) * (1 - eta) * (1 - zeta),
                (1 + xi) * (1 - eta) * (1 + zeta),
                (1 + xi) * (1 + eta) * (1 + zeta),
                (1 - xi) * (1 + eta) * (1 + zeta),
                (1 - xi) * (1 - eta) * (1 + zeta)
            ]) / 8.0
            
            # Gauss point coordinates
            x_G_k = np.dot(N, x_ver)
            y_G_k = np.dot(N, y_ver)
            z_G_k = np.dot(N, z_ver)
            x_G_list.append(x_G_k)
            y_G_list.append(y_G_k)
            z_G_list.append(z_G_k)
            
            # Shape function derivatives w.r.t. reference coordinates
            dN_dxi = np.array([
                (1 - eta) * (1 - zeta),
                (1 + eta) * (1 - zeta),
                -(1 + eta) * (1 - zeta),
                -(1 - eta) * (1 - zeta),
                (1 - eta) * (1 + zeta),
                (1 + eta) * (1 + zeta),
                -(1 + eta) * (1 + zeta),
                -(1 - eta) * (1 + zeta)
            ]) / 8.0
            
            dN_deta = np.array([
                -(1 + xi) * (1 - zeta),
                (1 + xi) * (1 - zeta),
                (1 - xi) * (1 - zeta),
                -(1 - xi) * (1 - zeta),
                -(1 + xi) * (1 + zeta),
                (1 + xi) * (1 + zeta),
                (1 - xi) * (1 + zeta),
                -(1 - xi) * (1 + zeta)
            ]) / 8.0
            
            dN_dzeta = np.array([
                -(1 + xi) * (1 - eta),
                -(1 + xi) * (1 + eta),
                -(1 - xi) * (1 + eta),
                -(1 - xi) * (1 - eta),
                (1 + xi) * (1 - eta),
                (1 + xi) * (1 + eta),
                (1 - xi) * (1 + eta),
                (1 - xi) * (1 - eta)
            ]) / 8.0
            
            # Jacobian matrix
            J11 = np.dot(dN_dxi, x_ver)
            J12 = np.dot(dN_deta, x_ver)
            J13 = np.dot(dN_dzeta, x_ver)
            J21 = np.dot(dN_dxi, y_ver)
            J22 = np.dot(dN_deta, y_ver)
            J23 = np.dot(dN_dzeta, y_ver)
            J31 = np.dot(dN_dxi, z_ver)
            J32 = np.dot(dN_deta, z_ver)
            J33 = np.dot(dN_dzeta, z_ver)
            
            J = np.array([[J11, J12, J13],
                         [J21, J22, J23],
                         [J31, J32, J33]])
            
            det_J_time_weight.append(np.linalg.det(J) * weights[k])
    
    # Stack coordinates into array
    if x_G_list:
        x_G = np.stack([np.array(x_G_list), np.array(y_G_list), np.array(z_G_list)], axis=1)
        det_J_time_weight = np.array(det_J_time_weight)
    else:
        x_G = np.empty((0, 3))
        det_J_time_weight = np.empty(0)
    
    return x_G, det_J_time_weight


def x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary(cell_nodes_boundary_x, cell_nodes_boundary_z, 
                                                            y_coords_on_boundary, x_G_domain, weight_G_domain):
    """
    Vectorized version of 2D boundary Gauss points in 3D space.
    
    Computes Gauss points on 2D boundaries (constant y-coordinate).
    
    Parameters:
    -----------
    cell_nodes_boundary_x, cell_nodes_boundary_z : ndarray
        Boundary cell node coordinates (n_cells, 4)
    y_coords_on_boundary : float
        Fixed y-coordinate of the boundary
    x_G_domain : list
        Gauss point locations in 2D reference coordinates
    weight_G_domain : list
        Gauss quadrature weights
    
    Returns:
    --------
    x_G : list
        Physical coordinates of boundary Gauss points
    det_J_time_weight : list
        Jacobian determinants times weights
    """
    
    cell_nodes_boundary_x = np.atleast_2d(cell_nodes_boundary_x)
    cell_nodes_boundary_z = np.atleast_2d(cell_nodes_boundary_z)
    
    n_cells = cell_nodes_boundary_x.shape[0]
    n_gauss = len(x_G_domain)
    
    x_G_list = []
    y_G_list = []
    z_G_list = []
    det_J_time_weight = []
    
    xi_eta = np.array(x_G_domain)  # (n_gauss, 2)
    weights = np.array(weight_G_domain)
    
    for i in range(n_cells):
        x_ver = cell_nodes_boundary_x[i, :]
        z_ver = cell_nodes_boundary_z[i, :]
        
        for k in range(n_gauss):
            xi, eta = xi_eta[k]
            
            # 2D shape functions for quadrilateral
            N = np.array([
                (1 - xi) * (1 - eta),
                (1 + xi) * (1 - eta),
                (1 + xi) * (1 + eta),
                (1 - xi) * (1 + eta)
            ]) / 4.0
            
            x_G_k = np.dot(N, x_ver)
            z_G_k = np.dot(N, z_ver)
            y_G_k = y_coords_on_boundary
            
            x_G_list.append(x_G_k)
            y_G_list.append(y_G_k)
            z_G_list.append(z_G_k)
            
            # Shape function derivatives
            dN_dxi = np.array([
                -(1 - eta),
                (1 - eta),
                (1 + eta),
                -(1 + eta)
            ]) / 4.0
            
            dN_deta = np.array([
                -(1 - xi),
                -(1 + xi),
                (1 + xi),
                (1 - xi)
            ]) / 4.0
            
            # Jacobian for 2D boundary
            J1 = np.dot(dN_dxi, x_ver)
            J2 = np.dot(dN_dxi, z_ver)
            J3 = np.dot(dN_deta, x_ver)
            J4 = np.dot(dN_deta, z_ver)
            
            det_J = J1 * J4 - J2 * J3
            det_J_time_weight.append(det_J * weights[k])
    
    # Stack coordinates into array
    if x_G_list:
        x_G = np.stack([np.array(x_G_list), np.array(y_G_list), np.array(z_G_list)], axis=1)
        det_J_time_weight = np.array(det_J_time_weight)
    else:
        x_G = np.empty((0, 3))
        det_J_time_weight = np.empty(0)
    
    return x_G, det_J_time_weight


def x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary_interface(cell_nodes_boundary_x, 
                                                                      cell_nodes_boundary_y, 
                                                                      cell_nodes_boundary_z, 
                                                                      x_G_domain, weight_G_domain):
    """
    Vectorized version of interface boundary Gauss points.
    
    Handles boundaries aligned with any coordinate plane (constant x, y, or z).
    
    Parameters:
    -----------
    cell_nodes_boundary_x, cell_nodes_boundary_y, cell_nodes_boundary_z : ndarray
        Boundary cell node coordinates (n_cells, 4)
    x_G_domain : list
        Gauss point locations in 2D reference coordinates
    weight_G_domain : list
        Gauss quadrature weights
    
    Returns:
    --------
    x_G : list
        Physical coordinates of boundary Gauss points
    det_J_time_weight : list
        Jacobian determinants times weights
    """
    
    cell_nodes_boundary_x = np.atleast_2d(cell_nodes_boundary_x)
    cell_nodes_boundary_y = np.atleast_2d(cell_nodes_boundary_y)
    cell_nodes_boundary_z = np.atleast_2d(cell_nodes_boundary_z)
    
    n_cells = cell_nodes_boundary_x.shape[0]
    n_gauss = len(x_G_domain)
    
    x_G_list = []
    y_G_list = []
    z_G_list = []
    det_J_time_weight = []
    
    xi_eta = np.array(x_G_domain)  # (n_gauss, 2)
    weights = np.array(weight_G_domain)
    
    for i in range(n_cells):
        x_ver = cell_nodes_boundary_x[i, :]
        y_ver = cell_nodes_boundary_y[i, :]
        z_ver = cell_nodes_boundary_z[i, :]
        
        # Determine which coordinate is constant
        x_const = np.allclose(x_ver, x_ver[0])
        y_const = np.allclose(y_ver, y_ver[0])
        z_const = np.allclose(z_ver, z_ver[0])
        
        for k in range(n_gauss):
            xi, eta = xi_eta[k]
            
            # 2D shape functions
            N = np.array([
                (1 - xi) * (1 - eta),
                (1 + xi) * (1 - eta),
                (1 + xi) * (1 + eta),
                (1 - xi) * (1 + eta)
            ]) / 4.0
            
            dN_dxi = np.array([
                -(1 - eta),
                (1 - eta),
                (1 + eta),
                -(1 + eta)
            ]) / 4.0
            
            dN_deta = np.array([
                -(1 - xi),
                -(1 + xi),
                (1 + xi),
                (1 - xi)
            ]) / 4.0
            
            if y_const:  # Constant y (x-z plane)
                x_G_k = np.dot(N, x_ver)
                y_G_k = y_ver[0]
                z_G_k = np.dot(N, z_ver)
                
                J1 = np.dot(dN_dxi, x_ver)
                J2 = np.dot(dN_dxi, z_ver)
                J3 = np.dot(dN_deta, x_ver)
                J4 = np.dot(dN_deta, z_ver)
                
            elif x_const:  # Constant x (y-z plane)
                x_G_k = x_ver[0]
                y_G_k = np.dot(N, y_ver)
                z_G_k = np.dot(N, z_ver)
                
                J1 = np.dot(dN_dxi, y_ver)
                J2 = np.dot(dN_dxi, z_ver)
                J3 = np.dot(dN_deta, y_ver)
                J4 = np.dot(dN_deta, z_ver)
                
            elif z_const:  # Constant z (x-y plane)
                x_G_k = np.dot(N, x_ver)
                y_G_k = np.dot(N, y_ver)
                z_G_k = z_ver[0]
                
                J1 = np.dot(dN_dxi, x_ver)
                J2 = np.dot(dN_dxi, y_ver)
                J3 = np.dot(dN_deta, x_ver)
                J4 = np.dot(dN_deta, y_ver)
            else:
                continue  # Skip if not a valid boundary
            
            x_G_list.append(x_G_k)
            y_G_list.append(y_G_k)
            z_G_list.append(z_G_k)
            det_J = J1 * J4 - J2 * J3
            det_J_time_weight.append(det_J * weights[k])
    
    # Stack coordinates into array
    if x_G_list:
        x_G = np.stack([np.array(x_G_list), np.array(y_G_list), np.array(z_G_list)], axis=1)
        det_J_time_weight = np.array(det_J_time_weight)
    else:
        x_G = np.empty((0, 3))
        det_J_time_weight = np.empty(0)
    
    return x_G, det_J_time_weight


def x_G_and_det_J_line_3d_fuelcell_1d_boundary(segments_source, x_G_line, weight_G_line):
    """
    VECTORIZED version of 1D line Gauss points in 3D space.
    
    Computes Gauss points along line segments (triple junctions).
    Removed @jit decorator and vectorized the computation.
    
    Parameters:
    -----------
    segments_source : ndarray
        Line segment endpoints (n_segments, 6) - [x1, y1, z1, x2, y2, z2]
    x_G_line : list or array
        Gauss point locations in 1D reference coordinate
    weight_G_line : list or array
        Gauss quadrature weights
    
    Returns:
    --------
    x_G_b_line : list
        Physical coordinates of line Gauss points
    det_J_b_time_weight_line : list
        Jacobian determinants times weights
    """
    x_G_line = np.array(x_G_line)
    weight_G_line = np.array(weight_G_line)
    
    n_segments = segments_source.shape[0]
    n_gauss = len(x_G_line)
    
    # Extract segment endpoints
    x1, y1, z1 = segments_source[:, 0], segments_source[:, 1], segments_source[:, 2]
    x2, y2, z2 = segments_source[:, 3], segments_source[:, 4], segments_source[:, 5]
    
    # Compute segment direction vectors
    dx = x2 - x1  # (n_segments,)
    dy = y2 - y1
    dz = z2 - z1
    
    # Compute segment length / 2 (Jacobian) for each segment
    # Use the maximum absolute difference to determine length
    abs_dx = np.abs(dx)
    abs_dy = np.abs(dy)
    abs_dz = np.abs(dz)
    
    # Find which direction has the largest variation
    max_vals = np.maximum(np.maximum(abs_dx, abs_dy), abs_dz)
    length_half = max_vals / 2  # (n_segments,)
    
    # Broadcast computation: for each segment, for each Gauss point
    # Shape: (n_segments, n_gauss)
    x1_expanded = x1[:, np.newaxis]  # (n_segments, 1)
    y1_expanded = y1[:, np.newaxis]
    z1_expanded = z1[:, np.newaxis]
    x2_expanded = x2[:, np.newaxis]
    y2_expanded = y2[:, np.newaxis]
    z2_expanded = z2[:, np.newaxis]
    
    dx_expanded = dx[:, np.newaxis]
    dy_expanded = dy[:, np.newaxis]
    dz_expanded = dz[:, np.newaxis]
    
    xi_expanded = x_G_line[np.newaxis, :]  # (1, n_gauss)
    weight_expanded = weight_G_line[np.newaxis, :]  # (1, n_gauss)
    
    # Map from reference [-1, 1] to physical coordinates
    # x_G = (x2 - x1)/2 * xi + (x2 + x1)/2
    x_G_k = dx_expanded / 2 * xi_expanded + (x2_expanded + x1_expanded) / 2  # (n_segments, n_gauss)
    y_G_k = dy_expanded / 2 * xi_expanded + (y2_expanded + y1_expanded) / 2
    z_G_k = dz_expanded / 2 * xi_expanded + (z2_expanded + z1_expanded) / 2
    
    # Jacobian times weight
    length_half_expanded = length_half[:, np.newaxis]  # (n_segments, 1)
    det_J_weight = length_half_expanded * weight_expanded  # (n_segments, n_gauss)
    
    # Return arrays directly instead of converting to list
    x_G_b_line = np.stack([x_G_k.ravel(), y_G_k.ravel(), z_G_k.ravel()], axis=1)
    det_J_b_time_weight_line = det_J_weight.ravel()
    
    return x_G_b_line, det_J_b_time_weight_line
