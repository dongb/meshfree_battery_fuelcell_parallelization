#workaround for the missing functions in the sparse library
def det_via_cholesky(matrix):
    """
    Calculate determinant using Cholesky decomposition.
    Applicable to: Symmetric Positive Definite (SPD) matrices ONLY.
    Fastest method for covariance matrices.
    """
    print("det_via_cholesky")
    try:
        # 1. Perform Cholesky decomposition
        # Returns lower triangular matrix L such that A = L @ L.T
        L = np.linalg.cholesky(matrix)
        
        # 2. Extract the diagonal elements of L
        diag_L = np.diag(L)
        
        # 3. Calculate the determinant
        # det(A) = det(L) * det(L.T) = det(L)^2
        # det(L) is simply the product of its diagonal elements.
        det = np.prod(diag_L) ** 2
        return det
        
    except np.linalg.LinAlgError:
        print("Error: Matrix is not positive definite. Cholesky failed.")
        return None

def det_via_eigenvalues(matrix):
    """
    Calculate determinant using eigenvalues.
    Applicable to: Any square matrix.
    """
    # 1. Compute all eigenvalues of the matrix
    # Note: This is computationally more expensive than LU decomposition.
    eig_vals = np.linalg.eigvals(matrix)
    
    # 2. The determinant is the product of all eigenvalues
    det = np.prod(eig_vals)
    
    # 3. Handle numerical precision issues
    # If the input matrix contains only real numbers, the determinant should be real.
    # However, eigvals might return complex numbers with negligible imaginary parts (e.g., 1e-15j).
    if np.isrealobj(matrix):
        det = np.real(det)
        
    return det

def manual_bmat_coo(blocks):
    """
    Workaround for bmat by manually merging COO data.
    This avoids intermediate matrix creation and is very memory efficient.
    """
    print("manual_bmat_coo")
    rows_list = []
    cols_list = []
    data_list = []
    
    # Current row offset in the global matrix
    row_offset = 0
    
    # Iterate through the grid of blocks
    for row_blocks in blocks:
        col_offset = 0 # Reset column offset for each new row of blocks
        max_row_height = 0
        
        for block in row_blocks:
            if block is None:
                continue
                
            # Ensure the block is in COO format to access row, col, data
            if not sparse.isspmatrix_coo(block):
                block = block.tocoo()
            
            # 1. Shift indices by current offsets
            # We append the global coordinates to the lists
            rows_list.append(block.row + row_offset)
            cols_list.append(block.col + col_offset)
            data_list.append(block.data)
            
            # 2. Update offsets
            # The next block in this row starts after the current block's width
            col_offset += block.shape[1]
            # Track the max height of this row of blocks
            max_row_height = max(max_row_height, block.shape[0])
            
        # Move the row offset down by the height of the current row of blocks
        row_offset += max_row_height
        
    # 3. Concatenate all data
    if not data_list:
        return sparse.coo_matrix((0, 0))
        
    all_rows = np.concatenate(rows_list)
    all_cols = np.concatenate(cols_list)
    all_data = np.concatenate(data_list)
    
    # 4. Create the final matrix
    # Shape is inferred from max indices, or can be calculated manually if needed
    total_rows = row_offset
    # Note: Assuming all block rows have same total width for simplicity
    total_cols = col_offset 
    
    return sparse.coo_matrix((all_data, (all_rows, all_cols)), shape=(total_rows, total_cols))

try:
    # Uncomment to force use of NumPy backend instead of CuPy
    #raise ImportError
    import cupynumeric as np
    print("cupynumeric imported")
    import legate_sparse as sparse
    print("legate_sparse imported")
    import legate_sparse.linalg as linalg
    print("legate_sparse.linalg imported")
    from legate.timing import time
    print("legate.timing imported")
    from legate_sparse import csr_array
    csr_matrix = csr_array
    print("legate_sparse.csr_array imported")
    #from legate_sparse.linalg import spsolve #debug
    #print("legate_sparse.linalg.spsolve imported")
    np.linalg.det = det_via_eigenvalues
    bmat = manual_bmat_coo 


    use_cupy = True
    print("Using Legate (GPU backend)")
    
except (RuntimeError, ImportError):
    from time import perf_counter_ns
    
    import numpy as np
    import scipy.sparse as sparse
    import scipy.sparse.linalg as linalg
    from scipy.sparse import csr_array, csc_matrix, csr_matrix, bmat, block_diag, vstack, diags
    from scipy.sparse.linalg import spsolve
    def time():
        """NumPy timing function using performance counter"""
        return perf_counter_ns() / 1000.0
    
    use_cupy = False
    print("Using NumPy (CPU backend)")
