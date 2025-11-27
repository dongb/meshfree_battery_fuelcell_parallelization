try:
    # Uncomment to force use of NumPy backend instead of CuPy
    # raise ImportError
    import cupy as np
    import cupyx.scipy.sparse as sparse
    import cupyx.scipy.sparse.linalg as linalg
    from cupyx.scipy.sparse import csc_matrix, csr_matrix, bmat, vstack, diags
    from cupyx.scipy.sparse.linalg import spsolve
    from time import perf_counter_ns
    
    # CuPy doesn't have csr_array or block_diag, use matrix equivalents
    csr_array = csr_matrix
    
    # Implement block_diag for CuPy
    def block_diag(mats, format=None, dtype=None):
        """Block diagonal sparse matrix from provided matrices."""
        from cupyx.scipy.sparse import bmat as cupyx_bmat
        n = len(mats)
        # Create block diagonal structure
        blocks = [[None] * n for _ in range(n)]
        for i, mat in enumerate(mats):
            blocks[i][i] = mat
        result = cupyx_bmat(blocks, format=format, dtype=dtype)
        return result
    
    def time():
        """Timing function using performance counter"""
        return perf_counter_ns() / 1000.0
    
    use_cupy = True
    print("Using CuPy (GPU backend)")
    
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