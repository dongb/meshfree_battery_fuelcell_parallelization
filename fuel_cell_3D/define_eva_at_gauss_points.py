from common import np
# Removed numba import - NumPy matrix multiplication is already optimized


def evaluate_at_gauss_points(shape_func, shape_func_b, u):
    """
    Evaluate solution at Gauss points using shape functions.
    
    VECTORIZED: Removed @jit decorator. NumPy's dot product is already
    highly optimized using BLAS/LAPACK libraries, making it faster than
    numba JIT for matrix-vector multiplication.
    
    Parameters:
    -----------
    shape_func : ndarray or sparse matrix
        Shape function matrix for domain Gauss points
    shape_func_b : ndarray or sparse matrix
        Shape function matrix for boundary Gauss points
    u : ndarray
        Solution vector at nodes
    
    Returns:
    --------
    u_G_domain : ndarray
        Solution values at domain Gauss points
    u_G_boundary : ndarray
        Solution values at boundary Gauss points
    """
    # NumPy's dot/matmul is vectorized and uses optimized BLAS
    # For sparse matrices, this automatically uses sparse matrix operations
    u_G_domain = np.dot(shape_func, u)
    u_G_boundary = np.dot(shape_func_b, u)
    
    return u_G_domain, u_G_boundary