"""
Unit tests for evaluation at Gauss points.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from define_eva_at_gauss_points import evaluate_at_gauss_points


class TestEvaluateAtGaussPoints:
    """Test suite for Gauss point evaluation."""
    
    def test_basic_evaluation(self):
        """Test basic shape function evaluation."""
        # Create simple shape function matrix (4 gauss points, 3 nodes)
        shape_func = np.array([
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
            [0.3, 0.5, 0.2],
            [0.2, 0.3, 0.5]
        ])
        
        # Node values
        u = np.array([1.0, 2.0, 3.0])
        
        # Evaluate
        u_G_domain, u_G_boundary = evaluate_at_gauss_points(
            shape_func, 
            shape_func, 
            u
        )
        
        # Check shapes
        assert u_G_domain.shape == (4,)
        assert u_G_boundary.shape == (4,)
        
        # Check partition of unity: sum of shape functions should be 1
        for i in range(4):
            expected = np.dot(shape_func[i, :], u)
            assert np.isclose(u_G_domain[i], expected)
    
    def test_constant_field(self):
        """Test evaluation of constant field."""
        shape_func = np.array([
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2]
        ])
        
        # Constant field
        u = np.array([5.0, 5.0, 5.0])
        
        u_G_domain, _ = evaluate_at_gauss_points(shape_func, shape_func, u)
        
        # With constant field and partition of unity, result should be constant
        assert np.allclose(u_G_domain, 5.0)
    
    def test_linear_field(self):
        """Test evaluation of linear field."""
        # Simple shape functions
        shape_func = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Linear field
        u = np.array([1.0, 2.0, 3.0])
        
        u_G_domain, _ = evaluate_at_gauss_points(shape_func, shape_func, u)
        
        # With identity shape functions, output should equal input
        assert np.allclose(u_G_domain, u)
    
    def test_different_boundary_shape_functions(self):
        """Test with different boundary and domain shape functions."""
        shape_func_domain = np.array([
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2]
        ])
        
        shape_func_boundary = np.array([
            [0.6, 0.2, 0.2],
            [0.3, 0.3, 0.4]
        ])
        
        u = np.array([1.0, 2.0, 3.0])
        
        u_G_domain, u_G_boundary = evaluate_at_gauss_points(
            shape_func_domain, 
            shape_func_boundary, 
            u
        )
        
        # Check that domain and boundary evaluations are different
        assert not np.allclose(u_G_domain, u_G_boundary)
        
        # But both should be valid evaluations
        assert np.all(np.isfinite(u_G_domain))
        assert np.all(np.isfinite(u_G_boundary))
    
    def test_large_system(self):
        """Test with larger system."""
        n_gauss = 100
        n_nodes = 50
        
        # Random shape functions (not necessarily physical, but for testing)
        shape_func = np.random.rand(n_gauss, n_nodes)
        shape_func = shape_func / shape_func.sum(axis=1, keepdims=True)  # Normalize
        
        u = np.random.rand(n_nodes)
        
        u_G_domain, u_G_boundary = evaluate_at_gauss_points(
            shape_func, 
            shape_func, 
            u
        )
        
        assert u_G_domain.shape == (n_gauss,)
        assert u_G_boundary.shape == (n_gauss,)
        assert np.all(np.isfinite(u_G_domain))
    
    def test_zero_field(self):
        """Test evaluation of zero field."""
        shape_func = np.array([
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2]
        ])
        
        u = np.zeros(3)
        
        u_G_domain, u_G_boundary = evaluate_at_gauss_points(
            shape_func, 
            shape_func, 
            u
        )
        
        assert np.allclose(u_G_domain, 0.0)
        assert np.allclose(u_G_boundary, 0.0)
    
    def test_boundary_values_independence(self):
        """Test that domain and boundary evaluations are independent."""
        shape_func_domain = np.array([[0.5, 0.5]])
        shape_func_boundary = np.array([[0.8, 0.2]])
        
        u = np.array([10.0, 20.0])
        
        u_G_domain, u_G_boundary = evaluate_at_gauss_points(
            shape_func_domain, 
            shape_func_boundary, 
            u
        )
        
        # Domain: 0.5*10 + 0.5*20 = 15
        assert np.isclose(u_G_domain[0], 15.0)
        
        # Boundary: 0.8*10 + 0.2*20 = 12
        assert np.isclose(u_G_boundary[0], 12.0)
    
    def test_single_gauss_point(self):
        """Test with single Gauss point."""
        shape_func = np.array([[0.4, 0.35, 0.25]])
        u = np.array([100.0, 200.0, 300.0])
        
        u_G_domain, _ = evaluate_at_gauss_points(shape_func, shape_func, u)
        
        expected = 0.4 * 100 + 0.35 * 200 + 0.25 * 300
        assert np.isclose(u_G_domain[0], expected)
    
    def test_data_types(self):
        """Test that function handles different data types correctly."""
        # Note: numba JIT requires same dtype for all inputs
        shape_func = np.array([[0.5, 0.5]], dtype=np.float64)
        u = np.array([1.0, 2.0], dtype=np.float64)
        
        u_G_domain, u_G_boundary = evaluate_at_gauss_points(
            shape_func, 
            shape_func, 
            u
        )
        
        # Should return numpy arrays
        assert isinstance(u_G_domain, np.ndarray)
        assert isinstance(u_G_boundary, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

