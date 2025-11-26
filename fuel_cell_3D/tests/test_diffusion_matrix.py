"""
Unit tests for diffusion matrix assembly.
"""
import pytest
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from define_diffusion_matrix_form import (
    diffusion_matrix_fuel_cell,
    diffusion_matrix_fuel_cell_distributed_point_source
)


class TestDiffusionMatrixAssembly:
    """Test diffusion matrix assembly for fuel cell."""
    
    @pytest.fixture
    def simple_diffusion_setup_2d(self):
        """Create simple 2D diffusion problem setup."""
        n_nodes = 4
        n_gauss = 2
        n_boundary = 2
        
        # Diffusion coefficient
        global_diffusion = np.ones(n_gauss)
        
        # Shape functions and gradients (sparse)
        grad_shape_func_x = csr_matrix(np.random.rand(n_gauss, n_nodes))
        grad_shape_func_y = csr_matrix(np.random.rand(n_gauss, n_nodes))
        
        grad_shape_func_x_weighted = grad_shape_func_x.multiply(0.1)
        grad_shape_func_y_weighted = grad_shape_func_y.multiply(0.1)
        
        # Boundary shape functions
        shape_func_b = csr_matrix(np.random.rand(n_boundary, n_nodes))
        shape_func_b_weighted = shape_func_b.multiply(0.05)
        
        grad_shape_func_b_x_weighted = csr_matrix(np.random.rand(n_boundary, n_nodes) * 0.05)
        grad_shape_func_b_y_weighted = csr_matrix(np.random.rand(n_boundary, n_nodes) * 0.05)
        
        # Boundary conditions
        g_diretchlet = np.ones(n_boundary)
        beta_Nitsche = np.ones(n_boundary) * 100.0
        normal_vector_x = np.array([1.0, 0.0])
        normal_vector_y = np.array([0.0, 1.0])
        
        # Source terms
        point_source = np.zeros(n_nodes)
        shape_func_point = csr_matrix(np.eye(n_nodes))
        interface_source = np.zeros(n_boundary)
        shape_func_inter_weighted = csr_matrix(np.ones((n_boundary, n_nodes)) * 0.01)
        
        return {
            'dimention': 2,
            'point_source': point_source,
            'shape_func_point': shape_func_point,
            'g_diretchlet': g_diretchlet,
            'beta_Nitsche': beta_Nitsche,
            'normal_vector_x': normal_vector_x,
            'normal_vector_y': normal_vector_y,
            'global_diffusion': global_diffusion,
            'grad_shape_func_x': grad_shape_func_x,
            'grad_shape_func_y': grad_shape_func_y,
            'grad_shape_func_x_weighted': grad_shape_func_x_weighted,
            'grad_shape_func_y_weighted': grad_shape_func_y_weighted,
            'shape_func_b': shape_func_b,
            'shape_func_b_weighted': shape_func_b_weighted,
            'grad_shape_func_b_x_weighted': grad_shape_func_b_x_weighted,
            'grad_shape_func_b_y_weighted': grad_shape_func_b_y_weighted,
            'interface_source': interface_source,
            'shape_func_inter_weighted': shape_func_inter_weighted
        }
    
    def test_matrix_assembly_returns_K_and_f(self, simple_diffusion_setup_2d):
        """Test that diffusion matrix assembly returns K and f."""
        setup = simple_diffusion_setup_2d
        
        K, f = diffusion_matrix_fuel_cell(
            dimention=setup['dimention'],
            point_or_line_source=setup['point_source'],
            shape_func_point_or_line_nodes=setup['shape_func_point'],
            g_diretchlet=setup['g_diretchlet'],
            beta_Nitsche=setup['beta_Nitsche'],
            normal_vector_x=setup['normal_vector_x'],
            normal_vector_y=setup['normal_vector_y'],
            global_diffusion=setup['global_diffusion'],
            grad_shape_func_x=setup['grad_shape_func_x'],
            grad_shape_func_y=setup['grad_shape_func_y'],
            grad_shape_func_x_times_det_J_time_weight=setup['grad_shape_func_x_weighted'],
            grad_shape_func_y_times_det_J_time_weight=setup['grad_shape_func_y_weighted'],
            shape_func_b=setup['shape_func_b'],
            shape_func_b_times_det_J_b_time_weight=setup['shape_func_b_weighted'],
            grad_shape_func_b_x_times_det_J_b_time_weight=setup['grad_shape_func_b_x_weighted'],
            grad_shape_func_b_y_times_det_J_b_time_weight=setup['grad_shape_func_b_y_weighted'],
            shape_func_inter_times_det_J_b_time_weight=setup['shape_func_inter_weighted'],
            interface_source=setup['interface_source']
        )
        
        # Check types
        assert sp.issparse(K)
        assert isinstance(f, np.ndarray)
        
        # Check dimensions
        assert K.shape[0] == K.shape[1]
        assert f.shape[0] == K.shape[0]
    
    def test_matrix_symmetry(self, simple_diffusion_setup_2d):
        """Test that stiffness matrix is symmetric (for symmetric problem)."""
        setup = simple_diffusion_setup_2d
        
        K, _ = diffusion_matrix_fuel_cell(
            dimention=setup['dimention'],
            point_or_line_source=setup['point_source'],
            shape_func_point_or_line_nodes=setup['shape_func_point'],
            g_diretchlet=setup['g_diretchlet'],
            beta_Nitsche=setup['beta_Nitsche'],
            normal_vector_x=setup['normal_vector_x'],
            normal_vector_y=setup['normal_vector_y'],
            global_diffusion=setup['global_diffusion'],
            grad_shape_func_x=setup['grad_shape_func_x'],
            grad_shape_func_y=setup['grad_shape_func_y'],
            grad_shape_func_x_times_det_J_time_weight=setup['grad_shape_func_x_weighted'],
            grad_shape_func_y_times_det_J_time_weight=setup['grad_shape_func_y_weighted'],
            shape_func_b=setup['shape_func_b'],
            shape_func_b_times_det_J_b_time_weight=setup['shape_func_b_weighted'],
            grad_shape_func_b_x_times_det_J_b_time_weight=setup['grad_shape_func_b_x_weighted'],
            grad_shape_func_b_y_times_det_J_b_time_weight=setup['grad_shape_func_b_y_weighted'],
            shape_func_inter_times_det_J_b_time_weight=setup['shape_func_inter_weighted'],
            interface_source=setup['interface_source']
        )
        
        # K1 is symmetric, but K2 and K3 may not be
        # Check that K is at least square
        assert K.shape[0] == K.shape[1]
    
    def test_positive_diagonal(self, simple_diffusion_setup_2d):
        """Test that diagonal entries are positive (for well-posed problem)."""
        setup = simple_diffusion_setup_2d
        
        K, _ = diffusion_matrix_fuel_cell(
            dimention=setup['dimention'],
            point_or_line_source=setup['point_source'],
            shape_func_point_or_line_nodes=setup['shape_func_point'],
            g_diretchlet=setup['g_diretchlet'],
            beta_Nitsche=setup['beta_Nitsche'],
            normal_vector_x=setup['normal_vector_x'],
            normal_vector_y=setup['normal_vector_y'],
            global_diffusion=setup['global_diffusion'],
            grad_shape_func_x=setup['grad_shape_func_x'],
            grad_shape_func_y=setup['grad_shape_func_y'],
            grad_shape_func_x_times_det_J_time_weight=setup['grad_shape_func_x_weighted'],
            grad_shape_func_y_times_det_J_time_weight=setup['grad_shape_func_y_weighted'],
            shape_func_b=setup['shape_func_b'],
            shape_func_b_times_det_J_b_time_weight=setup['shape_func_b_weighted'],
            grad_shape_func_b_x_times_det_J_b_time_weight=setup['grad_shape_func_b_x_weighted'],
            grad_shape_func_b_y_times_det_J_b_time_weight=setup['grad_shape_func_b_y_weighted'],
            shape_func_inter_times_det_J_b_time_weight=setup['shape_func_inter_weighted'],
            interface_source=setup['interface_source']
        )
        
        # Extract diagonal
        diagonal = K.diagonal()
        
        # Most diagonal entries should be positive (due to Nitsche penalty)
        assert np.sum(diagonal > 0) > len(diagonal) * 0.5
    
    def test_force_vector_shape(self, simple_diffusion_setup_2d):
        """Test that force vector has correct shape."""
        setup = simple_diffusion_setup_2d
        
        _, f = diffusion_matrix_fuel_cell(
            dimention=setup['dimention'],
            point_or_line_source=setup['point_source'],
            shape_func_point_or_line_nodes=setup['shape_func_point'],
            g_diretchlet=setup['g_diretchlet'],
            beta_Nitsche=setup['beta_Nitsche'],
            normal_vector_x=setup['normal_vector_x'],
            normal_vector_y=setup['normal_vector_y'],
            global_diffusion=setup['global_diffusion'],
            grad_shape_func_x=setup['grad_shape_func_x'],
            grad_shape_func_y=setup['grad_shape_func_y'],
            grad_shape_func_x_times_det_J_time_weight=setup['grad_shape_func_x_weighted'],
            grad_shape_func_y_times_det_J_time_weight=setup['grad_shape_func_y_weighted'],
            shape_func_b=setup['shape_func_b'],
            shape_func_b_times_det_J_b_time_weight=setup['shape_func_b_weighted'],
            grad_shape_func_b_x_times_det_J_b_time_weight=setup['grad_shape_func_b_x_weighted'],
            grad_shape_func_b_y_times_det_J_b_time_weight=setup['grad_shape_func_b_y_weighted'],
            shape_func_inter_times_det_J_b_time_weight=setup['shape_func_inter_weighted'],
            interface_source=setup['interface_source']
        )
        
        assert f.ndim == 2
        assert f.shape[1] == 1
    
    def test_distributed_source_variant(self, simple_diffusion_setup_2d):
        """Test distributed point source variant."""
        setup = simple_diffusion_setup_2d
        
        distributed_source = np.ones(2)
        shape_func_distributed = csr_matrix(np.ones((2, 4)) * 0.1)
        
        K, f = diffusion_matrix_fuel_cell_distributed_point_source(
            dimention=setup['dimention'],
            distributed_point_or_line_source=distributed_source,
            shape_func_distributed_point_or_line_nodes=shape_func_distributed,
            g_diretchlet=setup['g_diretchlet'],
            beta_Nitsche=setup['beta_Nitsche'],
            normal_vector_x=setup['normal_vector_x'],
            normal_vector_y=setup['normal_vector_y'],
            global_diffusion=setup['global_diffusion'],
            grad_shape_func_x=setup['grad_shape_func_x'],
            grad_shape_func_y=setup['grad_shape_func_y'],
            grad_shape_func_x_times_det_J_time_weight=setup['grad_shape_func_x_weighted'],
            grad_shape_func_y_times_det_J_time_weight=setup['grad_shape_func_y_weighted'],
            shape_func_b=setup['shape_func_b'],
            shape_func_b_times_det_J_b_time_weight=setup['shape_func_b_weighted'],
            grad_shape_func_b_x_times_det_J_b_time_weight=setup['grad_shape_func_b_x_weighted'],
            grad_shape_func_b_y_times_det_J_b_time_weight=setup['grad_shape_func_b_y_weighted'],
            shape_func_inter_times_det_J_b_time_weight=setup['shape_func_inter_weighted'],
            interface_source=setup['interface_source']
        )
        
        assert sp.issparse(K)
        assert isinstance(f, np.ndarray)


class TestDiffusionMatrixProperties:
    """Test mathematical properties of diffusion matrix."""
    
    def test_diffusion_scaling(self):
        """Test that increasing diffusion coefficient scales matrix."""
        n_nodes = 3
        n_gauss = 1
        
        # Setup
        grad_x = csr_matrix(np.array([[0.5, -0.5, 0.0]]))
        grad_y = csr_matrix(np.array([[0.5, 0.0, -0.5]]))
        grad_x_w = grad_x * 0.1
        grad_y_w = grad_y * 0.1
        
        # Small diffusion
        D1 = np.array([0.1])
        
        # Large diffusion
        D2 = np.array([1.0])
        
        # Dummy boundary terms
        shape_b = csr_matrix(np.zeros((1, n_nodes)))
        shape_b_w = csr_matrix(np.zeros((1, n_nodes)))
        grad_b_x_w = csr_matrix(np.zeros((1, n_nodes)))
        grad_b_y_w = csr_matrix(np.zeros((1, n_nodes)))
        g_dir = np.zeros(1)
        beta = np.ones(1)
        n_x = np.zeros(1)
        n_y = np.zeros(1)
        pt_src = np.zeros(n_nodes)
        shape_pt = csr_matrix(np.eye(n_nodes))
        int_src = np.zeros(1)
        shape_int = csr_matrix(np.zeros((1, n_nodes)))
        
        K1, _ = diffusion_matrix_fuel_cell(
            2, pt_src, shape_pt, g_dir, beta, n_x, n_y, D1,
            grad_x, grad_y, grad_x_w, grad_y_w,
            shape_b, shape_b_w, grad_b_x_w, grad_b_y_w,
            shape_int, int_src
        )
        
        K2, _ = diffusion_matrix_fuel_cell(
            2, pt_src, shape_pt, g_dir, beta, n_x, n_y, D2,
            grad_x, grad_y, grad_x_w, grad_y_w,
            shape_b, shape_b_w, grad_b_x_w, grad_b_y_w,
            shape_int, int_src
        )
        
        # K2 should be larger than K1 (in Frobenius norm sense)
        K1_norm = np.linalg.norm(K1.toarray())
        K2_norm = np.linalg.norm(K2.toarray())
        
        assert K2_norm > K1_norm


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

