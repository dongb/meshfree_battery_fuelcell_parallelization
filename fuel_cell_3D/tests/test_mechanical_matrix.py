"""
Unit tests for mechanical stiffness matrix assembly.
"""
import pytest
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from define_mechanical_stiffness_matrix import (
    mechanical_C_tensor_3d,
    mechanical_stiffness_matrix_3d_fuel_cell,
    mechanical_force_matrix_3d
)


class TestMechanicalCTensor:
    """Test mechanical stiffness tensor computation."""
    
    def test_c_tensor_basic(self):
        """Test basic C tensor computation."""
        num_gauss = 10
        D_damage = np.zeros((num_gauss, 1))
        lambda_mech = 1e10
        mu = 1e10
        
        # Zero rotation
        gauss_angle = np.zeros(num_gauss)
        gauss_rotation_axis = np.zeros((num_gauss, 3))
        gauss_rotation_axis[:, 0] = 1.0  # Rotate around x-axis (but angle=0)
        
        C, T_c = mechanical_C_tensor_3d(
            num_gauss, D_damage, lambda_mech, mu,
            gauss_angle, gauss_rotation_axis
        )
        
        # Check shape
        assert C.shape == (6, 6, num_gauss, 1)
        assert T_c.shape == (6, 6, num_gauss)
    
    def test_isotropic_material(self):
        """Test isotropic material properties."""
        num_gauss = 1
        D_damage = np.zeros((num_gauss, 1))
        
        E = 200e9  # Young's modulus (steel)
        nu = 0.3   # Poisson's ratio
        lambda_mech = E * nu / ((1 + nu) * (1 - 2*nu))
        mu = E / (2 * (1 + nu))
        
        # Zero rotation
        gauss_angle = np.zeros(num_gauss)
        gauss_rotation_axis = np.zeros((num_gauss, 3))
        gauss_rotation_axis[:, 0] = 1.0
        
        C, _ = mechanical_C_tensor_3d(
            num_gauss, D_damage, lambda_mech, mu,
            gauss_angle, gauss_rotation_axis
        )
        
        # Check diagonal components
        C11 = C[0, 0, 0, 0]
        C22 = C[1, 1, 0, 0]
        C33 = C[2, 2, 0, 0]
        
        # For isotropic material with zero rotation, C11 = C22 = C33
        assert np.isclose(C11, C22, rtol=1e-6)
        assert np.isclose(C22, C33, rtol=1e-6)
        
        # C11 should equal lambda + 2*mu
        expected = lambda_mech + 2*mu
        assert np.isclose(C11, expected, rtol=1e-6)
    
    def test_damage_reduces_stiffness(self):
        """Test that damage reduces stiffness."""
        num_gauss = 2
        D_damage_zero = np.zeros((num_gauss, 1))
        D_damage_half = np.ones((num_gauss, 1)) * 0.5
        
        lambda_mech = 1e10
        mu = 1e10
        gauss_angle = np.zeros(num_gauss)
        gauss_rotation_axis = np.zeros((num_gauss, 3))
        gauss_rotation_axis[:, 0] = 1.0
        
        C_zero, _ = mechanical_C_tensor_3d(
            num_gauss, D_damage_zero, lambda_mech, mu,
            gauss_angle, gauss_rotation_axis
        )
        
        C_half, _ = mechanical_C_tensor_3d(
            num_gauss, D_damage_half, lambda_mech, mu,
            gauss_angle, gauss_rotation_axis
        )
        
        # With 50% damage, stiffness should be 50% of original
        ratio = C_half[0, 0, 0, 0] / C_zero[0, 0, 0, 0]
        assert np.isclose(ratio, 0.5, rtol=0.01)
    
    def test_damage_capping(self):
        """Test that damage is capped at 0.9."""
        num_gauss = 1
        D_damage_excessive = np.ones((num_gauss, 1)) * 0.95  # Above 0.9 limit
        
        lambda_mech = 1e10
        mu = 1e10
        gauss_angle = np.zeros(num_gauss)
        gauss_rotation_axis = np.zeros((num_gauss, 3))
        gauss_rotation_axis[:, 0] = 1.0
        
        C, _ = mechanical_C_tensor_3d(
            num_gauss, D_damage_excessive, lambda_mech, mu,
            gauss_angle, gauss_rotation_axis
        )
        
        # After capping at 0.9, (1-D_damage) = 0.1
        # So stiffness should be 10% of undamaged
        D_damage_capped = np.ones((num_gauss, 1)) * 0.9
        C_capped, _ = mechanical_C_tensor_3d(
            num_gauss, D_damage_capped, lambda_mech, mu,
            gauss_angle, gauss_rotation_axis
        )
        
        assert np.isclose(C[0, 0, 0, 0], C_capped[0, 0, 0, 0], rtol=0.01)
    
    def test_rotation_matrix_properties(self):
        """Test rotation matrix properties."""
        num_gauss = 1
        D_damage = np.zeros((num_gauss, 1))
        lambda_mech = 1e10
        mu = 1e10
        
        # 90 degree rotation around z-axis
        gauss_angle = np.array([np.pi/2])
        gauss_rotation_axis = np.array([[0, 0, 1]])
        
        _, T_c = mechanical_C_tensor_3d(
            num_gauss, D_damage, lambda_mech, mu,
            gauss_angle, gauss_rotation_axis
        )
        
        # Extract rotation matrix for first Gauss point
        T = T_c[:, :, 0]
        
        # Check that T is finite
        assert np.all(np.isfinite(T))


class TestMechanicalStiffnessMatrix:
    """Test mechanical stiffness matrix assembly."""
    
    @pytest.fixture
    def simple_mechanical_setup(self):
        """Create simple mechanical problem setup."""
        n_nodes = 4
        n_gauss = 2
        
        # C tensor (simplified)
        C = np.zeros((6, 6, n_gauss, 1))
        for i in range(6):
            C[i, i, :, :] = 1e10  # Simplified diagonal stiffness
        
        # Gradient shape functions
        grad_x = csr_matrix(np.random.rand(n_gauss, n_nodes) * 0.1)
        grad_y = csr_matrix(np.random.rand(n_gauss, n_nodes) * 0.1)
        grad_z = csr_matrix(np.random.rand(n_gauss, n_nodes) * 0.1)
        
        grad_x_w = grad_x * 0.01
        grad_y_w = grad_y * 0.01
        grad_z_w = grad_z * 0.01
        
        # Fixed point boundary conditions
        n_fixed = 1
        shape_fixed = csr_matrix(np.ones((n_fixed, n_nodes)) * 0.25)
        shape_fixed_w = shape_fixed * 0.01
        grad_fixed_x = csr_matrix(np.random.rand(n_fixed, n_nodes) * 0.1)
        grad_fixed_y = csr_matrix(np.random.rand(n_fixed, n_nodes) * 0.1)
        grad_fixed_z = csr_matrix(np.random.rand(n_fixed, n_nodes) * 0.1)
        grad_fixed_x_w = grad_fixed_x * 0.01
        grad_fixed_y_w = grad_fixed_y * 0.01
        grad_fixed_z_w = grad_fixed_z * 0.01
        
        # Boundary shape functions (electrolyte)
        n_boundary = 1
        shape_b = csr_matrix(np.ones((n_boundary, n_nodes)) * 0.25)
        shape_b_w = shape_b * 0.01
        grad_b_x = csr_matrix(np.random.rand(n_boundary, n_nodes) * 0.1)
        grad_b_y = csr_matrix(np.random.rand(n_boundary, n_nodes) * 0.1)
        grad_b_z = csr_matrix(np.random.rand(n_boundary, n_nodes) * 0.1)
        grad_b_x_w = grad_b_x * 0.01
        grad_b_y_w = grad_b_y * 0.01
        grad_b_z_w = grad_b_z * 0.01
        
        # Normal vectors
        n_vec_x = np.array([1.0])
        n_vec_y = np.array([0.0])
        n_vec_z = np.array([0.0])
        
        # Penalty parameter
        beta = 1e10
        
        return {
            'C': C,
            'n_gauss': n_gauss,
            'grad_x': grad_x,
            'grad_y': grad_y,
            'grad_z': grad_z,
            'grad_x_w': grad_x_w,
            'grad_y_w': grad_y_w,
            'grad_z_w': grad_z_w,
            'beta': beta,
            'shape_fixed': shape_fixed,
            'shape_fixed_w': shape_fixed_w,
            'grad_fixed_x': grad_fixed_x,
            'grad_fixed_y': grad_fixed_y,
            'grad_fixed_z': grad_fixed_z,
            'grad_fixed_x_w': grad_fixed_x_w,
            'grad_fixed_y_w': grad_fixed_y_w,
            'grad_fixed_z_w': grad_fixed_z_w,
            'n_vec_x': n_vec_x,
            'n_vec_y': n_vec_y,
            'n_vec_z': n_vec_z,
            'shape_b': shape_b,
            'shape_b_w': shape_b_w,
            'grad_b_x': grad_b_x,
            'grad_b_y': grad_b_y,
            'grad_b_z': grad_b_z,
            'grad_b_x_w': grad_b_x_w,
            'grad_b_y_w': grad_b_y_w,
            'grad_b_z_w': grad_b_z_w
        }
    
    def test_stiffness_matrix_assembly(self, simple_mechanical_setup):
        """Test basic stiffness matrix assembly."""
        setup = simple_mechanical_setup
        
        K = mechanical_stiffness_matrix_3d_fuel_cell(
            setup['C'],
            setup['n_gauss'],
            setup['grad_x_w'], setup['grad_x'],
            setup['grad_y_w'], setup['grad_y'],
            setup['grad_z_w'], setup['grad_z'],
            setup['beta'],
            setup['shape_fixed'], setup['shape_fixed_w'],
            setup['grad_fixed_x'], setup['grad_fixed_x_w'],
            setup['grad_fixed_y'], setup['grad_fixed_y_w'],
            setup['grad_fixed_z'], setup['grad_fixed_z_w'],
            setup['n_vec_x'], setup['n_vec_y'], setup['n_vec_z'],
            setup['shape_b'], setup['shape_b_w'],
            setup['grad_b_x'], setup['grad_b_x_w'],
            setup['grad_b_y'], setup['grad_b_y_w'],
            setup['grad_b_z'], setup['grad_b_z_w']
        )
        
        # Check that K is sparse matrix
        assert sp.issparse(K)
        
        # Check that K is square
        assert K.shape[0] == K.shape[1]
        
        # K should be 3*n_nodes × 3*n_nodes (3 DOF per node)
        assert K.shape[0] % 3 == 0
    
    def test_stiffness_matrix_positive_diagonal(self, simple_mechanical_setup):
        """Test that diagonal entries are positive."""
        setup = simple_mechanical_setup
        
        K = mechanical_stiffness_matrix_3d_fuel_cell(
            setup['C'],
            setup['n_gauss'],
            setup['grad_x_w'], setup['grad_x'],
            setup['grad_y_w'], setup['grad_y'],
            setup['grad_z_w'], setup['grad_z'],
            setup['beta'],
            setup['shape_fixed'], setup['shape_fixed_w'],
            setup['grad_fixed_x'], setup['grad_fixed_x_w'],
            setup['grad_fixed_y'], setup['grad_fixed_y_w'],
            setup['grad_fixed_z'], setup['grad_fixed_z_w'],
            setup['n_vec_x'], setup['n_vec_y'], setup['n_vec_z'],
            setup['shape_b'], setup['shape_b_w'],
            setup['grad_b_x'], setup['grad_b_x_w'],
            setup['grad_b_y'], setup['grad_b_y_w'],
            setup['grad_b_z'], setup['grad_b_z_w']
        )
        
        # Most diagonal entries should be positive
        diagonal = K.diagonal()
        assert np.sum(diagonal > 0) > len(diagonal) * 0.5


class TestMechanicalForceMatrix:
    """Test mechanical force vector assembly."""
    
    def test_force_matrix_basic(self):
        """Test basic force vector computation."""
        n_nodes = 4
        n_gauss = 2
        
        x_G = np.random.rand(n_gauss, 3)
        
        # C tensor
        C = np.zeros((6, 6, n_gauss, 1))
        for i in range(6):
            C[i, i, :, :] = 1e10
        
        # Eigenstrain
        epsilon_D1 = np.ones((n_gauss, 1)) * 0.001
        epsilon_D2 = np.ones((n_gauss, 1)) * 0.001
        epsilon_D3 = np.ones((n_gauss, 1)) * 0.001
        epsilon_D4 = np.zeros((n_gauss, 1))
        epsilon_D5 = np.zeros((n_gauss, 1))
        epsilon_D6 = np.zeros((n_gauss, 1))
        
        # Gradient shape functions
        grad_x_w = csr_matrix(np.random.rand(n_gauss, n_nodes) * 0.01)
        grad_y_w = csr_matrix(np.random.rand(n_gauss, n_nodes) * 0.01)
        grad_z_w = csr_matrix(np.random.rand(n_gauss, n_nodes) * 0.01)
        
        f = mechanical_force_matrix_3d(
            x_G, C, epsilon_D1, epsilon_D2, epsilon_D3,
            epsilon_D4, epsilon_D5, epsilon_D6,
            grad_x_w, grad_y_w, grad_z_w
        )
        
        # Check shape
        assert f.shape[0] == 3 * n_nodes
        assert f.shape[1] == 1
        
        # Check that force is finite
        assert np.all(np.isfinite(f))
    
    def test_force_zero_eigenstrain(self):
        """Test that zero eigenstrain gives zero force."""
        n_nodes = 3
        n_gauss = 1
        
        x_G = np.array([[0.5, 0.5, 0.5]])
        
        C = np.zeros((6, 6, n_gauss, 1))
        for i in range(6):
            C[i, i, :, :] = 1e10
        
        # Zero eigenstrain
        epsilon_D1 = np.zeros((n_gauss, 1))
        epsilon_D2 = np.zeros((n_gauss, 1))
        epsilon_D3 = np.zeros((n_gauss, 1))
        epsilon_D4 = np.zeros((n_gauss, 1))
        epsilon_D5 = np.zeros((n_gauss, 1))
        epsilon_D6 = np.zeros((n_gauss, 1))
        
        grad_x_w = csr_matrix(np.ones((n_gauss, n_nodes)) * 0.01)
        grad_y_w = csr_matrix(np.ones((n_gauss, n_nodes)) * 0.01)
        grad_z_w = csr_matrix(np.ones((n_gauss, n_nodes)) * 0.01)
        
        f = mechanical_force_matrix_3d(
            x_G, C, epsilon_D1, epsilon_D2, epsilon_D3,
            epsilon_D4, epsilon_D5, epsilon_D6,
            grad_x_w, grad_y_w, grad_z_w
        )
        
        # Force should be near zero
        assert np.allclose(f, 0.0, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

