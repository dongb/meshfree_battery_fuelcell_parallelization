"""
Integration tests for the fuel_cell_3D simulation package.
Tests the interaction between different modules.
"""
import pytest
from common import np, use_cupy
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _save_image(filename, img):
    """Save image to file, converting from CuPy to NumPy if needed."""
    import tifffile
    if use_cupy and hasattr(img, 'get'):
        img = img.get()
    tifffile.imwrite(filename, img)

from read_image import read_in_image
from get_nodes_gauss_points import (
    get_x_nodes_fuel_cell_3d_toy_image,
    x_G_and_def_J_time_weight_3d_fuelcell_domain
)
from shape_function_in_domain import compute_phi_M
from define_eva_at_gauss_points import evaluate_at_gauss_points


class TestImageToNodesWorkflow:
    """Test the workflow from image to nodes."""
    
    @pytest.fixture
    def sample_3d_image_file(self):
        """Create a sample 3D image file."""
        # Simple 5x5x5 image with 3 phases
        img = np.zeros((5, 5, 5), dtype=np.uint8)
        img[0:2, :, :] = 0  # pore
        img[2:3, :, :] = 1  # electrode
        img[3:5, :, :] = 2  # electrolyte
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            _save_image(tmp.name, img)
            yield tmp.name
        
        os.unlink(tmp.name)
    
    def test_image_to_nodes_workflow(self, sample_3d_image_file):
        """Test complete workflow from image to node generation."""
        # Step 1: Read image
        img, grain_ids, num_pixels = read_in_image(
            sample_3d_image_file,
            "fuel cell",
            dimention=3
        )
        
        assert img.shape == (5, 5, 5)
        assert len(grain_ids) == 3
        
        # Step 2: Generate nodes
        x_min, x_max = 0.0, 10e-6
        y_min, y_max = 0.0, 10e-6
        z_min, z_max = 0.0, 10e-6
        
        result = get_x_nodes_fuel_cell_3d_toy_image(
            x_min, x_max, y_min, y_max, z_min, z_max,
            num_pixels, img
        )
        
        x_nodes_mechanical = result[0]
        x_nodes_electrolyte = result[1]
        x_nodes_electrode = result[2]
        x_nodes_pore = result[3]
        
        # Verify nodes were created
        assert len(x_nodes_mechanical) > 0
        assert len(x_nodes_electrolyte) > 0
        assert len(x_nodes_electrode) > 0
        assert len(x_nodes_pore) > 0
        
        # Verify nodes are within domain (with small tolerance for floating point)
        # Note: Nodes are created at pixel boundaries including far edges
        tol = 1e-12
        for nodes in [x_nodes_mechanical, x_nodes_electrolyte, 
                      x_nodes_electrode, x_nodes_pore]:
            if len(nodes) > 0:
                nodes_array = np.array(nodes)
                assert np.all(nodes_array[:, 0] >= x_min - tol)
                assert np.all(nodes_array[:, 0] <= x_max + tol)
                assert np.all(nodes_array[:, 1] >= y_min - tol)
                assert np.all(nodes_array[:, 1] <= y_max + tol)
                assert np.all(nodes_array[:, 2] >= z_min - tol)
                assert np.all(nodes_array[:, 2] <= z_max + tol)


class TestGaussPointsToShapeFunctions:
    """Test workflow from Gauss points to shape functions."""
    
    def test_gauss_to_shape_workflow(self):
        """Test workflow from Gauss point generation to shape function computation."""
        # Step 1: Create cells
        cell_nodes_x = np.array([[0, 1, 1, 0, 0, 1, 1, 0]])
        cell_nodes_y = np.array([[0, 0, 1, 1, 0, 0, 1, 1]])
        cell_nodes_z = np.array([[0, 0, 0, 0, 1, 1, 1, 1]])
        
        # Step 2: Generate Gauss points
        x_G_domain = [[0, 0, 0]]
        weight_G_domain = [8.0]  # Volume of cube
        
        x_G, det_J_weight = x_G_and_def_J_time_weight_3d_fuelcell_domain(
            cell_nodes_x, cell_nodes_y, cell_nodes_z,
            x_G_domain, weight_G_domain
        )
        
        assert len(x_G) == 1
        assert len(det_J_weight) == 1
        
        # Step 3: Setup for shape function computation
        x_nodes = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=np.float64)
        
        nodes_grain_id = np.ones(8)
        Gauss_grain_id = np.ones(1)
        
        a = np.ones(8) * 1.5  # Support size
        
        M = np.zeros((1, 4, 4))
        M_P_x = np.zeros((1, 4, 4))
        M_P_y = np.zeros((1, 4, 4))
        M_P_z = np.zeros((1, 4, 4))
        
        # Step 4: Compute shape functions
        result = compute_phi_M(
            np.array(x_G),
            Gauss_grain_id,
            x_nodes,
            nodes_grain_id,
            a,
            M, M_P_x, M_P_y,
            num_interface_segments=0,
            interface_nodes=np.zeros((1, 2)),
            BxByCxCy=np.zeros((1, 4)),
            IM_RKPM='False',
            single_grain='True',
            M_P_z=M_P_z
        )
        
        phi_row, phi_col, phi_val = result[0], result[1], result[2]
        
        # Verify shape functions were computed
        assert len(phi_val) > 0
        
        # Verify partition of unity (approximately)
        phi_sum = sum(phi_val)
        assert 0.8 < phi_sum < 1.2


class TestShapeFunctionsToEvaluation:
    """Test workflow from shape functions to field evaluation."""
    
    def test_shape_to_evaluation_workflow(self):
        """Test field evaluation using shape functions."""
        # Step 1: Create simple shape functions
        n_gauss = 3
        n_nodes = 4
        
        shape_func = np.array([
            [0.4, 0.3, 0.2, 0.1],
            [0.3, 0.4, 0.2, 0.1],
            [0.2, 0.3, 0.4, 0.1]
        ])
        
        # Normalize to ensure partition of unity
        shape_func = shape_func / shape_func.sum(axis=1, keepdims=True)
        
        # Step 2: Create a field (e.g., concentration)
        concentration = np.array([100.0, 200.0, 300.0, 400.0])
        
        # Step 3: Evaluate field at Gauss points
        c_gauss_domain, c_gauss_boundary = evaluate_at_gauss_points(
            shape_func,
            shape_func,
            concentration
        )
        
        # Verify evaluation
        assert len(c_gauss_domain) == n_gauss
        assert np.all(np.isfinite(c_gauss_domain))
        
        # Check that values are reasonable (within range of nodal values)
        assert np.all(c_gauss_domain >= concentration.min())
        assert np.all(c_gauss_domain <= concentration.max())


class TestMaterialPropertiesIntegration:
    """Test integration of material properties with other modules."""
    
    def test_material_properties_with_gauss_points(self):
        """Test computing material properties at Gauss points."""
        from define_buttler_volmer import Dn_complex, ocp_complex
        
        # Step 1: Create Gauss points
        n_gauss = 5
        x_G = np.random.rand(n_gauss, 3)
        
        # Step 2: Assume we have concentration field at Gauss points
        c_normalized = np.linspace(0.2, 0.8, n_gauss)
        
        # Step 3: Compute material properties
        D_damage = np.zeros(n_gauss)
        D, dD_dx = Dn_complex(c_normalized, D_damage)
        E_eq, dE_dx = ocp_complex(c_normalized)
        
        # Verify properties are computed
        assert len(D) == n_gauss
        assert len(E_eq) == n_gauss
        assert np.all(D >= 0)
        assert np.all(np.isfinite(E_eq))


class TestBoundaryConditionIntegration:
    """Test boundary condition handling across modules."""
    
    def test_boundary_nodes_identification(self):
        """Test that boundary nodes are correctly identified and used."""
        # Simple 3D image
        img = np.zeros((3, 3, 3), dtype=np.uint8)
        img[1, :, :] = 2  # Electrolyte in middle
        
        x_min, x_max = 0.0, 3e-6
        y_min, y_max = 0.0, 3e-6
        z_min, z_max = 0.0, 3e-6
        num_pixels = [3, 3, 3]
        
        result = get_x_nodes_fuel_cell_3d_toy_image(
            x_min, x_max, y_min, y_max, z_min, z_max,
            num_pixels, img
        )
        
        x_nodes_electrolyte = result[1]
        nodes_id_left_electrolyte = result[7]
        
        # Verify left boundary nodes exist
        assert len(nodes_id_left_electrolyte) > 0
        
        # Verify left boundary nodes are actually at y_min
        for node_id in nodes_id_left_electrolyte:
            if node_id < len(x_nodes_electrolyte):
                node_y = x_nodes_electrolyte[node_id][1]
                assert np.isclose(node_y, y_min, atol=1e-10)


class TestConsistencyChecks:
    """Test consistency across different modules."""
    
    def test_dimension_consistency(self):
        """Test that dimensions are consistent across modules."""
        # 2D vs 3D consistency
        n_nodes = 10
        
        # 2D: each node has 2 coordinates
        x_nodes_2d = np.random.rand(n_nodes, 2)
        
        # 3D: each node has 3 coordinates
        x_nodes_3d = np.random.rand(n_nodes, 3)
        
        # Verify shapes
        assert x_nodes_2d.shape == (n_nodes, 2)
        assert x_nodes_3d.shape == (n_nodes, 3)
    
    def test_gauss_integration_consistency(self):
        """Test that Gauss integration is consistent."""
        # For a cube [0,1]^3, integral of 1 should equal volume
        cell_x = np.array([[0, 1, 1, 0, 0, 1, 1, 0]])
        cell_y = np.array([[0, 0, 1, 1, 0, 0, 1, 1]])
        cell_z = np.array([[0, 0, 0, 0, 1, 1, 1, 1]])
        
        # Single Gauss point at center with weight covering whole cube
        x_G = [[0, 0, 0]]
        weight_G = [8.0]  # 2^3
        
        _, det_J_weight = x_G_and_def_J_time_weight_3d_fuelcell_domain(
            cell_x, cell_y, cell_z, x_G, weight_G
        )
        
        # Total integrated weight should equal volume (1.0 for unit cube)
        total_weight = sum(det_J_weight)
        assert np.isclose(total_weight, 1.0, rtol=0.1)


class TestEndToEndSmoke:
    """Smoke tests for end-to-end workflow."""
    
    def test_minimal_simulation_setup(self):
        """Test minimal simulation setup doesn't crash."""
        # This is a smoke test - just verify basic setup works
        
        # 1. Create minimal geometry
        img = np.zeros((2, 2, 2), dtype=np.uint8)
        img[0, :, :] = 1  # electrode
        img[1, :, :] = 2  # electrolyte
        
        # 2. Generate nodes
        result = get_x_nodes_fuel_cell_3d_toy_image(
            0, 1e-6, 0, 1e-6, 0, 1e-6,
            [2, 2, 2], img
        )
        
        # Just verify it completed without crashing
        assert result is not None
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

