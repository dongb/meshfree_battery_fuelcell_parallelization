"""
Unit tests for node and Gauss point generation functions.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from get_nodes_gauss_points import (
    get_x_nodes_fuel_cell_3d_toy_image,
    x_G_and_def_J_time_weight_3d_fuelcell_domain,
    x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary,
    x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary_interface,
    x_G_and_det_J_line_3d_fuelcell_1d_boundary
)


class TestGetNodesFromImage:
    """Test node generation from 3D images."""
    
    @pytest.fixture
    def simple_3d_image(self):
        """Create a simple 3x3x3 test image."""
        img = np.zeros((3, 3, 3), dtype=np.uint8)
        # Phase 0: pore
        img[0, :, :] = 0
        # Phase 1: electrode
        img[1, :, :] = 1
        # Phase 2: electrolyte
        img[2, :, :] = 2
        return img
    
    def test_get_nodes_basic(self, simple_3d_image):
        """Test basic node generation from image."""
        x_min, x_max = 0.0, 10e-6
        y_min, y_max = 0.0, 10e-6
        z_min, z_max = 0.0, 10e-6
        num_pixels_xyz = [3, 3, 3]
        
        result = get_x_nodes_fuel_cell_3d_toy_image(
            x_min, x_max, y_min, y_max, z_min, z_max,
            num_pixels_xyz, simple_3d_image
        )
        
        # Unpack results
        (x_nodes_mechanical, x_nodes_electrolyte, x_nodes_electrode, 
         x_nodes_pore, segments_source, *_) = result
        
        # Check that nodes were created
        assert len(x_nodes_mechanical) > 0
        assert len(x_nodes_electrolyte) > 0
        assert len(x_nodes_electrode) > 0
        assert len(x_nodes_pore) > 0
    
    def test_node_coordinates_within_bounds(self, simple_3d_image):
        """Test that generated nodes are within domain bounds."""
        x_min, x_max = 0.0, 10e-6
        y_min, y_max = 0.0, 20e-6
        z_min, z_max = 0.0, 10e-6
        num_pixels_xyz = [3, 3, 3]
        
        result = get_x_nodes_fuel_cell_3d_toy_image(
            x_min, x_max, y_min, y_max, z_min, z_max,
            num_pixels_xyz, simple_3d_image
        )
        
        x_nodes_mechanical = result[0]
        
        for node in x_nodes_mechanical:
            assert x_min <= node[0] <= x_max
            assert y_min <= node[1] <= y_max
            assert z_min <= node[2] <= z_max
    
    def test_segments_source_structure(self, simple_3d_image):
        """Test that segment sources have correct structure."""
        x_min, x_max = 0.0, 10e-6
        y_min, y_max = 0.0, 10e-6
        z_min, z_max = 0.0, 10e-6
        num_pixels_xyz = [3, 3, 3]
        
        result = get_x_nodes_fuel_cell_3d_toy_image(
            x_min, x_max, y_min, y_max, z_min, z_max,
            num_pixels_xyz, simple_3d_image
        )
        
        segments_source = result[4]
        
        # Each segment should have 6 coordinates (2 points × 3D)
        if len(segments_source) > 0:
            assert segments_source.shape[1] == 6
    
    def test_boundary_node_identification(self, simple_3d_image):
        """Test that boundary nodes are correctly identified."""
        x_min, x_max = 0.0, 10e-6
        y_min, y_max = 0.0, 10e-6
        z_min, z_max = 0.0, 10e-6
        num_pixels_xyz = [3, 3, 3]
        
        result = get_x_nodes_fuel_cell_3d_toy_image(
            x_min, x_max, y_min, y_max, z_min, z_max,
            num_pixels_xyz, simple_3d_image
        )
        
        # Check boundary node IDs
        nodes_id_left_electrolyte = result[7]
        nodes_id_right_electrode = result[8]
        nodes_id_right_pore = result[9]
        
        # All should be lists/arrays of node IDs
        assert isinstance(nodes_id_left_electrolyte, (list, np.ndarray))
        assert isinstance(nodes_id_right_electrode, (list, np.ndarray))
        assert isinstance(nodes_id_right_pore, (list, np.ndarray))


class TestGaussPointGeneration3D:
    """Test Gauss point generation in 3D domain."""
    
    def test_gauss_points_cube(self):
        """Test Gauss point generation for a cube."""
        # Simple cube cell
        cell_nodes_x = np.array([[0, 1, 1, 0, 0, 1, 1, 0]])
        cell_nodes_y = np.array([[0, 0, 1, 1, 0, 0, 1, 1]])
        cell_nodes_z = np.array([[0, 0, 0, 0, 1, 1, 1, 1]])
        
        # 2x2x2 Gauss quadrature
        x_G_domain = [
            [-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)],
            [1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)],
            [-1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)],
            [1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)],
            [-1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)],
            [1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)],
            [-1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
            [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]
        ]
        weight_G_domain = [1.0] * 8
        
        x_G, det_J_weight = x_G_and_def_J_time_weight_3d_fuelcell_domain(
            cell_nodes_x, cell_nodes_y, cell_nodes_z,
            x_G_domain, weight_G_domain
        )
        
        # Should have 8 Gauss points (1 cell × 8 points per cell)
        assert len(x_G) == 8
        assert len(det_J_weight) == 8
        
        # All Gauss points should be within the cube [0,1]^3
        for point in x_G:
            assert 0 <= point[0] <= 1
            assert 0 <= point[1] <= 1
            assert 0 <= point[2] <= 1
        
        # Det(J) * weight should be positive
        for dj_w in det_J_weight:
            assert dj_w > 0
    
    def test_gauss_points_jacobian_scaling(self):
        """Test that Jacobian scales correctly with element size."""
        # Small cube
        cell_small_x = np.array([[0, 0.1, 0.1, 0, 0, 0.1, 0.1, 0]])
        cell_small_y = np.array([[0, 0, 0.1, 0.1, 0, 0, 0.1, 0.1]])
        cell_small_z = np.array([[0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1]])
        
        # Large cube
        cell_large_x = np.array([[0, 1, 1, 0, 0, 1, 1, 0]])
        cell_large_y = np.array([[0, 0, 1, 1, 0, 0, 1, 1]])
        cell_large_z = np.array([[0, 0, 0, 0, 1, 1, 1, 1]])
        
        x_G_domain = [[0, 0, 0]]
        weight_G_domain = [1.0]
        
        _, det_J_small = x_G_and_def_J_time_weight_3d_fuelcell_domain(
            cell_small_x, cell_small_y, cell_small_z,
            x_G_domain, weight_G_domain
        )
        
        _, det_J_large = x_G_and_def_J_time_weight_3d_fuelcell_domain(
            cell_large_x, cell_large_y, cell_large_z,
            x_G_domain, weight_G_domain
        )
        
        # Volume ratio should be (10)^3 = 1000
        volume_ratio = det_J_large[0] / det_J_small[0]
        assert np.isclose(volume_ratio, 1000.0, rtol=0.01)
    
    def test_multiple_cells(self):
        """Test Gauss points for multiple cells."""
        # Two cubes
        cell_nodes_x = np.array([
            [0, 1, 1, 0, 0, 1, 1, 0],
            [1, 2, 2, 1, 1, 2, 2, 1]
        ])
        cell_nodes_y = np.array([
            [0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1]
        ])
        cell_nodes_z = np.array([
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1]
        ])
        
        x_G_domain = [[0, 0, 0]]
        weight_G_domain = [1.0]
        
        x_G, det_J_weight = x_G_and_def_J_time_weight_3d_fuelcell_domain(
            cell_nodes_x, cell_nodes_y, cell_nodes_z,
            x_G_domain, weight_G_domain
        )
        
        # Should have 2 Gauss points (2 cells × 1 point per cell)
        assert len(x_G) == 2
        assert len(det_J_weight) == 2


class TestBoundaryGaussPoints:
    """Test Gauss point generation on boundaries."""
    
    def test_2d_boundary_in_3d(self):
        """Test 2D boundary Gauss points in 3D space."""
        # Square boundary at y = 0
        cell_boundary_x = np.array([[0, 1, 1, 0]])
        cell_boundary_z = np.array([[0, 0, 1, 1]])
        y_coordinate = 0.0
        
        x_G_rec = [[-1/np.sqrt(3), -1/np.sqrt(3)]]
        weight_G_rec = [1.0]
        
        x_G, det_J_weight = x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary(
            cell_boundary_x, cell_boundary_z, y_coordinate,
            x_G_rec, weight_G_rec
        )
        
        assert len(x_G) == 1
        assert len(det_J_weight) == 1
        
        # Check that y-coordinate is fixed
        assert np.isclose(x_G[0][1], y_coordinate)
        
        # Check that boundary point is within the square
        assert 0 <= x_G[0][0] <= 1
        assert 0 <= x_G[0][2] <= 1
    
    def test_interface_boundary_detection(self):
        """Test interface boundary Gauss points."""
        # Interface boundary (all nodes have same y)
        cell_boundary_x = np.array([[0, 1, 1, 0]])
        cell_boundary_y = np.array([[0.5, 0.5, 0.5, 0.5]])
        cell_boundary_z = np.array([[0, 0, 1, 1]])
        
        x_G_rec = [[0, 0]]
        weight_G_rec = [1.0]
        
        x_G, det_J_weight = x_G_b_and_det_J_b_time_weight_3d_fuelcell_2d_boundary_interface(
            cell_boundary_x, cell_boundary_y, cell_boundary_z,
            x_G_rec, weight_G_rec
        )
        
        assert len(x_G) > 0
        assert len(det_J_weight) > 0
        
        # Check that y-coordinate is constant
        assert np.allclose(x_G[0][1], 0.5)


class TestLineGaussPoints:
    """Test Gauss point generation on line boundaries."""
    
    def test_vertical_line_segment(self):
        """Test line Gauss points for vertical segment."""
        # Vertical line from (1, 1, 0) to (1, 1, 2)
        segments = np.array([[1.0, 1.0, 0.0, 1.0, 1.0, 2.0]])
        
        x_G_line = [0.0]
        weight_G_line = [2.0]
        
        x_G, det_J_weight = x_G_and_det_J_line_3d_fuelcell_1d_boundary(
            segments, x_G_line, weight_G_line
        )
        
        assert len(x_G) == 1
        assert len(det_J_weight) == 1
        
        # Check midpoint
        assert np.isclose(x_G[0][0], 1.0)
        assert np.isclose(x_G[0][1], 1.0)
        assert np.isclose(x_G[0][2], 1.0)  # Middle of [0, 2]
        
        # Check Jacobian (line length / 2 * weight)
        expected_jacobian = 2.0 / 2.0 * 2.0  # length/2 * weight
        assert np.isclose(det_J_weight[0], expected_jacobian)
    
    def test_horizontal_line_segment(self):
        """Test line Gauss points for horizontal segment."""
        # Horizontal line from (0, 1, 1) to (2, 1, 1)
        segments = np.array([[0.0, 1.0, 1.0, 2.0, 1.0, 1.0]])
        
        x_G_line = [0.0]
        weight_G_line = [2.0]
        
        x_G, det_J_weight = x_G_and_det_J_line_3d_fuelcell_1d_boundary(
            segments, x_G_line, weight_G_line
        )
        
        assert len(x_G) == 1
        
        # Check midpoint
        assert np.isclose(x_G[0][0], 1.0)  # Middle of [0, 2]
        assert np.isclose(x_G[0][1], 1.0)
        assert np.isclose(x_G[0][2], 1.0)
    
    def test_multiple_segments(self):
        """Test multiple line segments."""
        segments = np.array([
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ])
        
        x_G_line = [0.0]
        weight_G_line = [2.0]
        
        x_G, det_J_weight = x_G_and_det_J_line_3d_fuelcell_1d_boundary(
            segments, x_G_line, weight_G_line
        )
        
        # Should have 2 Gauss points (one per segment)
        assert len(x_G) == 2
        assert len(det_J_weight) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

