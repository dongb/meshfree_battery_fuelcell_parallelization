"""
Pytest configuration and shared fixtures for fuel_cell_3D tests.
"""
import pytest
from common import np
import tempfile
import os
import tifffile


@pytest.fixture(scope="session")
def sample_material_constants():
    """Material constants for testing."""
    return {
        'Fday': 9.6485e4,
        'R': 8.3145e0,
        'T': 1273.2,
        'E_electrolyte': 132.69e9,
        'nu_electrolyte': 0.33,
        'E_electrode': 130.0e9,
        'nu_electrode': 0.33,
        'diffusion_electrolyte': 0.035,
        'diffusion_electrode': 1.0e-9,
        'diffusion_pore': 1.0e-7
    }


@pytest.fixture(scope="session")
def gauss_quadrature_2d():
    """2D Gauss quadrature points and weights."""
    return {
        'points': [[-1/np.sqrt(3), -1/np.sqrt(3)],
                   [-1/np.sqrt(3), 1/np.sqrt(3)],
                   [1/np.sqrt(3), -1/np.sqrt(3)],
                   [1/np.sqrt(3), 1/np.sqrt(3)]],
        'weights': [1.0, 1.0, 1.0, 1.0]
    }


@pytest.fixture(scope="session")
def gauss_quadrature_3d():
    """3D Gauss quadrature points and weights."""
    s = 1/np.sqrt(3)
    return {
        'points': [
            [-s, -s, -s], [s, -s, -s],
            [-s, s, -s], [s, s, -s],
            [-s, -s, s], [s, -s, s],
            [-s, s, s], [s, s, s]
        ],
        'weights': [1.0] * 8
    }


@pytest.fixture
def simple_cube_mesh():
    """Simple cube mesh for testing."""
    return {
        'nodes': np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=np.float64),
        'cell_x': np.array([[0, 1, 1, 0, 0, 1, 1, 0]]),
        'cell_y': np.array([[0, 0, 1, 1, 0, 0, 1, 1]]),
        'cell_z': np.array([[0, 0, 0, 0, 1, 1, 1, 1]])
    }


@pytest.fixture
def temp_test_image_3d():
    """Create a temporary 3D test image."""
    img = np.zeros((4, 4, 4), dtype=np.uint8)
    img[0:2, :, :] = 0  # pore
    img[2, :, :] = 1    # electrode
    img[3, :, :] = 2    # electrolyte
    
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        tifffile.imwrite(tmp.name, img)
        yield tmp.name
    
    # Cleanup
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)


@pytest.fixture
def tolerance_config():
    """Tolerance configuration for numerical tests."""
    return {
        'rtol': 1e-5,
        'atol': 1e-8,
        'partition_of_unity_tol': 0.1,
        'shape_func_bound_tol': 0.05
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Helper functions for tests

def create_simple_shape_functions(n_gauss, n_nodes, normalized=True):
    """Create simple shape functions for testing."""
    shape_func = np.random.rand(n_gauss, n_nodes)
    if normalized:
        shape_func = shape_func / shape_func.sum(axis=1, keepdims=True)
    return shape_func


def verify_partition_of_unity(shape_func, tol=0.1):
    """Verify partition of unity property."""
    row_sums = shape_func.sum(axis=1)
    return np.allclose(row_sums, 1.0, atol=tol)


def create_test_concentration_field(n_nodes, c_min=0.0, c_max=1.0):
    """Create a test concentration field."""
    return np.linspace(c_min, c_max, n_nodes)

