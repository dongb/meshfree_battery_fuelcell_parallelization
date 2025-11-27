"""
Unit tests for read_image module.
Tests image reading functionality for fuel cell microstructure.
"""
import pytest
from common import np, use_cupy
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _save_image(filename, img):
    """Save image to file, converting from CuPy to NumPy if needed."""
    import tifffile
    if use_cupy and hasattr(img, 'get'):
        img = img.get()
    tifffile.imwrite(filename, img)

from read_image import read_in_image


class TestReadInImage:
    """Test suite for image reading functionality."""
    
    @pytest.fixture
    def temp_image_2d(self):
        """Create a temporary 2D test image."""
        # Create a simple 2D image with 3 phases: 0=pore, 1=electrode, 2=electrolyte
        img = np.zeros((10, 10), dtype=np.uint8)
        img[0:5, 0:5] = 0  # pore
        img[5:10, 0:5] = 1  # electrode
        img[:, 5:10] = 2  # electrolyte
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            _save_image(tmp.name, img)
            yield tmp.name
        
        # Cleanup
        os.unlink(tmp.name)
    
    @pytest.fixture
    def temp_image_3d(self):
        """Create a temporary 3D test image."""
        # Create a simple 3D image with 3 phases
        img = np.zeros((10, 10, 10), dtype=np.uint8)
        img[0:5, :, :] = 0  # pore
        img[5:8, :, :] = 1  # electrode
        img[8:10, :, :] = 2  # electrolyte
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            _save_image(tmp.name, img)
            yield tmp.name
        
        # Cleanup
        os.unlink(tmp.name)
    
    def test_read_2d_image(self, temp_image_2d):
        """Test reading a 2D image."""
        img, grain_ids, num_pixels = read_in_image(
            temp_image_2d, 
            "fuel cell", 
            dimention=2
        )
        
        assert img.shape == (10, 10)
        assert len(grain_ids) == 3  # 3 unique phases
        assert 0 in grain_ids
        assert 1 in grain_ids
        assert 2 in grain_ids
        assert len(num_pixels) == 2
        assert num_pixels[0] == 10
        assert num_pixels[1] == 10
    
    def test_read_3d_image(self, temp_image_3d):
        """Test reading a 3D image."""
        img, grain_ids, num_pixels = read_in_image(
            temp_image_3d, 
            "fuel cell", 
            dimention=3
        )
        
        assert img.shape == (10, 10, 10)
        assert len(grain_ids) == 3
        assert len(num_pixels) == 3
        assert num_pixels[0] == 10
        assert num_pixels[1] == 10
        assert num_pixels[2] == 10
    
    def test_grain_id_count(self, temp_image_2d):
        """Test that grain IDs are correctly identified."""
        img, grain_ids, _ = read_in_image(temp_image_2d, "fuel cell", 2)
        
        # Check all expected grain IDs are present
        for grain_id in [0, 1, 2]:
            assert grain_id in grain_ids
    
    def test_pixel_dimensions(self, temp_image_3d):
        """Test that pixel dimensions are correctly extracted."""
        img, _, num_pixels = read_in_image(temp_image_3d, "fuel cell", 3)
        
        expected_shape = img.shape
        assert num_pixels[0] == expected_shape[0]
        assert num_pixels[1] == expected_shape[1]
        assert num_pixels[2] == expected_shape[2]
    
    def test_image_data_type(self, temp_image_2d):
        """Test that returned image is ndarray (numpy or cupy)."""
        img, _, _ = read_in_image(temp_image_2d, "fuel cell", 2)
        # Check it's an array from the backend we're using
        assert hasattr(img, 'shape') and hasattr(img, 'dtype')
    
    def test_grain_ids_are_integers(self, temp_image_3d):
        """Test that grain IDs are integers."""
        _, grain_ids, _ = read_in_image(temp_image_3d, "fuel cell", 3)
        
        for grain_id in grain_ids:
            assert isinstance(grain_id, (int, np.integer))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

