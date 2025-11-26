# Fuel Cell 3D Test Suite

Comprehensive test suite for the fuel_cell_3D simulation package.

## Overview

This test suite provides comprehensive coverage of the fuel cell 3D simulation codebase, including:

- **Unit Tests**: Test individual functions and modules in isolation
- **Integration Tests**: Test interactions between modules
- **Property Tests**: Verify mathematical properties and physical constraints

## Test Structure

```
tests/
├── __init__.py                    # Package initialization
├── conftest.py                    # Shared fixtures and configuration
├── pytest.ini                     # Pytest configuration
├── requirements-test.txt          # Test dependencies
├── README.md                      # This file
├── test_read_image.py            # Image reading tests
├── test_buttler_volmer.py        # Butler-Volmer equations tests
├── test_eva_at_gauss_points.py   # Gauss point evaluation tests
├── test_get_nodes_gauss_points.py # Node and Gauss point generation
├── test_shape_functions.py       # Shape function computation
├── test_diffusion_matrix.py      # Diffusion matrix assembly
├── test_mechanical_matrix.py     # Mechanical matrix assembly
└── test_integration.py           # Integration tests
```

## Running Tests

### Run all tests
```bash
cd fuel_cell_3D/tests
pytest
```

### Run specific test file
```bash
pytest test_read_image.py
```

### Run specific test class or function
```bash
pytest test_read_image.py::TestReadInImage::test_read_2d_image
```

### Run with coverage
```bash
pytest --cov=.. --cov-report=html
```

### Run parallel tests (faster)
```bash
pytest -n auto
```

### Run only unit tests
```bash
pytest -m unit
```

### Run only integration tests
```bash
pytest -m integration
```

### Skip slow tests
```bash
pytest -m "not slow"
```

## Test Categories

### Unit Tests

Test individual functions and modules:

- `test_read_image.py`: Image I/O operations
- `test_buttler_volmer.py`: Material property functions
- `test_eva_at_gauss_points.py`: Field evaluation
- `test_get_nodes_gauss_points.py`: Geometry generation
- `test_shape_functions.py`: Shape function computation
- `test_diffusion_matrix.py`: Diffusion matrix assembly
- `test_mechanical_matrix.py`: Mechanical matrix assembly

### Integration Tests

Test module interactions:

- `test_integration.py`: End-to-end workflow tests

## Key Test Features

### 1. Image Processing
- Reading 2D and 3D TIFF images
- Grain identification
- Dimension extraction

### 2. Geometry Generation
- Node generation from images
- Gauss point generation (domain, boundary, line)
- Boundary node identification

### 3. Shape Functions
- RKPM shape function computation
- Moment matrix computation
- Gradient calculations
- Partition of unity verification

### 4. Material Properties
- Butler-Volmer kinetics
- Diffusivity functions
- Open circuit potential
- Lattice parameters

### 5. Matrix Assembly
- Diffusion stiffness matrix
- Mechanical stiffness matrix
- Force vector assembly
- Boundary condition enforcement

### 6. Physical Constraints
- Partition of unity
- Shape function boundedness
- Matrix symmetry properties
- Positive definiteness

## Adding New Tests

When adding new functionality, please add corresponding tests:

1. **Unit tests** for new functions
2. **Integration tests** if the function interacts with other modules
3. **Property tests** for mathematical properties

Example test structure:

```python
import pytest
import numpy as np

class TestNewFeature:
    """Test suite for new feature."""
    
    @pytest.fixture
    def setup_data(self):
        """Create test data."""
        return {'data': np.array([1, 2, 3])}
    
    def test_basic_functionality(self, setup_data):
        """Test basic use case."""
        result = new_function(setup_data['data'])
        assert result is not None
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Empty input
        result = new_function(np.array([]))
        assert result.size == 0
    
    def test_properties(self):
        """Test mathematical properties."""
        result = new_function(np.array([1, 2, 3]))
        assert np.all(result >= 0)  # Non-negative
```

## Continuous Integration

The test suite is designed to be run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    cd fuel_cell_3D/tests
    pytest --cov=.. --cov-report=xml -v
```

## Troubleshooting

### Import Errors
If you see import errors, make sure the parent directory is in the Python path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Numba Warnings
Numba may issue warnings during first run. These can be ignored or filtered:
```bash
pytest --disable-warnings
```

### Memory Issues
For large tests, you may need to increase available memory or run tests serially:
```bash
pytest -n 1  # Run serially
```

## Test Coverage

Aim for >80% code coverage. Generate coverage report:

```bash
pytest --cov=.. --cov-report=html
# Open htmlcov/index.html in browser
```

## Performance Testing

For performance-critical code, use pytest-benchmark:

```python
def test_performance(benchmark):
    result = benchmark(expensive_function, args)
    assert result is not None
```

## Known Issues

1. Some tests may be slow due to Numba JIT compilation on first run
2. Large 3D images may require significant memory
3. Some numerical tests may have platform-dependent tolerances

## Contributing

When contributing tests:

1. Follow existing test structure
2. Use descriptive test names
3. Add docstrings explaining what is tested
4. Use fixtures for common setup
5. Test both success and failure cases
6. Verify edge cases and boundary conditions

## Contact

For questions or issues with tests, please open an issue on the repository.

