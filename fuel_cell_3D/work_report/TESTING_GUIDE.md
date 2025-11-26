# Complete Testing Guide for fuel_cell_3D

## Quick Start

```bash
cd /home/bod/workspace/meshfree_battery_fuelcell_parallelization/fuel_cell_3D

# Install test dependencies
pip install -r tests/requirements-test.txt

# Run all tests
./run_tests.sh

# Or use pytest directly
pytest tests -v
```

## What's Been Created

### Test Files (93 test cases total)

1. **`tests/test_read_image.py`** - 8 tests
   - Image I/O operations
   - 2D and 3D image handling
   - Grain identification
   - Dimension extraction

2. **`tests/test_buttler_volmer.py`** - 24 tests
   - Exchange current density functions
   - Lattice parameter calculations
   - Diffusivity with damage
   - Open circuit potential
   - Butler-Volmer kinetics

3. **`tests/test_eva_at_gauss_points.py`** - 10 tests
   - Field evaluation at Gauss points
   - Shape function interpolation
   - Boundary vs domain evaluation

4. **`tests/test_get_nodes_gauss_points.py`** - 14 tests
   - Node generation from images
   - 3D domain Gauss points
   - 2D boundary Gauss points
   - 1D line Gauss points
   - Jacobian computation

5. **`tests/test_shape_functions.py`** - 11 tests
   - RKPM shape function computation
   - Moment matrix calculation
   - Shape function gradients
   - Partition of unity verification

6. **`tests/test_diffusion_matrix.py`** - 7 tests
   - Diffusion matrix assembly
   - Matrix properties (symmetry, positivity)
   - Force vector assembly
   - Boundary condition enforcement

7. **`tests/test_mechanical_matrix.py`** - 9 tests
   - Mechanical stiffness tensor
   - Damage model
   - Stiffness matrix assembly
   - Force vector with eigenstrain

8. **`tests/test_integration.py`** - 10 tests
   - End-to-end workflow tests
   - Module interaction tests
   - Consistency checks

### Configuration Files

- **`tests/conftest.py`**: Shared fixtures and test utilities
- **`tests/pytest.ini`**: Pytest configuration
- **`tests/requirements-test.txt`**: Test dependencies
- **`tests/README.md`**: Detailed testing documentation
- **`run_tests.sh`**: Convenient test runner script

### Documentation

- **`TEST_SUITE_SUMMARY.md`**: Complete test suite overview
- **`TESTING_GUIDE.md`**: This guide

## Test Categories

### Unit Tests
Test individual functions in isolation:
```bash
pytest tests/test_read_image.py -v
pytest tests/test_buttler_volmer.py -v
```

### Integration Tests
Test module interactions:
```bash
pytest tests/test_integration.py -v
```

### Run by Marker
```bash
pytest tests -m unit          # Unit tests only
pytest tests -m integration   # Integration tests only
pytest tests -m "not slow"    # Skip slow tests
```

## Test Runner Options

```bash
# Basic usage
./run_tests.sh                 # Run all tests

# With options
./run_tests.sh --unit          # Unit tests only
./run_tests.sh --integration   # Integration tests only
./run_tests.sh --coverage      # With coverage report
./run_tests.sh --parallel      # Parallel execution
./run_tests.sh --verbose       # Verbose output

# Combined
./run_tests.sh --coverage --parallel
```

## Coverage Analysis

```bash
# Generate HTML coverage report
./run_tests.sh --coverage

# View report
firefox htmlcov/index.html  # Linux
open htmlcov/index.html     # macOS
start htmlcov/index.html    # Windows
```

## Pytest Advanced Usage

### Run Specific Test
```bash
pytest tests/test_read_image.py::TestReadInImage::test_read_2d_image -v
```

### Run Tests Matching Pattern
```bash
pytest tests -k "matrix" -v        # All tests with "matrix" in name
pytest tests -k "not slow" -v      # Exclude slow tests
```

### Parallel Execution
```bash
pytest tests -n auto               # Auto-detect CPU count
pytest tests -n 4                  # Use 4 workers
```

### Stop on First Failure
```bash
pytest tests -x                    # Stop on first failure
pytest tests --maxfail=3           # Stop after 3 failures
```

### Show Print Statements
```bash
pytest tests -s                    # Show print output
```

### Detailed Tracebacks
```bash
pytest tests --tb=long             # Long tracebacks
pytest tests --tb=short            # Short tracebacks (default)
```

## Test Development Workflow

### 1. Write Tests First (TDD)
```python
# tests/test_new_feature.py
def test_new_function():
    """Test new functionality."""
    result = new_function(input_data)
    assert result == expected_output
```

### 2. Run Tests (They Should Fail)
```bash
pytest tests/test_new_feature.py -v
```

### 3. Implement Feature
```python
# new_module.py
def new_function(input_data):
    # Implementation
    return output
```

### 4. Tests Pass
```bash
pytest tests/test_new_feature.py -v
```

## Common Test Patterns

### Testing with Fixtures
```python
@pytest.fixture
def sample_data():
    return {'x': np.array([1, 2, 3])}

def test_with_fixture(sample_data):
    result = process(sample_data['x'])
    assert result is not None
```

### Parametrized Tests
```python
@pytest.mark.parametrize("input,expected", [
    (0.0, 0.0),
    (0.5, 0.5),
    (1.0, 1.0),
])
def test_multiple_inputs(input, expected):
    assert function(input) == expected
```

### Testing Exceptions
```python
def test_raises_error():
    with pytest.raises(ValueError):
        function_that_should_fail()
```

### Testing Warnings
```python
def test_warning():
    with pytest.warns(UserWarning):
        function_that_warns()
```

## Debugging Tests

### Use pdb
```python
def test_debug():
    import pdb; pdb.set_trace()
    result = function()
    assert result is not None
```

### Verbose Output
```bash
pytest tests -vv -s
```

### Show Local Variables on Failure
```bash
pytest tests --showlocals
```

## Best Practices

### ✅ DO
- Write descriptive test names
- Use fixtures for setup/teardown
- Test edge cases and boundary conditions
- Keep tests independent
- Use appropriate assertions
- Document test purpose with docstrings

### ❌ DON'T
- Test implementation details
- Create interdependent tests
- Use global state
- Test framework code
- Write tests without assertions

## Continuous Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r fuel_cell_3D/tests/requirements-test.txt
      - name: Run tests
        run: |
          cd fuel_cell_3D
          pytest tests --cov=. --cov-report=xml -v
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Performance Testing

### Benchmark Tests
```python
def test_performance(benchmark):
    result = benchmark(expensive_function, args)
    assert result is not None
```

### Profiling
```bash
pytest tests --profile
pytest tests --profile-svg
```

## Troubleshooting

### Import Errors
```python
# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Numba Warnings
```bash
pytest tests --disable-warnings
```

### Out of Memory
```bash
pytest tests -n 1  # Run serially
```

### Slow Tests
```bash
pytest tests -m "not slow" -v
```

## Test Maintenance

### Update Tests When:
- Adding new features
- Fixing bugs
- Refactoring code
- Changing APIs

### Review Coverage
```bash
./run_tests.sh --coverage
# Aim for >80% coverage
```

### Run Tests Before:
- Committing code
- Creating pull requests
- Deploying to production

## Example Test Session

```bash
# Terminal 1: Watch mode (requires pytest-watch)
cd fuel_cell_3D
ptw tests -- -v

# Terminal 2: Make changes
vim shape_function_in_domain.py

# Tests automatically rerun on save
```

## Test Results Interpretation

### All Green ✓
```
tests/test_read_image.py::TestReadInImage::test_read_2d_image PASSED
tests/test_read_image.py::TestReadInImage::test_read_3d_image PASSED
...
========== 93 passed in 45.23s ==========
```

### Some Failed ✗
```
tests/test_shape_functions.py::TestComputePhiM::test_partition_of_unity FAILED

AssertionError: assert 0.95 < 1.0 < 1.05
Partition of unity violated
```

### Skipped Tests ⊘
```
tests/test_integration.py::TestEndToEnd::test_full_simulation SKIPPED
Reason: Requires large test data
```

## Getting Help

### Documentation
- Read `tests/README.md` for detailed information
- Check `TEST_SUITE_SUMMARY.md` for overview

### Pytest Help
```bash
pytest --help
pytest --markers  # Show available markers
```

### Community
- pytest documentation: https://docs.pytest.org/
- NumPy testing: https://numpy.org/doc/stable/reference/testing.html
- SciPy testing: https://docs.scipy.org/doc/scipy/dev/testing.html

## Summary

You now have a complete, professional-grade test suite for `fuel_cell_3D` with:

✅ 93 comprehensive test cases
✅ Unit and integration tests
✅ Shared fixtures and utilities
✅ Coverage reporting
✅ Parallel execution support
✅ CI/CD ready
✅ Comprehensive documentation
✅ Easy-to-use test runner

**Get started now:**
```bash
cd fuel_cell_3D
./run_tests.sh
```

Happy testing! 🧪

