#!/bin/bash
# Test runner script for fuel_cell_3D

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==================================${NC}"
echo -e "${GREEN}Fuel Cell 3D Test Suite${NC}"
echo -e "${GREEN}==================================${NC}"
echo ""

# Check if in correct directory
if [ ! -d "tests" ]; then
    echo -e "${RED}Error: tests directory not found${NC}"
    echo "Please run this script from the fuel_cell_3D directory"
    exit 1
fi

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${YELLOW}pytest not found. Installing dependencies...${NC}"
    pip install -r tests/requirements-test.txt
fi

# Parse command line arguments
TEST_TYPE="all"
COVERAGE=false
PARALLEL=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--unit)
            TEST_TYPE="unit"
            shift
            ;;
        -i|--integration)
            TEST_TYPE="integration"
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: ./run_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -u, --unit          Run only unit tests"
            echo "  -i, --integration   Run only integration tests"
            echo "  -c, --coverage      Generate coverage report"
            echo "  -p, --parallel      Run tests in parallel"
            echo "  -v, --verbose       Verbose output"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest tests"

if [ "$TEST_TYPE" == "unit" ]; then
    echo -e "${YELLOW}Running unit tests only${NC}"
    PYTEST_CMD="$PYTEST_CMD -m unit"
elif [ "$TEST_TYPE" == "integration" ]; then
    echo -e "${YELLOW}Running integration tests only${NC}"
    PYTEST_CMD="$PYTEST_CMD -m integration"
else
    echo -e "${YELLOW}Running all tests${NC}"
fi

if [ "$COVERAGE" = true ]; then
    echo -e "${YELLOW}Coverage enabled${NC}"
    PYTEST_CMD="$PYTEST_CMD --cov=. --cov-report=html --cov-report=term-missing"
fi

if [ "$PARALLEL" = true ]; then
    echo -e "${YELLOW}Parallel execution enabled${NC}"
    PYTEST_CMD="$PYTEST_CMD -n auto"
fi

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -vv"
else
    PYTEST_CMD="$PYTEST_CMD -v"
fi

echo ""
echo -e "${GREEN}Running: $PYTEST_CMD${NC}"
echo ""

# Run tests
if $PYTEST_CMD; then
    echo ""
    echo -e "${GREEN}==================================${NC}"
    echo -e "${GREEN}All tests passed! ✓${NC}"
    echo -e "${GREEN}==================================${NC}"
    
    if [ "$COVERAGE" = true ]; then
        echo ""
        echo -e "${YELLOW}Coverage report generated in htmlcov/index.html${NC}"
    fi
    
    exit 0
else
    echo ""
    echo -e "${RED}==================================${NC}"
    echo -e "${RED}Some tests failed! ✗${NC}"
    echo -e "${RED}==================================${NC}"
    exit 1
fi

