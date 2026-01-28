#!/bin/bash

# Test execution script for SYCL kernels
# This script runs all SYCL kernel tests and generates a report

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_DIR="build"
OUTPUT_DIR="test_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_FILE="${OUTPUT_DIR}/sycl_test_report_${TIMESTAMP}.txt"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            REPORT_FILE="${OUTPUT_DIR}/sycl_test_report_${TIMESTAMP}.txt"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --build-dir DIR     Specify build directory (default: build)"
            echo "  --output-dir DIR    Specify output directory for results (default: test_results)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}=== SYCL Kernel Test Execution Script ===${NC}"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Initialize report
cat > $REPORT_FILE << EOF
SYCL Kernel Test Report
=======================
Generated: $(date)
Build Directory: $BUILD_DIR
Output Directory: $OUTPUT_DIR

EOF

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: Build directory '$BUILD_DIR' not found.${NC}"
    echo -e "${YELLOW}Please build the project first using ./build_sycl.sh${NC}"
    exit 1
fi

cd $BUILD_DIR

# Check if test executables exist
TESTS=(test_addVectors test_policy_map test_se_layer_nhwc test_globalAvgPool test_softmax)

echo -e "${BLUE}Checking for test executables...${NC}"
AVAILABLE_TESTS=()
for test in "${TESTS[@]}"; do
    if [ -f "$test" ] && [ -x "$test" ]; then
        AVAILABLE_TESTS+=("$test")
        echo -e "${GREEN}  ✓ $test${NC}"
    else
        echo -e "${YELLOW}  ✗ $test (not found)${NC}"
    fi
done

if [ ${#AVAILABLE_TESTS[@]} -eq 0 ]; then
    echo -e "${RED}Error: No test executables found.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Found ${#AVAILABLE_TESTS[@]} test(s). Running tests...${NC}"
echo ""

# Track test results
PASSED=0
FAILED=0
TOTAL=${#AVAILABLE_TESTS[@]}

# Run each test
for test_exe in "${AVAILABLE_TESTS[@]}"; do
    echo -e "${BLUE}Running $test_exe...${NC}"
    echo "Running $test_exe..." >> $REPORT_FILE
    echo "----------------------------------------" >> $REPORT_FILE

    # Run test and capture output
    if ./$test_exe 2>&1 | tee -a $REPORT_FILE; then
        echo -e "${GREEN}  ✓ $test_exe PASSED${NC}"
        echo "Result: PASSED" >> $REPORT_FILE
        ((PASSED++))
    else
        echo -e "${RED}  ✗ $test_exe FAILED${NC}"
        echo "Result: FAILED" >> $REPORT_FILE
        ((FAILED++))
    fi

    echo "" >> $REPORT_FILE
    echo ""
done

# Summary
echo ""
echo -e "${BLUE}=== Test Summary ===${NC}"
echo -e "Total tests: $TOTAL"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

# Add summary to report
cat >> $REPORT_FILE << EOF

=== Test Summary ===
Total tests: $TOTAL
Passed: $PASSED
Failed: $FAILED

EOF

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed successfully!${NC}"
    echo "Overall Result: ALL TESTS PASSED" >> $REPORT_FILE
    exit 0
else
    echo -e "${RED}Some tests failed. Check the report for details.${NC}"
    echo "Overall Result: SOME TESTS FAILED" >> $REPORT_FILE
    exit 1
fi