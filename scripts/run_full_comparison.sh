#!/bin/bash

# Full CUDA vs SYCL Comparison Script
# This script executes both CUDA and SYCL tests and compares the results

set -e

# Source configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"
CONFIG_FILE="$PROJECT_ROOT/config/ssh_config.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found at $CONFIG_FILE"
    echo "Please copy config/ssh_config_template.json to config/ssh_config.json and configure"
    exit 1
fi

# Parse arguments
KERNEL_NAME=${1:-"test_kernel"}
TEST_ARGS=${2:-""}
CLEANUP=${3:-"true"}

# Parse tolerances from config
RTOL=$(jq -r '.comparison.relative_tolerance' "$CONFIG_FILE")
ATOL=$(jq -r '.comparison.absolute_tolerance' "$CONFIG_FILE")

echo "=== Full CUDA vs SYCL Comparison for $KERNEL_NAME ==="
echo "Relative Tolerance: $RTOL"
echo "Absolute Tolerance: $ATOL"

# Step 1: Execute CUDA tests
echo
echo "Step 1: Executing CUDA tests on NVIDIA server..."
$SCRIPT_DIR/remote_cuda_executor.sh "$KERNEL_NAME" "$TEST_ARGS"

# Find CUDA results directory
CUDA_RESULTS_DIR=$(ls -1dt "$PROJECT_ROOT/results/cuda_outputs/"*"$KERNEL_NAME" | head -1)
echo "CUDA results directory: $CUDA_RESULTS_DIR"

# Step 2: Execute SYCL tests
echo
echo "Step 2: Executing SYCL tests on Intel server..."
$SCRIPT_DIR/remote_sycl_executor.sh "$KERNEL_NAME" "$TEST_ARGS"

# Find SYCL results directory
SYCL_RESULTS_DIR=$(ls -1dt "$PROJECT_ROOT/results/sycl_outputs/"*"$KERNEL_NAME" | head -1)
echo "SYCL results directory: $SYCL_RESULTS_DIR"

# Step 3: Compare results
echo
echo "Step 3: Comparing CUDA and SYCL results..."
python3 "$SCRIPT_DIR/compare_results.py" \
    --kernel "$KERNEL_NAME" \
    --cuda-dir "$CUDA_RESULTS_DIR" \
    --sycl-dir "$SYCL_RESULTS_DIR" \
    --rtol "$RTOL" \
    --atol "$ATOL" \
    --output-dir "$PROJECT_ROOT/results/comparison_reports"

# Step 4: Generate performance summary
echo
echo "Step 4: Generating performance summary..."

# Load the latest comparison report
LATEST_REPORT=$(ls -1t "$PROJECT_ROOT/results/comparison_reports/comparison_${KERNEL_NAME}_"*.json | head -1)

if [ -f "$LATEST_REPORT" ]; then
    # Extract performance metrics
    python3 - <<EOF
import json
import sys

report_file = "$LATEST_REPORT"
with open(report_file, 'r') as f:
    report = json.load(f)

print("=== Test Results Summary ===")
print(f"Kernel: {report['kernel']}")
print(f"Timestamp: {report['timestamp']}")

if 'summary' in report:
    summary = report['summary']
    print(f"\nNumerical Validation:")
    print(f"  Overall Match: {summary['overall_match']}")
    print(f"  Total Comparisons: {summary['total_comparisons']}")
    print(f"  Passed: {summary['matched_comparisons']}")
    print(f"  Failed: {summary['failed_comparisons']}")

if 'performance_comparison' in report:
    perf = report['performance_comparison']
    print(f"\nPerformance Comparison:")
    print(f"  CUDA Execution Time: {perf['cuda_execution_time']:.4f}s")
    print(f"  SYCL Execution Time: {perf['sycl_execution_time']:.4f}s")
    print(f"  SYCL Speedup: {perf['speedup_ratio']:.2f}x")
    print(f"  SYCL Performance: {perf['performance_percentage']:.1f}% of CUDA")

if not report.get('summary', {}).get('overall_match', True):
    print(f"\n⚠️  WARNING: Numerical validation failed!")
    print("Check the detailed report for mismatch analysis:")
    print(f"  {report_file}")
    sys.exit(1)
else:
    print(f"\n✓ SUCCESS: Numerical validation passed!")
    print(f"Detailed report: {report_file}")
EOF
else
    echo "Warning: No comparison report found"
fi

# Step 5: Cleanup if requested
if [ "$CLEANUP" = "true" ]; then
    echo
    echo "Step 5: Cleanup temporary files..."
    # Keep only the latest results for each kernel
    find "$PROJECT_ROOT/results/cuda_outputs" -name "*$KERNEL_NAME*" -type d | sort -r | tail -n +2 | xargs rm -rf 2>/dev/null || true
    find "$PROJECT_ROOT/results/sycl_outputs" -name "*$KERNEL_NAME*" -type d | sort -r | tail -n +2 | xargs rm -rf 2>/dev/null || true
fi

echo
echo "=== Comparison completed successfully ==="

# Return success/failure based on numerical validation
if [ -f "$LATEST_REPORT" ]; then
    MATCH_STATUS=$(python3 -c "import json; print(json.load(open('$LATEST_REPORT')).get('summary', {}).get('overall_match', False))")
    if [ "$MATCH_STATUS" = "True" ]; then
        exit 0
    else
        exit 1
    fi
fi

exit 0