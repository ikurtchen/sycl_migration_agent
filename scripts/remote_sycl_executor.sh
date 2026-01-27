#!/bin/bash

# Remote SYCL Execution Script
# Usage: ./remote_sycl_executor.sh <kernel_name> <test_args>

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

# Parse configuration
INTEL_HOST=$(jq -r '.intel_server.host' "$CONFIG_FILE")
INTEL_USER=$(jq -r '.intel_server.user' "$CONFIG_FILE")
INTEL_KEY=$(jq -r '.intel_server.key_file' "$CONFIG_FILE" | sed 's|^~/|'"$HOME"'/|')
INTEL_WORKSPACE=$(jq -r '.intel_server.workspace' "$CONFIG_FILE")
ONEAPI_PATH=$(jq -r '.intel_server.oneapi_path' "$CONFIG_FILE")
DEVICE_SELECTOR=$(jq -r '.intel_server.device_selector' "$CONFIG_FILE")
TIMEOUT=$(jq -r '.execution.timeout_seconds' "$CONFIG_FILE")

# Parse arguments
KERNEL_NAME=${1:-"test_kernel"}
TEST_ARGS=${2:-""}

# Create results directory
RESULTS_DIR="$PROJECT_ROOT/results/sycl_outputs/$(date +%Y%m%d_%H%M%S)_$KERNEL_NAME"
mkdir -p "$RESULTS_DIR"

echo "=== SYCL Remote Execution for $KERNEL_NAME ==="
echo "Intel Server: $INTEL_HOST"
echo "Workspace: $INTEL_WORKSPACE"
echo "Results Directory: $RESULTS_DIR"

# Create remote execution script
REMOTE_SCRIPT=$(cat <<'EOF'
#!/bin/bash
set -e

# Set oneAPI environment
source $ONEAPI_PATH/setvars.sh > /dev/null 2>&1

# Set device selector
export ONEAPI_DEVICE_SELECTOR=$DEVICE_SELECTOR

# Change to workspace
cd $WORKSPACE

# List available devices
echo "Available SYCL devices:"
sycl-ls

# Clean previous builds
rm -f *.o *.exe core results_*.json output_*.bin

# Build tests
echo "Building SYCL tests..."
if [ -f "CMakeLists.txt" ]; then
    mkdir -p build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j$(nproc)
    cd ..
else
    echo "No CMakeLists.txt found, looking for Makefile..."
    if [ -f "Makefile" ]; then
        make clean
        make -j$(nproc)
    else
        echo "Error: No build system found"
        exit 1
    fi
fi

# Run tests
echo "Running SYCL tests..."
TEST_EXECUTABLE="./build/${KERNEL_NAME}_test"
if [ ! -f "$TEST_EXECUTABLE" ]; then
    TEST_EXECUTABLE="./${KERNEL_NAME}_test"
fi

if [ ! -f "$TEST_EXECUTABLE" ]; then
    echo "Error: Test executable not found"
    exit 1
fi

# Run with GTest JSON output and timing
START_TIME=$(date +%s.%N)
$TEST_EXECUTABLE $TEST_ARGS --gtest_output=json:results_$KERNEL_NAME.json
END_TIME=$(date +%s.%N)

# Calculate execution time
EXECUTION_TIME=$(echo "$END_TIME - $START_TIME" | bc)
echo "Execution time: $EXECUTION_TIME seconds"

# Get device info
DEVICE_INFO=$(sycl-ls | head -1)

# Create benchmark data
cat > benchmark_$KERNEL_NAME.json <<BENCHMARK_EOF
{
  "kernel": "$KERNEL_NAME",
  "platform": "SYCL",
  "timestamp": "$(date -Iseconds)",
  "execution_time_seconds": $EXECUTION_TIME,
  "device_selector": "$DEVICE_SELECTOR",
  "device_info": "$DEVICE_INFO"
}
BENCHMARK_EOF

# Package results
tar -czf results_$KERNEL_NAME.tar.gz results_$KERNEL_NAME.json benchmark_$KERNEL_NAME.json output_*.bin *.log 2>/dev/null || true

echo "SYCL execution completed successfully"
EOF
)

# Replace variables in remote script
REMOTE_SCRIPT=$(echo "$REMOTE_SCRIPT" | sed "s|\$ONEAPI_PATH|$ONEAPI_PATH|g")
REMOTE_SCRIPT=$(echo "$REMOTE_SCRIPT" | sed "s|\$WORKSPACE|$INTEL_WORKSPACE|g")
REMOTE_SCRIPT=$(echo "$REMOTE_SCRIPT" | sed "s|\$DEVICE_SELECTOR|$DEVICE_SELECTOR|g")
REMOTE_SCRIPT=$(echo "$REMOTE_SCRIPT" | sed "s|\$KERNEL_NAME|$KERNEL_NAME|g")
REMOTE_SCRIPT=$(echo "$REMOTE_SCRIPT" | sed "s|\$TEST_ARGS|$TEST_ARGS|g")

# Execute remote script
echo "Executing on remote server..."
ssh -i "$INTEL_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=30 "$INTEL_USER@$INTEL_HOST" "$REMOTE_SCRIPT"

# Transfer results back
echo "Transferring results..."
scp -i "$INTEL_KEY" -o StrictHostKeyChecking=no "$INTEL_USER@$INTEL_HOST:$INTEL_WORKSPACE/results_$KERNEL_NAME.tar.gz" "$RESULTS_DIR/"

# Extract results
cd "$RESULTS_DIR"
tar -xzf results_$KERNEL_NAME.tar.gz

# Clean up remote
if [ "$(jq -r '.execution.cleanup_remote_after' "$CONFIG_FILE")" = "true" ]; then
    ssh -i "$INTEL_KEY" -o StrictHostKeyChecking=no "$INTEL_USER@$INTEL_HOST" "cd $INTEL_WORKSPACE && rm -f results_$KERNEL_NAME.tar.gz results_$KERNEL_NAME.json benchmark_$KERNEL_NAME.json output_*.bin"
fi

echo "SYCL execution completed. Results in: $RESULTS_DIR"
echo "Generated files:"
ls -la "$RESULTS_DIR"