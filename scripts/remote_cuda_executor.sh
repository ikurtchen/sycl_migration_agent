#!/bin/bash

# Remote CUDA Execution Script
# Usage: ./remote_cuda_executor.sh <kernel_name> <test_args>

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
NVIDIA_HOST=$(jq -r '.nvidia_server.host' "$CONFIG_FILE")
NVIDIA_USER=$(jq -r '.nvidia_server.user' "$CONFIG_FILE")
NVIDIA_KEY=$(jq -r '.nvidia_server.key_file' "$CONFIG_FILE" | sed 's|^~/|'"$HOME"'/|')
NVIDIA_WORKSPACE=$(jq -r '.nvidia_server.workspace' "$CONFIG_FILE")
CUDA_PATH=$(jq -r '.nvidia_server.cuda_path' "$CONFIG_FILE")
GPU_ID=$(jq -r '.nvidia_server.gpu_id' "$CONFIG_FILE")
TIMEOUT=$(jq -r '.execution.timeout_seconds' "$CONFIG_FILE")

# Parse arguments
KERNEL_NAME=${1:-"test_kernel"}
TEST_ARGS=${2:-""}

# Create results directory
RESULTS_DIR="$PROJECT_ROOT/results/cuda_outputs/$(date +%Y%m%d_%H%M%S)_$KERNEL_NAME"
mkdir -p "$RESULTS_DIR"

echo "=== CUDA Remote Execution for $KERNEL_NAME ==="
echo "NVIDIA Server: $NVIDIA_HOST"
echo "Workspace: $NVIDIA_WORKSPACE"
echo "Results Directory: $RESULTS_DIR"

# Create remote execution script
REMOTE_SCRIPT=$(cat <<'EOF'
#!/bin/bash
set -e

# Set CUDA environment
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Change to workspace
cd $WORKSPACE

# List available GPUs
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits

# Clean previous builds
rm -f *.o *.exe core results_*.json output_*.bin

# Build tests
echo "Building CUDA tests..."
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
echo "Running CUDA tests..."
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

# Create benchmark data
cat > benchmark_$KERNEL_NAME.json <<BENCHMARK_EOF
{
  "kernel": "$KERNEL_NAME",
  "platform": "CUDA",
  "timestamp": "$(date -Iseconds)",
  "execution_time_seconds": $EXECUTION_TIME,
  "gpu_info": {
    "device_id": $GPU_ID,
    "name": "$(nvidia-smi --query-gpu=name --id=$GPU_ID --format=csv,noheader,nounits)"
  }
}
BENCHMARK_EOF

# Package results
tar -czf results_$KERNEL_NAME.tar.gz results_$KERNEL_NAME.json benchmark_$KERNEL_NAME.json output_*.bin *.log 2>/dev/null || true

echo "CUDA execution completed successfully"
EOF
)

# Replace variables in remote script
REMOTE_SCRIPT=$(echo "$REMOTE_SCRIPT" | sed "s|\$CUDA_PATH|$CUDA_PATH|g")
REMOTE_SCRIPT=$(echo "$REMOTE_SCRIPT" | sed "s|\$WORKSPACE|$NVIDIA_WORKSPACE|g")
REMOTE_SCRIPT=$(echo "$REMOTE_SCRIPT" | sed "s|\$GPU_ID|$GPU_ID|g")
REMOTE_SCRIPT=$(echo "$REMOTE_SCRIPT" | sed "s|\$KERNEL_NAME|$KERNEL_NAME|g")
REMOTE_SCRIPT=$(echo "$REMOTE_SCRIPT" | sed "s|\$TEST_ARGS|$TEST_ARGS|g")

# Execute remote script
echo "Executing on remote server..."
ssh -i "$NVIDIA_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=30 "$NVIDIA_USER@$NVIDIA_HOST" "$REMOTE_SCRIPT"

# Transfer results back
echo "Transferring results..."
scp -i "$NVIDIA_KEY" -o StrictHostKeyChecking=no "$NVIDIA_USER@$NVIDIA_HOST:$NVIDIA_WORKSPACE/results_$KERNEL_NAME.tar.gz" "$RESULTS_DIR/"

# Extract results
cd "$RESULTS_DIR"
tar -xzf results_$KERNEL_NAME.tar.gz

# Clean up remote
if [ "$(jq -r '.execution.cleanup_remote_after' "$CONFIG_FILE")" = "true" ]; then
    ssh -i "$NVIDIA_KEY" -o StrictHostKeyChecking=no "$NVIDIA_USER@$NVIDIA_HOST" "cd $NVIDIA_WORKSPACE && rm -f results_$KERNEL_NAME.tar.gz results_$KERNEL_NAME.json benchmark_$KERNEL_NAME.json output_*.bin"
fi

echo "CUDA execution completed. Results in: $RESULTS_DIR"
echo "Generated files:"
ls -la "$RESULTS_DIR"