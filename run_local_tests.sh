#!/bin/bash
#
# Local Test Runner for CUDA and SYCL
#
# This script builds and runs CUDA and SYCL tests locally with proper
# environment setup, logging, and result collection.
#
# Usage:
#   ./run_local_tests.sh [OPTIONS]
#   ./run_local_tests.sh --cuda-only
#   ./run_local_tests.sh --sycl-only
#   ./run_local_tests.sh --clean
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
RUN_CUDA=true
RUN_SYCL=true
CLEAN_BUILD=false
VERBOSE=false
BENCHMARK=true

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
CUDA_DIR="$PROJECT_ROOT/cuda"
SYCL_DIR="$PROJECT_ROOT/sycl"
RESULTS_DIR="$PROJECT_ROOT/results"
LOGS_DIR="$PROJECT_ROOT/logs"

# Timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_LOGS_DIR="$LOGS_DIR/run_$TIMESTAMP"
CUDA_RESULTS_DIR="$RESULTS_DIR/cuda_$TIMESTAMP"
SYCL_RESULTS_DIR="$RESULTS_DIR/sycl_$TIMESTAMP"

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --cuda-only)
                RUN_CUDA=true
                RUN_SYCL=false
                shift
                ;;
            --sycl-only)
                RUN_SYCL=true
                RUN_CUDA=false
                shift
                ;;
            --clean)
                CLEAN_BUILD=true
                shift
                ;;
            --verbose|-v)
                VERBOSE=true
                shift
                ;;
            --no-benchmark)
                BENCHMARK=false
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                show_help
                exit 1
                ;;
        esac
    done
}

# Display help information
show_help() {
    cat << EOF
Local Test Runner for CUDA and SYCL

Usage:
    $0 [OPTIONS]

Options:
    --cuda-only        Run only CUDA tests
    --sycl-only        Run only SYCL tests
    --clean            Clean build directories before building
    --verbose, -v      Enable verbose output
    --no-benchmark     Disable benchmark collection
    --help, -h         Show this help message

Examples:
    $0                          # Run both CUDA and SYCL tests
    $0 --cuda-only             # Run only CUDA tests
    $0 --sycl-only --clean     # Clean and run only SYCL tests
    $0 --verbose --clean       # Clean build with verbose output

EOF
}

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date +'%Y-%m-%d %H:%M:%S') - $1"
}

# Setup directories
setup_directories() {
    log_info "Setting up directories..."

    mkdir -p "$RUN_LOGS_DIR"
    mkdir -p "$CUDA_RESULTS_DIR"
    mkdir -p "$SYCL_RESULTS_DIR"
    mkdir -p "$RESULTS_DIR/comparisons"

    # Create subdirectories for test outputs
    mkdir -p "$CUDA_RESULTS_DIR/test_outputs"
    mkdir -p "$CUDA_RESULTS_DIR/benchmarks"
    mkdir -p "$SYCL_RESULTS_DIR/test_outputs"
    mkdir -p "$SYCL_RESULTS_DIR/benchmarks"

    log_success "Directories created"
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."

    local missing_deps=()

    # Check build tools
    if ! command -v cmake &> /dev/null; then
        missing_deps+=("cmake")
    fi

    if ! command -v make &> /dev/null; then
        missing_deps+=("make")
    fi

    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi

    # Check for CUDA if running CUDA tests
    if [ "$RUN_CUDA" = true ]; then
        if ! command -v nvcc &> /dev/null; then
            missing_deps+=("nvcc (CUDA compiler)")
        fi
    fi

    # Check for SYCL if running SYCL tests
    if [ "$RUN_SYCL" = true ]; then
        if ! command -v icpx &> /dev/null && ! command -v dpcpp &> /dev/null; then
            missing_deps+=("icpx/dpcpp (SYCL compiler)")
        fi
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        exit 1
    fi

    log_success "All dependencies found"
}

# Setup environment
setup_environment() {
    log_info "Setting up environment..."

    # CUDA environment
    if [ "$RUN_CUDA" = true ]; then
        if [ -f "/usr/local/cuda/bin/nvcc" ]; then
            export PATH="/usr/local/cuda/bin:$PATH"
            export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
            log_info "CUDA environment setup (Path: /usr/local/cuda)"
        elif command -v nvcc &> /dev/null; then
            cuda_path=$(dirname $(dirname $(which nvcc)))
            export PATH="$cuda_path/bin:$PATH"
            export LD_LIBRARY_PATH="$cuda_path/lib64:$LD_LIBRARY_PATH"
            log_info "CUDA environment setup (Path: $cuda_path)"
        else
            log_warning "Could not find CUDA installation"
        fi
    fi

    # SYCL environment
    if [ "$RUN_SYCL" = true ]; then
        if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
            source "/opt/intel/oneapi/setvars.sh" > /dev/null 2>&1
            log_info "SYCL environment setup (oneAPI)"
        elif [ -f "/opt/intel/oneapi/compiler/latest/env/vars.sh" ]; then
            source "/opt/intel/oneapi/compiler/latest/env/vars.sh" > /dev/null 2>&1
            log_info "SYCL environment setup (compiler)"
        else
            log_warning "Could not find oneAPI/SYCL installation"
        fi
    fi

    log_success "Environment setup complete"
}

# Clean build directories
clean_build() {
    log_info "Cleaning build directories..."

    if [ -d "$CUDA_DIR/build" ]; then
        rm -rf "$CUDA_DIR/build"
        log_info "Cleaned CUDA build directory"
    fi

    if [ -d "$SYCL_DIR/build" ]; then
        rm -rf "$SYCL_DIR/build"
        log_info "Cleaned SYCL build directory"
    fi

    log_success "Cleanup complete"
}

# Build CUDA project
build_cuda() {
    log_info "Building CUDA project..."

    if [ ! -d "$CUDA_DIR" ]; then
        log_error "CUDA directory not found: $CUDA_DIR"
        return 1
    fi

    cd "$CUDA_DIR"

    # Create build directory
    mkdir -p build
    cd build

    # Configure with CMake
    local cmake_args=(
        "-DCMAKE_BUILD_TYPE=Release"
        "-DENABLE_BENCHMARKING=$BENCHMARK"
    )

    if [ "$VERBOSE" = true ]; then
        cmake_args+=("-DCMAKE_VERBOSE_MAKEFILE=ON")
    fi

    log_info "Configuring CUDA project..."
    if ! cmake "${cmake_args[@]}" .. 2>&1 | tee "$RUN_LOGS_DIR/cuda_cmake.log"; then
        log_error "CUDA CMake configuration failed"
        return 1
    fi

    # Build
    local build_args=("-j$(nproc)")
    if [ "$VERBOSE" = true ]; then
        build_args+=("VERBOSE=1")
    fi

    log_info "Building CUDA project..."
    if ! make "${build_args[@]}" 2>&1 | tee "$RUN_LOGS_DIR/cuda_build.log"; then
        log_error "CUDA build failed"
        return 1
    fi

    log_success "CUDA build completed successfully"
    return 0
}

# Build SYCL project
build_sycl() {
    log_info "Building SYCL project..."

    if [ ! -d "$SYCL_DIR" ]; then
        log_error "SYCL directory not found: $SYCL_DIR"
        return 1
    fi

    cd "$SYCL_DIR"

    # Create build directory
    mkdir -p build
    cd build

    # Configure with CMake
    local cmake_args=(
        "-DCMAKE_BUILD_TYPE=Release"
       ("-DENABLE_BENCHMARKING=$BENCHMARK")
    )

    if [ "$VERBOSE" = true ]; then
        cmake_args+=("-DCMAKE_VERBOSE_MAKEFILE=ON")
    fi

    log_info "Configuring SYCL project..."
    if ! cmake "${cmake_args[@]}" .. 2>&1 | tee "$RUN_LOGS_DIR/sycl_cmake.log"; then
        log_error "SYCL CMake configuration failed"
        return 1
    fi

    # Build
    local build_args=("-j$(nproc)")
    if [ "$VERBOSE" = true ]; then
        build_args+=("VERBOSE=1")
    fi

    log_info "Building SYCL project..."
    if ! make "${build_args[@]}" 2>&1 | tee "$RUN_LOGS_DIR/sycl_build.log"; then
        log_error "SYCL build failed"
        return 1
    fi

    log_success "SYCL build completed successfully"
    return 0
}

# Run CUDA tests
run_cuda_tests() {
    log_info "Running CUDA tests..."

    local test_executable="$CUDA_DIR/build/tests/test_runner"
    local test_log="$RUN_LOGS_DIR/cuda_tests.log"
    local benchmark_file="$CUDA_RESULTS_DIR/benchmarks/benchmark.json"
    local test_results_file="$CUDA_RESULTS_DIR/test_outputs/test_results.json"

    if [ ! -f "$test_executable" ]; then
        log_error "CUDA test executable not found: $test_executable"
        return 1
    fi

    cd "$CUDA_RESULTS_DIR"

    # Setup test environment
    export CUDA_VISIBLE_DEVICES="0"

    # Run tests
    local test_command="$test_executable --gtest_output=json:test_results.json"

    if [ "$BENCHMARK" = true ]; then
        test_command="$test_command --benchmark_out=benchmark.json --benchmark_format=json"
    fi

    log_info "Executing CUDA tests..."

    if [ "$VERBOSE" = true ]; then
        if ! $test_command 2>&1 | tee "$test_log"; then
            log_error "CUDA tests failed (see log: $test_log)"
            return 1
        fi
    else
        if ! $test_command > "$test_log" 2>&1; then
            log_error "CUDA tests failed (see log: $test_log)"
            return 1
        fi
    fi

    log_success "CUDA tests completed successfully"
    return 0
}

# Run SYCL tests
run_sycl_tests() {
    log_info "Running SYCL tests..."

    local test_executable="$SYCL_DIR/build/tests/test_runner"
    local test_log="$RUN_LOGS_DIR/sycl_tests.log"
    local benchmark_file="$SYCL_RESULTS_DIR/benchmarks/benchmark.json"
    local test_results_file="$SYCL_RESULTS_DIR/test_outputs/test_results.json"

    if [ ! -f "$test_executable" ]; then
        log_error "SYCL test executable not found: $test_executable"
        return 1
    fi

    cd "$SYCL_RESULTS_DIR"

    # Setup test environment
    export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"

    # Run tests
    local test_command="$test_executable --gtest_output=json:test_results.json"

    if [ "$BENCHMARK" = true ]; then
        test_command="$test_command --benchmark_out=benchmark.json --benchmark_format=json"
    fi

    log_info "Executing SYCL tests..."

    if [ "$VERBOSE" = true ]; then
        if ! $test_command 2>&1 | tee "$test_log"; then
            log_error "SYCL tests failed (see log: $test_log)"
            return 1
        fi
    else
        if ! $test_command > "$test_log" 2>&1; then
            log_error "SYCL tests failed (see log: $test_log)"
            return 1
        fi
    fi

    log_success "SYCL tests completed successfully"
    return 0
}

# Generate test summary
generate_summary() {
    log_info "Generating test summary..."

    local summary_file="$RESULTS_DIR/comparisons/summary_$TIMESTAMP.json"

    # Create JSON summary
    cat > "$summary_file" << EOF
{
    "timestamp": "$TIMESTAMP",
    "run_config": {
        "cuda": $RUN_CUDA,
        "sycl": $RUN_SYCL,
        "clean_build": $CLEAN_BUILD,
        "benchmarking": $BENCHMARK
    },
    "results": {
EOF

    local first_entry=true

    # CUDA results
    if [ "$RUN_CUDA" = true ] && [ -f "$CUDA_RESULTS_DIR/test_outputs/test_results.json" ]; then
        if [ "$first_entry" = false ]; then
            echo "," >> "$summary_file"
        else
            first_entry=false
        fi
        echo '"cuda":' >> "$summary_file"
        cat "$CUDA_RESULTS_DIR/test_outputs/test_results.json" >> "$summary_file"
    fi

    # SYCL results
    if [ "$RUN_SYCL" = true ] && [ -f "$SYCL_RESULTS_DIR/test_outputs/test_results.json" ]; then
        if [ "$first_entry" = false ]; then
            echo "," >> "$summary_file"
        else
            first_entry=false
        fi
        echo '"sycl":' >> "$summary_file"
        cat "$SYCL_RESULTS_DIR/test_outputs/test_results.json" >> "$summary_file"
    fi

    cat >> "$summary_file" << EOF
    },
    "artifact_paths": {
        "cuda_results": "$CUDA_RESULTS_DIR",
        "sycl_results": "$SYCL_RESULTS_DIR",
        "logs": "$RUN_LOGS_DIR"
    }
}
EOF

    log_success "Test summary generated: $summary_file"
}

# Main execution
main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}    Local Test Runner for CUDA and SYCL${NC}"
    echo -e "${BLUE}================================--------${NC}"
    echo

    parse_arguments "$@"

    # Setup
    setup_directories
    check_dependencies
    setup_environment

    # Clean if requested
    if [ "$CLEAN_BUILD" = true ]; then
        clean_build
    fi

    local cuda_passed=false
    local sycl_passed=false

    # Build and run CUDA tests
    if [ "$RUN_CUDA" = true ]; then
        echo -e "\n${YELLOW}=== CUDA Tests ===${NC}\n"
        if build_cuda && run_cuda_tests; then
            cuda_passed=true
            log_success "CUDA tests PASSED"
        else
            log_error "CUDA tests FAILED"
        fi
    fi

    # Build and run SYCL tests
    if [ "$RUN_SYCL" = true ]; then
        echo -e "\n${YELLOW}=== SYCL Tests ===${NC}\n"
        if build_sycl && run_sycl_tests; then
            sycl_passed=true
            log_success "SYCL tests PASSED"
        else
            log_error "SYCL tests FAILED"
        fi
    fi

    # Generate summary
    generate_summary

    # Final status
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}           Final Test Status${NC}"
    echo -e "${BLUE}================================--------${NC}"

    local overall_status=true

    if [ "$RUN_CUDA" = true ]; then
        if [ "$cuda_passed" = true ]; then
            echo -e "CUDA: ${GREEN}✓ PASSED${NC}"
        else
            echo -e "CUDA: ${RED}✗ FAILED${NC}"
            overall_status=false
        fi
    fi

    if [ "$RUN_SYCL" = true ]; then
        if [ "$sycl_passed" = true ]; then
            echo -e "SYCL: ${GREEN}✓ PASSED${NC}"
        else
            echo -e "SYCL: ${RED}✗ FAILED${NC}"
            overall_status=false
        fi
    fi

    if [ "$overall_status" = true ]; then
        echo -e "\n${GREEN}All tests completed successfully!${NC}"
        echo -e "Results saved to: ${BLUE}$RESULTS_DIR${NC}"
        echo -e "Logs saved to: ${BLUE}$RUN_LOGS_DIR${NC}"
        exit 0
    else
        echo -e "\n${RED}Some tests failed. Check logs for details.${NC}"
        echo -e "Logs saved to: ${BLUE}$RUN_LOGS_DIR${NC}"
        exit 1
    fi
}

# Trap cleanup
cleanup() {
    # Return to original directory
    cd "$PROJECT_ROOT"
}

trap cleanup EXIT

# Run main function with all arguments
main "$@"