# Remote Execution Setup for CUDA/SYCL Validation

## Overview

This document provides the setup instructions for remote execution of CUDA and SYCL kernels on NVIDIA and Intel GPU servers respectively, enabling numerical comparison between the two implementations.

## Prerequisites

### 1. NVIDIA GPU Server (CUDA Benchmark)
- CUDA Toolkit 11.0+ installed
- NVIDIA GPU with Compute Capability 7.0+
- SSH access configured
- cuBLAS, cuDNN libraries available
- Google Test compiled with CUDA support

### 2. Intel GPU Server (SYCL Target)
- Intel oneAPI DPC++/C++ Compiler 2024+
- Intel GPU (Data Center GPU Max or Arc series)
- SSH access configured
- Intel oneAPI MKL and GPU Runtime
- Google Test compiled with SYCL support

## Configuration

### NVIDIA GPU Server Configuration
```json
{
  "server": {
    "hostname": "nvidia-gpu-server.example.com",
    "username": "user",
    "port": 22
  },
  "environment": {
    "CUDA_VISIBLE_DEVICES": "0",
    "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
    "PATH": "/usr/local/cuda/bin:$PATH"
  },
  "build": {
    "cmake_path": "/usr/bin/cmake",
    "make_path": "/usr/bin/make",
    "cuda_architectures": ["70", "75", "80", "86"]
  }
}
```

### Intel GPU Server Configuration
```json
{
  "server": {
    "hostname": "intel-gpu-server.example.com",
    "username": "user",
    "port": 22
  },
  "environment": {
    "ONEAPI_DEVICE_SELECTOR": "level_zero",
    "LD_LIBRARY_PATH": "/opt/intel/oneapi/compiler/latest/linux/lib:/opt/intel/oneapi/mkl/latest/lib",
    "PATH": "/opt/intel/oneapi/compiler/latest/linux/bin:$PATH"
  },
  "build": {
    "cmake_path": "/opt/intel/oneapi/compiler/latest/linux/bin/cmake",
    "make_path": "/usr/bin/make",
    "sycl_target": "intel_gpu"
  }
}
```

## Execution Workflow

### Phase 1: CUDA Execution
1. **Build CUDA Tests**:
   ```bash
   ssh nvidia-gpu-server.example.com
   cd cuda
   mkdir -p build && cd build
   cmake .. -DCMAKE_CXX_COMPILER=g++ -DCUDA_ARCHITECTURES=80
   make -j$(nproc)
   ```

2. **Run CUDA Tests**:
   ```bash
   mkdir -p ../results/cuda_outputs
   ./tests/cuda_test_runner --gtest_output=json:../results/cuda_test_results.json
   ```

3. **Verify CUDA Outputs**:
   - Check for all kernel outputs in `results/cuda_outputs/`
   - Validate benchmark metrics JSON files
   - Ensure no test failures

### Phase 2: SYCL Execution
1. **Build SYCL Tests**:
   ```bash
   ssh intel-gpu-server.example.com
   cd sycl
   source /opt/intel/oneapi/setvars.sh
   mkdir -p build && cd build
   cmake .. -DCMAKE_CXX_COMPILER=icpx -DINTEL_GPU_TARGET="spir64"
   make -j$(nproc)
   ```

2. **Run SYCL Tests**:
   ```bash
   mkdir -p ../results/sycl_outputs
   ./tests/sycl_test_runner --gtest_output=json:../results/sycl_test_results.json
   ```

3. **Verify SYCL Outputs**:
   - Check for all kernel outputs in `results/sycl_outputs/`
   - Validate benchmark metrics JSON files
   - Ensure no test failures

### Phase 3: Comparison and Validation
1. **Copy Results**:
   ```bash
   rsync -av nvidia-gpu-server:cuda/results/ ./results/cuda/
   rsync -av intel-gpu-server:sycl/results/ ./results/sycl/
   ```

2. **Run Comparison Tool**:
   ```bash
   python compare_results.py --cuda ./results/cuda/ --sycl ./results/sycl/
   ```

## Error Tolerances

| Data Type | Absolute Tolerance | Relative Tolerance |
|-----------|-------------------|-------------------|
| float32   | 1e-6             | 1e-6             |
| float16   | 1e-3             | 1e-3             |

## Test Matrix

### Kernels to Compare

| Kernel Name | Input Size | Data Type | Special Notes |
|-------------|------------|-----------|---------------|
| addVectors | 1024, 4096 | float, half | Multiple activation functions |
| addBiasBatched | N=2, C=64, N=128 | float | Test with/without Nstride |
| batchNorm | N=2, C=64, N=128 | float | Various epsilon values |
| globalAvgPool | N=2, C=64, elements=64 | float, half | NHWC layout for fp16 |
| SE_Layer_NHWC | N=2, C=64, se_K=32 | half | Test with/without skip |
| softmax | C=64 (optimized), C=128 (general) | float, half | Test numerical stability |
| policyMap | moves=1858, N=2 | float | Chess policy mapping |
| expandPlanes | planes=112, N=2 | float, half | Both NHWC and NCHW |
| layer_norm | N=2, H=8, W=8, C=64 | float, half | Test with/without skip |
| copyTypeConverted | size=4096 | float↔half | Type conversion accuracy |
| globalScale | N=2, C=64, N=128 | float, half | Different layouts |
| NCHWtoNHWC | N=2, C=64, H=8, W=8 | float | Layout conversion |

## Performance Metrics

Collected for each kernel:
- **Execution Time** (ms) - Average over 100 runs
- **Throughput** (GB/s) - Memory bandwidth utilization
- **GFLOPS** - Compute throughput (for applicable kernels)
- **Device Utilization** - GPU resource usage
- **Memory Usage** - Peak memory consumption

## Troubleshooting

### CUDA Side Issues
1. **Kernel Launch Failures**: Check CUDA compute capability
2. **Memory Leaks**: Verify CUDA free operations
3. **Synchronization Errors**: Ensure proper cudaDeviceSynchronize() calls

### SYCL Side Issues
1. **Device Selection**: Verify ONEAPI_DEVICE_SELECTOR
2. **Compilation Errors**: Check Intel oneAPI environment sourcing
3. **Runtime Errors**: Validate Intel GPU driver and runtime

### Comparison Issues
1. **Tolerance Failures**: Check numerical precision differences
2. **Missing Files**: Ensure test output directories are created
3. **Format Mismatches**: Verify identical input patterns

## Automation Script

A Python automation script is available at:
`/localdisk/kurt/workspace/code/ai_coding/sycl_migration_agent/scripts/remote_runner.py`

Usage:
```bash
python remote_runner.py --config execution_config.json
```

This script will:
1. Connect to both servers
2. Build and run CUDA tests
3. Build and run SYCL tests
4. Collect all results
5. Run comparison analysis
6. Generate validation report

## Report Generation

Final report includes:
- **Executive Summary**: Migration completeness percentage
- **Numerical Validation**: Pass/fail status for each kernel
- **Performance Comparison**: CUDA vs SYCL performance metrics
- **Recommendations**: Next steps and optimization opportunities