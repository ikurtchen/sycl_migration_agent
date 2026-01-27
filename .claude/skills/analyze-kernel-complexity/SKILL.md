---
name: analyze-kernel-complexity
description: "perform deep analysis of CUDA kernel computational characteristics for performance modeling"
---

# analyze-kernel-complexity

Performs deep analysis of CUDA kernel computational characteristics for performance modeling.

## Description

This skill analyzes individual CUDA kernels to determine their computational and memory characteristics, calculate arithmetic intensity, and project theoretical performance. Essential for Phase 7 performance optimization.

## Usage

```bash
python analyze_kernel_complexity.py <cuda_file> <kernel_name> [--platform <platform_spec.json>] [--input-size <input_size>] [--output <kernel_complexity.json>]
```

### Arguments

- `cuda_file`: Path to CUDA source file containing the kernel
- `kernel_name`: Name of the kernel function to analyze
- `--platform`: Optional JSON file with GPU platform specifications
- `--input-size`: Optional input dimensions (e.g., "N=4096,M=4096")
- `--output`: Optional JSON file to save analysis results

### Examples

```bash
# Basic analysis
python analyze_kernel_complexity.py matmul.cu matrixMul

# With platform specs
python analyze_kernel_complexity.py matmul.cu matrixMul --platform intel_max_1550.json

# With input dimensions
python analyze_kernel_complexity.py matmul.cu matrixMul --input-size "N=4096"

# save output json file
python analyze_kernel_complexity.py matmul.cu matrixMul --output "/path/to/kernel_complexity.json"
```

## Output Format

```json
{
  "kernel_name": "matrixMul",
  "algorithm_type": "dense_matrix_multiplication",
  "compute_characteristics": {
    "total_operations": 137438953472,
    "flops": 137438953472,
    "operation_breakdown": {
      "fma": 68719476736,
      "add": 0,
      "mul": 0,
      "div": 0,
      "special": 0
    },
    "operation_types": {
      "fp32": 137438953472,
      "fp16": 0,
      "int32": 0
    },
    "instruction_mix": {
      "fma_percentage": 100.0,
      "memory_percentage": 0.0,
      "control_percentage": 0.0
    }
  },
  "memory_characteristics": {
    "bytes_read": 201326592,
    "bytes_written": 67108864,
    "total_bytes": 268435456,
    "access_pattern": "tiled_coalesced",
    "reuse_factor": 64,
    "working_set_size_mb": 256.0,
    "cache_behavior": "good_spatial_locality"
  },
  "arithmetic_intensity": {
    "value": 512.0,
    "classification": "compute_bound",
    "description": "512.0 FLOPs per byte - highly compute intensive"
  },
  "parallelism": {
    "total_threads": 16777216,
    "work_per_thread": 8,
    "thread_divergence": "none",
    "synchronization_points": 2,
    "critical_sections": 0
  },
  "roofline_analysis": {
    "ridge_point": 13.6,
    "performance_regime": "compute_bound",
    "limiting_factor": "compute_throughput"
  },
  "complexity_score": {
    "overall": "moderate",
    "compute": "high",
    "memory": "moderate",
    "control": "low"
  }
}
```

## Analysis Components

### 1. Operation Counting

Analyzes kernel code to count:
- **FMA operations**: Fused multiply-add (counts as 2 FLOPs)
- **Arithmetic operations**: ADD, MUL, DIV
- **Special functions**: sqrt, sin, cos, exp, log
- **Integer operations**: Bitwise, integer arithmetic

### 2. Memory Analysis

Calculates:
- **Data movement**: Bytes read from/written to global memory
- **Access patterns**: Coalesced, strided, random
- **Reuse distance**: How often data is reused
- **Working set size**: Memory footprint

### 3. Arithmetic Intensity Calculation

```
AI = Total FLOPs / Total Bytes Transferred

Classification:
- AI < 1.0:   Memory-bound (bandwidth-limited)
- 1.0 < AI < 10: Balanced
- AI > 10:    Compute-bound (FLOPS-limited)
```

### 4. Parallelism Assessment

Evaluates:
- Total thread count from launch configuration
- Work distribution per thread
- Thread divergence (branches)
- Synchronization overhead
- Load balancing

### 5. Roofline Positioning

Determines kernel position on roofline model:
- Ridge point: Threshold between memory and compute bound
- Performance regime: Which resource limits performance
- Optimization opportunities

## Platform Specification Format

```json
{
  "name": "Intel Data Center GPU Max 1550",
  "compute": {
    "xe_cores": 128,
    "vector_engines_per_core": 8,
    "matrix_engines_per_core": 8,
    "peak_fp32_tflops": 22.2,
    "peak_fp16_tflops": 44.4,
    "peak_int8_tops": 88.8,
    "subgroup_size": 16
  },
  "memory": {
    "hbm_gb": 128,
    "bandwidth_gbps": 3200,
    "l3_cache_mb": 408,
    "l1_cache_kb": 64
  },
  "derived": {
    "ridge_point_fp32": 144.4
  }
}
```

## Complexity Scoring

### Overall Complexity

- **Simple**: Single loop, no shared memory, < 100 operations/thread
- **Moderate**: Tiling, shared memory, 100-1000 operations/thread  
- **Complex**: Multi-stage, dynamic workload, > 1000 operations/thread

### Component Scores

Each aspect (compute, memory, control) scored separately:
- **Low**: Straightforward, predictable
- **Moderate**: Some complexity, requires tuning
- **High**: Significant complexity, needs expert optimization

## Algorithm Pattern Recognition

Automatically identifies common patterns:

### Matrix Operations
- `dense_matrix_multiplication`: GEMM patterns
- `sparse_matrix_multiplication`: SpMV, SpMM
- `matrix_transpose`: Memory-bound transpose
- `triangular_solve`: TRSM patterns

### Vector Operations
- `vector_addition`: Element-wise operations
- `dot_product`: Reduction patterns
- `vector_scaling`: AXPY patterns

### Convolution
- `2d_convolution`: Image/signal processing
- `3d_convolution`: Volumetric operations
- `separable_convolution`: Optimizable patterns

### Reduction
- `parallel_reduction`: Tree-based reduction
- `segmented_reduction`: Per-segment reductions

## Integration Example

```python
# In performance_optimizer subagent
analysis = execute_skill("analyze-kernel-complexity", args=[
    "matmul.cu",
    "matrixMul",
    "--platform", "intel_max_1550.json",
    "--input-size", "N=4096"
])

# Use results for optimization planning
if analysis["arithmetic_intensity"]["classification"] == "memory_bound":
    # Focus on memory optimizations
    apply_memory_optimizations()
elif analysis["roofline_analysis"]["performance_regime"] == "compute_bound":
    # Focus on compute optimizations
    apply_compute_optimizations()
```

## Performance Projection

Calculates theoretical performance:

```python
def project_performance(analysis, platform):
    ai = analysis["arithmetic_intensity"]["value"]
    flops = analysis["compute_characteristics"]["flops"]
    bytes = analysis["memory_characteristics"]["total_bytes"]

    # Compute-bound time
    compute_time = flops / (platform["peak_fp32_tflops"] * 1e12)

    # Memory-bound time  
    memory_time = bytes / (platform["bandwidth_gbps"] * 1e9)

    # Actual time is the maximum (bottleneck)
    theoretical_time = max(compute_time, memory_time)

    return {
        "theoretical_time_ms": theoretical_time * 1000,
        "theoretical_gflops": flops / theoretical_time / 1e9,
        "bottleneck": "compute" if compute_time > memory_time else "memory"
    }
```

## Advanced Features

### Template Parameter Analysis

Handles templated kernels:
```cpp
template<int TILE_SIZE>
__global__ void tiledMatMul(...)
```

Analyzes for different template values.

### Dynamic Kernel Features

Detects:
- Dynamic shared memory allocation
- Runtime-dependent control flow
- Variable work distribution

### Optimization Opportunities

Identifies:
- Vectorization potential
- Fusion opportunities
- Redundant memory accesses
- Suboptimal access patterns

## Validation

Cross-validates analysis against:
- Known algorithm complexities (e.g., O(N³) for matrix multiply)
- Empirical measurements when available
- Literature benchmarks

## Limitations

- Static analysis only (no runtime profiling)
- May not capture all dynamic behavior
- Assumes ideal memory access patterns
- Template kernels require instantiation

## Related Skills

- `scan-cuda-repo`: Provides kernel inventory for batch analysis
- `profile-gpu-kernel`: Validates analysis with actual measurements
- `optimize-sycl-kernel`: Uses analysis for optimization decisions

## Skill Metadata

```yaml
name: analyze-kernel-complexity
version: 1.0.0
category: analysis
dependencies:
  - python >= 3.8
  - numpy
inputs:
  - cuda_file: string
  - kernel_name: string
  - platform_spec: JSON (optional)
  - input_size: string (optional)
outputs:
  - complexity_analysis: JSON object
execution_time: 1-5 seconds per kernel
accuracy: 85-95% for standard patterns
```
