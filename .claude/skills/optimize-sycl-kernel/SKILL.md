---
name: optimize-sycl-kernel
description: "optimize SYCL kernels for Intel GPUs using profiling and analysis data"
---

# optimize-sycl-kernel

Applies Intel GPU-specific optimizations to SYCL kernels based on profiling data and analysis.

## Description

This skill transforms SYCL kernels by applying targeted optimizations for Intel GPUs. It uses profiling metrics, theoretical analysis, and best practices to maximize performance across memory bandwidth, compute throughput, and parallelism.

## Usage

```bash
python optimize_sycl_kernel.py <sycl_file> <kernel_name> [options]
```

### Arguments

- `sycl_file`: Path to SYCL source file
- `kernel_name`: Name of kernel to optimize
- `--profiling-data`: JSON file with profiling metrics
- `--analysis-data`: JSON file with complexity analysis
- `--platform`: Intel GPU platform (e.g., "pvc", "acm-g10")
- `--optimization-level`: Level of optimization (conservative, balanced, aggressive)
- `--output`: Output file for optimized kernel

### Examples

```bash
# Basic optimization
python optimize_sycl_kernel.py matmul.cpp matrixMul \
    --profiling-data profiling_results/matrixMul_metrics.json \
    --output matmul_optimized.cpp

# Aggressive optimization for specific platform
python optimize_sycl_kernel.py matmul.cpp matrixMul \
    --profiling-data profiling_results/matrixMul_metrics.json \
    --analysis-data analysis/matrixMul_analysis.json \
    --platform pvc \
    --optimization-level aggressive \
    --output matmul_optimized.cpp
```

## Optimization Categories

### 1. Memory Optimizations

#### Coalesced Access Pattern

**Before:**
```cpp
// Strided access
for (int i = item.get_global_id(0); i < n; i += stride) {
    output[i] = input[i * stride];
}
```

**After:**
```cpp
// Coalesced access
int idx = item.get_global_id(0);
if (idx < n) {
    output[idx] = input[idx];
}
```

#### Shared/Local Memory Usage

**Before:**
```cpp
// Direct global memory access
float sum = 0;
for (int i = 0; i < N; i++) {
    sum += data[i];
}
```

**After:**
```cpp
// Use local memory for data reuse
sycl::local_accessor<float, 1> local_data(sycl::range<1>(256), h);

// ... in kernel
// Cooperative load to local memory
local_data[item.get_local_id(0)] = data[item.get_global_id(0)];
item.barrier();

// Access from local memory
float sum = 0;
for (int i = 0; i < 256; i++) {
    sum += local_data[i];
}
```

#### Vectorization

**Before:**
```cpp
// Scalar loads
for (int i = item.get_global_id(0); i < n; i++) {
    c[i] = a[i] + b[i];
}
```

**After:**
```cpp
// Vectorized loads (4-wide)
int idx = item.get_global_id(0) * 4;
if (idx + 3 < n) {
    sycl::vec<float, 4> va = *(sycl::vec<float, 4>*)&a[idx];
    sycl::vec<float, 4> vb = *(sycl::vec<float, 4>*)&b[idx];
    sycl::vec<float, 4> vc = va + vb;
    *(sycl::vec<float, 4>*)&c[idx] = vc;
}
```

### 2. Compute Optimizations

#### Subgroup Operations

**Before:**
```cpp
// Manual reduction across work-group
float val = compute_local();
local_data[item.get_local_id(0)] = val;
item.barrier();

// Tree reduction
for (int stride = 128; stride > 0; stride /= 2) {
    if (item.get_local_id(0) < stride) {
        local_data[item.get_local_id(0)] += local_data[item.get_local_id(0) + stride];
    }
    item.barrier();
}
```

**After:**
```cpp
// Efficient subgroup reduction
float val = compute_local();
auto sg = item.get_sub_group();
float sg_sum = sycl::reduce_over_group(sg, val, sycl::plus<>());

// Only one item per subgroup writes to local memory
if (sg.get_local_id() == 0) {
    local_data[sg.get_group_id()] = sg_sum;
}
item.barrier();

// Final reduction across subgroups
if (item.get_local_id(0) < num_subgroups) {
    float final_sum = sycl::reduce_over_group(sg, local_data[item.get_local_id(0)], sycl::plus<>());
}
```

#### Work-Group Size Tuning

**Before:**
```cpp
// Generic work-group size
q.parallel_for(
    sycl::nd_range<1>(global_size, sycl::range<1>(64)),
    [=](sycl::nd_item<1> item) { ... }
);
```

**After:**
```cpp
// Optimized for Intel GPU (Data Center GPU Max)
constexpr int OPTIMAL_WG_SIZE = 256;  // Multiple of subgroup size (16)

q.parallel_for(
    sycl::nd_range<1>(
        sycl::range<1>((global_size + OPTIMAL_WG_SIZE - 1) / OPTIMAL_WG_SIZE * OPTIMAL_WG_SIZE),
        sycl::range<1>(OPTIMAL_WG_SIZE)
    ),
    [=](sycl::nd_item<1> item) { ... }
);
```

#### Loop Unrolling

**Before:**
```cpp
for (int i = 0; i < 8; i++) {
    sum += data[i];
}
```

**After:**
```cpp
#pragma unroll
for (int i = 0; i < 8; i++) {
    sum += data[i];
}
```

### 3. Intel-Specific Optimizations

#### Subgroup Size Specification

```cpp
[[intel::reqd_sub_group_size(16)]]  // Force subgroup size 16
void optimized_kernel(...) {
    auto sg = item.get_sub_group();
    // Subgroup operations guaranteed to work on 16 items
}
```

#### Kernel Arguments Restriction

```cpp
[[intel::kernel_args_restrict]]  // Tell compiler pointers don't alias
void matrixMul(...) {
    // Compiler can optimize more aggressively
}
```

#### SIMD Width Hints

```cpp
[[intel::num_simd_work_items(16)]]  // SIMD width hint
void vectorized_kernel(...) {
    // Vectorization hint for compiler
}
```

#### Prefetching

```cpp
// Prefetch upcoming data
sycl::ext::oneapi::experimental::prefetch(ptr + prefetch_distance);
```

### 4. Algorithm-Specific Optimizations

#### Matrix Multiplication (Tiled)

**Before - Naive:**
```cpp
void matrixMul(sycl::queue& q, float* C, float* A, float* B, int N) {
    q.parallel_for(sycl::nd_range<2>(
        sycl::range<2>(N, N),
        sycl::range<2>(16, 16)),
        [=](sycl::nd_item<2> item) {
            int row = item.get_global_id(0);
            int col = item.get_global_id(1);

            float sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        });
}
```

**After - Tiled with Local Memory:**
```cpp
constexpr int TILE_SIZE = 32;

[[intel::reqd_sub_group_size(16)]]
[[intel::kernel_args_restrict]]
void matrixMul_optimized(sycl::queue& q, float* C, float* A, float* B, int N) {
    q.submit([&](sycl::handler& h) {
        // Local memory for tiles
        sycl::local_accessor<float, 2> tileA(sycl::range<2>(TILE_SIZE, TILE_SIZE), h);
        sycl::local_accessor<float, 2> tileB(sycl::range<2>(TILE_SIZE, TILE_SIZE), h);

        h.parallel_for(
            sycl::nd_range<2>(
                sycl::range<2>(N, N),
                sycl::range<2>(TILE_SIZE, TILE_SIZE)
            ),
            [=](sycl::nd_item<2> item) {
                int row = item.get_global_id(0);
                int col = item.get_global_id(1);
                int local_row = item.get_local_id(0);
                int local_col = item.get_local_id(1);

                float sum = 0;

                // Tile iteration
                for (int t = 0; t < N / TILE_SIZE; t++) {
                    // Cooperative load to local memory
                    tileA[local_row][local_col] = A[row * N + (t * TILE_SIZE + local_col)];
                    tileB[local_row][local_col] = B[(t * TILE_SIZE + local_row) * N + col];

                    item.barrier();

                    // Compute using local memory
                    #pragma unroll 8
                    for (int k = 0; k < TILE_SIZE; k++) {
                        sum += tileA[local_row][k] * tileB[k][local_col];
                    }

                    item.barrier();
                }

                C[row * N + col] = sum;
            });
    });
}
```

#### Reduction Pattern

**Before - Sequential:**
```cpp
float reduce(float* data, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}
```

**After - Parallel with Subgroups:**
```cpp
[[intel::reqd_sub_group_size(16)]]
float reduce_optimized(sycl::queue& q, float* data, int n) {
    constexpr int WG_SIZE = 256;
    int num_wg = (n + WG_SIZE - 1) / WG_SIZE;

    float* partial_sums = sycl::malloc_device<float>(num_wg, q);

    // First reduction: data -> partial_sums
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> local_sum(sycl::range<1>(WG_SIZE / 16), h);

        h.parallel_for(
            sycl::nd_range<1>(num_wg * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) {
                int idx = item.get_global_id(0);
                auto sg = item.get_sub_group();

                // Each item reads one element
                float val = (idx < n) ? data[idx] : 0.0f;

                // Subgroup reduction
                float sg_sum = sycl::reduce_over_group(sg, val, sycl::plus<>());

                // One item per subgroup stores to local memory
                if (sg.get_local_id() == 0) {
                    local_sum[sg.get_group_id()] = sg_sum;
                }
                item.barrier();

                // Final reduction within work-group
                if (item.get_local_id(0) < WG_SIZE / 16) {
                    float wg_sum = sycl::reduce_over_group(sg, local_sum[item.get_local_id(0)], sycl::plus<>());

                    if (item.get_local_id(0) == 0) {
                        partial_sums[item.get_group(0)] = wg_sum;
                    }
                }
            });
    }).wait();

    // Second reduction on CPU (small array)
    std::vector<float> h_partial(num_wg);
    q.memcpy(h_partial.data(), partial_sums, num_wg * sizeof(float)).wait();

    float result = 0;
    for (float v : h_partial) result += v;

    sycl::free(partial_sums, q);
    return result;
}
```

## Optimization Selection Algorithm

```python
def select_optimizations(profiling_data, analysis_data):
    """Select appropriate optimizations based on profiling and analysis."""
    optimizations = []

    # Check memory bottleneck
    if profiling_data['memory_bandwidth_utilization'] > 70:
        optimizations.append({
            'type': 'memory',
            'techniques': [
                'local_memory_tiling',
                'increase_arithmetic_intensity',
                'kernel_fusion'
            ]
        })

    # Check compute bottleneck
    if analysis_data['arithmetic_intensity'] > 100:
        optimizations.append({
            'type': 'compute',
            'techniques': [
                'subgroup_operations',
                'vectorization',
                'loop_unrolling'
            ]
        })

    # Check occupancy
    if profiling_data['eu_active_percent'] < 75:
        optimizations.append({
            'type': 'occupancy',
            'techniques': [
                'tune_work_group_size',
                'reduce_register_pressure',
                'decrease_local_memory_usage'
            ]
        })

    # Check subgroup efficiency
    if 'subgroup_efficiency' in profiling_data:
        if profiling_data['subgroup_efficiency'] < 80:
            optimizations.append({
                'type': 'subgroup',
                'techniques': [
                    'use_subgroup_operations',
                    'specify_subgroup_size',
                    'reduce_divergence'
                ]
            })

    return optimizations
```

## Optimization Application

```python
class SYCLOptimizer:
    """Apply optimizations to SYCL kernels."""

    def __init__(self, profiling_data, analysis_data, platform):
        self.profiling_data = profiling_data
        self.analysis_data = analysis_data
        self.platform = platform
        self.optimizations_applied = []

    def optimize_kernel(self, kernel_code, kernel_name):
        """Apply all applicable optimizations."""
        optimized_code = kernel_code

        # Select optimizations
        opts = select_optimizations(self.profiling_data, self.analysis_data)

        for opt in opts:
            if opt['type'] == 'memory':
                optimized_code = self.apply_memory_optimizations(optimized_code)
            elif opt['type'] == 'compute':
                optimized_code = self.apply_compute_optimizations(optimized_code)
            elif opt['type'] == 'occupancy':
                optimized_code = self.apply_occupancy_optimizations(optimized_code)
            elif opt['type'] == 'subgroup':
                optimized_code = self.apply_subgroup_optimizations(optimized_code)

        # Add Intel-specific attributes
        optimized_code = self.add_intel_attributes(optimized_code, kernel_name)

        return {
            'optimized_code': optimized_code,
            'optimizations_applied': self.optimizations_applied
        }

    def apply_memory_optimizations(self, code):
        """Apply memory-focused optimizations."""
        # Detect and apply vectorization
        if 'for' in code and 'a[i]' in code:
            code = self.vectorize_loops(code)
            self.optimizations_applied.append('vectorization')

        # Add local memory for tiling
        if 'matrix' in code.lower():
            code = self.add_tiling(code)
            self.optimizations_applied.append('tiling')

        return code

    def apply_subgroup_operations(self, code):
        """Replace work-group operations with subgroup operations."""
        # Find reduction patterns
        if 'barrier' in code and 'sum' in code:
            code = self.use_subgroup_reduce(code)
            self.optimizations_applied.append('subgroup_reduction')

        return code

    def add_intel_attributes(self, code, kernel_name):
        """Add Intel-specific kernel attributes."""
        attributes = []

        # Subgroup size
        if self.platform in ['pvc', 'acm-g10']:
            attributes.append('[[intel::reqd_sub_group_size(16)]]')

        # Kernel args restrict
        attributes.append('[[intel::kernel_args_restrict]]')

        # Add attributes before kernel
        attr_str = '\n'.join(attributes) + '\n'
        code = code.replace(f'void {kernel_name}', 
                           f'{attr_str}void {kernel_name}')

        return code
```

## Performance Validation

```python
def validate_optimization(original_metrics, optimized_metrics):
    """Validate that optimizations improved performance."""
    improvement = {
        'execution_time': (original_metrics['time_ms'] - optimized_metrics['time_ms']) / original_metrics['time_ms'] * 100,
        'throughput': (optimized_metrics['gflops'] - original_metrics['gflops']) / original_metrics['gflops'] * 100
    }

    # Check for regression
    if improvement['execution_time'] < 0:
        print(f"WARNING: Optimization caused {abs(improvement['execution_time']):.1f}% regression!")
        return False

    print(f"Optimization improved performance by {improvement['execution_time']:.1f}%")
    return True
```

## Output Format

```json
{
  "kernel_name": "matrixMul",
  "original_file": "matmul.cpp",
  "optimized_file": "matmul_optimized.cpp",
  "optimizations_applied": [
    {
      "type": "memory",
      "technique": "tiling",
      "description": "Added 32x32 tiling with local memory",
      "expected_improvement": "30-50% for memory-bound kernels"
    },
    {
      "type": "subgroup",
      "technique": "subgroup_operations",
      "description": "Replaced manual reductions with subgroup operations",
      "expected_improvement": "10-20% reduction in synchronization overhead"
    },
    {
      "type": "intel_attributes",
      "technique": "kernel_attributes",
      "description": "Added [[intel::reqd_sub_group_size(16)]] and [[intel::kernel_args_restrict]]",
      "expected_improvement": "5-15% from compiler optimizations"
    }
  ],
  "performance_projection": {
    "original_gflops": 10500,
    "projected_gflops": 15750,
    "expected_improvement_percent": 50.0
  }
}
```

## Related Skills

- `analyze-kernel-complexity`: Provides theoretical analysis for optimization decisions
- `profile-gpu-kernel`: Provides profiling data to guide optimizations
- `compare-numerical-results`: Validates optimized kernel correctness

## Skill Metadata

```yaml
name: optimize-sycl-kernel
version: 1.0.0
category: optimization
dependencies:
  - python >= 3.8
inputs:
  - sycl_file: string
  - kernel_name: string
  - profiling_data: JSON
  - analysis_data: JSON (optional)
  - platform: string
outputs:
  - optimized_code: C++ source
  - optimization_report: JSON
execution_time: 1-5 seconds
```
