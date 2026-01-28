# SYCL Multi-Head Attention Implementation

## Overview

This directory contains a SYCL-based implementation of the Fused Multi-Head Attention (MHA) kernel, designed as a drop-in replacement for the CUTLASS-based CUDA version used in Leela Chess Zero.

## Files

- `fused_attention_sycl.cpp` - Main SYCL MHA kernel implementation
- `mha_integration_example.cpp` - Example showing how to use the SYCL MHA
- `MHA_README.md` - This documentation file

## API Compatibility

The SYCL MHA maintains the same function signature pattern as the CUDA version:

```cpp
// CUDA version (in cutlass_kernels.cu)
void fusedMHA(void* output, void* mha_q, void* mha_k, void* mha_v, void* skip,
              int batch_size, int num_heads, int depth, cudaStream_t stream);

// SYCL version (in fused_attention_sycl.cpp)
void fusedMHA(void* output, void* mha_q, void* mha_k, void* mha_v, void* skip,
              int batch_size, int num_heads, int depth, sycl::queue& queue);
```

The only difference is the stream parameter type:
- CUDA: `cudaStream_t stream`
- SYCL: `sycl::queue& queue`

## Key Features

### 1. Tensor Layout Support
- **BMHK (Batch-MultiHead-Height-Key)** layout matching CUTLASS MHA
- Supports the same memory strides and indexing patterns

### 2. Bias Support
- **With bias**: Implements attention scores + bias
- **Without bias**: Standard attention computation
- Automatically detected based on skip pointer parameter

### 3. Performance Optimizations
- **Tiled computation**: Processes queries and keys in 64x64 tiles
- **Share memory**: Local accessor for efficient data reuse
- **Intel GPU optimizations**:
  - `[[intel::reqd_sub_group_size(16)]]` for optimal subgroup size
  - Efficient memory access patterns
  - Subgroup-based parallelism where applicable

### 4. Numerical Equivalence
- Same attention computation formula as CUTLASS
- Identical scaling factor: `1.0f / sqrt(depth)`
- Softmax with numerical stability
- Half-precision (FP16) support

## Implementation Details

### Memory Layout
```
Query:   [batch][head][query][depth]
Key:     [batch][head][key][depth]
Value:   [batch][head][key][depth]
Output:  [batch][head][query][depth]
Bias:    [batch][head][query][key] (optional)
```

### Tile-based Processing
- **Tile size**: 64 queries × 64 keys
- **Shared memory**: Uses local accessors for tiles
- **Kernel launch**: 3D ND-range (batch, head, query)

### Attention Computation Steps
1. **Query-Key dot product**: Q · K^T
2. **Scaling**: Divide by √depth
3. **Bias addition** (if bias provided)
4. **Softmax**: Across keys for each query
5. **Value weighting**: Attention scores × V
6. **Output**: Weighted sum of values

## Usage Example

```cpp
#include "neural/backends/sycl/kernels.h"

// Initialize SYCL queue
sycl::queue queue(sycl::default_selector());

// MHA configuration
const int batch_size = 2;
const int num_heads = 4;
const int depth = 32;

// Allocate memory (using USM for simplicity)
sycl::half* d_query = sycl::malloc_device<sycl::half>(...);
sycl::half* d_key = sycl::malloc_device<sycl::half>(...);
sycl::half* d_value = sycl::malloc_device<sycl::half>(...);
sycl::half* d_output = sycl::malloc_device<sycl::half>(...);

// Call MHA without bias
lczero::sycl_backend::fusedMHA(d_output, d_query, d_key, d_value,
                              nullptr, batch_size, num_heads, depth, queue);

// Or call MHA with bias
sycl::half* d_bias = sycl::malloc_device<sycl::half>(...);
lczero::sycl_backend::fusedMHA(d_output, d_query, d_key, d_value,
                              d_bias, batch_size, num_heads, depth, queue);

// Cleanup
sycl::free(d_query, queue);
// ... free other allocations
```

## Performance Characteristics

### Theoretical Performance
- **Compute bound**: O(N² × D) attention complexity
- **Memory bound**: O(Q × K + K × V) per tile
- **Parallel efficiency**: Limited by the O(N²) attention computation

### Expected Performance on Intel GPUs
- **Data Center GPU Max**: Theoretical FP16 TFLOPS
- **Arc Series**: Consumer-grade performance
- **Memory bandwidth**: Critical for large depth values

### Optimization Opportunities
1. **Vectorization**: Use `sycl::vec<float, 4>` for memory loads
2. **Prefetching**: Load next tile while computing current
3. **Execution caches**: Cache frequently accessed bias matrices
4. **Work-group tuning**: Optimize for specific Intel GPU generations

## Testing and Validation

### Unit Tests
- `test_fused_mha.cpp`: Comprehensive test suite
- Tests with/without bias
- Numerical validation against reference implementation
- Parameter validation

### Integration Testing
- `mha_integration_example.cpp`: End-to-end example
- Tests actual API compatibility
- Performance benchmarking

### Validation Metrics
- **Numerical accuracy**: < 0.1% error vs reference
- **Regression testing**: Same results as CUDA version
- **Performance**: Target 80%+ of theoretical peak

## Migration Guide

From CUTLASS to SYCL:
1. Replace `cudaStream_t` with `sycl::queue&`
2. Update memory allocation to use SYCL USM or buffers
3. Ensure proper synchronization with `queue.wait()` if needed
4. Update error handling to catch SYCL exceptions
5. No changes needed to tensor layouts or parameters

## Known Limitations

1. **Fixed tile size**: Currently uses 64×64 tiles (matching CUTLASS)
2. **FP16 only**: Half-precision only in this implementation
3. **Fixed dimensions**: Assumes 64 queries/keys per head (from LC0)

## Future Enhancements

1. **Dynamic tile sizes**: Support arbitrary query/key counts
2. **FP32 support**: Add single-precision variant
3. **Flash attention**: Implement memory-efficient variant
4. **Multi-head variants**: Support different head dimensions
5. **Performance tuning**: Auto-tuning for different Intel GPUs

## Dependencies

- **Intel oneAPI DPC++**: For SYCL compiler and runtime
- **SYCL 2020**: Using SYCL 2020 specification features
- **OpenCL/SPIR-V**: Target for Intel GPU backends

## Building

```bash
mkdir build && cd build
cmake ..
make fused_attention_sycl
```

The kernel will be compiled with Intel GPU optimizations by default.