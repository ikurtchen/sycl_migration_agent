# expandPlanes Translation Mapping Reference

## Core Translation Mappings

### Thread Index Mapping
| CUDA | SYCL | Description |
|------|------|-------------|
| `threadIdx.x + blockDim.x * blockIdx.x` | `item.get_global_id(0)` | Global thread ID |
| `threadIdx.x` | `item.get_local_id(0)` | Thread ID within work-group |
| `blockDim.x` | `item.get_local_range(0)` | Work-group size |
| `blockIdx.x` | `item.get_group(0)` | Work-group ID |

### Kernel Launch Configuration
| CUDA | SYCL | Description |
|------|------|-------------|
| `<<<blocks, kBlockSize, 0, stream>>>` | `parallel_for(nd_range<1>(global, local))` | Kernel launch |
| `const int kBlockSize = 256` | `const int kBlockSize = 256` | Work-group size |
| `int blocks = DivUp(threads, kBlockSize)` | `int blocks = DivUp(threads, kBlockSize)` | Number of work-groups |
| `global range = blocks * kBlockSize` | `sycl::range<1>(blocks * kBlockSize)` | Global range |
| `local range = kBlockSize` | `sycl::range<1>(kBlockSize)` | Local range |

### Memory Operations
| CUDA | SYCL | Description |
|------|------|-------------|
| `cudaMallocManaged` | `sycl::malloc_shared` | Shared memory allocation |
| `cudaMemcpy` | `q.memcpy()` | Memory copy |
| `cudaFree` | `sycl::free()` | Memory deallocation |
| `cudaStream_t stream` | `sycl::queue& q` | Execution queue |

### Data Types
| CUDA | SYCL | Description |
|------|------|-------------|
| `half` | `sycl::half` | Half precision float |
| `float` | `float` | Single precision float |
| `uint64_t` | `uint64_t` | 64-bit unsigned integer |

### Error Handling
| CUDA | SYCL | Description |
|------|------|-------------|
| `ReportCUDAErrors(cudaGetLastError())` | `try..catch(sycl::exception)` | Error handling |
| `cudaError_t status` | `sycl::exception const& e` | Error object |

##Kernel-Specific Translations

### expandPlanes_kernel_NHWC

#### CUDA Implementation
```cpp
template <typename T>
__global__ void expandPlanes_kernel_NHWC(T* output, const uint64_t* masks,
                                         const T* values, int n) {
  const int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index >= n * 8 * 8) return;

  const int planeIndex = index % kInputPlanes;
  const int boardIndex = index / (kInputPlanes * 8 * 8);
  const int sqIndex = (index / kInputPlanes) & 0x3F;

  uint64_t mask = masks[boardIndex * kInputPlanes + planeIndex];

  T op = 0;
  bool set = !!(mask & (1ull << sqIndex));
  if (set) {
    op = values[boardIndex * kInputPlanes + planeIndex];
  }
  output[index] = op;
}
```

#### SYCL Implementation
```cpp
template <typename T>
void expandPlanes_kernel_NHWC(sycl::queue& q, T* output, const uint64_t* masks,
                             const T* values, int n) {
  int threads = n * 8 * 8;
  const int kBlockSize = 256;
  int blocks = DivUp(threads, kBlockSize);

  q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(
      sycl::range<1>(blocks * kBlockSize),
      sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        const int index = item.get_global_id(0);
        if (index >= n * 8 * 8) return;

        const int planeIndex = index % kInputPlanes;
        const int boardIndex = index / (kInputPlanes * 8 * 8);
        const int sqIndex = (index / kInputPlanes) & 0x3F;

        uint64_t mask = masks[boardIndex * kInputPlanes + planeIndex];

        T op = static_cast<T>(0);
        bool set = !!(mask & (1ull << sqIndex));
        if (set) {
          op = values[boardIndex * kInputPlanes + planeIndex];
        }
        output[index] = op;
      });
  }).wait();
}
```

### expandPlanes_kernel_NCHW

#### CUDA Implementation
```cpp
template <typename T>
__global__ void expandPlanes_kernel_NCHW(T* output, const uint64_t* masks,
                                         const T* values, unsigned n) {
  unsigned index = threadIdx.x + blockDim.x * blockIdx.x;

  index *= 2;
  unsigned planeIndex = index >> 6;

  if (planeIndex >= n) return;

  uint64_t mask = masks[planeIndex];

  int sqIndex = index & 0x3F;
  T op[2] = {0, 0};

  bool set = !!(mask & (1ull << sqIndex));
  if (set) {
    op[0] = values[planeIndex];
  }
  sqIndex++;
  set = !!(mask & (1ull << sqIndex));
  if (set) {
    op[1] = values[planeIndex];
  }
  output[index + 0] = op[0];
  output[index + 1] = op[1];
}
```

#### SYCL Implementation
```cpp
template <typename T>
void expandPlanes_kernel_NCHW(sycl::queue& q, T* output, const uint64_t* masks,
                             const T* values, unsigned n) {
  unsigned threads = n * 8 * 8 / 2;
  const int blockSize = 256;
  unsigned blocks = DivUp(threads, blockSize);

  q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(
      sycl::range<1>(blocks * blockSize),
      sycl::range<1>(blockSize)),
      [=](sycl::nd_item<1> item) {
        unsigned index = item.get_global_id(0);

        index *= 2;
        unsigned planeIndex = index >> 6;

        if (planeIndex >= n) return;

        uint64_t mask = masks[planeIndex];

        int sqIndex = index & 0x3F;
        T op[2] = {static_cast<T>(0), static_cast<T>(0)};

        bool set = !!(mask & (1ull << sqIndex));
        if (set) {
          op[0] = values[planeIndex];
        }

        sqIndex++;
        set = !!(mask & (1ull << sqIndex));
        if (set) {
          op[1] = values[planeIndex];
        }

        output[index + 0] = op[0];
        output[index + 1] = op[1];
      });
  }).wait();
}
```

## Key Differences Highlighted

1. **Execution Model**:
   - CUDA: Kernel functions marked with `__global__`
   - SYCL: Regular functions that submit kernels to queues

2. **Memory Access**:
   - CUDA: Direct pointer access
   - SYCL: Same direct pointer access with USM

3. **Type Casting**:
   - CUDA: Implicit casting using `(T)value`
   - SYCL: Explicit casting using `static_cast<T>(value)`

4. **Initialization**:
   - CUDA: `{0, 0}` array initialization
   - SYCL: `{static_cast<T>(0), static_cast<T>(0)}` explicit typing

5. **Error Handling**:
   - CUDA: Post-kernel error checking
   - SYCL: Exception-based error handling with try/catch