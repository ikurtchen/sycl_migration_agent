---
name: scan-cuda-repo
description: "recursively scan a repository for CUDA files and extract kernel information for migration planning"
---

# scan-cuda-repo

Recursively scans a repository to find all CUDA files and extract kernel information.

## Description

This skill performs comprehensive repository scanning to identify CUDA source files, extract kernel functions, analyze their features, and assess migration complexity. It's the foundation for planning CUDA-to-SYCL migrations.

## Usage

```bash
python scan_cuda_repo.py <repository_path> [--output <output.json>]
```

### Arguments

- `repository_path`: Path to the repository root directory (required)
- `--output`: Optional JSON file to save scan results

### Examples

```bash
# Scan current directory
python scan_cuda_repo.py .

# Scan specific repository
python scan_cuda_repo.py /path/to/cuda/project

# Save results to file
python scan_cuda_repo.py /path/to/cuda/project --output scan_results.json
```

## Output Format

```json
{
  "repository": "/path/to/repo",
  "scan_date": "2026-01-26T10:30:00",
  "total_cuda_files": 15,
  "cuda_files": [
    "src/kernels/matmul.cu",
    "src/kernels/vectorAdd.cu"
  ],
  "total_kernels": 42,
  "kernels": [
    {
      "name": "matrixMul",
      "file": "src/kernels/matmul.cu",
      "line": 45,
      "signature": "__global__ void matrixMul(float* C, float* A, float* B, int N)",
      "complexity": "moderate",
      "features": ["shared_memory", "syncthreads"],
      "dependencies": [],
      "translation_notes": "Standard tiled matrix multiplication"
    }
  ],
  "external_libraries": ["cuBLAS", "Thrust"],
  "cuda_version_required": "11.0+",
  "migration_challenges": [
    "Uses cooperative groups in 3 kernels - requires careful SYCL translation",
    "One kernel uses inline PTX - needs manual review"
  ],
  "complexity_breakdown": {
    "simple": 25,
    "moderate": 15,
    "complex": 2
  }
}
```

## Features Detected

The skill identifies these CUDA features in kernels:

### Memory Features
- `shared_memory`: `__shared__` declarations
- `constant_memory`: `__constant__` declarations
- `texture_memory`: Texture objects or references
- `surface_memory`: Surface objects

### Synchronization
- `syncthreads`: `__syncthreads()` calls
- `syncwarp`: `__syncwarp()` calls
- `cooperative_groups`: Cooperative groups API usage

### Advanced Features
- `warp_shuffle`: Warp shuffle operations (`__shfl_*`)
- `atomics`: Atomic operations (atomicAdd, atomicCAS, etc.)
- `inline_ptx`: Inline PTX or assembly code
- `dynamic_parallelism`: Device-side kernel launches

## Complexity Assessment

Kernels are classified as:

- **Simple**: Few features, < 50 lines, straightforward translation
- **Moderate**: Multiple features, 50-200 lines, requires careful translation
- **Complex**: Advanced features (inline PTX, cooperative groups), > 200 lines, needs expert review

## External Library Detection

Identifies usage of:
- cuBLAS (linear algebra)
- cuFFT (Fast Fourier Transform)
- cuRAND (random number generation)
- cuSPARSE (sparse matrix operations)
- cuDNN (deep learning primitives)
- Thrust (parallel algorithms)
- CUB (CUDA building blocks)

## Integration with Claude Code

When invoked by Claude Code agents:

```python
# In subagent or main agent
results = execute_skill("scan-cuda-repo", args=["./cuda-repo"])

# Access results
for kernel in results["kernels"]:
    if kernel["complexity"] == "complex":
        # Flag for manual review
        add_to_review_queue(kernel)
```

## Implementation Details

### Directory Exclusions

Automatically skips:
- `.git`, `.svn`, `.hg` (version control)
- `build`, `bin`, `lib` (build artifacts)
- `__pycache__`, `.pytest_cache` (Python cache)
- `node_modules` (if mixed projects)

### Kernel Extraction Algorithm

1. Uses regex pattern matching for `__global__` functions
2. Extracts full signature including template parameters
3. Finds matching braces to extract kernel body
4. Analyzes body for CUDA-specific features
5. Generates unique kernel identifier

### Error Handling

- Gracefully handles non-UTF8 files
- Continues scanning if individual files fail
- Reports warnings for unreadable files
- Validates CUDA syntax heuristically

## Performance

- Typical scan speed: ~1000 files/second
- Memory usage: ~100MB for large repositories
- Parallel scanning for repositories > 1000 files

## Limitations

- Does not perform deep semantic analysis
- May miss dynamically generated kernels
- Requires valid CUDA syntax for accurate extraction
- Template kernels are simplified in output

## Related Skills

- `analyze-kernel-complexity`: Deep analysis of individual kernels
- `generate-cmake`: Uses scan results to create build files
- `create-cuda-tests`: Uses kernel inventory for test generation

## Skill Metadata

```yaml
name: scan-cuda-repo
version: 1.0.0
category: analysis
dependencies:
  - python >= 3.8
  - No external packages required
inputs:
  - repository_path: string
outputs:
  - scan_results: JSON object
execution_time: 1-30 seconds (depends on repo size)
```
