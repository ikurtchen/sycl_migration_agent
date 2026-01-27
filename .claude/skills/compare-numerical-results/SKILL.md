---
name: compare-numerical-results
description: "Compares CUDA and SYCL kernel outputs for numerical correctness."
---

# compare-numerical-results

Compares CUDA and SYCL kernel outputs for numerical correctness.

## Description

This skill compares the numerical outputs of CUDA and SYCL kernels by reading binary output files, checking for shape consistency, and evaluating element-wise closeness within specified tolerances. It generates a detailed report on mismatches, including match rates and sample discrepancies.

## Usage

### Arguments

`cuda_file`: Path to CUDA output binary file
`sycl_file`: Path to SYCL output binary file
`--dtype` : Data type (float32, float64, int32, etc.), default: 'float32'
`--rtol`: Relative tolerance (default: 1e-5)
`--atol`: Absolute tolerance (default: 1e-8)
`--report`: Save text report to file
`--json`: Save JSON report to file

### Examples
```bash
# Basic comparison
python compare_numerical_results.py cuda_output.bin sycl_output.bin

# With custom tolerance
python compare_numerical_results.py cuda_output.bin sycl_output.bin \
    --rtol 1e-4 --atol 1e-6

# Save reports
python compare_numerical_results.py cuda_output.bin sycl_output.bin \
    --report comparison.txt --json comparison.json

# Different data type
python compare_numerical_results.py cuda_output.bin sycl_output.bin \
    --dtype float64
```
