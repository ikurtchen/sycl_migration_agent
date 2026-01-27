---
name: profile-gpu-kernel
description: "Performance profiling and analysis for GPU kernels using vendor-specific tools"
---

# profile-gpu-kernel

Performance profiling and analysis for GPU kernels using vendor-specific tools.

## Description

This skill integrates with GPU profiling tools (NVIDIA Nsight, Intel VTune/Advisor) to collect detailed performance metrics, identify bottlenecks, and generate roofline analysis for optimization guidance.

## Usage

```bash
python profile_gpu_kernel.py <platform> <executable> [options]
```

### Arguments

- `platform`: Either "nvidia" or "intel"
- `executable`: Path to test executable containing kernel
- `--kernel-name`: Specific kernel to profile
- `--input-size`: Input dimensions for profiling run
- `--output-dir`: Directory for profiling results
- `--tool`: Profiling tool (nsight-compute, nsight-systems, vtune, advisor)
- `--metrics`: Specific metrics to collect

### Examples

```bash
# Profile CUDA kernel with Nsight Compute
python profile_gpu_kernel.py nvidia ./cuda_tests \
    --kernel-name matrixMul \
    --tool nsight-compute \
    --output-dir ./profiling_results

# Profile SYCL kernel with Intel VTune
python profile_gpu_kernel.py intel ./sycl_tests \
    --kernel-name matrixMul \
    --tool vtune \
    --output-dir ./profiling_results

# Roofline analysis with Intel Advisor
python profile_gpu_kernel.py intel ./sycl_tests \
    --kernel-name matrixMul \
    --tool advisor \
    --metrics roofline
```

## Supported Profiling Tools

### NVIDIA Platform

#### 1. **Nsight Compute** (Kernel-level profiling)

```bash
# Command generated
ncu --set full \
    --kernel-name matrixMul \
    --launch-skip 5 \
    --launch-count 10 \
    --export profiling_results/matrixMul \
    ./cuda_tests
```

**Metrics Collected:**
- Achieved occupancy
- Warp execution efficiency
- Memory throughput (global, shared, L1, L2)
- Compute throughput (SM efficiency, IPC)
- Memory bandwidth utilization
- Cache hit rates

#### 2. **Nsight Systems** (System-level profiling)

```bash
# Command generated
nsys profile \
    --trace=cuda,nvtx \
    --output=profiling_results/timeline \
    --stats=true \
    ./cuda_tests
```

**Metrics Collected:**
- Kernel launch overhead
- Memory transfer times
- Timeline visualization
- API call statistics

### Intel Platform

#### 1. **Intel VTune Profiler** (GPU Hotspots)

```bash
# Command generated
vtune -collect gpu-hotspots \
    -result-dir profiling_results/vtune \
    -knob sampling-interval=1 \
    -- ./sycl_tests
```

**Metrics Collected:**
- GPU Utilization %
- EU Array Active/Stalled %
- Memory Bandwidth Utilization
- L3 Hit Rate
- EU Thread Occupancy

#### 2. **Intel Advisor** (Roofline Analysis)

```bash
# Survey collection
advisor --collect=survey \
    --project-dir=profiling_results/advisor \
    -- ./sycl_tests

# FLOP analysis
advisor --collect=tripcounts \
    --flop \
    --project-dir=profiling_results/advisor \
    -- ./sycl_tests

# Generate roofline
advisor --report=roofline \
    --project-dir=profiling_results/advisor \
    --report-output=profiling_results/roofline.html
```

**Metrics Collected:**
- Arithmetic Intensity
- Roofline positioning
- Performance vs theoretical peak
- Vectorization efficiency
- Memory bandwidth usage

## Output Format

```json
{
  "kernel": "matrixMul",
  "platform": "intel",
  "tool": "vtune",
  "timestamp": "2026-01-26T15:30:00",
  "execution_metrics": {
    "total_time_ms": 2.34,
    "kernel_time_ms": 2.10,
    "overhead_ms": 0.24,
    "throughput_gflops": 15234.5
  },
  "gpu_utilization": {
    "gpu_busy_percent": 87.5,
    "eu_active_percent": 82.3,
    "eu_stalled_percent": 12.1,
    "eu_idle_percent": 5.6
  },
  "memory_metrics": {
    "bandwidth_utilization_percent": 65.4,
    "bandwidth_achieved_gbps": 2092.8,
    "bandwidth_theoretical_gbps": 3200.0,
    "l3_hit_rate_percent": 78.3,
    "bytes_transferred_gb": 4.38
  },
  "compute_metrics": {
    "flops_total": 137438953472,
    "gflops_achieved": 15234.5,
    "gflops_theoretical": 22200.0,
    "efficiency_percent": 68.6,
    "instruction_mix": {
      "fma_percent": 85.3,
      "memory_percent": 12.1,
      "other_percent": 2.6
    }
  },
  "roofline_analysis": {
    "arithmetic_intensity": 512.0,
    "performance_regime": "compute_bound",
    "distance_from_roofline_percent": 31.4,
    "limiting_factor": "compute_throughput"
  },
  "bottlenecks": [
    {
      "type": "suboptimal_occupancy",
      "severity": "medium",
      "description": "EU occupancy at 82%, target >90%",
      "recommendation": "Increase work-group size or reduce register usage"
    },
    {
      "type": "memory_bandwidth",
      "severity": "low",
      "description": "Memory bandwidth at 65% utilization",
      "recommendation": "Consider memory access coalescing"
    }
  ],
  "raw_output_files": [
    "profiling_results/vtune/vtune.vtune",
    "profiling_results/vtune/summary.txt"
  ]
}
```

## Profiling Workflow

### 1. Preparation

```python
def prepare_profiling_run(kernel_name, input_size):
    """Prepare environment for profiling."""
    # Set optimal input size for profiling (not too small, not too large)
    if not input_size:
        input_size = determine_optimal_size(kernel_name)

    # Create output directory
    os.makedirs("profiling_results", exist_ok=True)

    # Set environment variables
    os.environ['ONEAPI_DEVICE_SELECTOR'] = 'level_zero:gpu'

    return {
        "input_size": input_size,
        "output_dir": "profiling_results"
    }
```

### 2. Tool Execution

```python
def run_vtune_profiling(executable, kernel_name):
    """Run Intel VTune GPU hotspots analysis."""
    cmd = [
        "vtune",
        "-collect", "gpu-hotspots",
        "-result-dir", f"profiling_results/vtune_{kernel_name}",
        "-knob", "sampling-interval=1",
        "--", executable
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse VTune output
    metrics = parse_vtune_output(f"profiling_results/vtune_{kernel_name}")

    return metrics
```

### 3. Metrics Extraction

```python
def parse_vtune_output(result_dir):
    """Extract metrics from VTune results."""
    # Generate summary report
    subprocess.run([
        "vtune", "-report", "summary",
        "-result-dir", result_dir,
        "-format", "csv",
        "-report-output", f"{result_dir}/summary.csv"
    ])

    # Parse CSV
    metrics = {}
    with open(f"{result_dir}/summary.csv") as f:
        # Parse VTune CSV format
        # ... extraction logic

    return metrics
```

### 4. Roofline Generation

```python
def generate_roofline(executable, advisor_project):
    """Generate roofline analysis with Intel Advisor."""
    # Survey
    subprocess.run([
        "advisor", "--collect=survey",
        f"--project-dir={advisor_project}",
        "--", executable
    ])

    # FLOP analysis
    subprocess.run([
        "advisor", "--collect=tripcounts",
        "--flop",
        f"--project-dir={advisor_project}",
        "--", executable
    ])

    # Generate roofline report
    subprocess.run([
        "advisor", "--report=roofline",
        f"--project-dir={advisor_project}",
        f"--report-output={advisor_project}/roofline.html"
    ])

    # Parse roofline data
    roofline_data = parse_advisor_roofline(advisor_project)

    return roofline_data
```

## Metrics Reference

### GPU Utilization Metrics

- **GPU Busy %**: Percentage of time GPU had work
- **EU Active %**: Execution units actively computing
- **EU Stalled %**: Execution units waiting (memory, barriers)
- **EU Idle %**: Execution units with no work

**Target**: >85% GPU Busy, >80% EU Active

### Memory Metrics

- **Bandwidth Utilization %**: Achieved / Theoretical bandwidth
- **L3 Hit Rate %**: Percentage of L3 cache hits
- **Bytes Transferred**: Total data movement

**Target**: >70% bandwidth utilization for memory-bound kernels

### Compute Metrics

- **GFLOPS**: Achieved floating-point operations per second
- **Efficiency %**: Achieved / Theoretical performance
- **IPC**: Instructions per cycle

**Target**: >70% efficiency for compute-bound kernels

### Occupancy Metrics

- **Thread Occupancy %**: Active threads / Max threads
- **Wave Occupancy %**: Active wavefronts / Max wavefronts
- **Register Usage**: Registers per thread

**Target**: >75% occupancy

## Bottleneck Identification

### Memory Bottlenecks

```python
def identify_memory_bottlenecks(metrics):
    """Detect memory-related performance issues."""
    bottlenecks = []

    if metrics['memory_bandwidth_utilization'] > 80:
        bottlenecks.append({
            "type": "memory_bandwidth_saturated",
            "severity": "high",
            "recommendation": "Reduce memory accesses or improve cache reuse"
        })

    if metrics['l3_hit_rate'] < 50:
        bottlenecks.append({
            "type": "poor_cache_locality",
            "severity": "medium",
            "recommendation": "Improve spatial/temporal locality"
        })

    return bottlenecks
```

### Compute Bottlenecks

```python
def identify_compute_bottlenecks(metrics):
    """Detect compute-related performance issues."""
    bottlenecks = []

    if metrics['eu_active_percent'] < 70:
        bottlenecks.append({
            "type": "low_eu_utilization",
            "severity": "high",
            "recommendation": "Increase parallelism or reduce divergence"
        })

    if metrics['compute_efficiency'] < 60:
        bottlenecks.append({
            "type": "suboptimal_instruction_mix",
            "severity": "medium",
            "recommendation": "Profile instruction mix and optimize"
        })

    return bottlenecks
```

## Visualization

### Roofline Plot Generation

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_roofline(platform_specs, kernel_metrics, output_file):
    """Generate roofline plot."""
    # Platform specs
    peak_gflops = platform_specs['peak_fp32_tflops'] * 1000
    bandwidth_gbps = platform_specs['bandwidth_gbps']

    # Ridge point
    ridge_point = peak_gflops / bandwidth_gbps

    # Roofline
    ai_range = np.logspace(-2, 3, 100)
    roofline = np.minimum(
        peak_gflops,
        ai_range * bandwidth_gbps
    )

    # Plot
    plt.figure(figsize=(10, 6))
    plt.loglog(ai_range, roofline, 'b-', linewidth=2, label='Roofline')

    # Plot kernel
    kernel_ai = kernel_metrics['arithmetic_intensity']
    kernel_gflops = kernel_metrics['achieved_gflops']

    plt.loglog(kernel_ai, kernel_gflops, 'ro', markersize=10, 
              label=kernel_metrics['name'])

    plt.xlabel('Arithmetic Intensity (FLOPs/Byte)')
    plt.ylabel('Performance (GFLOPS)')
    plt.title('Roofline Model')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig(output_file)
    plt.close()
```

## Integration with Optimization

```python
def profile_and_optimize(kernel, platform):
    """Profile kernel and suggest optimizations."""
    # Run profiling
    metrics = profile_kernel(kernel, platform)

    # Identify bottlenecks
    bottlenecks = identify_all_bottlenecks(metrics)

    # Generate recommendations
    optimizations = []

    for bottleneck in bottlenecks:
        if bottleneck['type'] == 'memory_bandwidth_saturated':
            optimizations.append({
                "strategy": "reduce_memory_traffic",
                "techniques": [
                    "Use shared/local memory for data reuse",
                    "Increase arithmetic intensity",
                    "Consider fusion with other kernels"
                ]
            })
        elif bottleneck['type'] == 'low_occupancy':
            optimizations.append({
                "strategy": "increase_occupancy",
                "techniques": [
                    "Reduce register usage",
                    "Decrease shared memory usage",
                    "Tune work-group size"
                ]
            })

    return {
        "metrics": metrics,
        "bottlenecks": bottlenecks,
        "optimizations": optimizations
    }
```

## Related Skills

- `analyze-kernel-complexity`: Provides theoretical analysis to compare against
- `optimize-sycl-kernel`: Uses profiling results to guide optimizations
- `execute-remote-ssh`: Runs profiling on remote servers

## Skill Metadata

```yaml
name: profile-gpu-kernel
version: 1.0.0
category: performance
dependencies:
  - python >= 3.8
  - matplotlib, numpy (for visualization)
  - nsight-compute or nsight-systems (NVIDIA)
  - vtune or advisor (Intel)
inputs:
  - platform: string (nvidia|intel)
  - executable: string
  - kernel_name: string
outputs:
  - profiling_metrics: JSON object
  - raw_profiler_outputs: files
execution_time: 10-120 seconds
```
