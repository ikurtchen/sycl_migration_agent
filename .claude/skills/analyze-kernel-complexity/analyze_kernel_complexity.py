#!/usr/bin/env python3
"""
Skill: analyze-kernel-complexity
Performs deep analysis of CUDA kernel computational characteristics for performance modeling.
"""

import re
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class KernelComplexityAnalyzer:
    """Analyzes CUDA kernels for computational and memory characteristics."""

    def __init__(self, platform_spec: Optional[Dict] = None):
        """
        Initialize analyzer with optional platform specifications.

        Args:
            platform_spec: Dictionary with GPU platform specs (FLOPS, bandwidth, etc.)
        """
        self.platform_spec = platform_spec or {}

    def analyze_kernel(self, cuda_file: str, kernel_name: str, 
                      input_size: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Analyze a CUDA kernel's computational characteristics.

        Args:
            cuda_file: Path to CUDA source file
            kernel_name: Name of kernel to analyze
            input_size: Dictionary of input dimensions (e.g., {"N": 4096})

        Returns:
            Comprehensive analysis dictionary
        """
        # Read CUDA file
        with open(cuda_file, 'r', encoding='utf-8', errors='ignore') as f:
            cuda_code = f.read()

        # Extract kernel code
        kernel_code = self._extract_kernel(cuda_code, kernel_name)
        if not kernel_code:
            return {"error": f"Kernel {kernel_name} not found in {cuda_file}"}

        # Determine input size if not provided
        if not input_size:
            input_size = self._infer_input_size(kernel_code, kernel_name)

        # Analyze different aspects
        compute_chars = self._analyze_compute(kernel_code, input_size)
        memory_chars = self._analyze_memory(kernel_code, input_size)
        parallelism = self._analyze_parallelism(kernel_code, input_size)

        # Calculate arithmetic intensity
        arithmetic_intensity = self._calculate_arithmetic_intensity(
            compute_chars, memory_chars
        )

        # Roofline analysis
        roofline = self._roofline_analysis(
            arithmetic_intensity, compute_chars, memory_chars
        )

        # Overall complexity scoring
        complexity = self._calculate_complexity_score(
            kernel_code, compute_chars, memory_chars, parallelism
        )

        # Recognize algorithm pattern
        algorithm_type = self._recognize_algorithm_pattern(kernel_name, kernel_code)

        return {
            "kernel_name": kernel_name,
            "source_file": cuda_file,
            "algorithm_type": algorithm_type,
            "input_size": input_size,
            "compute_characteristics": compute_chars,
            "memory_characteristics": memory_chars,
            "arithmetic_intensity": arithmetic_intensity,
            "parallelism": parallelism,
            "roofline_analysis": roofline,
            "complexity_score": complexity
        }

    def _extract_kernel(self, cuda_code: str, kernel_name: str) -> Optional[str]:
        """Extract kernel function code."""
        pattern = rf'__global__\s+.*?\b{kernel_name}\s*\([^)]*\)\s*\{{'
        match = re.search(pattern, cuda_code, re.DOTALL)

        if not match:
            return None

        # Find matching closing brace
        start = match.end() - 1
        brace_count = 1
        end = start + 1

        while end < len(cuda_code) and brace_count > 0:
            if cuda_code[end] == '{':
                brace_count += 1
            elif cuda_code[end] == '}':
                brace_count -= 1
            end += 1

        return cuda_code[match.start():end]

    def _infer_input_size(self, kernel_code: str, kernel_name: str) -> Dict[str, int]:
        """Infer reasonable input sizes from kernel parameters."""
        # Look for common size parameters
        size_patterns = [
            r'\bint\s+([NMK])\b',
            r'\bsize_t\s+([NMK])\b',
            r'\bconst\s+int\s+([NMK])\b'
        ]

        sizes = {}
        for pattern in size_patterns:
            matches = re.findall(pattern, kernel_code)
            for match in matches:
                if match not in sizes:
                    # Default size for analysis
                    sizes[match] = 4096

        # If no sizes found, use generic N
        if not sizes:
            sizes['N'] = 4096

        return sizes

    def _analyze_compute(self, kernel_code: str, 
                        input_size: Dict[str, int]) -> Dict[str, Any]:
        """Analyze computational characteristics."""
        # Count operations
        operations = self._count_operations(kernel_code)

        # Determine problem size
        total_threads = self._estimate_total_threads(kernel_code, input_size)

        # Calculate total FLOPs
        ops_per_thread = operations['total_per_thread']
        total_ops = ops_per_thread * total_threads

        # Instruction mix
        total_compute = sum([
            operations['fma'], operations['add'], 
            operations['mul'], operations['div']
        ])

        instruction_mix = {
            'fma_percentage': (operations['fma'] / total_compute * 100) if total_compute > 0 else 0,
            'memory_percentage': 0,  # Calculated from memory analysis
            'control_percentage': (operations['control'] / ops_per_thread * 100) if ops_per_thread > 0 else 0
        }

        return {
            'total_operations': int(total_ops),
            'flops': int(total_ops),
            'operation_breakdown': {
                'fma': int(operations['fma'] * total_threads),
                'add': int(operations['add'] * total_threads),
                'mul': int(operations['mul'] * total_threads),
                'div': int(operations['div'] * total_threads),
                'special': int(operations['special'] * total_threads)
            },
            'operation_types': {
                'fp32': int(total_ops),  # Assume FP32 unless specified
                'fp16': 0,
                'int32': int(operations['integer'] * total_threads)
            },
            'instruction_mix': instruction_mix
        }

    def _count_operations(self, kernel_code: str) -> Dict[str, int]:
        """Count different types of operations in kernel."""
        ops = {
            'fma': 0,
            'add': 0,
            'mul': 0,
            'div': 0,
            'special': 0,
            'integer': 0,
            'control': 0,
            'total_per_thread': 0
        }

        # Count FMA patterns (a * b + c)
        fma_pattern = r'\w+\s*\*\s*\w+\s*\+\s*\w+'
        ops['fma'] = len(re.findall(fma_pattern, kernel_code))

        # Count additions (excluding FMA)
        add_pattern = r'\+(?!=)'
        ops['add'] = len(re.findall(add_pattern, kernel_code)) - ops['fma']

        # Count multiplications (excluding FMA)
        mul_pattern = r'\*(?!=)'
        ops['mul'] = len(re.findall(mul_pattern, kernel_code)) - ops['fma']

        # Count divisions
        div_pattern = r'/(?!=)'
        ops['div'] = len(re.findall(div_pattern, kernel_code))

        # Count special functions
        special_funcs = ['sqrt', 'exp', 'log', 'sin', 'cos', 'tan', 'pow']
        for func in special_funcs:
            ops['special'] += len(re.findall(rf'\b{func}\s*\(', kernel_code))

        # Count control flow
        ops['control'] = len(re.findall(r'\b(if|for|while)\b', kernel_code))

        # Total operations per thread (FMA counts as 2 FLOPs)
        ops['total_per_thread'] = (ops['fma'] * 2 + ops['add'] + ops['mul'] + 
                                   ops['div'] + ops['special'] * 10)

        return ops

    def _estimate_total_threads(self, kernel_code: str, 
                                input_size: Dict[str, int]) -> int:
        """Estimate total number of threads from kernel code."""
        # Look for common patterns
        if 'N * N' in kernel_code or 'N*N' in kernel_code:
            N = input_size.get('N', 4096)
            return N * N
        elif 'M * N' in kernel_code or 'M*N' in kernel_code:
            M = input_size.get('M', 4096)
            N = input_size.get('N', 4096)
            return M * N
        else:
            # Default to N
            return input_size.get('N', 4096)

    def _analyze_memory(self, kernel_code: str, 
                       input_size: Dict[str, int]) -> Dict[str, Any]:
        """Analyze memory access characteristics."""
        # Count memory accesses
        reads = self._count_memory_reads(kernel_code)
        writes = self._count_memory_writes(kernel_code)

        # Estimate bytes transferred
        total_threads = self._estimate_total_threads(kernel_code, input_size)
        bytes_per_element = 4  # Assume float32

        bytes_read = reads * total_threads * bytes_per_element
        bytes_written = writes * total_threads * bytes_per_element
        total_bytes = bytes_read + bytes_written

        # Analyze access pattern
        access_pattern = self._analyze_access_pattern(kernel_code)

        # Estimate reuse factor
        reuse_factor = self._estimate_reuse_factor(kernel_code, reads, writes)

        # Working set size
        working_set_mb = self._estimate_working_set(input_size)

        return {
            'bytes_read': int(bytes_read),
            'bytes_written': int(bytes_written),
            'total_bytes': int(total_bytes),
            'access_pattern': access_pattern,
            'reuse_factor': reuse_factor,
            'working_set_size_mb': working_set_mb,
            'cache_behavior': self._predict_cache_behavior(access_pattern, working_set_mb)
        }

    def _count_memory_reads(self, kernel_code: str) -> int:
        """Count memory read operations."""
        # Array accesses on right side of assignment
        read_pattern = r'\w+\[[^\]]+\](?!\s*=)'
        return len(re.findall(read_pattern, kernel_code))

    def _count_memory_writes(self, kernel_code: str) -> int:
        """Count memory write operations."""
        # Array accesses on left side of assignment
        write_pattern = r'\w+\[[^\]]+\]\s*='
        return len(re.findall(write_pattern, kernel_code))

    def _analyze_access_pattern(self, kernel_code: str) -> str:
        """Determine memory access pattern."""
        if '__shared__' in kernel_code and 'threadIdx' in kernel_code:
            return 'tiled_coalesced'
        elif 'threadIdx.x' in kernel_code and '[idx]' in kernel_code:
            return 'coalesced'
        elif 'stride' in kernel_code.lower():
            return 'strided'
        else:
            return 'irregular'

    def _estimate_reuse_factor(self, kernel_code: str, 
                               reads: int, writes: int) -> int:
        """Estimate how many times data is reused."""
        if '__shared__' in kernel_code:
            # Shared memory implies reuse
            return 16
        elif 'for' in kernel_code and reads > writes:
            # Loop with more reads than writes
            return reads // max(writes, 1)
        else:
            return 1

    def _estimate_working_set(self, input_size: Dict[str, int]) -> float:
        """Estimate working set size in MB."""
        total_elements = 1
        for size in input_size.values():
            total_elements *= size

        bytes_per_element = 4  # float32
        bytes_total = total_elements * bytes_per_element
        mb = bytes_total / (1024 * 1024)

        return round(mb, 2)

    def _predict_cache_behavior(self, access_pattern: str, 
                                working_set_mb: float) -> str:
        """Predict cache behavior."""
        if access_pattern in ['coalesced', 'tiled_coalesced']:
            return 'good_spatial_locality'
        elif working_set_mb < 1.0:
            return 'fits_in_cache'
        elif access_pattern == 'strided':
            return 'poor_spatial_locality'
        else:
            return 'cache_thrashing_likely'

    def _calculate_arithmetic_intensity(self, compute: Dict, 
                                        memory: Dict) -> Dict[str, Any]:
        """Calculate arithmetic intensity (FLOPs per byte)."""
        flops = compute['flops']
        bytes_transferred = memory['total_bytes']

        if bytes_transferred == 0:
            ai = float('inf')
        else:
            ai = flops / bytes_transferred

        # Classify
        if ai < 1.0:
            classification = 'memory_bound'
            description = f"{ai:.2f} FLOPs/byte - memory bandwidth limited"
        elif ai < 10.0:
            classification = 'balanced'
            description = f"{ai:.2f} FLOPs/byte - balanced compute and memory"
        else:
            classification = 'compute_bound'
            description = f"{ai:.2f} FLOPs/byte - compute throughput limited"

        return {
            'value': round(ai, 2),
            'classification': classification,
            'description': description
        }

    def _analyze_parallelism(self, kernel_code: str, 
                            input_size: Dict[str, int]) -> Dict[str, Any]:
        """Analyze parallelism characteristics."""
        total_threads = self._estimate_total_threads(kernel_code, input_size)

        # Count operations per thread
        ops = self._count_operations(kernel_code)
        work_per_thread = ops['total_per_thread']

        # Detect divergence
        divergence = self._detect_divergence(kernel_code)

        # Count synchronization points
        sync_points = kernel_code.count('__syncthreads()')

        return {
            'total_threads': total_threads,
            'work_per_thread': work_per_thread,
            'thread_divergence': divergence,
            'synchronization_points': sync_points,
            'critical_sections': 0  # Would need more sophisticated analysis
        }

    def _detect_divergence(self, kernel_code: str) -> str:
        """Detect potential thread divergence."""
        # Look for conditional statements with threadIdx
        if re.search(r'if\s*\([^)]*threadIdx', kernel_code):
            return 'high'
        elif 'if' in kernel_code:
            return 'moderate'
        else:
            return 'none'

    def _roofline_analysis(self, arithmetic_intensity: Dict, 
                          compute: Dict, memory: Dict) -> Dict[str, Any]:
        """Perform roofline analysis."""
        ai = arithmetic_intensity['value']

        # Calculate ridge point from platform specs
        if self.platform_spec:
            peak_gflops = self.platform_spec.get('peak_fp32_tflops', 22.2) * 1000
            bandwidth_gbps = self.platform_spec.get('bandwidth_gbps', 3200)
            ridge_point = peak_gflops / bandwidth_gbps
        else:
            # Default Intel Data Center GPU Max values
            ridge_point = 6.9375  # 22200 / 3200

        # Determine performance regime
        if ai < ridge_point:
            regime = 'memory_bound'
            limiting_factor = 'memory_bandwidth'
        else:
            regime = 'compute_bound'
            limiting_factor = 'compute_throughput'

        return {
            'ridge_point': round(ridge_point, 2),
            'performance_regime': regime,
            'limiting_factor': limiting_factor
        }

    def _calculate_complexity_score(self, kernel_code: str, 
                                    compute: Dict, memory: Dict,
                                    parallelism: Dict) -> Dict[str, str]:
        """Calculate overall complexity scores."""
        # Lines of code
        loc = len([line for line in kernel_code.split('\n') 
                  if line.strip() and not line.strip().startswith('//')])

        # Compute complexity
        ops_per_thread = parallelism['work_per_thread']
        if ops_per_thread < 100:
            compute_complexity = 'low'
        elif ops_per_thread < 1000:
            compute_complexity = 'moderate'
        else:
            compute_complexity = 'high'

        # Memory complexity
        if '__shared__' in kernel_code:
            memory_complexity = 'moderate'
        elif 'texture' in kernel_code.lower():
            memory_complexity = 'high'
        else:
            memory_complexity = 'low'

        # Control complexity
        control_statements = compute['instruction_mix']['control_percentage']
        if control_statements < 5:
            control_complexity = 'low'
        elif control_statements < 15:
            control_complexity = 'moderate'
        else:
            control_complexity = 'high'

        # Overall
        complexity_levels = [compute_complexity, memory_complexity, control_complexity]
        if 'high' in complexity_levels or loc > 200:
            overall = 'complex'
        elif 'moderate' in complexity_levels or loc > 50:
            overall = 'moderate'
        else:
            overall = 'simple'

        return {
            'overall': overall,
            'compute': compute_complexity,
            'memory': memory_complexity,
            'control': control_complexity
        }

    def _recognize_algorithm_pattern(self, kernel_name: str, 
                                     kernel_code: str) -> str:
        """Recognize common algorithm patterns."""
        name_lower = kernel_name.lower()
        code_lower = kernel_code.lower()

        # Matrix operations
        if 'matmul' in name_lower or 'gemm' in name_lower:
            return 'dense_matrix_multiplication'
        elif 'transpose' in name_lower:
            return 'matrix_transpose'
        elif 'spmv' in name_lower or 'sparse' in name_lower:
            return 'sparse_matrix_vector'

        # Vector operations
        elif 'vector' in name_lower and 'add' in name_lower:
            return 'vector_addition'
        elif 'dot' in name_lower or 'reduce' in name_lower:
            return 'reduction'
        elif 'saxpy' in name_lower or 'axpy' in name_lower:
            return 'vector_scaling'

        # Convolution
        elif 'conv' in name_lower:
            if '3d' in name_lower:
                return '3d_convolution'
            else:
                return '2d_convolution'

        # FFT
        elif 'fft' in name_lower:
            return 'fft'

        # Stencil
        elif 'stencil' in name_lower:
            return 'stencil_computation'

        # Default
        else:
            return 'general_computation'


def load_platform_spec(platform_file: str) -> Dict[str, Any]:
    """Load platform specifications from JSON file."""
    try:
        with open(platform_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load platform spec: {e}")
        return {}


def parse_input_size(size_str: str) -> Dict[str, int]:
    """Parse input size string like 'N=4096,M=2048'."""
    sizes = {}
    if size_str:
        for pair in size_str.split(','):
            key, value = pair.split('=')
            sizes[key.strip()] = int(value.strip())
    return sizes


def main():
    parser = argparse.ArgumentParser(
        description='Analyze CUDA kernel computational characteristics'
    )
    parser.add_argument('cuda_file', help='Path to CUDA source file')
    parser.add_argument('kernel_name', help='Name of kernel to analyze')
    parser.add_argument('--platform', help='JSON file with platform specifications')
    parser.add_argument('--input-size', help='Input dimensions (e.g., N=4096,M=4096)')
    parser.add_argument('--output', help='Output JSON file')

    args = parser.parse_args()

    # Load platform spec if provided
    platform_spec = None
    if args.platform:
        platform_spec = load_platform_spec(args.platform)

    # Parse input size
    input_size = parse_input_size(args.input_size) if args.input_size else None

    # Analyze kernel
    analyzer = KernelComplexityAnalyzer(platform_spec)
    results = analyzer.analyze_kernel(args.cuda_file, args.kernel_name, input_size)

    # Output results
    output_json = json.dumps(results, indent=2)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_json)
        print(f"Analysis written to {args.output}")
    else:
        print(output_json)

    return 0


if __name__ == '__main__':
    sys.exit(main())
