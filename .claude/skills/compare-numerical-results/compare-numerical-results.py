#!/usr/bin/env python3
"""
Skill: compare-numerical-results
Compares CUDA and SYCL kernel outputs for numerical correctness.
"""

import numpy as np
import json
import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional


class NumericalComparator:
    """Compare numerical results from CUDA and SYCL kernels."""

    def __init__(self, rtol: float = 1e-5, atol: float = 1e-8):
        """
        Initialize comparator with tolerance settings.

        Args:
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
        """
        self.rtol = rtol
        self.atol = atol

    def compare_binary_files(self, cuda_file: str, sycl_file: str,
                           dtype: str = 'float32') -> Dict[str, Any]:
        """
        Compare binary output files from CUDA and SYCL kernels.

        Args:
            cuda_file: Path to CUDA output binary
            sycl_file: Path to SYCL output binary
            dtype: Data type of arrays ('float32', 'float64', 'int32', etc.)

        Returns:
            Comparison report dictionary
        """
        # Check if files exist
        if not os.path.exists(cuda_file):
            return {
                "status": "error",
                "error": f"CUDA file not found: {cuda_file}"
            }

        if not os.path.exists(sycl_file):
            return {
                "status": "error",
                "error": f"SYCL file not found: {sycl_file}"
            }

        try:
            # Convert dtype string to numpy dtype
            np_dtype = getattr(np, dtype)

            cuda_data = np.fromfile(cuda_file, dtype=np_dtype)
            sycl_data = np.fromfile(sycl_file, dtype=np_dtype)
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to load files: {e}"
            }

        return self.compare_arrays(cuda_data, sycl_data, cuda_file, sycl_file)

    def compare_arrays(self, cuda_array: np.ndarray, 
                      sycl_array: np.ndarray,
                      cuda_file: str = "cuda_output",
                      sycl_file: str = "sycl_output") -> Dict[str, Any]:
        """
        Compare two numpy arrays for numerical equivalence.

        Args:
            cuda_array: CUDA kernel output
            sycl_array: SYCL kernel output
            cuda_file: Source file name for reporting
            sycl_file: Source file name for reporting

        Returns:
            Detailed comparison report
        """
        report = {
            "status": "unknown",
            "cuda_file": cuda_file,
            "sycl_file": sycl_file,
            "rtol": self.rtol,
            "atol": self.atol,
        }

        # Shape check
        if cuda_array.shape != sycl_array.shape:
            report["status"] = "shape_mismatch"
            report["cuda_shape"] = list(cuda_array.shape)
            report["sycl_shape"] = list(sycl_array.shape)
            report["error"] = f"Shape mismatch: CUDA {cuda_array.shape} vs SYCL {sycl_array.shape}"
            return report

        total_elements = cuda_array.size
        report["total_elements"] = int(total_elements)
        report["cuda_shape"] = list(cuda_array.shape)
        report["sycl_shape"] = list(sycl_array.shape)
        report["dtype"] = str(cuda_array.dtype)

        # Handle special cases (NaN, Inf)
        cuda_has_nan = np.any(np.isnan(cuda_array))
        sycl_has_nan = np.any(np.isnan(sycl_array))
        cuda_has_inf = np.any(np.isinf(cuda_array))
        sycl_has_inf = np.any(np.isinf(sycl_array))

        report["special_values"] = {
            "cuda_has_nan": bool(cuda_has_nan),
            "sycl_has_nan": bool(sycl_has_nan),
            "cuda_has_inf": bool(cuda_has_inf),
            "sycl_has_inf": bool(sycl_has_inf)
        }

        # Element-wise comparison
        close = np.isclose(cuda_array, sycl_array, rtol=self.rtol, atol=self.atol, 
                          equal_nan=True)
        matching_elements = np.sum(close)
        match_rate = matching_elements / total_elements

        report["matching_elements"] = int(matching_elements)
        report["match_rate"] = float(match_rate)
        report["match_percentage"] = float(match_rate * 100)

        # Statistical analysis
        abs_diff = np.abs(cuda_array - sycl_array)

        # Filter out NaN and Inf for statistics
        valid_mask = np.isfinite(abs_diff)
        valid_abs_diff = abs_diff[valid_mask]

        if len(valid_abs_diff) > 0:
            report["statistics"] = {
                "max_absolute_diff": float(np.max(valid_abs_diff)),
                "mean_absolute_diff": float(np.mean(valid_abs_diff)),
                "median_absolute_diff": float(np.median(valid_abs_diff)),
                "std_absolute_diff": float(np.std(valid_abs_diff)),
                "min_absolute_diff": float(np.min(valid_abs_diff))
            }
        else:
            report["statistics"] = {
                "max_absolute_diff": 0.0,
                "mean_absolute_diff": 0.0,
                "median_absolute_diff": 0.0,
                "std_absolute_diff": 0.0,
                "min_absolute_diff": 0.0
            }

        # For floating point, compute relative error where possible
        if np.issubdtype(cuda_array.dtype, np.floating):
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_diff = abs_diff / np.abs(cuda_array)
                rel_diff_valid = rel_diff[np.isfinite(rel_diff)]

                if len(rel_diff_valid) > 0:
                    report["statistics"]["max_relative_diff"] = float(np.max(rel_diff_valid))
                    report["statistics"]["mean_relative_diff"] = float(np.mean(rel_diff_valid))
                    report["statistics"]["median_relative_diff"] = float(np.median(rel_diff_valid))

        # Determine status
        if match_rate == 1.0:
            report["status"] = "perfect_match"
            return report
        elif match_rate >= 0.9999:
            report["status"] = "excellent_match"
        elif match_rate >= 0.99:
            report["status"] = "good_match"
        elif match_rate >= 0.90:
            report["status"] = "partial_match"
        else:
            report["status"] = "mismatch"

        # Analyze mismatches
        mismatch_indices = np.where(~close)[0]
        report["mismatch_count"] = len(mismatch_indices)

        # Pattern analysis
        report["mismatch_pattern"] = self._analyze_mismatch_pattern(
            mismatch_indices, total_elements, cuda_array.shape
        )

        # Sample mismatches (up to 20)
        sample_size = min(20, len(mismatch_indices))
        sample_indices = mismatch_indices[:sample_size]

        report["sample_mismatches"] = []
        for idx in sample_indices:
            cuda_val = cuda_array.flat[idx]
            sycl_val = sycl_array.flat[idx]
            abs_diff_val = abs_diff.flat[idx]

            mismatch_info = {
                "flat_index": int(idx),
                "multi_index": self._flat_to_multi_index(idx, cuda_array.shape),
                "cuda_value": float(cuda_val) if np.isfinite(cuda_val) else str(cuda_val),
                "sycl_value": float(sycl_val) if np.isfinite(sycl_val) else str(sycl_val),
                "absolute_diff": float(abs_diff_val) if np.isfinite(abs_diff_val) else str(abs_diff_val),
            }

            # Add relative diff if applicable
            if np.isfinite(cuda_val) and cuda_val != 0:
                mismatch_info["relative_diff"] = float(abs_diff_val / abs(cuda_val))
            else:
                mismatch_info["relative_diff"] = None

            report["sample_mismatches"].append(mismatch_info)

        # Diagnosis suggestions
        report["diagnosis"] = self._diagnose_mismatches(report)

        return report

    def _flat_to_multi_index(self, flat_idx: int, shape: Tuple) -> List[int]:
        """Convert flat index to multi-dimensional index."""
        if len(shape) == 1:
            return [flat_idx]

        indices = []
        remaining = flat_idx

        for dim_size in reversed(shape):
            indices.insert(0, remaining % dim_size)
            remaining //= dim_size

        return indices

    def _analyze_mismatch_pattern(self, mismatch_indices: np.ndarray,
                                 total_elements: int, 
                                 shape: Tuple) -> Dict[str, Any]:
        """Analyze spatial/structural patterns in mismatches."""
        pattern = {
            "type": "unknown",
            "description": "",
            "percentage": 0.0
        }

        if len(mismatch_indices) == 0:
            pattern["type"] = "none"
            pattern["description"] = "No mismatches found"
            return pattern

        pattern["percentage"] = (len(mismatch_indices) / total_elements) * 100

        # Check if mismatches are at boundaries
        if len(shape) > 1:
            # For multi-dimensional arrays, check edges
            sample_size = min(100, len(mismatch_indices))
            multi_indices = [self._flat_to_multi_index(idx, shape) 
                           for idx in mismatch_indices[:sample_size]]

            edge_mismatches = sum(1 for idx in multi_indices 
                                if any(i == 0 or i == s-1 
                                      for i, s in zip(idx, shape)))

            if sample_size > 0 and edge_mismatches / sample_size > 0.7:
                pattern["type"] = "boundary"
                pattern["description"] = "Mismatches concentrated at array boundaries"
                return pattern

        # Check if mismatches are random or clustered
        if len(mismatch_indices) > 2:
            spacing = np.diff(sorted(mismatch_indices[:1000]))
            if len(spacing) > 0:
                mean_spacing = np.mean(spacing)
                std_spacing = np.std(spacing)

                if std_spacing / (mean_spacing + 1e-10) < 0.5:
                    pattern["type"] = "periodic"
                    pattern["description"] = f"Mismatches occur periodically (mean spacing: {mean_spacing:.1f})"
                elif std_spacing / (mean_spacing + 1e-10) > 2.0:
                    pattern["type"] = "clustered"
                    pattern["description"] = "Mismatches appear clustered in certain regions"
                else:
                    pattern["type"] = "scattered"
                    pattern["description"] = "Mismatches randomly scattered throughout array"

        return pattern

    def _diagnose_mismatches(self, report: Dict[str, Any]) -> List[str]:
        """Provide diagnostic suggestions based on mismatch patterns."""
        diagnoses = []

        match_rate = report.get("match_rate", 0)
        max_abs_diff = report["statistics"]["max_absolute_diff"]
        pattern_type = report.get("mismatch_pattern", {}).get("type", "unknown")

        # High match rate but not perfect
        if match_rate > 0.999:
            diagnoses.append("Excellent match rate (>99.9%) - likely minor floating-point precision differences")
            diagnoses.append("Recommendation: Consider this a successful match")
        elif match_rate > 0.99:
            diagnoses.append("Very high match rate (>99%) - likely acceptable precision differences")
            diagnoses.append("Recommendation: Review specific mismatches or adjust tolerance")

        # Boundary issues
        if pattern_type == "boundary":
            diagnoses.append("Mismatches concentrated at array boundaries")
            diagnoses.append("Recommendation: Check boundary condition handling in SYCL kernel")
            diagnoses.append("Recommendation: Verify index guard conditions (e.g., if (idx < N))")
            diagnoses.append("Recommendation: Check global/local range calculations")

        # Large differences
        if max_abs_diff > 1.0:
            diagnoses.append(f"Large differences detected (max: {max_abs_diff:.2e})")
            diagnoses.append("Recommendation: Verify algorithm correctness in SYCL translation")
            diagnoses.append("Recommendation: Check for uninitialized memory")
            diagnoses.append("Recommendation: Look for potential race conditions or missing synchronization")

        # Many mismatches
        if match_rate < 0.9:
            diagnoses.append(f"Low match rate ({match_rate*100:.1f}%) indicates systematic issue")
            diagnoses.append("Recommendation: Review thread indexing translation (threadIdx, blockIdx)")
            diagnoses.append("Recommendation: Verify memory access patterns")
            diagnoses.append("Recommendation: Check synchronization points (__syncthreads → item.barrier)")

        # Periodic pattern
        if pattern_type == "periodic":
            diagnoses.append("Periodic mismatch pattern detected")
            diagnoses.append("Recommendation: Check stride calculations or tiling logic")
            diagnoses.append("Recommendation: Verify loop index translations")

        # Clustered pattern
        if pattern_type == "clustered":
            diagnoses.append("Clustered mismatches suggest localized issue")
            diagnoses.append("Recommendation: Check work-group size and memory access patterns")
            diagnoses.append("Recommendation: Verify shared/local memory usage")

        # Special values
        special = report.get("special_values", {})
        if special.get("cuda_has_nan") or special.get("sycl_has_nan"):
            diagnoses.append("NaN values detected in output")
            diagnoses.append("Recommendation: Check for division by zero or invalid operations")

        if special.get("cuda_has_inf") or special.get("sycl_has_inf"):
            diagnoses.append("Infinity values detected in output")
            diagnoses.append("Recommendation: Check for overflow conditions")

        return diagnoses

    def generate_report(self, comparison_result: Dict[str, Any], 
                       output_file: str = None) -> str:
        """Generate human-readable comparison report."""

        if comparison_result.get("status") == "error":
            report = f"ERROR: {comparison_result.get('error')}\n"
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(report)
            return report

        lines = []
        lines.append("=" * 80)
        lines.append("CUDA vs SYCL Numerical Comparison Report")
        lines.append("=" * 80)
        lines.append("")

        # Files
        lines.append("Files Compared:")
        lines.append(f"  CUDA: {comparison_result.get('cuda_file', 'N/A')}")
        lines.append(f"  SYCL: {comparison_result.get('sycl_file', 'N/A')}")
        lines.append("")

        # Status
        status = comparison_result["status"]
        status_symbols = {
            "perfect_match": "✓✓✓",
            "excellent_match": "✓✓",
            "good_match": "✓",
            "partial_match": "~",
            "mismatch": "✗"
        }
        status_symbol = status_symbols.get(status, "?")
        lines.append(f"Status: {status_symbol} {status.upper().replace('_', ' ')}")
        lines.append("")

        # Basic info
        lines.append("Array Information:")
        lines.append(f"  Shape: {comparison_result['cuda_shape']}")
        lines.append(f"  Data Type: {comparison_result.get('dtype', 'unknown')}")
        lines.append(f"  Total Elements: {comparison_result['total_elements']:,}")
        lines.append(f"  Matching Elements: {comparison_result['matching_elements']:,}")
        lines.append(f"  Match Rate: {comparison_result['match_percentage']:.4f}%")
        lines.append("")

        # Tolerance
        lines.append("Comparison Tolerance:")
        lines.append(f"  Relative: {comparison_result['rtol']:.2e}")
        lines.append(f"  Absolute: {comparison_result['atol']:.2e}")
        lines.append("")

        # Special values
        if comparison_result.get("special_values"):
            special = comparison_result["special_values"]
            if any(special.values()):
                lines.append("Special Values:")
                if special.get("cuda_has_nan"):
                    lines.append("  ⚠ CUDA output contains NaN values")
                if special.get("sycl_has_nan"):
                    lines.append("  ⚠ SYCL output contains NaN values")
                if special.get("cuda_has_inf"):
                    lines.append("  ⚠ CUDA output contains Inf values")
                if special.get("sycl_has_inf"):
                    lines.append("  ⚠ SYCL output contains Inf values")
                lines.append("")

        # Statistics
        stats = comparison_result.get("statistics", {})
        if stats:
            lines.append("Difference Statistics:")
            lines.append(f"  Max Absolute Diff: {stats.get('max_absolute_diff', 0):.6e}")
            lines.append(f"  Mean Absolute Diff: {stats.get('mean_absolute_diff', 0):.6e}")
            lines.append(f"  Median Absolute Diff: {stats.get('median_absolute_diff', 0):.6e}")
            lines.append(f"  Std Dev Absolute Diff: {stats.get('std_absolute_diff', 0):.6e}")
            if "max_relative_diff" in stats:
                lines.append(f"  Max Relative Diff: {stats['max_relative_diff']:.6e}")
                lines.append(f"  Mean Relative Diff: {stats.get('mean_relative_diff', 0):.6e}")
            lines.append("")

        # Mismatch pattern
        if comparison_result.get("mismatch_pattern"):
            pattern = comparison_result["mismatch_pattern"]
            lines.append(f"Mismatch Pattern: {pattern['type'].upper()}")
            if pattern.get("description"):
                lines.append(f"  {pattern['description']}")
            if pattern.get("percentage"):
                lines.append(f"  Percentage: {pattern['percentage']:.4f}%")
            lines.append("")

        # Sample mismatches
        if comparison_result.get("sample_mismatches"):
            lines.append(f"Sample Mismatches (showing up to {len(comparison_result['sample_mismatches'])}):")
            lines.append(f"{'Index':<20} {'CUDA Value':<20} {'SYCL Value':<20} {'Abs Diff':<15}")
            lines.append("-" * 80)

            for sample in comparison_result["sample_mismatches"][:20]:
                multi_idx = str(tuple(sample['multi_index']))
                cuda_val = sample['cuda_value']
                sycl_val = sample['sycl_value']
                abs_diff = sample['absolute_diff']

                # Format values
                if isinstance(cuda_val, str):
                    cuda_str = cuda_val
                else:
                    cuda_str = f"{cuda_val:.6e}"

                if isinstance(sycl_val, str):
                    sycl_str = sycl_val
                else:
                    sycl_str = f"{sycl_val:.6e}"

                if isinstance(abs_diff, str):
                    diff_str = abs_diff
                else:
                    diff_str = f"{abs_diff:.6e}"

                lines.append(f"{multi_idx:<20} {cuda_str:<20} {sycl_str:<20} {diff_str:<15}")
            lines.append("")

        # Diagnosis
        if comparison_result.get("diagnosis"):
            lines.append("Diagnostic Suggestions:")
            for i, suggestion in enumerate(comparison_result["diagnosis"], 1):
                lines.append(f"  {i}. {suggestion}")
            lines.append("")

        lines.append("=" * 80)

        report = "\n".join(lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)

        return report


def compare_numerical_results(cuda_file: str, sycl_file: str,
                             dtype: str = 'float32',
                             rtol: float = 1e-5, 
                             atol: float = 1e-8,
                             report_file: str = None,
                             json_file: str = None) -> Dict[str, Any]:
    """
    Main function to compare CUDA and SYCL outputs.

    Args:
        cuda_file: Path to CUDA output binary
        sycl_file: Path to SYCL output binary  
        dtype: Data type of arrays
        rtol: Relative tolerance
        atol: Absolute tolerance
        report_file: Optional path to save text report
        json_file: Optional path to save JSON report

    Returns:
        Comparison result dictionary
    """
    comparator = NumericalComparator(rtol=rtol, atol=atol)
    result = comparator.compare_binary_files(cuda_file, sycl_file, dtype)

    # Generate and print report
    report_text = comparator.generate_report(result, report_file)
    print(report_text)

    # Save JSON if requested
    if json_file:
        with open(json_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nJSON report saved to: {json_file}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Compare CUDA and SYCL kernel outputs for numerical correctness'
    )
    parser.add_argument('cuda_file', help='Path to CUDA output binary file')
    parser.add_argument('sycl_file', help='Path to SYCL output binary file')
    parser.add_argument('--dtype', default='float32',
                       help='Data type (float32, float64, int32, etc.)')
    parser.add_argument('--rtol', type=float, default=1e-5,
                       help='Relative tolerance (default: 1e-5)')
    parser.add_argument('--atol', type=float, default=1e-8,
                       help='Absolute tolerance (default: 1e-8)')
    parser.add_argument('--report', help='Save text report to file')
    parser.add_argument('--json', help='Save JSON report to file')

    args = parser.parse_args()

    result = compare_numerical_results(
        args.cuda_file,
        args.sycl_file,
        args.dtype,
        args.rtol,
        args.atol,
        args.report,
        args.json
    )

    # Exit with error code if comparison failed
    if result.get("status") == "error":
        return 1
    elif result.get("status") in ["mismatch", "partial_match"]:
        return 2
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
