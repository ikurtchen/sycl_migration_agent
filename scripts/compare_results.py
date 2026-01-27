#!/usr/bin/env python3
"""
SYCL vs CUDA Results Comparison Tool
Compares numerical outputs and generates detailed reports
"""

import json
import numpy as np
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

def load_binary_file(filepath, dtype=np.float32):
    """Load binary data from file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Binary file not found: {filepath}")

    return np.fromfile(filepath, dtype=dtype)

def load_json_file(filepath):
    """Load JSON data from file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    with open(filepath, 'r') as f:
        return json.load(f)

def compare_arrays(cuda_data, sycl_data, rtol=1e-6, atol=1e-9):
    """Compare two numpy arrays and return comparison metrics"""

    if cuda_data.shape != sycl_data.shape:
        return {
            "match": False,
            "error": f"Shape mismatch: CUDA {cuda_data.shape} vs SYCL {sycl_data.shape}",
            "shape_mismatch": True
        }

    # Use numpy.isclose for comparison
    close = np.isclose(cuda_data, sycl_data, rtol=rtol, atol=atol)
    match_count = np.sum(close)
    total_count = close.size
    match_rate = match_count / total_count

    # Find mismatches
    if match_rate < 1.0:
        mismatches = np.where(~close)[0]
        differences = np.abs(cuda_data - sycl_data)
        max_diff = np.max(differences)
        mean_diff = np.mean(differences)

        # Sample mismatches for report
        sample_size = min(10, len(mismatches))
        sample_indices = np.random.choice(mismatches, sample_size, replace=False) if len(mismatches) > sample_size else mismatches

        sample_mismatches = []
        for idx in sample_indices:
            sample_mismatches.append({
                "index": int(idx),
                "cuda": float(cuda_data[idx]),
                "sycl": float(sycl_data[idx]),
                "diff": float(cuda_data[idx] - sycl_data[idx]),
                "relative_error": float(abs(cuda_data[idx] - sycl_data[idx]) / (abs(cuda_data[idx]) + 1e-12))
            })

        return {
            "match": False,
            "match_rate": match_rate,
            "total_elements": int(total_count),
            "mismatches": int(total_count - match_count),
            "max_absolute_diff": float(max_diff),
            "mean_absolute_diff": float(mean_diff),
            "sample_mismatches": sample_mismatches
        }

    return {
        "match": True,
        "match_rate": 1.0,
        "total_elements": int(total_count),
        "mismatches": 0
    }

def generate_comparison_report(cuda_results_dir, sycl_results_dir, kernel_name, rtol=1e-6, atol=1e-9):
    """Generate comprehensive comparison report"""

    report = {
        "kernel": kernel_name,
        "timestamp": datetime.now().isoformat(),
        "tolerances": {
            "relative": rtol,
            "absolute": atol
        },
        "cuda_results_dir": cuda_results_dir,
        "sycl_results_dir": sycl_results_dir,
        "comparisons": {}
    }

    # Load benchmark data
    cuda_benchmark = None
    sycl_benchmark = None

    cuda_benchmark_file = Path(cuda_results_dir) / f"benchmark_{kernel_name}.json"
    sycl_benchmark_file = Path(sycl_results_dir) / f"benchmark_{kernel_name}.json"

    if cuda_benchmark_file.exists():
        cuda_benchmark = load_json_file(cuda_benchmark_file)
        report["cuda_benchmark"] = cuda_benchmark

    if sycl_benchmark_file.exists():
        sycl_benchmark = load_json_file(sycl_benchmark_file)
        report["sycl_benchmark"] = sycl_benchmark

    # Find all output binary files
    cuda_dir = Path(cuda_results_dir)
    sycl_dir = Path(sycl_results_dir)

    cuda_outputs = list(cuda_dir.glob("output_*.bin"))
    sycl_outputs = list(sycl_dir.glob("output_*.bin"))

    if not cuda_outputs and not sycl_outputs:
        # Try to find JSON results
        cuda_json = list(cuda_dir.glob("results_*.json"))
        sycl_json = list(sycl_dir.glob("results_*.json"))

        if cuda_json and sycl_json:
            return compare_json_results(cuda_json[0], sycl_json[0], kernel_name, rtol, atol)

    # Compare binary outputs
    overall_match = True
    total_comparisons = 0

    for cuda_file in cuda_outputs:
        filename = cuda_file.name
        sycl_file = sycl_dir / filename

        if not sycl_file.exists():
            print(f"Warning: SYCL output file not found: {filename}")
            continue

        try:
            # Load binary data
            cuda_data = load_binary_file(cuda_file)
            sycl_data = load_binary_file(sycl_file)

            # Compare arrays
            comparison_result = compare_arrays(cuda_data, sycl_data, rtol, atol)
            comparison_result["file"] = filename
            comparison_result["cuda_file"] = str(cuda_file)
            comparison_result["sycl_file"] = str(sycl_file)

            report["comparisons"][filename] = comparison_result

            if not comparison_result["match"]:
                overall_match = False

            total_comparisons += 1

        except Exception as e:
            print(f"Error comparing {filename}: {e}")
            report["comparisons"][filename] = {
                "match": False,
                "error": str(e)
            }

    # Add summary
    report["summary"] = {
        "overall_match": overall_match,
        "total_comparisons": total_comparisons,
        "matched_comparisons": sum(1 for c in report["comparisons"].values() if c.get("match", False)),
        "failed_comparisons": sum(1 for c in report["comparisons"].values() if not c.get("match", False))
    }

    # Add performance comparison if available
    if cuda_benchmark and sycl_benchmark:
        cuda_time = cuda_benchmark.get("execution_time_seconds", 0)
        sycl_time = sycl_benchmark.get("execution_time_seconds", 0)

        if cuda_time > 0 and sycl_time > 0:
            speedup = cuda_time / sycl_time
            report["performance_comparison"] = {
                "cuda_execution_time": cuda_time,
                "sycl_execution_time": sycl_time,
                "speedup_ratio": speedup,
                "performance_percentage": (1.0 / speedup) * 100 if speedup > 0 else 0
            }

    return report

def compare_json_results(cuda_json_file, sycl_json_file, kernel_name, rtol, atol):
    """Compare JSON test results"""

    cuda_data = load_json_file(cuda_json_file)
    sycl_data = load_json_file(sycl_json_file)

    report = {
        "kernel": kernel_name,
        "timestamp": datetime.now().isoformat(),
        "comparison_type": "json",
        "cuda_results_file": str(cuda_json_file),
        "sycl_results_file": str(sycl_json_file)
    }

    # Compare test outcomes
    cuda_tests = cuda_data.get("testsuites", [])
    sycl_tests = sycl_data.get("testsuites", [])

    report["test_comparison"] = {
        "cuda_total_tests": sum(ts.get("tests", 0) for ts in cuda_tests),
        "sycl_total_tests": sum(ts.get("tests", 0) for ts in sycl_tests),
        "cuda_passed": sum(ts.get("tests", 0) - ts.get("failures", 0) for ts in cuda_tests),
        "sycl_passed": sum(ts.get("tests", 0) - ts.get("failures", 0) for ts in sycl_tests)
    }

    report["summary"] = {
        "overall_match": report["test_comparison"]["cuda_passed"] == report["test_comparison"]["sycl_passed"],
        "test_counts_match": report["test_comparison"]["cuda_total_tests"] == report["test_comparison"]["sycl_total_tests"]
    }

    return report

def main():
    parser = argparse.ArgumentParser(description="Compare CUDA and SYCL execution results")
    parser.add_argument("--kernel", required=True, help="Kernel name for comparison")
    parser.add_argument("--cuda-dir", required=True, help="CUDA results directory")
    parser.add_argument("--sycl-dir", required=True, help="SYCL results directory")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-9, help="Absolute tolerance")
    parser.add_argument("--output-dir", default=None, help="Output directory for report")

    args = parser.parse_args()

    # Generate comparison report
    report = generate_comparison_report(
        args.cuda_dir,
        args.sycl_dir,
        args.kernel,
        args.rtol,
        args.atol
    )

    # Determine output location
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("/localdisk/kurt/workspace/code/ai_coding/sycl_migration_agent/results/comparison_reports")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"comparison_{args.kernel}_{timestamp}.json"

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    # Generate human-readable summary
    summary_file = output_dir / f"comparison_{args.kernel}_{timestamp}_summary.txt"

    with open(summary_file, 'w') as f:
        f.write(f"=== CUDA vs SYCL Comparison Report ===\n")
        f.write(f"Kernel: {args.kernel}\n")
        f.write(f"Timestamp: {report['timestamp']}\n")
        f.write(f"Tolerances: rtol={args.rtol}, atol={args.atol}\n\n")

        if "summary" in report:
            summary = report["summary"]
            f.write(f"Overall Match: {summary['overall_match']}\n")
            f.write(f"Total Comparisons: {summary['total_comparisons']}\n")
            f.write(f"Matched: {summary['matched_comparisons']}\n")
            f.write(f"Failed: {summary['failed_comparisons']}\n\n")

        if "performance_comparison" in report:
            perf = report["performance_comparison"]
            f.write(f"Performance Comparison:\n")
            f.write(f"  CUDA Time: {perf['cuda_execution_time']:.4f}s\n")
            f.write(f"  SYCL Time: {perf['sycl_execution_time']:.4f}s\n")
            f.write(f"  Speedup: {perf['speedup_ratio']:.2f}x\n")
            f.write(f"  SYCL Performance: {perf['performance_percentage']:.1f}% of CUDA\n\n")

        # Show mismatches if any
        for filename, comparison in report.get("comparisons", {}).items():
            if not comparison.get("match", True):
                f.write(f"Mismatch in {filename}:\n")
                f.write(f"  Match Rate: {comparison.get('match_rate', 0)*100:.2f}%\n")
                f.write(f"  Mismatches: {comparison.get('mismatches', 'N/A')}/{comparison.get('total_elements', 'N/A')}\n")
                if 'max_absolute_diff' in comparison:
                    f.write(f"  Max Difference: {comparison['max_absolute_diff']:.2e}\n")
                if 'sample_mismatches' in comparison:
                    f.write(f"  Sample Mismatches:\n")
                    for i, sample in enumerate(comparison['sample_mismatches'][:3], 1):
                        f.write(f"    {i}. Index {sample['index']}: CUDA={sample['cuda']:.6f}, SYCL={sample['sycl']:.6f}, diff={sample['diff']:.2e}\n")
                f.write("\n")

    print(f"Comparison completed!")
    print(f"Report saved: {report_file}")
    print(f"Summary saved: {summary_file}")

    return 0 if report["summary"]["overall_match"] else 1

if __name__ == "__main__":
    sys.exit(main())