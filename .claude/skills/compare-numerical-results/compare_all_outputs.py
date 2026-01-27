#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, '.claude/skills/compare-numerical-results')

def compare_binary_files(cuda_file, sycl_file, dtype='float32', rtol=1e-5, atol=1e-8):
    import numpy as np

    # Read CUDA output
    cuda_data = np.fromfile(cuda_file, dtype=np.dtype(dtype))
    # Read SYCL output
    sycl_data = np.fromfile(sycl_file, dtype=np.dtype(dtype))

    if cuda_data.shape != sycl_data.shape:
        return {
            'match': False,
            'shape_mismatch': True,
            'cuda_shape': cuda_data.shape,
            'sycl_shape': sycl_data.shape
        }

    # Calculate differences
    abs_diff = np.abs(cuda_data - sycl_data)
    rel_diff = np.divide(abs_diff,
                         np.maximum(np.abs(cuda_data), np.abs(sycl_data)),
                         out=np.full_like(abs_diff, np.inf),
                         where=(cuda_data != 0) | (sycl_data != 0))

    matches = np.isclose(cuda_data, sycl_data, rtol=rtol, atol=atol)
    match_count = np.sum(matches)
    total_count = len(cuda_data)

    return {
        'match': match_count == total_count,
        'shape_mismatch': False,
        'total': total_count,
        'matches': match_count,
        'match_rate': (match_count / total_count * 100) if total_count > 0 else 0,
        'max_abs_diff': float(np.max(abs_diff)) if len(abs_diff) > 0 else 0,
        'mean_abs_diff': float(np.mean(abs_diff)) if len(abs_diff) > 0 else 0,
        'max_rel_diff': float(np.max(rel_diff)) if len(rel_diff) > 0 else 0,
        'mean_rel_diff': float(np.mean(rel_diff)) if len(rel_diff) > 0 else 0,
    }

def main():
    cuda_output_dir = 'results/cuda_outputs/cuda_outputs'
    sycl_output_dir = 'results/sycl_outputs/sycl_outputs'

    # Get all output files
    cuda_files = sorted([f for f in os.listdir(cuda_output_dir) if f.endswith('.bin')])

    print("=" * 80)
    print("CUDA vs SYCL Numerical Comparison - All Outputs")
    print("=" * 80)

    results = []
    all_passed = True

    for f in cuda_files:
        cuda_path = os.path.join(cuda_output_dir, f)
        sycl_path = os.path.join(sycl_output_dir, f)

        if not os.path.exists(sycl_path):
            print(f"\n[SKIP] {f} - SYCL output not found")
            all_passed = False
            continue

        result = compare_binary_files(cuda_path, sycl_path)
        results.append((f, result))

        status = "✓" if result['match'] else "✗"
        print(f"\n{status} {f}")
        print(f"    Elements: {result['total']:,} | Matches: {result['matches']:,} ({result['match_rate']:.1f}%)")
        if not result['match'] and not result['shape_mismatch']:
            print(f"    Max Abs Diff: {result['max_abs_diff']:.2e} | Max Rel Diff: {result['max_rel_diff']:.2e}")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    total_tests = len(results)
    passed_tests = sum(1 for _, r in results if r['match'])

    print(f"\nTotal tests compared: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")

    if passed_tests == total_tests:
        print("\n✓✓✓ ALL TESTS PASSED - Perfect Numerical Match!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())