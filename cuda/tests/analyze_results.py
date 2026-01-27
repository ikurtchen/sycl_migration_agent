#!/usr/bin/env python3
"""
CUDA VectorAdd Test Results Analyzer
This script analyzes test results, benchmarks, and generates comparison reports
"""

import json
import os
import sys
import struct
import numpy as np
from datetime import datetime
import argparse

class TestAnalyzer:
    def __init__(self, test_dir):
        self.test_dir = test_dir
        self.cuda_outputs_dir = os.path.join(test_dir, 'cuda_outputs')
        self.cuda_inputs_dir = os.path.join(test_dir, 'cuda_inputs')
        self.test_logs_dir = os.path.join(test_dir, 'test_logs')

    def load_binary_file(self, filepath):
        """Load binary data from file"""
        try:
            with open(filepath, 'rb') as f:
                # Read all data as float32
                data = []
                while True:
                    chunk = f.read(4)  # 4 bytes per float
                    if not chunk:
                        break
                    try:
                        value = struct.unpack('<f', chunk)[0]
                        data.append(value)
                    except:
                        break
                return np.array(data, dtype=np.float32)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def analyze_benchmarks(self):
        """Analyze benchmark results from JSON files"""
        benchmarks = {}

        for filename in os.listdir(self.cuda_outputs_dir):
            if filename.startswith('vectorAdd_benchmark_') and filename.endswith('.json'):
                filepath = os.path.join(self.cuda_outputs_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        size = data.get('size')
                        benchmarks[size] = {
                            'time_ms': data.get('time_ms'),
                            'gflops': data.get('gflops'),
                            'kernel': data.get('kernel')
                        }
                except Exception as e:
                    print(f"Error parsing benchmark file {filename}: {e}")

        return benchmarks

    def compute_statistics(self, data):
        """Compute basic statistics for data array"""
        if data is None or len(data) == 0:
            return {}

        return {
            'count': len(data),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'sum': float(np.sum(data))
        }

    def analyze_output_files(self):
        """Analyze all output files and compute statistics"""
        results = {}

        for filename in os.listdir(self.cuda_outputs_dir):
            if filename.endswith('_output.bin'):
                test_name = filename.replace('vectorAdd_', '').replace('_output.bin', '')
                filepath = os.path.join(self.cuda_outputs_dir, filename)

                data = self.load_binary_file(filepath)
                if data is not None:
                    results[test_name] = self.compute_statistics(data)

        return results

    def generate_report(self):
        """Generate a comprehensive test report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_directory': self.test_dir,
            'benchmarks': self.analyze_benchmarks(),
            'output_analysis': self.analyze_output_files()
        }

        return report

    def save_report(self, output_file):
        """Save analysis report to JSON file"""
        report = self.generate_report()

        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to: {output_file}")
        except Exception as e:
            print(f"Error saving report: {e}")

    def print_summary(self):
        """Print a summary of the test results"""
        print("\n" + "="*60)
        print("CUDA VECTORADD TEST ANALYSIS SUMMARY")
        print("="*60)

        # Benchmark summary
        benchmarks = self.analyze_benchmarks()
        if benchmarks:
            print("\nBENCHMARK RESULTS:")
            print("-" * 40)
            for size, data in sorted(benchmarks.items()):
                print(f"Size: {size:>10} | Time: {data['time_ms']:>8.3f} ms | GFLOPS: {data['gflops']:>8.2f}")

        # Output analysis
        outputs = self.analyze_output_files()
        if outputs:
            print("\nOUTPUT STATISTICS:")
            print("-" * 40)
            for test_name, stats in outputs.items():
                print(f"\nTest: {test_name}")
                print(f"  Count: {stats['count']}")
                print(f"  Mean:  {stats['mean']:.6f}")
                print(f"  Std:   {stats['std']:.6f}")
                print(f"  Min:   {stats['min']:.6f}")
                print(f"  Max:   {stats['max']:.6f}")

        print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Analyze CUDA VectorAdd test results')
    parser.add_argument('--test-dir',
                       default='/localdisk/kurt/workspace/code/ai_coding/sycl_migration_agent/cuda/tests',
                       help='Test directory path')
    parser.add_argument('--output', '-o',
                       help='Output JSON report file path')
    parser.add_argument('--summary', '-s', action='store_true',
                       help='Print summary to console')

    args = parser.parse_args()

    # Create analyzer
    analyzer = TestAnalyzer(args.test_dir)

    # Print summary if requested
    if args.summary:
        analyzer.print_summary()

    # Save report if output specified
    if args.output:
        analyzer.save_report(args.output)
    else:
        # Default output file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_output = f"{args.test_dir}/test_results_analysis_{timestamp}.json"
        analyzer.save_report(default_output)

if __name__ == '__main__':
    main()