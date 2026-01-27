#!/usr/bin/env python3
"""
Skill: scan_cuda_repo
Recursively scans a repository for CUDA files and extracts kernel information.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any

def scan_cuda_repo(repo_path: str) -> Dict[str, Any]:
    """
    Scan repository for CUDA files and extract kernel information.

    Args:
        repo_path: Path to the repository root

    Returns:
        Dictionary containing scan results with kernel inventory
    """
    cuda_extensions = {'.cu', '.cuh', '.cuda'}
    cuda_files = []
    kernels = []

    # Recursively find CUDA files
    for root, dirs, files in os.walk(repo_path):
        # Skip common non-source directories
        dirs[:] = [d for d in dirs if d not in {'.git', 'build', 'bin', 'lib', '__pycache__'}]

        for file in files:
            if Path(file).suffix in cuda_extensions:
                file_path = os.path.join(root, file)
                cuda_files.append(file_path)

                # Extract kernels from this file
                kernels.extend(extract_kernels_from_file(file_path))

    # Analyze complexity
    for kernel in kernels:
        kernel['complexity'] = assess_complexity(kernel)

    # Detect external libraries
    external_libs = detect_external_libraries(cuda_files)

    # Identify migration challenges
    challenges = identify_challenges(kernels)

    return {
        "repository": repo_path,
        "total_cuda_files": len(cuda_files),
        "cuda_files": cuda_files,
        "total_kernels": len(kernels),
        "kernels": kernels,
        "external_libraries": external_libs,
        "migration_challenges": challenges
    }

def extract_kernels_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Extract kernel functions from a CUDA file."""
    kernels = []

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return kernels

    # Regex patterns
    kernel_pattern = re.compile(
        r'__global__\s+(?:__launch_bounds__\([^)]+\)\s+)?'
        r'(?:void|[\w<>:,\s\*&]+)\s+'
        r'(\w+)\s*\([^)]*\)',
        re.MULTILINE
    )

    for match in kernel_pattern.finditer(content):
        kernel_name = match.group(1)

        # Find line number
        line_num = content[:match.start()].count('\n') + 1

        # Extract full signature (rough approximation)
        signature_start = match.start()
        signature_end = content.find('{', signature_start)
        if signature_end == -1:
            signature_end = match.end()
        signature = content[signature_start:signature_end].strip()

        # Analyze kernel features
        features = analyze_kernel_features(content, match.start(), signature_end)

        kernels.append({
            "name": kernel_name,
            "file": file_path,
            "line": line_num,
            "signature": signature,
            "features": features,
            "dependencies": []
        })

    return kernels

def analyze_kernel_features(content: str, start: int, end: int) -> List[str]:
    """Analyze which CUDA features a kernel uses."""
    # Extract kernel body (approximate)
    brace_count = 0
    kernel_start = content.find('{', start)
    if kernel_start == -1:
        return []

    kernel_end = kernel_start
    for i in range(kernel_start, len(content)):
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                kernel_end = i
                break

    kernel_body = content[kernel_start:kernel_end]

    features = []

    # Detect features
    if '__shared__' in kernel_body:
        features.append('shared_memory')
    if '__syncthreads()' in kernel_body:
        features.append('syncthreads')
    if '__syncwarp()' in kernel_body:
        features.append('syncwarp')
    if '__shfl' in kernel_body:
        features.append('warp_shuffle')
    if 'cooperative_groups' in kernel_body:
        features.append('cooperative_groups')
    if 'atomicAdd' in kernel_body or 'atomicCAS' in kernel_body:
        features.append('atomics')
    if 'texture<' in kernel_body or 'cudaTextureObject_t' in kernel_body:
        features.append('texture_memory')
    if '__constant__' in content[:start]:
        features.append('constant_memory')
    if 'asm(' in kernel_body or '__asm__' in kernel_body:
        features.append('inline_ptx')

    return features

def assess_complexity(kernel: Dict[str, Any]) -> str:
    """Assess kernel migration complexity based on features."""
    features = kernel.get('features', [])

    # Complex features that require careful translation
    complex_features = {'inline_ptx', 'cooperative_groups', 'dynamic_parallelism',
                       'texture_memory', 'surface_memory'}

    if any(f in complex_features for f in features):
        return 'complex'
    elif len(features) > 3:
        return 'moderate'
    else:
        return 'simple'

def detect_external_libraries(cuda_files: List[str]) -> List[str]:
    """Detect which external CUDA libraries are used."""
    libraries = set()

    lib_patterns = {
        'cuBLAS': re.compile(r'#include\s+[<"]cublas'),
        'cuFFT': re.compile(r'#include\s+[<"]cufft'),
        'cuRAND': re.compile(r'#include\s+[<"]curand'),
        'cuSPARSE': re.compile(r'#include\s+[<"]cusparse'),
        'cuDNN': re.compile(r'#include\s+[<"]cudnn'),
        'Thrust': re.compile(r'#include\s+[<"]thrust/'),
        'CUB': re.compile(r'#include\s+[<"]cub/'),
    }

    for file_path in cuda_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                for lib_name, pattern in lib_patterns.items():
                    if pattern.search(content):
                        libraries.add(lib_name)
        except Exception:
            continue

    return sorted(list(libraries))

def identify_challenges(kernels: List[Dict[str, Any]]) -> List[str]:
    """Identify potential migration challenges."""
    challenges = []

    # Count specific features
    inline_ptx_count = sum(1 for k in kernels if 'inline_ptx' in k.get('features', []))
    coop_groups_count = sum(1 for k in kernels if 'cooperative_groups' in k.get('features', []))
    texture_count = sum(1 for k in kernels if 'texture_memory' in k.get('features', []))

    if inline_ptx_count > 0:
        challenges.append(f"{inline_ptx_count} kernel(s) use inline PTX - requires manual review")

    if coop_groups_count > 0:
        challenges.append(f"{coop_groups_count} kernel(s) use cooperative groups - careful SYCL translation needed")

    if texture_count > 0:
        challenges.append(f"{texture_count} kernel(s) use texture memory - map to SYCL images or samplers")

    return challenges

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Scan code repo for CUDA kernels")
    parser.add_argument("repo_path", help="Code repo path")
    parser.add_argument("--output", help="Ouput JSON file path")

    args = parser.parse_args()
    results = scan_cuda_repo(args.repo_path)

    if args.output:
        with open(args.output, 'w') as out_file:
            json.dump(results, out_file, indent=2)
    else:
        print(json.dumps(results, indent=2))
