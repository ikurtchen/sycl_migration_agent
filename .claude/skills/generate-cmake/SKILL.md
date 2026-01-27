---
name: generate-cmake
description: "generate CMake build system files for CUDA and SYCL projects"
---

# generate-cmake

Generates CMakeLists.txt files for CUDA and SYCL projects with proper configuration.

## Description

This skill creates production-ready CMake build configurations for GPU projects, handling compiler detection, dependency management, test framework integration, and cross-platform compatibility.

## Usage

```bash
python generate_cmake.py <project_type> <source_dir> [options]
```

### Arguments

- `project_type`: Either "cuda" or "sycl"
- `source_dir`: Directory containing source files
- `--output`: Output path for CMakeLists.txt (default: source_dir/CMakeLists.txt)
- `--project-name`: Project name (default: derived from directory)
- `--cuda-arch`: CUDA architecture (e.g., "sm_80,sm_86")
- `--sycl-backend`: SYCL backend ("level_zero", "opencl", "cuda")
- `--enable-tests`: Include Google Test framework
- `--install-prefix`: Installation directory

### Examples

```bash
# Generate CUDA build configuration
python generate_cmake.py cuda ./cuda-kernels --enable-tests

# Generate SYCL build configuration with specific backend
python generate_cmake.py sycl ./sycl-kernels --sycl-backend level_zero --enable-tests

# Custom CUDA architectures
python generate_cmake.py cuda ./cuda-kernels --cuda-arch "sm_80,sm_86,sm_90"
```

## Generated CUDA CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.18)
project(CUDAKernels LANGUAGES CUDA CXX)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# CUDA architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
endif()

# Kernel library
file(GLOB_RECURSE CUDA_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh"
)

add_library(cuda_kernels ${CUDA_SOURCES})

target_include_directories(cuda_kernels PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cuda_kernels PUBLIC
    CUDA::cudart
)

# Optimization flags
target_compile_options(cuda_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --extra-device-vectorization
        -lineinfo
    >
)

# Google Test integration (if enabled)
if(BUILD_TESTING)
    enable_testing()
    include(FetchContent)

    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/release-1.12.1.zip
    )
    FetchContent_MakeAvailable(googletest)

    # Test sources
    file(GLOB_RECURSE TEST_SOURCES 
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu"
    )

    add_executable(cuda_tests ${TEST_SOURCES})

    target_link_libraries(cuda_tests PRIVATE
        cuda_kernels
        GTest::gtest_main
        GTest::gmock
    )

    include(GoogleTest)
    gtest_discover_tests(cuda_tests)
endif()

# Installation
install(TARGETS cuda_kernels
    EXPORT cuda_kernels-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include
)

# Export targets
install(EXPORT cuda_kernels-targets
    FILE cuda_kernels-config.cmake
    NAMESPACE cuda_kernels::
    DESTINATION lib/cmake/cuda_kernels
)
```

## Generated SYCL CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(SYCLKernels LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect SYCL compiler
if(NOT DEFINED CMAKE_CXX_COMPILER)
    # Try Intel DPC++ first
    find_program(ICPX_COMPILER icpx)
    if(ICPX_COMPILER)
        set(CMAKE_CXX_COMPILER ${ICPX_COMPILER})
    else()
        # Fall back to clang++ with SYCL support
        find_program(CLANGXX_COMPILER clang++)
        if(CLANGXX_COMPILER)
            set(CMAKE_CXX_COMPILER ${CLANGXX_COMPILER})
        endif()
    endif()
endif()

# SYCL compilation flags
set(SYCL_FLAGS "-fsycl")

# Backend-specific flags
if(SYCL_BACKEND STREQUAL "level_zero")
    list(APPEND SYCL_FLAGS "-fsycl-targets=spir64_gen")
elseif(SYCL_BACKEND STREQUAL "cuda")
    list(APPEND SYCL_FLAGS "-fsycl-targets=nvptx64-nvidia-cuda")
elseif(SYCL_BACKEND STREQUAL "opencl")
    list(APPEND SYCL_FLAGS "-fsycl-targets=spir64")
endif()

# Apply SYCL flags globally
add_compile_options(${SYCL_FLAGS})
add_link_options(${SYCL_FLAGS})

# Kernel library
file(GLOB_RECURSE SYCL_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
)

add_library(sycl_kernels ${SYCL_SOURCES})

target_include_directories(sycl_kernels PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Intel-specific optimizations
target_compile_options(sycl_kernels PRIVATE
    -O3
    -fno-finite-math-only  # Preserve NaN/Inf handling
)

# Google Test integration (if enabled)
if(BUILD_TESTING)
    enable_testing()
    include(FetchContent)

    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/release-1.12.1.zip
    )
    FetchContent_MakeAvailable(googletest)

    # Test sources
    file(GLOB_RECURSE TEST_SOURCES 
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
    )

    add_executable(sycl_tests ${TEST_SOURCES})

    target_link_libraries(sycl_tests PRIVATE
        sycl_kernels
        GTest::gtest_main
        GTest::gmock
    )

    include(GoogleTest)
    gtest_discover_tests(sycl_tests)
endif()

# Installation
install(TARGETS sycl_kernels
    EXPORT sycl_kernels-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include
)

# Export targets
install(EXPORT sycl_kernels-targets
    FILE sycl_kernels-config.cmake
    NAMESPACE sycl_kernels::
    DESTINATION lib/cmake/sycl_kernels
)
```

## Configuration Options

### CUDA-Specific Options

```cmake
# Architecture selection
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;90" CACHE STRING "CUDA architectures")

# Optimization levels
set(CUDA_NVCC_FLAGS "-O3 --use_fast_math" CACHE STRING "NVCC optimization flags")

# Debug information
option(CUDA_DEBUG_INFO "Include debug info in CUDA kernels" OFF)

# Separable compilation (for device linking)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
```

### SYCL-Specific Options

```cmake
# Backend selection
set(SYCL_BACKEND "level_zero" CACHE STRING "SYCL backend (level_zero/opencl/cuda)")

# Device architectures (Intel GPU)
set(SYCL_DEVICE_ARCH "pvc" CACHE STRING "Intel GPU architecture (pvc/acm-g10)")

# Ahead-of-time compilation
option(SYCL_AOT_COMPILE "Enable AOT compilation for target device" ON)

# Optimization flags
set(SYCL_OPT_FLAGS "-O3 -fp-model=precise" CACHE STRING "SYCL optimization flags")
```

## Advanced Features

### Multi-Configuration Support

```cmake
# Support both Debug and Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS 
    "Debug" "Release" "RelWithDebInfo" "MinSizeRel")

# Configuration-specific flags
target_compile_options(sycl_kernels PRIVATE
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
)
```

### Dependency Management

```cmake
# External libraries
find_package(oneMKL CONFIG)
if(oneMKL_FOUND)
    target_link_libraries(sycl_kernels PUBLIC MKL::MKL_DPCPP)
    target_compile_definitions(sycl_kernels PUBLIC USE_ONEMKL)
endif()

# Conditional features
option(USE_DOUBLE_PRECISION "Enable double precision" ON)
if(USE_DOUBLE_PRECISION)
    target_compile_definitions(sycl_kernels PUBLIC USE_DOUBLE)
endif()
```

### Cross-Platform Compatibility

```cmake
# Platform detection
if(WIN32)
    target_compile_definitions(sycl_kernels PRIVATE PLATFORM_WINDOWS)
elseif(UNIX AND NOT APPLE)
    target_compile_definitions(sycl_kernels PRIVATE PLATFORM_LINUX)
elseif(APPLE)
    target_compile_definitions(sycl_kernels PRIVATE PLATFORM_MACOS)
endif()

# Compiler-specific settings
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    # Intel DPC++ specific
    target_compile_options(sycl_kernels PRIVATE -qopenmp)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Clang with SYCL support
    target_compile_options(sycl_kernels PRIVATE -fopenmp)
endif()
```

## Integration with Testing

### Google Test Setup

```cmake
# Fetch and configure Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)

# Prevent Google Test from overriding our compiler/linker options
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable CTest
enable_testing()
include(GoogleTest)

# Auto-discover tests
gtest_discover_tests(sycl_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES
        ENVIRONMENT "ONEAPI_DEVICE_SELECTOR=level_zero:gpu"
)
```

### Benchmark Integration

```cmake
# Google Benchmark (optional)
option(BUILD_BENCHMARKS "Build benchmark suite" OFF)

if(BUILD_BENCHMARKS)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.0
    )
    FetchContent_MakeAvailable(benchmark)

    add_executable(kernel_benchmarks
        benchmarks/bench_matmul.cpp
        benchmarks/bench_vector.cpp
    )

    target_link_libraries(kernel_benchmarks PRIVATE
        sycl_kernels
        benchmark::benchmark
    )
endif()
```

## Output Directory Structure

```
project/
├── CMakeLists.txt                 # Generated root CMake
├── src/
│   ├── kernels/
│   └── CMakeLists.txt             # Optional subdirectory CMake
├── include/
│   └── kernels/
├── tests/
│   ├── CMakeLists.txt
│   └── *.cpp
├── build/                         # Build directory
│   ├── cuda_kernels              # Built library
│   ├── sycl_kernels
│   └── *_tests                   # Test executables
└── install/                       # Install prefix
    ├── lib/
    ├── include/
    └── bin/
```

## Validation

The skill validates:
- Source files exist
- Compiler is available
- Required dependencies are present
- CMake version compatibility
- Platform support

## Error Handling

```python
def generate_cmake(project_type, source_dir, options):
    # Validate inputs
    if not os.path.exists(source_dir):
        raise ValueError(f"Source directory not found: {source_dir}")

    if project_type not in ["cuda", "sycl"]:
        raise ValueError("project_type must be 'cuda' or 'sycl'")

    # Check for source files
    sources = find_sources(source_dir, project_type)
    if not sources:
        raise Warning(f"No {project_type} source files found in {source_dir}")

    # Generate CMakeLists.txt
    cmake_content = render_template(project_type, sources, options)

    # Write to file
    output_path = options.get("output", 
                             os.path.join(source_dir, "CMakeLists.txt"))

    with open(output_path, 'w') as f:
        f.write(cmake_content)

    return {"status": "success", "output": output_path}
```

## Related Skills

- `scan-cuda-repo`: Provides source file inventory
- `create-cuda-tests`: Generates test files referenced in CMake
- `translate-cuda-to-sycl`: Creates SYCL sources to be built

## Skill Metadata

```yaml
name: generate-cmake
version: 1.0.0
category: build
dependencies:
  - python >= 3.8
  - cmake >= 3.18 (CUDA) or >= 3.20 (SYCL)
inputs:
  - project_type: string (cuda|sycl)
  - source_dir: string
  - options: dict (optional)
outputs:
  - cmake_file: file path
  - status: success|error
execution_time: < 1 second
```
