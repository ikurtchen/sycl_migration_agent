cmake_minimum_required(VERSION 3.18)
project(LC0_CUDAKernels LANGUAGES CUDA CXX VERSION 1.0.0)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Configuration-specific flags
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "RelWithDebInfo" "MinSizeRel")

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# CUDA architectures - based on meson.build analysis
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Support multiple architectures for compatibility including H20 at CC 9.0
    # Prioritize newer architectures
    set(CMAKE_CUDA_ARCHITECTURES 90 86 80 75 70)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/neural/backends/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# Find CUDA libraries
find_library(CUBLAS_LIB cublas PATHS
    /opt/cuda/lib64
    /usr/local/cuda/lib64
    /usr/lib/cuda/lib64
    DOC "cuBLAS library"
)

find_library(CUDNN_LIB cudnn PATHS
    /opt/cuda/lib64
    /usr/local/cuda/lib64
    /usr/lib/cuda/lib64
    DOC "cuDNN library"
)

# Check for CUTLASS support
option(USE_CUTLASS "Enable CUTLASS kernels" ON)
if(USE_CUTLASS)
    # Check if we can use CUTLASS (requires CUDA 11.0+ for compute capability 8.0)
    set(CUDA_MIN_COMPUTE_CAPABILITY 80)
    foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
        if(arch GREATER_EQUAL CUDA_MIN_COMPUTE_CAPABILITY)
            set(HAS_CUTLASS_SUPPORT TRUE)
            break()
        endif()
    endforeach()

    if(HAS_CUTLASS_SUPPORT)
        add_definitions(-DUSE_CUTLASS)
        message(STATUS "CUTLASS support enabled")
    else()
        message(STATUS "CUTLASS disabled - requires compute capability >= ${CUDA_MIN_COMPUTE_CAPABILITY}")
        set(USE_CUTLASS OFF)
    endif()
endif()

# Gather all CUDA sources
set(CUDA_SOURCES
    src/neural/backends/cuda/fp16_kernels.cu
    src/neural/backends/cuda/common_kernels.cu
)

# Add CUTLASS kernels if enabled
if(USE_CUTLASS)
    list(APPEND CUDA_SOURCES src/neural/backends/cuda/cutlass_kernels.cu)
endif()

# Extra CUDA source from ONNX backend (if exists)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/neural/backends/onnx/onnx_kernels.cu)
    list(APPEND CUDA_SOURCES src/neural/backends/onnx/onnx_kernels.cu)
endif()

# Additional C++ sources
set(CPP_SOURCES
    src/neural/backends/cuda/layers.cc
    src/neural/backends/cuda/network_cuda.cc
)

# Add cuDNN backend if library is found
if(CUDNN_LIB)
    list(APPEND CPP_SOURCES src/neural/backends/cuda/network_cudnn.cc)
    add_definitions(-DUSE_CUDNN)
    message(STATUS "cuDNN backend enabled")
else()
    message(STATUS "cuDNN backend disabled - library not found")
endif()

# Create the main CUDA kernels library
add_library(lc0_cuda_kernels
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)

# Set target properties
set_target_properties(lc0_cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(lc0_cuda_kernels PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/neural/backends/cuda>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>
    $<INSTALL_INTERFACE:include>
)

# Link CUDA libraries
target_link_libraries(lc0_cuda_kernels PUBLIC
    CUDA::cudart
    ${CUBLAS_LIB}
)

# Link cuDNN if available
if(CUDNN_LIB)
    target_link_libraries(lc0_cuda_kernels PUBLIC ${CUDNN_LIB})
endif()

# Platform-specific compilation flags
if(WIN32)
    target_compile_options(lc0_cuda_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/DNOMINMAX>
        $<$<COMPILE_LANGUAGE:CXX>:/D_CRT_SECURE_NO_WARNINGS>
    )
    target_compile_options(lc0_cuda_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/DNOMINMAX>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/D_CRT_SECURE_NO_WARNINGS>
    )
else()
    target_compile_options(lc0_cuda_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
    )
endif()

# Optimization flags for different build types
target_compile_options(lc0_cuda_kernels PRIVATE
    $<$<CONFIG:Debug>:
        $<$<COMPILE_LANGUAGE:CXX>:-g -O0>
        $<$<COMPILE_LANGUAGE:CUDA>:-g -G>
    >
    $<$<CONFIG:Release>:
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    >
    $<$<CONFIG:RelWithDebInfo>:
        $<$<COMPILE_LANGUAGE:CXX>:-O2 -g -DNDEBUG>
        $<$<COMPILE_LANGUAGE:CUDA>:-O2 -g>
    >
)

# Additional CUDA optimization flags
target_compile_options(lc0_cuda_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        $<$<CONFIG:Release>:--extra-device-vectorization>
        $<$<CONFIG:Release>:--ftz=true>
        $<$<CONFIG:Release>:--prec-div=false>
        $<$<CONFIG:Release>:--prec-sqrt=false>
        $<$<CONFIG:Release>:--fmad=true>
    >
)

# Google Test integration for testing
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    enable_testing()

    # Use system-installed Google Test
    find_package(GTest REQUIRED)

    # Add subdirectory for tests
    add_subdirectory(tests)
endif()

# Optional benchmark build
option(BUILD_BENCHMARKS "Build performance benchmarks" OFF)
if(BUILD_BENCHMARKS)
    # Use system-installed Google Benchmark
    find_package(benchmark)

    add_executable(lc0_cuda_benchmarks
        benchmarks/benchmark_kernels.cpp
    )

    target_link_libraries(lc0_cuda_benchmarks PRIVATE
        lc0_cuda_kernels
        benchmark::benchmark
    )
endif()

# Installation configuration
include(GNUInstallDirs)

install(TARGETS lc0_cuda_kernels
    EXPORT lc0_cuda_kernels-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(DIRECTORY src/neural/backends/cuda/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lc0_cuda
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)

# Export targets for use by other projects
install(EXPORT lc0_cuda_kernels-targets
    FILE lc0_cuda_kernels-config.cmake
    NAMESPACE lc0_cuda::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lc0_cuda_kernels
)

# Create config file for finding the package
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/lc0_cuda_kernels-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/lc0_cuda_kernels-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lc0_cuda_kernels
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/lc0_cuda_kernels-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/lc0_cuda_kernels-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/lc0_cuda_kernels-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lc0_cuda_kernels
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== LC0 CUDA Kernels Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUTLASS support: ${USE_CUTLASS}")
message(STATUS "cuDNN support: ${CUDNN_LIB}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "Build benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "=====================================")
message(STATUS "")