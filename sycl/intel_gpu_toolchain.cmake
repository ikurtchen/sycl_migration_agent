# CMake toolchain file for Intel GPU compilation with SYCL/DPC++
# Usage: cmake -DCMAKE_TOOLCHAIN_FILE=intel_gpu_toolchain.cmake ..

# Set the target system
set(CMAKE_SYSTEM_NAME Linux)

# Find Intel oneAPI DPC++ compiler
if(DEFINED ENV{ONEAPI_ROOT})
    set(ONEAPI_ROOT $ENV{ONEAPI_ROOT})
else()
    set(ONEAPI_ROOT /opt/intel/oneapi)
endif()

# Common paths for Intel oneAPI
set(ONEAPI_COMPILER_PATH "${ONEAPI_ROOT}/compiler/latest/linux/bin")
set(ONEAPI_DPCPP_PATH "${ONEAPI_ROOT}/dpcpp/latest/linux/bin")

# Find the DPC++ compiler
find_program(SYCL_COMPILER
    NAMES icpx
    PATHS ${ONEAPI_COMPILER_PATH}
          ${ONEAPI_DPCPP_PATH}
          /usr/local/bin
    DOC "Intel oneAPI DPC++/SYCL compiler"
)

if(NOT SYCL_COMPILER)
    message(FATAL_ERROR "Intel oneAPI DPC++ compiler not found. Please ensure oneAPI is installed.")
endif()

# Set the compilers
set(CMAKE_CXX_COMPILER ${SYCL_COMPILER})

# SYCL flags for Intel GPU
set(SYCL_FLAGS "-fsycl")

# Target specific Intel GPU architectures
# Available options:
#   - spir64_gen (generic Intel GPU)
#   - spir64_gen-unknown-linux (for Intel Data Center GPU Max series)
#   - spir64_gen-unknown-linux-sycldevice (for Intel Arc GPUs)
set(INTEL_GPU_TARGET "spir64_gen" CACHE STRING "Target Intel GPU architecture")

# Add target-specific flags
list(APPEND SYCL_FLAGS "-fsycl-targets=${INTEL_GPU_TARGET}")

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DSYCL_DEBUG")

# Apply SYCL flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SYCL_FLAGS}")

# Export variables
set(SYCL_COMPILER ${SYCL_COMPILER} CACHE PATH "SYCL compiler path")
set(INTEL_GPU_TARGET ${INTEL_GPU_TARGET} CACHE STRING "Intel GPU target")

message(STATUS "Using Intel oneAPI DPC++ compiler: ${SYCL_COMPILER}")
message(STATUS "Target GPU architecture: ${INTEL_GPU_TARGET}")