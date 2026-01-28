cmake_minimum_required(VERSION 3.20)
project(SYCLKernels CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find SYCL/DPC++ compiler
find_package(DPCPP QUIET)

if(DPCPP_FOUND)
  set(SYCL_COMPILER ${DPCPP_COMPILER})
elseif(DEFINED ENV{SYCL_COMPILER})
  set(SYCL_COMPILER $ENV{SYCL_COMPILER})
else()
  # Try to find Intel oneAPI DPC++ compiler
  find_program(SYCL_COMPILER
    NAMES icpx icx dpc++
    PATHS /opt/intel/oneapi/compiler/latest/linux/bin/
          /usr/local/bin/
          /opt/intel/oneapi/compiler/latest/linux/bin-llvm/
  )
endif()

if(NOT SYCL_COMPILER)
  message(FATAL_ERROR "SYCL compiler not found. Please install Intel oneAPI DPC++ or set SYCL_COMPILER environment variable.")
endif()

message(STATUS "Using SYCL compiler: ${SYCL_COMPILER}")

# Set compiler
set(CMAKE_CXX_COMPILER ${SYCL_COMPILER})

# SYCL compilation flags
add_compile_options(-fsycl)
add_link_options(-fsycl)

# Architecture-specific flags for Intel GPUs
option(SYCL_ARCH_INTEL_GPU "Enable Intel GPU compilation" ON)
if(SYCL_ARCH_INTEL_GPU)
  add_compile_options(-fsycl-targets=spir64_gen)
endif()

# Debug options
option(SYCL_DEBUG "Enable SYCL debug mode" OFF)
if(SYCL_DEBUG)
  add_compile_options(-g -O0 -DSYCL_DEBUG)
  add_definitions(-DSYCL_DEBUG)
else()
  add_compile_options(-O3)
endif()

# Include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neural
)

# Source files for kernel library
set(KERNEL_SOURCES
  src/neural/backends/sycl/common_kernels.cpp
  src/neural/backends/sycl/fp16_kernels.cpp
  src/neural/backends/sycl/fused_attention_sycl.cpp
)

# Header files
set(KERNEL_HEADERS
  src/neural/backends/sycl/kernels.h
  src/neural/backends/sycl/fp16_kernels.h
  src/neural/backends/sycl/sycl_common.h
  src/neural/tables/activation_function.h
)

# Create kernel library
add_library(sycl_kernels STATIC ${KERNEL_SOURCES} ${KERNEL_HEADERS})

# Set target properties
set_target_properties(sycl_kernels PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Link SYCL library
target_link_libraries(sycl_kernels sycl)

# Use system-installed Google Test for testing
find_package(GTest REQUIRED)

enable_testing()

# Test executable for addVectors
add_executable(test_addVectors
  test_addVectors.cpp
  ${KERNEL_SOURCES}
)

target_link_libraries(test_addVectors
  GTest::gtest_main
  sycl
)

set_target_properties(test_addVectors PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Test executable for policy map
add_executable(test_policy_map
  tests/test_policy_map.cpp
  ${KERNEL_SOURCES}
)

target_link_libraries(test_policy_map
  GTest::gtest_main
  sycl
)

set_target_properties(test_policy_map PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Test executable for SE Layer NHWC
add_executable(test_se_layer_nhwc
  tests/test_se_layer_nhwc.cpp
  ${KERNEL_SOURCES}
)

target_link_libraries(test_se_layer_nhwc
  GTest::gtest_main
  sycl
)

set_target_properties(test_se_layer_nhwc PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Check if there are additional test files in the parent tests directory
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_globalAvgPool.cpp")
  add_executable(test_globalAvgPool
    ../tests/test_globalAvgPool.cpp
    ${KERNEL_SOURCES}
  )

  target_link_libraries(test_globalAvgPool
    GTest::gtest_main
    sycl
  )

  set_target_properties(test_globalAvgPool PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
  )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_softmax_sycl.cpp")
  add_executable(test_softmax
    ../tests/test_softmax_sycl.cpp
    ${KERNEL_SOURCES}
  )

  target_link_libraries(test_softmax
    GTest::gtest_main
    sycl
  )

  set_target_properties(test_softmax PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
  )
endif()

# Register tests with Google Test
gtest_discover_tests(test_addVectors)
gtest_discover_tests(test_policy_map)
gtest_discover_tests(test_se_layer_nhwc)

if(TARGET test_globalAvgPool)
  gtest_discover_tests(test_globalAvgPool)
endif()

if(TARGET test_softmax)
  gtest_discover_tests(test_softmax)
endif()

# Test executable for expandPlanes
# Commented out - test file does not exist
# add_executable(test_expandPlanes
#   tests/test_expandPlanes.cpp
#   ${KERNEL_SOURCES}
# )

# target_link_libraries(test_expandPlanes
#   sycl
# )

# set_target_properties(test_expandPlanes PROPERTIES
#   CXX_STANDARD 17
#   CXX_STANDARD_REQUIRED ON
# )

# Register expandPlanes test with Google Test
# gtest_discover_tests(test_expandPlanes)

# Test executable for preprocess attention
add_executable(test_preprocess_attention
  tests/test_preprocess_attention.cpp
  ${KERNEL_SOURCES}
)

target_link_libraries(test_preprocess_attention
  sycl
)

set_target_properties(test_preprocess_attention PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)



# Custom targets for running specific tests
add_custom_target(run_addVectors
  COMMAND test_addVectors
  DEPENDS test_addVectors
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running addVectors tests"
)

add_custom_target(run_policy_map
  COMMAND test_policy_map
  DEPENDS test_policy_map
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running policy map tests"
)

add_custom_target(run_se_layer_nhwc
  COMMAND test_se_layer_nhwc
  DEPENDS test_se_layer_nhwc
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running SE Layer NHWC tests"
)

if(TARGET test_globalAvgPool)
  add_custom_target(run_globalAvgPool
    COMMAND test_globalAvgPool
    DEPENDS test_globalAvgPool
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running global average pool tests"
  )
endif()

if(TARGET test_softmax)
  add_custom_target(run_softmax
    COMMAND test_softmax
    DEPENDS test_softmax
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running softmax tests"
  )
endif()

add_custom_target(run_expandPlanes
  COMMAND test_expandPlanes
  DEPENDS test_expandPlanes
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running expandPlanes tests"
)

# Custom target to run all tests
add_custom_target(run_all_tests
  COMMAND ${CMAKE_CTEST_COMMAND} --verbose
  DEPENDS test_addVectors test_policy_map test_se_layer_nhwc test_expandPlanes
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running all SYCL kernel tests"
)

# Installation rules
install(TARGETS sycl_kernels
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(FILES ${KERNEL_HEADERS}
  DESTINATION include/sycl/neural/backends/sycl
)

# Integration example for MHA
add_executable(mha_integration_example
  src/neural/backends/sycl/mha_integration_example.cpp
  ${KERNEL_SOURCES}
)

target_link_libraries(mha_integration_example
  sycl
)

set_target_properties(mha_integration_example PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== SYCL Kernel Build Configuration ===")
message(STATUS "SYCL Compiler: ${SYCL_COMPILER}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Intel GPU Support: ${SYCL_ARCH_INTEL_GPU}")
message(STATUS "Debug Mode: ${SYCL_DEBUG}")
message(STATUS "==========================================")
message(STATUS "")