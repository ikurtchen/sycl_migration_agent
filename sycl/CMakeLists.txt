cmake_minimum_required(VERSION 3.20)
project(VectorAddSYCL CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# SYCL Compiler Detection
# =============================================================================

# Try to find Intel DPC++/SYCL compiler
find_program(ICPX_COMPILER
    NAMES icpx
    DOC "Intel DPC++/SYCL compiler"
)

# If no specific compiler found, check for standard compilers with SYCL support
if(NOT ICPX_COMPILER)
    # Check if current compiler supports SYCL
    if(CMAKE_CXX_COMPILER)
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -fsycl -x c++ - /dev/null 2>&1
            RESULT_VARIABLE SYCL_TEST_RESULT
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(SYCL_TEST_RESULT EQUAL 0)
            message(STATUS "Current compiler ${CMAKE_CXX_COMPILER} supports SYCL")
        else()
            message(WARNING "Current compiler may not support SYCL. Consider installing Intel oneAPI DPC++/C++ Compiler.")
        endif()
    endif()
else()
    # Set the compiler explicitly
    set(CMAKE_CXX_COMPILER ${ICPX_COMPILER})
    message(STATUS "Using Intel DPC++/SYCL compiler: ${ICPX_COMPILER}")
endif()

# Verify compiler capabilities
if(CMAKE_CXX_COMPILER)
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} --version
        OUTPUT_VARIABLE COMPILER_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    message(STATUS "Compiler version: ${COMPILER_VERSION}")
endif()

# =============================================================================
# SYCL Compilation Flags
# =============================================================================

# Basic SYCL flags
add_compile_options(-fsycl)

# Intel GPU target (can be overridden)
set(SYCL_TARGET "spir64_gen" CACHE STRING "SYCL target architecture")
add_compile_options(-fsycl-targets=${SYCL_TARGET})

# Optional device-specific optimizations (can be enabled via cmake -DDEVICE_ID=8800)
if(DEFINED DEVICE_ID)
    add_compile_options(-Xsycl-target-backend=${SYCL_TARGET} "-device=${DEVICE_ID}")
    message(STATUS "Using device-specific optimizations for device ID: ${DEVICE_ID}")
endif()

# Performance optimizations
add_compile_options(-O3)
add_compile_options(-ffast-math)
add_compile_options(-ftree-vectorize)

# Debug options (can be toggled with -DCMAKE_BUILD_TYPE=Debug)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -DSYCL_DEBUG)
else()
    add_compile_options(-DNDEBUG)
endif()

# Link options
add_link_options(-fsycl)
add_link_options(-fsycl-targets=${SYCL_TARGET})

# =============================================================================
# Build Configuration Options
# =============================================================================

# Option to enable CPU fallback (works with multiple backend targets)
option(ENABLE_CPU_FALLBACK "Enable CPU fallback when GPU is not available" ON)
if(ENABLE_CPU_FALLBACK)
    # Add both GPU and CPU targets for fallback capability
    if(SYCL_TARGET STREQUAL "spir64_gen")
        add_compile_options(-fsycl-targets=spir64_gen,spir64)
        add_link_options(-fsycl-targets=spir64_gen,spir64)
        message(STATUS "CPU fallback enabled with Intel GPU and CPU targets")
    endif()
    add_compile_definitions(ENABLE_CPU_FALLBACK=1)
endif()

# Option to enable performance profiling
option(ENABLE_PROFILING "Enable kernel performance profiling" OFF)
if(ENABLE_PROFILING)
    add_compile_definitions(ENABLE_PROFILING=1)
    add_compile_options(-fsycl-verbose)
endif()

# Intel GPU B60 specific optimizations option
option(ENABLE_B60_OPTIMIZATIONS "Enable Intel GPU B60 specific optimizations" OFF)
if(ENABLE_B60_OPTIMIZATIONS)
    add_compile_options(-Xsycl-target-backend=${SYCL_TARGET} "-device=8800")
    add_compile_definitions(WORK_GROUP_SIZE=256 VECTOR_SIZE=4)
    message(STATUS "Intel GPU B60 optimizations enabled")
endif()

# =============================================================================
# Source Files and Target Configuration
# =============================================================================

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Source files
set(SYCL_SOURCES
    src/vectorAdd.cpp
)

# Header files
set(SYCL_HEADERS
    src/vectorAdd.h
)

# Create the executable
add_executable(vectorAdd_sycl ${SYCL_SOURCES} ${SYCL_HEADERS})

# =============================================================================
# Target-Specific Configuration
# =============================================================================

# Set target properties
set_target_properties(vectorAdd_sycl PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Add include directories for the target
target_include_directories(vectorAdd_sycl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# =============================================================================
# Optional: Testing Support (if Google Test is available)
# =============================================================================

option(BUILD_TESTS "Build unit tests for SYCL kernels" OFF)
if(BUILD_TESTS)
    # Try to find Google Test
    find_package(GTest QUIET)

    if(GTest_FOUND)
        enable_testing()

        # Test executable
        add_executable(vectorAdd_sycl_tests
            tests/test_vectorAdd.cpp
            src/vectorAdd.cpp
        )

        target_link_libraries(vectorAdd_sycl_tests
            GTest::GTest
            GTest::Main
        )

        target_include_directories(vectorAdd_sycl_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${GTEST_INCLUDE_DIRS}
        )

        # Add test case
        add_test(NAME VectorAddSYCLTest COMMAND vectorAdd_sycl_tests)
    else()
        message(WARNING "Google Test not found. Tests will not be built.")
    endif()
endif()

# =============================================================================
# Installation Configuration
# =============================================================================

# Install the executable
install(TARGETS vectorAdd_sycl
    RUNTIME DESTINATION bin
    COMPONENT Runtime
)

# Install headers (if needed for library usage)
if(ENABLE_INSTALL_HEADERS)
    install(FILES ${SYCL_HEADERS}
        DESTINATION include
        COMPONENT Development
    )
endif()

# =============================================================================
# Print Configuration Summary
# =============================================================================

message(STATUS "")
message(STATUS "=== SYCL/DPC++ Configuration Summary ===")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "SYCL Target: ${SYCL_TARGET}")
if(DEFINED DEVICE_ID)
    message(STATUS "Device ID: ${DEVICE_ID}")
endif()
message(STATUS "CPU Fallback: ${ENABLE_CPU_FALLBACK}")
message(STATUS "Profiling: ${ENABLE_PROFILING}")
message(STATUS "B60 Optimizations: ${ENABLE_B60_OPTIMIZATIONS}")
message(STATUS "Tests: ${BUILD_TESTS}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "===========================================")

# =============================================================================
# Helper Targets
# =============================================================================

# Add a clean target for removing build artifacts
add_custom_target(clean-all
    COMMAND ${CMAKE_BUILD_TOOL} clean
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/cmake_install.cmake
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/Makefile
    COMMENT "Cleaning all build artifacts"
)

# Add a run target for easy execution
add_custom_target(run
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vectorAdd_sycl
    DEPENDS vectorAdd_sycl
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running vectorAdd_sycl executable"
)

# Add a benchmark target
add_custom_target(benchmark
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vectorAdd_sycl
    DEPENDS vectorAdd_sycl
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running vectorAdd_sycl benchmark"
)