cmake_minimum_required(VERSION 3.20)
project(VectorAddSYCLTests CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Try to find Intel DPC++/SYCL compiler
find_program(ICPX_COMPILER
    NAMES icpx
    DOC "Intel DPC++/SYCL compiler"
)

if(NOT ICPX_COMPILER)
    set(CMAKE_CXX_COMPILER icpx)
else()
    set(CMAKE_CXX_COMPILER ${ICPX_COMPILER})
endif()

message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")

# Basic SYCL flags
add_compile_options(-fsycl)

# Intel GPU target (use spir64 without specific device to avoid ocloc errors)
set(SYCL_TARGET "spir64")
add_compile_options(-fsycl-targets=${SYCL_TARGET})

# CPU fallback for compatibility
add_compile_options(-fsycl-targets=${SYCL_TARGET},spir64)

# Performance optimizations
add_compile_options(-O3)
add_compile_options(-ffast-math)

# Link options
add_link_options(-fsycl)
add_link_options(-fsycl-targets=${SYCL_TARGET},spir64)

# Find Google Test
find_package(GTest REQUIRED)

if(GTest_FOUND)
    message(STATUS "Google Test found: ${GTEST_VERSION}")
    enable_testing()

    # Source files
    set(TEST_SOURCES
        test_vectorAdd_sycl.cpp
        vectorAdd_kernel_sycl.cpp
    )

    # Test executable
    add_executable(vectorAdd_sycl_test ${TEST_SOURCES})

    # Link libraries
    target_link_libraries(vectorAdd_sycl_test
        GTest::GTest
        GTest::Main
    )

    # Add test
    add_test(NAME VectorAddSYCLTest COMMAND vectorAdd_sycl_test)

    message(STATUS "Test executable: vectorAdd_sycl_test")
else()
    message(FATAL_ERROR "Google Test not found. Please install libgtest-dev.")
endif()

message(STATUS "=== SYCL Test Configuration ===")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "SYCL Target: ${SYCL_TARGET}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "=================================")