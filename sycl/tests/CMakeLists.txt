# CMakeLists.txt for SYCL PolicyMap kernel test
cmake_minimum_required(VERSION 3.18)
project(PolicyMapTest)

# Find SYCL implementation (Intel oneAPI DPC++/LLVM)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(SYCL sycl)
endif()

# Try to find oneAPI DPC++ as a fallback
if(NOT SYCL_FOUND)
    find_program(DPCPP_EXECUTABLE
        NAMES clang++ icx
        DOC "DPC++/LLVM compiler for SYCL"
    )

    if(DPCPP_EXECUTABLE)
        set(SYCL_FOUND TRUE)
        set(SYCL_COMPILER "${DPCPP_EXECUTABLE}")
        set(SYCL_CXX_FLAGS "-fsycl")
        message(STATUS "Found DPC++ compiler: ${DPCPP_EXECUTABLE}")
    else()
        message(FATAL_ERROR "SYCL compiler not found. Please install Intel oneAPI DPC++ or another SYCL implementation.")
    endif()
endif()

# Create test executable
add_executable(test_policy_map
    test_policy_map.cpp
    ../src/neural/backends/sycl/common_kernels.cpp
)

# Set include directories
target_include_directories(test_policy_map PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

# Set compiler flags for SYCL
if(SYCL_CXX_FLAGS)
    target_compile_options(test_policy_map PRIVATE ${SYCL_CXX_FLAGS})
endif()

# Link with SYCL if needed
if(SYCL_LIBRARIES)
    target_link_libraries(test_policy_map PRIVATE ${SYCL_LIBRARIES})
endif()

# Add compile definitions
target_compile_definitions(test_policy_map PRIVATE
    # Add any necessary definitions here
)

# Set C++ standard
set_property(TARGET test_policy_map PROPERTY CXX_STANDARD 17)
set_property(TARGET test_policy_map PROPERTY CXX_STANDARD_REQUIRED ON)