cmake_minimum_required(VERSION 3.10)
project(VectorAddTests)

# Find CUDA
find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)

# Find Google Test
find_package(GTest REQUIRED)
find_package (Threads REQUIRED)

# Include CUDA directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)

# Set CUDA host and device compiler
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

# Create executable for vectorAdd tests
cuda_add_executable(
    vectorAdd_test
    test_vectorAdd.cpp
    vectorAdd_kernel.cu
    OPTIONS
    -std=c++14
)

# Link libraries
target_link_libraries(
    vectorAdd_test
    ${CUDA_LIBRARIES}
    ${GTEST_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    cuda
)

# Set CUDA properties
set_target_properties(vectorAdd_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Add test to CTest
add_test(NAME VectorAddTest COMMAND vectorAdd_test)

# Install rules
install(TARGETS vectorAdd_test DESTINATION bin)
