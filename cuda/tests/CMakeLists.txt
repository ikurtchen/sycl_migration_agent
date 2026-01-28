cmake_minimum_required(VERSION 3.18)
project(CudaKernelTests)

# Find CUDA
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
enable_language(CUDA)

# Find Google Test
find_package(GTest REQUIRED)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 90;80;70;60)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${GTEST_INCLUDE_DIRS})

# Common test sources
set(COMMON_TEST_SOURCES
    common/test_utils.h
    common/base_test_fixture.h
)

# FP16 Kernel Tests
add_executable(fp16_tests
    fp16/se_layer_nhwc_test.cu
    fp16/output_input_transform_test.cu
    ${COMMON_TEST_SOURCES}
)
target_link_libraries(fp16_tests GTest::gtest_main ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES} pthread)

# Math Kernel Tests
add_executable(math_tests
    math/add_vectors_test.cu
    math/add_bias_test.cu
    math/batch_norm_test.cu
    math/softmax_test.cu
    math/reduce_sum_test.cu
    ${COMMON_TEST_SOURCES}
)
target_link_libraries(math_tests GTest::gtest_main ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES} pthread)

# Neural Network Kernel Tests
add_executable(nn_tests
    nn/layer_norm_test.cu
    nn/policy_head_test.cu
    nn/value_head_test.cu
    nn/winograd_transform_test.cu
    ${COMMON_TEST_SOURCES}
)
target_link_libraries(nn_tests GTest::gtest_main ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES} pthread)

# Reduction and Transform Tests
add_executable(transform_tests
    reduction/transpose_test.cu
    reduction/reshape_test.cu
    reduction/atomic_sum_test.cu
    ${COMMON_TEST_SOURCES}
)
target_link_libraries(transform_tests GTest::gtest_main ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES} pthread)

# Create output directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/cuda_outputs)