cmake_minimum_required(VERSION 3.18)
project(VectorAdd_CUDA CUDA CXX)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Check CUDA version
message(STATUS "Found CUDA Toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")

# Set CUDA architecture flags
# This allows compiling for different GPU architectures
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;75;80;86")
endif()

# Enable separable compilation for better device code linking
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Add CUDA compile flags
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
)

# Create the vectorAdd executable
add_executable(vectorAdd_cuda
    src/vectorAdd.cu
)

# Link CUDA runtime library
target_link_libraries(vectorAdd_cuda
    CUDA::cudart
)

# Set output directory
set_target_properties(vectorAdd_cuda PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add custom target for cleaning
add_custom_target(clean-cuda
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
    COMMENT "Cleaning CUDA build artifacts"
)

# Print build configuration
message(STATUS "=== CUDA Build Configuration ===")
message(STATUS "CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Runtime Library: ${CUDA_CUDART_LIBRARY}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

# Installation rules
install(TARGETS vectorAdd_cuda
    RUNTIME DESTINATION bin
    COMPONENT applications
)

# Package configuration
set(CPACK_PACKAGE_NAME "VectorAdd_CUDA")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CUDA Vector Addition Example")
include(CPack)
