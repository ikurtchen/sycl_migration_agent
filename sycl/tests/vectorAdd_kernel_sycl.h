#ifndef VECTORADD_KERNEL_SYCL_H
#define VECTORADD_KERNEL_SYCL_H

#include <sycl/sycl.hpp>
#include <vector>
#include <stdexcept>

/**
 * @brief Launch the vector addition kernel on SYCL device
 *
 * @param A Pointer to first input vector
 * @param B Pointer to second input vector
 * @param C Pointer to output vector
 * @param numElements Number of elements in each vector
 * @return int 0 on success, error code on failure
 */
int launchVectorAddSYCL(const float* A, const float* B, float* C, int numElements);

/**
 * @brief Vector addition kernel function
 *
 * This function adds two vectors element-wise using a SYCL queue.
 * The kernel is executed with optimal work-group sizes for Intel GPUs.
 *
 * @param A First input vector
 * @param B Second input vector
 * @param C Output vector (A + B)
 * @param numElements Number of elements to process
 * @return 0 on success, -1 on failure
 */
int vectorAddSYCL(const float* A, const float* B, float* C, int numElements);

/**
 * @brief Select and configure the best available SYCL device
 *
 * @param sycl::queue reference to configure
 * @return true if GPU device selected, false for CPU fallback
 */
bool selectOptimalDevice(sycl::queue& q);

#endif // VECTORADD_KERNEL_SYCL_H