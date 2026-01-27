#ifndef VECTORADD_KERNEL_H
#define VECTORADD_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements);

#ifdef __cplusplus
}
#endif

#endif // VECTORADD_KERNEL_H