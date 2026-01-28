/*
  Integration example showing how to use the SYCL MHA kernel
  as a drop-in replacement for the CUDA CUTLASS version
*/

#include "neural/backends/sycl/kernels.h"
#include "neural/backends/sycl/sycl_common.h"
#include <iostream>
#include <vector>

using namespace lczero;

int main() {
  try {
    // Initialize SYCL queue
    sycl::queue queue(sycl::default_selector());
    std::cout << "Running on: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

    // MHA configuration (matching CUTLASS MHA parameters)
    const int batch_size = 2;
    const int num_heads = 4;
    const int depth = 32;
    const int kNumQueries = 64;
    const int kNumKeys = 64;

    // Calculate buffer sizes
    const int query_size = batch_size * num_heads * kNumQueries * depth;
    const int key_size = batch_size * num_heads * kNumKeys * depth;
    const int value_size = batch_size * num_heads * kNumKeys * depth;
    const int bias_size = batch_size * num_heads * kNumQueries * kNumKeys;
    const int output_size = batch_size * num_heads * kNumQueries * depth;

    // Allocate device memory using USM (Unified Shared Memory)
    sycl::half* d_query = sycl::malloc_device<sycl::half>(query_size, queue);
    sycl::half* d_key = sycl::malloc_device<sycl::half>(key_size, queue);
    sycl::half* d_value = sycl::malloc_device<sycl::half>(value_size, queue);
    sycl::half* d_bias = sycl::malloc_device<sycl::half>(bias_size, queue);
    sycl::half* d_output = sycl::malloc_device<sycl::half>(output_size, queue);

    // Initialize data with test values (simplified for demo)
    std::vector<sycl::half> h_query(query_size, sycl::half(0.5f));
    std::vector<sycl::half> h_key(key_size, sycl::half(0.3f));
    std::vector<sycl::half> h_value(value_size, sycl::half(0.7f));
    std::vector<sycl::half> h_bias(bias_size, sycl::half(0.1f));
    std::vector<sycl::half> h_output(output_size);

    // Copy data to device
    queue.memcpy(d_query, h_query.data(), query_size * sizeof(sycl::half));
    queue.memcpy(d_key, h_key.data(), key_size * sizeof(sycl::half));
    queue.memcpy(d_value, h_value.data(), value_size * sizeof(sycl::half));
    queue.memcpy(d_bias, h_bias.data(), bias_size * sizeof(sycl::half));
    queue.wait();

    std::cout << "Running SYCL MHA without bias..." << std::endl;
    sycl_backend::fusedMHA(d_output, d_query, d_key, d_value, nullptr,
                           batch_size, num_heads, depth, queue);

    // Copy result back
    queue.memcpy(h_output.data(), d_output, output_size * sizeof(sycl::half));
    queue.wait();

    std::cout << "SYCL MHA without bias completed successfully!" << std::endl;
    std::cout << "Sample output values: "
              << static_cast<float>(h_output[0]) << ", "
              << static_cast<float>(h_output[1]) << ", "
              << static_cast<float>(h_output[2]) << std::endl;

    std::cout << "Running SYCL MHA with bias..." << std::endl;
    sycl_backend::fusedMHA(d_output, d_query, d_key, d_value, d_bias,
                           batch_size, num_heads, depth, queue);

    // Copy result back
    queue.memcpy(h_output.data(), d_output, output_size * sizeof(sycl::half));
    queue.wait();

    std::cout << "SYCL MHA with bias completed successfully!" << std::endl;
    std::cout << "Sample output values with bias: "
              << static_cast<float>(h_output[0]) << ", "
              << static_cast<float>(h_output[1]) << ", "
              << static_cast<float>(h_output[2]) << std::endl;

    // Cleanup
    sycl::free(d_query, queue);
    sycl::free(d_key, queue);
    sycl::free(d_value, queue);
    sycl::free(d_bias, queue);
    sycl::free(d_output, queue);

    std::cout << "Integration test completed successfully!" << std::endl;
    return 0;

  } catch (const sycl::exception& e) {
    std::cerr << "SYCL exception: " << e.what() << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Standard exception: " << e.what() << std::endl;
    return 1;
  }
}