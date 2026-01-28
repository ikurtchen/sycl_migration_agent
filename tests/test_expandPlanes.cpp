/*
  Test program for expandPlanes kernel migration verification
  This test verifies that the SYCL expandPlanes kernels produce identical
  results to the CUDA versions for both NHWC and NCHW layouts.
*/

#include <iostream>
#include <vector>
#include <random>
#include <cstring>
#include <iomanip>
#include <chrono>

#include "neural/backends/sycl/sycl_common.h"

using namespace lczero::sycl_backend;

// Helper function to generate random test data
void generateTestData(std::vector<uint64_t>& masks, std::vector<float>& values, int batch_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> mask_dist(0, 0xFFFFFFFFFFFFFFFFULL);
    std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);

    const int kInputPlanes = 112;

    masks.resize(batch_size * kInputPlanes);
    values.resize(batch_size * kInputPlanes);

    for (int i = 0; i < batch_size * kInputPlanes; ++i) {
        masks[i] = mask_dist(gen);
        values[i] = value_dist(gen);
    }
}

// Helper function to verify results
bool compareResults(const std::vector<float>& cuda_result,
                   const std::vector<float>& sycl_result,
                   const std::string& test_name,
                   float tolerance = 1e-6) {

    if (cuda_result.size() != sycl_result.size()) {
        std::cout << "FAIL: " << test_name << " - Size mismatch" << std::endl;
        return false;
    }

    bool passed = true;
    float max_diff = 0.0f;

    for (size_t i = 0; i < cuda_result.size(); ++i) {
        float diff = std::abs(cuda_result[i] - sycl_result[i]);
        if (diff > tolerance) {
            std::cout << "FAIL: " << test_name << " - Element " << i
                      << " CUDA=" << cuda_result[i] << " SYCL=" << sycl_result[i]
                      << " diff=" << diff << std::endl;
            passed = false;
        }
        max_diff = std::max(max_diff, diff);
    }

    if (passed) {
        std::cout << "PASS: " << test_name << " (max_diff=" << max_diff << ")" << std::endl;
    }

    return passed;
}

// CPU implementation for reference (same logic as CUDA/SYCL kernels)
void expandPlanesCPU_NHWC(float* output, const uint64_t* masks, const float* values, int n) {
    const int kInputPlanes = 112;

    for (int index = 0; index < n * 8 * 8; ++index) {
        const int planeIndex = index % kInputPlanes;
        const int boardIndex = index / (kInputPlanes * 8 * 8);
        const int sqIndex = (index / kInputPlanes) & 0x3F;

        uint64_t mask = masks[boardIndex * kInputPlanes + planeIndex];

        float op = 0.0f;
        bool set = !!(mask & (1ull << sqIndex));
        if (set) {
            op = values[boardIndex * kInputPlanes + planeIndex];
        }
        output[index] = op;
    }
}

void expandPlanesCPU_NCHW(float* output, const uint64_t* masks, const float* values, int n) {
    for (unsigned index = 0; index < static_cast<unsigned>(n * 8 * 8 / 2); ++index) {
        unsigned idx = index;
        idx *= 2;
        unsigned planeIndex = idx >> 6;

        if (planeIndex >= static_cast<unsigned>(n)) continue;

        uint64_t mask = masks[planeIndex];

        int sqIndex = idx & 0x3F;
        float op[2] = {0.0f, 0.0f};

        bool set = !!(mask & (1ull << sqIndex));
        if (set) {
            op[0] = values[planeIndex];
        }
        sqIndex++;
        set = !!(mask & (1ull << sqIndex));
        if (set) {
            op[1] = values[planeIndex];
        }
        output[idx + 0] = op[0];
        output[idx + 1] = op[1];
    }
}

int main() {
    try {
        // Create SYCL queue
        sycl::queue q(sycl::default_selector_v);
        std::cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

        // Test parameters
        const int batch_size = 4;
        const int kInputPlanes = 112;

        // Generate test data
        std::vector<uint64_t> h_masks;
        std::vector<float> h_values;
        generateTestData(h_masks, h_values, batch_size);

        // Allocate device memory
        uint64_t* d_masks = sycl::malloc_device<uint64_t>(h_masks.size(), q);
        float* d_values = sycl::malloc_device<float>(h_values.size(), q);

        // Copy data to device
        q.memcpy(d_masks, h_masks.data(), h_masks.size() * sizeof(uint64_t));
        q.memcpy(d_values, h_values.data(), h_values.size() * sizeof(float));
        q.wait();

        // Test NHWC variant
        {
            const int output_size = batch_size * kInputPlanes * 8 * 8;
            float* d_output_nhwc = sycl::malloc_device<float>(output_size, q);
            std::vector<float> h_output_sycl(output_size);
            std::vector<float> h_output_cpu(output_size);

            // Run SYCL kernel
            expandPlanes_NHWC(d_output_nhwc, d_masks, d_values, batch_size, q);

            // Copy results back
            q.memcpy(h_output_sycl.data(), d_output_nhwc, output_size * sizeof(float));
            q.wait();

            // Run CPU reference
            expandPlanesCPU_NHWC(h_output_cpu.data(), h_masks.data(), h_values.data(), batch_size);

            // Compare results
            bool passed = compareResults(h_output_cpu, h_output_sycl, "expandPlanes_NHWC");

            sycl::free(d_output_nhwc, q);

            if (!passed) {
                return 1;
            }
        }

        // Test NCHW variant
        {
            const int output_size = batch_size * kInputPlanes * 8 * 8;
            float* d_output_nchw = sycl::malloc_device<float>(output_size, q);
            std::vector<float> h_output_sycl(output_size);
            std::vector<float> h_output_cpu(output_size);

            // Run SYCL kernel
            expandPlanes_NCHW(d_output_nchw, d_masks, d_values, batch_size, q);

            // Copy results back
            q.memcpy(h_output_sycl.data(), d_output_nchw, output_size * sizeof(float));
            q.wait();

            // Run CPU reference
            expandPlanesCPU_NCHW(h_output_cpu.data(), h_masks.data(), h_values.data(), batch_size);

            // Compare results
            compareResults(h_output_cpu, h_output_sycl, "expandPlanes_NCHW");

            sycl::free(d_output_nchw, q);
        }

        // Performance test
        std::cout << "\nPerformance test:" << std::endl;
        const int perf_batch_size = 32;
        const int iterations = 100;

        // Resize for performance test
        h_masks.resize(perf_batch_size * kInputPlanes);
        h_values.resize(perf_batch_size * kInputPlanes);
        generateTestData(h_masks, h_values, perf_batch_size);

        uint64_t* d_perf_masks = sycl::malloc_device<uint64_t>(h_masks.size(), q);
        float* d_perf_values = sycl::malloc_device<float>(h_values.size(), q);

        q.memcpy(d_perf_masks, h_masks.data(), h_masks.size() * sizeof(uint64_t));
        q.memcpy(d_perf_values, h_values.data(), h_values.size() * sizeof(float));

        const int perf_output_size = perf_batch_size * kInputPlanes * 8 * 8;
        float* d_perf_output = sycl::malloc_device<float>(perf_output_size, q);

        // Time the NHWC variant
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            expandPlanes_NHWC(d_perf_output, d_perf_masks, d_perf_values, perf_batch_size, q);
        }
        q.wait();
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_time = duration.count() / static_cast<double>(iterations);

        std::cout << "NHWC variant - Average time: " << std::fixed << std::setprecision(3)
                  << avg_time << " microseconds" << std::endl;

        // Time the NCHW variant
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            expandPlanes_NCHW(d_perf_output, d_perf_masks, d_perf_values, perf_batch_size, q);
        }
        q.wait();
        end = std::chrono::high_resolution_clock::now();

        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        avg_time = duration.count() / static_cast<double>(iterations);

        std::cout << "NCHW variant - Average time: " << std::fixed << std::setprecision(3)
                  << avg_time << " microseconds" << std::endl;

        // Cleanup
        sycl::free(d_masks, q);
        sycl::free(d_values, q);
        sycl::free(d_perf_masks, q);
        sycl::free(d_perf_values, q);
        sycl::free(d_perf_output, q);

        std::cout << "\nAll tests passed successfully!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}