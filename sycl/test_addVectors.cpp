#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

#include <CL/sycl.hpp>
#include "src/neural/backends/sycl/kernels.h"
#include "src/neural/tables/activation_function.h"

using namespace lczero;
using namespace sycl_backend;

// Function to compare results with tolerance
bool compareVectors(const float* a, const float* b, int size, float tolerance = 1e-5) {
    for (int i = 0; i < size; ++i) {
        if (std::abs(a[i] - b[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

// CPU reference implementation
void addVectorsCPU(float* c, const float* a, const float* b, int size, int asize, int bsize, lczero::sycl_backend::ActivationFunction activation) {
    for (int i = 0; i < size; ++i) {
        float aVal = a ? a[i % asize] : 0.0f;
        float bVal = b ? b[i % bsize] : 0.0f;
        float cVal = aVal + bVal;

        // Apply activation function
        switch (activation) {
            case lczero::sycl_backend::ActivationFunction::ACTIVATION_RELU:
                if (cVal < 0) cVal = 0;
                break;
            case lczero::sycl_backend::ActivationFunction::ACTIVATION_RELU_2:
                if (cVal < 0) cVal = 0;
                cVal *= cVal;
                break;
            case lczero::sycl_backend::ActivationFunction::ACTIVATION_NONE:
                break;
            default:
                std::cout << "Unsupported activation function for CPU test" << std::endl;
                break;
        }

        c[i] = cVal;
    }
}

int main() {
    try {
        // Initialize SYCL queue
        sycl::queue q(sycl::default_selector{},
                      [](sycl::exception_list eL) {
                          for (auto e : eL) {
                              std::cerr << "SYCL Exception: " << e.what() << std::endl;
                          }
                      });

        std::cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

        // Test parameters
        const int size = 1024;
        const int asize = 256;
        const int bsize = 512;

        std::vector<float> host_a(asize);
        std::vector<float> host_b(bsize);
        std::vector<float> host_c(size);
        std::vector<float> reference_c(size);

        // Initialize data with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

        for (int i = 0; i < asize; ++i) {
            host_a[i] = dis(gen);
        }
        for (int i = 0; i < bsize; ++i) {
            host_b[i] = dis(gen);
        }

        // Allocate device memory
        float* device_a = sycl::malloc_device<float>(asize, q);
        float* device_b = sycl::malloc_device<float>(bsize, q);
        float* device_c = sycl::malloc_device<float>(size, q);

        // Copy data to device
        q.memcpy(device_a, host_a.data(), asize * sizeof(float));
        q.memcpy(device_b, host_b.data(), bsize * sizeof(float));
        q.wait();

        // Test different activation functions
        std::vector<ActivationFunction> activations = {
            ActivationFunction::ACTIVATION_NONE,
            ActivationFunction::ACTIVATION_RELU,
            ActivationFunction::ACTIVATION_RELU_2
        };

        std::vector<std::string> activation_names = {
            "NONE", "RELU", "RELU_2"
        };

        for (size_t test = 0; test < activations.size(); ++test) {
            std::cout << "\nTesting with activation: " << activation_names[test] << std::endl;

            // Run SYCL kernel
            addVectors<float>(device_c, device_a, device_b, size, asize, bsize, activations[test], q);
            q.wait();

            // Copy result back
            q.memcpy(host_c.data(), device_c, size * sizeof(float));
            q.wait();

            // CPU reference
            addVectorsCPU(reference_c.data(), host_a.data(), host_b.data(), size, asize, bsize, activations[test]);

            // Compare results
            bool passed = compareVectors(host_c.data(), reference_c.data(), size);

            std::cout << "Test " << activation_names[test] << ": " << (passed ? "PASSED" : "FAILED") << std::endl;

            if (!passed) {
                std::cout << "First few SYCL results: ";
                for (int i = 0; i < 5; ++i) {
                    std::cout << std::fixed << std::setprecision(6) << host_c[i] << " ";
                }
                std::cout << std::endl;

                std::cout << "First few CPU results: ";
                for (int i = 0; i < 5; ++i) {
                    std::cout << std::fixed << std::setprecision(6) << reference_c[i] << " ";
                }
                std::cout << std::endl;
            }
        }

        // Clean up
        sycl::free(device_a, q);
        sycl::free(device_b, q);
        sycl::free(device_c, q);

        std::cout << "\nAll tests completed!" << std::endl;
        return 0;

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}