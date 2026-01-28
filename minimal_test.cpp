#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

// Simple activation function for testing
enum class ActivationFunction {
    ACTIVATION_NONE,
    ACTIVATION_RELU,
    ACTIVATION_RELU_2
};

inline float activate(float cVal, ActivationFunction activation) {
    switch (activation) {
        case ActivationFunction::ACTIVATION_RELU:
            if (cVal < 0) cVal = 0;
            break;
        case ActivationFunction::ACTIVATION_RELU_2:
            if (cVal < 0) cVal = 0;
            cVal *= cVal;
            break;
        case ActivationFunction::ACTIVATION_NONE:
            break;
        default:
            break;
    }
    return cVal;
}

// CPU reference implementation
void addVectorsCPU(float* c, const float* a, const float* b, int size, ActivationFunction activation) {
    for (int i = 0; i < size; ++i) {
        float cVal = a[i] + b[i];
        cVal = activate(cVal, activation);
        c[i] = cVal;
    }
}

// SYCL kernel implementation
void addVectorsSYCL(sycl::queue& q, float* c, const float* a, const float* b, int size, ActivationFunction activation) {
    sycl::buffer<float, 1> buf_c(c, size);
    sycl::buffer<float, 1> buf_a(const_cast<float*>(a), size);
    sycl::buffer<float, 1> buf_b(const_cast<float*>(b), size);

    q.submit([&](sycl::handler& h) {
        auto acc_c = buf_c.get_access<sycl::access::mode::write>(h);
        auto acc_a = buf_a.get_access<sycl::access::mode::read>(h);
        auto acc_b = buf_b.get_access<sycl::access::mode::read>(h);

        h.parallel_for<class vector_add>(sycl::range<1>(size), [=](sycl::id<1> i) {
            float cVal = acc_a[i] + acc_b[i];

            // Apply activation function
            switch (activation) {
                case ActivationFunction::ACTIVATION_RELU:
                    if (cVal < 0) cVal = 0;
                    break;
                case ActivationFunction::ACTIVATION_RELU_2:
                    if (cVal < 0) cVal = 0;
                    cVal *= cVal;
                    break;
                case ActivationFunction::ACTIVATION_NONE:
                    break;
                default:
                    break;
            }

            acc_c[i] = cVal;
        });
    }).wait();
}

int main() {
    try {
        // Set up queue with GPU selector
        sycl::queue q(sycl::default_selector_v);

        std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

        const size_t N = 1024;
        std::vector<float> a(N), b(N), c_cpu(N), c_sycl(N);

        // Initialize input data
        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i) * 0.5f - 256.0f;  // Mix of positive and negative
            b[i] = static_cast<float>(N - i) * 0.3f;
        }

        // Test with different activation functions
        std::vector<ActivationFunction> activations = {
            ActivationFunction::ACTIVATION_NONE,
            ActivationFunction::ACTIVATION_RELU,
            ActivationFunction::ACTIVATION_RELU_2
        };

        const char* activation_names[] = {"NONE", "RELU", "RELU_2"};

        for (size_t test_idx = 0; test_idx < activations.size(); ++test_idx) {
            auto activation = activations[test_idx];
            std::cout << "\nTesting activation: " << activation_names[test_idx] << std::endl;

            // Clear results
            std::fill(c_cpu.begin(), c_cpu.end(), 0.0f);
            std::fill(c_sycl.begin(), c_sycl.end(), 0.0f);

            // CPU reference
            addVectorsCPU(c_cpu.data(), a.data(), b.data(), N, activation);

            // SYCL implementation
            addVectorsSYCL(q, c_sycl.data(), a.data(), b.data(), N, activation);

            // Verify results
            bool correct = true;
            for (size_t i = 0; i < N && correct; ++i) {
                if (std::abs(c_cpu[i] - c_sycl[i]) > 1e-5) {
                    correct = false;
                    std::cout << "Mismatch at index " << i << ": CPU=" << c_cpu[i]
                              << ", SYCL=" << c_sycl[i] << std::endl;
                }
            }

            if (correct) {
                std::cout << "✓ Test " << activation_names[test_idx] << " PASSED!" << std::endl;
            } else {
                std::cout << "✗ Test " << activation_names[test_idx] << " FAILED!" << std::endl;
            }
        }

        std::cout << "\nAll tests completed!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}