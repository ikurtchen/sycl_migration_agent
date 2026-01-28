#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <random>

int main() {
    // Set up queue with GPU selector
    sycl::queue q(sycl::gpu_selector_v);

    std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    const size_t N = 1024 * 1024;
    std::vector<float> a(N), b(N), c(N);

    // Initialize input data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (size_t i = 0; i < N; ++i) {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }

    // Allocate SYCL buffers
    sycl::buffer<float, 1> buf_a(a.data(), N);
    sycl::buffer<float, 1> buf_b(b.data(), N);
    sycl::buffer<float, 1> buf_c(c.data(), N);

    // Submit kernel
    sycl::event e = q.submit([&](sycl::handler& h) {
        auto acc_a = buf_a.get_access<sycl::access::mode::read>(h);
        auto acc_b = buf_b.get_access<sycl::access::mode::read>(h);
        auto acc_c = buf_c.get_access<sycl::access::mode::write>(h);

        h.parallel_for<class vector_add>(sycl::range<1>(N), [=](sycl::id<1> i) {
            acc_c[i] = acc_a[i] + acc_b[i];
        });
    });

    e.wait();

    // Verify results
    auto acc_c = buf_c.get_access<sycl::access::mode::read>();
    bool correct = true;
    for (size_t i = 0; i < N && correct; ++i) {
        if (std::abs(acc_c[i] - (a[i] + b[i])) > 1e-5) {
            correct = false;
            std::cout << "Mismatch at index " << i << ": expected " << a[i] + b[i]
                      << ", got " << acc_c[i] << std::endl;
        }
    }

    if (correct) {
        std::cout << "Vector addition test PASSED!" << std::endl;
    } else {
        std::cout << "Vector addition test FAILED!" << std::endl;
    }

    return correct ? 0 : 1;
}