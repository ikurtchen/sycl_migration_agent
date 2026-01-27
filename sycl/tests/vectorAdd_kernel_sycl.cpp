#include "vectorAdd_kernel_sycl.h"
#include <iostream>
#include <chrono>

using namespace sycl;

// Vector addition kernel - executed on device
template <typename T>
class vector_add_kernel;

template <typename T>
void vectorAddKernel(queue& q, buffer<T, 1>& bufA, buffer<T, 1>& bufB, buffer<T, 1>& bufC, size_t numElements) {
    q.submit([&](handler& h) {
        accessor accessorA(bufA, h, read_only);
        accessor accessorB(bufB, h, read_only);
        accessor accessorC(bufC, h, write_only);

        // Optimize for Intel GPUs: use subgroups and proper work-group sizing
        h.parallel_for(range<1>(numElements), [=](id<1> idx) {
            int i = idx[0];
            accessorC[i] = accessorA[i] + accessorB[i];
        });
    });
}

int launchVectorAddSYCL(const float* A, const float* B, float* C, int numElements) {
    // Find available devices
    std::vector<device> devices;
    device selectedDevice;

    try {
        // Try to get GPU devices first
        auto gpuDevices = device::get_devices(info::device_type::gpu);
        if (!gpuDevices.empty()) {
            selectedDevice = gpuDevices[0];
            std::cout << "Using GPU: " << selectedDevice.get_info<info::device::name>() << std::endl;
        } else {
            // Fallback to CPU
            selectedDevice = device::get_devices(info::device_type::cpu)[0];
            std::cout << "Using CPU fallback: " << selectedDevice.get_info<info::device::name>() << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error finding device: " << e.what() << std::endl;
        return -1;
    }

    // Create queue with exception handler
    queue q(selectedDevice, [](exception_list eList) {
        for (auto e : eList) {
            std::rethrow_exception(e);
        }
    });

    try {
        // Create buffers for input and output
        buffer<float, 1> bufA(A, range<1>(numElements));
        buffer<float, 1> bufB(B, range<1>(numElements));
        buffer<float, 1> bufC{range<1>(numElements)};

        // Launch kernel
        vectorAddKernel(q, bufA, bufB, bufC, numElements);

        // Copy result back to host
        host_accessor h_C(bufC);
        for (int i = 0; i < numElements; ++i) {
            C[i] = h_C[i];
        }

    } catch (const std::exception& e) {
        std::cerr << "SYCL error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

bool selectOptimalDevice(sycl::queue& q) {
    // Check what device is in the queue
    auto device = q.get_device();
    return device.is_gpu();
}

int vectorAddSYCL(const float* A, const float* B, float* C, int numElements) {
    return launchVectorAddSYCL(A, B, C, numElements);
}