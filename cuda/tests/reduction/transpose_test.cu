#include <gtest/gtest.h>
#include "../common/base_test_fixture.h"

// Test kernel for matrix transpose
__global__ void transpose_kernel(const float* input, float* output,
                                int rows, int cols) {
    __shared__ float tile[32][32];

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    // Load from global to shared memory with coalesced access
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    __syncthreads();

    // Write to global memory with coalesced access (transposed)
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    if (x < rows && y < cols) {
        output[x * rows + y] = tile[threadIdx.x][threadIdx.y];
    }
}

// Simple transpose without shared memory
__global__ void transpose_naive_kernel(const float* input, float* output,
                                       int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        output[x * rows + y] = input[y * cols + x];
    }
}

// Transpose with padding for non-square matrices
__global__ void transpose_padded_kernel(const float* input, float* output,
                                        int rows, int cols) {
    __shared__ float tile[32][32 + 1];  // Padding to avoid bank conflicts

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    __syncthreads();

    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    if (x < rows && y < cols) {
        output[x * rows + y] = tile[threadIdx.x][threadIdx.y];
    }
}

void launchTranspose(const float* input, float* output, int rows, int cols);

class TransposeTest : public BaseCudaTest<float> {
protected:
    void SetUp() override {
        BaseCudaTest<float>::SetUp();

        // Test dimensions
        rows = 1024;
        cols = 1024;
        total_size = rows * cols;

        // Allocate memory
        allocateMemory(&d_input, &h_input, total_size);
        allocateMemory(&d_output, &h_output, total_size);
        allocateMemory(&d_reference, &h_reference, total_size);

        // Initialize test data
        initializeRandom(h_input, total_size, -100.0f, 100.0f);

        // Copy to device
        copyToDevice(d_input, h_input, total_size);
    }

    void TearDown() override {
        BaseCudaTest<float>::TearDown();

        freeMemory(d_input, h_input);
        freeMemory(d_output, h_output);
        freeMemory(d_reference, h_reference);
    }

    void cpuTranspose(const float* input, float* output, int rows, int cols) {
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                output[x * rows + y] = input[y * cols + x];
            }
        }
    }

    int rows, cols, total_size;
    float *d_input, *d_output, *d_reference;
    float *h_input, *h_output, *h_reference;
};

TEST_F(TransposeTest, SharedMemoryCorrectness) {
    // Launch kernel with shared memory
    dim3 block(32, 32);
    dim3 grid((cols + 31) / 32, (rows + 31) / 32);

    transpose_kernel<<<grid, block>>>(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_output, d_output, total_size);

    // CPU reference
    cpuTranspose(h_input, h_reference, rows, cols);

    // Compare results
    for (int i = 0; i < total_size; ++i) {
        EXPECT_NEAR(h_output[i], h_reference[i], 1e-5f) << "Shared memory transpose mismatch at index " << i;
    }

    // Save outputs
    saveOutput("transpose_shared_output.bin", h_output, total_size);
    saveInput("transpose_input.bin", h_input, total_size);
}

TEST_F(TransposeTest, NaiveCorrectness) {
    // Launch naive kernel
    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);

    transpose_naive_kernel<<<grid, block>>>(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_output, d_output, total_size);

    // CPU reference
    cpuTranspose(h_input, h_reference, rows, cols);

    // Compare results
    for (int i = 0; i < total_size; ++i) {
        EXPECT_NEAR(h_output[i], h_reference[i], 1e-5f) << "Naive transpose mismatch at index " << i;
    }

    // Save output
    saveOutput("transpose_naive_output.bin", h_output, total_size);
}

TEST_F(TransposeTest, PaddedCorrectness) {
    // Launch padded kernel
    dim3 block(32, 32);
    dim3 grid((cols + 31) / 32, (rows + 31) / 32);

    transpose_padded_kernel<<<grid, block>>>(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_output, d_output, total_size);

    // CPU reference
    cpuTranspose(h_input, h_reference, rows, cols);

    // Compare results
    for (int i = 0; i < total_size; ++i) {
        EXPECT_NEAR(h_output[i], h_reference[i], 1e-5f) << "Padded transpose mismatch at index " << i;
    }

    // Save output
    saveOutput("transpose_padded_output.bin", h_output, total_size);
}

TEST_F(TransposeTest, BenchmarkSharedMemory) {
    dim3 block(32, 32);
    dim3 grid((cols + 31) / 32, (rows + 31) / 32);

    auto kernel_func = [&]() {
        transpose_kernel<<<grid, block>>>(d_input, d_output, rows, cols);
    };

    BenchmarkResult result = benchmarkKernel(
        kernel_func, total_size, "transpose_shared", 10, 100);

    std::cout << "Transpose Shared Memory - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/transpose_shared_benchmark.json");
}

TEST_F(TransposeTest, BenchmarkNaive) {
    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);

    auto kernel_func = [&]() {
        transpose_naive_kernel<<<grid, block>>>(d_input, d_output, rows, cols);
    };

    BenchmarkResult result = benchmarkKernel(
        kernel_func, total_size, "transpose_naive", 10, 100);

    std::cout << "Transpose Naive - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/transpose_naive_benchmark.json");
}

TEST_F(TransposeTest, DifferentSizes) {
    struct MatrixSize {
        int rows, cols;
    };

    std::vector<MatrixSize> test_sizes = {
        {256, 256},      // Square
        {512, 1024},     // Rectangular (wide)
        {1024, 512},     // Rectangular (tall)
        {2048, 2048},    // Large square
        {64, 2048},      // Very wide
        {2048, 64}       // Very tall
    };

    for (auto size : test_sizes) {
        int test_total = size.rows * size.cols;

        float *test_input, *test_output, *test_reference;
        allocateMemory(&test_input, &h_input, test_total);
        allocateMemory(&test_output, &h_output, test_total);
        allocateMemory(&test_reference, &h_reference, test_total);

        initializeRandom(h_input, test_total, -100.0f, 100.0f);
        copyToDevice(test_input, h_input, test_total);

        // Test shared memory version
        dim3 block(32, 32);
        dim3 grid((size.cols + 31) / 32, (size.rows + 31) / 32);

        transpose_kernel<<<grid, block>>>(test_input, test_output, size.rows, size.cols);
        CUDA_CHECK(cudaDeviceSynchronize());

        copyFromDevice(h_output, test_output, test_total);

        // CPU reference
        cpuTranspose(h_input, h_reference, size.rows, size.cols);

        // Compare a few random elements
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, test_total - 1);
        for (int i = 0; i < 20; ++i) {
            int idx = dis(gen);
            EXPECT_NEAR(h_output[idx], h_reference[idx], 1e-4f)
                << "Size " << size.rows << "x" << size.cols << " mismatch at index " << idx;
        }

        std::string filename = "transpose_" + std::to_string(size.rows) +
                               "x" + std::to_string(size.cols) + ".bin";
        saveOutput(filename.c_str(), h_output, test_total);

        freeMemory(test_input, h_input);
        freeMemory(test_output, h_output);
        freeMemory(test_reference, h_reference);
    }
}

TEST_F(TransposeTest, PerformanceComparison) {
    dim3 block_shared(32, 32);
    dim3 grid_shared((cols + 31) / 32, (rows + 31) / 32);

    dim3 block_naive(16, 16);
    dim3 grid_naive((cols + 15) / 16, (rows + 15) / 16);

    // Benchmark shared memory version
    CudaTimer timer1;
    timer1.start();
    for (int i = 0; i < 100; ++i) {
        transpose_kernel<<<grid_shared, block_shared>>>(d_input, d_output, rows, cols);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer1.stop();
    float time_shared = timer1.elapsed_ms() / 100.0f;

    // Benchmark naive version
    CudaTimer timer2;
    timer2.start();
    for (int i = 0; i < 100; ++i) {
        transpose_naive_kernel<<<grid_naive, block_naive>>>(d_input, d_output, rows, cols);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer2.stop();
    float time_naive = timer2.elapsed_ms() / 100.0f;

    // Benchmark padded version
    CudaTimer timer3;
    timer3.start();
    for (int i = 0; i < 100; ++i) {
        transpose_padded_kernel<<<grid_shared, block_shared>>>(d_input, d_output, rows, cols);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer3.stop();
    float time_padded = timer3.elapsed_ms() / 100.0f;

    std::cout << "Shared memory transpose time: " << time_shared << " ms" << std::endl;
    std::cout << "Naive transpose time: " << time_naive << " ms" << std::endl;
    std::cout << "Padded transpose time: " << time_padded << " ms" << std::endl;
    std::cout << "Shared vs Naive speedup: " << (time_naive / time_shared) << "x" << std::endl;
    std::cout << "Padded vs Shared speedup: " << (time_shared / time_padded) << "x" << std::endl;

    // Save performance comparison
    std::ofstream file("cuda_outputs/transpose_performance_comparison.json");
    file << "{\n";
    file << "  \"shared_memory_ms\": " << time_shared << ",\n";
    file << "  \"naive_ms\": " << time_naive << ",\n";
    file << "  \"padded_ms\": " << time_padded << ",\n";
    file << "  \"shared_vs_naive_speedup\": " << (time_naive / time_shared) << ",\n";
    file << "  \"padded_vs_shared_speedup\": " << (time_shared / time_padded) << "\n";
    file << "}\n";
    file.close();
}

// Dummy implementation of launch function for test linking
void launchTranspose(const float* input, float* output, int rows, int cols) {
    dim3 block(32, 32);
    dim3 grid((cols + 31) / 32, (rows + 31) / 32);
    transpose_kernel<<<grid, block>>>(input, output, rows, cols);
}