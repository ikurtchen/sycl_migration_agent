#include <gtest/gtest.h>
#include "../common/base_test_fixture.h"

// Test kernel for 1D to 2D reshape
__global__ void reshape_1d_to_2d_kernel(const float* input, float* output,
                                        int size, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int row = idx / cols;
        int col = idx % cols;
        output[row * cols + col] = input[idx];
    }
}

// Test kernel for 2D to 1D reshape
__global__ void reshape_2d_to_1d_kernel(const float* input, float* output,
                                        int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
        int idx = y * cols + x;
        output[idx] = input[y * cols + x];
    }
}

// Test kernel for ND reshape (general case)
__global__ void reshape_nd_kernel(const float* input, float* output,
                                 const int* input_dims, const int* output_dims,
                                 int input_rank, int output_rank, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    // Convert 1D index to input multi-dim indices
    int temp_idx = idx;
    int input_stride = 1;
    for (int i = input_rank - 1; i >= 0; --i) {
        int dim_idx = temp_idx % input_dims[i];
        temp_idx /= input_dims[i];
        input_stride *= input_dims[i];
    }

    // Copy the value (reshape is just memory layout change)
    output[idx] = input[idx];
}

// Test kernel for batch reshape
__global__ void reshape_batch_kernel(const float* input, float* output,
                                    int batch_size, int input_size_per_batch,
                                    int output_size_per_batch) {
    int batch_idx = blockIdx.x;
    int elem_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx < batch_size && elem_idx < min(input_size_per_batch, output_size_per_batch)) {
        int input_idx = batch_idx * input_size_per_batch + elem_idx;
        int output_idx = batch_idx * output_size_per_batch + elem_idx;
        output[output_idx] = input[input_idx];
    }
}

void launchReshape1DTo2D(const float* input, float* output, int size, int rows, int cols);
void launchReshape2DTo1D(const float* input, float* output, int rows, int cols);

class ReshapeTest : public BaseCudaTest<float> {
protected:
    void SetUp() override {
        BaseCudaTest<float>::SetUp();

        // Test dimensions
        // Test 1D to 2D: 1024 elements -> 32x32 matrix
        size_1d = 1024;
        rows_2d = 32;
        cols_2d = 32;
        size_2d = rows_2d * cols_2d;

        // Allocate memory for 1D to 2D test
        allocateMemory(&d_input_1d, &h_input_1d, size_1d);
        allocateMemory(&d_output_2d, &h_output_2d, size_2d);

        // Test 2D to 1D: 64x64 matrix -> 4096 elements
        rows_2d_test = 64;
        cols_2d_test = 64;
        size_1d_test = rows_2d_test * cols_2d_test;

        // Allocate memory for 2D to 1D test
        allocateMemory(&d_input_2d, &h_input_2d, size_1d_test);
        allocateMemory(&d_output_1d, &h_output_1d, size_1d_test);

        // Initialize test data
        initializeRandom(h_input_1d, size_1d, -100.0f, 100.0f);
        initializeRandom(h_input_2d, size_1d_test, -100.0f, 100.0f);

        // Copy to device
        copyToDevice(d_input_1d, h_input_1d, size_1d);
        copyToDevice(d_input_2d, h_input_2d, size_1d_test);
    }

    void TearDown() override {
        BaseCudaTest<float>::TearDown();

        freeMemory(d_input_1d, h_input_1d);
        freeMemory(d_output_2d, h_output_2d);
        freeMemory(d_input_2d, h_input_2d);
        freeMemory(d_output_1d, h_output_1d);
    }

    void cpuReshape1DTo2D(const float* input, float* output, int size, int rows, int cols) {
        for (int idx = 0; idx < size; ++idx) {
            int row = idx / cols;
            int col = idx % cols;
            output[row * cols + col] = input[idx];
        }
    }

    void cpuReshape2DTo1D(const float* input, float* output, int rows, int cols) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                output[i * cols + j] = input[i * cols + j];
            }
        }
    }

    int size_1d, rows_2d, cols_2d, size_2d;
    int rows_2d_test, cols_2d_test, size_1d_test;
    float *d_input_1d, *d_output_2d, *d_input_2d, *d_output_1d;
    float *h_input_1d, *h_output_2d, *h_input_2d, *h_output_1d;
};

TEST_F(ReshapeTest, Reshape1DTo2DCorrectness) {
    // Launch kernel
    int block_size = 256;
    int grid_size = (size_1d + block_size - 1) / block_size;

    reshape_1d_to_2d_kernel<<<grid_size, block_size>>>(d_input_1d, d_output_2d, size_1d, rows_2d, cols_2d);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_output_2d, d_output_2d, size_2d);

    // CPU reference
    float* h_reference = new float[size_2d];
    cpuReshape1DTo2D(h_input_1d, h_reference, size_1d, rows_2d, cols_2d);

    // Compare results
    for (int i = 0; i < size_2d; ++i) {
        EXPECT_NEAR(h_output_2d[i], h_reference[i], 1e-5f) << "1D to 2D reshape mismatch at index " << i;
    }

    // Save outputs
    saveOutput("reshape_1d_to_2d.bin", h_output_2d, size_2d);
    saveInput("reshape_1d_input.bin", h_input_1d, size_1d);

    delete[] h_reference;
}

TEST_F(ReshapeTest, Reshape2DTo1DCorrectness) {
    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((cols_2d_test + 15) / 16, (rows_2d_test + 15) / 16);

    reshape_2d_to_1d_kernel<<<grid, block>>>(d_input_2d, d_output_1d, rows_2d_test, cols_2d_test);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_output_1d, d_output_1d, size_1d_test);

    // CPU reference
    float* h_reference = new float[size_1d_test];
    cpuReshape2DTo1D(h_input_2d, h_reference, rows_2d_test, cols_2d_test);

    // Compare results
    for (int i = 0; i < size_1d_test; ++i) {
        EXPECT_NEAR(h_output_1d[i], h_reference[i], 1e-5f) << "2D to 1D reshape mismatch at index " << i;
    }

    // Save outputs
    saveOutput("reshape_2d_to_1d.bin", h_output_1d, size_1d_test);
    saveInput("reshape_2d_input.bin", h_input_2d, size_1d_test);

    delete[] h_reference;
}

TEST_F(ReshapeTest, RoundTripCorrectness) {
    // 1D -> 2D -> 1D round trip
    int block_size = 256;
    int grid_size = (size_1d + block_size - 1) / block_size;

    // 1D to 2D
    reshape_1d_to_2d_kernel<<<grid_size, block_size>>>(d_input_1d, d_output_2d, size_1d, rows_2d, cols_2d);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2D to 1D
    dim3 block_2d(16, 16);
    dim3 grid_2d((cols_2d + 15) / 16, (rows_2d + 15) / 16);
    reshape_2d_to_1d_kernel<<<grid_2d, block_2d>>>(d_output_2d, d_input_2d, rows_2d, cols_2d);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_output_1d, d_input_2d, size_1d);

    // Should match original
    for (int i = 0; i < size_1d; ++i) {
        EXPECT_NEAR(h_output_1d[i], h_input_1d[i], 1e-5f) << "Round trip mismatch at index " << i;
    }

    // Save output
    saveOutput("reshape_round_trip.bin", h_output_1d, size_1d);
}

TEST_F(ReshapeTest, Benchmark1DTo2D) {
    int block_size = 256;
    int grid_size = (size_1d + block_size - 1) / block_size;

    auto kernel_func = [&]() {
        reshape_1d_to_2d_kernel<<<grid_size, block_size>>>(d_input_1d, d_output_2d, size_1d, rows_2d, cols_2d);
    };

    BenchmarkResult result = benchmarkKernel(
        kernel_func, size_1d, "reshape_1d_to_2d", 100, 1000);

    std::cout << "Reshape 1D to 2D - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/reshape_1d_to_2d_benchmark.json");
}

TEST_F(ReshapeTest, Benchmark2DTo1D) {
    dim3 block(16, 16);
    dim3 grid((cols_2d_test + 15) / 16, (rows_2d_test + 15) / 16);

    auto kernel_func = [&]() {
        reshape_2d_to_1d_kernel<<<grid, block>>>(d_input_2d, d_output_1d, rows_2d_test, cols_2d_test);
    };

    BenchmarkResult result = benchmarkKernel(
        kernel_func, size_1d_test, "reshape_2d_to_1d", 100, 1000);

    std::cout << "Reshape 2D to 1D - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/reshape_2d_to_1d_benchmark.json");
}

TEST_F(ReshapeTest, DifferentSizes) {
    struct ReshapeSize {
        int input_size, rows, cols;
    };

    std::vector<ReshapeSize> test_sizes = {
        {256, 16, 16},
        {1000, 20, 50},
        {4096, 64, 64},
        {10000, 100, 100},
        {65536, 256, 256}
    };

    for (auto size : test_sizes) {
        float *test_input, *test_output;
        allocateMemory(&test_input, &h_input_1d, size.input_size);
        allocateMemory(&test_output, &h_output_2d, size.input_size);

        initializeRandom(h_input_1d, size.input_size, -100.0f, 100.0f);
        copyToDevice(test_input, h_input_1d, size.input_size);

        int block_size = 256;
        int grid_size = (size.input_size + block_size - 1) / block_size;

        reshape_1d_to_2d_kernel<<<grid_size, block_size>>>(
            test_input, test_output, size.input_size, size.rows, size.cols);
        CUDA_CHECK(cudaDeviceSynchronize());

        copyFromDevice(h_output_2d, test_output, size.input_size);

        // Verify by checking reshape reversibility
        float *test_roundtrip;
        allocateMemory(&test_roundtrip, &h_output_1d, size.input_size);

        dim3 block_2d(16, 16);
        dim3 grid_2d((size.cols + 15) / 16, (size.rows + 15) / 16);

        reshape_2d_to_1d_kernel<<<grid_2d, block_2d>>>(
            test_output, test_roundtrip, size.rows, size.cols);
        CUDA_CHECK(cudaDeviceSynchronize());

        copyFromDevice(h_output_1d, test_roundtrip, size.input_size);

        for (int i = 0; i < min(1000, size.input_size); ++i) {
            EXPECT_NEAR(h_output_1d[i], h_input_1d[i], 1e-4f)
                << "Size " << size.input_size << " reshape roundtrip mismatch at index " << i;
        }

        std::string filename = "reshape_size_" + std::to_string(size.input_size) +
                               "_" + std::to_string(size.rows) + "x" +
                               std::to_string(size.cols) + ".bin";
        saveOutput(filename.c_str(), h_output_2d, size.input_size);

        freeMemory(test_input, h_input_1d);
        freeMemory(test_output, h_output_2d);
        freeMemory(test_roundtrip, h_output_1d);
    }
}

TEST_F(ReshapeTest, BatchReshape) {
    int batch_size = 8;
    int input_size_per_batch = 512;
    int output_size_per_batch = 1024;
    int total_input_size = batch_size * input_size_per_batch;
    int total_output_size = batch_size * output_size_per_batch;

    float *d_batch_input, *d_batch_output;
    float *h_batch_input, *h_batch_output;

    allocateMemory(&d_batch_input, &h_batch_input, total_input_size);
    allocateMemory(&d_batch_output, &h_batch_output, total_output_size);

    initializeRandom(h_batch_input, total_input_size, -100.0f, 100.0f);
    copyToDevice(d_batch_input, h_batch_input, total_input_size);

    dim3 block(1, 256);
    dim3 grid(batch_size, (input_size_per_batch + 255) / 256);

    reshape_batch_kernel<<<grid, block>>>(
        d_batch_input, d_batch_output, batch_size,
        input_size_per_batch, output_size_per_batch);
    CUDA_CHECK(cudaDeviceSynchronize());

    copyFromDevice(h_batch_output, d_batch_output, total_output_size);

    // Verify first element of each batch matches
    for (int b = 0; b < batch_size; ++b) {
        EXPECT_EQ(h_batch_output[b * output_size_per_batch],
                 h_batch_input[b * input_size_per_batch])
            << "Batch " << b << " first element mismatch";
    }

    saveOutput("reshape_batch_output.bin", h_batch_output, total_output_size);

    freeMemory(d_batch_input, h_batch_input);
    freeMemory(d_batch_output, h_batch_output);
}

// Dummy implementations of launch functions for test linking
void launchReshape1DTo2D(const float* input, float* output, int size, int rows, int cols) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    reshape_1d_to_2d_kernel<<<grid_size, block_size>>>(input, output, size, rows, cols);
}

void launchReshape2DTo1D(const float* input, float* output, int rows, int cols) {
    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    reshape_2d_to_1d_kernel<<<grid, block>>>(input, output, rows, cols);
}