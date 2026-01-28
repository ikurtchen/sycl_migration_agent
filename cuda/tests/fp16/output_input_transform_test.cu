#include <gtest/gtest.h>
#include "../common/base_test_fixture.h"

// Simplified version of OutputInputTransformKernel_fp16_shmem_board
template<int BLOCK_SIZE>
__global__ void output_input_transform_kernel(half* output, const half* input,
                                             int width, int height) {
    __shared__ half shared_mem[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Load tile to shared memory
    int x = bx * BLOCK_SIZE + tx;
    int y = by * BLOCK_SIZE + ty;

    if (x < width && y < height) {
        shared_mem[ty][tx] = input[y * width + x];
    } else {
        shared_mem[ty][tx] = __float2half(0.0f);
    }

    __syncthreads();

    // Perform transformation (simple transpose as example)
    int out_x = by * BLOCK_SIZE + tx;
    int out_y = bx * BLOCK_SIZE + ty;

    if (out_x < height && out_y < width) {
        output[out_y * height + out_x] = shared_mem[tx][ty] * __float2half(2.0f);
    }
}

void launchOutputInputTransform(half* output, const half* input,
                                int width, int height);

class OutputInputTransformTest : public FP16Test {
protected:
    void SetUp() override {
        FP16Test::SetUp();

        // Test dimensions
        width = 512;
        height = 512;
        total_size = width * height;

        // Allocate memory
        allocateMemory(&d_input, &h_input, total_size);
        allocateMemory(&d_output, &h_output, total_size);

        // Initialize test data
        initializeRandom(h_input, total_size, half(-1.0f), half(1.0f));

        // Copy to device
        copyToDevice(d_input, h_input, total_size);
    }

    void TearDown() override {
        FP16Test::TearDown();

        freeMemory(d_input, h_input);
        freeMemory(d_output, h_output);
    }

    int width, height, total_size;
    half *d_input, *d_output;
    half *h_input, *h_output;
};

TEST_F(OutputInputTransformTest, Correctness8x8) {
    const int BLOCK_SIZE = 8;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    output_input_transform_kernel<BLOCK_SIZE><<<grid, block>>>(
        d_output, d_input, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_output, d_output, total_size);

    // Save outputs
    saveOutput("output_input_transform_8x8.bin", h_output, total_size);
    saveInput("output_input_transform_input.bin", h_input, total_size);

    // Basic sanity checks
    for (int i = 0; i < total_size; ++i) {
        float val = __half2float(h_output[i]);
        EXPECT_LE(abs(val), 2.0f) << "Output too large at index " << i;
    }
}

TEST_F(OutputInputTransformTest, Correctness16x16) {
    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    output_input_transform_kernel<BLOCK_SIZE><<<grid, block>>>(
        d_output, d_input, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_output, d_output, total_size);

    // Save outputs
    saveOutput("output_input_transform_16x16.bin", h_output, total_size);

    // Basic sanity checks
    for (int i = 0; i < total_size; ++i) {
        float val = __half2float(h_output[i]);
        EXPECT_LE(abs(val), 2.0f) << "Output too large at index " << i;
    }
}

TEST_F(OutputInputTransformTest, Benchmark8x8) {
    const int BLOCK_SIZE = 8;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    auto kernel_func = [&]() {
        output_input_transform_kernel<BLOCK_SIZE><<<grid, block>>>(
            d_output, d_input, width, height);
    };

    BenchmarkResult result = benchmarkKernel(
        kernel_func, total_size, "output_input_transform_8x8", 5, 50);

    std::cout << "Output Input Transform 8x8 - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/output_input_transform_8x8_benchmark.json");
}

TEST_F(OutputInputTransformTest, Benchmark16x16) {
    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    auto kernel_func = [&]() {
        output_input_transform_kernel<BLOCK_SIZE><<<grid, block>>>(
            d_output, d_input, width, height);
    };

    BenchmarkResult result = benchmarkKernel(
        kernel_func, total_size, "output_input_transform_16x16", 5, 50);

    std::cout << "Output Input Transform 16x16 - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/output_input_transform_16x16_benchmark.json");
}

TEST_F(OutputInputTransformTest, DifferentSizes) {
    struct TestSize {
        int w, h;
    };

    std::vector<TestSize> test_sizes = {
        {256, 256}, {1024, 1024}, {64, 1024}
    };

    for (auto size : test_sizes) {
        int test_total = size.w * size.h;
        half *test_input, *test_output;
        allocateMemory(&test_input, &h_input, test_total);
        allocateMemory(&test_output, &h_output, test_total);

        initializeRandom(h_input, test_total, half(-1.0f), half(1.0f));
        copyToDevice(test_input, h_input, test_total);

        const int BLOCK_SIZE = 16;
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((size.w + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (size.h + BLOCK_SIZE - 1) / BLOCK_SIZE);

        output_input_transform_kernel<BLOCK_SIZE><<<grid, block>>>(
            test_output, test_input, size.w, size.h);
        CUDA_CHECK(cudaDeviceSynchronize());

        copyFromDevice(h_output, test_output, test_total);

        std::string filename = "output_input_transform_" + std::to_string(size.w) +
                               "x" + std::to_string(size.h) + ".bin";
        saveOutput(filename.c_str(), h_output, test_total);

        freeMemory(test_input, h_input);
        freeMemory(test_output, h_output);
    }
}