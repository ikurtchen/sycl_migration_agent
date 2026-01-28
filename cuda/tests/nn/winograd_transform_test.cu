#include <gtest/gtest.h>
#include "../common/base_test_fixture.h"

// Winograd transform F(2x2, 3x3) transforms
// Input transform: G * g * G^T
__global__ void winograd_input_transform_kernel(const float* input, float* output,
                                               int batch_size, int channels,
                                               int height, int width) {
    // Each thread block processes one tile
    int tile_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int batch_idx = blockIdx.z;

    if (batch_idx >= batch_size || channel_idx >= channels) return;

    // Calculate tile position
    int tiles_x = (width + 1) / 2;
    int tiles_y = (height + 1) / 2;
    int tile_x = tile_idx % tiles_x;
    int tile_y = tile_idx / tiles_x;

    // Original input indices
    int base_x = tile_x * 2;
    int base_y = tile_y * 2;

    // G matrix for Winograd F(2x2, 3x3)
    // G = [[1, 0, -1, 0],
    //      [0, 1, 1, 0],
    //      [0, -1, 1, 0],
    //      [1, 0, 1, 0]]

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    // Load 4x4 patch
    float patch[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            int y = base_y - 1 + i;
            int x = base_x - 1 + j;

            if (x >= 0 && x < width && y >= 0 && y < height) {
                int input_idx = batch_idx * channels * height * width +
                               channel_idx * height * width + y * width + x;
                patch[i][j] = input[input_idx];
            } else {
                patch[i][j] = 0.0f;
            }
        }
    }

    // Apply Winograd transform: output[ty][tx] = sum_i sum_j G[ty][i] * patch[i][j] * G[tx][j]
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float g_elem = 0.0f;

            // G matrix elements
            if (ty == 0) {
                if (i == 0) g_elem = 1.0f;
                else if (i == 2) g_elem = -1.0f;
                else g_elem = 0.0f;
            } else if (ty == 1) {
                if (i == 1) g_elem = 1.0f;
                else if (i == 2) g_elem = 1.0f;
                else g_elem = 0.0f;
            } else if (ty == 2) {
                if (i == 1) g_elem = -1.0f;
                else if (i == 2) g_elem = 1.0f;
                else g_elem = 0.0f;
            } else if (ty == 3) {
                if (i == 0) g_elem = 1.0f;
                else if (i == 2) g_elem = 1.0f;
                else g_elem = 0.0f;
            }

            float g_T_elem = 0.0f;
            if (tx == 0) {
                if (j == 0) g_T_elem = 1.0f;
                else if (j == 2) g_T_elem = -1.0f;
                else g_T_elem = 0.0f;
            } else if (tx == 1) {
                if (j == 1) g_T_elem = 1.0f;
                else if (j == 2) g_T_elem = 1.0f;
                else g_T_elem = 0.0f;
            } else if (tx == 2) {
                if (j == 1) g_T_elem = -1.0f;
                else if (j == 2) g_T_elem = 1.0f;
                else g_T_elem = 0.0f;
            } else if (tx == 3) {
                if (j == 0) g_T_elem = 1.0f;
                else if (j == 2) g_T_elem = 1.0f;
                else g_T_elem = 0.0f;
            }

            sum += g_elem * patch[i][j] * g_T_elem;
        }
    }

    // Store result
    int tiles_per_channel = tiles_x * tiles_y;
    int output_idx = batch_idx * channels * tiles_per_channel * 16 +
                     channel_idx * tiles_per_channel * 16 +
                     tile_idx * 16 + ty * 4 + tx;
    output[output_idx] = sum;
}

// Output transform: A^T * (U ⊙ V) * A
__global__ void winograd_output_transform_kernel(const float* input, float* output,
                                                int batch_size, int channels,
                                                int height, int width) {
    // Each thread block processes one output tile
    int tile_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int batch_idx = blockIdx.z;

    if (batch_idx >= batch_size || channel_idx >= channels) return;

    // Calculate tile position
    int tiles_x = (width + 1) / 2;
    int tiles_y = (height + 1) / 2;
    int tile_x = tile_idx % tiles_x;
    int tile_y = tile_idx / tiles_x;

    // Output position
    int out_y = tile_y * 2;
    int out_x = tile_x * 2;

    // Load transformed patch (4x4)
    float patch[4][4];
    int tiles_per_channel = tiles_x * tiles_y;
    int base_idx = batch_idx * channels * tiles_per_channel * 16 +
                   channel_idx * tiles_per_channel * 16 +
                   tile_idx * 16;

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            patch[i][j] = input[base_idx + i * 4 + j];
        }
    }

    // Apply output transform
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (ty < 2 && tx < 2) {
        // A matrix for Winograd F(2x2, 3x3)
        // A = [[1, 0, 1, 0],
        //      [0, 1, 1, 0]]

        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            float a_T_elem = 0.0f;
            if (ty == 0) {
                if (i == 0) a_T_elem = 1.0f;
                else if (i == 2) a_T_elem = 1.0f;
                else a_T_elem = 0.0f;
            } else if (ty == 1) {
                if (i == 1) a_T_elem = 1.0f;
                else if (i == 2) a_T_elem = 1.0f;
                else a_T_elem = 0.0f;
            }

            float inner_sum = 0.0f;
            for (int j = 0; j < 4; ++j) {
                float a_elem = 0.0f;
                if (tx == 0) {
                    if (j == 0) a_elem = 1.0f;
                    else if (j == 2) a_elem = 1.0f;
                    else a_elem = 0.0f;
                } else if (tx == 1) {
                    if (j == 1) a_elem = 1.0f;
                    else if (j == 2) a_elem = 1.0f;
                    else a_elem = 0.0f;
                }

                inner_sum += patch[i][j] * a_elem;
            }
            sum += a_T_elem * inner_sum;
        }

        // Store result
        if (out_x + tx < width && out_y + ty < height) {
            int output_idx = batch_idx * channels * height * width +
                           channel_idx * height * width +
                           (out_y + ty) * width + (out_x + tx);
            output[output_idx] = sum;
        }
    }
}

void launchWinogradInputTransform(const float* input, float* output,
                                 int batch_size, int channels, int height, int width);
void launchWinogradOutputTransform(const float* input, float* output,
                                  int batch_size, int channels, int height, int width);

class WinogradTransformTest : public BaseCudaTest<float> {
protected:
    void SetUp() override {
        BaseCudaTest<float>::SetUp();

        // Test dimensions
        batch_size = 4;
        channels = 32;
        height = 32;
        width = 32;

        // Calculate transformed dimensions
        tiles_x = (width + 1) / 2;
        tiles_y = (height + 1) / 2;
        tiles_per_channel = tiles_x * tiles_y;
        transformed_size = batch_size * channels * tiles_per_channel * 16;
        input_size = batch_size * channels * height * width;

        // Allocate memory
        allocateMemory(&d_input, &h_input, input_size);
        allocateMemory(&d_output, &h_output, input_size);
        allocateMemory(&d_transformed, &h_transformed, transformed_size);

        // Initialize test data
        initializeRandom(h_input, input_size, -1.0f, 1.0f);

        // Copy to device
        copyToDevice(d_input, h_input, input_size);
    }

    void TearDown() override {
        BaseCudaTest<float>::TearDown();

        freeMemory(d_input, h_input);
        freeMemory(d_output, h_output);
        freeMemory(d_transformed, h_transformed);
    }

    int batch_size, channels, height, width;
    int tiles_x, tiles_y, tiles_per_channel;
    int input_size, transformed_size;
    float *d_input, *d_output, *d_transformed;
    float *h_input, *h_output, *h_transformed;
};

TEST_F(WinogradTransformTest, InputTransformCorrectness) {
    // Launch input transform
    dim3 block(4, 4);
    dim3 grid(tiles_per_channel, channels, batch_size);

    winograd_input_transform_kernel<<<grid, block>>>(
        d_input, d_transformed, batch_size, channels, height, width);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy transformed data back
    copyFromDevice(h_transformed, d_transformed, transformed_size);

    // Check that transformed data is not all zero
    int non_zero_count = 0;
    for (int i = 0; i < min(1000, transformed_size); ++i) {
        if (abs(h_transformed[i]) > 1e-6f) {
            non_zero_count++;
        }
    }
    EXPECT_GT(non_zero_count, 0) << "Transformed data should not be all zero";

    // Check for reasonable values
    for (int i = 0; i < min(1000, transformed_size); ++i) {
        EXPECT_LT(abs(h_transformed[i]), 100.0f) << "Transformed values should be reasonable at index " << i;
        EXPECT_FALSE(isnan(h_transformed[i])) << "NaN detected in transformed data at index " << i;
        EXPECT_FALSE(isinf(h_transformed[i])) << "Inf detected in transformed data at index " << i;
    }

    // Save outputs
    saveOutput("winograd_input_transform.bin", h_transformed, transformed_size);
    saveInput("winograd_input.bin", h_input, input_size);
}

TEST_F(WinogradTransformTest, OutputTransformCorrectness) {
    // First run input transform
    dim3 block(4, 4);
    dim3 grid(tiles_per_channel, channels, batch_size);

    winograd_input_transform_kernel<<<grid, block>>>(
        d_input, d_transformed, batch_size, channels, height, width);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Then run output transform
    winograd_output_transform_kernel<<<grid, block>>>(
        d_transformed, d_output, batch_size, channels, height, width);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_output, d_output, input_size);

    // Check that output is not all zero
    int non_zero_count = 0;
    for (int i = 0; i < min(10000, input_size); ++i) {
        if (abs(h_output[i]) > 1e-6f) {
            non_zero_count++;
        }
    }
    EXPECT_GT(non_zero_count, 0) << "Output should not be all zero";

    // Check for reasonable values
    for (int i = 0; i < min(10000, input_size); ++i) {
        EXPECT_LT(abs(h_output[i]), 1000.0f) << "Output values should be reasonable at index " << i;
        EXPECT_FALSE(isnan(h_output[i])) << "NaN detected in output at index " << i;
        EXPECT_FALSE(isinf(h_output[i])) << "Inf detected in output at index " << i;
    }

    // Save outputs
    saveOutput("winograd_output_transform.bin", h_output, input_size);
}

TEST_F(WinogradTransformTest, EndToEndCorrectness) {
    // Run complete transform pipeline
    launchWinogradInputTransform(d_input, d_transformed, batch_size, channels, height, width);
    CUDA_CHECK(cudaDeviceSynchronize());

    launchWinogradOutputTransform(d_transformed, d_output, batch_size, channels, height, width);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_output, d_output, input_size);

    // Basic sanity checks
    float min_val = *std::min_element(h_output, h_output + input_size);
    float max_val = *std::max_element(h_output, h_output + input_size);
    float avg_val = 0.0f;
    for (int i = 0; i < input_size; ++i) {
        avg_val += h_output[i];
    }
    avg_val /= input_size;

    EXPECT_lt(abs(avg_val), 10.0f) << "Average value should be reasonable";
    EXPECT_LT(abs(max_val), 100.0f) << "Maximum value should be reasonable";
    EXPECT_LT(abs(min_val), 100.0f) << "Minimum value should be reasonable";

    // Save final output
    saveOutput("winograd_transform_final.bin", h_output, input_size);
}

TEST_F(WinogradTransformTest, BenchmarkInputTransform) {
    dim3 block(4, 4);
    dim3 grid(tiles_per_channel, channels, batch_size);

    auto kernel_func = [&]() {
        winograd_input_transform_kernel<<<grid, block>>>(
            d_input, d_transformed, batch_size, channels, height, width);
    };

    // Calculate FLOPs (simplified)
    size_t flops_per_tile = 4 * 4 * 16;  // Rough estimate
    size_t total_flops = batch_size * channels * tiles_per_channel * flops_per_tile;

    BenchmarkResult result = benchmarkKernel(
        kernel_func, total_flops / 2, "winograd_input_transform", 5, 50);

    std::cout << "Winograd Input Transform - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/winograd_input_transform_benchmark.json");
}

TEST_F(WinogradTransformTest, BenchmarkOutputTransform) {
    // First run input transform
    dim3 block(4, 4);
    dim3 grid(tiles_per_channel, channels, batch_size);

    winograd_input_transform_kernel<<<grid, block>>>(
        d_input, d_transformed, batch_size, channels, height, width);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto kernel_func = [&]() {
        winograd_output_transform_kernel<<<grid, block>>>(
            d_transformed, d_output, batch_size, channels, height, width);
    };

    size_t flops_per_tile = 2 * 2 * 16;  // Rough estimate
    size_t total_flops = batch_size * channels * tiles_per_channel * flops_per_tile;

    BenchmarkResult result = benchmarkKernel(
        kernel_func, total_flops / 2, "winograd_output_transform", 5, 50);

    std::cout << "Winograd Output Transform - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/winograd_output_transform_benchmark.json");
}

TEST_F(WinogradTransformTest, DifferentSizes) {
    struct TestSize {
        int h, w;
    };

    std::vector<TestSize> test_sizes = {
        {16, 16}, {64, 64}, {128, 128}
    };

    for (auto test_size : test_sizes) {
        int test_tiles_x = (test_size.w + 1) / 2;
        int test_tiles_y = (test_size.h + 1) / 2;
        int test_tiles_per_channel = test_tiles_x * test_tiles_y;
        int test_input_size = batch_size * channels * test_size.h * test_size.w;
        int test_transformed_size = batch_size * channels * test_tiles_per_channel * 16;

        float *test_input, *test_transformed, *test_output;
        allocateMemory(&test_input, &h_input, test_input_size);
        allocateMemory(&test_transformed, &h_transformed, test_transformed_size);
        allocateMemory(&test_output, &h_output, test_input_size);

        initializeRandom(h_input, test_input_size, -1.0f, 1.0f);
        copyToDevice(test_input, h_input, test_input_size);

        dim3 block(4, 4);
        dim3 test_grid(test_tiles_per_channel, channels, batch_size);

        // Input transform
        winograd_input_transform_kernel<<<test_grid, block>>>(
            test_input, test_transformed, batch_size, channels, test_size.h, test_size.w);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Output transform
        winograd_output_transform_kernel<<<test_grid, block>>>(
            test_transformed, test_output, batch_size, channels, test_size.h, test_size.w);
        CUDA_CHECK(cudaDeviceSynchronize());

        copyFromDevice(h_output, test_output, test_input_size);

        // Basic check
        int non_zero_count = 0;
        for (int i = 0; i < min(10000, test_input_size); ++i) {
            if (abs(h_output[i]) > 1e-6f) {
                non_zero_count++;
            }
        }
        EXPECT_GT(non_zero_count, 0) << "Size " << test_size.h << "x" << test_size.w << " should have non-zero output";

        std::string filename = "winograd_transform_" + std::to_string(test_size.h) +
                               "x" + std::to_string(test_size.w) + ".bin";
        saveOutput(filename.c_str(), h_output, test_input_size);

        freeMemory(test_input, h_input);
        freeMemory(test_transformed, h_transformed);
        freeMemory(test_output, h_output);
    }
}

// Dummy implementations of launch functions for test linking
void launchWinogradInputTransform(const float* input, float* output,
                                 int batch_size, int channels, int height, int width) {
    int tiles_x = (width + 1) / 2;
    int tiles_y = (height + 1) / 2;
    dim3 block(4, 4);
    dim3 grid(tiles_x * tiles_y, channels, batch_size);

    winograd_input_transform_kernel<<<grid, block>>>(input, output, batch_size, channels, height, width);
}

void launchWinogradOutputTransform(const float* input, float* output,
                                  int batch_size, int channels, int height, int width) {
    int tiles_x = (width + 1) / 2;
    int tiles_y = (height + 1) / 2;
    dim3 block(4, 4);
    dim3 grid(tiles_x * tiles_y, channels, batch_size);

    winograd_output_transform_kernel<<<grid, block>>>(input, output, batch_size, channels, height, width);
}