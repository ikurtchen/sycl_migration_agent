#include <gtest/gtest.h>
#include "../common/base_test_fixture.h"

// Test kernel for FP16LayerNorm
__global__ void fp16_layer_norm_kernel(half* input, const half* gamma, const half* beta,
                                       float epsilon, int hidden_size, int batch_size) {
    int row_idx = blockIdx.x;  // Each block processes one row
    int tid = threadIdx.x;

    extern __shared__ float shared_mem[];

    if (row_idx >= batch_size) return;

    half* row_input = input + row_idx * hidden_size;

    // Convert to float for accuracy
    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        sum += __half2float(row_input[i]);
    }
    shared_mem[tid] = sum;
    __syncthreads();

    // Reduce sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    float mean = shared_mem[0] / hidden_size;

    // Compute variance
    float var = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float diff = __half2float(row_input[i]) - mean;
        var += diff * diff;
    }
    shared_mem[tid] = var;
    __syncthreads();

    // Reduce variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    float variance = shared_mem[0] / hidden_size;

    // Apply layer norm
    float std_dev = sqrtf(variance + epsilon);
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float normalized = (__half2float(row_input[i]) - mean) / std_dev;
        row_input[i] = __float2half(__half2float(gamma[i]) * normalized + __half2float(beta[i]));
    }
}

// Float version for comparison
__global__ void layer_norm_kernel(float* input, const float* gamma, const float* beta,
                                  float epsilon, int hidden_size, int batch_size) {
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared_mem[];

    if (row_idx >= batch_size) return;

    float* row_input = input + row_idx * hidden_size;

    // Compute mean
    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        sum += row_input[i];
    }
    shared_mem[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    float mean = shared_mem[0] / hidden_size;

    // Compute variance
    float var = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float diff = row_input[i] - mean;
        var += diff * diff;
    }
    shared_mem[tid] = var;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    float variance = shared_mem[0] / hidden_size;
    float std_dev = sqrtf(variance + epsilon);

    // Apply layer norm
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float normalized = (row_input[i] - mean) / std_dev;
        row_input[i] = gamma[i] * normalized + beta[i];
    }
}

void launchFp16LayerNorm(half* input, const half* gamma, const half* beta,
                         float epsilon, int hidden_size, int batch_size);

class Fp16LayerNormTest : public FP16Test {
protected:
    void SetUp() override {
        FP16Test::SetUp();

        // Test dimensions
        batch_size = 32;
        hidden_size = 768;  // Common transformer hidden size
        total_size = batch_size * hidden_size;
        epsilon = 1e-6f;

        // Allocate memory
        allocateMemory(&d_input, &h_input, total_size);
        allocateMemory(&d_gamma, &h_gamma, hidden_size);
        allocateMemory(&d_beta, &h_beta, hidden_size);

        // Initialize test data
        initializeRandom(h_input, total_size, half(-2.0f), half(2.0f));
        initializeRandom(h_gamma, hidden_size, half(0.5f), half(1.5f));
        initializeRandom(h_beta, hidden_size, half(-0.5f), half(0.5f));

        // Copy to device
        copyToDevice(d_input, h_input, total_size);
        copyToDevice(d_gamma, h_gamma, hidden_size);
        copyToDevice(d_beta, h_beta, hidden_size);
    }

    void TearDown() override {
        FP16Test::TearDown();

        freeMemory(d_input, h_input);
        freeMemory(d_gamma, h_gamma);
        freeMemory(d_beta, h_beta);
    }

    int batch_size, hidden_size, total_size;
    float epsilon;
    half *d_input, *d_gamma, *d_beta;
    half *h_input, *h_gamma, *h_beta;
};

TEST_F(Fp16LayerNormTest, Correctness) {
    // Create float version for reference
    float *f_input = new float[total_size];
    float *f_gamma = new float[hidden_size];
    float *f_beta = new float[hidden_size];
    float *f_input_orig = new float[total_size];

    // Convert to float
    for (int i = 0; i < total_size; ++i) {
        f_input[i] = f_input_orig[i] = __half2float(h_input[i]);
    }
    for (int i = 0; i < hidden_size; ++i) {
        f_gamma[i] = __half2float(h_gamma[i]);
        f_beta[i] = __half2float(h_beta[i]);
    }

    // Launch FP16 kernel
    int block_size = 256;
    int shared_mem_size = block_size * sizeof(float);
    dim3 grid(batch_size);

    fp16_layer_norm_kernel<<<grid, block_size, shared_mem_size>>>(
        d_input, d_gamma, d_beta, epsilon, hidden_size, batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_input, d_input, total_size);

    // Launch float version for reference
    float *d_f_input, *d_f_gamma, *d_f_beta;
    CUDA_CHECK(cudaMalloc(&d_f_input, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_f_gamma, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_f_beta, hidden_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_f_input, f_input, total_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_gamma, f_gamma, hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f_beta, f_beta, hidden_size * sizeof(float), cudaMemcpyHostToDevice));

    layer_norm_kernel<<<grid, block_size, shared_mem_size>>>(
        d_f_input, d_f_gamma, d_f_beta, epsilon, hidden_size, batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(f_input, d_f_input, total_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare results
    for (int i = 0; i < total_size; ++i) {
        float fp16_result = __half2float(h_input[i]);
        float fp32_result = f_input[i];
        EXPECT_NEAR(fp16_result, fp32_result, 1e-3f) << "LayerNorm mismatch at index " << i;
    }

    // Check normalization properties for first row
    half* first_row = h_input;
    float row_mean = 0.0f, row_var = 0.0f;
    for (int i = 0; i < hidden_size; ++i) {
        float val = __half2float(first_row[i]);
        row_mean += val;
    }
    row_mean /= hidden_size;

    for (int i = 0; i < hidden_size; ++i) {
        float val = __half2float(first_row[i]);
        row_var += (val - row_mean) * (val - row_mean);
    }
    row_var /= hidden_size;

    EXPECT_NEAR(row_mean, 0.0f, 1e-3f) << "Mean should be normalized to 0";
    EXPECT_NEAR(row_var, 1.0f, 1e-2f) << "Variance should be normalized to 1";

    // Save outputs
    saveOutput("fp16_layer_norm_output.bin", h_input, total_size);
    saveOutput("fp16_layer_norm_input.bin", f_input_orig, total_size);

    // Cleanup
    delete[] f_input;
    delete[] f_gamma;
    delete[] f_beta;
    delete[] f_input_orig;
    CUDA_CHECK(cudaFree(d_f_input));
    CUDA_CHECK(cudaFree(d_f_gamma));
    CUDA_CHECK(cudaFree(d_f_beta));
}

TEST_F(Fp16LayerNormTest, Benchmark) {
    int block_size = 256;
    int shared_mem_size = block_size * sizeof(float);
    dim3 grid(batch_size);

    auto kernel_func = [&]() {
        fp16_layer_norm_kernel<<<grid, block_size, shared_mem_size>>>(
            d_input, d_gamma, d_beta, epsilon, hidden_size, batch_size);
    };

    BenchmarkResult result = benchmarkKernel(
        kernel_func, total_size, "fp16_layer_norm", 10, 100);

    std::cout << "FP16 LayerNorm - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/fp16_layer_norm_benchmark.json");
}

TEST_F(Fp16LayerNormTest, DifferentConfigurations) {
    struct Config {
        int batch, hidden;
    };

    std::vector<Config> configs = {
        {8, 512},
        {16, 768},
        {32, 1024},
        {64, 1536}
    };

    for (auto config : configs) {
        int test_total = config.batch * config.hidden;

        half *test_input, *test_gamma, *test_beta;
        allocateMemory(&test_input, &h_input, test_total);
        allocateMemory(&test_gamma, &h_gamma, config.hidden);
        allocateMemory(&test_beta, &h_beta, config.hidden);

        initializeRandom(h_input, test_total, half(-2.0f), half(2.0f));
        initializeRandom(h_gamma, config.hidden, half(0.5f), half(1.5f));
        initializeRandom(h_beta, config.hidden, half(-0.5f), half(0.5f));

        copyToDevice(test_input, h_input, test_total);
        copyToDevice(test_gamma, h_gamma, config.hidden);
        copyToDevice(test_beta, h_beta, config.hidden);

        int block_size = 256;
        int shared_mem_size = block_size * sizeof(float);
        dim3 test_grid(config.batch);

        fp16_layer_norm_kernel<<<test_grid, block_size, shared_mem_size>>>(
            test_input, test_gamma, test_beta, epsilon, config.hidden, config.batch);
        CUDA_CHECK(cudaDeviceSynchronize());

        copyFromDevice(h_input, test_input, test_total);

        // Check first row normalization
        half* first_row = h_input;
        float row_mean = 0.0f;
        for (int i = 0; i < config.hidden; ++i) {
            row_mean += __half2float(first_row[i]);
        }
        row_mean /= config.hidden;

        EXPECT_NEAR(row_mean, 0.0f, 1e-2f)
            << "Config " << config.batch << "x" << config.hidden << " mean not normalized";

        std::string filename = "fp16_layer_norm_" + std::to_string(config.batch) +
                               "x" + std::to_string(config.hidden) + ".bin";
        saveOutput(filename.c_str(), h_input, test_total);

        freeMemory(test_input, h_input);
        freeMemory(test_gamma, h_gamma);
        freeMemory(test_beta, h_beta);
    }
}