#pragma once

#include "cuda_utils.h"
#include "performance_test.h"
#include "gpu_runtime.h"
#include <iomanip>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <utility>

namespace cuda_perf {

// ============================================================
// Timing Statistics Collector
// ============================================================

class ExecutionTimer {
public:
    void add_sample(double seconds) { samples_.push_back(seconds); }

    // Trimmed mean (discard min and max)
    double trimmed_mean() const {
        if (samples_.empty()) return 0.0;
        if (samples_.size() == 1) return samples_[0];
        if (samples_.size() == 2) return (samples_[0] + samples_[1]) / 2.0;

        std::vector<double> sorted = samples_;
        std::sort(sorted.begin(), sorted.end());
        return std::accumulate(sorted.begin() + 1, sorted.end() - 1, 0.0) / (sorted.size() - 2);
    }

    double minimum() const {
        if (samples_.empty()) return 0.0;
        return *std::min_element(samples_.begin(), samples_.end());
    }

    double maximum() const {
        if (samples_.empty()) return 0.0;
        return *std::max_element(samples_.begin(), samples_.end());
    }

    double spread() const {
        double avg = trimmed_mean();
        if (avg == 0.0) return 0.0;
        return (maximum() - minimum()) / avg;
    }

    size_t count() const { return samples_.size(); }

private:
    std::vector<double> samples_;
};

// ============================================================
// L1 Cache Implementation
// ============================================================

namespace l1_cache_impl {

using data_type = float;
data_type* device_array_A;
data_type* device_array_B;

__global__ void data_initializer(data_type* buffer, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < count; i += blockDim.x * gridDim.x) {
        buffer[i] = 2.71828f;
    }
}

template <int ARRAY_N, int TOTAL_ITERS, int BLOCK_DIM>
__global__ void computation_kernel(data_type* __restrict__ output_ptr,
                                   const data_type* __restrict__ input_ptr,
                                   int param_offset) {
    data_type local_sum = 0.0f;

    input_ptr += threadIdx.x;

    for (int iter = 0; iter < TOTAL_ITERS; iter++) {
        input_ptr += param_offset;
        const data_type* second_ptr = input_ptr + ARRAY_N;

        for (int i = 0; i < ARRAY_N; i += BLOCK_DIM) {
            local_sum += input_ptr[i] * second_ptr[i];
        }

        local_sum *= 3.14159f;
    }

    if (local_sum == 9999.0f)
        output_ptr[threadIdx.x] += local_sum;
}

template <int ARRAY_N, int TOTAL_ITERS, int BLOCK_SIZE>
__host__ void launch_kernel(int grid_count) {
    computation_kernel<ARRAY_N, TOTAL_ITERS, BLOCK_SIZE><<<grid_count, BLOCK_SIZE>>>(
        device_array_A, device_array_B, 0);
}

template <int ARRAY_N, int BLOCK_SIZE>
void run_measurement() {
    const size_t repeat_count = 1000000000ULL / ARRAY_N + 2;

    cudaDeviceProp props;
    int dev_id;
    CUDA_CHECK(cudaGetDevice(&dev_id));
    CUDA_CHECK(cudaGetDeviceProperties(&props, dev_id));
    int grid_count = props.multiProcessorCount * 1;

    ExecutionTimer timer;

    for (int sample = 0; sample < 15; sample++) {
        size_t buffer_elems = 2 * ARRAY_N + sample * 1282;

        CUDA_CHECK(cudaMalloc(&device_array_A, buffer_elems * sizeof(data_type)));
        data_initializer<<<52, 256>>>(device_array_A, buffer_elems);
        CUDA_CHECK(cudaMalloc(&device_array_B, buffer_elems * sizeof(data_type)));
        data_initializer<<<52, 256>>>(device_array_B, buffer_elems);

        device_array_A += sample;
        device_array_B += sample;

        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start_ev, stop_ev;
        CUDA_CHECK(cudaEventCreate(&start_ev));
        CUDA_CHECK(cudaEventCreate(&stop_ev));

        CUDA_CHECK(cudaEventRecord(start_ev));
        launch_kernel<ARRAY_N, repeat_count, BLOCK_SIZE>(grid_count);
        CUDA_CHECK(cudaEventRecord(stop_ev));
        CUDA_CHECK(cudaEventSynchronize(stop_ev));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_ev, stop_ev));
        timer.add_sample(ms / 1000.0);

        CUDA_CHECK(cudaEventDestroy(start_ev));
        CUDA_CHECK(cudaEventDestroy(stop_ev));
        CUDA_CHECK(cudaFree(device_array_A - sample));
        CUDA_CHECK(cudaFree(device_array_B - sample));
    }

    double data_volume = 2.0 * ARRAY_N * sizeof(data_type);
    double bandwidth_gbps = data_volume * grid_count * repeat_count / timer.minimum() / 1.0e9;

    std::cout << std::fixed << std::setprecision(0);
    std::cout << std::setw(10) << (data_volume / 1024.0) << " kB";
    std::cout << std::setw(10) << (timer.trimmed_mean() * 1000.0) << "ms";
    std::cout << std::setprecision(1) << std::setw(10) << (timer.spread() * 100.0) << "%";
    std::cout << std::setprecision(1) << std::setw(10) << bandwidth_gbps << " GB/s";
    std::cout << "\n";
}

} // namespace l1_cache_impl

// ============================================================
// Helper to run template measurements at compile time
// ============================================================

namespace detail {
template <typename Func, std::size_t... Is>
constexpr void for_each_index(std::index_sequence<Is...>, Func&& func) {
    (func(std::integral_constant<std::size_t, Is>{}), ...);
}

template <std::size_t N, typename Func>
constexpr void for_each_index(Func&& func) {
    for_each_index(std::make_index_sequence<N>{}, std::forward<Func>(func));
}
} // namespace detail

// ============================================================
// L1 Cache Sweep Test
// ============================================================

class L1CacheSweepTest {
public:
    void execute(const TestConfig& config = TestConfig()) {
        std::cout << "\n===================================================\n";
        std::cout << "L1 Cache Bandwidth Sweep Test\n";
        std::cout << "===================================================\n\n";

        std::cout << std::left << std::setw(13) << "data set"
                  << std::right << std::setw(12) << "exec time"
                  << std::setw(11) << "spread"
                  << std::setw(15) << "Eff. bw\n";
        std::cout << std::string(51, '-') << "\n";

        // Run all test configurations using compile-time loop
        // Initial sizes
        l1_cache_impl::run_measurement<128, 128>();
        l1_cache_impl::run_measurement<256, 256>();
        l1_cache_impl::run_measurement<512, 512>();
        l1_cache_impl::run_measurement<3 * 256, 256>();

        // Powers of 512 from 1x to 16x
        detail::for_each_index<16>([](auto index) {
            constexpr std::size_t multiplier = index + 1;
            l1_cache_impl::run_measurement<multiplier * 512, 512>();
        });

        std::cout << "\n";
    }
};

// ============================================================
// L2 Cache Implementation
// ============================================================

namespace l2_cache_impl {

using data_type = double;
data_type* device_buffer_X;
data_type* device_buffer_Y;

__global__ void data_init_double(data_type* buffer, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < count; i += blockDim.x * gridDim.x) {
        buffer[i] = 1.41421;
    }
}

template <int ARRAY_N, int BLOCK_DIM>
__global__ void l2_computation_kernel(data_type* __restrict__ output_ptr,
                                     const data_type* __restrict__ input_ptr,
                                     int mult_param) {
    data_type accumulator = 0.0;

    for (int iter = 0; iter < ARRAY_N / 2; iter++) {
        int calc_idx =
            (blockDim.x * mult_param * iter + (blockIdx.x % mult_param) * BLOCK_DIM) * 2 +
            threadIdx.x;
        accumulator += input_ptr[calc_idx] * input_ptr[calc_idx + BLOCK_DIM];
    }

    accumulator *= 2.23606;

    if (threadIdx.x > 7777 || accumulator == 99.99)
        output_ptr[threadIdx.x] += accumulator;
}

template <int ARRAY_N>
void run_l2_measurement(int block_multiplier) {
    constexpr int BLOCK_DIM = 1024;
    constexpr int GRID_COUNT = 200000;

    ExecutionTimer timer;

    for (int sample = 0; sample < 11; sample++) {
        size_t buf_elems = block_multiplier * BLOCK_DIM * ARRAY_N + sample * 128;

        CUDA_CHECK(cudaMalloc(&device_buffer_X, buf_elems * sizeof(data_type)));
        data_init_double<<<52, 256>>>(device_buffer_X, buf_elems);
        CUDA_CHECK(cudaMalloc(&device_buffer_Y, buf_elems * sizeof(data_type)));
        data_init_double<<<52, 256>>>(device_buffer_Y, buf_elems);

        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start_ev, stop_ev;
        CUDA_CHECK(cudaEventCreate(&start_ev));
        CUDA_CHECK(cudaEventCreate(&stop_ev));

        CUDA_CHECK(cudaEventRecord(start_ev));
        l2_computation_kernel<ARRAY_N, BLOCK_DIM><<<GRID_COUNT, BLOCK_DIM>>>(
            device_buffer_X, device_buffer_Y, block_multiplier);
        CUDA_CHECK(cudaEventRecord(stop_ev));
        CUDA_CHECK(cudaEventSynchronize(stop_ev));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_ev, stop_ev));
        timer.add_sample(ms / 1000.0);

        CUDA_CHECK(cudaEventDestroy(start_ev));
        CUDA_CHECK(cudaEventDestroy(stop_ev));
        CUDA_CHECK(cudaFree(device_buffer_X));
        CUDA_CHECK(cudaFree(device_buffer_Y));
    }

    double data_vol = ARRAY_N * BLOCK_DIM * sizeof(data_type);
    double bandwidth_gbps = data_vol * GRID_COUNT / timer.minimum() / 1.0e9;

    std::cout << std::fixed << std::setprecision(0);
    std::cout << std::setw(10) << (data_vol / 1024.0) << " kB";
    std::cout << std::setw(10) << (data_vol * block_multiplier / 1024.0) << " kB";
    std::cout << std::setw(10) << (timer.trimmed_mean() * 1000.0) << "ms";
    std::cout << std::setprecision(1) << std::setw(10) << (timer.spread() * 100.0) << "%";
    std::cout << std::setprecision(1) << std::setw(10) << bandwidth_gbps << " GB/s";
    std::cout << "\n";
}

} // namespace l2_cache_impl

// ============================================================
// L2 Cache Sweep Test
// ============================================================

class L2CacheSweepTest {
public:
    void execute(const TestConfig& config = TestConfig()) {
        std::cout << "\n===================================================\n";
        std::cout << "L2 Cache Bandwidth Sweep Test\n";
        std::cout << "===================================================\n\n";

        std::cout << std::left << std::setw(13) << "data set"
                  << std::setw(12) << "exec data"
                  << std::right << std::setw(12) << "exec time"
                  << std::setw(11) << "spread"
                  << std::setw(15) << "Eff. bw\n";
        std::cout << std::string(63, '-') << "\n";

        constexpr int ARRAY_N = 64;

        for (int i = 3; i < 10000; i += std::max(1, static_cast<int>(i * 0.1))) {
            l2_cache_impl::run_l2_measurement<ARRAY_N>(i);
        }

        std::cout << "\n";
    }
};

// ============================================================
// Unified Interface
// ============================================================

enum class CacheTarget { L1, L2 };

class CacheBenchmarkSuite {
public:
    CacheBenchmarkSuite(CacheTarget target) : target_(target) {}

    void execute(const TestConfig& config = TestConfig()) {
        if (target_ == CacheTarget::L1) {
            L1CacheSweepTest test;
            test.execute(config);
        } else {
            L2CacheSweepTest test;
            test.execute(config);
        }
    }

private:
    CacheTarget target_;
};

} // namespace cuda_perf
