#pragma once

#include "performance_test.h"
#include "cuda_utils.h"

namespace cuda_perf {

// Initialization kernel
__global__ void init_arrays_kernel(double* a, double* b, double* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = 1.0;
        b[idx] = 2.0;
        c[idx] = 0.0;
    }
}

// STREAM kernel implementations
template<typename T>
__global__ void stream_copy_kernel(const T* __restrict__ src,
                                   T* __restrict__ dst,
                                   size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

template<typename T>
__global__ void stream_scale_kernel(const T* __restrict__ src,
                                    T* __restrict__ dst,
                                    T scalar,
                                    size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = scalar * src[idx];
    }
}

template<typename T>
__global__ void stream_add_kernel(const T* __restrict__ src1,
                                  const T* __restrict__ src2,
                                  T* __restrict__ dst,
                                  size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src1[idx] + src2[idx];
    }
}

template<typename T>
__global__ void stream_triad_kernel(const T* __restrict__ src1,
                                    const T* __restrict__ src2,
                                    T* __restrict__ dst,
                                    T scalar,
                                    size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src1[idx] + scalar * src2[idx];
    }
}

enum class StreamOperation {
    COPY,   // a = b
    SCALE,  // a = scalar * b
    ADD,    // a = b + c
    TRIAD   // a = b + scalar * c
};

class StreamBenchmark : public PerformanceTest {
public:
    StreamBenchmark(size_t array_size, StreamOperation op,
                    int block_size = 256)
        : array_size_(array_size)
        , operation_(op)
        , block_size_(block_size)
        , scalar_(3.5)
        , device_id_(0) {
    }

    std::string name() const override {
        switch (operation_) {
            case StreamOperation::COPY:  return "STREAM_Copy";
            case StreamOperation::SCALE: return "STREAM_Scale";
            case StreamOperation::ADD:   return "STREAM_Add";
            case StreamOperation::TRIAD: return "STREAM_Triad";
        }
        return "STREAM_Unknown";
    }

    std::string description() const override {
        std::ostringstream oss;
        oss << "STREAM " << name() << " operation\n";
        oss << "Array size: " << (array_size_ * sizeof(double) / 1024.0 / 1024.0)
            << " MB (" << array_size_ << " elements)\n";
        oss << "Block size: " << block_size_;
        return oss.str();
    }

    bool initialize() override {
        CUDA_CHECK(cudaSetDevice(device_id_));

        // Allocate device memory
        d_a_.resize(array_size_);
        d_b_.resize(array_size_);
        d_c_.resize(array_size_);

        if (!d_a_.is_valid() || !d_b_.is_valid() || !d_c_.is_valid()) {
            std::cerr << "Failed to allocate device memory" << std::endl;
            return false;
        }

        // Initialize arrays with some values
        init_arrays_kernel<<<(array_size_ + 255) / 256, 256>>>(
            d_a_.data(), d_b_.data(), d_c_.data(), array_size_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        return true;
    }

    void cleanup() override {
        d_a_.resize(0);
        d_b_.resize(0);
        d_c_.resize(0);
    }

    void run_warmup() override {
        execute_kernel();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    double run_single_test() override {
        Timer timer;
        execute_kernel();
        CUDA_CHECK(cudaDeviceSynchronize());
        return timer.elapsed_milliseconds();
    }

protected:
    void print_results(const PerformanceMetrics& metrics) override {
        double avg_time_sec = metrics.trimmed_mean() / 1000.0;
        size_t bytes_per_elem = get_bytes_per_element();
        double total_bytes = array_size_ * bytes_per_elem;
        double bandwidth_gb_s = total_bytes / avg_time_sec / (1024.0 * 1024.0 * 1024.0);

        std::cout << "\nResults:\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Average time:      " << metrics.trimmed_mean() << " ms\n";
        std::cout << "  Bandwidth:         " << std::setprecision(2)
                  << bandwidth_gb_s << " GB/s\n";
        std::cout << "  Bytes per element: " << bytes_per_elem << "\n";
        std::cout << "  Total bytes:       " << std::setprecision(2)
                  << (total_bytes / 1024.0 / 1024.0) << " MB\n";
        std::cout << "  Coeff. of variation: " << std::setprecision(2)
                  << (metrics.coefficient_of_variation() * 100.0) << "%\n";
    }

private:
    void execute_kernel() {
        int grid_size = (array_size_ + block_size_ - 1) / block_size_;
        dim3 grid(grid_size);
        dim3 block(block_size_);

        switch (operation_) {
            case StreamOperation::COPY:
                stream_copy_kernel<double><<<grid, block>>>(d_b_.data(), d_c_.data(), array_size_);
                break;
            case StreamOperation::SCALE:
                stream_scale_kernel<double><<<grid, block>>>(d_b_.data(), d_c_.data(),
                                                             scalar_, array_size_);
                break;
            case StreamOperation::ADD:
                stream_add_kernel<double><<<grid, block>>>(d_a_.data(), d_b_.data(),
                                                           d_c_.data(), array_size_);
                break;
            case StreamOperation::TRIAD:
                stream_triad_kernel<double><<<grid, block>>>(d_a_.data(), d_b_.data(),
                                                             d_c_.data(), scalar_, array_size_);
                break;
        }
        CUDA_CHECK(cudaGetLastError());
    }

    size_t get_bytes_per_element() const {
        switch (operation_) {
            case StreamOperation::COPY:
                return 2 * sizeof(double);  // read + write
            case StreamOperation::SCALE:
                return 2 * sizeof(double);  // read + write
            case StreamOperation::ADD:
                return 3 * sizeof(double);  // 2 reads + 1 write
            case StreamOperation::TRIAD:
                return 3 * sizeof(double);  // 2 reads + 1 write
        }
        return 0;
    }

    size_t array_size_;
    StreamOperation operation_;
    int block_size_;
    double scalar_;
    int device_id_;

    CudaDeviceBuffer<double> d_a_;
    CudaDeviceBuffer<double> d_b_;
    CudaDeviceBuffer<double> d_c_;
};

// Run all STREAM operations
class StreamBenchmarkSuite {
public:
    void execute(size_t array_size, const TestConfig& config = TestConfig()) {
        std::cout << "\n===================================================\n";
        std::cout << "STREAM Benchmark Suite\n";
        std::cout << "Array size: " << (array_size * sizeof(double) / 1024.0 / 1024.0)
                  << " MB (" << array_size << " elements)\n";
        std::cout << "===================================================\n\n";

        struct Result {
            std::string name;
            double bandwidth_gb_s;
            double avg_time_ms;
            double cv_percent;
        };

        std::vector<Result> results;

        // Run each operation
        std::vector<StreamOperation> ops = {
            StreamOperation::COPY,
            StreamOperation::SCALE,
            StreamOperation::ADD,
            StreamOperation::TRIAD
        };

        // Phase 1: Extended warmup for all operations to stabilize GPU state
        // This is critical when running after other tests (bandwidth, cache)
        const size_t EXTENDED_WARMUP = 10;  // Extra warmup iterations
        for (auto op : ops) {
            StreamBenchmark warmup_test(array_size, op);
            if (warmup_test.initialize()) {
                for (size_t i = 0; i < EXTENDED_WARMUP; ++i) {
                    warmup_test.run_warmup();
                }
                warmup_test.cleanup();
            }
        }

        // Small delay to let GPU frequency stabilize
        CUDA_CHECK(cudaDeviceSynchronize());

        // Phase 2: Actual measurements
        for (auto op : ops) {
            StreamBenchmark test(array_size, op);
            TestConfig test_config = config;
            test_config.verbose = false;

            if (!test.initialize()) {
                std::cerr << "Failed to initialize " << test.name() << std::endl;
                continue;
            }

            // Standard warmup before measurement
            for (size_t i = 0; i < test_config.warmup_iterations; ++i) {
                test.run_warmup();
            }

            PerformanceMetrics metrics;
            for (size_t i = 0; i < test_config.measurement_iterations; ++i) {
                double time_ms = test.run_single_test();
                metrics.add_sample(time_ms);
            }

            test.cleanup();

            double avg_time_sec = metrics.trimmed_mean() / 1000.0;
            size_t bytes_per_elem = 0;
            switch (op) {
                case StreamOperation::COPY:
                case StreamOperation::SCALE:
                    bytes_per_elem = 2 * sizeof(double);
                    break;
                case StreamOperation::ADD:
                case StreamOperation::TRIAD:
                    bytes_per_elem = 3 * sizeof(double);
                    break;
            }
            double total_bytes = array_size * bytes_per_elem;
            double bandwidth_gb_s = total_bytes / avg_time_sec / (1024.0 * 1024.0 * 1024.0);

            Result result;
            result.name = test.name();
            result.bandwidth_gb_s = bandwidth_gb_s;
            result.avg_time_ms = metrics.trimmed_mean();
            result.cv_percent = metrics.coefficient_of_variation() * 100.0;
            results.push_back(result);
        }

        // Print summary table
        std::cout << std::left << std::setw(15) << "Operation"
                  << std::right << std::setw(15) << "Bandwidth (GB/s)"
                  << std::setw(15) << "Time (ms)"
                  << std::setw(12) << "CV (%)\n";
        std::cout << std::string(57, '-') << "\n";

        for (const auto& result : results) {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << std::left << std::setw(15) << result.name;
            std::cout << std::right << std::setw(15) << result.bandwidth_gb_s;
            std::cout << std::setw(15) << result.avg_time_ms;
            std::cout << std::setw(12) << result.cv_percent << "\n";
        }
        std::cout << "\n";
    }
};

} // namespace cuda_perf
