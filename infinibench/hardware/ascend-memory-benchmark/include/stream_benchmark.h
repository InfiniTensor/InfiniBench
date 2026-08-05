#pragma once

#include "acl_utils.h"

#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_mul.h>

#include <limits>

namespace npu_perf {

class AclTensorDescriptor {
public:
    AclTensorDescriptor(void* data, size_t element_count) {
        if (element_count >
            static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
            throw std::runtime_error("STREAM array is too large");
        }

        dims_[0] = static_cast<int64_t>(element_count);
        tensor_ = aclCreateTensor(dims_, 1, ACL_FLOAT, strides_, 0,
                                  ACL_FORMAT_ND, dims_, 1, data);
        if (tensor_ == nullptr) {
            throw std::runtime_error("aclCreateTensor failed");
        }
    }

    ~AclTensorDescriptor() {
        if (tensor_ != nullptr) {
            aclDestroyTensor(tensor_);
        }
    }

    AclTensorDescriptor(const AclTensorDescriptor&) = delete;
    AclTensorDescriptor& operator=(const AclTensorDescriptor&) = delete;

    aclTensor* get() const { return tensor_; }

private:
    int64_t dims_[1] = {0};
    int64_t strides_[1] = {1};
    aclTensor* tensor_ = nullptr;
};

class AclScalarDescriptor {
public:
    explicit AclScalarDescriptor(float value) : value_(value) {
        scalar_ = aclCreateScalar(&value_, ACL_FLOAT);
        if (scalar_ == nullptr) {
            throw std::runtime_error("aclCreateScalar failed");
        }
    }

    ~AclScalarDescriptor() {
        if (scalar_ != nullptr) {
            aclDestroyScalar(scalar_);
        }
    }

    AclScalarDescriptor(const AclScalarDescriptor&) = delete;
    AclScalarDescriptor& operator=(const AclScalarDescriptor&) = delete;

    aclScalar* get() const { return scalar_; }

private:
    float value_;
    aclScalar* scalar_ = nullptr;
};

class AclnnOperation {
public:
    using ExecuteFn = aclnnStatus (*)(void*, uint64_t, aclOpExecutor*,
                                      aclrtStream);

    AclnnOperation(uint64_t workspace_size, aclOpExecutor* executor,
                   ExecuteFn execute)
        : workspace_size_(workspace_size), executor_(executor),
          execute_(execute) {
        if (executor_ == nullptr) {
            throw std::runtime_error("ACLNN did not create an executor");
        }

        try {
            ACL_CHECK(aclSetAclOpExecutorRepeatable(executor_));
            workspace_ = AclDeviceBuffer(workspace_size_);
        } catch (...) {
            aclDestroyAclOpExecutor(executor_);
            executor_ = nullptr;
            throw;
        }
    }

    ~AclnnOperation() {
        if (executor_ != nullptr) {
            aclDestroyAclOpExecutor(executor_);
        }
    }

    AclnnOperation(const AclnnOperation&) = delete;
    AclnnOperation& operator=(const AclnnOperation&) = delete;

    void run(aclrtStream stream) {
        ACL_CHECK(execute_(workspace_.data(), workspace_size_, executor_,
                           stream));
    }

private:
    uint64_t workspace_size_ = 0;
    aclOpExecutor* executor_ = nullptr;
    ExecuteFn execute_ = nullptr;
    AclDeviceBuffer workspace_;
};

class StreamBenchmarkTest {
public:
    void execute(size_t array_size, const TestConfig& cfg = TestConfig()) {
        ACL_CHECK(aclrtSetDevice(cfg.device_id));

        AclStream queue;
        int warmup = cfg.warmup_iterations;
        int measure = cfg.measure_iterations;

        using T = float;
        size_t total_bytes = array_size * sizeof(T);

        std::cout << "\n===================================================\n";
        std::cout << "STREAM Benchmark Suite\n";
        std::cout << "Array size: " << (total_bytes / 1024.0 / 1024.0)
                  << " MB (" << array_size << " elements)\n";
        std::cout << "===================================================\n\n";

        AclDeviceBuffer d_a(total_bytes);
        AclDeviceBuffer d_b(total_bytes);
        AclDeviceBuffer d_c(total_bytes);

        AclHostBuffer h_init(total_bytes);
        T* h_ptr = static_cast<T*>(h_init.data());
        initialize_buffer(d_a, h_ptr, array_size, static_cast<T>(1.0));
        initialize_buffer(d_b, h_ptr, array_size, static_cast<T>(2.0));
        initialize_buffer(d_c, h_ptr, array_size, static_cast<T>(0.0));

        AclTensorDescriptor tensor_a(d_a.data(), array_size);
        AclTensorDescriptor tensor_b(d_b.data(), array_size);
        AclTensorDescriptor tensor_c(d_c.data(), array_size);
        AclScalarDescriptor scale_scalar(3.0f);
        AclScalarDescriptor add_alpha(1.0f);
        AclScalarDescriptor triad_alpha(3.0f);

        uint64_t scale_workspace_size = 0;
        aclOpExecutor* scale_executor = nullptr;
        ACL_CHECK(aclnnMulsGetWorkspaceSize(
            tensor_b.get(), scale_scalar.get(), tensor_c.get(),
            &scale_workspace_size, &scale_executor));
        AclnnOperation scale_op(scale_workspace_size, scale_executor,
                                aclnnMuls);

        uint64_t add_workspace_size = 0;
        aclOpExecutor* add_executor = nullptr;
        ACL_CHECK(aclnnAddGetWorkspaceSize(
            tensor_a.get(), tensor_b.get(), add_alpha.get(), tensor_c.get(),
            &add_workspace_size, &add_executor));
        AclnnOperation add_op(add_workspace_size, add_executor, aclnnAdd);

        uint64_t triad_workspace_size = 0;
        aclOpExecutor* triad_executor = nullptr;
        ACL_CHECK(aclnnAddGetWorkspaceSize(
            tensor_a.get(), tensor_b.get(), triad_alpha.get(), tensor_c.get(),
            &triad_workspace_size, &triad_executor));
        AclnnOperation triad_op(triad_workspace_size, triad_executor,
                                aclnnAdd);

        std::vector<Result> results;
        double copy_bytes = static_cast<double>(2 * total_bytes);
        results.push_back(measure_operation(
            "STREAM_Copy", copy_bytes, warmup, measure, queue, [&]() {
                ACL_CHECK(aclrtMemcpyAsync(
                    d_c.data(), total_bytes, d_b.data(), total_bytes,
                    ACL_MEMCPY_DEVICE_TO_DEVICE, queue.get()));
            }));
        validate_output("STREAM_Copy", d_c.data(), array_size, 2.0f);

        double scale_bytes = static_cast<double>(2 * total_bytes);
        results.push_back(measure_operation(
            "STREAM_Scale", scale_bytes, warmup, measure, queue,
            [&]() { scale_op.run(queue.get()); }));
        validate_output("STREAM_Scale", d_c.data(), array_size, 6.0f);

        double add_bytes = static_cast<double>(3 * total_bytes);
        results.push_back(measure_operation(
            "STREAM_Add", add_bytes, warmup, measure, queue,
            [&]() { add_op.run(queue.get()); }));
        validate_output("STREAM_Add", d_c.data(), array_size, 3.0f);

        double triad_bytes = static_cast<double>(3 * total_bytes);
        results.push_back(measure_operation(
            "STREAM_Triad", triad_bytes, warmup, measure, queue,
            [&]() { triad_op.run(queue.get()); }));
        validate_output("STREAM_Triad", d_c.data(), array_size, 7.0f);

        std::cout << std::left << std::setw(16) << "Operation"
                  << std::right << std::setw(18) << "Bandwidth (GB/s)"
                  << std::setw(14) << "Time (ms)"
                  << std::setw(10) << "CV (%)\n";
        std::cout << std::string(58, '-') << "\n";
        for (const auto& result : results) {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << std::left << std::setw(16) << result.name;
            std::cout << std::right << std::setw(18) << result.bandwidth;
            std::cout << std::setw(14) << result.time_ms;
            std::cout << std::setw(10) << std::setprecision(2)
                      << result.cv_percent << "\n";
        }
        std::cout << "\n";
    }

private:
    struct Result {
        std::string name;
        double bandwidth;
        double time_ms;
        double cv_percent;
    };

    static void initialize_buffer(AclDeviceBuffer& device_buffer,
                                  float* host_buffer, size_t element_count,
                                  float value) {
        for (size_t i = 0; i < element_count; ++i) {
            host_buffer[i] = value;
        }
        size_t bytes = element_count * sizeof(float);
        ACL_CHECK(aclrtMemcpy(device_buffer.data(), bytes, host_buffer, bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE));
    }

    template <typename Operation>
    static Result measure_operation(const std::string& name, double bytes,
                                    int warmup, int measure, AclStream& queue,
                                    Operation operation) {
        for (int i = 0; i < warmup; ++i) {
            operation();
            queue.sync();
        }

        PerfMetrics bandwidth;
        for (int i = 0; i < measure; ++i) {
            queue.sync();
            auto start = std::chrono::high_resolution_clock::now();
            operation();
            queue.sync();
            auto stop = std::chrono::high_resolution_clock::now();
            double seconds =
                std::chrono::duration<double>(stop - start).count();
            bandwidth.add((bytes / 1e9) / seconds);
        }

        double average = bandwidth.trimmed_mean();
        double time_ms = average == 0.0 ? 0.0 : (bytes / 1e9) / average * 1000;
        return {name, average, time_ms, bandwidth.cv() * 100.0};
    }

    static void validate_output(const std::string& operation,
                                void* device_data, size_t element_count,
                                float expected) {
        if (element_count == 0) {
            throw std::runtime_error("STREAM array size must be positive");
        }

        AclHostBuffer samples(2 * sizeof(float));
        float* sample_values = static_cast<float*>(samples.data());
        ACL_CHECK(aclrtMemcpy(sample_values, sizeof(float), device_data,
                              sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));

        const auto* bytes = static_cast<const unsigned char*>(device_data);
        const void* last = bytes + (element_count - 1) * sizeof(float);
        ACL_CHECK(aclrtMemcpy(sample_values + 1, sizeof(float), last,
                              sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));

        for (int i = 0; i < 2; ++i) {
            if (std::fabs(sample_values[i] - expected) > 1e-4f) {
                std::ostringstream message;
                message << operation << " validation failed: expected "
                        << expected << ", got " << sample_values[i];
                throw std::runtime_error(message.str());
            }
        }
    }
};

} // namespace npu_perf
