#pragma once

#include "acl_utils.h"

namespace npu_perf {

// STREAM benchmark using D2D memcpy for Copy.
// Note: STREAM Scale/Add/Triad require device-side compute kernels
// (custom Ascend C operators). This implementation measures Copy bandwidth
// which is the primary indicator of device memory bandwidth.
// Scale/Add/Triad are estimated from Copy bandwidth.

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

        // Initialize device buffers
        AclHostBuffer h_init(total_bytes);
        T* h_ptr = static_cast<T*>(h_init.data());
        for (size_t i = 0; i < array_size; ++i) {
            h_ptr[i] = static_cast<T>(1.0);
        }
        ACL_CHECK(aclrtMemcpy(d_a.data(), total_bytes, h_init.data(),
                              total_bytes, ACL_MEMCPY_HOST_TO_DEVICE));

        for (size_t i = 0; i < array_size; ++i) {
            h_ptr[i] = static_cast<T>(2.0);
        }
        ACL_CHECK(aclrtMemcpy(d_b.data(), total_bytes, h_init.data(),
                              total_bytes, ACL_MEMCPY_HOST_TO_DEVICE));

        for (size_t i = 0; i < array_size; ++i) {
            h_ptr[i] = static_cast<T>(0.0);
        }
        ACL_CHECK(aclrtMemcpy(d_c.data(), total_bytes, h_init.data(),
                              total_bytes, ACL_MEMCPY_HOST_TO_DEVICE));

        queue.sync();

        struct Result { std::string name; double bw; double ms; double cv; };
        std::vector<Result> results;

        // ---- STREAM Copy: dst[i] = src[i]  (2 * N * sizeof(T) bytes) ----
        {
            for (int i = 0; i < warmup; ++i) {
                ACL_CHECK(aclrtMemcpyAsync(d_c.data(), total_bytes, d_b.data(),
                             total_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, queue.get()));
                queue.sync();
            }
            PerfMetrics bw_m;
            for (int i = 0; i < measure; ++i) {
                queue.sync();
                auto t0 = std::chrono::high_resolution_clock::now();
                ACL_CHECK(aclrtMemcpyAsync(d_c.data(), total_bytes, d_b.data(),
                             total_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, queue.get()));
                queue.sync();
                auto t1 = std::chrono::high_resolution_clock::now();
                double sec = std::chrono::duration<double>(t1 - t0).count();
                bw_m.add(((double)2 * sizeof(T) * array_size / 1e9) / sec);
            }
            double avg = bw_m.trimmed_mean();
            results.push_back({"STREAM_Copy", avg,
                ((double)2 * sizeof(T) * array_size / 1e9) / avg * 1000,
                bw_m.cv() * 100.0});
        }

        // ---- STREAM Scale: dst[i] = scalar * src[i]  (2 * N * sizeof(T)) ----
        // Estimated from D2D copy bandwidth (compute is memory-bound)
        {
            // Scale has same data movement as Copy: 1 read + 1 write
            // Without Ascend C custom kernel, we measure D2D copy as approximation
            for (int i = 0; i < warmup; ++i) {
                ACL_CHECK(aclrtMemcpyAsync(d_c.data(), total_bytes, d_b.data(),
                             total_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, queue.get()));
                queue.sync();
            }
            PerfMetrics bw_m;
            for (int i = 0; i < measure; ++i) {
                queue.sync();
                auto t0 = std::chrono::high_resolution_clock::now();
                ACL_CHECK(aclrtMemcpyAsync(d_c.data(), total_bytes, d_b.data(),
                             total_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, queue.get()));
                queue.sync();
                auto t1 = std::chrono::high_resolution_clock::now();
                double sec = std::chrono::duration<double>(t1 - t0).count();
                bw_m.add(((double)2 * sizeof(T) * array_size / 1e9) / sec);
            }
            double avg = bw_m.trimmed_mean();
            results.push_back({"STREAM_Scale", avg,
                ((double)2 * sizeof(T) * array_size / 1e9) / avg * 1000,
                bw_m.cv() * 100.0});
        }

        // ---- STREAM Add: dst[i] = src1[i] + src2[i]  (3 * N * sizeof(T)) ----
        // Uses two D2D copies to approximate 2 reads + 1 write
        {
            for (int i = 0; i < warmup; ++i) {
                ACL_CHECK(aclrtMemcpyAsync(d_c.data(), total_bytes, d_a.data(),
                             total_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, queue.get()));
                ACL_CHECK(aclrtMemcpyAsync(d_c.data(), total_bytes, d_b.data(),
                             total_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, queue.get()));
                queue.sync();
            }
            PerfMetrics bw_m;
            for (int i = 0; i < measure; ++i) {
                queue.sync();
                auto t0 = std::chrono::high_resolution_clock::now();
                ACL_CHECK(aclrtMemcpyAsync(d_c.data(), total_bytes, d_a.data(),
                             total_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, queue.get()));
                ACL_CHECK(aclrtMemcpyAsync(d_c.data(), total_bytes, d_b.data(),
                             total_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, queue.get()));
                queue.sync();
                auto t1 = std::chrono::high_resolution_clock::now();
                double sec = std::chrono::duration<double>(t1 - t0).count();
                // 2 D2D copies = 4 reads + 2 writes total device-side
                // Effective bytes for Add: 3 * N * sizeof(T)
                bw_m.add(((double)3 * sizeof(T) * array_size / 1e9) / sec);
            }
            double avg = bw_m.trimmed_mean();
            results.push_back({"STREAM_Add", avg,
                ((double)3 * sizeof(T) * array_size / 1e9) / avg * 1000,
                bw_m.cv() * 100.0});
        }

        // ---- STREAM Triad: dst[i] = src1[i] + scalar * src2[i]  (3 * N * sizeof(T)) ----
        // Same approach as Add (2 D2D copies)
        {
            for (int i = 0; i < warmup; ++i) {
                ACL_CHECK(aclrtMemcpyAsync(d_c.data(), total_bytes, d_a.data(),
                             total_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, queue.get()));
                ACL_CHECK(aclrtMemcpyAsync(d_c.data(), total_bytes, d_b.data(),
                             total_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, queue.get()));
                queue.sync();
            }
            PerfMetrics bw_m;
            for (int i = 0; i < measure; ++i) {
                queue.sync();
                auto t0 = std::chrono::high_resolution_clock::now();
                ACL_CHECK(aclrtMemcpyAsync(d_c.data(), total_bytes, d_a.data(),
                             total_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, queue.get()));
                ACL_CHECK(aclrtMemcpyAsync(d_c.data(), total_bytes, d_b.data(),
                             total_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, queue.get()));
                queue.sync();
                auto t1 = std::chrono::high_resolution_clock::now();
                double sec = std::chrono::duration<double>(t1 - t0).count();
                bw_m.add(((double)3 * sizeof(T) * array_size / 1e9) / sec);
            }
            double avg = bw_m.trimmed_mean();
            results.push_back({"STREAM_Triad", avg,
                ((double)3 * sizeof(T) * array_size / 1e9) / avg * 1000,
                bw_m.cv() * 100.0});
        }

        std::cout << std::left << std::setw(16) << "Operation"
                  << std::right << std::setw(18) << "Bandwidth (GB/s)"
                  << std::setw(14) << "Time (ms)"
                  << std::setw(10) << "CV (%)\n";
        std::cout << std::string(58, '-') << "\n";
        for (const auto& r : results) {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << std::left << std::setw(16) << r.name;
            std::cout << std::right << std::setw(18) << r.bw;
            std::cout << std::setw(14) << r.ms;
            std::cout << std::setw(10) << std::setprecision(2) << r.cv << "\n";
        }
        std::cout << "\n"
                  << "NOTE: Scale/Add/Triad bandwidth is estimated from D2D memcpy.\n"
                  << "      For precise results, use Ascend C custom operators.\n\n";
    }
};

} // namespace npu_perf
