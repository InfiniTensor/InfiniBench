#pragma once

#include "acl_utils.h"

namespace npu_perf {

// Ascend NPUs have a different memory hierarchy than GPUs:
// - No L1/L2 cache in the GPU sense
// - Instead: L1 (instruction/data cache per AI Core), L2 (unified buffer),
//   and HBM (global memory)
// This test does a D2D bandwidth sweep at different sizes to characterize
// the memory hierarchy, similar to how CUDA L1/L2 tests work.

class CacheBenchmarkTest {
public:
    void execute(const TestConfig& cfg = TestConfig()) {
        ACL_CHECK(aclrtSetDevice(0));

        AclStream queue;
        int warmup = cfg.warmup_iterations;
        int measure = cfg.measure_iterations;
        const int repeat = 10;

        std::cout << "\n===================================================\n";
        std::cout << "Memory Hierarchy Sweep Test (D2D Bandwidth)\n";
        std::cout << "===================================================\n\n";

        std::cout << std::left << std::setw(13) << "data set"
                  << std::setw(12) << "exec data"
                  << std::right << std::setw(12) << "exec time"
                  << std::setw(11) << "spread"
                  << std::setw(15) << "Eff. bw\n";
        std::cout << std::string(63, '-') << "\n";

        // Sweep from 4KB to 256MB
        std::vector<size_t> sizes_kb;
        for (size_t s = 4; s <= 512; s *= 2) sizes_kb.push_back(s);
        for (size_t s = 1024; s <= 8192; s *= 2) sizes_kb.push_back(s);
        for (size_t s = 10240; s <= 65536; s += 4096) sizes_kb.push_back(s);
        for (size_t s = 65536; s <= 262144; s *= 2) sizes_kb.push_back(s);

        size_t max_bytes = 512ULL * 1024 * 1024;
        AclDeviceBuffer src(max_bytes);
        AclDeviceBuffer dst(max_bytes);

        // Initialize buffers
        AclHostBuffer h_buf(max_bytes);
        memset(h_buf.data(), 0xCD, max_bytes);
        ACL_CHECK(aclrtMemcpy(src.data(), max_bytes, h_buf.data(),
                              max_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(dst.data(), max_bytes, h_buf.data(),
                              max_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        queue.sync();

        for (size_t skb : sizes_kb) {
            size_t bytes = skb * 1024;
            if (bytes > max_bytes) break;

            // Warmup
            for (int i = 0; i < warmup; ++i) {
                for (int r = 0; r < repeat; ++r) {
                    ACL_CHECK(aclrtMemcpyAsync(dst.data(), max_bytes, src.data(),
                                 bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, queue.get()));
                }
                queue.sync();
            }

            // Measure
            PerfMetrics time_metrics;
            for (int i = 0; i < measure; ++i) {
                queue.sync();
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int r = 0; r < repeat; ++r) {
                    ACL_CHECK(aclrtMemcpyAsync(dst.data(), max_bytes, src.data(),
                                 bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, queue.get()));
                }
                queue.sync();
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                time_metrics.add(ms);
            }

            double avg_time_ms = time_metrics.trimmed_mean();
            double total_data = 2.0 * bytes * repeat;
            double bw_gbps = total_data / (avg_time_ms / 1e3) / 1e9;

            std::cout << std::fixed << std::setprecision(0);
            std::cout << std::left << std::setw(13)
                      << std::to_string(bytes / 1024) + " kB";
            std::cout << std::setw(12)
                      << std::to_string(bytes * repeat / 1024) + " kB";
            std::cout << std::right << std::setw(12)
                      << std::setprecision(0) << avg_time_ms << "ms";
            std::cout << std::setprecision(1) << std::setw(11)
                      << (time_metrics.cv() * 100.0) << "%";
            std::cout << std::setprecision(1) << std::setw(15)
                      << bw_gbps << " GB/s";
            std::cout << "\n";
        }

        std::cout << "\n";
    }
};

} // namespace npu_perf
