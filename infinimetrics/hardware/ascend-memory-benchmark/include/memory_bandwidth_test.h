#pragma once

#include "acl_utils.h"

namespace npu_perf {

class MemoryBandwidthTest {
public:
    void execute(const TestConfig& cfg = TestConfig()) {
        ACL_CHECK(aclrtSetDevice(0));

        const size_t max_bytes = 2ULL * 1024 * 1024 * 1024;  // 2 GB
        const int warmup = cfg.warmup_iterations;
        const int measure = cfg.measure_iterations;

        AclStream queue;

        AclHostBuffer host_buf(max_bytes);
        AclDeviceBuffer dev1(max_bytes);
        AclDeviceBuffer dev2(max_bytes);

        memset(host_buf.data(), 0xAB, max_bytes);
        ACL_CHECK(aclrtMemcpy(dev1.data(), max_bytes, host_buf.data(), max_bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(dev2.data(), max_bytes, host_buf.data(), max_bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE));

        std::vector<size_t> sizes_kb = {
            64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
            32768, 65536, 131072, 262144, 524288, 1048576
        };

        auto print_table_header = [&]() {
            std::cout << std::left << std::setw(15) << "Size (MB)"
                      << std::right << std::setw(12) << "Time (ms)"
                      << std::setw(18) << "Bandwidth (GB/s)"
                      << std::setw(10) << "CV (%)\n";
            std::cout << std::string(55, '-') << "\n";
        };

        // ---- H2D ----
        std::cout << "\n===================================================\n";
        std::cout << "Memory Copy Bandwidth Sweep Test\n";
        std::cout << "Direction: Host to Device\n";
        std::cout << "===================================================\n\n";
        print_table_header();
        for (size_t skb : sizes_kb) {
            size_t bytes = skb * 1024;
            if (bytes > max_bytes) break;

            for (int i = 0; i < warmup; ++i) {
                ACL_CHECK(aclrtMemcpyAsync(dev1.data(), max_bytes, host_buf.data(),
                             bytes, ACL_MEMCPY_HOST_TO_DEVICE, queue.get()));
                queue.sync();
            }

            PerfMetrics bw;
            for (int i = 0; i < measure; ++i) {
                queue.sync();
                auto t0 = std::chrono::high_resolution_clock::now();
                ACL_CHECK(aclrtMemcpyAsync(dev1.data(), max_bytes, host_buf.data(),
                             bytes, ACL_MEMCPY_HOST_TO_DEVICE, queue.get()));
                queue.sync();
                auto t1 = std::chrono::high_resolution_clock::now();
                double sec = std::chrono::duration<double>(t1 - t0).count();
                bw.add((bytes / 1e9) / sec);
            }

            double avg_bw = bw.trimmed_mean();
            double avg_time_ms = (bytes / 1e9) / avg_bw * 1000;
            std::cout << std::fixed << std::setprecision(2);
            std::cout << std::left << std::setw(15) << (bytes / 1024.0 / 1024.0);
            std::cout << std::right << std::setw(12) << std::setprecision(3) << avg_time_ms;
            std::cout << std::setw(18) << std::setprecision(2) << avg_bw;
            std::cout << std::setw(10) << std::setprecision(1)
                      << (bw.cv() * 100.0) << "\n";
        }
        std::cout << "\n";

        // ---- D2H ----
        std::cout << "===================================================\n";
        std::cout << "Memory Copy Bandwidth Sweep Test\n";
        std::cout << "Direction: Device to Host\n";
        std::cout << "===================================================\n\n";
        print_table_header();
        for (size_t skb : sizes_kb) {
            size_t bytes = skb * 1024;
            if (bytes > max_bytes) break;

            for (int i = 0; i < warmup; ++i) {
                ACL_CHECK(aclrtMemcpyAsync(host_buf.data(), max_bytes, dev1.data(),
                             bytes, ACL_MEMCPY_DEVICE_TO_HOST, queue.get()));
                queue.sync();
            }

            PerfMetrics bw;
            for (int i = 0; i < measure; ++i) {
                queue.sync();
                auto t0 = std::chrono::high_resolution_clock::now();
                ACL_CHECK(aclrtMemcpyAsync(host_buf.data(), max_bytes, dev1.data(),
                             bytes, ACL_MEMCPY_DEVICE_TO_HOST, queue.get()));
                queue.sync();
                auto t1 = std::chrono::high_resolution_clock::now();
                double sec = std::chrono::duration<double>(t1 - t0).count();
                bw.add((bytes / 1e9) / sec);
            }

            double avg_bw = bw.trimmed_mean();
            double avg_time_ms = (bytes / 1e9) / avg_bw * 1000;
            std::cout << std::fixed << std::setprecision(2);
            std::cout << std::left << std::setw(15) << (bytes / 1024.0 / 1024.0);
            std::cout << std::right << std::setw(12) << std::setprecision(3) << avg_time_ms;
            std::cout << std::setw(18) << std::setprecision(2) << avg_bw;
            std::cout << std::setw(10) << std::setprecision(1)
                      << (bw.cv() * 100.0) << "\n";
        }
        std::cout << "\n";

        // ---- D2D ----
        std::cout << "===================================================\n";
        std::cout << "Memory Copy Bandwidth Sweep Test\n";
        std::cout << "Direction: Device to Device\n";
        std::cout << "===================================================\n\n";
        std::cout << "NOTE: Small sizes may reflect cache bandwidth, not DRAM bandwidth.\n\n";
        print_table_header();
        for (size_t skb : sizes_kb) {
            size_t bytes = skb * 1024;
            if (bytes > max_bytes) break;

            for (int i = 0; i < warmup; ++i) {
                ACL_CHECK(aclrtMemcpyAsync(dev2.data(), max_bytes, dev1.data(),
                             bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, queue.get()));
                queue.sync();
            }

            PerfMetrics bw;
            for (int i = 0; i < measure; ++i) {
                queue.sync();
                auto t0 = std::chrono::high_resolution_clock::now();
                ACL_CHECK(aclrtMemcpyAsync(dev2.data(), max_bytes, dev1.data(),
                             bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, queue.get()));
                queue.sync();
                auto t1 = std::chrono::high_resolution_clock::now();
                double sec = std::chrono::duration<double>(t1 - t0).count();
                bw.add((bytes / 1e9) / sec);
            }

            double avg_bw = bw.trimmed_mean();
            double avg_time_ms = (bytes / 1e9) / avg_bw * 1000;
            std::cout << std::fixed << std::setprecision(2);
            std::cout << std::left << std::setw(15) << (bytes / 1024.0 / 1024.0);
            std::cout << std::right << std::setw(12) << std::setprecision(3) << avg_time_ms;
            std::cout << std::setw(18) << std::setprecision(2) << avg_bw;
            std::cout << std::setw(10) << std::setprecision(1)
                      << (bw.cv() * 100.0) << "\n";
        }
        std::cout << "\n";

        // ---- Bidirectional ----
        std::cout << "===================================================\n";
        std::cout << "Memory Copy Bandwidth Sweep Test\n";
        std::cout << "Direction: Bidirectional\n";
        std::cout << "===================================================\n\n";
        print_table_header();
        {
            AclStream q1, q2;

            for (size_t skb : sizes_kb) {
                size_t bytes = skb * 1024;
                if (bytes > max_bytes) break;

                for (int i = 0; i < warmup; ++i) {
                    ACL_CHECK(aclrtMemcpyAsync(dev1.data(), max_bytes, host_buf.data(),
                                 bytes, ACL_MEMCPY_HOST_TO_DEVICE, q1.get()));
                    ACL_CHECK(aclrtMemcpyAsync(host_buf.data(), max_bytes, dev2.data(),
                                 bytes, ACL_MEMCPY_DEVICE_TO_HOST, q2.get()));
                    q1.sync();
                    q2.sync();
                }

                PerfMetrics bw;
                for (int i = 0; i < measure; ++i) {
                    q1.sync();
                    q2.sync();
                    auto t0 = std::chrono::high_resolution_clock::now();

                    ACL_CHECK(aclrtMemcpyAsync(dev1.data(), max_bytes, host_buf.data(),
                                 bytes, ACL_MEMCPY_HOST_TO_DEVICE, q1.get()));
                    ACL_CHECK(aclrtMemcpyAsync(host_buf.data(), max_bytes, dev2.data(),
                                 bytes, ACL_MEMCPY_DEVICE_TO_HOST, q2.get()));

                    q1.sync();
                    q2.sync();
                    auto t1 = std::chrono::high_resolution_clock::now();

                    double sec = std::chrono::duration<double>(t1 - t0).count();
                    bw.add((2.0 * bytes / 1e9) / sec);
                }

                double avg_bw = bw.trimmed_mean();
                double avg_time_ms = (2.0 * bytes / 1e9) / avg_bw * 1000;
                std::cout << std::fixed << std::setprecision(2);
                std::cout << std::left << std::setw(15) << (bytes / 1024.0 / 1024.0);
                std::cout << std::right << std::setw(12) << std::setprecision(3) << avg_time_ms;
                std::cout << std::setw(18) << std::setprecision(2) << avg_bw;
                std::cout << std::setw(10) << std::setprecision(1)
                          << (bw.cv() * 100.0) << "\n";
            }
        }
        std::cout << "\n";
    }
};

} // namespace npu_perf
