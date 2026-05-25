#pragma once

#include "cnrt_utils.h"

namespace mlu_perf {

class MemoryBandwidthTest {
public:
    void execute(const TestConfig& cfg = TestConfig()) {
        MLU_CHECK(cnrtSetDevice(0));

        const size_t max_bytes = 2ULL * 1024 * 1024 * 1024;  // 2 GB
        const int warmup = cfg.warmup_iterations;
        const int measure = cfg.measure_iterations;

        cnrtQueue_t queue;
        MLU_CHECK(cnrtQueueCreate(&queue));

        void* host_buf;
        void* dev1;
        void* dev2;
        MLU_CHECK(cnrtHostMalloc(&host_buf, max_bytes));
        MLU_CHECK(cnrtMalloc(&dev1, max_bytes));
        MLU_CHECK(cnrtMalloc(&dev2, max_bytes));

        memset(host_buf, 0xAB, max_bytes);
        MLU_CHECK(cnrtMemcpy(dev1, host_buf, max_bytes, cnrtMemcpyHostToDev));
        MLU_CHECK(cnrtMemcpy(dev2, host_buf, max_bytes, cnrtMemcpyHostToDev));

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
                MLU_CHECK(cnrtMemcpyAsync(dev1, host_buf, bytes, queue, cnrtMemcpyHostToDev));
                MLU_CHECK(cnrtQueueSync(queue));
            }

            PerfMetrics bw;
            for (int i = 0; i < measure; ++i) {
                cnrtNotifier_t ns, ne;
                MLU_CHECK(cnrtNotifierCreate(&ns));
                MLU_CHECK(cnrtNotifierCreate(&ne));

                MLU_CHECK(cnrtPlaceNotifier(ns, queue));
                MLU_CHECK(cnrtMemcpyAsync(dev1, host_buf, bytes, queue, cnrtMemcpyHostToDev));
                MLU_CHECK(cnrtPlaceNotifier(ne, queue));
                MLU_CHECK(cnrtQueueSync(queue));

                float us;
                MLU_CHECK(cnrtNotifierDuration(ns, ne, &us));
                double sec = us / 1e6;
                bw.add((bytes / 1e9) / sec);

                MLU_CHECK(cnrtNotifierDestroy(ns));
                MLU_CHECK(cnrtNotifierDestroy(ne));
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
                MLU_CHECK(cnrtMemcpyAsync(host_buf, dev1, bytes, queue, cnrtMemcpyDevToHost));
                MLU_CHECK(cnrtQueueSync(queue));
            }

            PerfMetrics bw;
            for (int i = 0; i < measure; ++i) {
                cnrtNotifier_t ns, ne;
                MLU_CHECK(cnrtNotifierCreate(&ns));
                MLU_CHECK(cnrtNotifierCreate(&ne));

                MLU_CHECK(cnrtPlaceNotifier(ns, queue));
                MLU_CHECK(cnrtMemcpyAsync(host_buf, dev1, bytes, queue, cnrtMemcpyDevToHost));
                MLU_CHECK(cnrtPlaceNotifier(ne, queue));
                MLU_CHECK(cnrtQueueSync(queue));

                float us;
                MLU_CHECK(cnrtNotifierDuration(ns, ne, &us));
                double sec = us / 1e6;
                bw.add((bytes / 1e9) / sec);

                MLU_CHECK(cnrtNotifierDestroy(ns));
                MLU_CHECK(cnrtNotifierDestroy(ne));
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
                MLU_CHECK(cnrtMemcpyAsync(dev2, dev1, bytes, queue, cnrtMemcpyDevToDev));
                MLU_CHECK(cnrtQueueSync(queue));
            }

            PerfMetrics bw;
            for (int i = 0; i < measure; ++i) {
                cnrtNotifier_t ns, ne;
                MLU_CHECK(cnrtNotifierCreate(&ns));
                MLU_CHECK(cnrtNotifierCreate(&ne));

                MLU_CHECK(cnrtPlaceNotifier(ns, queue));
                MLU_CHECK(cnrtMemcpyAsync(dev2, dev1, bytes, queue, cnrtMemcpyDevToDev));
                MLU_CHECK(cnrtPlaceNotifier(ne, queue));
                MLU_CHECK(cnrtQueueSync(queue));

                float us;
                MLU_CHECK(cnrtNotifierDuration(ns, ne, &us));
                double sec = us / 1e6;
                bw.add((bytes / 1e9) / sec);

                MLU_CHECK(cnrtNotifierDestroy(ns));
                MLU_CHECK(cnrtNotifierDestroy(ne));
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
            cnrtQueue_t q1, q2;
            MLU_CHECK(cnrtQueueCreate(&q1));
            MLU_CHECK(cnrtQueueCreate(&q2));

            for (size_t skb : sizes_kb) {
                size_t bytes = skb * 1024;
                if (bytes > max_bytes) break;

                for (int i = 0; i < warmup; ++i) {
                    MLU_CHECK(cnrtMemcpyAsync(dev1, host_buf, bytes, q1, cnrtMemcpyHostToDev));
                    MLU_CHECK(cnrtMemcpyAsync(host_buf, dev2, bytes, q2, cnrtMemcpyDevToHost));
                    MLU_CHECK(cnrtQueueSync(q1));
                    MLU_CHECK(cnrtQueueSync(q2));
                }

                PerfMetrics bw;
                for (int i = 0; i < measure; ++i) {
                    cnrtNotifier_t ns1, ne1, ns2, ne2;
                    MLU_CHECK(cnrtNotifierCreate(&ns1));
                    MLU_CHECK(cnrtNotifierCreate(&ne1));
                    MLU_CHECK(cnrtNotifierCreate(&ns2));
                    MLU_CHECK(cnrtNotifierCreate(&ne2));

                    MLU_CHECK(cnrtPlaceNotifier(ns1, q1));
                    MLU_CHECK(cnrtMemcpyAsync(dev1, host_buf, bytes, q1, cnrtMemcpyHostToDev));
                    MLU_CHECK(cnrtPlaceNotifier(ne1, q1));

                    MLU_CHECK(cnrtPlaceNotifier(ns2, q2));
                    MLU_CHECK(cnrtMemcpyAsync(host_buf, dev2, bytes, q2, cnrtMemcpyDevToHost));
                    MLU_CHECK(cnrtPlaceNotifier(ne2, q2));

                    MLU_CHECK(cnrtQueueSync(q1));
                    MLU_CHECK(cnrtQueueSync(q2));

                    float us1, us2;
                    MLU_CHECK(cnrtNotifierDuration(ns1, ne1, &us1));
                    MLU_CHECK(cnrtNotifierDuration(ns2, ne2, &us2));

                    double sec = std::max(us1, us2) / 1e6;
                    bw.add((2.0 * bytes / 1e9) / sec);

                    MLU_CHECK(cnrtNotifierDestroy(ns1));
                    MLU_CHECK(cnrtNotifierDestroy(ne1));
                    MLU_CHECK(cnrtNotifierDestroy(ns2));
                    MLU_CHECK(cnrtNotifierDestroy(ne2));
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

            MLU_CHECK(cnrtQueueDestroy(q1));
            MLU_CHECK(cnrtQueueDestroy(q2));
        }

        cnrtFreeHost(host_buf);
        cnrtFree(dev1);
        cnrtFree(dev2);
        MLU_CHECK(cnrtQueueDestroy(queue));
        std::cout << "\n";
    }
};

} // namespace mlu_perf
