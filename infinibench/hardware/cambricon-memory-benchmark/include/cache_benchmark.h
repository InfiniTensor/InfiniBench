#pragma once

#include "cnrt_utils.h"
#include "nram_utils.h"

namespace mlu_perf {

// ============================================================
// NRAM Bandwidth Kernel
// ============================================================
// Load data into each core's explicitly managed NRAM, then repeatedly run
// vector additions. Each __bang_add performs two reads and one write.

template <typename T>
__mlu_global__ void nram_add_kernel(T* dst, const T* src, size_t n,
                                    int repeat) {
    __nram__ char nram_raw[kNramBytes];
    T* buf_a;
    size_t chunk = prepare_nram_layout<T>(nram_raw, 2, &buf_a);
    if (chunk == 0) return;

    T* buf_b = buf_a + chunk;

    size_t per_core = (n + taskDim - 1) / taskDim;
    size_t start = taskId * per_core;
    size_t end = start + per_core > n ? n : start + per_core;
    if (start >= end) return;

    size_t cnt = end - start;
    if (cnt > chunk) cnt = chunk;
    __memcpy(buf_a, src + start, cnt * sizeof(T), GDRAM2NRAM);

    // Repeat __bang_add alternating between two buffers
    for (int r = 0; r < repeat; r++) {
        __bang_add(buf_b, buf_a, buf_a, cnt);
        __bang_add(buf_a, buf_b, buf_b, cnt);
    }

    __memcpy(dst + start, buf_a, cnt * sizeof(T), NRAM2GDRAM);
}

// ============================================================
// GDRAM<->NRAM Copy Kernel (used by the L2 cache test)
// ============================================================

template <typename T>
__mlu_global__ void cache_rw_kernel(T* dst, const T* src, size_t n, int repeat) {
    __nram__ char nram_raw[kNramBytes];
    T* buf;
    size_t chunk = prepare_nram_layout<T>(nram_raw, 1, &buf);
    if (chunk == 0) return;

    size_t per_core = (n + taskDim - 1) / taskDim;
    size_t start = taskId * per_core;
    size_t end = start + per_core > n ? n : start + per_core;
    if (start >= end) return;

    for (int r = 0; r < repeat; r++) {
        for (size_t off = start; off < end; off += chunk) {
            size_t c = off + chunk > end ? end - off : chunk;
            __memcpy(buf, src + off, c * sizeof(T), GDRAM2NRAM);
            __memcpy(dst + off, buf, c * sizeof(T), NRAM2GDRAM);
        }
    }
}

// ============================================================
// NRAM Bandwidth Test
// ============================================================

class NRAMBandwidthTest {
public:
    void execute(const TestConfig& cfg = TestConfig()) {
        MLU_CHECK(cnrtSetDevice(cfg.device_id));

        cnrtQueue_t queue;
        MLU_CHECK(cnrtQueueCreate(&queue));

        cnrtDeviceProp_t prop;
        MLU_CHECK(cnrtGetDeviceProperties(&prop, cfg.device_id));
        int total_cores = prop.clusterCount * prop.McorePerCluster;

        cnrtDim3_t dim;
        dim.x = prop.McorePerCluster;
        dim.y = prop.clusterCount;
        dim.z = 1;
        cnrtFunctionType_t k_type = cnrtFuncTypeUnion1;

        using T = float;
        int warmup = cfg.warmup_iterations;
        int measure = cfg.measure_iterations;

        std::cout << "\n===================================================\n";
        std::cout << "NRAM Bandwidth Test (BANG Kernel)\n";
        std::cout << "Cores: " << total_cores << "\n";
        std::cout << "===================================================\n\n";

        // Use the maximum NRAM chunk: two buffers in 240 KB, about 120 KB each.
        size_t nram_bytes = kNramBytes - kNramAlignment;
        size_t chunk = nram_bytes / (2 * sizeof(T));
        chunk = (chunk / (kNramAlignment / sizeof(T)))
                * (kNramAlignment / sizeof(T));
        size_t chunk_bytes = chunk * sizeof(T);
        size_t total_elements = chunk * total_cores;
        size_t total_bytes = total_elements * sizeof(T);

        // Large repeat to amortize per-call overhead
        // Aligned with CUDA L1: ~1e9 / ARRAY_N + 2
        size_t repeat_count = 1000000000ULL / chunk + 2;

        void* src = nullptr;
        void* dst = nullptr;
        MLU_CHECK(cnrtMalloc(&src, total_bytes));
        MLU_CHECK(cnrtMalloc(&dst, total_bytes));

        std::cout << "Chunk size per core: " << (chunk_bytes / 1024) << " kB\n";
        std::cout << "Repeat count: " << repeat_count << "\n\n";

        // Warmup
        for (int i = 0; i < warmup; ++i) {
            nram_add_kernel<T><<<dim, k_type, queue>>>(
                (T*)dst, (const T*)src, total_elements, (int)repeat_count);
            MLU_CHECK(cnrtQueueSync(queue));
        }

        // Measure
        PerfMetrics bw_metrics;
        for (int i = 0; i < measure; ++i) {
            cnrtNotifier_t ns, ne;
            MLU_CHECK(cnrtNotifierCreate(&ns));
            MLU_CHECK(cnrtNotifierCreate(&ne));

            MLU_CHECK(cnrtPlaceNotifier(ns, queue));
            nram_add_kernel<T><<<dim, k_type, queue>>>(
                (T*)dst, (const T*)src, total_elements, (int)repeat_count);
            MLU_CHECK(cnrtPlaceNotifier(ne, queue));
            MLU_CHECK(cnrtQueueSync(queue));

            float us;
            MLU_CHECK(cnrtNotifierDuration(ns, ne, &us));
            double sec = us / 1e6;

            // Two __bang_add calls per repeat, each reading two NRAM inputs.
            double data_volume = 4.0 * chunk_bytes;
            double total_bw = data_volume * total_cores * repeat_count / sec / 1e9;
            bw_metrics.add(total_bw);

            MLU_CHECK(cnrtNotifierDestroy(ns));
            MLU_CHECK(cnrtNotifierDestroy(ne));
        }

        double avg_bw = bw_metrics.trimmed_mean();
        double avg_time_sec = (4.0 * chunk_bytes * total_cores * repeat_count / 1e9)
                              / avg_bw;

        // Also compute TFLOPS: 2 adds per iter, each is 1 FLOP per element
        double total_flops = 2.0 * chunk * total_cores * repeat_count;
        double tflops = total_flops / avg_time_sec / 1e12;

        std::cout << std::left << std::setw(20) << "NRAM chunk/core"
                  << std::right << std::setw(12) << "Time (ms)"
                  << std::setw(15) << "Eff. BW (GB/s)"
                  << std::setw(12) << "TFLOPS"
                  << std::setw(10) << "Spread\n";
        std::cout << std::string(69, '-') << "\n";

        std::cout << std::fixed << std::setprecision(1);
        std::cout << std::left << std::setw(20)
                  << std::to_string(chunk_bytes / 1024) + " kB";
        std::cout << std::right << std::setw(12) << std::setprecision(1)
                  << avg_time_sec * 1000;
        std::cout << std::setw(15) << std::setprecision(1) << avg_bw;
        std::cout << std::setw(12) << std::setprecision(1) << tflops;
        std::cout << std::setw(10) << std::setprecision(1)
                  << (bw_metrics.cv() * 100.0) << "%\n";

        cnrtFree(src);
        cnrtFree(dst);
        MLU_CHECK(cnrtQueueDestroy(queue));
        std::cout << "\n";
    }
};

// ============================================================
// L2 Cache Bandwidth Sweep Test
// ============================================================

class L2CacheBandwidthTest {
public:
    void execute(const TestConfig& cfg = TestConfig()) {
        MLU_CHECK(cnrtSetDevice(cfg.device_id));

        cnrtQueue_t queue;
        MLU_CHECK(cnrtQueueCreate(&queue));

        cnrtDeviceProp_t prop;
        MLU_CHECK(cnrtGetDeviceProperties(&prop, cfg.device_id));
        int total_cores = prop.clusterCount * prop.McorePerCluster;
        size_t l2_bytes = prop.maxL2CacheSize;

        cnrtDim3_t dim;
        dim.x = prop.McorePerCluster;
        dim.y = prop.clusterCount;
        dim.z = 1;
        cnrtFunctionType_t k_type = cnrtFuncTypeUnion1;

        using T = float;
        const int repeat = 10;
        int warmup = cfg.warmup_iterations;
        int measure = cfg.measure_iterations;

        std::cout << "\n===================================================\n";
        std::cout << "L2 Cache Bandwidth Sweep Test (BANG Kernel)\n";
        std::cout << "Cores: " << total_cores
                  << ", L2 Cache: " << (l2_bytes / 1024) << " kB\n";
        std::cout << "===================================================\n\n";

        std::cout << std::left << std::setw(13) << "data set"
                  << std::setw(12) << "exec data"
                  << std::right << std::setw(12) << "exec time"
                  << std::setw(11) << "spread"
                  << std::setw(15) << "Eff. bw\n";
        std::cout << std::string(63, '-') << "\n";

        // Sweep from 256KB to 128MB
        // L2 is ~40MB, so <40MB should show high bandwidth (L2 hit)
        // Data sets larger than L2 should show lower DRAM-backed bandwidth.
        std::vector<size_t> sizes_kb;
        for (size_t s = 256; s <= 8192; s *= 2) sizes_kb.push_back(s);
        for (size_t s = 10240; s <= 65536; s += 4096) sizes_kb.push_back(s);
        for (size_t s = 65536; s <= 131072; s *= 2) sizes_kb.push_back(s);

        size_t max_bytes = 256ULL * 1024 * 1024;
        void* src = nullptr;
        void* dst = nullptr;
        MLU_CHECK(cnrtMalloc(&src, max_bytes));
        MLU_CHECK(cnrtMalloc(&dst, max_bytes));

        for (size_t skb : sizes_kb) {
            size_t bytes = skb * 1024;
            if (bytes > max_bytes) break;
            size_t n = bytes / sizeof(T);

            // Warmup
            for (int i = 0; i < warmup; ++i) {
                cache_rw_kernel<T><<<dim, k_type, queue>>>(
                    (T*)dst, (const T*)src, n, repeat);
                MLU_CHECK(cnrtQueueSync(queue));
            }

            // Measure
            PerfMetrics time_metrics;
            for (int i = 0; i < measure; ++i) {
                cnrtNotifier_t ns, ne;
                MLU_CHECK(cnrtNotifierCreate(&ns));
                MLU_CHECK(cnrtNotifierCreate(&ne));

                MLU_CHECK(cnrtPlaceNotifier(ns, queue));
                cache_rw_kernel<T><<<dim, k_type, queue>>>(
                    (T*)dst, (const T*)src, n, repeat);
                MLU_CHECK(cnrtPlaceNotifier(ne, queue));
                MLU_CHECK(cnrtQueueSync(queue));

                float us;
                MLU_CHECK(cnrtNotifierDuration(ns, ne, &us));
                time_metrics.add(us / 1e3);  // ms

                MLU_CHECK(cnrtNotifierDestroy(ns));
                MLU_CHECK(cnrtNotifierDestroy(ne));
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

        cnrtFree(src);
        cnrtFree(dst);
        MLU_CHECK(cnrtQueueDestroy(queue));
        std::cout << "\n";
    }
};

} // namespace mlu_perf
