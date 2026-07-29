#pragma once

#include "cnrt_utils.h"
#include <cstring>

namespace mlu_perf {

// NRAM: 240KB per core, single buffer manually partitioned
#define NRAM_MAX (1024 * 240)
#define ALIGN 128

// ---- init kernel ----
template <typename T>
__mlu_global__ void init_kernel(T* a, T* b, T* c, size_t n,
                                 T va, T vb, T vc) {
    __nram__ char nram_raw[NRAM_MAX];
    char* aligned = (char*)(((size_t)nram_raw + ALIGN - 1) & ~(ALIGN - 1));
    size_t usable = NRAM_MAX - (aligned - nram_raw);
    size_t chunk = usable / (3 * sizeof(T));
    chunk = (chunk / (ALIGN / sizeof(T))) * (ALIGN / sizeof(T));
    if (chunk == 0) return;

    T* na = (T*)aligned;
    T* nb = na + chunk;
    T* nc = nb + chunk;

    size_t per_core = (n + taskDim - 1) / taskDim;
    size_t start = taskId * per_core;
    size_t end = start + per_core > n ? n : start + per_core;

    for (size_t off = start; off < end; off += chunk) {
        size_t cnt = off + chunk > end ? end - off : chunk;
        __bang_write_value(na, cnt, va);
        __bang_write_value(nb, cnt, vb);
        __bang_write_value(nc, cnt, vc);
        __memcpy(a + off, na, cnt * sizeof(T), NRAM2GDRAM);
        __memcpy(b + off, nb, cnt * sizeof(T), NRAM2GDRAM);
        __memcpy(c + off, nc, cnt * sizeof(T), NRAM2GDRAM);
    }
}

// ---- STREAM Copy: dst[i] = src[i]  (2 * N * sizeof(T) bytes moved) ----
template <typename T>
__mlu_global__ void stream_copy_kernel(T* dst, const T* src, size_t n) {
    __nram__ char nram_raw[NRAM_MAX];
    char* aligned = (char*)(((size_t)nram_raw + ALIGN - 1) & ~(ALIGN - 1));
    size_t usable = NRAM_MAX - (aligned - nram_raw);
    size_t chunk = usable / sizeof(T);
    chunk = (chunk / (ALIGN / sizeof(T))) * (ALIGN / sizeof(T));
    if (chunk == 0) return;

    T* buf = (T*)aligned;

    size_t per_core = (n + taskDim - 1) / taskDim;
    size_t start = taskId * per_core;
    size_t end = start + per_core > n ? n : start + per_core;

    for (size_t off = start; off < end; off += chunk) {
        size_t cnt = off + chunk > end ? end - off : chunk;
        __memcpy(buf, src + off, cnt * sizeof(T), GDRAM2NRAM);
        __memcpy(dst + off, buf, cnt * sizeof(T), NRAM2GDRAM);
    }
}

// ---- STREAM Scale: dst[i] = scalar * src[i]  (2 * N * sizeof(T)) ----
template <typename T>
__mlu_global__ void stream_scale_kernel(T* dst, const T* src, T scalar, size_t n) {
    __nram__ char nram_raw[NRAM_MAX];
    char* aligned = (char*)(((size_t)nram_raw + ALIGN - 1) & ~(ALIGN - 1));
    size_t usable = NRAM_MAX - (aligned - nram_raw);
    size_t chunk = usable / (2 * sizeof(T));
    chunk = (chunk / (ALIGN / sizeof(T))) * (ALIGN / sizeof(T));
    if (chunk == 0) return;

    T* ns = (T*)aligned;
    T* nd = ns + chunk;

    size_t per_core = (n + taskDim - 1) / taskDim;
    size_t start = taskId * per_core;
    size_t end = start + per_core > n ? n : start + per_core;

    for (size_t off = start; off < end; off += chunk) {
        size_t cnt = off + chunk > end ? end - off : chunk;
        __memcpy(ns, src + off, cnt * sizeof(T), GDRAM2NRAM);
        __bang_mul_scalar(nd, ns, scalar, cnt);
        __memcpy(dst + off, nd, cnt * sizeof(T), NRAM2GDRAM);
    }
}

// ---- STREAM Add: dst[i] = src1[i] + src2[i]  (3 * N * sizeof(T)) ----
template <typename T>
__mlu_global__ void stream_add_kernel(T* dst, const T* src1, const T* src2, size_t n) {
    __nram__ char nram_raw[NRAM_MAX];
    char* aligned = (char*)(((size_t)nram_raw + ALIGN - 1) & ~(ALIGN - 1));
    size_t usable = NRAM_MAX - (aligned - nram_raw);
    size_t chunk = usable / (3 * sizeof(T));
    chunk = (chunk / (ALIGN / sizeof(T))) * (ALIGN / sizeof(T));
    if (chunk == 0) return;

    T* na = (T*)aligned;
    T* nb = na + chunk;
    T* nc = nb + chunk;

    size_t per_core = (n + taskDim - 1) / taskDim;
    size_t start = taskId * per_core;
    size_t end = start + per_core > n ? n : start + per_core;

    for (size_t off = start; off < end; off += chunk) {
        size_t cnt = off + chunk > end ? end - off : chunk;
        __memcpy(na, src1 + off, cnt * sizeof(T), GDRAM2NRAM);
        __memcpy(nb, src2 + off, cnt * sizeof(T), GDRAM2NRAM);
        __bang_add(nc, na, nb, cnt);
        __memcpy(dst + off, nc, cnt * sizeof(T), NRAM2GDRAM);
    }
}

// ---- STREAM Triad: dst[i] = src1[i] + scalar * src2[i]  (3 * N * sizeof(T)) ----
template <typename T>
__mlu_global__ void stream_triad_kernel(T* dst, const T* src1, const T* src2,
                                         T scalar, size_t n) {
    __nram__ char nram_raw[NRAM_MAX];
    char* aligned = (char*)(((size_t)nram_raw + ALIGN - 1) & ~(ALIGN - 1));
    size_t usable = NRAM_MAX - (aligned - nram_raw);
    size_t chunk = usable / (3 * sizeof(T));
    chunk = (chunk / (ALIGN / sizeof(T))) * (ALIGN / sizeof(T));
    if (chunk == 0) return;

    T* na = (T*)aligned;
    T* nb = na + chunk;
    T* nc = nb + chunk;

    size_t per_core = (n + taskDim - 1) / taskDim;
    size_t start = taskId * per_core;
    size_t end = start + per_core > n ? n : start + per_core;

    for (size_t off = start; off < end; off += chunk) {
        size_t cnt = off + chunk > end ? end - off : chunk;
        __memcpy(na, src1 + off, cnt * sizeof(T), GDRAM2NRAM);
        __memcpy(nb, src2 + off, cnt * sizeof(T), GDRAM2NRAM);
        __bang_mul_scalar(nb, nb, scalar, cnt);
        __bang_add(nc, na, nb, cnt);
        __memcpy(dst + off, nc, cnt * sizeof(T), NRAM2GDRAM);
    }
}

// ============================================================
// Benchmark suite
// ============================================================

class StreamBenchmarkTest {
public:
    void execute(size_t array_size, const TestConfig& cfg = TestConfig()) {
        MLU_CHECK(cnrtSetDevice(cfg.device_id));

        cnrtQueue_t queue;
        MLU_CHECK(cnrtQueueCreate(&queue));

        cnrtDeviceProp_t prop;
        MLU_CHECK(cnrtGetDeviceProperties(&prop, cfg.device_id));
        int total_cores = prop.clusterCount * prop.McorePerCluster;

        using T = float;

        cnrtDim3_t dim;
        dim.x = prop.McorePerCluster;
        dim.y = prop.clusterCount;
        dim.z = 1;
        cnrtFunctionType_t k_type = cnrtFuncTypeUnion1;

        int warmup = cfg.warmup_iterations;
        int measure = cfg.measure_iterations;

        std::cout << "\n===================================================\n";
        std::cout << "STREAM Benchmark Suite\n";
        std::cout << "Array size: " << (array_size * sizeof(T) / 1024.0 / 1024.0)
                  << " MB (" << array_size << " elements)\n";
        std::cout << "===================================================\n\n";

        T* d_a; T* d_b; T* d_c;
        MLU_CHECK(cnrtMalloc(reinterpret_cast<void**>(&d_a), array_size * sizeof(T)));
        MLU_CHECK(cnrtMalloc(reinterpret_cast<void**>(&d_b), array_size * sizeof(T)));
        MLU_CHECK(cnrtMalloc(reinterpret_cast<void**>(&d_c), array_size * sizeof(T)));

        init_kernel<T><<<dim, k_type, queue>>>(d_a, d_b, d_c, array_size,
                                                  (T)1.0, (T)2.0, (T)0.0);
        MLU_CHECK(cnrtQueueSync(queue));

        struct Result { std::string name; double bw; double ms; double cv; };
        std::vector<Result> results;

        // --- STREAM Copy ---
        {
            for (int i = 0; i < warmup; ++i) {
                stream_copy_kernel<T><<<dim, k_type, queue>>>(d_c, d_b, array_size);
                MLU_CHECK(cnrtQueueSync(queue));
            }
            PerfMetrics bw_m;
            for (int i = 0; i < measure; ++i) {
                cnrtNotifier_t ns, ne;
                MLU_CHECK(cnrtNotifierCreate(&ns));
                MLU_CHECK(cnrtNotifierCreate(&ne));
                MLU_CHECK(cnrtPlaceNotifier(ns, queue));
                stream_copy_kernel<T><<<dim, k_type, queue>>>(d_c, d_b, array_size);
                MLU_CHECK(cnrtPlaceNotifier(ne, queue));
                MLU_CHECK(cnrtQueueSync(queue));
                float us;
                MLU_CHECK(cnrtNotifierDuration(ns, ne, &us));
                bw_m.add(((double)2 * sizeof(T) * array_size / 1e9) / (us / 1e6));
                MLU_CHECK(cnrtNotifierDestroy(ns));
                MLU_CHECK(cnrtNotifierDestroy(ne));
            }
            double avg = bw_m.trimmed_mean();
            results.push_back({"STREAM_Copy", avg,
                ((double)2 * sizeof(T) * array_size / 1e9) / avg * 1000,
                bw_m.cv() * 100.0});
        }

        // --- STREAM Scale ---
        {
            for (int i = 0; i < warmup; ++i) {
                stream_scale_kernel<T><<<dim, k_type, queue>>>(d_c, d_b, (T)3.5, array_size);
                MLU_CHECK(cnrtQueueSync(queue));
            }
            PerfMetrics bw_m;
            for (int i = 0; i < measure; ++i) {
                cnrtNotifier_t ns, ne;
                MLU_CHECK(cnrtNotifierCreate(&ns));
                MLU_CHECK(cnrtNotifierCreate(&ne));
                MLU_CHECK(cnrtPlaceNotifier(ns, queue));
                stream_scale_kernel<T><<<dim, k_type, queue>>>(d_c, d_b, (T)3.5, array_size);
                MLU_CHECK(cnrtPlaceNotifier(ne, queue));
                MLU_CHECK(cnrtQueueSync(queue));
                float us;
                MLU_CHECK(cnrtNotifierDuration(ns, ne, &us));
                bw_m.add(((double)2 * sizeof(T) * array_size / 1e9) / (us / 1e6));
                MLU_CHECK(cnrtNotifierDestroy(ns));
                MLU_CHECK(cnrtNotifierDestroy(ne));
            }
            double avg = bw_m.trimmed_mean();
            results.push_back({"STREAM_Scale", avg,
                ((double)2 * sizeof(T) * array_size / 1e9) / avg * 1000,
                bw_m.cv() * 100.0});
        }

        // --- STREAM Add ---
        {
            for (int i = 0; i < warmup; ++i) {
                stream_add_kernel<T><<<dim, k_type, queue>>>(d_c, d_a, d_b, array_size);
                MLU_CHECK(cnrtQueueSync(queue));
            }
            PerfMetrics bw_m;
            for (int i = 0; i < measure; ++i) {
                cnrtNotifier_t ns, ne;
                MLU_CHECK(cnrtNotifierCreate(&ns));
                MLU_CHECK(cnrtNotifierCreate(&ne));
                MLU_CHECK(cnrtPlaceNotifier(ns, queue));
                stream_add_kernel<T><<<dim, k_type, queue>>>(d_c, d_a, d_b, array_size);
                MLU_CHECK(cnrtPlaceNotifier(ne, queue));
                MLU_CHECK(cnrtQueueSync(queue));
                float us;
                MLU_CHECK(cnrtNotifierDuration(ns, ne, &us));
                bw_m.add(((double)3 * sizeof(T) * array_size / 1e9) / (us / 1e6));
                MLU_CHECK(cnrtNotifierDestroy(ns));
                MLU_CHECK(cnrtNotifierDestroy(ne));
            }
            double avg = bw_m.trimmed_mean();
            results.push_back({"STREAM_Add", avg,
                ((double)3 * sizeof(T) * array_size / 1e9) / avg * 1000,
                bw_m.cv() * 100.0});
        }

        // --- STREAM Triad ---
        {
            for (int i = 0; i < warmup; ++i) {
                stream_triad_kernel<T><<<dim, k_type, queue>>>(d_c, d_a, d_b, (T)3.5, array_size);
                MLU_CHECK(cnrtQueueSync(queue));
            }
            PerfMetrics bw_m;
            for (int i = 0; i < measure; ++i) {
                cnrtNotifier_t ns, ne;
                MLU_CHECK(cnrtNotifierCreate(&ns));
                MLU_CHECK(cnrtNotifierCreate(&ne));
                MLU_CHECK(cnrtPlaceNotifier(ns, queue));
                stream_triad_kernel<T><<<dim, k_type, queue>>>(d_c, d_a, d_b, (T)3.5, array_size);
                MLU_CHECK(cnrtPlaceNotifier(ne, queue));
                MLU_CHECK(cnrtQueueSync(queue));
                float us;
                MLU_CHECK(cnrtNotifierDuration(ns, ne, &us));
                bw_m.add(((double)3 * sizeof(T) * array_size / 1e9) / (us / 1e6));
                MLU_CHECK(cnrtNotifierDestroy(ns));
                MLU_CHECK(cnrtNotifierDestroy(ne));
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
        std::cout << "\n";

        cnrtFree(d_a); cnrtFree(d_b); cnrtFree(d_c);
        MLU_CHECK(cnrtQueueDestroy(queue));
    }
};

} // namespace mlu_perf
