#pragma once

#include <cstddef>

namespace mlu_perf {

constexpr size_t kNramBytes = 1024 * 240;
constexpr size_t kNramAlignment = 128;

template <typename T>
__mlu_device__ inline size_t prepare_nram_layout(
    char* raw, size_t buffer_count, T** base) {
    char* aligned = (char*)(((size_t)raw + kNramAlignment - 1)
                            & ~(kNramAlignment - 1));
    size_t usable = kNramBytes - (aligned - raw);
    size_t chunk = usable / (buffer_count * sizeof(T));
    size_t alignment_elements = kNramAlignment / sizeof(T);
    *base = (T*)aligned;
    return (chunk / alignment_elements) * alignment_elements;
}

} // namespace mlu_perf
