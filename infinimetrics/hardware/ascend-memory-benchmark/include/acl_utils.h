#pragma once

#include <acl/acl.h>
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cmath>
#include <cstring>

namespace npu_perf {

// ACL error checking macro
#define ACL_CHECK(call) \
    do { \
        aclError ret = call; \
        if (ret != ACL_SUCCESS) { \
            std::ostringstream oss; \
            oss << "ACL error at " << __FILE__ << ":" << __LINE__ \
                << ": error code=" << ret; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

// Host-side high resolution timer
class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    Timer() : start_(Clock::now()) {}
    void reset() { start_ = Clock::now(); }

    double elapsed_seconds() const {
        return std::chrono::duration<double>(Clock::now() - start_).count();
    }
    double elapsed_ms() const { return elapsed_seconds() * 1000.0; }

private:
    TimePoint start_;
};

// Statistics collector
class PerfMetrics {
public:
    void add(double v) { samples_.push_back(v); }

    double mean() const {
        if (samples_.empty()) return 0.0;
        return std::accumulate(samples_.begin(), samples_.end(), 0.0) / samples_.size();
    }

    double trimmed_mean() const {
        if (samples_.size() <= 2) return mean();
        auto s = samples_;
        std::sort(s.begin(), s.end());
        return std::accumulate(s.begin() + 1, s.end() - 1, 0.0) / (s.size() - 2);
    }

    double min_val() const {
        if (samples_.empty()) return 0.0;
        return *std::min_element(samples_.begin(), samples_.end());
    }

    double max_val() const {
        if (samples_.empty()) return 0.0;
        return *std::max_element(samples_.begin(), samples_.end());
    }

    double cv() const {
        double avg = mean();
        if (avg == 0.0) return 0.0;
        double var = 0.0;
        for (double v : samples_) var += (v - avg) * (v - avg);
        var /= samples_.size();
        return std::sqrt(var) / avg;
    }

    size_t count() const { return samples_.size(); }

private:
    std::vector<double> samples_;
};

struct TestConfig {
    int warmup_iterations = 5;
    int measure_iterations = 10;
    int device_id = 0;
    bool verbose = true;
};

// RAII wrapper for ACL device memory
class AclDeviceBuffer {
public:
    AclDeviceBuffer() : data_(nullptr), size_(0) {}
    explicit AclDeviceBuffer(size_t bytes) : data_(nullptr), size_(bytes) {
        if (bytes > 0) {
            ACL_CHECK(aclrtMalloc(&data_, size_, ACL_MEM_MALLOC_HUGE_FIRST));
        }
    }
    ~AclDeviceBuffer() {
        if (data_) aclrtFree(data_);
    }

    AclDeviceBuffer(const AclDeviceBuffer&) = delete;
    AclDeviceBuffer& operator=(const AclDeviceBuffer&) = delete;

    AclDeviceBuffer(AclDeviceBuffer&& o) noexcept : data_(o.data_), size_(o.size_) {
        o.data_ = nullptr; o.size_ = 0;
    }
    AclDeviceBuffer& operator=(AclDeviceBuffer&& o) noexcept {
        if (this != &o) {
            if (data_) aclrtFree(data_);
            data_ = o.data_; size_ = o.size_;
            o.data_ = nullptr; o.size_ = 0;
        }
        return *this;
    }

    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t size() const { return size_; }
    bool is_valid() const { return data_ != nullptr; }

private:
    void* data_;
    size_t size_;
};

// RAII wrapper for ACL host (pinned) memory
class AclHostBuffer {
public:
    AclHostBuffer() : data_(nullptr), size_(0) {}
    explicit AclHostBuffer(size_t bytes) : data_(nullptr), size_(bytes) {
        if (bytes > 0) {
            ACL_CHECK(aclrtMallocHost(&data_, size_));
        }
    }
    ~AclHostBuffer() {
        if (data_) aclrtFreeHost(data_);
    }

    AclHostBuffer(const AclHostBuffer&) = delete;
    AclHostBuffer& operator=(const AclHostBuffer&) = delete;

    AclHostBuffer(AclHostBuffer&& o) noexcept : data_(o.data_), size_(o.size_) {
        o.data_ = nullptr; o.size_ = 0;
    }
    AclHostBuffer& operator=(AclHostBuffer&& o) noexcept {
        if (this != &o) {
            if (data_) aclrtFreeHost(data_);
            data_ = o.data_; size_ = o.size_;
            o.data_ = nullptr; o.size_ = 0;
        }
        return *this;
    }

    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t size() const { return size_; }
    bool is_valid() const { return data_ != nullptr; }

private:
    void* data_;
    size_t size_;
};

// RAII wrapper for ACL stream
class AclStream {
public:
    AclStream() : stream_(nullptr) {
        ACL_CHECK(aclrtCreateStream(&stream_));
    }
    ~AclStream() {
        if (stream_) aclrtDestroyStream(stream_);
    }

    AclStream(const AclStream&) = delete;
    AclStream& operator=(const AclStream&) = delete;

    void sync() {
        ACL_CHECK(aclrtSynchronizeStream(stream_));
    }

    aclrtStream get() const { return stream_; }

private:
    aclrtStream stream_;
};

// Device info
struct NpuDeviceInfo {
    static void print(int device_id = 0) {
        size_t free_mem = 0, total_mem = 0;
        ACL_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem));

        std::cout << "Device " << device_id << ": Ascend NPU\n";
        std::cout << "  Total HBM Memory:  "
                  << (total_mem / 1024.0 / 1024.0 / 1024.0) << " GB\n";
        std::cout << "  Free HBM Memory:   "
                  << (free_mem / 1024.0 / 1024.0 / 1024.0) << " GB\n";
    }
};

inline int get_device_count() {
    uint32_t count = 0;
    ACL_CHECK(aclrtGetDeviceCount(&count));
    return static_cast<int>(count);
}

// ACL initialization guard (call once per process)
class AclInitGuard {
public:
    AclInitGuard() {
        ACL_CHECK(aclInit(nullptr));
    }
    ~AclInitGuard() {
        aclFinalize();
    }
};

} // namespace npu_perf
