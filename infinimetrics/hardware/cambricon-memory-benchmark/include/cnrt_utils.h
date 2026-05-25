#pragma once

#include <cnrt.h>
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

namespace mlu_perf {

// Own check macro with exceptions (cnrt.h's CNRT_CHECK calls exit)
#define MLU_CHECK(call) \
    do { \
        cnrtRet_t ret = call; \
        if (ret != CNRT_RET_SUCCESS) { \
            std::ostringstream oss; \
            oss << "CNRT error at " << __FILE__ << ":" << __LINE__ \
                << ": " << cnrtGetErrorStr(ret) \
                << " (code=" << ret << ")"; \
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

// Statistics
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

    double cv() const {
        double avg = mean();
        if (avg == 0.0) return 0.0;
        double var = 0.0;
        for (double v : samples_) var += (v - avg) * (v - avg);
        var /= samples_.size();
        return std::sqrt(var) / avg;
    }

private:
    std::vector<double> samples_;
};

struct TestConfig {
    int warmup_iterations = 5;
    int measure_iterations = 10;
    bool verbose = true;
};

// Device info
struct MluDeviceInfo {
    static void print(int device_id = 0) {
        cnrtDeviceProp_t prop;
        MLU_CHECK(cnrtGetDeviceProperties(&prop, device_id));

        std::cout << "Device " << device_id << ": " << prop.name << "\n";
        std::cout << "  Total Memory:       "
                  << prop.totalMem << " MB\n";
        std::cout << "  L2 Cache Size:      "
                  << (prop.maxL2CacheSize / 1024.0) << " KB\n";
        std::cout << "  Clusters:           " << prop.clusterCount << "\n";
        std::cout << "  Cores per Cluster:  " << prop.McorePerCluster << "\n";
        std::cout << "  Max Block Dim X:    " << prop.maxDim[0] << "\n";
    }
};

inline int get_device_count() {
    unsigned int count;
    MLU_CHECK(cnrtGetDeviceCount(&count));
    return static_cast<int>(count);
}

} // namespace mlu_perf
