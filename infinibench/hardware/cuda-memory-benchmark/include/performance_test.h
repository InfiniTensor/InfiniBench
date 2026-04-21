#pragma once

#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <memory>

namespace cuda_perf {

// High-resolution timer
class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    Timer() : start_time_(Clock::now()) {}

    void reset() {
        start_time_ = Clock::now();
    }

    double elapsed_seconds() const {
        auto end_time = Clock::now();
        std::chrono::duration<double> diff = end_time - start_time_;
        return diff.count();
    }

    double elapsed_milliseconds() const {
        return elapsed_seconds() * 1000.0;
    }

private:
    TimePoint start_time_;
};

// Statistics collector for performance measurements
class PerformanceMetrics {
public:
    void add_sample(double value) {
        samples_.push_back(value);
    }

    double average() const {
        if (samples_.empty()) return 0.0;
        return std::accumulate(samples_.begin(), samples_.end(), 0.0) / samples_.size();
    }

    double median() const {
        if (samples_.empty()) return 0.0;
        auto sorted = samples_;
        std::sort(sorted.begin(), sorted.end());
        size_t n = sorted.size();
        if (n % 2 == 0) {
            return (sorted[n/2 - 1] + sorted[n/2]) / 2.0;
        }
        return sorted[n/2];
    }

    double min_value() const {
        if (samples_.empty()) return 0.0;
        return *std::min_element(samples_.begin(), samples_.end());
    }

    double max_value() const {
        if (samples_.empty()) return 0.0;
        return *std::max_element(samples_.begin(), samples_.end());
    }

    // Trimmed mean (remove min and max)
    double trimmed_mean() const {
        if (samples_.size() <= 2) return average();
        auto sorted = samples_;
        std::sort(sorted.begin(), sorted.end());
        double sum = std::accumulate(sorted.begin() + 1, sorted.end() - 1, 0.0);
        return sum / (sorted.size() - 2);
    }

    // Coefficient of variation (relative standard deviation)
    double coefficient_of_variation() const {
        double avg = average();
        if (avg == 0.0) return 0.0;
        double variance = 0.0;
        for (double sample : samples_) {
            double diff = sample - avg;
            variance += diff * diff;
        }
        variance /= samples_.size();
        return std::sqrt(variance) / avg;
    }

    size_t sample_count() const {
        return samples_.size();
    }

    void clear() {
        samples_.clear();
    }

private:
    std::vector<double> samples_;
};

// Configuration for running tests
struct TestConfig {
    size_t warmup_iterations = 5;  // Ensure GPU is warmed up
    size_t measurement_iterations = 10;
    bool verbose = true;
    bool output_csv = false;
};

// Base class for all performance tests
class PerformanceTest {
public:
    virtual ~PerformanceTest() = default;

    virtual std::string name() const = 0;
    virtual std::string description() const = 0;

    virtual bool initialize() = 0;
    virtual void cleanup() = 0;

    virtual void run_warmup() = 0;
    virtual double run_single_test() = 0;

    void execute(const TestConfig& config = TestConfig()) {
        print_header();

        if (!initialize()) {
            std::cerr << "ERROR: Failed to initialize test " << name() << std::endl;
            return;
        }

        // Warmup phase
        if (config.verbose) {
            std::cout << "  Warming up..." << std::endl;
        }
        for (size_t i = 0; i < config.warmup_iterations; ++i) {
            run_warmup();
        }

        // Measurement phase
        PerformanceMetrics metrics;
        if (config.verbose) {
            std::cout << "  Running " << config.measurement_iterations
                      << " measurements..." << std::endl;
        }

        for (size_t i = 0; i < config.measurement_iterations; ++i) {
            double time_ms = run_single_test();
            metrics.add_sample(time_ms);
        }

        // Print results
        print_results(metrics);

        cleanup();
    }

protected:
    virtual void print_header() {
        std::cout << "\n========================================\n";
        std::cout << "Test: " << name() << "\n";
        std::cout << "Description: " << description() << "\n";
        std::cout << "========================================\n";
    }

    virtual void print_results(const PerformanceMetrics& metrics) {
        std::cout << "\nResults:\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Average time:      " << metrics.average() << " ms\n";
        std::cout << "  Median time:       " << metrics.median() << " ms\n";
        std::cout << "  Min time:          " << metrics.min_value() << " ms\n";
        std::cout << "  Max time:          " << metrics.max_value() << " ms\n";
        std::cout << "  Trimmed mean:      " << metrics.trimmed_mean() << " ms\n";
        std::cout << "  Coeff. of variation: " << std::setprecision(2)
                  << (metrics.coefficient_of_variation() * 100.0) << "%\n";
    }
};

} // namespace cuda_perf
