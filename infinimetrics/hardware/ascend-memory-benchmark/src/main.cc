#include <iostream>
#include <string>
#include "acl_utils.h"
#include "memory_bandwidth_test.h"
#include "stream_benchmark.h"
#include "cache_benchmark.h"

using namespace npu_perf;

void print_banner() {
    std::cout << R"(
================================================================
        NPU Performance Benchmark Suite v1.0
        Ascend Memory & Cache Testing
================================================================
)" << std::endl;
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  --all              Run all tests (default)\n"
              << "  --memory           Run memory bandwidth tests only\n"
              << "  --stream           Run STREAM benchmark only\n"
              << "  --cache            Run memory hierarchy sweep test\n"
              << "  --device <id>      Specify NPU device ID (default: 0)\n"
              << "  --iterations <n>   Number of measurement iterations (default: 10)\n"
              << "  --array-size <n>   Array size for STREAM test (default: 67108864)\n"
              << "  --help             Show this help\n";
}

struct Config {
    bool run_all = true;
    bool run_memory = false;
    bool run_stream = false;
    bool run_cache = false;
    int device_id = 0;
    int iterations = 10;
    size_t array_size = 67108864;
};

Config parse_args(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { print_usage(argv[0]); exit(0); }
        else if (arg == "--all") { cfg.run_all = true; }
        else if (arg == "--memory") { cfg.run_all = false; cfg.run_memory = true; }
        else if (arg == "--stream") { cfg.run_all = false; cfg.run_stream = true; }
        else if (arg == "--cache") { cfg.run_all = false; cfg.run_cache = true; }
        else if (arg == "--device" && i + 1 < argc) { cfg.device_id = std::atoi(argv[++i]); }
        else if (arg == "--iterations" && i + 1 < argc) { cfg.iterations = std::atoi(argv[++i]); }
        else if (arg == "--array-size" && i + 1 < argc) { cfg.array_size = std::atoll(argv[++i]); }
        else { std::cerr << "Unknown option: " << arg << "\n"; print_usage(argv[0]); exit(1); }
    }
    return cfg;
}

int main(int argc, char* argv[]) {
    try {
        print_banner();
        Config cfg = parse_args(argc, argv);

        // Initialize ACL
        AclInitGuard acl_guard;

        // System info
        std::cout << "=== System Information ===\n";
        int dev_count = get_device_count();
        std::cout << "NPU Devices: " << dev_count << "\n";

        if (cfg.device_id >= dev_count) {
            std::cerr << "Error: Device ID " << cfg.device_id << " not available\n";
            return 1;
        }
        // Set device FIRST — aclrtGetMemInfo requires an active context
        ACL_CHECK(aclrtSetDevice(cfg.device_id));

        for (int i = 0; i < dev_count; ++i) {
            std::cout << "\n";
            NpuDeviceInfo::print(i);
        }
        std::cout << "\n";

        TestConfig tc;
        tc.warmup_iterations = 5;
        tc.measure_iterations = cfg.iterations;

        std::cout << "=== Test Configuration ===\n"
                  << "Device ID:         " << cfg.device_id << "\n"
                  << "Iterations:        " << cfg.iterations << "\n"
                  << "Stream array size: " << cfg.array_size
                  << " elements (" << cfg.array_size * sizeof(float) / 1024.0 / 1024.0 << " MB)\n";

        if (cfg.run_all || cfg.run_memory) {
            MemoryBandwidthTest test;
            test.execute(tc);
        }

        if (cfg.run_all || cfg.run_stream) {
            StreamBenchmarkTest test;
            test.execute(cfg.array_size, tc);
        }

        if (cfg.run_all || cfg.run_cache) {
            CacheBenchmarkTest test;
            test.execute(tc);
        }

        ACL_CHECK(aclrtResetDevice(cfg.device_id));

        std::cout << "\nAll tests completed successfully.\n\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << "\n";
        return 1;
    }
}
