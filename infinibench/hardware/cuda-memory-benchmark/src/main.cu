#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <memory>

// Include all test suites
#include "performance_test.h"
#include "memory_bandwidth_test.h"
#include "stream_benchmark.h"
#include "cache_benchmark.h"

using namespace cuda_perf;

void print_banner() {
    std::cout << R"(
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║        CUDA Performance Benchmark Suite v1.0                  ║
║                                                               ║
║        Comprehensive GPU Memory & Cache Testing               ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
)" << std::endl;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --all                    Run all tests (default)\n";
    std::cout << "  --memory                 Run memory bandwidth tests only\n";
    std::cout << "  --stream                 Run STREAM benchmark only\n";
    std::cout << "  --cache                  Run cache benchmarks only\n";
    std::cout << "  --device <id>            Specify CUDA device ID (default: 0)\n";
    std::cout << "  --iterations <n>         Number of measurement iterations (default: 10)\n";
    std::cout << "  --array-size <size>      Array size for STREAM test (default: 67108864)\n";
    std::cout << "  --quiet                  Reduce output verbosity\n";
    std::cout << "  --help                   Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " --all\n";
    std::cout << "  " << program_name << " --stream --array-size 134217728\n";
    std::cout << "  " << program_name << " --memory\n";
    std::cout << "  " << program_name << " --cache\n";
}

struct Config {
    bool run_all = true;
    bool run_memory = false;
    bool run_stream = false;
    bool run_cache = false;
    int device_id = 0;
    int iterations = 10;
    size_t array_size = 67108864;  // 64M elements (512 MB for double)
    bool verbose = true;
};

Config parse_arguments(int argc, char* argv[]) {
    Config config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            exit(0);
        }
        else if (arg == "--all") {
            config.run_all = true;
        }
        else if (arg == "--memory") {
            config.run_all = false;
            config.run_memory = true;
        }
        else if (arg == "--stream") {
            config.run_all = false;
            config.run_stream = true;
        }
        else if (arg == "--cache") {
            config.run_all = false;
            config.run_cache = true;
        }
        else if (arg == "--device" && i + 1 < argc) {
            config.device_id = std::atoi(argv[++i]);
        }
        else if (arg == "--iterations" && i + 1 < argc) {
            config.iterations = std::atoi(argv[++i]);
        }
        else if (arg == "--array-size" && i + 1 < argc) {
            config.array_size = static_cast<size_t>(std::atoll(argv[++i]));
        }
        else if (arg == "--quiet") {
            config.verbose = false;
        }
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            exit(1);
        }
    }

    return config;
}

void print_system_info() {
    std::cout << "\n=== System Information ===\n";

    int device_count = get_device_count();
    std::cout << "CUDA Devices: " << device_count << "\n";

    for (int i = 0; i < device_count; ++i) {
        CudaDeviceInfo info = CudaDeviceInfo::get(i);
        std::cout << "\n";
        info.print();
    }

    // Print runtime information
    int runtime_version;
    cudaRuntimeGetVersion(&runtime_version);
    std::cout << "\nCUDA Runtime Version: "
              << runtime_version / 1000 << "."
              << (runtime_version % 1000) / 10 << "\n";

    int driver_version;
    cudaDriverGetVersion(&driver_version);
    std::cout << "CUDA Driver Version:  "
              << driver_version / 1000 << "."
              << (driver_version % 1000) / 10 << "\n";
}

int main(int argc, char* argv[]) {
    try {
        print_banner();

        // Parse command line arguments
        Config config = parse_arguments(argc, argv);

        // Print system information
        print_system_info();

        // Set device
        int device_count = get_device_count();
        if (config.device_id >= device_count) {
            std::cerr << "Error: Device ID " << config.device_id
                      << " not available (only " << device_count << " devices)\n";
            return 1;
        }
        CUDA_CHECK(cudaSetDevice(config.device_id));

        // Configure test settings
        TestConfig test_config;
        test_config.warmup_iterations = 5;  
        test_config.measurement_iterations = config.iterations;
        test_config.verbose = config.verbose;

        std::cout << "\n=== Test Configuration ===\n";
        std::cout << "Device ID:         " << config.device_id << "\n";
        std::cout << "Iterations:        " << config.iterations << "\n";
        std::cout << "Stream array size: " << config.array_size << " elements ("
                  << (config.array_size * sizeof(double) / 1024.0 / 1024.0) << " MB)\n";

        // Run selected tests
        if (config.run_all || config.run_memory) {
            std::cout << "\n";
            std::cout << "██████████████████████████████████████████████████████\n";
            std::cout << "█       MEMORY BANDWIDTH TESTS                       █\n";
            std::cout << "██████████████████████████████████████████████████████\n";

            // Host to Device (Pinned)
            MemoryCopySweepTest h2d_pinned(CopyDirection::HOST_TO_DEVICE, true);
            h2d_pinned.execute(test_config);

            // Device to Host (Pinned)
            MemoryCopySweepTest d2h_pinned(CopyDirection::DEVICE_TO_HOST, true);
            d2h_pinned.execute(test_config);

            // Device to Device
            MemoryCopySweepTest d2d(CopyDirection::DEVICE_TO_DEVICE, true);
            d2d.execute(test_config);

            // Bidirectional (H2D + D2H concurrently)
            MemoryCopySweepTest bidi(CopyDirection::BIDIRECTIONAL, true);
            bidi.execute(test_config);
        }

        if (config.run_all || config.run_stream) {
            std::cout << "\n";
            std::cout << "██████████████████████████████████████████████████████\n";
            std::cout << "█           STREAM BENCHMARK                         █\n";
            std::cout << "██████████████████████████████████████████████████████\n";

            StreamBenchmarkSuite stream_suite;
            stream_suite.execute(config.array_size, test_config);
        }

        if (config.run_all || config.run_cache) {
            std::cout << "\n";
            std::cout << "██████████████████████████████████████████████████████\n";
            std::cout << "█           CACHE BENCHMARKS                         █\n";
            std::cout << "██████████████████████████████████████████████████████\n";

            // L1 Cache test
            CacheBenchmarkSuite l1_test(CacheTarget::L1);
            l1_test.execute(test_config);

            // L2 Cache test
            CacheBenchmarkSuite l2_test(CacheTarget::L2);
            l2_test.execute(test_config);
        }

        // Final summary
        std::cout << "\n";
        std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                                                               ║\n";
        std::cout << "║              All Tests Completed Successfully                 ║\n";
        std::cout << "║                                                               ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\nERROR: Unknown exception occurred" << std::endl;
        return 1;
    }
}
