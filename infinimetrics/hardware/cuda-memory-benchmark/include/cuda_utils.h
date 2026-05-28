#pragma once

#include "gpu_runtime.h"
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace cuda_perf {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::ostringstream oss; \
            oss << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                << ": " << cudaGetErrorString(error); \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

// RAII wrapper for CUDA device memory
template<typename T>
class CudaDeviceBuffer {
public:
    explicit CudaDeviceBuffer(size_t count = 0)
        : data_(nullptr), size_(count * sizeof(T)) {
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(&data_, size_));
        }
    }

    ~CudaDeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    // Disable copy
    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

    // Enable move
    CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                cudaFree(data_);
            }
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void resize(size_t count) {
        size_t new_size = count * sizeof(T);
        if (new_size != size_) {
            if (data_ != nullptr) {
                CUDA_CHECK(cudaFree(data_));
            }
            size_ = new_size;
            if (count > 0) {
                CUDA_CHECK(cudaMalloc(&data_, size_));
            } else {
                data_ = nullptr;
            }
        }
    }

    T* data() { return static_cast<T*>(data_); }
    const T* data() const { return static_cast<const T*>(data_); }
    size_t size_bytes() const { return size_; }
    size_t count() const { return size_ / sizeof(T); }
    bool is_valid() const { return data_ != nullptr; }

private:
    void* data_;
    size_t size_;
};

// RAII wrapper for CUDA pinned host memory
template<typename T>
class CudaPinnedBuffer {
public:
    explicit CudaPinnedBuffer(size_t count = 0)
        : data_(nullptr), size_(count * sizeof(T)) {
        if (count > 0) {
            CUDA_CHECK(cudaMallocHost(&data_, size_));
        }
    }

    ~CudaPinnedBuffer() {
        if (data_ != nullptr) {
            cudaFreeHost(data_);
        }
    }

    // Disable copy
    CudaPinnedBuffer(const CudaPinnedBuffer&) = delete;
    CudaPinnedBuffer& operator=(const CudaPinnedBuffer&) = delete;

    // Enable move
    CudaPinnedBuffer(CudaPinnedBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    CudaPinnedBuffer& operator=(CudaPinnedBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                CUDA_CHECK(cudaFreeHost(data_));
            }
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void resize(size_t count) {
        size_t new_size = count * sizeof(T);
        if (new_size != size_) {
            if (data_ != nullptr) {
                CUDA_CHECK(cudaFreeHost(data_));
            }
            size_ = new_size;
            if (count > 0) {
                CUDA_CHECK(cudaMallocHost(&data_, size_));
            } else {
                data_ = nullptr;
            }
        }
    }

    T* data() { return static_cast<T*>(data_); }
    const T* data() const { return static_cast<const T*>(data_); }
    size_t size_bytes() const { return size_; }
    size_t count() const { return size_ / sizeof(T); }
    bool is_valid() const { return data_ != nullptr; }

private:
    void* data_;
    size_t size_;
};

// RAII wrapper for CUDA stream
class CudaStream {
public:
    CudaStream() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~CudaStream() {
        cudaStreamDestroy(stream_);
    }

    // Disable copy
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_;
};

// RAII wrapper for CUDA event
class CudaEvent {
public:
    CudaEvent() {
        CUDA_CHECK(cudaEventCreate(&event_));
    }

    ~CudaEvent() {
        cudaEventDestroy(event_);
    }

    // Disable copy
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    void record(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }

    void synchronize() {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }

    float elapsed_time(const CudaEvent& start) const {
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, event_));
        return ms;
    }

    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_;
};

// CUDA device information
struct CudaDeviceInfo {
    int device_id;
    std::string name;
    int compute_capability_major;
    int compute_capability_minor;
    size_t total_global_mem;
    size_t shared_mem_per_block;
    int max_threads_per_block;
    int multi_processor_count;
    int clock_rate;
    size_t l2_cache_size;
    int max_threads_per_multiprocessor;

    static CudaDeviceInfo get(int device_id = 0) {
        CudaDeviceInfo info;
        info.device_id = device_id;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

        info.name = prop.name;
        info.compute_capability_major = prop.major;
        info.compute_capability_minor = prop.minor;
        info.total_global_mem = prop.totalGlobalMem;
        info.shared_mem_per_block = prop.sharedMemPerBlock;
        info.max_threads_per_block = prop.maxThreadsPerBlock;
        info.multi_processor_count = prop.multiProcessorCount;
#if defined(GPU_PLATFORM_MUSA)
        info.clock_rate = 0;
#elif CUDART_VERSION >= 11000
        info.clock_rate = 0;
#else
        info.clock_rate = prop.clockRate;
#endif
        info.l2_cache_size = prop.l2CacheSize;
        info.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;

        return info;
    }

    void print() const {
        std::cout << "Device " << device_id << ": " << name << "\n";
        std::cout << "  Compute Capability: " << compute_capability_major
                  << "." << compute_capability_minor << "\n";
        std::cout << "  Total Global Memory: "
                  << (total_global_mem / 1024.0 / 1024.0 / 1024.0) << " GB\n";
        std::cout << "  L2 Cache Size: "
                  << (l2_cache_size / 1024.0) << " KB\n";
        std::cout << "  Multiprocessors: " << multi_processor_count << "\n";
        std::cout << "  Max Threads per Block: " << max_threads_per_block << "\n";
    }
};

// Get number of CUDA devices
inline int get_device_count() {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

} // namespace cuda_perf
