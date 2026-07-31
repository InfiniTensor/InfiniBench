#pragma once

/// @file gpu_runtime.h
/// Unified GPU runtime header for cross-platform GPU computing.
///
/// - GPU_PLATFORM_MUSA: Moore Threads MUSA  (maps cuda* -> musa*)
/// - GPU_PLATFORM_HIP:  Hygon DCU / AMD ROCm (maps cuda* -> hip*)
/// - Otherwise:         Native CUDA (NVIDIA, MetaX cu-bridge, CoreX)

#ifdef GPU_PLATFORM_MUSA
// ---- MUSA backend (Moore Threads) ----
#include <musa_runtime.h>

// --- Error types ---
#define cudaError_t              musaError_t
#define cudaSuccess              musaSuccess
#define cudaGetErrorString       musaGetErrorString
#define cudaGetLastError         musaGetLastError

// --- Device management ---
#define cudaSetDevice            musaSetDevice
#define cudaGetDevice            musaGetDevice
#define cudaGetDeviceCount       musaGetDeviceCount
#define cudaGetDeviceProperties  musaGetDeviceProperties
#define cudaDeviceSynchronize    musaDeviceSynchronize
#define cudaDeviceProp           musaDeviceProp

// --- Memory management ---
#define cudaMalloc               musaMalloc
#define cudaFree                 musaFree
#define cudaMallocHost           musaMallocHost
#define cudaFreeHost             musaFreeHost
#define cudaMemcpy               musaMemcpy
#define cudaMemcpyAsync          musaMemcpyAsync
#define cudaMemset               musaMemset

// --- Stream ---
#define cudaStream_t             musaStream_t
#define cudaStreamCreate         musaStreamCreate
#define cudaStreamDestroy        musaStreamDestroy
#define cudaStreamSynchronize    musaStreamSynchronize

// --- Event ---
#define cudaEvent_t              musaEvent_t
#define cudaEventCreate          musaEventCreate
#define cudaEventDestroy         musaEventDestroy
#define cudaEventRecord          musaEventRecord
#define cudaEventSynchronize     musaEventSynchronize
#define cudaEventElapsedTime     musaEventElapsedTime

// --- Memory copy constants ---
#define cudaMemcpyHostToDevice   musaMemcpyHostToDevice
#define cudaMemcpyDeviceToHost   musaMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice musaMemcpyDeviceToDevice

// --- Version ---
#define cudaRuntimeGetVersion    musaRuntimeGetVersion
#define cudaDriverGetVersion     musaDriverGetVersion

#elif defined(GPU_PLATFORM_HIP)
// ---- HIP backend (Hygon DCU, AMD ROCm) ----
#include <hip/hip_runtime.h>

// --- Error types ---
#define cudaError_t       hipError_t
#define cudaSuccess        hipSuccess
#define cudaGetErrorString hipGetErrorString

// --- Memory management ---
#define cudaMalloc         hipMalloc
#define cudaFree           hipFree
#define cudaMallocHost     hipMallocHost
#define cudaFreeHost       hipFreeHost
#define cudaMemset         hipMemset

// --- Stream ---
#define cudaStream_t       hipStream_t
#define cudaStreamCreate   hipStreamCreate
#define cudaStreamDestroy  hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize

// --- Event ---
#define cudaEvent_t        hipEvent_t
#define cudaEventCreate    hipEventCreate
#define cudaEventDestroy   hipEventDestroy
#define cudaEventRecord    hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime

// --- Device management ---
#define cudaSetDevice          hipSetDevice
#define cudaGetDevice          hipGetDevice
#define cudaGetDeviceCount     hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceSynchronize  hipDeviceSynchronize
#define cudaGetLastError       hipGetLastError

// --- Device property struct ---
#define cudaDeviceProp         hipDeviceProp_t

// --- Memory copy ---
#define cudaMemcpy             hipMemcpy
#define cudaMemcpyAsync        hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice

// --- Version ---
#define cudaRuntimeGetVersion  hipRuntimeGetVersion
#define cudaDriverGetVersion   hipDriverGetVersion

// Provide CUDART_VERSION equivalent for HIP
#ifndef CUDART_VERSION
#define CUDART_VERSION (HIP_VERSION_MAJOR * 1000 + HIP_VERSION_MINOR * 10)
#endif

#else
// ---- CUDA backend (NVIDIA, MetaX cu-bridge, CoreX) ----
#include <cuda_runtime.h>
#endif
