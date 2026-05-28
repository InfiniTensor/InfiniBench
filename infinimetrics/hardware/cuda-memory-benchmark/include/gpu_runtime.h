#pragma once

/// @file gpu_runtime.h
/// Unified GPU runtime header for cross-platform GPU computing.
///
/// When GPU_PLATFORM_HIP is defined (e.g. Hygon DCU / AMD ROCm),
/// CUDA API names are mapped to their HIP equivalents via macros.
/// Otherwise, the native CUDA runtime is used as-is.

#ifdef GPU_PLATFORM_HIP
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
