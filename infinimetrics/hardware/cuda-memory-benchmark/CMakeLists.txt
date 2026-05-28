cmake_minimum_required(VERSION 3.18)

# Platform detection
if(NOT DEFINED PLATFORM)
    message(FATAL_ERROR "PLATFORM is not set. Use -DPLATFORM=cuda, -DPLATFORM=metax, or -DPLATFORM=corex")
endif()

if(PLATFORM STREQUAL "cuda")
    project(CudaPerfSuite LANGUAGES CXX CUDA VERSION 1.0.0)
elseif(PLATFORM STREQUAL "metax")
    project(CudaPerfSuite LANGUAGES CXX VERSION 1.0.0)
elseif(PLATFORM STREQUAL "corex")
    project(CudaPerfSuite LANGUAGES CXX CUDA VERSION 1.0.0)
else()
    message(FATAL_ERROR "Unsupported PLATFORM: ${PLATFORM}. Use 'cuda', 'metax', or 'corex'")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(CUDA_SOURCES
    src/main.cu
)

set(CXX_SOURCES
    # Add any .cpp files here if needed
)

# ---- Platform-specific settings ----

if(PLATFORM STREQUAL "cuda")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING "CUDA architectures")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -lineinfo")

    add_executable(cuda_perf_suite ${CUDA_SOURCES} ${CXX_SOURCES})

elseif(PLATFORM STREQUAL "metax")
    # Use cu-bridge compiler
    set(CMAKE_CXX_COMPILER "$ENV{CUDA_PATH}/bin/nvcc")
    set(CMAKE_C_COMPILER "$ENV{CUDA_PATH}/bin/nvcc")

    # Treat .cu as CXX for cu-bridge
    set_source_files_properties(src/main.cu PROPERTIES LANGUAGE CXX)

    add_executable(cuda_perf_suite ${CUDA_SOURCES} ${CXX_SOURCES})

elseif(PLATFORM STREQUAL "corex")
    # CoreX adapted CMake handles CUDA compiler automatically (clang++)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES "ivcore20" CACHE STRING "CoreX GPU architectures")

    include_directories($ENV{COREX_PATH}/include)
    link_directories($ENV{COREX_PATH}/lib64)

    add_executable(cuda_perf_suite ${CUDA_SOURCES} ${CXX_SOURCES})
endif()

# Link libraries
target_link_libraries(cuda_perf_suite ${CMAKE_THREADS_LIB_INIT})

# Installation
install(TARGETS cuda_perf_suite RUNTIME DESTINATION bin)

# Print configuration
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Project:         ${PROJECT_NAME}")
message(STATUS "  Version:         ${PROJECT_VERSION}")
message(STATUS "  Platform:        ${PLATFORM}")
message(STATUS "  Build type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:    ${CMAKE_CXX_STANDARD}")
if(PLATFORM STREQUAL "cuda")
    message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
elseif(PLATFORM STREQUAL "corex")
    message(STATUS "  CoreX arch:      ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "")
