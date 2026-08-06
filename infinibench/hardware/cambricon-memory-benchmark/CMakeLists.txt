cmake_minimum_required(VERSION 3.18)

# Find NEUWARE / CNToolkit
if(DEFINED ENV{NEUWARE_HOME})
    set(NEUWARE_HOME $ENV{NEUWARE_HOME})
else()
    set(NEUWARE_HOME "/usr/local/neuware")
endif()
message(STATUS "NEUWARE_HOME: ${NEUWARE_HOME}")

# Find cncc compiler (used as C++ compiler for Cambricon)
find_program(CNCC cncc HINTS ${NEUWARE_HOME}/bin)
if(NOT CNCC)
    message(FATAL_ERROR "cncc not found. Set NEUWARE_HOME.")
endif()
message(STATUS "Found cncc: ${CNCC}")

set(CMAKE_CXX_COMPILER "${CNCC}")
project(MluPerfSuite VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(DEFINED ENV{MLU_ARCH})
    set(MLU_ARCH $ENV{MLU_ARCH})
else()
    set(MLU_ARCH "mtp_592")
endif()

# Find CNRT library
find_library(CNRT_LIB cnrt HINTS ${NEUWARE_HOME}/lib64 ${NEUWARE_HOME}/lib)
if(NOT CNRT_LIB)
    message(FATAL_ERROR "libcnrt not found in ${NEUWARE_HOME}")
endif()

# Find CNRT headers
find_path(CNRT_INCLUDE_DIR NAMES cnrt.h HINTS ${NEUWARE_HOME}/include)
if(NOT CNRT_INCLUDE_DIR)
    message(FATAL_ERROR "cnrt.h not found in ${NEUWARE_HOME}/include")
endif()

add_executable(mlu_perf_suite src/main.mlu)
target_include_directories(mlu_perf_suite PRIVATE
    ${CNRT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(mlu_perf_suite PRIVATE "--bang-mlu-arch=${MLU_ARCH}")
target_link_libraries(mlu_perf_suite ${CNRT_LIB} stdc++ m pthread)

set_source_files_properties(src/main.mlu PROPERTIES LANGUAGE CXX)

install(TARGETS mlu_perf_suite RUNTIME DESTINATION bin)

message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Project:    ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "  Build:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  cncc:       ${CNCC}")
message(STATUS "  MLU arch:   ${MLU_ARCH}")
message(STATUS "  CNRT:       ${CNRT_LIB}")
message(STATUS "")
