cmake_minimum_required(VERSION 3.18)
project(NpuPerfSuite VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find Ascend CANN Toolkit
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME $ENV{ASCEND_HOME_PATH})
elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(ASCEND_HOME $ENV{ASCEND_TOOLKIT_HOME})
else()
    set(ASCEND_HOME "/usr/local/Ascend/ascend-toolkit/latest")
endif()
message(STATUS "ASCEND_HOME: ${ASCEND_HOME}")

# Detect architecture for library path
execute_process(
    COMMAND uname -m
    OUTPUT_VARIABLE ARCH_TYPE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(ARCH_TYPE STREQUAL "aarch64")
    set(ARCH_DIR "aarch64-linux")
else()
    set(ARCH_DIR "x86_64-linux")
endif()

set(ASCEND_INCLUDE_DIR "${ASCEND_HOME}/${ARCH_DIR}/include")
set(ASCEND_LIB_DIR "${ASCEND_HOME}/${ARCH_DIR}/lib64")

# Verify headers exist
find_path(ACL_INCLUDE_DIR NAMES acl/acl.h HINTS ${ASCEND_INCLUDE_DIR})
if(NOT ACL_INCLUDE_DIR)
    message(FATAL_ERROR "acl/acl.h not found in ${ASCEND_INCLUDE_DIR}. "
                        "Set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME.")
endif()

# Find Ascend runtime and operator API libraries
find_library(ASCENDCL_LIB ascendcl HINTS ${ASCEND_LIB_DIR})
if(NOT ASCENDCL_LIB)
    message(FATAL_ERROR "libascendcl.so not found in ${ASCEND_LIB_DIR}")
endif()

find_library(OPAPI_LIB opapi HINTS ${ASCEND_LIB_DIR})
if(NOT OPAPI_LIB)
    message(FATAL_ERROR "libopapi.so not found in ${ASCEND_LIB_DIR}")
endif()

find_library(NNOPBASE_LIB nnopbase HINTS ${ASCEND_LIB_DIR})
if(NOT NNOPBASE_LIB)
    message(FATAL_ERROR "libnnopbase.so not found in ${ASCEND_LIB_DIR}")
endif()

add_executable(npu_perf_suite src/main.cc)
target_include_directories(npu_perf_suite PRIVATE
    ${ACL_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(npu_perf_suite
    ${ASCENDCL_LIB}
    ${OPAPI_LIB}
    ${NNOPBASE_LIB}
    pthread
)
target_compile_options(npu_perf_suite PRIVATE -Wall -O3)

install(TARGETS npu_perf_suite RUNTIME DESTINATION bin)

message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Project:    ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "  Build:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  Arch:       ${ARCH_DIR}")
message(STATUS "  Include:    ${ACL_INCLUDE_DIR}")
message(STATUS "  ascendcl:   ${ASCENDCL_LIB}")
message(STATUS "  opapi:      ${OPAPI_LIB}")
message(STATUS "  nnopbase:   ${NNOPBASE_LIB}")
message(STATUS "")
