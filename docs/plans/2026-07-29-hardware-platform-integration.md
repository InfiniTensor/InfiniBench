# Hardware Platform Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Merge `hardware_adapt` and `cambricon_test` into `master` while preserving existing CUDA adapter behavior and adding testable support for MetaX, Iluvatar, Hygon, Moore Threads, Cambricon, and Ascend.

**Architecture:** Keep `HardwareTestAdapter` as the public entry point and preserve its existing constructor, methods, CUDA command shape, output parsing, and metric names. Add a small platform registry behind the adapter for platform-specific detection, build commands, binary paths, capabilities, and output parsers. CUDA-compatible platforms reuse the CUDA benchmark source; Cambricon and Ascend retain native benchmark directories.

**Tech Stack:** Python 3.10, pytest, C++17, CUDA/CMake, MetaX cu-bridge, CoreX, DTK/HIP, MUSA, CNToolkit/CNRT/BANG C, CANN/AscendCL.

---

### Task 1: Capture Existing CUDA Behavior

**Files:**
- Create: `tests/test_hardware_adapter.py`
- Modify: `infinimetrics/hardware/hardware_adapter.py`

**Steps:**
1. Add tests for the existing constructor argument `cuda_perf_path`, default binary path, `_build_command`, memory parsing, STREAM parsing, cache parsing, and metric names.
2. Run `pytest tests/test_hardware_adapter.py -v` against the unmerged baseline and record passing behavior.
3. Do not change implementation until the compatibility tests pass.

### Task 2: Merge Source Histories

**Files:**
- Merge: `origin/hardware_adapt`
- Merge: `origin/cambricon_test`
- Resolve: `infinimetrics/hardware/hardware_adapter.py`
- Modify: `infinimetrics/dispatcher.py`

**Steps:**
1. Merge `origin/hardware_adapt` with a merge commit.
2. Merge `origin/cambricon_test` without committing.
3. Preserve all non-conflicting native benchmark source directories.
4. Resolve `hardware_adapter.py` using the baseline compatibility tests.
5. Register `cudaunified`, `cuda`, `metax`, `corex`, `iluvatar`, `hygon`, `moore`, `cambricon`, and `ascend` hardware frameworks.

### Task 3: Add the Platform Registry

**Files:**
- Modify: `infinimetrics/hardware/hardware_adapter.py`
- Modify: `infinimetrics/common/constants.py`
- Test: `tests/test_hardware_adapter.py`

**Steps:**
1. Define immutable platform specifications for binary path, build platform, native benchmark directory, supported tests, aliases, and detection probes.
2. Resolve platform in this order: explicit `config.device`, testcase framework, runtime detection, CUDA fallback.
3. Preserve `_build_cuda_project`, `_build_command`, `_execute_test`, and `_parse_output` compatibility wrappers.
4. Preserve existing CUDA metric names: `hardware.mem_sweep_*`, `hardware.stream_*`, and `hardware.gpu_cache_*`.
5. Put platform identity in result configuration instead of metric names.
6. Add tests for every alias, resolution priority, build command, and unsupported capability.

### Task 4: Repair Native Platform Benchmarks

**Files:**
- Modify: `infinimetrics/hardware/cambricon-memory-benchmark/CMakeLists.txt`
- Modify: `infinimetrics/hardware/cambricon-memory-benchmark/src/main.mlu`
- Modify: `infinimetrics/hardware/cambricon-memory-benchmark/include/*.h`
- Modify: `infinimetrics/hardware/ascend-memory-benchmark/src/main.cc`
- Modify: `infinimetrics/hardware/ascend-memory-benchmark/include/*.h`
- Test: `tests/test_hardware_adapter.py`

**Steps:**
1. Pass the selected device ID into every native test instead of resetting to device 0.
2. Wire Cambricon cache flags to the existing NRAM and L2 implementations.
3. Fix the Cambricon CMake source filename.
4. Parse Cambricon cache into the existing cache metric schema.
5. Report Ascend D2D hierarchy sweep separately from CUDA L1/L2 cache metrics.
6. Mark estimated Ascend STREAM operations in result metadata and preserve their existing numeric output.
7. Keep default allocation sizes and CLI defaults unchanged.

### Task 5: Automated Verification

**Files:**
- Modify: `tests/test_hardware_adapter.py`
- Create: `test_inputs/configs/hardware.cambricon.Comprehensive.json`
- Create: `test_inputs/configs/hardware.ascend.Comprehensive.json`
- Modify: platform benchmark documentation

**Steps:**
1. Run `pytest tests/test_hardware_adapter.py -v`.
2. Run the complete repository test suite.
3. Run `git diff --check`.
4. Parse every modified Python file with `compileall`.
5. Validate build scripts with `bash -n`.

### Task 6: Hardware Verification

**Steps:**
1. On A100, select one verified idle GPU and build/run CUDA memory, STREAM, cache, and comprehensive modes.
2. On MetaX, Iluvatar, Moore Threads, Cambricon, and Ascend hosts, build in the prepared platform container and run `--help` plus a short STREAM smoke test.
3. Test Hygon when an SSH host or DTK environment is available; otherwise report it explicitly as unverified.
4. Save commit, command, device selection, stdout, stderr, and exit code for every platform.
5. Re-run automated tests after any platform-specific fix.

### Task 7: Integrate Master

**Steps:**
1. Review the final diff against `origin/master`.
2. Confirm the original local dirty worktree is untouched.
3. Fetch the latest `origin/master` and merge it if necessary.
4. Push the verified integration commit to `origin/master` with a normal fast-forward push.
5. Report the final commit and per-platform verification matrix.
