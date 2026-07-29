# Hardware Benchmarks

InfiniBench provides one hardware adapter for NVIDIA CUDA and six additional
accelerator platforms. Existing CUDA command shapes and metric names are kept
unchanged.

## Platforms

| Platform | Testcase framework | Build command | Binary |
| --- | --- | --- | --- |
| NVIDIA CUDA | `cuda`, `cudaUnified` | `bash build.sh --platform cuda` | `cuda-memory-benchmark/build/cuda_perf_suite` |
| MetaX | `metax` | `bash build.sh --platform metax` | `cuda-memory-benchmark/build/cuda_perf_suite` |
| Iluvatar CoreX | `corex`, `iluvatar` | `bash build.sh --platform corex` | `cuda-memory-benchmark/build/cuda_perf_suite` |
| Hygon DCU | `hygon` | `bash build.sh --platform hygon` | `cuda-memory-benchmark/build/cuda_perf_suite` |
| Moore Threads | `moore` | `bash build.sh --platform moore` | `cuda-memory-benchmark/build/cuda_perf_suite` |
| Cambricon | `cambricon` | `bash build.sh` | `cambricon-memory-benchmark/build/mlu_perf_suite` |
| Ascend | `ascend` | `bash build.sh` | `ascend-memory-benchmark/build/npu_perf_suite` |

Run each build command from its benchmark directory. All binaries use the same
test selectors and common arguments:

```bash
./build/<binary> --all
./build/<binary> --memory
./build/<binary> --stream --iterations 3 --array-size 1048576
./build/<binary> --cache --device 0
```

## Platform Selection

`HardwareTestAdapter` resolves a platform in this order:

1. `config.device`, when supplied.
2. The framework in a testcase such as `hardware.cambricon.Stream`.
3. The locally installed accelerator toolchain.
4. CUDA as the compatibility fallback.

The aliases `nvidia`, `musa`, `mthreads`, `mlu`, and `npu` are also accepted as
explicit device values. A selected non-CUDA platform is recorded in the result
configuration as `platform`; metric names remain compatible with CUDA results.

## Device Visibility

Restrict the process to an idle physical device before starting a benchmark.
The selected physical device is renumbered to device 0 inside the process.

| Platform | Visibility variable |
| --- | --- |
| NVIDIA, MetaX, Iluvatar | `CUDA_VISIBLE_DEVICES` |
| Hygon | `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` |
| Moore Threads | `MUSA_VISIBLE_DEVICES` |
| Cambricon | `MLU_VISIBLE_DEVICES` |
| Ascend | `ASCEND_RT_VISIBLE_DEVICES` |

For example:

```bash
CUDA_VISIBLE_DEVICES=2 ./build/cuda_perf_suite --stream --device 0
MLU_VISIBLE_DEVICES=3 ./build/mlu_perf_suite --stream --device 0
ASCEND_RT_VISIBLE_DEVICES=4 ./build/npu_perf_suite --stream --device 0
```

## Container Notes

Hygon DTK containers may require the host driver libraries to be mounted at
the path expected by DTK:

```bash
docker run --rm --privileged \
  --mount type=bind,source=/opt/hyhal,target=/opt/hyhal,readonly \
  <dtk-image> bash
```

Without this mount, management tools can list DCUs while HIP applications fail
to load `libhsa-runtime64.so` or `libhydmi.so`.

Some Ascend development images can return exit code 137 after the benchmark has
printed its completion message. The adapter intentionally treats every nonzero
exit code as a failure; fix the container lifecycle rather than suppressing that
error in application code.
