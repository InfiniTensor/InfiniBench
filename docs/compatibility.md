# CUDA Compatibility Tests

The compatibility adapter compiles and runs CUDA Samples on NVIDIA and
CUDA-compatible accelerator toolchains. It reports compilation, execution,
failure, and waived-sample counts without treating a partial pass rate as an
adapter execution error.

## Test Input

```json
{
  "run_id": "cuda_samples.nvidia.quick",
  "testcase": "compatibility.CudaSamples.PassRate",
  "config": {
    "platform": "nvidia",
    "sample_filter": ["vectorAdd", "matrixMul", "clock"],
    "timeout_per_sample": 180,
    "jobs": 4
  }
}
```

Initialize the bundled CUDA Samples revision before running compatibility tests:

```bash
git submodule update --init submodules/cuda-samples
```

The adapter uses `submodules/cuda-samples` by default. An optional
`cuda_samples_dir` override must contain a `Samples` directory. The adapter
recursively discovers `Samples/<category>/<sample>` directories containing a
Makefile or a standalone CMake project. CMake grouping manifests that only
aggregate child directories are excluded. A requested sample name that is not
found, or an empty `sample_filter`, is a configuration error instead of a
successful zero-sample result.

The bundled submodule pins the CMake-based `master` revision at `7b601789`.
The upstream `batch_test` branch provides the Makefiles used by the original
compatibility workflow. The adapter supports both layouts; use `build_system`
to select `cmake`, `make`, or the default `auto` detection.

CMake samples are configured through a temporary wrapper project. The wrapper
sets `CUDA_ARCHITECTURES` on every generated target after the sample manifest
has been evaluated, so manifests that set their own default architecture list
cannot override the requested `sms` value.

## Platform Toolchains

Platform aliases and compiler candidates are shared with the hardware adapter
through `infinibench.hardware.constants`. Compatibility-only architecture
values and Make arguments live in `infinibench.common.constants`.

| Platform | Default compiler | Default architecture |
| --- | --- | --- |
| NVIDIA | `nvcc` | `80` |
| MetaX | `cucc` (falls back to `mxcc`) | `70` |
| Iluvatar CoreX (BI-V150/TG150) | `/usr/local/corex/bin/clang++` | `ivcore11` |

The supported canonical platform names are `cuda`, `metax`, and `corex`.
The existing aliases `nvidia` and `iluvatar` are also accepted.

The CoreX default targets BI-V150/TG150. Override `sms` and `make_args`
together when testing a different Iluvatar architecture.

Set `compiler`, `sms`, or `make_args` in the input when the installed vendor
SDK uses a wrapper or different target. Arguments are passed directly as an
argument list; shell expansion is not performed. The resolved default compiler
and architecture are added to the result config when they were not explicit in
the input. For MetaX, the adapter also infers `MACA_PATH` from the resolved
`cucc` or `mxcc` location when the variable is unset. An explicit `MACA_PATH`
is preserved.

For non-NVIDIA Makefile builds, the default arguments remove NVIDIA-only
`--threads`, `-gencode`, and `-m64` flags. Platform support is declared only
after its compile and runtime workflow has been validated on target hardware.

## Metrics

- `compile_passed` and `compile_failed` cover all discovered samples.
- `run_passed` and `run_failed` cover samples that produced an executable.
- `run_skipped` counts CUDA Samples that explicitly return a waived result.
- A sample that does not run because compilation failed has `run_result:
  "not_run"` and is not included in `run_skipped`.
- `run_pass_rate` keeps the original end-to-end definition: run passes divided
  by all selected samples.
- `details` records each sample path and the final compiler or runtime error.

`result_code: 0` means the compatibility test completed and produced valid
measurements. It does not mean every sample passed; use the pass-rate metrics
for that decision.
