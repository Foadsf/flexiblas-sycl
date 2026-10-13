# SYCL offload hook for FlexiBLAS

A [FlexiBLAS](https://github.com/mpimd-csc/flexiblas) hook that runs large `sgemm`/`dgemm` calls on a SYCL device, using the generic SYCL BLAS kernels from [oneMath](https://github.com/uxlfoundation/oneMath). Everything else goes to whichever BLAS backend FlexiBLAS has selected (OpenBLAS, BLIS, MKL, NETLIB, ...). Applications need no changes: if they link against FlexiBLAS, setting `FLEXIBLAS_HOOK` is enough.

Status: early prototype. Only `?gemm` is offloaded so far.

## How it works

Two libraries, so that each can be built with the right compiler:

- `sycl_engine` (C++/SYCL, built with a SYCL compiler such as open DPC++): copies the host matrices to device memory, calls oneMath's generic `gemm`, copies `C` back and waits. It declines a call (returns non-zero, outputs untouched) when the problem is below the threshold, the device lacks FP64 for `dgemm`, or anything fails.
- `flexiblas_hook_sycl` (plain C, built with the same compiler as FlexiBLAS against its installed headers `flexiblas_backend.h` and `flexiblas_real_calls.h`): exports `hook_sgemm`/`hook_dgemm`, loads the engine on the first call rather than during FlexiBLAS initialisation, and passes declined calls down the FlexiBLAS hook chain. `SYCL_HOOK_EAGER_INIT=1` starts the engine during initialisation instead; on the tested setup both work.

## Settings

| Variable | Meaning | Default |
|---|---|---|
| `FLEXIBLAS_HOOK` | hook to load, e.g. `flexiblas_hook_sycl.dll` | unset |
| `SYCL_ENGINE_LIBRARY` | engine library to load | `sycl_engine.dll` / `libsycl_engine.so` |
| `SYCL_ENGINE_GEMM_MIN_FLOPS` | offload only if `2*m*n*k` is at least this | `2*512^3` |
| `SYCL_ENGINE_VERBOSE` | `1` prints the device and one line per offloaded call | `0` |
| `ONEAPI_DEVICE_SELECTOR` | which SYCL device to use (DPC++) | default device |

## Building (Windows, as tested)

- Engine: `build_engine.cmd` (open DPC++ `clang-cl -fsycl` + VS 2026, against a oneMath build with `-DENABLE_GENERIC_BLAS_BACKEND=ON`; on Windows this needs uxlfoundation/oneMath#787).
- Hook and test: `build_hook_and_test.sh` in an MSYS2 UCRT64 shell, against FlexiBLAS 3.5.0 built with MinGW.

## Tested

Windows 11, Intel UHD Graphics (Tiger Lake, no FP64) through Level Zero, FlexiBLAS 3.5.0 (MinGW, NETLIB backend). `test/gemm_check.c` calls `sgemm_`/`dgemm_` through the Fortran ABI and compares with a reference GEMM: all transpose combinations, padded leading dimensions, `beta = 0` with NaN in `C`. All 17 cases pass with and without the hook; with the hook, the large `sgemm` cases run on the GPU, `dgemm` falls back to the host (no FP64), and small calls stay on the host. The same holds with `SYCL_HOOK_EAGER_INIT=1`.

Loading a hook by file name needs a FlexiBLAS fix for a use-after-unload in `__flexiblas_hook_add_from_file` (mpimd-csc/flexiblas#108, merged after 3.5.0). Also note that `FLEXIBLAS_HOOK` is split at `:`, so on Windows give a file name without a drive letter (found through `PATH` or the current directory).

## Performance

`bench/run_bench.sh` times square `sgemm` (`alpha=1, beta=0`) through FlexiBLAS, median of 5 calls after a first call. Measured 2026-10-13 on an i7-11850H laptop on mains power, Intel UHD Graphics (Tiger Lake, Xe-LP, 32 EU), OpenBLAS 0.3.31 (MSYS2, OpenMP) as the FlexiBLAS backend:

| n | OpenBLAS, CPU | hook, iGPU, untuned | hook, iGPU, `INTEL_GPU` tuning |
|---|---|---|---|
| 512 | 288 GFLOP/s | 33 | 9 |
| 1024 | 322 | 41 | 10 |
| 2048 | 317 | 40 | 10 |
| 4096 | 404 | 39 | 11 |

On this integrated GPU, offloading is about 10x slower than OpenBLAS at every size, so it does not pay off; the `INTEL_GPU` tuning of the generic backend is slower still here (it targets larger Intel GPUs). The first offloaded call also costs about 0.3 s for kernel JIT compilation. Copies are not the bottleneck (the GPU rate is flat with size); the generic kernels are. Discrete GPUs have not been measured yet. Until then, keep `SYCL_ENGINE_GEMM_MIN_FLOPS` high on integrated GPUs, or do not load the hook there.

Raw output: `bench/results-uhd-sgemm-untuned.txt`, `bench/results-uhd-sgemm-tuned.txt`.

Note: on Windows/MinGW, use OpenBLAS itself as the FlexiBLAS backend (`FLEXIBLAS=<path>\libopenblas.dll`). FlexiBLAS's wrapper backend `libflexiblas_openblasopenmp.dll` exports no BLAS symbols of its own and `GetProcAddress` does not search its dependencies, so every routine silently falls back to the reference NETLIB code (about 4 GFLOP/s at n=1024).

## License

Apache-2.0
