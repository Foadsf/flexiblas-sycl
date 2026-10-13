#!/usr/bin/env bash
# Build gemm_bench and run it three ways through FlexiBLAS with the OpenBLAS backend:
#   1. OpenBLAS only (no hook)
#   2. hook, every size offloaded to the SYCL device (threshold 0)
#   3. hook with the default threshold
# Run in an MSYS2 UCRT64 shell. REPS (default 5) and SIZES override the defaults.
set -uo pipefail
HOOK=$(cd "$(dirname "$0")/.." && pwd)
OUT=$HOOK/build
FB=${FLEXIBLAS_PREFIX:-$HOOK/../toolchains/flexiblas}
DPCPP=${DPCPP_ROOT:-$HOOK/../toolchains/dpcpp}
ONEMATH_BUILD=${ONEMATH_BUILD:-$HOOK/../spike/build-win}
REPS=${REPS:-5}
mkdir -p "$OUT"

gcc -O2 -o "$OUT/gemm_bench.exe" "$HOOK/bench/gemm_bench.c" -L"$FB/bin" -lflexiblas || exit 1

export PATH="$FB/bin:$FB/bin/flexiblas:/ucrt64/bin:$OUT:$DPCPP/bin:$ONEMATH_BUILD/bin:$PATH"
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}
# OpenBLAS itself as the backend, by full path. FlexiBLAS's own wrapper backend
# (bin/flexiblas/libflexiblas_openblasopenmp.dll) does not work on MinGW: GetProcAddress does
# not search a DLL's dependencies, so no BLAS symbol is found and everything falls back to NETLIB.
export FLEXIBLAS=${FLEXIBLAS:-$(cygpath -w /ucrt64/bin/libopenblas.dll)}
cd "$OUT"

echo "=== backend check ==="
FLEXIBLAS_VERBOSE=1 ./gemm_bench.exe s 1 64 2>&1 | grep -iE "Use .*BLAS|openblas" | head -3

echo; echo "=== 1. OpenBLAS (no hook) ==="
FLEXIBLAS_HOOK= ./gemm_bench.exe s "$REPS" ${SIZES:-}

echo; echo "=== 2. hook, everything offloaded ==="
FLEXIBLAS_HOOK=flexiblas_hook_sycl.dll SYCL_ENGINE_GEMM_MIN_FLOPS=0 ./gemm_bench.exe s "$REPS" ${SIZES:-}

echo; echo "=== 3. hook, default threshold ==="
FLEXIBLAS_HOOK=flexiblas_hook_sycl.dll ./gemm_bench.exe s "$REPS" ${SIZES:-}
