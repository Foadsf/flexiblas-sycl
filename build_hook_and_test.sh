#!/usr/bin/env bash
# Build the FlexiBLAS SYCL hook and the GEMM check with MSYS2 UCRT64 gcc, then run the check
# twice: plain FlexiBLAS (control) and with the hook offloading everything but the smallest
# case. Run with: C:\msys64\usr\bin\env MSYSTEM=UCRT64 /usr/bin/bash -lc "<this>"
set -uo pipefail
HOOK=$(cd "$(dirname "$0")" && pwd)
OUT=$HOOK/build
# Install prefix of FlexiBLAS, open DPC++ root and oneMath build tree (override as needed).
FB=${FLEXIBLAS_PREFIX:-$HOOK/../toolchains/flexiblas}
DPCPP=${DPCPP_ROOT:-$HOOK/../toolchains/dpcpp}
ONEMATH_BUILD=${ONEMATH_BUILD:-$HOOK/../spike/build-win}
mkdir -p "$OUT"

gcc -O2 -shared -o "$OUT/flexiblas_hook_sycl.dll" "$HOOK/flexiblas/hook_sycl.c" \
    -L"$FB/bin" -lflexiblas || exit 1
gcc -O2 -o "$OUT/gemm_check.exe" "$HOOK/test/gemm_check.c" -L"$FB/bin" -lflexiblas || exit 1
echo "HOOK_BUILD_OK"

# Runtime search path: FlexiBLAS, MinGW runtime, the engine and its DPC++/oneMath DLLs.
export PATH="$FB/bin:/ucrt64/bin:$OUT:$DPCPP/bin:$ONEMATH_BUILD/bin:$PATH"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
# FLEXIBLAS_HOOK splits on ':' (and MSYS turns ./x into C:/...), so pass a bare file name
# that LoadLibrary finds in the current directory.
cd "$OUT"

echo; echo "=== 1. control: FlexiBLAS without hook ==="
FLEXIBLAS_HOOK= ./gemm_check.exe; echo "exit=$?"

echo; echo "=== 2. hook, offload everything above 1000 flops ==="
FLEXIBLAS_HOOK=flexiblas_hook_sycl.dll SYCL_ENGINE_VERBOSE=1 SYCL_ENGINE_GEMM_MIN_FLOPS=1000 \
    ./gemm_check.exe; echo "exit=$?"

echo; echo "=== 3. same, but SYCL engine started during FlexiBLAS initialisation ==="
FLEXIBLAS_HOOK=flexiblas_hook_sycl.dll SYCL_ENGINE_VERBOSE=1 SYCL_ENGINE_GEMM_MIN_FLOPS=1000 \
    SYCL_HOOK_EAGER_INIT=1 ./gemm_check.exe; echo "exit=$?"
