#!/usr/bin/env bash
# Diagnose hook loading: run gemm_check with FlexiBLAS verbose output (V=level, default 2).
HOOK=$(cd "$(dirname "$0")" && pwd)
OUT=$HOOK/build
FB=${FLEXIBLAS_PREFIX:-$HOOK/../toolchains/flexiblas}
DPCPP=${DPCPP_ROOT:-$HOOK/../toolchains/dpcpp}
ONEMATH_BUILD=${ONEMATH_BUILD:-$HOOK/../spike/build-win}
export PATH="$FB/bin:/ucrt64/bin:$OUT:$DPCPP/bin:$ONEMATH_BUILD/bin:$PATH"
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}
cd "$OUT"
FLEXIBLAS_VERBOSE=${V:-2} FLEXIBLAS_HOOK=${HOOKNAME:-flexiblas_hook_sycl.dll} SYCL_ENGINE_VERBOSE=1 \
  SYCL_ENGINE_GEMM_MIN_FLOPS=${MIN_FLOPS:-1000} ./gemm_check.exe 2>&1 \
  | grep -viE "internal fallback|^<flexiblas> Load [a-z0-9_]+ from" | tail -${LINES_MAX:-40}
echo "exit=${PIPESTATUS[0]}"
