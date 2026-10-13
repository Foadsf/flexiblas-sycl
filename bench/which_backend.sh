#!/usr/bin/env bash
# Show which BLAS backend FlexiBLAS actually loads for gemm_bench (verbose log, filtered).
HOOK=$(cd "$(dirname "$0")/.." && pwd)
FB=${FLEXIBLAS_PREFIX:-$HOOK/../toolchains/flexiblas}
export PATH="$FB/bin:$FB/bin/flexiblas:/ucrt64/bin:$HOOK/build:$PATH"
# Set FLEXIBLAS to compare, e.g. the wrapper backend bin/flexiblas/libflexiblas_openblasopenmp.dll.
export FLEXIBLAS=${FLEXIBLAS:-$(cygpath -w /ucrt64/bin/libopenblas.dll)}
cd "$HOOK/build"
FLEXIBLAS_VERBOSE=3 ./gemm_bench.exe s 1 64 2>&1 | sed -n '/Trying to use/,/Load NETLIB/p' | grep -viE 'internal fallback|Load [a-z0-9_]+ from|Look up' | head -25
