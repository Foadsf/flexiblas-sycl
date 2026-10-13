/* SPDX-License-Identifier: Apache-2.0
 *
 * FlexiBLAS hook that offloads large ?GEMM calls to a SYCL device.
 *
 * The SYCL work lives in a separate engine library (sycl_engine), loaded at runtime, so
 * this hook can be built with the same C compiler as FlexiBLAS while the engine is built
 * with a SYCL compiler. Calls the engine declines, and all calls when the engine or a
 * device is unavailable, go down the FlexiBLAS hook chain to the selected backend.
 *
 * Environment:
 *   SYCL_ENGINE_LIBRARY  engine library to load (default: sycl_engine.dll / libsycl_engine.so)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Installed FlexiBLAS headers: hook registration/options and the chain functions. */
#include "flexiblas_backend.h"
#include "flexiblas_real_calls.h"

#ifdef _WIN32
#include <windows.h>
#define ENGINE_DEFAULT "sycl_engine.dll"
#else
#include <dlfcn.h>
#define ENGINE_DEFAULT "libsycl_engine.so"
#endif

FLEXIBLAS_HOOK_OPTIONS(FLEXIBLAS_HOOK_OPTIONS_END)

FLEXIBLAS_HOOK_REGISTER("SYCL", "SYCL", sycl,
                        "Offload large Level-3 BLAS calls to a SYCL device (oneMath generic backend)",
                        "Foad S. Farimani")

/* --- Engine, resolved at init --- */
typedef int (*engine_init_t)(void);
typedef void (*engine_finalize_t)(void);
typedef int (*engine_sgemm_t)(char, char, int64_t, int64_t, int64_t, float, const float *, int64_t,
                              const float *, int64_t, float, float *, int64_t);
typedef int (*engine_dgemm_t)(char, char, int64_t, int64_t, int64_t, double, const double *,
                              int64_t, const double *, int64_t, double, double *, int64_t);

static void *engine_handle = NULL;
static engine_finalize_t engine_finalize = NULL;
static engine_sgemm_t engine_sgemm = NULL;
static engine_dgemm_t engine_dgemm = NULL;

/* FlexiBLAS runs hook initialization while libflexiblas itself is being loaded, before
 * main() and before other runtimes in the program may be initialised (DllMain on Windows).
 * To stay out of that, and to cost nothing in programs that never call GEMM, the engine is
 * loaded on the first offloadable call, like the backends' post_init. */
#ifdef _WIN32
static INIT_ONCE engine_once = INIT_ONCE_STATIC_INIT;
#else
#include <pthread.h>
static pthread_once_t engine_once = PTHREAD_ONCE_INIT;
#endif

static void *open_library(const char *path) {
#ifdef _WIN32
    return (void *)LoadLibraryA(path);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

static void *find_symbol(void *handle, const char *name) {
#ifdef _WIN32
    return (void *)GetProcAddress((HMODULE)handle, name);
#else
    return dlsym(handle, name);
#endif
}

static void load_engine(void) {
    const char *path = getenv("SYCL_ENGINE_LIBRARY");
    if (!path || !*path) path = ENGINE_DEFAULT;

    engine_handle = open_library(path);
    if (!engine_handle) {
        fprintf(stderr, "[sycl-hook] cannot load %s; all calls use the host BLAS.\n", path);
        return;
    }
    engine_init_t init = (engine_init_t)find_symbol(engine_handle, "sycl_engine_init");
    engine_finalize = (engine_finalize_t)find_symbol(engine_handle, "sycl_engine_finalize");
    engine_sgemm = (engine_sgemm_t)find_symbol(engine_handle, "sycl_engine_sgemm");
    engine_dgemm = (engine_dgemm_t)find_symbol(engine_handle, "sycl_engine_dgemm");
    if (!init || init() != 0) {
        fprintf(stderr, "[sycl-hook] SYCL engine unavailable; all calls use the host BLAS.\n");
        engine_sgemm = NULL;
        engine_dgemm = NULL;
    }
}

#ifdef _WIN32
static BOOL CALLBACK load_engine_once(PINIT_ONCE once, PVOID param, PVOID *ctx) {
    (void)once; (void)param; (void)ctx;
    load_engine();
    return TRUE;
}
static void ensure_engine(void) { InitOnceExecuteOnce(&engine_once, load_engine_once, NULL, NULL); }
#else
static void ensure_engine(void) { pthread_once(&engine_once, load_engine); }
#endif

/* SYCL_HOOK_EAGER_INIT=1 loads the engine during FlexiBLAS initialisation instead, to test
 * whether the SYCL runtime tolerates being started there. */
FLEXIBLAS_HOOK_INIT_FUNCTION(void) {
    const char *eager = getenv("SYCL_HOOK_EAGER_INIT");
    if (eager && *eager == '1') ensure_engine();
}

FLEXIBLAS_HOOK_EXIT_FUNCTION(void) {
    if (engine_finalize) engine_finalize();
    engine_finalize = NULL;
    engine_sgemm = NULL;
    engine_dgemm = NULL;
}

void hook_sgemm(char *transa, char *transb, blasint *m, blasint *n, blasint *k, float *alpha,
                float *a, blasint *lda, float *b, blasint *ldb, float *beta, float *c, blasint *ldc,
                flexiblas_fortran_charlen_t transa_len, flexiblas_fortran_charlen_t transb_len) {
    ensure_engine();
    if (engine_sgemm && engine_sgemm(*transa, *transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta,
                                     c, *ldc) == 0)
        return;
    flexiblas_chain_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, transa_len,
                          transb_len);
}

void hook_dgemm(char *transa, char *transb, blasint *m, blasint *n, blasint *k, double *alpha,
                double *a, blasint *lda, double *b, blasint *ldb, double *beta, double *c,
                blasint *ldc, flexiblas_fortran_charlen_t transa_len,
                flexiblas_fortran_charlen_t transb_len) {
    ensure_engine();
    if (engine_dgemm && engine_dgemm(*transa, *transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta,
                                     c, *ldc) == 0)
        return;
    flexiblas_chain_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, transa_len,
                          transb_len);
}
