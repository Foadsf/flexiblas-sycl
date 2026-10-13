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
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#define HOOK_EXPORT __declspec(dllexport)
#define ENGINE_DEFAULT "sycl_engine.dll"
#else
#include <dlfcn.h>
#define HOOK_EXPORT __attribute__((visibility("default")))
#define ENGINE_DEFAULT "libsycl_engine.so"
#endif

#ifdef FLEXIBLAS_INTEGER8
typedef int64_t blasint;
#else
typedef int blasint;
#endif
typedef size_t fcharlen; /* gfortran >= 8, clang/flang, Intel */

/* --- FlexiBLAS hook registration (layout matches flexiblas_backend.h) --- */
typedef struct {
    char *name;
    char *cfg_name;
    char *desc;
    char *authors;
} flexiblas_hook_register_t;

typedef struct {
    char *name;
    char *desc;
    int type;
    char *def;
} flexiblas_option_t;

HOOK_EXPORT flexiblas_hook_register_t flexiblas_register = {
    "SYCL", "SYCL", "Offload large Level-3 BLAS calls to a SYCL device (oneMath generic backend)",
    "Foad S. Farimani"};
HOOK_EXPORT flexiblas_option_t flexiblas_options[] = {{NULL, NULL, 0, NULL}};

/* --- FlexiBLAS chain functions, provided by libflexiblas --- */
extern void flexiblas_chain_sgemm(void *transa, void *transb, void *m, void *n, void *k,
                                  void *alpha, void *a, void *lda, void *b, void *ldb, void *beta,
                                  void *c, void *ldc, fcharlen transa_len, fcharlen transb_len);
extern void flexiblas_chain_dgemm(void *transa, void *transb, void *m, void *n, void *k,
                                  void *alpha, void *a, void *lda, void *b, void *ldb, void *beta,
                                  void *c, void *ldc, fcharlen transa_len, fcharlen transb_len);

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
HOOK_EXPORT void __flexiblas_hook_initialize(void) {
    const char *eager = getenv("SYCL_HOOK_EAGER_INIT");
    if (eager && *eager == '1') ensure_engine();
}

HOOK_EXPORT void __flexiblas_hook_finalize(void) {
    if (engine_finalize) engine_finalize();
    engine_finalize = NULL;
    engine_sgemm = NULL;
    engine_dgemm = NULL;
}

HOOK_EXPORT void hook_sgemm(char *transa, char *transb, blasint *m, blasint *n, blasint *k,
                            float *alpha, float *a, blasint *lda, float *b, blasint *ldb,
                            float *beta, float *c, blasint *ldc, fcharlen transa_len,
                            fcharlen transb_len) {
    ensure_engine();
    if (engine_sgemm && engine_sgemm(*transa, *transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta,
                                     c, *ldc) == 0)
        return;
    flexiblas_chain_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, transa_len,
                          transb_len);
}

HOOK_EXPORT void hook_dgemm(char *transa, char *transb, blasint *m, blasint *n, blasint *k,
                            double *alpha, double *a, blasint *lda, double *b, blasint *ldb,
                            double *beta, double *c, blasint *ldc, fcharlen transa_len,
                            fcharlen transb_len) {
    ensure_engine();
    if (engine_dgemm && engine_dgemm(*transa, *transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta,
                                     c, *ldc) == 0)
        return;
    flexiblas_chain_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, transa_len,
                          transb_len);
}
