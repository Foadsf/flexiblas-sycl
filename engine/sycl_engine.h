/* SPDX-License-Identifier: Apache-2.0
 *
 * C interface of the SYCL offload engine. Plain C types only, so the engine can be
 * built with a SYCL compiler (e.g. clang-cl -fsycl) and called from code built with
 * any other C compiler (e.g. the MinGW-built FlexiBLAS hook).
 *
 * Every routine returns 0 when it has computed the result, and a non-zero value when it
 * declined the call (problem too small, unsupported precision on the device, or a
 * runtime error). On a non-zero return the output arguments are untouched and the
 * caller must compute the result itself.
 */
#ifndef SYCL_ENGINE_H
#define SYCL_ENGINE_H

#include <stdint.h>

#ifdef _WIN32
#ifdef SYCL_ENGINE_BUILD
#define SYCL_ENGINE_API __declspec(dllexport)
#else
#define SYCL_ENGINE_API __declspec(dllimport)
#endif
#else
#define SYCL_ENGINE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum { SYCL_ENGINE_DONE = 0, SYCL_ENGINE_DECLINED = 1, SYCL_ENGINE_ERROR = 2 };

/* Create the queue on the default SYCL device. Returns 0 if a device is usable. */
SYCL_ENGINE_API int sycl_engine_init(void);
SYCL_ENGINE_API void sycl_engine_finalize(void);
SYCL_ENGINE_API const char *sycl_engine_device_name(void);

/* Column-major GEMM, C = alpha * op(A) * op(B) + beta * C, with Fortran BLAS semantics.
 * trans: 'N', 'T' or 'C' (case-insensitive). */
SYCL_ENGINE_API int sycl_engine_sgemm(char transa, char transb, int64_t m, int64_t n, int64_t k,
                                      float alpha, const float *a, int64_t lda, const float *b,
                                      int64_t ldb, float beta, float *c, int64_t ldc);
SYCL_ENGINE_API int sycl_engine_dgemm(char transa, char transb, int64_t m, int64_t n, int64_t k,
                                      double alpha, const double *a, int64_t lda, const double *b,
                                      int64_t ldb, double beta, double *c, int64_t ldc);

#ifdef __cplusplus
}
#endif

#endif
