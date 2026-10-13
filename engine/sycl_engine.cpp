// SPDX-License-Identifier: Apache-2.0
//
// SYCL offload engine: runs column-major GEMM on a SYCL device through oneMath's
// generic SYCL BLAS backend. Inputs are ordinary host pointers (Fortran BLAS
// semantics); the engine copies them to device memory, computes, copies C back and
// waits before returning, so the caller sees a normal synchronous BLAS call.
//
// Environment:
//   SYCL_ENGINE_GEMM_MIN_FLOPS  offload only if 2*m*n*k >= this (default 2*512^3)
//   SYCL_ENGINE_VERBOSE         1 = one line per call on stderr

#define SYCL_ENGINE_BUILD
#include "sycl_engine.h"

#include <sycl/sycl.hpp>
#include "oneapi/math/blas.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

using generic_selector = oneapi::math::backend_selector<oneapi::math::backend::generic>;

struct Engine {
    sycl::queue queue;
    std::string device_name;
    bool fp64 = false;
    double min_flops = 2.0 * 512.0 * 512.0 * 512.0;
    int verbose = 0;
};

std::unique_ptr<Engine> g_engine;
std::mutex g_mutex;  // one queue, one call at a time

double env_double(const char *name, double fallback) {
    const char *v = std::getenv(name);
    return (v && *v) ? std::atof(v) : fallback;
}

bool to_transpose(char t, oneapi::math::transpose &out) {
    switch (std::toupper(static_cast<unsigned char>(t))) {
        case 'N': out = oneapi::math::transpose::nontrans; return true;
        case 'T': out = oneapi::math::transpose::trans; return true;
        case 'C': out = oneapi::math::transpose::conjtrans; return true;
        default: return false;
    }
}

// Number of columns of the stored array holding op(X) with the given leading dimension.
int64_t stored_cols(oneapi::math::transpose t, int64_t rows_of_op, int64_t cols_of_op) {
    return t == oneapi::math::transpose::nontrans ? cols_of_op : rows_of_op;
}

template <typename T>
int gemm(const char *name, char transa, char transb, int64_t m, int64_t n, int64_t k, T alpha,
         const T *a, int64_t lda, const T *b, int64_t ldb, T beta, T *c, int64_t ldc) {
    std::lock_guard<std::mutex> lock(g_mutex);
    Engine *e = g_engine.get();
    if (!e) return SYCL_ENGINE_DECLINED;

    oneapi::math::transpose ta, tb;
    if (!to_transpose(transa, ta) || !to_transpose(transb, tb)) return SYCL_ENGINE_DECLINED;
    if (m <= 0 || n <= 0 || k <= 0) return SYCL_ENGINE_DECLINED;  // leave edge cases to the host BLAS
    if (std::is_same<T, double>::value && !e->fp64) return SYCL_ENGINE_DECLINED;
    const double flops = 2.0 * double(m) * double(n) * double(k);
    if (flops < e->min_flops) return SYCL_ENGINE_DECLINED;

    // op(A) is m x k, op(B) is k x n, C is m x n; arrays are stored column-major.
    const size_t size_a = size_t(lda) * size_t(stored_cols(ta, m, k));
    const size_t size_b = size_t(ldb) * size_t(stored_cols(tb, k, n));
    const size_t size_c = size_t(ldc) * size_t(n);

    T *da = nullptr, *db = nullptr, *dc = nullptr;
    try {
        sycl::queue &q = e->queue;
        da = sycl::malloc_device<T>(size_a, q);
        db = sycl::malloc_device<T>(size_b, q);
        dc = sycl::malloc_device<T>(size_c, q);
        if (!da || !db || !dc) throw std::runtime_error("device allocation failed");

        q.memcpy(da, a, size_a * sizeof(T));
        q.memcpy(db, b, size_b * sizeof(T));
        // BLAS: with beta == 0, C is output only and may hold NaN/garbage on input.
        if (beta == T(0)) q.memset(dc, 0, size_c * sizeof(T));
        else q.memcpy(dc, c, size_c * sizeof(T));
        q.wait_and_throw();

        oneapi::math::blas::column_major::gemm(generic_selector{q}, ta, tb, m, n, k, alpha, da, lda,
                                               db, ldb, beta, dc, ldc)
            .wait_and_throw();

        q.memcpy(c, dc, size_c * sizeof(T)).wait_and_throw();
        sycl::free(da, q);
        sycl::free(db, q);
        sycl::free(dc, q);
    }
    catch (const std::exception &ex) {
        if (e->verbose) std::fprintf(stderr, "[sycl-engine] %s failed, declined: %s\n", name, ex.what());
        sycl::free(da, e->queue);
        sycl::free(db, e->queue);
        sycl::free(dc, e->queue);
        return SYCL_ENGINE_ERROR;
    }

    if (e->verbose) {
        std::fprintf(stderr, "[sycl-engine] %s %c%c m=%lld n=%lld k=%lld on %s\n", name, transa,
                     transb, (long long)m, (long long)n, (long long)k, e->device_name.c_str());
    }
    return SYCL_ENGINE_DONE;
}

}  // namespace

extern "C" {

int sycl_engine_init(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_engine) return 0;
    try {
        auto e = std::make_unique<Engine>();
        e->queue = sycl::queue(sycl::default_selector_v, sycl::property::queue::in_order{});
        sycl::device dev = e->queue.get_device();
        e->device_name = dev.get_info<sycl::info::device::name>();
        e->fp64 = dev.has(sycl::aspect::fp64);
        e->min_flops = env_double("SYCL_ENGINE_GEMM_MIN_FLOPS", e->min_flops);
        e->verbose = int(env_double("SYCL_ENGINE_VERBOSE", 0));
        if (e->verbose) {
            std::fprintf(stderr, "[sycl-engine] device: %s (fp64: %s), min flops: %g\n",
                         e->device_name.c_str(), e->fp64 ? "yes" : "no", e->min_flops);
        }
        g_engine = std::move(e);
        return 0;
    }
    catch (const std::exception &ex) {
        std::fprintf(stderr, "[sycl-engine] no usable SYCL device: %s\n", ex.what());
        return 1;
    }
}

void sycl_engine_finalize(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_engine.reset();
}

const char *sycl_engine_device_name(void) {
    return g_engine ? g_engine->device_name.c_str() : "";
}

int sycl_engine_sgemm(char transa, char transb, int64_t m, int64_t n, int64_t k, float alpha,
                      const float *a, int64_t lda, const float *b, int64_t ldb, float beta, float *c,
                      int64_t ldc) {
    return gemm<float>("sgemm", transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

int sycl_engine_dgemm(char transa, char transb, int64_t m, int64_t n, int64_t k, double alpha,
                      const double *a, int64_t lda, const double *b, int64_t ldb, double beta,
                      double *c, int64_t ldc) {
    return gemm<double>("dgemm", transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

}  // extern "C"
