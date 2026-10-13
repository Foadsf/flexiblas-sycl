/* SPDX-License-Identifier: Apache-2.0
 *
 * Calls sgemm_/dgemm_ through the Fortran BLAS ABI (as any application linked against
 * FlexiBLAS would) and compares each result with a naive reference GEMM.
 * Exit code 0 = all cases within tolerance.
 */
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

typedef int blasint;
extern void sgemm_(const char *, const char *, const blasint *, const blasint *, const blasint *,
                   const float *, const float *, const blasint *, const float *, const blasint *,
                   const float *, float *, const blasint *, size_t, size_t);
extern void dgemm_(const char *, const char *, const blasint *, const blasint *, const blasint *,
                   const double *, const double *, const blasint *, const double *,
                   const blasint *, const double *, double *, const blasint *, size_t, size_t);

static double rnd(void) { return (double)rand() / RAND_MAX - 0.5; }

/* Reference: C = alpha*op(A)*op(B) + beta*C in double, beta == 0 ignores C. */
static void ref_gemm(char ta, char tb, int m, int n, int k, double alpha, const double *a, int lda,
                     const double *b, int ldb, double beta, double *c, int ldc) {
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            double s = 0;
            for (int p = 0; p < k; ++p) {
                double av = (ta == 'N') ? a[i + (size_t)p * lda] : a[p + (size_t)i * lda];
                double bv = (tb == 'N') ? b[p + (size_t)j * ldb] : b[j + (size_t)p * ldb];
                s += av * bv;
            }
            c[i + (size_t)j * ldc] = alpha * s + (beta == 0 ? 0 : beta * c[i + (size_t)j * ldc]);
        }
}

static int run_case(int dbl, char ta, char tb, int m, int n, int k, int pad, double alpha,
                    double beta) {
    int lda = ((ta == 'N') ? m : k) + pad, ldb = ((tb == 'N') ? k : n) + pad, ldc = m + pad;
    size_t na = (size_t)lda * ((ta == 'N') ? k : m), nb = (size_t)ldb * ((tb == 'N') ? n : k),
           nc = (size_t)ldc * n;
    double *a = malloc(na * sizeof *a), *b = malloc(nb * sizeof *b), *c = malloc(nc * sizeof *c),
           *r = malloc(nc * sizeof *r);
    for (size_t i = 0; i < na; ++i) a[i] = rnd();
    for (size_t i = 0; i < nb; ++i) b[i] = rnd();
    for (size_t i = 0; i < nc; ++i) c[i] = (beta == 0) ? NAN : rnd(); /* beta=0: C must be ignored */
    for (size_t i = 0; i < nc; ++i) r[i] = c[i];
    ref_gemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, r, ldc);

    double err = 0;
    if (dbl) {
        dgemm_(&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, 1, 1);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                double d = fabs(c[i + (size_t)j * ldc] - r[i + (size_t)j * ldc]);
                if (!(d <= err)) err = d; /* also catches NaN */
            }
    } else {
        float *fa = malloc(na * sizeof *fa), *fb = malloc(nb * sizeof *fb), *fc = malloc(nc * sizeof *fc);
        for (size_t i = 0; i < na; ++i) fa[i] = (float)a[i];
        for (size_t i = 0; i < nb; ++i) fb[i] = (float)b[i];
        for (size_t i = 0; i < nc; ++i) fc[i] = (float)c[i];
        float fal = (float)alpha, fbe = (float)beta;
        sgemm_(&ta, &tb, &m, &n, &k, &fal, fa, &lda, fb, &ldb, &fbe, fc, &ldc, 1, 1);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                double d = fabs((double)fc[i + (size_t)j * ldc] - r[i + (size_t)j * ldc]);
                if (!(d <= err)) err = d;
            }
        free(fa); free(fb); free(fc);
    }
    /* Inputs are in [-0.5, 0.5]; scale the tolerance with k and the precision. */
    double tol = (dbl ? 1e-12 : 2e-5) * k;
    int ok = err <= tol;
    printf("%s %c%c m=%4d n=%4d k=%4d pad=%d beta=%g  max|err|=%.3g  tol=%.3g  %s\n",
           dbl ? "dgemm" : "sgemm", ta, tb, m, n, k, pad, beta, err, tol, ok ? "OK" : "FAIL");
    free(a); free(b); free(c); free(r);
    return ok;
}

int main(void) {
    const char tr[] = {'N', 'T'};
    int fails = 0;
    srand(42);
    for (int dbl = 0; dbl <= 1; ++dbl)
        for (int x = 0; x < 2; ++x)
            for (int y = 0; y < 2; ++y) {
                fails += !run_case(dbl, tr[x], tr[y], 600, 520, 700, 3, 1.5, 0.5);
                fails += !run_case(dbl, tr[x], tr[y], 600, 520, 700, 0, -1.0, 0.0);
            }
    fails += !run_case(0, 'N', 'N', 17, 9, 5, 1, 2.0, 1.0); /* small: below offload threshold */
    printf("%s (%d failing case%s)\n", fails ? "FAILED" : "ALL OK", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
