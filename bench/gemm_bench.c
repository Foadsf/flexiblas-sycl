/* SPDX-License-Identifier: Apache-2.0
 *
 * Times square sgemm_/dgemm_ calls through the Fortran BLAS ABI, i.e. through whatever
 * library the program is linked to (FlexiBLAS here, with or without the SYCL hook).
 *
 * Usage: gemm_bench [s|d] [reps] [n1 n2 ...]
 * Prints, per size: first-call time (includes any one-off setup such as kernel JIT),
 * median of the following calls, and GFLOP/s from the median.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
static double now(void) {
    static LARGE_INTEGER f;
    LARGE_INTEGER t;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
}
#else
#include <time.h>
static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
#endif

typedef int blasint;
extern void sgemm_(const char *, const char *, const blasint *, const blasint *, const blasint *,
                   const float *, const float *, const blasint *, const float *, const blasint *,
                   const float *, float *, const blasint *, size_t, size_t);
extern void dgemm_(const char *, const char *, const blasint *, const blasint *, const blasint *,
                   const double *, const double *, const blasint *, const double *,
                   const blasint *, const double *, double *, const blasint *, size_t, size_t);

static int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    int dbl = argc > 1 && argv[1][0] == 'd';
    int reps = argc > 2 ? atoi(argv[2]) : 5;
    int def_sizes[] = {128, 256, 512, 1024, 1536, 2048, 3072, 4096};
    int nsizes = argc > 3 ? argc - 3 : (int)(sizeof def_sizes / sizeof *def_sizes);
    double *t = malloc(reps * sizeof *t);

    printf("%-6s %6s %12s %12s %10s\n", dbl ? "dgemm" : "sgemm", "n", "first[ms]", "median[ms]",
           "GFLOP/s");
    for (int s = 0; s < nsizes; ++s) {
        int n = argc > 3 ? atoi(argv[3 + s]) : def_sizes[s];
        size_t nn = (size_t)n * n, es = dbl ? sizeof(double) : sizeof(float);
        void *a = malloc(nn * es), *b = malloc(nn * es), *c = malloc(nn * es);
        for (size_t i = 0; i < nn; ++i) {
            double v = (double)rand() / RAND_MAX - 0.5;
            if (dbl) { ((double *)a)[i] = v; ((double *)b)[i] = -v; ((double *)c)[i] = 0; }
            else { ((float *)a)[i] = (float)v; ((float *)b)[i] = (float)-v; ((float *)c)[i] = 0; }
        }
        const char tr = 'N';
        double first = 0;
        for (int r = -1; r < reps; ++r) {
            double t0 = now();
            if (dbl) {
                const double one = 1, zero = 0;
                dgemm_(&tr, &tr, &n, &n, &n, &one, a, &n, b, &n, &zero, c, &n, 1, 1);
            } else {
                const float one = 1, zero = 0;
                sgemm_(&tr, &tr, &n, &n, &n, &one, a, &n, b, &n, &zero, c, &n, 1, 1);
            }
            double dt = now() - t0;
            if (r < 0) first = dt; else t[r] = dt;
        }
        qsort(t, reps, sizeof *t, cmp);
        double med = t[reps / 2];
        printf("%-6s %6d %12.2f %12.2f %10.1f\n", dbl ? "dgemm" : "sgemm", n, 1e3 * first,
               1e3 * med, 2.0 * n * (double)n * n / med * 1e-9);
        fflush(stdout);
        free(a); free(b); free(c);
    }
    free(t);
    return 0;
}
