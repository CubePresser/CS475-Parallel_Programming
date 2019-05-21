#include "simd.p4.h"

#define NUMTRIES    25

#ifndef ARRAYSIZE
#define ARRAYSIZE   1000
#endif 

int main() {
    FILE* outfile = fopen("results.txt", "a");
    if(!outfile) {
        fprintf(stderr, "Error opening results.txt\n");
        return -1;
    }

    float* a = new float[ARRAYSIZE]();
    float* b = new float[ARRAYSIZE]();
    float* c = new float[ARRAYSIZE]();

    double maxSpeedup = 0.0;
    double maxSpeedupRed = 0.0;

    for(int i = 0; i < NUMTRIES; i++) {
        double start, end, elapsed1, elapsed2;

        // Array multiplication
        start = omp_get_wtime();
        NonSimdMul(a, b, c, ARRAYSIZE);
        end = omp_get_wtime();

        elapsed1 = end - start;

        start = omp_get_wtime();
        SimdMul(a, b, c, ARRAYSIZE);
        end = omp_get_wtime();

        elapsed2 = end - start;

        double speedup = elapsed1 / elapsed2;

        // Array multiplication with reduction
        start = omp_get_wtime();
        NonSimdMulSum(a, b, ARRAYSIZE);
        end = omp_get_wtime();

        elapsed1 = end - start;

        start = omp_get_wtime();
        SimdMulSum(a, b, ARRAYSIZE);
        end = omp_get_wtime();

        elapsed2 = end - start;
        
        double speedupRed = elapsed1 / elapsed2;

        // Set max speedups
        if(speedup > maxSpeedup) maxSpeedup = speedup;
        if(speedupRed > maxSpeedupRed) maxSpeedupRed = speedupRed;
    }

    printf(
        "========================\nArraysize: %d\nArr-mult-speedup: %f\nArr-mult-red-speedup: %f\n========================\n\n",
        ARRAYSIZE, maxSpeedup, maxSpeedupRed
    );

    fprintf(outfile, "%f\t%f\n", maxSpeedup, maxSpeedupRed);

    return 0;
}
