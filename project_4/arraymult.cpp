#include "simd.p4.h"

int main() {
    double time1 = omp_get_wtime();
    printf("%f\n", time1);
}
