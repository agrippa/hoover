#include <stdio.h>
#include <stdlib.h>

#include "hvr_irregular_matrix.h"
#include "hoover.h"

#define N_VERTICES 1000000
#define N_REPEATS 20

int main(int argc, char **argv) {
    hvr_irr_matrix_t s;
    hvr_irr_matrix_init(N_VERTICES, 1024ULL * 1024ULL * 1024ULL, &s);

    const unsigned long long start = hvr_current_time_us();
    for (int r = 0; r < N_REPEATS; r++) {
        for (unsigned i = 0; i < N_VERTICES - N_REPEATS; i++) {
            unsigned neighbor = i + r;

            hvr_irr_matrix_set(i, neighbor, BIDIRECTIONAL, &s);
        }
    }
    const unsigned long long elapsed = hvr_current_time_us() - start;
    printf("# vertices = %u, # repeats = %u, took %f ms\n", N_VERTICES,
            N_REPEATS, (double)elapsed / 1000.0);

    return 0;
}
