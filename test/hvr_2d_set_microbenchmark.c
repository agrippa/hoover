#include "hvr_2d_edge_set.h"

#define N_VERTICES 10000000
#define N_REPEATS 20

int main(int argc, char **argv) {
    hvr_2d_edge_set_t s;
    hvr_2d_edge_set_init(&s, N_VERTICES, 10000);

    const unsigned long long start = hvr_current_time_us();
    for (int r = 0; r < N_REPEATS; i++) {
        for (unsigned i = 0; i < N_VERTICES; i++) {
            unsigned neighbor = i + r;

            hvr_2d_set(i, neighbor, BIDIRECTIONAL, &s);
        }
    }
    const unsigned long long elapsed = hvr_current_time_us() - start;
    printf("# vertices = %u, # repeats = %u, took %f ms\n", N_VERTICES,
            N_REPEATS, (double)elapsed / 1000.0);

    return 0;
}
