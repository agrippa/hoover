#include "hvr_map.h"
#include "hoover.h"

#define N_VERTICES 1000000
#define N_REPEATS 20

int main(int argc, char **argv) {
    hvr_map_t m;
    hvr_map_init(&m,
            100000,/* prealloc segs */
            1073741824, /* prealloc vals */
            65536, /* prealloc nodes */
            8,
            EDGE_INFO);

    const unsigned long long start = hvr_current_time_us();
    for (int r = 0; r < N_REPEATS; r++) {
        for (unsigned i = 0; i < N_VERTICES - N_REPEATS; i++) {
            unsigned neighbor = i + r;

            hvr_map_val_t val;
            val.edge_info = construct_edge_info(neighbor, BIDIRECTIONAL);
            hvr_map_add(i, val, &m);
        }
    }
    const unsigned long long elapsed = hvr_current_time_us() - start;
    printf("# vertices = %u, # repeats = %u, took %f ms\n", N_VERTICES,
            N_REPEATS, (double)elapsed / 1000.0);

    return 0;
}
