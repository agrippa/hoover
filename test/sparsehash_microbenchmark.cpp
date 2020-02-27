#include "hoover.h"
#include <sparsehash/sparse_hash_map>

#define N_VERTICES 1000000
#define N_REPEATS 50

using google::sparse_hash_map;

typedef sparse_hash_map<hvr_vertex_id_t, void*> map_t;

int main(void) {
    map_t m;

    unsigned long long start = hvr_current_time_us();
    for (int r = 0; r < N_REPEATS; r++) {
        for (unsigned i = 0; i < N_VERTICES - N_REPEATS; i++) {
            unsigned neighbor = i + r;

            m.insert(std::pair<hvr_vertex_id_t, void*>(i, (void*)neighbor));
        }
    }
    unsigned long long elapsed = hvr_current_time_us() - start;
    printf("# vertices = %u, # repeats = %u, took %f ms\n", N_VERTICES,
            N_REPEATS, (double)elapsed / 1000.0);

    start = hvr_current_time_us();
    for (int r = 0; r < N_REPEATS; r++) {
        for (unsigned i = 0; i < N_VERTICES - N_REPEATS; i++) {
            m.find(i);
        }
    }
    elapsed = hvr_current_time_us() - start;
    printf("# vertices = %u, # repeats = %u, took %f ms\n", N_VERTICES,
            N_REPEATS, (double)elapsed / 1000.0);


    return 0;
}
