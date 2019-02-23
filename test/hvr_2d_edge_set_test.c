#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "hvr_2d_edge_set.h"

#define SDIM 10000000
#define REPEATS 10000

int main(int argc, char **argv) {
    hvr_2d_edge_set_t es;
    hvr_2d_edge_set_init(&es, SDIM, REPEATS + 2);

    hvr_edge_type_t all_edge_types[4] = {BIDIRECTIONAL, DIRECTED_IN, DIRECTED_OUT, NO_EDGE};

    for (int i = 0; i < 4; i++) {
        hvr_2d_set(0, 0, all_edge_types[i], &es);
        assert(hvr_2d_get(0, 0, &es) == all_edge_types[i]);
    }

    // Try setting the one right next to it and make sure it doesn't change
    for (int i = 0; i < 4; i++) {
        hvr_2d_set(0, 1, all_edge_types[i], &es);
        assert(hvr_2d_get(0, 1, &es) == all_edge_types[i]);
        assert(hvr_2d_get(0, 0, &es) == NO_EDGE);
    }

    for (int i = 0; i < 4; i++) {
        hvr_2d_set(SDIM - 1, SDIM - 1, all_edge_types[i], &es);
        assert(hvr_2d_get(SDIM - 1, SDIM - 1, &es) == all_edge_types[i]);
    }

    for (unsigned r = 0; r < REPEATS; r++) {
        size_t i = rand();
        i = i % SDIM;
        size_t j = rand();
        j = j % SDIM;

        int edge_type_index = rand();
        edge_type_index = edge_type_index % 4;

        hvr_2d_set(i, j, all_edge_types[edge_type_index], &es);
        assert(hvr_2d_get(i, j, &es) == all_edge_types[edge_type_index]);
    }

    printf("Success!\n");

    return 0;
}
