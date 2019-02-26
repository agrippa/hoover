#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "hvr_irregular_matrix.h"

#define SDIM 10000000
#define REPEATS 10000

int main(int argc, char **argv) {
    hvr_irr_matrix_t es;
    hvr_irr_matrix_init(SDIM, 1024ULL * 1024ULL * 1024ULL, &es);

    hvr_edge_type_t all_edge_types[4] = {BIDIRECTIONAL, DIRECTED_IN,
        DIRECTED_OUT, NO_EDGE};

    for (int i = 0; i < 4; i++) {
        hvr_irr_matrix_set(0, 0, all_edge_types[i], &es);
        assert(hvr_irr_matrix_get(0, 0, &es) == all_edge_types[i]);
    }

    // Try setting the one right next to it and make sure it doesn't change
    for (int i = 0; i < 4; i++) {
        hvr_irr_matrix_set(0, 1, all_edge_types[i], &es);
        assert(hvr_irr_matrix_get(0, 1, &es) == all_edge_types[i]);
        assert(hvr_irr_matrix_get(0, 0, &es) == NO_EDGE);
    }

    for (int i = 0; i < 4; i++) {
        hvr_irr_matrix_set(SDIM - 1, SDIM - 1, all_edge_types[i], &es);
        assert(hvr_irr_matrix_get(SDIM - 1, SDIM - 1, &es) == all_edge_types[i]);
    }

    for (unsigned r = 0; r < REPEATS; r++) {
        size_t i = rand();
        i = i % SDIM;
        size_t j = rand();
        j = j % SDIM;

        int edge_type_index = rand();
        edge_type_index = edge_type_index % 4;

        hvr_irr_matrix_set(i, j, all_edge_types[edge_type_index], &es);
        assert(hvr_irr_matrix_get(i, j, &es) ==
                all_edge_types[edge_type_index]);
    }

    hvr_irr_matrix_t es2;
    hvr_irr_matrix_init(SDIM, 1024ULL * 1024ULL * 1024ULL, &es2);

    hvr_irr_matrix_set(SDIM - 2, 1, BIDIRECTIONAL, &es2);
    hvr_irr_matrix_set(SDIM - 2, 2, DIRECTED_IN, &es2);
    hvr_irr_matrix_set(SDIM - 2, 3, DIRECTED_OUT, &es2);
    hvr_irr_matrix_set(SDIM - 2, 4, NO_EDGE, &es2);
    hvr_irr_matrix_set(SDIM - 2, 5, BIDIRECTIONAL, &es2);

    uint64_t vals[10];
    hvr_edge_type_t edges[10];
    size_t nvals;
    hvr_irr_matrix_linearize(SDIM - 2, vals, edges, &nvals, 10, &es2);

    assert(nvals == 4);
    assert(vals[0] == 1 && edges[0] == BIDIRECTIONAL);
    assert(vals[1] == 2 && edges[1] == DIRECTED_IN);
    assert(vals[2] == 3 && edges[2] == DIRECTED_OUT);
    assert(vals[3] == 5 && edges[3] == BIDIRECTIONAL);

    printf("Success!\n");

    return 0;
}
