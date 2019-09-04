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
        hvr_irr_matrix_set(0, 0, all_edge_types[i], IMPLICIT_EDGE, &es, 0);
        assert(hvr_irr_matrix_get(0, 0, &es) == all_edge_types[i]);
    }

    // Try setting the one right next to it and make sure it doesn't change
    for (int i = 0; i < 4; i++) {
        hvr_irr_matrix_set(0, 1, all_edge_types[i], IMPLICIT_EDGE, &es, 0);
        assert(hvr_irr_matrix_get(0, 1, &es) == all_edge_types[i]);
        assert(hvr_irr_matrix_get(0, 0, &es) == NO_EDGE);
    }

    for (int i = 0; i < 4; i++) {
        hvr_irr_matrix_set(SDIM - 1, SDIM - 1, all_edge_types[i], IMPLICIT_EDGE, &es, 0);
        assert(hvr_irr_matrix_get(SDIM - 1, SDIM - 1, &es) == all_edge_types[i]);
    }

    for (unsigned r = 0; r < REPEATS; r++) {
        size_t i = rand();
        i = i % SDIM;
        size_t j = rand();
        j = j % SDIM;

        int edge_type_index = rand();
        edge_type_index = edge_type_index % 4;

        hvr_irr_matrix_set(i, j, all_edge_types[edge_type_index], IMPLICIT_EDGE, &es, 0);
        assert(hvr_irr_matrix_get(i, j, &es) ==
                all_edge_types[edge_type_index]);
    }

    hvr_irr_matrix_t es2;
    hvr_irr_matrix_init(SDIM, 1024ULL * 1024ULL * 1024ULL, &es2);

    hvr_irr_matrix_set(SDIM - 2, 1, BIDIRECTIONAL, IMPLICIT_EDGE, &es2, 0);
    hvr_irr_matrix_set(SDIM - 2, 2, DIRECTED_IN, IMPLICIT_EDGE, &es2, 0);
    hvr_irr_matrix_set(SDIM - 2, 3, DIRECTED_OUT, IMPLICIT_EDGE, &es2, 0);
    hvr_irr_matrix_set(SDIM - 2, 4, NO_EDGE, IMPLICIT_EDGE, &es2, 0);
    hvr_irr_matrix_set(SDIM - 2, 5, BIDIRECTIONAL, IMPLICIT_EDGE, &es2, 0);

    hvr_edge_info_t vals[10];
    unsigned nvals = hvr_irr_matrix_linearize(SDIM - 2, vals, 10, &es2);

    assert(nvals == 4);
    assert(EDGE_INFO_VERTEX(vals[0]) == 1 && EDGE_INFO_EDGE(vals[0]) == BIDIRECTIONAL);
    assert(EDGE_INFO_VERTEX(vals[1]) == 2 && EDGE_INFO_EDGE(vals[1]) == DIRECTED_IN);
    assert(EDGE_INFO_VERTEX(vals[2]) == 3 && EDGE_INFO_EDGE(vals[2]) == DIRECTED_OUT);
    assert(EDGE_INFO_VERTEX(vals[3]) == 5 && EDGE_INFO_EDGE(vals[3]) == BIDIRECTIONAL);

    hvr_edge_info_t *vals_ptr = NULL;
    nvals = hvr_irr_matrix_linearize_zero_copy(SDIM - 2, &vals_ptr, &es2);

    assert(nvals == 4);
    assert(EDGE_INFO_VERTEX(vals_ptr[0]) == 1 && EDGE_INFO_EDGE(vals_ptr[0]) == BIDIRECTIONAL);
    assert(EDGE_INFO_VERTEX(vals_ptr[1]) == 2 && EDGE_INFO_EDGE(vals_ptr[1]) == DIRECTED_IN);
    assert(EDGE_INFO_VERTEX(vals_ptr[2]) == 3 && EDGE_INFO_EDGE(vals_ptr[2]) == DIRECTED_OUT);
    assert(EDGE_INFO_VERTEX(vals_ptr[3]) == 5 && EDGE_INFO_EDGE(vals_ptr[3]) == BIDIRECTIONAL);

    printf("Success!\n");

    return 0;
}
