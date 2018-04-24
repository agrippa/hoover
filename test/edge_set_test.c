/* For license: see LICENSE.txt file at top-level */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "hoover.h"

void assert_only_edge_not_set(unsigned i, unsigned j, hvr_edge_set_t *set) {
    for (unsigned ii = 0; ii < 10; ii++) {
        for (unsigned jj = 0; jj < 10; jj++) {
            if (ii == i && jj == j) {
                if (hvr_have_edge(ii, jj, set) != 0) {
                    fprintf(stderr, "Expected edge (%u, %u) to not be set, but "
                            "was.\n", ii, jj);
                    abort();
                }
            } else {
                if (hvr_have_edge(ii, jj, set) != 1) {
                    fprintf(stderr, "Expected edge (%u, %u) to be set, but "
                            "was not.\n", ii, jj);
                    abort();
                }
            }
        }
    }
}

void assert_only_edge_set(unsigned i, unsigned j, hvr_edge_set_t *set) {
    for (unsigned ii = 0; ii < 10; ii++) {
        for (unsigned jj = 0; jj < 10; jj++) {
            if (ii == i && jj == j) {
                if (hvr_have_edge(ii, jj, set) != 1) {
                    fprintf(stderr, "Expected edge (%u, %u) to be set, but "
                            "was not.\n", ii, jj);
                    abort();
                }
            } else {
                if (hvr_have_edge(ii, jj, set) != 0) {
                    fprintf(stderr, "Expected edge (%u, %u) to not be set, but "
                            "was.\n", ii, jj);
                    abort();
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    hvr_edge_set_t *set = hvr_create_empty_edge_set();

    for (unsigned i = 0; i < 10; i++) {
        for (unsigned j = 0; j < 10; j++) {
            assert(hvr_have_edge(i, j, set) == 0);
        }
    }

    for (unsigned i = 0; i < 10; i++) {
        for (unsigned j = 0; j < 10; j++) {
            hvr_add_edge(i, j, set);

            if (hvr_have_edge(i, j, set) != 1) {
                printf("Expected to find (%d, %d) in set:\n", i, j);
                hvr_print_edge_set(set);
                abort();
            }

            if (j < 9) {
                assert(hvr_have_edge(i, j + 1, set) == 0);
            }
        }
    }

    hvr_clear_edge_set(set);

    for (unsigned i = 0; i < 10; i++) {
        for (unsigned j = 0; j < 10; j++) {
            assert(hvr_have_edge(i, j, set) == 0);
        }
    }

    hvr_add_edge(5, 5, set);
    assert_only_edge_set(5, 5, set);
    hvr_clear_edge_set(set);

    hvr_add_edge(3, 6, set);
    assert_only_edge_set(3, 6, set);

    hvr_clear_edge_set(set);

    for (unsigned i = 0; i < 10; i++) {
        for (unsigned j = 0; j < 10; j++) {
            assert(hvr_have_edge(i, j, set) == 0);
        }
    }

    for (int i = 1; i < 10; i += 2) {
        hvr_add_edge(5, i, set);
    }
    for (int i = 0; i < 10; i++) {
        if (i % 2 == 0) {
            assert(hvr_have_edge(5, i, set) == 0);
        } else {
            if (hvr_have_edge(5, i, set) != 1) {
                fprintf(stderr, "Expected edge (5, %d) but it wasn't there.\n",
                        i);
                hvr_print_edge_set(set);
                abort();
            }
        }
    }
    assert(hvr_count_edges(5, set) == 5);

    hvr_release_edge_set(set);

    printf("Passed!\n");

    return 0;
}
