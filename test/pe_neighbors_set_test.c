#include <stdio.h>
#include <assert.h>
#include <hoover.h>

#define NPES 100

int main(int argc, char **argv) {

    hvr_ctx_t ctx;
    hvr_ctx_create(&ctx);
    ((hvr_internal_ctx_t *)ctx)->npes = NPES;

    hvr_pe_neighbors_set_t *set = hvr_create_empty_pe_neighbors_set(ctx);

    for (int i = 0; i < NPES; i++) {
        hvr_pe_neighbors_set_insert(i, set);
        assert(hvr_pe_neighbors_set_contains(i, set) == 1);
        for (int j = 0; j < NPES; j++) {
            if (j != i) assert(hvr_pe_neighbors_set_contains(j, set) == 0);
        }
        hvr_pe_neighbors_set_clear(i, set);
        for (int j = 0; j < NPES; j++) {
            assert(hvr_pe_neighbors_set_contains(j, set) == 0);
        }
    }

    hvr_pe_neighbors_set_insert(2, set);
    hvr_pe_neighbors_set_insert(8, set);
    for (int i = 0; i < NPES; i++) {
        if (i == 2 || i == 8) {
            assert(hvr_pe_neighbors_set_contains(i, set) == 1);
        } else {
            assert(hvr_pe_neighbors_set_contains(i, set) == 0);
        }
    }

    printf("Passed!\n");

    return 0;
}
