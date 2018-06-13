/* For license: see LICENSE.txt file at top-level */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <shmem.h>

#include "hoover.h"

int main(int argc, char **argv) {
    shmem_init();

    hvr_ctx_t ctx;
    hvr_ctx_create(&ctx);

    hvr_sparse_vec_t *vec = hvr_sparse_vec_create_n(1, ctx);

    // Everything should be zero
    for (unsigned i = 0; i < 100; i++) {
        assert(hvr_sparse_vec_get(i, vec, ctx) == 0.0);
    }

    hvr_sparse_vec_set(3, 42.0, vec, ctx);
    ctx->timestep += 1; // need to increment to make changes readable
    for (unsigned i = 0; i < 100; i++) {
        if (i == 3) {
            assert(hvr_sparse_vec_get(i, vec, ctx) == 42.0);
        } else {
            assert(hvr_sparse_vec_get(i, vec, ctx) == 0.0);
        }
    }

    hvr_sparse_vec_set(5, 43.0, vec, ctx);
    ctx->timestep += 1; // need to increment to make changes readable
    for (unsigned i = 0; i < 100; i++) {
        if (i == 3) {
            assert(hvr_sparse_vec_get(i, vec, ctx) == 42.0);
        } else if (i == 5) {
            assert(hvr_sparse_vec_get(i, vec, ctx) == 43.0);
        } else {
            assert(hvr_sparse_vec_get(i, vec, ctx) == 0.0);
        }
    }

    hvr_sparse_vec_set(3, 44.0, vec, ctx);
    ctx->timestep += 1; // need to increment to make changes readable
     for (unsigned i = 0; i < 100; i++) {
        if (i == 3) {
            assert(hvr_sparse_vec_get(i, vec, ctx) == 44.0);
        } else if (i == 5) {
            assert(hvr_sparse_vec_get(i, vec, ctx) == 43.0);
        } else {
            assert(hvr_sparse_vec_get(i, vec, ctx) == 0.0);
        }
    }

    hvr_finalize(ctx);
    shmem_finalize();

    printf("Passed!\n");

    return 0;
}
