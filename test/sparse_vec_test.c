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
    hvr_graph_id_t graph = hvr_graph_create(ctx);

    hvr_sparse_vec_t *vec = hvr_sparse_vec_create_n(1, graph, ctx);

    hvr_sparse_vec_set(3, 42.0, vec, ctx);
    // needed to increment to make changes readable
    finalize_actor_for_timestep(vec, ctx->timestep);
    ctx->timestep += 1;

    assert(hvr_sparse_vec_get(3, vec, ctx) == 42.0);

    hvr_sparse_vec_set(5, 43.0, vec, ctx);
    finalize_actor_for_timestep(vec, ctx->timestep);
    ctx->timestep += 1; // need to increment to make changes readable

    assert(hvr_sparse_vec_get(3, vec, ctx) == 42.0);
    assert(hvr_sparse_vec_get(5, vec, ctx) == 43.0);

    hvr_sparse_vec_set(3, 44.0, vec, ctx);
    finalize_actor_for_timestep(vec, ctx->timestep);
    ctx->timestep += 1; // need to increment to make changes readable
    assert(hvr_sparse_vec_get(3, vec, ctx) == 44.0);
    assert(hvr_sparse_vec_get(5, vec, ctx) == 43.0);

    hvr_finalize(ctx);
    shmem_finalize();

    printf("Passed!\n");

    return 0;
}
