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

    hvr_vertex_t *vec = hvr_vertex_create(ctx);

    hvr_vertex_set(3, 42.0, vec, ctx);

    assert(hvr_vertex_get(3, vec, ctx) == 42.0);

    hvr_vertex_set(5, 43.0, vec, ctx);

    assert(hvr_vertex_get(3, vec, ctx) == 42.0);
    assert(hvr_vertex_get(5, vec, ctx) == 43.0);

    hvr_vertex_set(3, 44.0, vec, ctx);
    assert(hvr_vertex_get(3, vec, ctx) == 44.0);
    assert(hvr_vertex_get(5, vec, ctx) == 43.0);

    hvr_finalize(ctx);
    shmem_finalize();

    printf("Passed!\n");

    return 0;
}
