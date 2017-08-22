#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <shmem.h>

#include "hoover.h"

static long long *p_wrk = NULL;
static long *p_sync = NULL;

void hvr_init(const vertex_id_t n_local_nodes, vertex_id_t *vertex_ids,
        void *vertex_metadata, const size_t vertex_metadata_size,
        vertex_id_t *edges, size_t *edge_offsets, hvr_ctx_t *out_ctx) {
    hvr_internal_ctx_t *new_ctx = (hvr_internal_ctx_t *)malloc(sizeof(*new_ctx));
    assert(new_ctx);

    p_wrk = (long long *)shmem_malloc(
            SHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(*p_wrk));
    p_sync = (long *)shmem_malloc(
            SHMEM_REDUCE_SYNC_SIZE * sizeof(*p_sync));
    assert(p_wrk && p_sync);

    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        p_sync[i] = SHMEM_SYNC_VALUE;
    }

    new_ctx->pe = shmem_my_pe();
    new_ctx->npes = shmem_n_pes();

    long long *tmp_n_nodes = (long long *)shmem_malloc(2 * sizeof(*tmp_n_nodes));
    assert(tmp_n_nodes);
    tmp_n_nodes[0] = n_local_nodes;

    new_ctx->n_local_nodes = n_local_nodes;
    shmem_longlong_sum_to_all(&tmp_n_nodes[1], &tmp_n_nodes[0], 1, 0, 0,
            new_ctx->npes, p_wrk, p_sync);
    shmem_barrier_all();
    shmem_free(tmp_n_nodes);
    new_ctx->n_global_nodes = tmp_n_nodes[1];

    new_ctx->vertex_ids = vertex_ids;
    new_ctx->vertex_metadata = vertex_metadata;
    new_ctx->vertex_metadata_size = vertex_metadata_size;

    new_ctx->edges = edges;
    new_ctx->edge_offsets;

    *out_ctx = new_ctx;
}

void hvr_cleanup(hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    free(ctx);
}
