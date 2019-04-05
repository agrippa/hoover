#ifndef _HVR_VERTEX_ITER_H
#define _HVR_VERTEX_ITER_H

#include <pthread.h>

#include "hvr_vertex_pool.h"
#include "hvr_vertex.h"

typedef struct _hvr_vertex_iter_t {
    hvr_range_node_t *current_chunk;
    unsigned index_for_current_chunk;
    hvr_vertex_pool_t *pool;
    hvr_internal_ctx_t *ctx;
    int include_all;
} hvr_vertex_iter_t;

typedef struct _hvr_conc_vertex_iter_t {
    hvr_vertex_iter_t child;

    unsigned max_chunk_size;
    unsigned n_chunks_generated;
    pthread_mutex_t mutex;
} hvr_conc_vertex_iter_t;

typedef struct _hvr_conc_vertex_subiter_t {
    hvr_vertex_iter_t child;
    unsigned max_index_for_current_chunk;
} hvr_conc_vertex_subiter_t;

void hvr_vertex_iter_init(hvr_vertex_iter_t *iter,
        hvr_internal_ctx_t *ctx);

void hvr_vertex_iter_all_init(hvr_vertex_iter_t *iter,
        hvr_internal_ctx_t *ctx);

hvr_vertex_t *hvr_vertex_iter_next(hvr_vertex_iter_t *iter);

void hvr_conc_vertex_iter_init(hvr_conc_vertex_iter_t *iter,
        unsigned max_chunk_size, hvr_internal_ctx_t *ctx);

int hvr_conc_vertex_iter_next_chunk(hvr_conc_vertex_iter_t *iter,
        hvr_conc_vertex_subiter_t *chunk);

hvr_vertex_t *hvr_conc_vertex_iter_next(
        hvr_conc_vertex_subiter_t *chunk);

#endif // _HVR_VERTEX_ITER_H
