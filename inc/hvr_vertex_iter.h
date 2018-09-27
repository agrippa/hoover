#ifndef _HVR_VERTEX_ITER_H
#define _HVR_VERTEX_ITER_H

#include "hvr_vertex_pool.h"
#include "hvr_vertex.h"

typedef struct _hvr_vertex_iter_t {
    hvr_vertex_range_node_t *current_chunk;
    unsigned index_for_current_chunk;
    hvr_vertex_pool_t *pool;
    hvr_internal_ctx_t *ctx;
    int include_all;
} hvr_vertex_iter_t;

void hvr_vertex_iter_init(hvr_vertex_iter_t *iter,
        hvr_internal_ctx_t *ctx);

void hvr_vertex_iter_all_init(hvr_vertex_iter_t *iter,
        hvr_internal_ctx_t *ctx);

hvr_vertex_t *hvr_vertex_iter_next(hvr_vertex_iter_t *iter);

#endif // _HVR_VERTEX_ITER_H
