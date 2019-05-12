#ifndef _HVR_VERTEX_ITER_H
#define _HVR_VERTEX_ITER_H

#include <pthread.h>

#include "hvr_vertex.h"

typedef struct _hvr_vertex_iter_t {
    int include_all;
    hvr_internal_ctx_t *ctx;
    hvr_map_t *cache_map;

    unsigned current_bucket;
    hvr_map_seg_t *current_seg;
    unsigned current_seg_index;


} hvr_vertex_iter_t;

void hvr_vertex_iter_init(hvr_vertex_iter_t *iter,
        hvr_internal_ctx_t *ctx);

void hvr_vertex_iter_all_init(hvr_vertex_iter_t *iter,
        hvr_internal_ctx_t *ctx);

hvr_vertex_t *hvr_vertex_iter_next(hvr_vertex_iter_t *iter);

#endif // _HVR_VERTEX_ITER_H
