#include "hoover.h"
#include "hvr_vertex_iter.h"

/*
 * Don't iterate over vertices that were created during our current internal
 * iteration, as they will not have updated edge information.
 */
static inline int is_valid_vertex(hvr_vertex_t *vertex, hvr_vertex_iter_t *iter) {
    return (VERTEX_ID_PE(vertex->id) == iter->ctx->pe) &&
        (iter->include_all || vertex->creation_iter < iter->ctx->iter);
}

static inline int hvr_vertex_iter_next_helper(int *inout_bucket,
        hvr_map_seg_t **inout_seg, unsigned *inout_seg_index,
        hvr_vertex_iter_t *iter) {
    int bucket = *inout_bucket;
    while (bucket < HVR_MAP_BUCKETS) {
        hvr_map_seg_t *seg = (bucket == *inout_bucket ? *inout_seg :
                iter->cache_map->buckets[bucket]);
        while (seg) {
            unsigned seg_index = (seg == *inout_seg ? *inout_seg_index : 0);
            while (seg_index < seg->nkeys) {
                hvr_vertex_cache_node_t *node = seg->data[seg_index].data;
                if (is_valid_vertex(&node->vert, iter)) {
                    *inout_bucket = bucket;
                    *inout_seg = seg;
                    *inout_seg_index = seg_index;
                    return 1;
                }
                seg_index++;
            }
            seg = seg->next;
        }
        bucket++;
    }

    return 0;
}

static void hvr_vertex_iter_init_helper(hvr_vertex_iter_t *iter,
        hvr_internal_ctx_t *ctx, int include_all) {
    // Seek to first valid vertex
    iter->include_all = include_all;
    iter->ctx = ctx;
    iter->cache_map = &ctx->vec_cache.cache_map;

    iter->current_bucket = 0;
    iter->current_seg = iter->cache_map->buckets[0];
    iter->current_seg_index = 0;
}

void hvr_vertex_iter_init(hvr_vertex_iter_t *iter,
        hvr_internal_ctx_t *ctx) {
    hvr_vertex_iter_init_helper(iter, ctx, 0);
}

void hvr_vertex_iter_all_init(hvr_vertex_iter_t *iter,
        hvr_internal_ctx_t *ctx) {
    hvr_vertex_iter_init_helper(iter, ctx, 1);
}

hvr_vertex_t *hvr_vertex_iter_next(hvr_vertex_iter_t *iter) {
    int bucket = iter->current_bucket;
    hvr_map_seg_t *seg = iter->current_seg;
    unsigned seg_index = iter->current_seg_index;

    int success = hvr_vertex_iter_next_helper(&bucket, &seg, &seg_index, iter);

    iter->current_bucket = bucket;
    iter->current_seg = seg;
    iter->current_seg_index = seg_index + 1;

    if (success) {
        hvr_vertex_cache_node_t *node = seg->data[seg_index].data;
        hvr_vertex_t *result = &node->vert;
        return result;
    } else {
        return NULL;
    }
}
