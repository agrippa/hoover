#include "hoover.h"
#include "hvr_vertex_iter.h"

/*
 * Don't iterate over vertices that were created during our current internal
 * iteration, as they will not have updated edge information.
 */
static inline int is_valid_vertex(const hvr_vertex_t *vertex, const int pe,
        const hvr_time_t iter, const int include_all) {
    assert(VERTEX_ID_PE(vertex->id) == pe);
    return (include_all || vertex->creation_iter < iter);
}

static void hvr_vertex_iter_init_helper(hvr_vertex_iter_t *iter,
        hvr_internal_ctx_t *ctx, int include_all) {
    // Seek to first valid vertex
    iter->include_all = include_all;
    iter->ctx = ctx;
    iter->cache_map = &ctx->vec_cache.cache_map;

    hvr_vertex_cache_node_t *curr = ctx->vec_cache.locals_head;
    while (curr && !is_valid_vertex(&curr->vert, ctx->pe, ctx->iter,
                include_all)) {
        curr = curr->locals_next;
    }
    iter->curr = curr;
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
    const int pe = iter->ctx->pe;
    const hvr_time_t sim_iter = iter->ctx->iter;
    const int include_all = iter->include_all;

    hvr_vertex_t *result = NULL;
    if (iter->curr) {
        hvr_vertex_cache_node_t *curr = iter->curr;
        result = &curr->vert;

        // Seek to the next
        curr = curr->locals_next;
        while (curr && !is_valid_vertex(&curr->vert, pe, sim_iter,
                    include_all)) {
            curr = curr->locals_next;
        }
        iter->curr = curr;
    }
    return result;
}
