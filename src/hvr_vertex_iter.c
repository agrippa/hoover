#include "hoover.h"
#include "hvr_vertex_iter.h"

static int is_valid_vertex(hvr_vertex_t *vertex, hvr_vertex_iter_t *iter) {
    return 1;
}

static void hvr_vertex_iter_init_helper(hvr_vertex_iter_t *iter,
        hvr_internal_ctx_t *ctx) {
    iter->current_chunk = ctx->pool->used_list;
    iter->index_for_current_chunk = 0;
    iter->pool = ctx->pool;
    iter->ctx = ctx;

    // May be NULL if no vertices are allocated yet
    if (iter->current_chunk) {
        // If the first vertex isn't valid, need to iterate to the next one
        hvr_vertex_t *first = iter->pool->pool +
            (iter->current_chunk->start_index + iter->index_for_current_chunk);
        if (!is_valid_vertex(first, iter)) {
            hvr_vertex_iter_next(iter);
        }
    }
}

void hvr_vertex_iter_init(hvr_vertex_iter_t *iter, hvr_internal_ctx_t *ctx) {
    hvr_vertex_iter_init_helper(iter, ctx);
}

hvr_vertex_t *hvr_vertex_iter_next(hvr_vertex_iter_t *iter) {
    if (iter->current_chunk == NULL) {
        return NULL;
    } else {
        hvr_vertex_t *result = iter->pool->pool +
            (iter->current_chunk->start_index + iter->index_for_current_chunk);
        // Haven't already found that we're at the end
        int found = 0;
        while (!found) {
            iter->index_for_current_chunk += 1;
            if (iter->index_for_current_chunk == iter->current_chunk->length) {
                // Move to next chunk, which may be NULL
                iter->current_chunk = iter->current_chunk->next;
                iter->index_for_current_chunk = 0;
            }

            if (iter->current_chunk == NULL) {
                // Reached the end of the list
                found = 1;
            } else {
                // Check if the current vertex is one that we want to visit
                hvr_vertex_t *vertex = iter->pool->pool +
                    (iter->current_chunk->start_index +
                     iter->index_for_current_chunk);

                /*
                 * deleted_timestamp is initialized to HVR_MAX_TIMESTEP.
                 *
                 * created_timestamp < current timestep ensures we don't iterate
                 * over any vertices created in the current timestep.
                 */
                if (is_valid_vertex(vertex, iter)) {
                    found = 1;
                }
            }
        }
        return result;
    }
}
