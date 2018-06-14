#include "hoover.h"
#include "hvr_vertex_iter.h"

void hvr_vertex_iter_init(hvr_vertex_iter_t *iter,
        hvr_graph_id_t target_graphs, hvr_internal_ctx_t *ctx) {
    iter->current_chunk = ctx->pool->used_list;
    iter->index_for_current_chunk = 0;
    iter->pool = ctx->pool;
    iter->ctx = ctx;
    iter->target_graphs = target_graphs;
}

hvr_sparse_vec_t *hvr_vertex_iter_next(hvr_vertex_iter_t *iter) {
    if (iter->current_chunk == NULL) {
        return NULL;
    } else {
        hvr_sparse_vec_t *result = iter->pool->pool +
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
                hvr_sparse_vec_t *vertex = iter->pool->pool +
                    (iter->current_chunk->start_index +
                     iter->index_for_current_chunk);
                if (vertex->created_timestamp < iter->ctx->timestep &&
                        (vertex->graph & iter->target_graphs) != 0) {
                    found = 1;
                }
            }
        }
        return result;
    }
}
