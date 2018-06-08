#include "hvr_vertex_iter.h"

        hvr_sparse_vec_range_node_t *iter = ctx->pool->used_list;
        size_t count_vertices = 0;
        while (iter) {
            for (unsigned a = 0; a < iter->length; a++) {
                hvr_sparse_vec_t *curr = ctx->pool->pool +
                    (iter->start_index + a);
                if (curr->created_timestamp < ctx->timestep) {
                    sum_n_neighbors += update_local_actor_metadata(curr,
                            to_couple_with,
                            &fetch_neighbors_time, &quiet_neighbors_time,
                            &update_metadata_time, ctx,
                            vec_caches, &quiet_counter);
                    count_vertices++;
                }
            }
            iter = iter->next;
        }


void hvr_vertex_iter_init(hvr_vertex_iter_t *iter, hvr_sparse_vec_t *pool) {
    iter->next = pool->used_list;
    iter->index_for_current_chunk = 0;
    iter->pool = pool;
}

hvr_sparse_vec_t *hvr_vertex_iter_next(hvr_vertex_iter_t *iter) {
    hvr_sparse_vec_t *result = iter->next;
    if (result) {
        // Haven't already found that we're at the end
    }
    return result;
}
