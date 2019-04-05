#include "hoover.h"
#include "hvr_vertex_iter.h"

/*
 * Don't iterate over vertices that were created during our current internal
 * iteration, as they will not have updated edge information.
 */
static int is_valid_vertex(hvr_vertex_t *vertex, hvr_vertex_iter_t *iter) {
    return iter->include_all || vertex->creation_iter < iter->ctx->iter;
}

static void hvr_vertex_iter_init_helper(hvr_vertex_iter_t *iter,
        hvr_internal_ctx_t *ctx, int include_all) {
    iter->current_chunk = ctx->pool.tracker.used_list;
    iter->index_for_current_chunk = 0;
    iter->pool = &ctx->pool;
    iter->ctx = ctx;
    iter->include_all = include_all;

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

void hvr_vertex_iter_init(hvr_vertex_iter_t *iter,
        hvr_internal_ctx_t *ctx) {
    hvr_vertex_iter_init_helper(iter, ctx, 0);
}

void hvr_vertex_iter_all_init(hvr_vertex_iter_t *iter,
        hvr_internal_ctx_t *ctx) {
    hvr_vertex_iter_init_helper(iter, ctx, 1);
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

void hvr_conc_vertex_iter_init(hvr_conc_vertex_iter_t *iter,
        unsigned max_chunk_size, hvr_internal_ctx_t *ctx) {
    hvr_vertex_iter_init(&iter->child, ctx);

    iter->max_chunk_size = max_chunk_size;

    int err = pthread_mutex_init(&iter->mutex, NULL);
    assert(err == 0);
}

int hvr_conc_vertex_iter_next_chunk(hvr_conc_vertex_iter_t *iter,
        hvr_conc_vertex_subiter_t *chunk) {
    int success = 1;

    int err = pthread_mutex_lock(&iter->mutex);
    assert(err == 0);

    if (iter->child.current_chunk == NULL) {
        success = 0;
    } else {
        // Set up the subchunk
        memcpy(&chunk->child, &iter->child, sizeof(chunk->child));
        chunk->max_index_for_current_chunk =
            chunk->child.index_for_current_chunk + iter->max_chunk_size;
        if (chunk->max_index_for_current_chunk >
                chunk->child.current_chunk->length) {
            chunk->max_index_for_current_chunk =
                chunk->child.current_chunk->length;
        }

        if (chunk->max_index_for_current_chunk ==
                chunk->child.current_chunk->length) {
            // Go to the next chunk, the remainder of this chunk is handled
            iter->child.current_chunk = iter->child.current_chunk->next;
            iter->child.index_for_current_chunk = 0;
        } else {
            // Just move the current index
            iter->child.index_for_current_chunk =
                chunk->max_index_for_current_chunk;
        }
    }

    err = pthread_mutex_unlock(&iter->mutex);
    assert(err == 0);

    return success;
}

hvr_vertex_t *hvr_conc_vertex_iter_next(hvr_conc_vertex_subiter_t *chunk) {
    if (chunk->child.index_for_current_chunk >=
            chunk->max_index_for_current_chunk) {
        return NULL;
    }

    return hvr_vertex_iter_next(&chunk->child);
}

