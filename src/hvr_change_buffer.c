#include <string.h>

#include "hvr_change_buffer.h"
#include "hvr_internal.h"

void hvr_buffered_changes_init(hvr_buffered_changes_t *changes) {
    memset(changes, 0x00, sizeof(*changes));
}

void hvr_buffered_changes_add(hvr_change_t *change,
        hvr_buffered_changes_t *changes) {
    hvr_timestep_changes_node_t *node = (hvr_timestep_changes_node_t *)malloc(
            sizeof(*node));
    assert(node);
    memcpy(&(node->change), change, sizeof(*change));
    node->next = NULL;

    hvr_timestep_changes_t *prev = NULL;
    hvr_timestep_changes_t *iter = changes->timestep_list;
    while (iter && iter->timestamp < change->timestamp) {
        prev = iter;
        iter = iter->next;
    }

    if (iter && iter->timestamp == change->timestamp) {
        // Insert in iter
        node->next = iter->changes_list;
        iter->changes_list = node;
    } else {
        // Insert before iter. iter may be NULL.
        hvr_timestep_changes_t *new_timestep = (hvr_timestep_changes_t *)malloc(
                sizeof(*new_timestep));
        assert(new_timestep);
        new_timestep->timestamp = change->timestamp;
        new_timestep->changes_list = node;
        new_timestep->next = NULL;

        if (prev == NULL && iter == NULL) {
            // Empty list
            changes->timestep_list = new_timestep;
        } else if (prev == NULL) {
            // iter is non-null, insert at start of timestep list
            assert(changes->timestep_list == iter);
            new_timestep->next = changes->timestep_list;
            changes->timestep_list = new_timestep;
        } else if (iter == NULL) {
            // prev is non-null, insert at end of timestep list
            assert(prev->next == NULL);
            prev->next = new_timestep;
        } else {
            // Both are non-null
            new_timestep->next = iter;
            prev->next = new_timestep;
        }
    }
}

hvr_timestep_changes_t *hvr_buffered_changes_remove_any(hvr_time_t max_timestep,
        hvr_buffered_changes_t *changes) {
    if (changes->timestep_list &&
            changes->timestep_list->timestamp <= max_timestep) {
        hvr_timestep_changes_t *result = changes->timestep_list;
        changes->timestep_list = result->next;
        result->next = NULL;
        return result;
    } else {
        return NULL;
    }
}

void hvr_buffered_changes_free(hvr_timestep_changes_t *changes) {
    hvr_timestep_changes_node_t *iter = changes->changes_list;
    while (iter) {
        hvr_timestep_changes_node_t *next = iter->next;
        free(iter);
        iter = next;
    }
    free(changes);
}


void hvr_vertex_from_change(hvr_change_t *change, hvr_sparse_vec_t *vertex,
        hvr_internal_ctx_t *ctx) {

    hvr_sparse_vec_init_with_const_attrs(vertex, change->graph,
            change->const_features, change->const_values,
            change->n_const_features, ctx);
    vertex->id = change->id;

    for (unsigned i = 0; i < change->size; i++) {
        hvr_sparse_vec_set_internal(change->features[i], change->values[i],
                vertex, 0);
    }

    finalize_actor_for_timestep(vertex, 0);
}
