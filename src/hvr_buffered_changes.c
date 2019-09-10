/* For license: see LICENSE.txt file at top-level */

#include <string.h>
#include <stdio.h>

#include "hvr_buffered_changes.h"

void hvr_buffered_changes_init(size_t nallocated,
        hvr_buffered_changes_t *changes) {
    changes->pool_mem = (hvr_buffered_change_t *)malloc(
            nallocated * sizeof(changes->pool_mem[0]));
    assert(changes->pool_mem);
    changes->nallocated = nallocated;

    changes->head = NULL;
    changes->pool = changes->pool_mem;
    for (int i = 0; i < nallocated - 1; i++) {
        changes->pool_mem[i].next = changes->pool_mem + (i + 1);
    }
    changes->pool_mem[nallocated - 1].next = NULL;
}

void hvr_buffered_changes_edge_create(hvr_vertex_id_t base_id,
        hvr_vertex_id_t neighbor_id, hvr_edge_type_t edge,
        hvr_buffered_changes_t *changes) {
    hvr_buffered_change_t *change = changes->pool;
    if (!change) {
        fprintf(stderr, "ERROR Failed allocating a change object, increase "
                "HVR_BUFFERED_CHANGES_ALLOCATED (currently %lu)\n",
                changes->nallocated);
        abort();
    }
    changes->pool = change->next;
    
    change->next = changes->head;
    changes->head = change;

    change->is_edge_create = 1;
    change->change.edge.base_id = base_id;
    change->change.edge.neighbor_id = neighbor_id;
    change->change.edge.edge = edge;
}

void hvr_buffered_changes_delete_vertex(hvr_vertex_id_t to_delete,
        hvr_buffered_changes_t *changes) {
    hvr_buffered_change_t *change = changes->pool;
    if (!change) {
        fprintf(stderr, "ERROR Failed allocating a change object, increase "
                "HVR_BUFFERED_CHANGES_ALLOCATED (currently %lu)\n",
                changes->nallocated);
        abort();
    }
    changes->pool = change->next;
    
    change->next = changes->head;
    changes->head = change;

    change->is_edge_create = 0;
    change->change.del.to_delete = to_delete;
}

int hvr_buffered_changes_poll(hvr_buffered_changes_t *changes,
        hvr_buffered_change_t *out_change) {
    if (changes->head == NULL) {
        return 0;
    }

    hvr_buffered_change_t *head = changes->head;
    changes->head = head->next;
    memcpy(out_change, head, sizeof(*head));

    head->next = changes->pool;
    changes->pool = head;

    return 1;
}

void hvr_buffered_changes_destroy(hvr_buffered_changes_t *changes) {
    free(changes->pool_mem);
}

