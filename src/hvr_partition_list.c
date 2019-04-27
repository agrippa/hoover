#include "hvr_partition_list.h"

void hvr_partition_list_init(hvr_partition_t n_partitions,
        hvr_partition_list_t *l) {
    l->n_partitions = n_partitions;
    l->lists = (hvr_vertex_t **)malloc(n_partitions * sizeof(l->lists[0]));
    assert(l->lists);
    memset(l->lists, 0x00, n_partitions * sizeof(l->lists[0]));
}

void hvr_partition_list_destroy(hvr_partition_list_t *l) {
    free(l->lists);
}

static void prepend_to_partition_list_helper(hvr_vertex_t *curr,
        hvr_partition_t partition, hvr_partition_list_t *l,
        hvr_internal_ctx_t *ctx) {
    assert(partition != HVR_INVALID_PARTITION);

    // Prepend to new partition list
    curr->prev_in_partition = NULL;
    curr->next_in_partition = l->lists[partition];
    if (l->lists[partition]) {
        l->lists[partition]->prev_in_partition = curr;
    }
    l->lists[partition] = curr;
}

void prepend_to_partition_list(hvr_vertex_t *curr,
        hvr_partition_list_t *l, hvr_internal_ctx_t *ctx) {
    hvr_partition_t partition = wrap_actor_to_partition(curr, ctx);

    prepend_to_partition_list_helper(curr, partition, l, ctx);
}

void remove_from_partition_list_helper(hvr_vertex_t *vert,
        hvr_partition_t partition, hvr_partition_list_t *l,
        hvr_internal_ctx_t *ctx) {
    assert(partition != HVR_INVALID_PARTITION);

    if (vert->next_in_partition && vert->prev_in_partition) {
        // Remove from current partition list
        vert->prev_in_partition->next_in_partition =
            vert->next_in_partition;
        vert->next_in_partition->prev_in_partition =
            vert->prev_in_partition;
    } else if (vert->next_in_partition) {
        // prev is NULL, at head of a non-empty list
        assert(l->lists[partition] == vert);
        l->lists[partition] = vert->next_in_partition;
        l->lists[partition]->prev_in_partition = NULL;
    } else if (vert->prev_in_partition) {
        // next is NULL, at tail of a non-empty list
        vert->prev_in_partition->next_in_partition = NULL;
    } else { // both NULL
        assert(l->lists[partition] == vert);
        // Only entry in list
        l->lists[partition] = NULL;
    }
}

void remove_from_partition_list(hvr_vertex_t *curr,
        hvr_partition_list_t *l, hvr_internal_ctx_t *ctx) {
    hvr_partition_t partition = wrap_actor_to_partition(curr, ctx);

    remove_from_partition_list_helper(curr, partition, l, ctx);
}

void update_partition_list_membership(hvr_vertex_t *curr,
        hvr_partition_t old_partition, hvr_partition_list_t *l,
        hvr_internal_ctx_t *ctx) {
    hvr_partition_t new_partition = wrap_actor_to_partition(curr, ctx);

    if (new_partition != old_partition) {
        // Remove from current list
        remove_from_partition_list_helper(curr, old_partition, l, ctx);

        // Prepend to new partition list
        prepend_to_partition_list_helper(curr, new_partition, l, ctx);
    }
}

