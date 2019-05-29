#include "hvr_map.h"
#include "hvr_partition_list.h"

void hvr_partition_list_init(hvr_partition_t n_partitions,
        hvr_partition_list_t *l) {
    l->n_partitions = n_partitions;
    int segs = 1024;
    if (getenv("HVR_PARTITION_LIST_SEGS")) {
        segs = atoi(getenv("HVR_PARTITION_LIST_SEGS"));
    }
    hvr_map_init(&l->map, segs);
}

void hvr_partition_list_destroy(hvr_partition_list_t *l) {
    hvr_map_destroy(&l->map);
}

static void prepend_to_partition_list_helper(hvr_vertex_t *curr,
        hvr_partition_t partition, hvr_partition_list_t *l,
        hvr_internal_ctx_t *ctx) {
    assert(partition != HVR_INVALID_PARTITION);

    hvr_vertex_t *head = hvr_map_get(partition, &l->map);
    curr->prev_in_partition = NULL;
    curr->next_in_partition = head;

    if (head) {
        head->prev_in_partition = curr;
    }
    hvr_map_add(partition, curr, 1, &l->map);
}

hvr_partition_t prepend_to_partition_list(hvr_vertex_t *curr,
        hvr_partition_list_t *l, hvr_internal_ctx_t *ctx) {
    hvr_partition_t partition = wrap_actor_to_partition(curr, ctx);

    prepend_to_partition_list_helper(curr, partition, l, ctx);
    return partition;
}

void remove_from_partition_list_helper(const hvr_vertex_t *vert,
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
        hvr_vertex_t *head = hvr_map_get(partition, &l->map);
        assert(head == vert);
        hvr_vertex_t *new_head = head->next_in_partition;
        new_head->prev_in_partition = NULL;
        hvr_map_add(partition, new_head, 1, &l->map);
    } else if (vert->prev_in_partition) {
        // next is NULL, at tail of a non-empty list
        vert->prev_in_partition->next_in_partition = NULL;
    } else { // both NULL
        hvr_vertex_t *head = hvr_map_get(partition, &l->map);
        assert(head == vert);
        // Only entry in list
        hvr_map_remove(partition, head, &l->map);
    }
}

void remove_from_partition_list(hvr_vertex_t *curr,
        hvr_partition_list_t *l, hvr_internal_ctx_t *ctx) {
    hvr_partition_t partition = wrap_actor_to_partition(curr, ctx);

    remove_from_partition_list_helper(curr, partition, l, ctx);
}

void update_partition_list_membership(hvr_vertex_t *curr,
        hvr_partition_t old_partition, hvr_partition_t optional_new_partition,
        hvr_partition_list_t *l, hvr_internal_ctx_t *ctx) {
    hvr_partition_t new_partition =
        (optional_new_partition == HVR_INVALID_PARTITION ?
         wrap_actor_to_partition(curr, ctx) : optional_new_partition);

    if (new_partition != old_partition) {
        // Remove from current list
        remove_from_partition_list_helper(curr, old_partition, l, ctx);

        // Prepend to new partition list
        prepend_to_partition_list_helper(curr, new_partition, l, ctx);
    }
}

hvr_vertex_t *hvr_partition_list_head(hvr_partition_t part,
        hvr_partition_list_t *l) {
    return hvr_map_get(part, &l->map);
}

size_t hvr_partition_list_mem_used(hvr_partition_list_t *l) {
    size_t capacity, used;
    hvr_map_size_in_bytes(&l->map, &capacity, &used, 0);
    return capacity;
}

