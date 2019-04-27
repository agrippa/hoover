#include <assert.h>
#include <shmem.h>

#include "hoover.h"
#include "hvr_vertex_pool.h"

static hvr_range_node_t *create_node(unsigned start_index,
        unsigned length, hvr_range_tracker_t *tracker)  {
    hvr_range_node_t *node = tracker->preallocated_nodes;
    assert(node);
    tracker->preallocated_nodes = node->next;

    node->start_index = start_index;
    node->length = length;
    node->next = NULL;
    node->prev = NULL;
    return node;
}

static void release_node(hvr_range_node_t *node, hvr_range_tracker_t *tracker) {
    node->next = tracker->preallocated_nodes;
    tracker->preallocated_nodes = node;
}

/*
 * Returns new head for the list, which may be the same as the old head.
 *
 * Removes the range of values starting at start_index (inclusive) and including
 * the following nvecs values.
 */
static hvr_range_node_t *remove_from_range(unsigned start_index,
        unsigned nvecs, hvr_range_node_t *list_head,
        hvr_range_tracker_t *tracker) {
    // Find the node containing the range start_index to start_index + nvecs
    hvr_range_node_t *curr = list_head;
    while (curr && !(start_index >= curr->start_index &&
                start_index < curr->start_index + curr->length)) {
        curr = curr->next;
    }

    /*
     * Assert that the whole block at start_index is included in our current
     * range node.
     */
    assert(curr);
    assert(start_index >= curr->start_index &&
            start_index < curr->start_index + curr->length);
    assert(start_index + nvecs > curr->start_index &&
            start_index + nvecs <= curr->start_index + curr->length);

    unsigned nvecs_before = start_index - curr->start_index;
    unsigned nvecs_after = (curr->start_index + curr->length) -
        (start_index + nvecs);
    hvr_range_node_t *prev = curr->prev;
    hvr_range_node_t *next = curr->next;

    hvr_range_node_t *new_head = NULL;
    if (nvecs_before == 0 && nvecs_after == 0) {
        // Remove whole node
        if (prev == NULL && next == NULL) {
            // Only element in the list
            assert(list_head == curr);
            new_head = NULL;
        } else if (prev == NULL) {
            // First element in a list of at least 2 elements
            assert(list_head == curr);
            new_head = next;
            new_head->prev = NULL;
        } else if (next == NULL) {
            // Last element in a list of at least 2 elements
            prev->next = NULL;
            new_head = list_head;
        } else {
            // Interior element
            prev->next = next;
            next->prev = prev;
            new_head = list_head;
        }

        release_node(curr, tracker);
    } else if (nvecs_before > 0 && nvecs_after > 0) {
        // Pull interior section out of current node
        hvr_range_node_t *new_node = create_node(start_index + nvecs,
                nvecs_after, tracker);
        new_node->next = curr->next;
        new_node->prev = curr;

        if (curr->next) {
            curr->next->prev = new_node;
        }

        curr->length = nvecs_before;
        curr->next = new_node;

        new_head = list_head;
    } else if (nvecs_before == 0) {
        // Just move this node's start forward
        curr->start_index += nvecs;
        curr->length -= nvecs;
        new_head = list_head;
    } else if (nvecs_after == 0) {
        // Move this node's end backwards
        curr->length -= nvecs;
        new_head = list_head;
    } else {
        assert(0);
    }

    return new_head;
}

static hvr_range_node_t *add_to_range(unsigned start_index,
        unsigned nvecs, hvr_range_node_t *list_head,
        hvr_range_tracker_t *tracker) {
    hvr_range_node_t *prev = NULL;
    hvr_range_node_t *next = list_head;
    while (next && next->start_index < start_index) {
        prev = next;
        next = next->next;
    }

    /*
     * Prev is the last node that has a start offset < start_index. So, we need
     * to assert that this range doesn't overlap with prev or next and then
     * merge with one or both.
     */
    assert(prev == NULL || start_index >= prev->start_index + prev->length);
    assert(next == NULL || start_index + nvecs <= next->start_index);

    hvr_range_node_t *new_head = NULL;
    if (prev && next && start_index == prev->start_index + prev->length &&
            start_index + nvecs == next->start_index) {
        // Merge with both, both must be non-NULL
        prev->length += (nvecs + next->length);
        prev->next = next->next;
        if (next->next) {
            next->next->prev = prev;
        }
        release_node(next, tracker);
        new_head = list_head;
    } else if ((prev == NULL ||
                start_index > prev->start_index + prev->length) &&
            (next == NULL || start_index + nvecs < next->start_index)) {
        // Merge with neither, either may be NULL
        if (prev == NULL && next == NULL) {
            // Both prev and next are NULL, empty list
            assert(list_head == NULL);
            new_head = create_node(start_index, nvecs, tracker);
        } else if (prev == NULL) {
            // prev is NULL, inserting at the front of a non-empty list
            assert(list_head == next);
            new_head = create_node(start_index, nvecs, tracker);
            new_head->next = next;
            next->prev = new_head;
        } else if (next == NULL) {
            // next is NULL, inserting at end of a non-empty list
            hvr_range_node_t *new_node = create_node(start_index,
                    nvecs, tracker);
            prev->next = new_node;
            new_node->prev = prev;
            new_head = list_head;
        } else { // both non-NULL
            hvr_range_node_t *new_node = create_node(start_index,
                    nvecs, tracker);
            prev->next = new_node;
            new_node->prev = prev;
            new_node->next = next;
            next->prev = new_node;
            new_head = list_head;
        }
    } else if (prev && start_index == prev->start_index + prev->length) {
        // Merge with prev, must be non-NULL
        prev->length += nvecs;
        new_head = list_head;
    } else if (next && start_index + nvecs == next->start_index) {
        // Merge with next
        next->start_index -= nvecs;
        next->length += nvecs;
        new_head = list_head;
    } else {
        assert(0);
    }

    return new_head;
}

void hvr_range_tracker_init(size_t capacity, int n_nodes,
        hvr_range_tracker_t *tracker) {
    hvr_range_node_t *prealloc = (hvr_range_node_t *)malloc(
            n_nodes * sizeof(*prealloc));
    assert(prealloc || n_nodes == 0);
    tracker->preallocated_nodes = prealloc;
    tracker->mem = prealloc;
    for (int i = 0; i < n_nodes - 1; i++) {
        prealloc[i].next = prealloc + (i + 1);
    }
    prealloc[n_nodes - 1].next = NULL;

    tracker->used_list = NULL;
    tracker->free_list = create_node(0, capacity, tracker);
    tracker->capacity = capacity;
    tracker->used = 0;
    tracker->n_nodes = n_nodes;
    
}

void hvr_range_tracker_destroy(hvr_range_tracker_t *tracker) {
    free(tracker->mem);
}

size_t hvr_range_tracker_reserve(size_t space, hvr_range_tracker_t *tracker) {
    // Greedily find the first free node large enough to satisfy this request
    hvr_range_node_t *curr = tracker->free_list;
    size_t nfree = 0;
    while (curr && curr->length < space) {
        nfree += curr->length;
        curr = curr->next;
    }

    if (curr == NULL) {
        fprintf(stderr, "HOOVER> ERROR Ran out of vertices in the pool "
                "on PE %d. # free = %lu, allocating %lu, total %lu\n",
                shmem_my_pe(), nfree, space, tracker->capacity);
        abort();
    }

    const unsigned alloc_start_index = curr->start_index;
    tracker->free_list = remove_from_range(alloc_start_index, space,
            tracker->free_list, tracker);
    tracker->used_list = add_to_range(alloc_start_index, space,
            tracker->used_list, tracker);
    tracker->used += space;

    return alloc_start_index;
}

void hvr_range_tracker_release(size_t offset, size_t space,
        hvr_range_tracker_t *tracker) {

    tracker->used_list = remove_from_range(offset, space, tracker->used_list,
            tracker);
    tracker->free_list = add_to_range(offset, space, tracker->free_list,
            tracker);
    tracker->used -= space;
}

static size_t hvr_range_tracker_used(hvr_range_tracker_t *tracker) {
    size_t neles = 0;

    hvr_range_node_t *iter = tracker->used_list;
    while (iter) {
        neles += iter->length;
        iter = iter->next;
    }

    return neles;
}

size_t hvr_range_tracker_size_in_bytes(hvr_range_tracker_t *tracker) {
    size_t nbytes = 0;
    hvr_range_node_t *iter = tracker->free_list;
    while (iter) {
        nbytes += sizeof(*iter);
        iter = iter->next;
    }

    iter = tracker->used_list;
    while (iter) {
        nbytes += sizeof(*iter);
        iter = iter->next;
    }

    iter = tracker->preallocated_nodes;
    while (iter) {
        nbytes += sizeof(*iter);
        iter = iter->next;
    }

    return nbytes;
}

void hvr_vertex_pool_create(size_t pool_size, size_t n_nodes,
        hvr_vertex_pool_t *pool) {
    pool->pool = (hvr_vertex_t *)malloc(pool_size * sizeof(hvr_vertex_t));
    if (pool->pool == NULL) {
        fprintf(stderr, "PE %d failed allocating sparse vec pool of size %lu "
                "bytes\n", shmem_my_pe(), pool_size * sizeof(hvr_vertex_t));
        abort();
    }

    for (unsigned i = 0; i < pool_size; i++) {
        (pool->pool)[i].id = HVR_INVALID_VERTEX_ID;
    }

    hvr_range_tracker_init(pool_size, n_nodes, &pool->tracker);
    pool->pool_size = pool_size;
}

hvr_vertex_t *hvr_alloc_vertices(unsigned nvecs, hvr_ctx_t in_ctx) {
    if (nvecs == 0) {
        return NULL;
    }

    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    hvr_vertex_pool_t *pool = &ctx->pool;
    const size_t alloc_start_index = hvr_range_tracker_reserve(nvecs,
            &pool->tracker);

    // Initialize each of the reserved vectors, including giving them valid IDs
    hvr_vertex_t *allocated = pool->pool + alloc_start_index;
    for (size_t i = 0; i < nvecs; i++) {
        hvr_vertex_init(&allocated[i], ctx);
    }
    return allocated;
}

void hvr_free_vertices(hvr_vertex_t *vecs, unsigned nvecs,
        hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    hvr_vertex_pool_t *pool = &ctx->pool;
    const size_t pool_index = vecs - pool->pool;

    hvr_range_tracker_release(pool_index, nvecs, &pool->tracker);

    // Indicate to anyone scanning the pool that these are invalid vertices
    for (unsigned i = 0; i < nvecs; i++) {
        vecs[i].id = HVR_INVALID_VERTEX_ID;
    }
}

size_t hvr_n_allocated(hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    return ctx->pool.tracker.used;
}

void hvr_pool_size_in_bytes(size_t *used, size_t *allocated, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    hvr_vertex_pool_t *pool = &ctx->pool;

    size_t tracker_size = hvr_range_tracker_size_in_bytes(&pool->tracker);

    *used = tracker_size + hvr_range_tracker_used(&pool->tracker) *
        sizeof(hvr_vertex_t);
    *allocated = tracker_size + pool->pool_size * sizeof(hvr_vertex_t);
}

void hvr_vertex_pool_destroy(hvr_vertex_pool_t *pool) {
    free(pool->pool);
    hvr_range_tracker_destroy(&pool->tracker);
}
