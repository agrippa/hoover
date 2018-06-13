#include <assert.h>
#include <shmem.h>

#include "hoover.h"
#include "hvr_sparse_vec_pool.h"

static hvr_sparse_vec_range_node_t *remove_from_range(unsigned start_index,
        unsigned nvecs, hvr_sparse_vec_range_node_t *list_head);

static hvr_sparse_vec_range_node_t *add_to_range(unsigned start_index,
        unsigned nvecs, hvr_sparse_vec_range_node_t *list_head);

static hvr_sparse_vec_range_node_t *create_node(unsigned start_index,
        unsigned length)  {
    hvr_sparse_vec_range_node_t *node =
        (hvr_sparse_vec_range_node_t *)malloc(sizeof(*node));
    assert(node);

    node->start_index = start_index;
    node->length = length;
    node->next = NULL;
    node->prev = NULL;
    return node;
}

/*
 * Returns new head for the list, which may be the same as the old head.
 *
 * Removes the range of values starting at start_index (inclusive) and including
 * the following nvecs values.
 */
static hvr_sparse_vec_range_node_t *remove_from_range(unsigned start_index,
        unsigned nvecs, hvr_sparse_vec_range_node_t *list_head) {
    // Find the node containing the range start_index to start_index + nvecs
    hvr_sparse_vec_range_node_t *curr = list_head;
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
    assert(start_index + nvecs >= curr->start_index &&
            start_index + nvecs <= curr->start_index + curr->length);

    unsigned nvecs_before = start_index - curr->start_index;
    unsigned nvecs_after = (curr->start_index + curr->length) -
        (start_index + nvecs);

    hvr_sparse_vec_range_node_t *new_head = NULL;
    if (nvecs_before == 0 && nvecs_after == 0) {
        // Remove whole node
        if (curr->prev == NULL && curr->next == NULL) {
            // Only element in the list
            assert(list_head == curr);
            new_head = NULL;
        } else if (curr->prev == NULL) {
            // First element in a list of at least 2 elements
            assert(list_head == curr);
            new_head = curr->next;
            new_head->prev = NULL;
        } else if (curr->next == NULL) {
            // Last element in a list of at least 2 elements
            curr->prev->next = NULL;
            new_head = list_head;
        } else {
            // Interior element
            curr->prev->next = curr->next;
            curr->next->prev = curr->prev;
            new_head = list_head;
        }

        free(curr);
    } else if (nvecs_before > 0 && nvecs_after > 0) {
        // Pull interior section out of current node
        hvr_sparse_vec_range_node_t *new_node = create_node(start_index + nvecs,
                nvecs_after);
        new_node->next = curr->next;
        new_node->prev = curr;

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
    assert(new_head);

    return new_head;
}

static hvr_sparse_vec_range_node_t *add_to_range(unsigned start_index,
        unsigned nvecs, hvr_sparse_vec_range_node_t *list_head) {
    hvr_sparse_vec_range_node_t *prev = NULL;
    hvr_sparse_vec_range_node_t *next = list_head;
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

    hvr_sparse_vec_range_node_t *new_head = NULL;
    if (prev && next && start_index == prev->start_index + prev->length &&
            start_index + nvecs == next->start_index) {
        // Merge with both, both must be non-NULL
        prev->length += (nvecs + next->length);
        prev->next = next->next;
        next->next->prev = prev;
        free(next);
        new_head = list_head;
    } else if ((prev == NULL ||
                start_index > prev->start_index + prev->length) &&
            (next == NULL || start_index + nvecs < next->start_index)) {
        // Merge with neither, either may be NULL
        if (prev == NULL && next == NULL) {
            // Both prev and next are NULL, empty list
            assert(list_head == NULL);
            new_head = create_node(start_index, nvecs);
        } else if (prev == NULL) {
            // prev is NULL, inserting at the front of a non-empty list
            assert(list_head == next);
            new_head = create_node(start_index, nvecs);
            new_head->next = next;
            next->prev = new_head;
        } else if (next == NULL) {
            // next is NULL, inserting at end of a non-empty list
            hvr_sparse_vec_range_node_t *new_node = create_node(start_index,
                    nvecs);
            prev->next = new_node;
            new_node->prev = prev;
            new_head = list_head;
        } else { // both non-NULL
            hvr_sparse_vec_range_node_t *new_node = create_node(start_index,
                    nvecs);
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

hvr_sparse_vec_pool_t *hvr_sparse_vec_pool_create(size_t pool_size) {
    hvr_sparse_vec_pool_t *pool = (hvr_sparse_vec_pool_t *)malloc(
            sizeof(*pool));
    assert(pool);

    pool->pool = (hvr_sparse_vec_t *)shmem_malloc(
            pool_size * sizeof(hvr_sparse_vec_t));
    if (pool->pool == NULL) {
        fprintf(stderr, "PE %d failed allocating sparse vec pool of size %lu "
                "bytes\n", shmem_my_pe(), pool_size * sizeof(hvr_sparse_vec_t));
        abort();
    }

    for (unsigned i = 0; i < pool_size; i++) {
        (pool->pool)[i].id = HVR_INVALID_VERTEX_ID;
    }

    pool->pool_size = pool_size;
    pool->free_list = create_node(0, pool_size);
    pool->used_list = NULL;

    return pool;
}

hvr_sparse_vec_t *hvr_alloc_sparse_vecs(unsigned nvecs,
        hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    hvr_sparse_vec_pool_t *pool = ctx->pool;

    // Greedily find the first free node large enough to satisfy this request
    hvr_sparse_vec_range_node_t *curr = pool->free_list;
    size_t nfree = 0;
    while (curr && curr->length < nvecs) {
        nfree += curr->length;
        curr = curr->next;
    }

    if (curr == NULL) {
        fprintf(stderr, "HOOVER> ERROR Ran out of sparse vectors in the pool "
                "on PE %d. # free = %lu, allocating %u\n", shmem_my_pe(),
                nfree, nvecs);
        abort();
    }

    const unsigned alloc_start_index = curr->start_index;
    pool->free_list = remove_from_range(alloc_start_index, nvecs,
            pool->free_list);
    pool->used_list = add_to_range(alloc_start_index, nvecs, pool->used_list);

    // Initialize each of the reserved vectors, including giving them valid IDs
    hvr_sparse_vec_t *allocated = pool->pool + alloc_start_index;
    for (size_t i = 0; i < nvecs; i++) {
        hvr_sparse_vec_init(&allocated[i], ctx);
    }
    return allocated;
}

void hvr_free_sparse_vecs(hvr_sparse_vec_t *vecs, unsigned nvecs,
        hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    hvr_sparse_vec_pool_t *pool = ctx->pool;
    const size_t pool_index = vecs - pool->pool;

    pool->used_list = remove_from_range(pool_index, nvecs, pool->used_list);
    pool->free_list = add_to_range(pool_index, nvecs, pool->free_list);

    // Indicate to anyone scanning the pool that these are invalid vertices
    for (unsigned i = 0; i < nvecs; i++) {
        vecs[i].id = HVR_INVALID_VERTEX_ID;
    }
}
