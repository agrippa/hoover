#include <assert.h>
#include <shmem.h>

#include "hoover.h"
#include "hvr_sparse_vec_pool.h"

static hvr_sparse_vec_pool_free_node_t *create_free_node(unsigned start_index,
        unsigned length)  {
    hvr_sparse_vec_pool_free_node_t *node =
        (hvr_sparse_vec_pool_free_node_t *)malloc(sizeof(*node));
    assert(node);

    node->start_index = start_index;
    node->length = length;
    node->next = NULL;
    node->prev = NULL;
    return node;
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

    pool->pool_size = pool_size;
    pool->free_list = create_free_node(0, pool_size);

    return pool;
}

hvr_sparse_vec_t *hvr_alloc_sparse_vecs(unsigned nvecs,
        hvr_sparse_vec_pool_t *pool) {
    // Greedily find the first free node large enough to satisfy this request
    hvr_sparse_vec_pool_free_node_t *curr = pool->free_list;
    size_t nfree = 0;
    while (curr && curr->length < nvecs) {
        nfree += curr->length;
        curr = curr->next;
    }

    if (curr == NULL) {
        fprintf(stderr, "HOOVER> ERROR Ran out of sparse vectors in the pool "
                "on PE %d. # free = %lu, allocating %lu\n", shmem_my_pe(),
                nfree, nvecs);
        abort();
    }

    unsigned alloc_start_index = curr->start_index;
    curr->start_index += nvecs;
    curr->length -= nvecs;
    if (curr->length == 0) {
        // Remove this node from the list
        if (curr->prev == NULL && curr->next == NULL) {
            // Only element in the list
            assert(pool->free_list == curr);
            pool->free_list = NULL;
        } else if (curr->prev == NULL) {
            // First element in a list of at least 2 elements
            assert(pool->free_list == curr);
            pool->free_list = curr->next;
            curr->next->prev = NULL;
        } else if (curr->next == NULL) {
            // Last element in a list of at least 2 elements
            curr->prev->next = NULL;
        } else {
            // Interior element
            curr->prev->next = curr->next;
            curr->next->prev = curr->prev;
        }

        free(curr);
    }

    // Initialize each of the reserved vectors
    hvr_sparse_vec_t *allocated = pool->pool + alloc_start_index;
    for (size_t i = 0; i < nvecs; i++) {
        hvr_sparse_vec_init(&allocated[i]);
    }
    return allocated;
}

void hvr_free_sparse_vecs(hvr_sparse_vec_t *vecs, unsigned nvecs,
        hvr_sparse_vec_pool_t *pool) {
    const size_t pool_index = vecs - pool->pool;
    hvr_sparse_vec_pool_free_node_t *prev = NULL;
    hvr_sparse_vec_pool_free_node_t *next = pool->free_list;
    while (next && next->start_index < pool_index) {
        prev = next;
        next = next->next;
    }

    if (prev == NULL && next == NULL) {
        // Empty free list, insert this as a new node
        hvr_sparse_vec_pool_free_node_t *node = create_free_node(pool_index,
                nvecs);
        pool->free_list = node;
    } else if (prev == NULL) {
        // Insert at front of pool free list
        assert(next == pool->free_list);

        if (pool_index + nvecs == next->start_index) {
            // Merge into first node
            next->start_index -= nvecs;
            next->length += nvecs;
        } else {
            // Create new node and prepend it
            hvr_sparse_vec_pool_free_node_t *node = create_free_node(
                    pool_index, nvecs);
            node->next = next;
            next->prev = node;
            pool->free_list = node;
        }
    } else if (next == NULL) {
        // Insert at end of pool free list. prev is the last node in the list
        if (prev->start_index + prev->length == pool_index) {
            // Merge into last node
            prev->length += nvecs;
        }  else {
            hvr_sparse_vec_pool_free_node_t *node = create_free_node(
                    pool_index, nvecs);
            prev->next = node;
            node->prev = prev;
        }
    } else {
        // Insert between prev and next
        if (prev->start_index + prev->length == pool_index &&
                pool_index + nvecs == next->start_index) {
            // Merge all three ranges together
            prev->length += (nvecs + next->length);
            prev->next = next->next;
            if (next->next) next->next->prev = prev;
            free(next);
        } else if (prev->start_index + prev->length == pool_index) {
            // Just merge with prev
            prev->length += nvecs;
        } else if (pool_index + nvecs == next->start_index) {
            // Just merge with next
            next->start_index -= nvecs;
            next->length += nvecs;
        } else {
            // No merge, just create a new node
            hvr_sparse_vec_pool_free_node_t *node = create_free_node(
                    pool_index, nvecs);
            prev->next = node;
            next->prev = node;
            node->next = next;
            node->prev = prev;
        }
    }
}
