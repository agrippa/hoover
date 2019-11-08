#include "hvr_sparse_arr.h"
#include "hvr_common.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

static inline void hvr_sparse_arr_seg_init(hvr_sparse_arr_seg_t *seg) {
    for (int i = 0; i < HVR_SPARSE_ARR_SEGMENT_SIZE; i++) {
        seg->seg[i] = nnil;
        seg->seg_size[i] = 0;
    }
    seg->next = NULL;
}

void hvr_sparse_arr_init(hvr_sparse_arr_t *arr, unsigned capacity) {
    unsigned nsegs = (capacity + HVR_SPARSE_ARR_SEGMENT_SIZE - 1) /
        HVR_SPARSE_ARR_SEGMENT_SIZE;

    arr->segs = (hvr_sparse_arr_seg_t **)malloc_helper(
            nsegs * sizeof(*(arr->segs)));
    assert(arr->segs);
    memset(arr->segs, 0x00, nsegs * sizeof(*(arr->segs)));

    arr->capacity = capacity;
    arr->nsegs = nsegs;

    int prealloc = 1024;
    if (getenv("HVR_SPARSE_ARR_SEGS")) {
        prealloc = atoi(getenv("HVR_SPARSE_ARR_SEGS"));
    }
    arr->preallocated = (hvr_sparse_arr_seg_t *)malloc_helper(
            prealloc * sizeof(arr->preallocated[0]));
    assert(arr->preallocated);
    for (unsigned i = 0; i < prealloc - 1; i++) {
        arr->preallocated[i].next = arr->preallocated + (i + 1);
    }
    arr->preallocated[prealloc - 1].next = NULL;
    arr->segs_pool = arr->preallocated;

    int pool_size = 1024 * 1024;
    if (getenv("HVR_SPARSE_ARR_POOL")) {
        pool_size = atoi(getenv("HVR_SPARSE_ARR_POOL"));
    }
    hvr_avl_node_allocator_init(&arr->avl_allocator, pool_size,
            "HVR_SPARSE_ARR_POOL");

    pool_size = 1024 * 1024;
    if (getenv("HVR_SPARSE_ARR_BUF_POOL")) {
        pool_size = atoi(getenv("HVR_SPARSE_ARR_BUF_POOL"));
    }
    arr->pool = malloc_helper(pool_size);
    assert(arr->pool);
    arr->allocator = create_mspace_with_base(arr->pool, pool_size, 0);
    assert(arr->allocator);
}

void hvr_sparse_arr_destroy(hvr_sparse_arr_t *arr) {
    free(arr->segs);
    free(arr->preallocated);
    destroy_mspace(arr->allocator);
    free(arr->pool);
}

void hvr_sparse_arr_insert(unsigned i, unsigned j, hvr_sparse_arr_t *arr) {
    assert(i < arr->capacity);

    const unsigned seg = i / HVR_SPARSE_ARR_SEGMENT_SIZE;
    const unsigned seg_index = i % HVR_SPARSE_ARR_SEGMENT_SIZE;

    if (arr->segs[seg] == NULL) {
        // No segment allocated yet
        arr->segs[seg] = arr->segs_pool;
        assert(arr->segs[seg]);
        arr->segs_pool = arr->segs_pool->next;

        hvr_sparse_arr_seg_init(arr->segs[seg]);
    }

    hvr_sparse_arr_seg_t *segment = arr->segs[seg];

    struct hvr_avl_node *exists = hvr_avl_find(segment->seg[seg_index], j);
    if (exists == nnil) {
        hvr_avl_insert(&(segment->seg[seg_index]), j, j, &arr->avl_allocator);
        segment->seg_size[seg_index] += 1;
    }
}

int hvr_sparse_arr_contains(unsigned i, unsigned j, hvr_sparse_arr_t *arr) {
    assert(i < arr->capacity);

    const unsigned seg = i / HVR_SPARSE_ARR_SEGMENT_SIZE;
    const unsigned seg_index = i % HVR_SPARSE_ARR_SEGMENT_SIZE;

    hvr_sparse_arr_seg_t *segment = arr->segs[seg];
    if (segment == NULL) {
        return 0;
    }

    struct hvr_avl_node *exists = hvr_avl_find(segment->seg[seg_index], j);
    return (exists != nnil);
}

void hvr_sparse_arr_remove(unsigned i, unsigned j, hvr_sparse_arr_t *arr) {
    assert(i < arr->capacity);

    const unsigned seg = i / HVR_SPARSE_ARR_SEGMENT_SIZE;
    const unsigned seg_index = i % HVR_SPARSE_ARR_SEGMENT_SIZE;

    hvr_sparse_arr_seg_t *segment = arr->segs[seg];
    if (segment == NULL) {
        return;
    }

    int success = hvr_avl_delete(&(segment->seg[seg_index]), j,
            &arr->avl_allocator);
    if (success) {
        segment->seg_size[seg_index] -= 1;
    }
}

void hvr_sparse_arr_remove_row(unsigned i, hvr_sparse_arr_t *arr) {
    const unsigned seg = i / HVR_SPARSE_ARR_SEGMENT_SIZE;
    const unsigned seg_index = i % HVR_SPARSE_ARR_SEGMENT_SIZE;

    hvr_sparse_arr_seg_t *segment = arr->segs[seg];
    if (segment == NULL) {
        return;
    }

    hvr_avl_delete_all(segment->seg[seg_index], &arr->avl_allocator);
    segment->seg[seg_index] = nnil;
    segment->seg_size[seg_index] = 0;
}

unsigned hvr_sparse_arr_linearize_row(unsigned i, uint64_t **out_arr,
        hvr_sparse_arr_t *arr) {
    assert(i < arr->capacity);

    const unsigned seg = i / HVR_SPARSE_ARR_SEGMENT_SIZE;
    const unsigned seg_index = i % HVR_SPARSE_ARR_SEGMENT_SIZE;

    hvr_sparse_arr_seg_t *segment = arr->segs[seg];
    if (segment == NULL) {
        *out_arr = NULL;
        return 0;
    }

    int n_stored_values = segment->seg_size[seg_index];
    if (n_stored_values == 0) {
        *out_arr = NULL;
    } else {
        uint64_t *keys_cache = (uint64_t *)mspace_malloc(arr->allocator,
                n_stored_values * sizeof(*keys_cache));
        assert(keys_cache);

        hvr_avl_serialize(segment->seg[seg_index], keys_cache,
                n_stored_values);

        *out_arr = keys_cache;
    }
    return n_stored_values;
}

void hvr_sparse_arr_release_row(uint64_t *out_arr, hvr_sparse_arr_t *arr) {
    if (out_arr) {
        mspace_free(arr->allocator, out_arr);
    }
}

size_t hvr_sparse_arr_used_bytes(hvr_sparse_arr_t *arr) {
    size_t nbytes = (arr->nsegs * sizeof(*(arr->segs))); // arr->segs
    for (unsigned s = 0; s < arr->nsegs; s++) {
        hvr_sparse_arr_seg_t *seg = arr->segs[s];
        if (seg) {
            for (unsigned i = 0; i < HVR_SPARSE_ARR_SEGMENT_SIZE; i++) {
                nbytes += seg->seg_size[i] * sizeof(seg->seg[0][0]);
            }
        }
    }
    return nbytes;
}
