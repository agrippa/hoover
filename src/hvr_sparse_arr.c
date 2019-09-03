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

    arr->segs = (hvr_sparse_arr_seg_t **)malloc_helper(nsegs * sizeof(*(arr->segs)));
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
    arr->pool = malloc_helper(pool_size);
    assert(arr->pool);
    memset(arr->pool, 0xff, pool_size);
    arr->tracker = create_mspace_with_base(arr->pool, pool_size, 0);
    assert(arr->tracker);
}

void hvr_sparse_arr_destroy(hvr_sparse_arr_t *arr) {
    free(arr->segs);
    free(arr->preallocated);
    destroy_mspace(arr->tracker);
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

    struct node *exists = hvr_avl_find(segment->seg[seg_index], j);
    if (exists == nnil) {
        hvr_avl_insert(&(segment->seg[seg_index]), j, arr->tracker);
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

    struct node *exists = hvr_avl_find(segment->seg[seg_index], j);
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

    int success = hvr_avl_delete(&(segment->seg[seg_index]), j, arr->tracker);
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

    hvr_avl_delete_all(segment->seg[seg_index], arr->tracker);
    segment->seg[seg_index] = NULL;
    segment->seg_size[seg_index] = 0;
}

unsigned hvr_sparse_arr_linearize_row(unsigned i, int **out_arr,
        hvr_sparse_arr_t *arr) {
    assert(i < arr->capacity);

    static int *cache = NULL;
    static int cache_size = 0;

    const unsigned seg = i / HVR_SPARSE_ARR_SEGMENT_SIZE;
    const unsigned seg_index = i % HVR_SPARSE_ARR_SEGMENT_SIZE;

    hvr_sparse_arr_seg_t *segment = arr->segs[seg];
    if (segment == NULL) {
        return 0;
    }

    int n_stored_values = segment->seg_size[seg_index];
    if (cache_size < n_stored_values) {
        cache = (int *)mspace_realloc(arr->tracker,
                cache,
                n_stored_values * sizeof(*cache));
        assert(cache);
        cache_size = n_stored_values;
    }

    hvr_avl_serialize(segment->seg[seg_index], cache, cache_size);

    *out_arr = cache;
    return n_stored_values;
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
