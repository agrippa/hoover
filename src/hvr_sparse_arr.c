#include "hvr_sparse_arr.h"
#include "hvr_common.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

static inline void hvr_sparse_arr_seg_init(hvr_sparse_arr_seg_t *seg) {
    memset(seg, 0x00, sizeof(*seg));
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

    // Check if this value is already stored
    int *stored_values = segment->seg[seg_index];
    const unsigned seg_len = segment->seg_lengths[seg_index];
    for (unsigned index = 0; index < seg_len; index++) {
        if (stored_values[index] == j) {
            return;
        }
    }

    if (seg_len == 0) {
        // First initialization
        const unsigned initial_capacity = 16;
        segment->seg_capacities[seg_index] = initial_capacity;
        segment->seg[seg_index] = (int *)mspace_malloc(arr->tracker,
                initial_capacity * sizeof(int));
        assert(segment->seg[seg_index]);
    } else if (seg_len == (segment->seg_capacities)[seg_index]) {
        // No more space left
        segment->seg_capacities[seg_index] *= 2;
        segment->seg[seg_index] = (int *)mspace_realloc(arr->tracker,
                segment->seg[seg_index],
                (segment->seg_capacities)[seg_index] * sizeof(int));
        assert(segment->seg[seg_index]);
    }

    (segment->seg[seg_index])[seg_len] = j;
    (segment->seg_lengths)[seg_index] = seg_len + 1;
}

int hvr_sparse_arr_contains(unsigned i, unsigned j, hvr_sparse_arr_t *arr) {
    assert(i < arr->capacity);

    const unsigned seg = i / HVR_SPARSE_ARR_SEGMENT_SIZE;
    const unsigned seg_index = i % HVR_SPARSE_ARR_SEGMENT_SIZE;

    hvr_sparse_arr_seg_t *segment = arr->segs[seg];
    if (segment == NULL) {
        return 0;
    }

    const int *stored_values = segment->seg[seg_index];
    const int n_stored_values = segment->seg_lengths[seg_index];

    for (int index = 0; index < n_stored_values; index++) {
        if (stored_values[index] == j) {
            return 1;
        }
    }

    return 0;
}

void hvr_sparse_arr_remove(unsigned i, unsigned j, hvr_sparse_arr_t *arr) {
    assert(i < arr->capacity);

    const unsigned seg = i / HVR_SPARSE_ARR_SEGMENT_SIZE;
    const unsigned seg_index = i % HVR_SPARSE_ARR_SEGMENT_SIZE;

    hvr_sparse_arr_seg_t *segment = arr->segs[seg];
    if (segment == NULL) {
        return;
    }

    int *stored_values = segment->seg[seg_index];
    int n_stored_values = segment->seg_lengths[seg_index];

    for (int index = 0; index < n_stored_values; index++) {
        if (stored_values[index] == j) {
            stored_values[index] = stored_values[n_stored_values - 1];
            segment->seg_lengths[seg_index] -= 1;
            return;
        }
    }
}

void hvr_sparse_arr_remove_row(unsigned i, hvr_sparse_arr_t *arr) {
    const unsigned seg = i / HVR_SPARSE_ARR_SEGMENT_SIZE;
    const unsigned seg_index = i % HVR_SPARSE_ARR_SEGMENT_SIZE;

    hvr_sparse_arr_seg_t *segment = arr->segs[seg];
    if (segment == NULL) {
        return;
    }

    int *stored_values = segment->seg[seg_index];
    if (stored_values) {
        mspace_free(arr->tracker, stored_values);
        segment->seg[seg_index] = NULL;
        segment->seg_lengths[seg_index] = 0;
    }
}

unsigned hvr_sparse_arr_row_length(unsigned i, hvr_sparse_arr_t *arr) {
    assert(i < arr->capacity);

    const unsigned seg = i / HVR_SPARSE_ARR_SEGMENT_SIZE;
    const unsigned seg_index = i % HVR_SPARSE_ARR_SEGMENT_SIZE;

    hvr_sparse_arr_seg_t *segment = arr->segs[seg];
    if (segment == NULL) {
        return 0;
    }
    return segment->seg_lengths[seg_index];
}

unsigned hvr_sparse_arr_linearize_row(unsigned i, int **out_arr,
        hvr_sparse_arr_t *arr) {
    assert(i < arr->capacity);

    const unsigned seg = i / HVR_SPARSE_ARR_SEGMENT_SIZE;
    const unsigned seg_index = i % HVR_SPARSE_ARR_SEGMENT_SIZE;

    hvr_sparse_arr_seg_t *segment = arr->segs[seg];
    if (segment == NULL) {
        return 0;
    }

    int *stored_values = segment->seg[seg_index];
    int n_stored_values = segment->seg_lengths[seg_index];

    *out_arr = stored_values;
    return n_stored_values;
}

size_t hvr_sparse_arr_used_bytes(hvr_sparse_arr_t *arr) {
    size_t nbytes = (arr->nsegs * sizeof(*(arr->segs))); // arr->segs
    for (unsigned s = 0; s < arr->nsegs; s++) {
        hvr_sparse_arr_seg_t *seg = arr->segs[s];
        if (seg) {
            for (unsigned i = 0; i < HVR_SPARSE_ARR_SEGMENT_SIZE; i++) {
                nbytes += seg->seg_capacities[i] * sizeof(seg->seg[0][0]);
            }
        }
    }
    return nbytes;
}
