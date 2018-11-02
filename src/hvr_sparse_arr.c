#include "hvr_sparse_arr.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

static void hvr_sparse_arr_seg_init(hvr_sparse_arr_seg_t *seg) {
    memset(seg, 0x00, sizeof(*seg));
}

void hvr_sparse_arr_init(hvr_sparse_arr_t *arr, unsigned capacity) {
    unsigned nsegs = (capacity + HVR_SPARSE_ARR_SEGMENT_SIZE - 1) /
        HVR_SPARSE_ARR_SEGMENT_SIZE;

    arr->segs = (hvr_sparse_arr_seg_t **)malloc(nsegs * sizeof(*(arr->segs)));
    assert(arr->segs);
    memset(arr->segs, 0x00, nsegs * sizeof(*(arr->segs)));

    arr->capacity = capacity;
    arr->nsegs = nsegs;
}

void hvr_sparse_arr_insert(unsigned i, unsigned j, hvr_sparse_arr_t *arr) {
    assert(i < arr->capacity);

    const unsigned seg = i / HVR_SPARSE_ARR_SEGMENT_SIZE;
    const unsigned seg_index = i % HVR_SPARSE_ARR_SEGMENT_SIZE;

    if (arr->segs[seg] == NULL) {
        // No segment allocated yet
        arr->segs[seg] = (hvr_sparse_arr_seg_t *)malloc(
                sizeof(hvr_sparse_arr_seg_t));
        assert(arr->segs[seg]);
        hvr_sparse_arr_seg_init(arr->segs[seg]);
    }

    hvr_sparse_arr_seg_t *segment = arr->segs[seg];

    // Check if this value is already stored
    int *stored_values = segment->seg[seg_index];
    for (unsigned index = 0; index < segment->seg_lengths[seg_index]; index++) {
        if (stored_values[index] == j) {
            return;
        }
    }

    if ((segment->seg_lengths)[seg_index] == 0) {
        // First initialization
        const unsigned initial_capacity = 16;
        segment->seg_capacities[seg_index] = initial_capacity;
        segment->seg[seg_index] = (int *)malloc(initial_capacity * sizeof(int));
        assert(segment->seg[seg_index]);
    } else if ((segment->seg_lengths)[seg_index] ==
            (segment->seg_capacities)[seg_index]) {
        // No more space left
        (segment->seg_capacities)[seg_index] *= 2;
        (segment->seg)[seg_index] = (int *)realloc(segment->seg[seg_index],
                (segment->seg_capacities)[seg_index] * sizeof(int));
        assert((segment->seg)[seg_index]);
    }

    stored_values = segment->seg[seg_index];
    stored_values[(segment->seg_lengths)[seg_index]] = j;
    (segment->seg_lengths)[seg_index] += 1;
}

int hvr_sparse_arr_contains(unsigned i, unsigned j, hvr_sparse_arr_t *arr) {
    assert(i < arr->capacity);

    const unsigned seg = i / HVR_SPARSE_ARR_SEGMENT_SIZE;
    const unsigned seg_index = i % HVR_SPARSE_ARR_SEGMENT_SIZE;

    hvr_sparse_arr_seg_t *segment = arr->segs[seg];
    if (segment == NULL) {
        return 0;
    }

    int *stored_values = segment->seg[seg_index];
    int n_stored_values = segment->seg_lengths[seg_index];

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
        unsigned *capacity, hvr_sparse_arr_t *arr) {
    assert(i < arr->capacity);

    const unsigned seg = i / HVR_SPARSE_ARR_SEGMENT_SIZE;
    const unsigned seg_index = i % HVR_SPARSE_ARR_SEGMENT_SIZE;

    hvr_sparse_arr_seg_t *segment = arr->segs[seg];
    if (segment == NULL) {
        return 0;
    }

    int *stored_values = segment->seg[seg_index];
    int n_stored_values = segment->seg_lengths[seg_index];

    if (*capacity < n_stored_values) {
        *capacity = n_stored_values;
        *out_arr = (int *)realloc(*out_arr, *capacity * sizeof(**out_arr));
        assert(*out_arr);
    }
    memcpy(*out_arr, stored_values, n_stored_values * sizeof(**out_arr));
    return n_stored_values;
}
