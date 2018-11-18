#ifndef _HVR_SPARSE_ARR_H
#define _HVR_SPARSE_ARR_H

#include <stdlib.h>

#define HVR_SPARSE_ARR_SEGMENT_SIZE 1024

typedef struct _hvr_sparse_arr_seg_t {
    int *seg[HVR_SPARSE_ARR_SEGMENT_SIZE];
    unsigned seg_lengths[HVR_SPARSE_ARR_SEGMENT_SIZE];
    unsigned seg_capacities[HVR_SPARSE_ARR_SEGMENT_SIZE];
} hvr_sparse_arr_seg_t;

typedef struct _hvr_sparse_arr_t {
    hvr_sparse_arr_seg_t **segs;
    unsigned capacity;
    unsigned nsegs;
} hvr_sparse_arr_t;

extern void hvr_sparse_arr_init(hvr_sparse_arr_t *arr, unsigned capacity);

extern void hvr_sparse_arr_insert(unsigned i, unsigned j,
        hvr_sparse_arr_t *arr);

extern int hvr_sparse_arr_contains(unsigned i, unsigned j,
        hvr_sparse_arr_t *arr);

extern void hvr_sparse_arr_remove(unsigned i, unsigned j,
        hvr_sparse_arr_t *arr);

extern unsigned hvr_sparse_arr_linearize_row(unsigned i, int **out_arr,
        unsigned *capacity, hvr_sparse_arr_t *arr);

extern unsigned hvr_sparse_arr_row_length(unsigned i, hvr_sparse_arr_t *arr);

extern size_t hvr_sparse_arr_used_bytes(hvr_sparse_arr_t *arr);

#endif
