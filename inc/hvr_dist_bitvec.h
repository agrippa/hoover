#ifndef _HVR_DISTRIBUTED_BITVECTOR
#define _HVR_DISTRIBUTED_BITVECTOR

#include "dlmalloc.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * This bitvector is distributed across PEs. Always the first dimension is used
 * for the distribution, with only contiguous lower dimensions stored on each
 * PE. For example, if you were to allocate a 3D bitvector of dimensions
 * (4, 4, 4) and distribute it across 2 PEs then elements (0:1, :, :) would be
 * stored on PE 0 and elements (2:3, :, :) would be stored on PE 1. It is an
 * error to specify a first dimension that is less than the number of PEs.
 */

typedef uint64_t hvr_dist_bitvec_size_t;
typedef uint64_t hvr_dist_bitvec_ele_t;
typedef struct _hvr_dist_bitvec_t {
    // Outermost dimension of the distributed vector
    hvr_dist_bitvec_size_t dim0;
    // Innermost dimension of the distributed vector
    hvr_dist_bitvec_size_t dim1;
    // Symmetrically allocated local portion of the vector
    hvr_dist_bitvec_ele_t *symm_vec;

    uint64_t *seq_nos;

    /*
     * Length of the inner dimension in units of sizeof(hvr_dist_bitvec_ele_t),
     * rounded up.
     */
    hvr_dist_bitvec_size_t dim1_length_in_words;
    hvr_dist_bitvec_size_t dim0_per_pe;

    void *pool;
    mspace tracker;
} hvr_dist_bitvec_t;

// A local (not symmetric) copy of the values for a single row of the bitvector.
typedef struct _hvr_dist_bitvec_local_subcopy_t {
    // The row coordinate of this local copy
    hvr_dist_bitvec_size_t coord0;
    // The length in elements of this copy
    hvr_dist_bitvec_size_t dim1;
    // The non-symmetric backing data
    hvr_dist_bitvec_ele_t *subvec;
    hvr_dist_bitvec_size_t dim1_length_in_words;

    uint64_t seq_no;
} hvr_dist_bitvec_local_subcopy_t;

void hvr_dist_bitvec_init(hvr_dist_bitvec_size_t dim0,
        hvr_dist_bitvec_size_t dim1, hvr_dist_bitvec_t *vec);

void hvr_dist_bitvec_set(hvr_dist_bitvec_size_t coord0,
        hvr_dist_bitvec_size_t coord1, hvr_dist_bitvec_t *vec);

void hvr_dist_bitvec_clear(hvr_dist_bitvec_size_t coord0,
        hvr_dist_bitvec_size_t coord1, hvr_dist_bitvec_t *vec);

int hvr_dist_bitvec_owning_pe(hvr_dist_bitvec_size_t coord0,
        hvr_dist_bitvec_t *vec);

void hvr_dist_bitvec_my_chunk(hvr_dist_bitvec_size_t *lower,
        hvr_dist_bitvec_size_t *upper, hvr_dist_bitvec_t *vec);

void hvr_dist_bitvec_local_subcopy_init(hvr_dist_bitvec_t *vec,
        hvr_dist_bitvec_local_subcopy_t *out);

void hvr_dist_bitvec_copy_locally(hvr_dist_bitvec_size_t coord0,
        hvr_dist_bitvec_t *vec, hvr_dist_bitvec_local_subcopy_t *out);

int hvr_dist_bitvec_local_subcopy_contains(hvr_dist_bitvec_size_t coord1,
        hvr_dist_bitvec_local_subcopy_t *vec);

void hvr_dist_bitvec_local_subcopy_copy(
        hvr_dist_bitvec_local_subcopy_t *dst,
        hvr_dist_bitvec_local_subcopy_t *src);

void hvr_dist_bitvec_local_subcopy_destroy(hvr_dist_bitvec_t *vec,
        hvr_dist_bitvec_local_subcopy_t *c);

size_t hvr_dist_bitvec_local_subcopy_bytes(
        hvr_dist_bitvec_local_subcopy_t *vec);

uint64_t hvr_dist_bitvec_get_seq_no(hvr_dist_bitvec_size_t coord0,
        hvr_dist_bitvec_t *vec);

#ifdef __cplusplus
}
#endif

#endif // _HVR_DISTRIBUTED_BITVECTOR
