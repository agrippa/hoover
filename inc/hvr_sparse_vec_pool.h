#ifndef _HVR_SPARSE_VEC_POOL_H
#define _HVR_SPARSE_VEC_POOL_H

#include "hvr_sparse_vec.h"

typedef struct _hvr_sparse_vec_range_node_t {
    unsigned start_index;
    unsigned length;
    struct _hvr_sparse_vec_range_node_t *next;
    struct _hvr_sparse_vec_range_node_t *prev;
} hvr_sparse_vec_range_node_t;

typedef struct _hvr_sparse_vec_pool_t {
    hvr_sparse_vec_t *pool;
    size_t pool_size;

    hvr_sparse_vec_range_node_t *free_list;
    hvr_sparse_vec_range_node_t *used_list;
} hvr_sparse_vec_pool_t;

/*
 * Allocate a fixed size pool of 'pool_size' sparse vectors from which we can
 * dynamically allocate (and free) HOOVER sparse vectors.
 */
hvr_sparse_vec_pool_t *hvr_sparse_vec_pool_create(size_t pool_size);

/*
 * Allocate 'nvecs' sparse vectors from the specified memory pool.
 */
hvr_sparse_vec_t *hvr_alloc_sparse_vecs(unsigned nvecs,
        hvr_ctx_t ctx);

/*
 * Release 'nvecs' sparse vectors starting at memory address vecs in the
 * specified memory pool.
 */
void hvr_free_sparse_vecs(hvr_sparse_vec_t *vecs, unsigned nvecs,
        hvr_ctx_t ctx);

#endif // _HVR_SPARSE_VEC_POOL_H
