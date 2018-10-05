#ifndef _HVR_SPARSE_VEC_POOL_H
#define _HVR_SPARSE_VEC_POOL_H

#include "hvr_vertex.h"
#include "hvr_common.h"

typedef struct _hvr_vertex_range_node_t {
    unsigned start_index;
    unsigned length;

    struct _hvr_vertex_range_node_t *next;
    struct _hvr_vertex_range_node_t *prev;
} hvr_vertex_range_node_t;

typedef struct _hvr_vertex_pool_t {
    hvr_vertex_t *pool;
    size_t pool_size;
    size_t nallocated;

    hvr_vertex_range_node_t *free_list;
    hvr_vertex_range_node_t *used_list;
} hvr_vertex_pool_t;

/*
 * Allocate a fixed size pool of 'pool_size' sparse vectors from which we can
 * dynamically allocate (and free) HOOVER sparse vectors.
 */
extern hvr_vertex_pool_t *hvr_vertex_pool_create(size_t pool_size);

/*
 * Allocate 'nvecs' sparse vectors from the specified memory pool.
 */
extern hvr_vertex_t *hvr_alloc_vertices(unsigned nvecs, hvr_ctx_t ctx);

/*
 * Release 'nvecs' sparse vectors starting at memory address vecs in the
 * specified memory pool.
 */
extern void hvr_free_vertices(hvr_vertex_t *vecs, unsigned nvecs,
        hvr_ctx_t ctx);

/*
 * Get the number of vertices that are allocated from this context's vertex
 * pool.
 */
extern size_t hvr_n_allocated(hvr_ctx_t ctx);

#endif // _HVR_SPARSE_VEC_POOL_H
