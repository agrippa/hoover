#ifndef _HVR_SPARSE_VEC_POOL_H
#define _HVR_SPARSE_VEC_POOL_H

#include "hvr_vertex.h"
#include "hvr_common.h"

typedef struct _hvr_range_node_t {
    unsigned start_index;
    unsigned length;

    struct _hvr_range_node_t *next;
    struct _hvr_range_node_t *prev;
} hvr_range_node_t;

typedef struct _hvr_range_tracker_t {
    hvr_range_node_t *free_list;
    hvr_range_node_t *used_list;
    hvr_range_node_t *preallocated_nodes;
    size_t capacity;
    size_t used;
    int n_nodes;
} hvr_range_tracker_t;

typedef struct _hvr_vertex_pool_t {
    hvr_vertex_t *pool;
    hvr_range_tracker_t tracker;
} hvr_vertex_pool_t;

/*
 * Allocate a fixed size pool of 'pool_size' sparse vectors from which we can
 * dynamically allocate (and free) HOOVER sparse vectors.
 */
extern void hvr_vertex_pool_create(size_t pool_size, size_t n_nodes,
        hvr_vertex_pool_t *pool);

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

extern size_t hvr_pool_size_in_bytes(hvr_ctx_t in_ctx);


extern void hvr_range_tracker_init(size_t capacity, int n_nodes,
        hvr_range_tracker_t *tracker);
extern size_t hvr_range_tracker_reserve(size_t space,
        hvr_range_tracker_t *tracker);
extern void hvr_range_tracker_release(size_t offset, size_t space,
        hvr_range_tracker_t *tracker);
extern size_t hvr_range_tracker_size_in_bytes(hvr_range_tracker_t *tracker);

#endif // _HVR_SPARSE_VEC_POOL_H
