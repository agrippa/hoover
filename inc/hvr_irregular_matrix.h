#ifndef _HVR_IRREGULAR_MATRIX_H
#define _HVR_IRREGULAR_MATRIX_H

#include "hvr_common.h"
#include "dlmalloc.h"

typedef struct _hvr_irr_matrix_t {
    hvr_edge_info_t **edges;
    unsigned *edges_capacity;
    unsigned *edges_len;
    size_t nvertices;

    void *pool;
    size_t pool_size;
    mspace allocator;
} hvr_irr_matrix_t;

void hvr_irr_matrix_init(size_t nvertices, size_t pool_size,
        hvr_irr_matrix_t *m);

hvr_edge_type_t hvr_irr_matrix_get(hvr_vertex_id_t i, hvr_vertex_id_t j,
        hvr_irr_matrix_t *m);

void hvr_irr_matrix_set(hvr_vertex_id_t i, hvr_vertex_id_t j, hvr_edge_type_t e,
        hvr_irr_matrix_t *m);

void hvr_irr_matrix_linearize(hvr_vertex_id_t i, hvr_vertex_id_t *out_vals,
        hvr_edge_type_t *out_edges, size_t *out_len, size_t capacity,
        hvr_irr_matrix_t *m);

void hvr_irr_matrix_usage(size_t *bytes_used, size_t *bytes_capacity,
        size_t *bytes_allocated, size_t *out_max_edges,
        size_t *out_max_edges_index, hvr_irr_matrix_t *m);

#endif
