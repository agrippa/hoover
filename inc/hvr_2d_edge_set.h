#ifndef _HVR_2D_EDGE_SET_H
#define _HVR_2D_EDGE_SET_H

#include "hvr_common.h"

/*
 * A data structure for storing sparse adjacency matrices, requiring 2-bits per
 * edge to store the type of edge.
 */

#define TILE_DIM 512
#define BITS_PER_EDGE 2
#define BITS_PER_BYTE 8
#define EDGES_PER_BYTE (BITS_PER_BYTE / BITS_PER_EDGE)
typedef uint64_t hvr_tile_ele_t;

#define TILE_SIZE_IN_BITS (TILE_DIM * TILE_DIM * BITS_PER_EDGE)
#define TILE_SIZE_IN_BYTES ((TILE_SIZE_IN_BITS + BITS_PER_BYTE - 1) / BITS_PER_BYTE)
#define TILE_SIZE_IN_ELES ((TILE_SIZE_IN_BYTES + sizeof(hvr_tile_ele_t) - 1) / sizeof(hvr_tile_ele_t))

typedef struct _hvr_2d_edge_set_tile_t {
    hvr_tile_ele_t tile[TILE_SIZE_IN_ELES];
    struct _hvr_2d_edge_set_tile_t *next;
} hvr_2d_edge_set_tile_t;

typedef struct _hvr_2d_edge_set_t {
    hvr_2d_edge_set_tile_t **tiles;
    hvr_2d_edge_set_tile_t *preallocated;
    uint64_t ntiles_per_dim;
    uint64_t dim;
} hvr_2d_edge_set_t;

void hvr_2d_edge_set_init(hvr_2d_edge_set_t *s, uint64_t dim,
        uint64_t max_n_tiles);

hvr_edge_type_t hvr_2d_get(uint64_t i, uint64_t j, hvr_2d_edge_set_t *s);

void hvr_2d_set(uint64_t i, uint64_t j, hvr_edge_type_t e,
        hvr_2d_edge_set_t *s);

void hvr_2d_linearize(uint64_t i, uint64_t *out_vals,
        hvr_edge_type_t *out_edges, size_t *out_len, size_t capacity,
        hvr_2d_edge_set_t *s);

#endif // _HVR_2D_EDGE_SET_H
