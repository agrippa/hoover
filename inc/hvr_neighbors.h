#ifndef _HVR_NEIGHBORS_H
#define _HVR_NEIGHBORS_H

#include <assert.h>
#include "hvr_vertex_cache.h"

typedef struct _hvr_neighbors_t {
    hvr_edge_info_t *l;
    unsigned l_len;
    unsigned iter;
    hvr_vertex_cache_t *cache;
} hvr_neighbors_t;

static inline void hvr_neighbors_seek_to_valid(hvr_neighbors_t *n) {
    while (n->iter < n->l_len) {
        hvr_vertex_cache_node_t *cached_neighbor = CACHE_NODE_BY_OFFSET(
                EDGE_INFO_VERTEX(n->l[n->iter]), n->cache);
        if (cached_neighbor->populated) {
            break;
        }
        n->iter += 1;
    }
}

static inline void hvr_neighbors_init(hvr_edge_info_t *l, unsigned l_len,
        hvr_vertex_cache_t *cache, hvr_neighbors_t *n) {
    n->l = l;
    n->l_len = l_len;
    n->iter = 0;
    n->cache = cache;
    hvr_neighbors_seek_to_valid(n);
}

static inline void hvr_neighbors_next(hvr_neighbors_t *n,
        hvr_vertex_t **out_neighbor, hvr_edge_type_t *out_type) {
    if (n->iter >= n->l_len) {
        *out_neighbor = NULL;
        return;
    }
    hvr_vertex_cache_node_t *cached_neighbor = CACHE_NODE_BY_OFFSET(
            EDGE_INFO_VERTEX(n->l[n->iter]), n->cache);
    assert(cached_neighbor->populated);
    *out_neighbor = &cached_neighbor->vert;
    *out_type = EDGE_INFO_EDGE(n->l[n->iter]);
    n->iter += 1;
    hvr_neighbors_seek_to_valid(n);
}

#endif // _HVR_NEIGHBORS_H
