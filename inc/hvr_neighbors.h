#ifndef _HVR_NEIGHBORS_H
#define _HVR_NEIGHBORS_H

#include <assert.h>
#ifdef __cplusplus
extern "C" {
#endif
#include "dlmalloc.h"
#ifdef __cplusplus
}
#endif
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

static inline void hvr_neighbors_init(hvr_vertex_id_t *l, unsigned l_len,
        hvr_vertex_cache_t *cache, mspace tracker, size_t pools_size,
        hvr_neighbors_t *n) {
    n->l = (hvr_edge_info_t *)mspace_malloc(tracker, l_len * sizeof(n->l[0]));
    if (n->l == NULL) {
        fprintf(stderr, "ERROR failed allocating neighbor list of %lu bytes, "
                "increase HVR_NEIGHBORS_LIST_POOL_SIZE (currently %lu)\n",
                l_len * sizeof(n->l[0]), pools_size);
        abort();
    }
    memcpy(n->l, l, l_len * sizeof(*l));

    n->l_len = l_len;
    n->iter = 0;
    n->cache = cache;
    hvr_neighbors_seek_to_valid(n);
}

static inline void hvr_neighbors_reset(hvr_neighbors_t *n) {
    n->iter = 0;
    hvr_neighbors_seek_to_valid(n);
}

static inline void hvr_neighbors_destroy(hvr_neighbors_t *n, mspace tracker) {
    mspace_free(tracker, n->l);
}

static inline int hvr_neighbors_next(hvr_neighbors_t *n,
        hvr_vertex_t **out_neighbor, hvr_edge_type_t *out_type) {
    if (n->iter >= n->l_len) {
        *out_neighbor = NULL;
        return 0;
    }
    hvr_vertex_cache_node_t *cached_neighbor = CACHE_NODE_BY_OFFSET(
            EDGE_INFO_VERTEX(n->l[n->iter]), n->cache);
    assert(cached_neighbor->populated);
    *out_neighbor = &cached_neighbor->vert;
    *out_type = EDGE_INFO_EDGE(n->l[n->iter]);
    n->iter += 1;
    hvr_neighbors_seek_to_valid(n);
    return 1;
}

#endif // _HVR_NEIGHBORS_H
