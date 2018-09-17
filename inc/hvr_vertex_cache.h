#ifndef _HVR_VERTEX_CACHE_H
#define _HVR_VERTEX_CACHE_H

#include "hvr_vertex.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HVR_CACHE_BUCKETS 128
#define CACHE_BUCKET(vert_id) ((vert_id) % HVR_CACHE_BUCKETS)

/*
 * A cache data structure used locally on each PE to store remote vertices that
 * have already been fetched. This data structure stores the remote vertex
 * itself.
 */
typedef struct _hvr_vertex_cache_node_t {
    // Contents of the vec itself
    hvr_vertex_t vert;
    /*
     * Number of edges that separate this vertex from closest vertex stored
     * locally on the current PE. Is zero for local vertices.
     */
    unsigned min_dist_from_local_vertex;

    /*
     * If this node is allocated, pointers to the next and previous cache node
     * in the same bucket.
     *
     * If this node is not allocated and is sitting in the free pool, pointers
     * to the next/prev node in the pool of free nodes.
     */
    struct _hvr_vertex_cache_node_t *next;
    struct _hvr_vertex_cache_node_t *prev;

    /*
     * For allocated nodes, construct a doubly-linked list with a total ordering
     * determined by the order in which a node was created (such that the least
     * recently allocated is at the tail).
     */
    struct _hvr_vertex_cache_node_t *lru_prev;
    struct _hvr_vertex_cache_node_t *lru_next;
} hvr_vertex_cache_node_t;

/*
 * Per-remote PE data structure used to store all fetched and cached vertices
 * from that remote PE. Vertices are hashed by their offset in the local
 * vertices of the remote PE and stored in separate buckets.
 */
typedef struct _hvr_vertex_cache_t {
    // Lists of vertices by vertex hash
    hvr_vertex_cache_node_t *buckets[HVR_CACHE_BUCKETS];

    /*
     * Pool of pre-allocated but unused vertex data structures. Used
     * to reduce system memory management calls and ensure we stay within a
     * fixed memory footprint.
     */
    hvr_vertex_cache_node_t *pool_head;
    unsigned pool_size;

    /*
     * LRU list of allocated vertex data structures, used to evict when we run
     * out of free nodes in the free pool.
     */
    hvr_vertex_cache_node_t *lru_head;
    hvr_vertex_cache_node_t *lru_tail;

    struct {
        unsigned long long quiet_counter;
        unsigned long long fetch_neighbors_time;
        unsigned long long quiet_neighbors_time;
        unsigned long long update_metadata_time;
        unsigned long long nhits;
        unsigned long long nmisses;
    } cache_perf_info;
} hvr_vertex_cache_t;

/*
 * Initializes an already allocated block of memory to store a vertex cache.
 * cache is assumed to point to a block of memory of at least
 * sizeof(hvr_vertex_cache_t) bytes.
 */
void hvr_vertex_cache_init(hvr_vertex_cache_t *cache);

/*
 * Given a vertex ID on a remote PE, look up that vertex in our local cache.
 * Only returns an entry if the newest timestep stored in that entry is new
 * enough given target_timestep. If no matching entry is found, returns NULL.
 *
 * May lead to evictions of very old cache entries that we now consider unusable
 * because of their age, as judged by CACHED_TIMESTEPS_TOLERANCE.
 */
hvr_vertex_cache_node_t *hvr_vertex_cache_lookup(hvr_vertex_id_t vert,
        hvr_vertex_cache_t *cache);

hvr_vertex_cache_node_t *hvr_vertex_cache_add(hvr_vertex_t *vert,
        unsigned min_dist_from_local_vertex, hvr_vertex_cache_t *cache);

#ifdef __cplusplus
}
#endif

#endif // _HVR_VERTEX_CACHE_H
