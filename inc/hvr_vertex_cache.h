#ifndef _HVR_VERTEX_CACHE_H
#define _HVR_VERTEX_CACHE_H

#include "hvr_vertex.h"
#include "hvr_map.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HVR_CACHE_BUCKETS 8192
#define CACHE_BUCKET(vert_id) ((vert_id) % HVR_CACHE_BUCKETS)

/*
 * A cache data structure used locally on each PE to store remote vertices that
 * have already been fetched. This data structure stores the remote vertex
 * itself.
 */
typedef struct _hvr_vertex_cache_node_t {
    // Contents of the vec itself
    hvr_vertex_t vert;

    // Partition for this vert. Think of this as a secondary key, after vert.id
    hvr_partition_t part;

    /*
     * Maintain lists of mirrored vertices in each partition to enable quick
     * iteration over a subset of mirrored vertices based on partition.
     */
    struct _hvr_vertex_cache_node_t *part_next;
    struct _hvr_vertex_cache_node_t *part_prev;
    struct _hvr_vertex_cache_node_t *tmp;

    struct _hvr_vertex_cache_node_t *local_neighbors_next;
    struct _hvr_vertex_cache_node_t *local_neighbors_prev;
} hvr_vertex_cache_node_t;

/*
 * Data structure used to store all fetched and cached vertices.
 */
typedef struct _hvr_vertex_cache_t {
    /*
     * Array of linked lists, allowing quick iteration over all mirrored
     * vertices in a given partition.
     */
    hvr_vertex_cache_node_t **partitions;
    hvr_partition_t npartitions;

    hvr_vertex_cache_node_t *local_neighbors_head;

    /*
     * A map datastructure used to enable quick lookup of
     * hvr_vertex_cache_node*, given a vertex ID.
     */
    hvr_map_t cache_map;

    /*
     * Pool of pre-allocated but unused vertex data structures. Used
     * to reduce system memory management calls and ensure we stay within a
     * fixed memory footprint.
     */
    hvr_vertex_cache_node_t *pool_head;
    hvr_vertex_cache_node_t *pool_mem;
    unsigned pool_size;

    uint8_t *dist_from_local_vert;

    // Keeps a count of mirrored vertices
    unsigned long long n_cached_vertices;

    struct {
        unsigned long long quiet_counter;
        unsigned long long fetch_neighbors_time;
        unsigned long long quiet_neighbors_time;
        unsigned long long update_metadata_time;
        unsigned long long nhits;
        unsigned long long nmisses;
    } cache_perf_info;
} hvr_vertex_cache_t;

static inline void set_dist_from_local_vert(hvr_vertex_cache_node_t *node,
        uint8_t dist, hvr_vertex_cache_t *cache) {
    size_t offset = node - cache->pool_mem;
    if (VERTEX_ID_PE(node->vert.id) != shmem_my_pe()) {
        cache->dist_from_local_vert[offset] = dist;
    }
}

static inline uint8_t get_dist_from_local_vert(hvr_vertex_cache_node_t *node,
        hvr_vertex_cache_t *cache) {
    size_t offset = node - cache->pool_mem;
    if (VERTEX_ID_PE(node->vert.id) == shmem_my_pe()) {
        return 0;
    } else {
        return cache->dist_from_local_vert[offset];
    }
}

static inline int local_neighbor_list_contains(hvr_vertex_cache_node_t *node) {
    return node->local_neighbors_next || node->local_neighbors_prev;
}

extern void hvr_vertex_cache_remove_from_local_neighbor_list(
        hvr_vertex_id_t vert, hvr_vertex_cache_t *cache);

extern void hvr_vertex_cache_add_to_local_neighbor_list(hvr_vertex_id_t id,
        hvr_internal_ctx_t *ctx);

/*
 * Initializes an already allocated block of memory to store a vertex cache.
 * cache is assumed to point to a block of memory of at least
 * sizeof(hvr_vertex_cache_t) bytes.
 */
extern void hvr_vertex_cache_init(hvr_vertex_cache_t *cache,
        hvr_partition_t npartitions);

/*
 * Given a vertex ID on a remote PE, look up that vertex in our local cache.
 * Only returns an entry if the newest timestep stored in that entry is new
 * enough given target_timestep. If no matching entry is found, returns NULL.
 *
 * May lead to evictions of very old cache entries that we now consider unusable
 * because of their age, as judged by CACHED_TIMESTEPS_TOLERANCE.
 */
extern hvr_vertex_cache_node_t *hvr_vertex_cache_lookup(hvr_vertex_id_t vert,
        hvr_vertex_cache_t *cache);

extern hvr_vertex_cache_node_t *hvr_vertex_cache_add(hvr_vertex_t *vert,
        hvr_partition_t part, hvr_vertex_cache_t *cache);

extern void hvr_vertex_cache_delete(hvr_vertex_t *vert,
        hvr_vertex_cache_t *cache);

#ifdef __cplusplus
}
#endif

#endif // _HVR_VERTEX_CACHE_H
