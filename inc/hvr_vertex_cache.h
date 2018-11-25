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
     * Number of edges that separate this vertex from closest vertex stored
     * locally on the current PE. Is zero for local vertices.
     */
    unsigned min_dist_from_local_vertex;

    /*
     * Maintain lists of mirrored vertices in each partition to enable quick
     * iteration over a subset of mirrored vertices based on partition.
     */
    struct _hvr_vertex_cache_node_t *part_next;
    struct _hvr_vertex_cache_node_t *part_prev;
    struct _hvr_vertex_cache_node_t *tmp;
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
    unsigned pool_size;

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
