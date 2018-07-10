/* For license: see LICENSE.txt file at top-level */

#ifndef _HOOVER_INTERNAL_H
#define _HOOVER_INTERNAL_H

#include "hvr_sparse_vec.h"
#include "hvr_avl_tree.h"

#define HVR_CACHE_BUCKETS 128

/*
 * A cache data structure used locally on each PE to store remote vertices that
 * have already been fetched. This data structure stores the remote vertex
 * itself.
 */
typedef struct _hvr_sparse_vec_cache_node_t {
    // ID of this vertex
    hvr_vertex_id_t vert;

    hvr_time_t requested_on;

    // Contents of the vec itself
    hvr_sparse_vec_t vec;

    /*
     * Flag indicating if this node is currently being filled asynchronously,
     * and we don't know if the contents are ready.
     */
    int pending_comm;

    // Pointer to the next cache node in the same bucket
    struct _hvr_sparse_vec_cache_node_t *next;
} hvr_sparse_vec_cache_node_t;

/*
 * Per-remote PE data structure used to store all fetched and cached vertices
 * from that remote PE. Vertices are hashed by their offset in the local
 * vertices of the remote PE and stored in separate buckets.
 */
typedef struct _hvr_sparse_vec_cache_t {
    // Lists of vertices by vertex hash
    hvr_sparse_vec_cache_node_t *buckets[HVR_CACHE_BUCKETS];

    /*
     * Pool of pre-allocated but unused vertex data structures. Used
     * to reduce system memory management calls and ensure we stay within a
     * fixed memory footprint.
     */
    hvr_sparse_vec_cache_node_t *pool;
    unsigned pool_size;

    // Performance metrics tracked per remote PE
    unsigned nhits, nmisses, nmisses_due_to_age;
} hvr_sparse_vec_cache_t;

/*
 * Initializes an already allocated block of memory to store a vertex cache.
 * cache is assumed to point to a block of memory of at least
 * sizeof(hvr_sparse_vec_cache_t) bytes.
 */
void hvr_sparse_vec_cache_init(hvr_sparse_vec_cache_t *cache);

/*
 * Given a vertex ID on a remote PE, look up that vertex in our local cache.
 * Only returns an entry if the newest timestep stored in that entry is new
 * enough given target_timestep. If no matching entry is found, returns NULL.
 *
 * May lead to evictions of very old cache entries that we now consider unusable
 * because of their age, as judged by CACHED_TIMESTEPS_TOLERANCE.
 */
hvr_sparse_vec_cache_node_t *hvr_sparse_vec_cache_lookup(hvr_vertex_id_t vert,
        hvr_sparse_vec_cache_t *cache, hvr_time_t target_timestep);

/*
 * Reserve a node in the specified cache to store the contents of the specified
 * vertex. This routine is useful for asynchronous fetches, allowing us to
 * reserve a persistent block of in-cache memory ahead-of-time and
 * asynchronously fetch into it.
 */
hvr_sparse_vec_cache_node_t *hvr_sparse_vec_cache_reserve(
        hvr_vertex_id_t vert, hvr_sparse_vec_cache_t *cache,
        hvr_time_t curr_timestep, int pe);

/*
 * Edge set utilities.
 *
 * An edge set is simply a space efficient data structure for storing edges
 * between vertices in the graph.
 *
 * We use an AVL tree of AVL trees to keep lookups O(logN). The AVL tree
 * directly reference from hvr_edge_set_t has a node per local vertex, for any
 * local vertex that has any edges. Each node in this top-level AVL tree itself
 * references an AVL subtree which stores the vertices that the local vertex has
 * edges with.
 */
typedef struct _hvr_edge_set_t {
    hvr_avl_tree_node_t *tree;
} hvr_edge_set_t;

extern hvr_edge_set_t *hvr_create_empty_edge_set();
extern void hvr_add_edge(const hvr_vertex_id_t local_vertex_id,
        const hvr_vertex_id_t global_vertex_id, hvr_edge_set_t *set);
extern int hvr_have_edge(const hvr_vertex_id_t local_vertex_id,
        const hvr_vertex_id_t global_vertex_id, hvr_edge_set_t *set);
extern size_t hvr_count_edges(const hvr_vertex_id_t local_vertex_id,
        hvr_edge_set_t *set);
extern void hvr_clear_edge_set(hvr_edge_set_t *set);
extern void hvr_release_edge_set(hvr_edge_set_t *set);
extern void hvr_print_edge_set(hvr_edge_set_t *set);

#endif // _HOOVER_INTERNAL_H
