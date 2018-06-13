/* For license: see LICENSE.txt file at top-level */

#ifndef _HOOVER_H
#define _HOOVER_H
#include <stdint.h>
#include <stdio.h>

#include "hvr_sparse_vec_pool.h"
#include "hvr_sparse_vec.h"
#include "hvr_common.h"
#include "hoover_internal.h"
#include "hvr_avl_tree.h"
#include "hvr_vertex_iter.h"

/*
 * High-level workflow of the HOOVER runtime:
 *
 *     hvr_init();
 *
 *     abort = false;
 *
 *     while (!abort) {
 *         update vertex metadata;
 *
 *         update edges based on metadata;
 *
 *         abort = check user-defined abort criteria;
 *     }
 *
 *     hvr_finalize();  // return to the user code, not a global barrier
 */

#define BITS_PER_BYTE 8
#define BITS_PER_WORD (BITS_PER_BYTE * sizeof(unsigned))

// Number of elements to cache in a hvr_set_t
#define PE_SET_CACHE_SIZE 100

#define MAX_TIMESTAMP (INT32_MAX - 1)

typedef unsigned long long bit_vec_element_type;

/*
 * Utilities used for storing a set of integers. This class is used for storing
 * sets of PEs, of partitions, and is flexible enough to store anything
 * integer-valued.
 *
 * Under the covers, hvr_set_t at its core is a bit vector, but also uses a
 * small fixed-size cache to enable quick iteration over all elements in the set
 * when only a few elements are set in a large bit vector.
 *
 * HOOVER user code is never expected to create sets, but may be required to
 * manipulate them.
 */
typedef struct _hvr_set_t {
    /*
     * A fixed-size list of elements set in this cache, enabling quick iteration
     * over contained elements as long as there are <= PE_SET_CACHE_SIZE
     * elements contained. When we can use cache, using it means we don't have
     * to iterate over the full bit_vector to look for all contained elements.
     */
    unsigned cache[PE_SET_CACHE_SIZE];

    // Number of elements in the cache
    int nelements;

    // Total number of elements inserted in this cache
    unsigned n_contained;

    // Backing bit vector
    bit_vec_element_type *bit_vector;
} hvr_set_t;

/*
 * Add a given value to this set.
 */
extern int hvr_set_insert(int pe, hvr_set_t *set);

/*
 * Check if a given value exists in this set.
 */
extern int hvr_set_contains(int pe, hvr_set_t *set);

/*
 * Count how many elements are in this set.
 */
extern unsigned hvr_set_count(hvr_set_t *set);

/*
 * Remove all elements from this set.
 */
extern void hvr_set_wipe(hvr_set_t *set);

/*
 * Free the memory used by the given set.
 */
extern void hvr_set_destroy(hvr_set_t *set);

/*
 * Create a human-readable string from the provided set.
 */
extern void hvr_set_to_string(hvr_set_t *set, char *buf, unsigned buflen);

/*
 * Return an array containing all values in the provided set.
 */
extern unsigned *hvr_set_non_zeros(hvr_set_t *set,
        unsigned *n_non_zeros, int *user_must_free);

/*
 * Callback type definitions to be defined by the user for the HOOVER runtime to
 * call into.
 *
 * hvr_update_metadata_func updates a given local vertex's attributes (metadata)
 * given a list of all vertices it has edges with (neighbors, n_neighbors).
 * Based on these updates, the HOOVER programmer can then choose to couple with
 * some other PEs by setting elements in couple_with.
 */
typedef void (*hvr_update_metadata_func)(hvr_sparse_vec_t *metadata,
        hvr_sparse_vec_t *neighbors, const size_t n_neighbors,
        hvr_set_t *couple_with, hvr_ctx_t ctx);

/*
 * Optional callback at the start of every timestep, usually used to update
 * non-graph data structures or insert/remove vertices.
 */
typedef void (*hvr_start_time_step)(hvr_vertex_iter_t *iter, hvr_ctx_t ctx);

/*
 * API for checking if the simulation for this PE should be aborted based on the
 * status of vertices on this PE.
 */
typedef int (*hvr_check_abort_func)(hvr_vertex_iter_t *iter,
        hvr_ctx_t ctx, hvr_sparse_vec_t *out_coupled_metric);

/*
 * API for checking if this PE might have any vertices that interact with
 * vertices on another PE.
 */
typedef int (*hvr_might_interact_func)(const uint16_t partition,
        hvr_set_t *partitions, uint16_t *interacting_partitions,
        unsigned *n_interacting_partitions,
        unsigned interacting_partitions_capacity, hvr_ctx_t ctx);

/*
 * API for calculating the problem space partition that a given vertex belongs
 * to.
 */
typedef uint16_t (*hvr_actor_to_partition)(hvr_sparse_vec_t *actor,
        hvr_ctx_t ctx);

/*
 * Per-PE data structure for storing all information about the running problem
 * so we don't have file scope variables. Enables the possibility in the future
 * of multiple HOOVER problems running on the same PE.
 */
typedef struct _hvr_internal_ctx_t {
    // Has the HOOVER runtime been initialized?
    int initialized;
    // The current PE (caches shmem_my_pe())
    int pe;
    // The number of PEs (caches shmem_n_pes())
    int npes;

    hvr_sparse_vec_pool_t *pool;

    // Number of partitions passed in by the user
    uint16_t n_partitions;

    // Set of edges for our local vertices
    hvr_edge_set_t *edges;

    // Current timestep
    hvr_time_t timestep;
    // Remotely visible current timestep
    volatile hvr_time_t *symm_timestep;

    /*
     * Used for tracking every other PE's current timestep to ensure we don't
     * get too out of sync.
     */
    hvr_time_t *all_pe_timesteps;
    hvr_time_t *all_pe_timesteps_buffer;
    long *all_pe_timesteps_locks;
    /*
     * For each partition, the last local timestep to have a local vertex inside
     * it.
     */
    hvr_time_t *last_timestep_using_partition;

    // Sets of the partitions that have been live in a recent time window.
    hvr_set_t *partition_time_window;
    hvr_set_t *other_pe_partition_time_window;
    hvr_set_t *tmp_partition_time_window;
    long *partition_time_window_lock;

    /*
     * Mapping from each local vertex to its partition. Globally visible and
     * concurrently accessible using the R/W lock actor_to_partition_lock.
     */
    long *actor_to_partition_lock;
    uint16_t *actor_to_partition_map;

    // User callbacks
    hvr_update_metadata_func update_metadata;
    hvr_might_interact_func might_interact;
    hvr_check_abort_func check_abort;
    hvr_actor_to_partition actor_to_partition;
    hvr_start_time_step start_time_step;

    /*
     * Distance threshold below which edges are automatically added between
     * vertices.
     */
    double connectivity_threshold;
    // Feature index range for the spatial features of each vertex
    unsigned min_spatial_feature, max_spatial_feature;
    // Limit on number of timesteps to run for current simulation
    hvr_time_t max_timestep;

    // List of asynchronously fetching vertices
    hvr_sparse_vec_cache_node_t **neighbor_buffer;
    // Buffer of edges for a given vertex, to be passed to update_metadata
    hvr_sparse_vec_t *buffered_neighbors;
    // PEs we are fetching each element of neighbor_buffer from
    int *buffered_neighbors_pes;

    long long *p_wrk;
    int *p_wrk_int;
    long *p_sync;

    // Used to write traces of the simulation out, for later visualization
    int dump_mode;
    FILE *dump_file;

    /*
     * Strict mode forces a global barrier on every iteration of the simulation.
     * For debug/dev, not prod.
     */
    int strict_mode;
    int *strict_counter_dest;
    int *strict_counter_src;

    /*
     * List of PEs that may have vertices my vertices interact with (i.e. have
     * edges with). No need to check any vertices in any PEs that are not in
     * this set.
     */
    hvr_set_t *my_neighbors;

    // Set of PEs we are in coupled execution with
    hvr_set_t *coupled_pes;
    // Values retrieved from each coupled PE on each timestep
    hvr_sparse_vec_t *coupled_pes_values;
    hvr_sparse_vec_t *coupled_pes_values_buffer;
    volatile long *coupled_lock;

    /*
     * An array of bit vectors, each of npes bits.
     *
     * Each bit vector is associated with a single partition, and signals the
     * PEs that have that partition active.
     *
     * These per-partition bit vectors are spread across all PEs.
     */
    unsigned *pes_per_partition;
    unsigned partitions_per_pe_vec_length_in_words;
    unsigned partitions_per_pe;
    unsigned *local_pes_per_partition_buffer;

    // Track hits/missed by the cached_timestamp field of each vertex
    unsigned n_vector_cache_hits, n_vector_cache_misses;

    // For debug printing
    char my_hostname[1024];

    // List of local vertices in each partition
    hvr_sparse_vec_t **partition_lists;
} hvr_internal_ctx_t;

// Must be called after shmem_init, zeroes out_ctx and fills in pe and npes
extern void hvr_ctx_create(hvr_ctx_t *out_ctx);

// Initialize the state of the simulation/ctx
extern void hvr_init(const uint16_t n_partitions,
        hvr_update_metadata_func update_metadata,
        hvr_might_interact_func might_interact,
        hvr_check_abort_func check_abort,
        hvr_actor_to_partition actor_to_partition,
        hvr_start_time_step start_time_step,
        const double connectivity_threshold,
        const unsigned min_spatial_feature_inclusive,
        const unsigned max_spatial_feature_inclusive,
        const hvr_time_t max_timestep, hvr_ctx_t ctx);

/*
 * Run the simulation. Returns when the local PE is done, but that return is not
 * collective.
 */
extern void hvr_body(hvr_ctx_t ctx);

// Collective call to clean up
extern void hvr_finalize(hvr_ctx_t ctx);

// Get the current timestep of the local PE
extern hvr_time_t hvr_current_timestep(hvr_ctx_t ctx);

// Get the PE ID we are running on
extern int hvr_my_pe(hvr_ctx_t ctx);

// Simple utility for time measurement in microseconds
extern unsigned long long hvr_current_time_us();

#endif
