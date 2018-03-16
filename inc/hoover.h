#ifndef _HOOVER_H
#define _HOOVER_H
#include <stdint.h>

#include "hvr_common.h"
#include "hvr_avl_tree.h"

/*
 * High-level workflow of a HOOVER program:
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
 *         abort = check abort criteria;
 *     }
 *
 *     hvr_finalize();  // return to the user code, not a global barrier
 */

#define BITS_PER_BYTE 8

#define HVR_BUCKETS 1024
#define HVR_BUCKET_SIZE 7

typedef struct _hvr_internal_ctx_t hvr_internal_ctx_t;
typedef hvr_internal_ctx_t *hvr_ctx_t;

typedef int32_t hvr_time_t;
#define MAX_TIMESTAMP (INT32_MAX - 1)

/*
 * Sparse vector for representing properties on each vertex, and accompanying
 * utilities.
 */
typedef struct _hvr_sparse_vec_t {
    // Globally unique ID for this node
    vertex_id_t id;

    // PE that owns this vertex
    int pe;

    // Values for each feature
    double values[HVR_BUCKETS][HVR_BUCKET_SIZE];

    // Feature IDs, all entries in each timestamp slot guaranteed unique
    unsigned features[HVR_BUCKETS][HVR_BUCKET_SIZE];

    // Number of features present in each bucket
    unsigned bucket_size[HVR_BUCKETS];

    // Timestamp for each value set, all entries guaranteed unique
    hvr_time_t timestamps[HVR_BUCKETS];

    hvr_time_t finalized[HVR_BUCKETS];

    // The oldest bucket or first unused bucket (used to evict quickly).
    volatile unsigned next_bucket;

    hvr_time_t cached_timestamp;
    unsigned cached_timestamp_index;
    struct _hvr_sparse_vec_t *next_in_partition;
} hvr_sparse_vec_t;

/*
 * Create nvecs new, empty vectors. Collective call.
 */
hvr_sparse_vec_t *hvr_sparse_vec_create_n(const size_t nvecs);

/*
 * Initialize an empty sparse vector.
 */
void hvr_sparse_vec_init(hvr_sparse_vec_t *vec);

/*
 * Set the specified feature to the specified value in the provided vector.
 */
void hvr_sparse_vec_set(const unsigned feature, const double val,
        hvr_sparse_vec_t *vec, hvr_ctx_t in_ctx);

/*
 * Get the value for the specified feature in the provided vector.
 */
double hvr_sparse_vec_get(const unsigned feature, hvr_sparse_vec_t *vec,
        hvr_ctx_t in_ctx);

/*
 * Write the contents of this sparse vector to buf as a human-readable string.
 */
void hvr_sparse_vec_dump(hvr_sparse_vec_t *vec, char *buf,
        const size_t buf_size, hvr_ctx_t in_ctx);

// Set the globally unique ID of this sparse vector
void hvr_sparse_vec_set_id(const vertex_id_t id, hvr_sparse_vec_t *vec);

// Get the globally unique ID of this sparse vector
vertex_id_t hvr_sparse_vec_get_id(hvr_sparse_vec_t *vec);

// Get the PE that is responsible for this sparse vector
int hvr_sparse_vec_get_owning_pe(hvr_sparse_vec_t *vec);

#define HVR_CACHE_BUCKETS 512

typedef struct _hvr_sparse_vec_cache_node_t {
    unsigned offset;
    hvr_sparse_vec_t vec;
    struct _hvr_sparse_vec_cache_node_t *next;
} hvr_sparse_vec_cache_node_t;

typedef struct _hvr_sparse_vec_cache_t {
    hvr_sparse_vec_cache_node_t *buckets[HVR_CACHE_BUCKETS];
    unsigned bucket_size[HVR_CACHE_BUCKETS];
    hvr_sparse_vec_cache_node_t *pool;
    unsigned nhits, nmisses, nmisses_due_to_age;

    int hvr_cache_max_bucket_size;
} hvr_sparse_vec_cache_t;

void hvr_sparse_vec_cache_init(hvr_sparse_vec_cache_t *cache);

void hvr_sparse_vec_cache_clear(hvr_sparse_vec_cache_t *cache);

hvr_sparse_vec_t *hvr_sparse_vec_cache_lookup(unsigned offset,
        hvr_sparse_vec_cache_t *cache, hvr_time_t target_timestep);

void hvr_sparse_vec_cache_insert(unsigned offset, hvr_sparse_vec_t *vec,
        hvr_sparse_vec_cache_t *cache);

/*
 * Edge set utilities.
 */
typedef struct _hvr_edge_set_t {
    hvr_avl_tree_node_t *tree;
} hvr_edge_set_t;

extern hvr_edge_set_t *hvr_create_empty_edge_set();
extern void hvr_add_edge(const vertex_id_t local_vertex_id,
        const vertex_id_t global_vertex_id, hvr_edge_set_t *set);
extern int hvr_have_edge(const vertex_id_t local_vertex_id,
        const vertex_id_t global_vertex_id, hvr_edge_set_t *set);
extern size_t hvr_count_edges(const vertex_id_t local_vertex_id,
        hvr_edge_set_t *set);
extern void hvr_clear_edge_set(hvr_edge_set_t *set);
extern void hvr_release_edge_set(hvr_edge_set_t *set);
extern void hvr_print_edge_set(hvr_edge_set_t *set);

typedef unsigned long long bit_vec_element_type;

/*
 * Utilities used for storing a set of PEs. This class is used for storing both
 * neighbor PE lists and PE coupling lists.
 */
#define PE_SET_CACHE_SIZE 100
typedef struct _hvr_set_t {
    unsigned cache[PE_SET_CACHE_SIZE];
    int nelements;
    unsigned n_contained;
    bit_vec_element_type *bit_vector;
} hvr_set_t;

extern hvr_set_t *hvr_create_empty_set(hvr_ctx_t ctx);
extern hvr_set_t *hvr_create_empty_set_custom(const unsigned nvals,
        hvr_ctx_t ctx);
extern void hvr_set_insert(int pe, hvr_set_t *set);
extern void hvr_set_clear(int pe, hvr_set_t *set);
extern int hvr_set_contains(int pe, hvr_set_t *set);
extern unsigned hvr_set_count(hvr_set_t *set);
extern void hvr_set_wipe(hvr_set_t *set);
extern void hvr_set_merge(hvr_set_t *set, hvr_set_t *other);
extern void hvr_set_destroy(hvr_set_t *set);
extern void hvr_set_to_string(hvr_set_t *set, char *buf, unsigned buflen);

/*
 * Callback type definitions to be defined by the user for the HOOVER runtime to
 * call into.
 */
typedef void (*hvr_update_metadata_func)(hvr_sparse_vec_t *metadata,
        hvr_sparse_vec_t *neighbors, const size_t n_neighbors,
        hvr_set_t *couple_with, hvr_ctx_t ctx);
/*
 * Signature for measuring the distance between two points of metadata. Used for
 * updating the graph's structure.
 */
typedef double (*hvr_sparse_vec_distance_measure_func)(hvr_sparse_vec_t *a,
        hvr_sparse_vec_t *b, hvr_ctx_t ctx);

/*
 * API for finding the owner of a given vertex.
 */
typedef void (*hvr_vertex_owner_func)(vertex_id_t vertex, unsigned *out_pe,
        size_t *out_local_offset);

/*
 * API for checking if the simulation for this PE should be aborted based on the
 * status of vertices on this PE.
 */
typedef int (*hvr_check_abort_func)(hvr_sparse_vec_t *vertices,
        const size_t n_vertices, hvr_ctx_t ctx,
        hvr_sparse_vec_t *out_coupled_metric);

/*
 * API for checking if this PE might have any vertices that interact with
 * vertices on another PE.
 */
typedef int (*hvr_might_interact_func)(const uint16_t partition,
        hvr_set_t *partitions, uint16_t *interacting_partitions,
        unsigned *n_interacting_partitions,
        unsigned interacting_partitions_capacity, hvr_ctx_t ctx);

typedef uint16_t (*hvr_actor_to_partition)(hvr_sparse_vec_t *actor,
        hvr_ctx_t ctx);

typedef struct _hvr_internal_ctx_t {
    int initialized;
    int pe;
    int npes;

    vertex_id_t n_local_vertices;
    long long *vertices_per_pe;
    vertex_id_t max_n_local_vertices;
    long long n_global_vertices;
    uint16_t n_partitions;

    hvr_sparse_vec_t *vertices;

    hvr_edge_set_t *edges;

    hvr_time_t timestep;
    volatile hvr_time_t *symm_timestep;
    hvr_time_t *all_pe_timesteps;
    hvr_time_t *all_pe_timesteps_buffer;
    long *all_pe_timesteps_locks;
    hvr_time_t *last_timestep_using_partition;
    hvr_set_t *partition_time_window;
    hvr_set_t *other_pe_partition_time_window;
    hvr_set_t *tmp_partition_time_window;

    long *actor_to_partition_locks;
    long *partition_time_window_lock;
    uint16_t *actor_to_partition_map;

    hvr_update_metadata_func update_metadata;
    hvr_might_interact_func might_interact;
    hvr_check_abort_func check_abort;
    hvr_vertex_owner_func vertex_owner;
    hvr_actor_to_partition actor_to_partition;
    double connectivity_threshold;
    unsigned min_spatial_feature, max_spatial_feature;
    hvr_time_t max_timestep;

    hvr_sparse_vec_t *buffer;

    long long *p_wrk;
    int *p_wrk_int;
    long *p_sync;

    int dump_mode;
    FILE *dump_file;

    int strict_mode;
    int *strict_counter_dest;
    int *strict_counter_src;

    hvr_set_t *my_neighbors;

    hvr_set_t *coupled_pes;
    hvr_sparse_vec_t *coupled_pes_values;
    hvr_sparse_vec_t *coupled_pes_values_buffer;
    volatile long *coupled_lock;

    unsigned n_vector_cache_hits, n_vector_cache_misses;

    char my_hostname[1024];

    hvr_sparse_vec_t **partition_lists;
} hvr_internal_ctx_t;

// Must be called after shmem_init
extern void hvr_ctx_create(hvr_ctx_t *out_ctx);

extern void hvr_init(const uint16_t n_partitions,
        const vertex_id_t n_local_vertices,
        hvr_sparse_vec_t *vertices,
        hvr_update_metadata_func update_metadata,
        hvr_might_interact_func might_interact,
        hvr_check_abort_func check_abort,
        hvr_vertex_owner_func vertex_owner,
        hvr_actor_to_partition actor_to_partition,
        const double connectivity_threshold,
        const unsigned min_spatial_feature_inclusive,
        const unsigned max_spatial_feature_inclusive,
        const hvr_time_t max_timestep, hvr_ctx_t ctx);

extern void hvr_body(hvr_ctx_t ctx);

extern void hvr_finalize(hvr_ctx_t ctx);

extern hvr_time_t hvr_current_timestep(hvr_ctx_t ctx);
extern unsigned long long hvr_current_time_us();

#endif
