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

#define HVR_MAX_SPARSE_VEC_CAPACITY 32

typedef struct _hvr_internal_ctx_t hvr_internal_ctx_t;
typedef hvr_internal_ctx_t *hvr_ctx_t;

/*
 * Sparse vector for representing properties on each vertex, and accompanying
 * utilities.
 */
typedef struct _hvr_sparse_vec_t {
    // Globally unique ID for this node
    vertex_id_t id;

    // Values for each feature
    double values[HVR_MAX_SPARSE_VEC_CAPACITY];

    // Feature IDs
    unsigned features[HVR_MAX_SPARSE_VEC_CAPACITY];

    // Timestamp for each value set
    uint64_t timestamp[HVR_MAX_SPARSE_VEC_CAPACITY];

    // Number of features set on this vertex
    unsigned nfeatures;
} hvr_sparse_vec_t;

/*
 * Create nvecs new, empty vectors. Collective call.
 */
hvr_sparse_vec_t *hvr_sparse_vec_create_n(const size_t nvecs);

/*
 * Set the specified feature to the specified value in the provided vector.
 */
void hvr_sparse_vec_set(const unsigned feature, const double val,
        hvr_sparse_vec_t *vec, hvr_ctx_t in_ctx);

/*
 * Get the value for the specified feature in the provided vector.
 */
double hvr_sparse_vec_get(const unsigned feature, const hvr_sparse_vec_t *vec,
        hvr_ctx_t in_ctx);

/*
 * Write the contents of this sparse vector to buf as a human-readable string.
 */
void hvr_sparse_vec_dump(hvr_sparse_vec_t *vec, char *buf,
        const size_t buf_size);

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

/*
 * Callback type definitions to be defined by the user for the HOOVER runtime to
 * call into.
 */
typedef void (*hvr_update_metadata_func)(hvr_sparse_vec_t *metadata,
        hvr_sparse_vec_t *neighbors, const size_t n_neighbors, hvr_ctx_t ctx);
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
        const size_t n_vertices, hvr_ctx_t ctx);

typedef struct _hvr_internal_ctx_t {
    int initialized;
    int pe;
    int npes;

    vertex_id_t n_local_vertices;
    long long *vertices_per_pe;
    long long n_global_vertices;

    hvr_sparse_vec_t *vertices;

    hvr_edge_set_t *edges;

    uint64_t timestep;

    hvr_update_metadata_func update_metadata;
    hvr_check_abort_func check_abort;
    hvr_sparse_vec_distance_measure_func distance_measure;
    hvr_vertex_owner_func vertex_owner;
    double connectivity_threshold;

    hvr_sparse_vec_t *buffer;

    int strict_mode;
    int *strict_counter_dest;
    int *strict_counter_src;
} hvr_internal_ctx_t;

extern void hvr_ctx_create(hvr_ctx_t *out_ctx);

extern void hvr_init(const vertex_id_t n_local_vertices,
        hvr_sparse_vec_t *vertices, hvr_edge_set_t *edges,
        hvr_update_metadata_func update_metadata,
        hvr_sparse_vec_distance_measure_func distance_measure,
        hvr_check_abort_func check_abort,
        hvr_vertex_owner_func vertex_owner,
        const double connectivity_threshold, hvr_ctx_t ctx);

extern void hvr_body(hvr_ctx_t ctx);

extern void hvr_finalize(hvr_ctx_t ctx);

extern uint64_t hvr_current_timestep(hvr_ctx_t ctx);
extern unsigned long long hvr_current_time_us();

#endif
