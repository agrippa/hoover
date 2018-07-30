/* For license: see LICENSE.txt file at top-level */

#ifndef _HVR_SPARSE_VEC_H
#define _HVR_SPARSE_VEC_H

#include "hvr_common.h"

// The maximum number of timesteps we record for each vertex.
#define HVR_BUCKETS 256

// The maximum number of features set on a vertex at any time.
#define HVR_BUCKET_SIZE 7

#define HVR_MAX_CONSTANT_ATTRS 3

/*
 * Sparse vector for representing properties on each vertex, and accompanying
 * utilities.
 *
 * Each sparse vector tracks its values for the current and last HVR_BUCKETS
 * timesteps. This is necessary so that other uncoupled PEs can read past
 * timestep data on any local vertex without accidentally seeing into the
 * "future". To some extent, you can think of this data structure as a sparse
 * vector replicated HVR_BUCKETS times, once for each historical timestep.
 */
typedef struct _hvr_sparse_vec_t {
    // Globally unique ID for this node if remotely accessible
    hvr_vertex_id_t id;

    hvr_time_t created_timestamp;
    hvr_time_t deleted_timestamp;
    hvr_graph_id_t graph;

    double const_values[HVR_MAX_CONSTANT_ATTRS];
    unsigned const_features[HVR_MAX_CONSTANT_ATTRS];
    unsigned n_const_features;

    // Values for each feature on each timestep
    double values[HVR_BUCKETS][HVR_BUCKET_SIZE];

    // Feature IDs, all entries in each timestamp slot guaranteed unique
    unsigned features[HVR_BUCKETS][HVR_BUCKET_SIZE];

    // Number of features present in each bucket
    unsigned bucket_size[HVR_BUCKETS];

    // Timestamp for each value set, all entries guaranteed unique
    hvr_time_t timestamps[HVR_BUCKETS];

    /*
     * Whether the timestep associated with each bucket is complete and the
     * entries for that timestep can be considered final.
     */
    hvr_time_t finalized[HVR_BUCKETS];

    // The oldest bucket or first unused bucket (used to evict quickly).
    volatile unsigned next_bucket;

    /*
     * We cache lookups for a single timestep to allow O(1) access for repeated
     * accesses to the same timestep.
     */
    hvr_time_t cached_timestamp;
    unsigned cached_timestamp_index;

    /*
     * Used to store lists of local vertices in each problem space partition.
     * Only valid for local vertices.
     */
    struct _hvr_sparse_vec_t *next_in_partition;
} hvr_sparse_vec_t;

/*
 * Create nvecs new, empty vectors. Collective call.
 */
hvr_sparse_vec_t *hvr_sparse_vec_create_n(const size_t nvecs,
        hvr_graph_id_t graph, hvr_ctx_t ctx);

/*
 * Remove these vertices from the graph.
 */
void hvr_sparse_vec_delete_n(hvr_sparse_vec_t *vecs,
        const size_t nvecs, hvr_ctx_t ctx);

/*
 * Initialize an empty sparse vector.
 */
void hvr_sparse_vec_init(hvr_sparse_vec_t *vec, hvr_graph_id_t graph,
        hvr_ctx_t ctx);

void hvr_sparse_vec_init_with_const_attrs(hvr_sparse_vec_t *vec,
        hvr_graph_id_t graph, unsigned *const_attr_features,
        double *const_attr_values, unsigned n_const_attrs, hvr_ctx_t in_ctx);

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

/*
 * Get the PE that is responsible for this sparse vector
 */
int hvr_sparse_vec_get_owning_pe(hvr_sparse_vec_t *vec);

/*
 * WARNING: In general, this API should never be used by HOOVER applications.
 *
 * We only expose it here to be used by testing code.
 */
void finalize_actor_for_timestep(hvr_sparse_vec_t *actor,
        const hvr_time_t timestep);

#endif // _HVR_SPARSE_VEC_H
