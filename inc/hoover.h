/* For license: see LICENSE.txt file at top-level */

#ifndef _HOOVER_H
#define _HOOVER_H
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "hvr_vertex_pool.h"
#include "hvr_vertex_cache.h"
#include "hvr_common.h"
#include "hvr_edge_set.h"
#include "hvr_vertex_iter.h"
#include "hvr_mailbox.h"
#include "hvr_set.h"
#include "hvr_dist_bitvec.h"
#include "hvr_sparse_arr.h"
#include "hvr_set_msg.h"
#include "hvr_msg_buf_pool.h"
#include "hvr_2d_edge_set.h"

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

/*
 * Callback type definitions to be defined by the user for the HOOVER runtime to
 * call into.
 *
 * hvr_update_metadata_func updates a given local vertex's attributes (metadata)
 * given a list of all vertices it has edges with (neighbors, n_neighbors).
 * Based on these updates, the HOOVER programmer can then choose to couple with
 * some other PEs by setting elements in couple_with.
 */
typedef void (*hvr_update_metadata_func)(hvr_vertex_t *vert,
        hvr_set_t *couple_with, hvr_ctx_t ctx);

/*
 * Optional callback at the start of every timestep, usually used to update
 * non-graph data structures or insert/remove vertices.
 */
typedef void (*hvr_start_time_step)(hvr_vertex_iter_t *iter,
        hvr_set_t *couple_with, hvr_ctx_t ctx);

/*
 * API for checking if the simulation for this PE should be aborted based on the
 * status of vertices on this PE.
 */
typedef void (*hvr_update_coupled_val_func)(hvr_vertex_iter_t *iter,
        hvr_ctx_t ctx, hvr_vertex_t *out_coupled_metric);

/*
 * Callback to check if this PE should leave the simulation.
 */
typedef int (*hvr_should_terminate_func)(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_vertex_t *local_coupled_val,
        hvr_vertex_t *all_coupled_vals,
        hvr_vertex_t *global_coupled_val,
        hvr_set_t *coupled_pes, int n_coupled_pes,
        int *updates_on_this_iter,
        hvr_set_t *terminated_coupled_pes);

/*
 * API for checking if this PE might have any vertices that interact with
 * vertices on another PE.
 */
typedef void (*hvr_might_interact_func)(const hvr_partition_t partition,
        hvr_partition_t *interacting_partitions,
        unsigned *n_interacting_partitions,
        unsigned interacting_partitions_capacity, hvr_ctx_t ctx);

/*
 * API for calculating the problem space partition that a given vertex belongs
 * to.
 */
typedef hvr_partition_t (*hvr_actor_to_partition)(hvr_vertex_t *actor,
        hvr_ctx_t ctx);

/*
 * API for checking if two vertices should have an edge between them, based on
 * application-specific logic.
 */
typedef hvr_edge_type_t (*hvr_should_have_edge)(hvr_vertex_t *target,
        hvr_vertex_t *candidate, hvr_ctx_t ctx);

/*
 * All message definitions.
 */
#define VERT_PER_UPDATE 16
typedef struct _hvr_vertex_update_t {
    hvr_vertex_t verts[VERT_PER_UPDATE];
    unsigned len;
} hvr_vertex_update_t;

typedef struct _hvr_partition_member_change_t {
    int pe;
    hvr_partition_t partition;
    int entered;
} hvr_partition_member_change_t;

typedef struct _hvr_dead_pe_msg_t {
    int pe;
} hvr_dead_pe_msg_t;

typedef struct _hvr_coupling_msg_t {
    int pe;
    int updates_on_this_iter;
    hvr_time_t iter;
    hvr_vertex_t val;
} hvr_coupling_msg_t;

typedef struct _start_iter_msg_t {
    int pe;
} start_iter_msg_t;

typedef struct _new_coupling_msg_t {
    int pe;
    hvr_time_t iter;
} new_coupling_msg_t;

typedef struct _new_coupling_msg_ack_t {
    int pe;
    int root_pe;
    int abort;
} new_coupling_msg_ack_t;

typedef struct _cluster_ack_msg_t {
    int pe;
    unsigned nforwards;
    int requires_resend;
} cluster_ack_msg_t;

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

    hvr_vertex_pool_t pool;
    hvr_partition_t *vertex_partitions; // only set while exiting

    // Number of partitions passed in by the user
    hvr_partition_t n_partitions;

    // Set of edges for our local vertices
    hvr_2d_edge_set_t edges;

    // Current iter
    hvr_time_t iter;

    // Sets of the partitions that have been live in a recent time window.
    hvr_set_t *subscriber_partition_time_window;
    hvr_set_t *producer_partition_time_window;
    hvr_set_t *new_subscriber_partition_time_window;
    hvr_set_t *new_producer_partition_time_window;

    // User callbacks
    hvr_update_metadata_func update_metadata;
    hvr_might_interact_func might_interact;
    hvr_update_coupled_val_func update_coupled_val;
    hvr_actor_to_partition actor_to_partition;
    hvr_should_have_edge should_have_edge;
    hvr_start_time_step start_time_step;
    hvr_should_terminate_func should_terminate;

    // Limit on execution time
    unsigned long long max_elapsed_seconds;

    // List of asynchronously fetching vertices
    hvr_vertex_cache_node_t **neighbor_buffer;

    long long *p_wrk;
    int *p_wrk_int;
    long *p_sync;

    // Used to write traces of the simulation out, for later visualization
    int dump_mode;
    int only_last_iter_dump;
    FILE *dump_file;
    FILE *edges_dump_file;

    /*
     * Strict mode forces a global barrier on every iteration of the simulation.
     * For debug/dev, not prod.
     */
    int strict_mode;
    int *strict_counter_dest;
    int *strict_counter_src;

    // Set of PEs we are in coupled execution with
    hvr_set_t *coupled_pes;
    int coupled_pes_root;
    hvr_set_msg_t coupled_pes_msg;

    // Set of PEs we were coupled with on the previous iteration
    hvr_set_t *prev_coupled_pes;
    // PEs we would like to be coupled with at the end of the current iteration.
    hvr_set_t *to_couple_with;
    hvr_set_msg_t to_couple_with_msg;
    hvr_set_msg_t terminating_pes_msg;
    hvr_set_msg_t new_all_terminated_cluster_pes_msg;

    hvr_set_t *received_from;
    // Accurate for everyone, using the dead_mailbox
    hvr_set_t *all_terminated_pes;
    // Only accurate for PEs in my cluster
    hvr_set_t *all_terminated_cluster_pes;
    hvr_set_t *prev_all_terminated_cluster_pes;
    hvr_set_t *new_all_terminated_cluster_pes;
    hvr_set_t *terminating_pes;
    hvr_set_t *other_terminating_pes;
    hvr_set_t **finalized_sets;
    hvr_set_t *other_to_couple_with;
    hvr_set_t *other_coupled_pes;
    hvr_set_t *already_coupled_with;

    // Values retrieved from each coupled PE
    hvr_vertex_t *coupled_pes_values;
    int *updates_on_this_iter;

    /*
     * An array of bit vectors, each of npes bits.
     *
     * Each bit vector is associated with a single partition, and signals the
     * PEs that have that partition active.
     *
     * These per-partition bit vectors are spread across all PEs.
     */
    hvr_partition_t *pes_per_partition;
    hvr_partition_t partitions_per_pe_vec_length_in_words;
    hvr_partition_t partitions_per_pe;
    hvr_partition_t *local_pes_per_partition_buffer;

    // For debug printing
    char my_hostname[1024];

    // List of local vertices in each partition
    hvr_vertex_t **local_partition_lists;

    /*
     * List of locally mirrored vertices in each partition (actually points into
     * vec_cache).
     */
    hvr_vertex_t **mirror_partition_lists;

    unsigned *partition_min_dist_from_local_vert;

    // Counter for which graph IDs have already been allocated
    hvr_graph_id_t allocated_graphs;

    hvr_vertex_cache_t vec_cache;

    hvr_dist_bitvec_local_subcopy_t local_partition_producers;
    hvr_dist_bitvec_local_subcopy_t local_terminated_pes;

    hvr_mailbox_t vertex_update_mailbox;
    hvr_mailbox_t vertex_delete_mailbox;
    hvr_mailbox_t forward_mailbox;

    // Actually used
    hvr_mailbox_t start_iter_mailbox;
    hvr_mailbox_t start_iter_ack_mailbox;
    hvr_mailbox_t coupling_mailbox;
    hvr_mailbox_t coupling_ack_and_dead_mailbox;
    hvr_mailbox_t cluster_mailbox;
    hvr_mailbox_t coupling_val_mailbox;
    hvr_mailbox_t to_couple_with_mailbox;
    hvr_mailbox_t root_info_mailbox;

    hvr_dist_bitvec_local_subcopy_t *producer_info;
    hvr_dist_bitvec_local_subcopy_t *dead_info;

    // Edge from PE -> partitions we need to notify it about
    hvr_sparse_arr_t pe_subscription_info;

    unsigned max_graph_traverse_depth;

    hvr_vertex_update_t *buffered_updates;
    hvr_vertex_update_t *buffered_deletes;

    hvr_msg_buf_pool_t msg_buf_pool;

#if 0
    hvr_msg_buf_node_t *buffered_msgs;
    unsigned n_buffered_msgs;
#endif

#define N_VERTICES_PER_BUF 10240
    hvr_partition_t *vert_partition_buf;

#define MAX_MODIFICATIONS 2048
    hvr_edge_info_t modify_buffer[MAX_MODIFICATIONS];
    hvr_edge_type_t modify_buffer_info[MAX_MODIFICATIONS];

    uint64_t modify_vert_buffer[MAX_MODIFICATIONS];
    hvr_edge_type_t modify_dir_buffer[MAX_MODIFICATIONS];
    hvr_vertex_t *vert_ptr_buffer[MAX_MODIFICATIONS];

    hvr_dist_bitvec_t partition_producers;
    hvr_dist_bitvec_t terminated_pes;

    hvr_partition_t *interacting;
} hvr_internal_ctx_t;

/*
 * Information on the execution of a problem after it completes which is
 * returned to the caller.
 */
typedef struct _hvr_exec_info {
    hvr_time_t executed_iters;
} hvr_exec_info;

// Must be called after shmem_init, zeroes out_ctx and fills in pe and npes
extern void hvr_ctx_create(hvr_ctx_t *out_ctx);

// Reserve a graph identifier for allocating vertices inside.
extern hvr_graph_id_t hvr_graph_create(hvr_ctx_t ctx);

// Initialize the state of the simulation/ctx
extern void hvr_init(const hvr_partition_t n_partitions,
        hvr_update_metadata_func update_metadata,
        hvr_might_interact_func might_interact,
        hvr_update_coupled_val_func update_coupled_val,
        hvr_actor_to_partition actor_to_partition,
        hvr_start_time_step start_time_step,
        hvr_should_have_edge should_have_edge,
        hvr_should_terminate_func should_terminate,
        unsigned long long max_elapsed_seconds,
        unsigned max_graph_traverse_depth,
        hvr_ctx_t ctx);

/*
 * Run the simulation. Returns when the local PE is done, but that return is not
 * collective.
 */
extern hvr_exec_info hvr_body(hvr_ctx_t ctx);

// Collective call to clean up
extern void hvr_finalize(hvr_ctx_t ctx);

// Get the PE ID we are running on
extern int hvr_my_pe(hvr_ctx_t ctx);

// Simple utility for time measurement in microseconds
extern unsigned long long hvr_current_time_us();

extern int hvr_get_neighbors(hvr_vertex_t *vert, hvr_vertex_t ***out_verts,
        hvr_edge_type_t **out_dirs, hvr_ctx_t in_ctx);

extern hvr_vertex_t *hvr_get_vertex(hvr_vertex_id_t vert_id, hvr_ctx_t ctx);

#ifdef __cplusplus
}
#endif

#endif
