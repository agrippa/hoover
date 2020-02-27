/* For license: see LICENSE.txt file at top-level */

#ifndef _HOOVER_H
#define _HOOVER_H

#include <stdint.h>
#include <stdio.h>

#include "hvr_vertex_cache.h"
#include "hvr_neighbors.h"
#include "hvr_common.h"
#include "hvr_vertex_iter.h"
#include "hvr_mailbox.h"
#include "hvr_mailbox_buffer.h"
#include "hvr_set.h"
#include "hvr_dist_bitvec.h"
#include "hvr_sparse_arr.h"
#include "hvr_set_msg.h"
#include "hvr_msg_buf_pool.h"
#include "hvr_irregular_matrix.h"
#include "hvr_buffered_msgs.h"
#include "hvr_buffered_changes.h"

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
        hvr_ctx_t ctx, hvr_vertex_t *out_coupled_metric,
        uint64_t n_msgs_recvd_this_iter, uint64_t n_msgs_sent_this_iter,
        uint64_t n_msgs_recvd_total, uint64_t n_msgs_sent_total);

/*
 * Callback to check if this PE should leave the simulation.
 */
typedef int (*hvr_should_terminate_func)(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_vertex_t *local_coupled_val,
        hvr_vertex_t *all_coupled_vals,
        hvr_set_t *coupled_pes, int n_coupled_pes,
        int *updates_on_this_iter,
        hvr_set_t *terminated_coupled_pes,
        uint64_t n_msgs_recvd_this_iter,
        uint64_t n_msgs_sent_this_iter,
        uint64_t n_msgs_recvd_total,
        uint64_t n_msgs_sent_total);

/*
 * API for checking if this PE might have any vertices that interact with
 * vertices on another PE. This callback function must be thread safe.
 */
typedef void (*hvr_might_interact_func)(const hvr_partition_t partition,
        hvr_partition_t *interacting_partitions,
        unsigned *n_interacting_partitions,
        unsigned interacting_partitions_capacity, hvr_ctx_t ctx);

/*
 * API for calculating the problem space partition that a given vertex belongs
 * to.
 */
typedef hvr_partition_t (*hvr_actor_to_partition)(const hvr_vertex_t *actor,
        hvr_ctx_t ctx);

/*
 * API for checking if two vertices should have an edge between them, based on
 * application-specific logic. This function must be symmetric. That is, given
 * any two vertices A and B, the return value of should_have_edge(A, B) must be
 * the inverse of should_have_edge(B, A).
 */
typedef hvr_edge_type_t (*hvr_should_have_edge)(const hvr_vertex_t *target,
        const hvr_vertex_t *candidate, hvr_ctx_t ctx);

/*
 * All message definitions.
 */
typedef struct _hvr_vertex_update_t {
    hvr_vertex_t vert;
    uint8_t is_invalidation;
} hvr_vertex_update_t;

typedef struct _hvr_edge_create_msg_t {
    hvr_vertex_t src;
    hvr_vertex_id_t target;
    hvr_edge_type_t edge;
    int is_forward;
} hvr_edge_create_msg_t;

typedef struct _hvr_update_msg_t {
    union {
        hvr_vertex_update_t vert_update;
        hvr_edge_create_msg_t edge_update;
    } payload;
    uint8_t is_vert_update;
} hvr_update_msg_t;

typedef struct _hvr_partition_member_change_t {
    int pe;
    hvr_partition_t partition;
    int entered;
} hvr_partition_member_change_t;

typedef struct _hvr_vertex_subscription_t {
    int pe;
    hvr_vertex_id_t vert;
    int entered;
} hvr_vertex_subscription_t;

typedef struct _hvr_dead_pe_msg_t {
    int pe;
} hvr_dead_pe_msg_t;

typedef struct _hvr_coupling_msg_t {
    int pe;
    int updates_on_this_iter;
    hvr_time_t iter;
    hvr_vertex_t val;
} hvr_coupling_msg_t;

typedef struct _new_coupling_msg_t {
    int pe;
    hvr_time_t iter;
} new_coupling_msg_t;

typedef struct _new_coupling_msg_ack_t {
    int pe;
    int root_pe;
    int abort;
} new_coupling_msg_ack_t;

typedef struct _inter_vert_msg_t {
    hvr_vertex_id_t dst;
    hvr_vertex_t payload;
} inter_vert_msg_t;

typedef struct _hvr_partition_list_t {
    hvr_map_t map;
    hvr_partition_t n_partitions;
} hvr_partition_list_t;

#include "hvr_partition_list.h"

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

    int user_mutation_allowed;

    hvr_partition_t *vertex_partitions; // only set while exiting

    // Number of partitions passed in by the user
    hvr_partition_t n_partitions;

    // Set of edges for our local vertices
    hvr_irr_matrix_t edges;

    // Current iter
    hvr_time_t iter;

    // Sets of the partitions that have been live in a recent time window.
    hvr_set_t *subscribed_partitions;
    hvr_set_t *produced_partitions;
    hvr_set_t *new_subscribed_partitions;
    hvr_set_t *new_produced_partitions;

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

    int *p_wrk_int;
    long *p_sync;

    // Used to write traces of the simulation out, for later visualization
    int dump_mode;
    int cache_dump_mode;
    int only_last_iter_dump;
    FILE *dump_file;
    FILE *edges_dump_file;
    FILE *cache_dump_file;

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
    hvr_set_t *other_to_couple_with;
    hvr_set_t *other_coupled_pes;
    hvr_set_t *already_coupled_with;

    // Values retrieved from each coupled PE
    hvr_vertex_t *coupled_pes_values;
    int *updates_on_this_iter;

    // For debug printing
    char my_hostname[1024];

    // List of local vertices in each partition
    hvr_partition_list_t local_partition_lists;

    /*
     * List of locally mirrored vertices in each partition (actually points into
     * vec_cache).
     */
    hvr_partition_list_t mirror_partition_lists;

    uint8_t *partition_min_dist_from_local_vert;

    // Counter for which graph IDs have already been allocated
    hvr_graph_id_t allocated_graphs;

    hvr_vertex_cache_t vec_cache;

    hvr_dist_bitvec_local_subcopy_t local_partition_producers;
    hvr_dist_bitvec_local_subcopy_t local_terminated_pes;

    hvr_mailbox_t vertex_update_mailbox;
    hvr_mailbox_t forward_mailbox;
    hvr_mailbox_t vertex_msg_mailbox;

    hvr_mailbox_t coupling_mailbox;
    hvr_mailbox_t coupling_ack_and_dead_mailbox;
    hvr_mailbox_t coupling_val_mailbox;
    hvr_mailbox_t to_couple_with_mailbox;
    hvr_mailbox_t root_info_mailbox;

    hvr_mailbox_t vert_sub_mailbox;

    hvr_mailbox_buffer_t vertex_update_mailbox_buffer;
    hvr_mailbox_buffer_t vert_sub_mailbox_buffer;

    hvr_map_t producer_info;
    hvr_map_t dead_info;

    hvr_time_t *next_producer_info_check;
    hvr_time_t *curr_producer_info_interval;

    /*
     * Mapping from partition -> remote PE subscribing to each partition
     * Dimensions: (# partitions x # PEs)
     */
    hvr_sparse_arr_t remote_partition_subs;
    /*
     * Mapping from local vertex offset -> remote PE subscribing to each vertex
     * Dimensions: (# pre-allocated vertices per PE x # PEs)
     */
    hvr_sparse_arr_t remote_vert_subs;
    /*
     * Mapping from remote PE -> subscriptions I have to vertices on that PE
     * Dimensions: (# PEs x # pre-allocated vertices per PE)
     */
    hvr_sparse_arr_t my_vert_subs;

    unsigned max_graph_traverse_depth;
    unsigned send_neighbor_updates_for_explicit_subs;

    hvr_msg_buf_pool_t msg_buf_pool;

#define N_VERTICES_PER_BUF 10240
    hvr_partition_t *vert_partition_buf;

#define MAX_MODIFICATIONS 65536
    hvr_vertex_id_t edge_buffer[MAX_MODIFICATIONS];

    hvr_dist_bitvec_t partition_producers;
    hvr_dist_bitvec_t terminated_pes;

    hvr_partition_t *interacting;

    hvr_vertex_t *recently_created;

    hvr_buffered_msgs_t buffered_msgs;

    hvr_partition_t *new_producer_partitions_list;
    hvr_partition_t *new_subscriber_partitions_list;

    hvr_partition_t *prev_producer_partitions_list;
    size_t n_prev_producer_partitions;
    hvr_partition_t *prev_subscriber_partitions_list;
    size_t n_prev_subscriber_partitions;

    size_t max_active_partitions;

    int any_needs_processing;

    mspace edge_list_allocator;
    void *edge_list_pool;
    size_t edge_list_pool_size;

    hvr_buffered_changes_t buffered_changes;

    void *neighbors_list_pool;
    size_t neighbors_list_pool_size;
    mspace neighbors_list_tracker;

    uint64_t n_msgs_recvd_this_iter;
    uint64_t n_msgs_recvd_total;

    uint64_t vertex_update_mailbox_nmsgs;
    uint64_t vertex_update_mailbox_nmsgs_total;
    uint64_t vertex_update_mailbox_nattempts;
} hvr_internal_ctx_t;

/*
 * Information on the execution of a problem after it completes which is
 * returned to the caller.
 */
typedef struct _hvr_exec_info {
    hvr_time_t executed_iters;
    unsigned long long start_hvr_body_us;
    unsigned long long start_hvr_body_iterations_us;
    unsigned long long start_hvr_body_wrapup_us;
    unsigned long long end_hvr_body_us;
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
        unsigned send_neighbor_updates_for_explicit_subs,
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

extern uint64_t hvr_neighbors_min(hvr_vertex_t *vert, unsigned feature,
        uint64_t init_val, hvr_ctx_t in_ctx);

extern void hvr_get_neighbors(hvr_vertex_t *vert, hvr_neighbors_t *neighbors,
        hvr_ctx_t in_ctx);

extern void hvr_release_neighbors(hvr_neighbors_t *n, hvr_ctx_t in_ctx);

extern void hvr_reset_neighbors(hvr_neighbors_t *n, hvr_ctx_t in_ctx);

extern void hvr_create_edge_with_vertex_id(hvr_vertex_t *base,
        hvr_vertex_id_t neighbor, hvr_edge_type_t edge, hvr_ctx_t in_ctx);

extern void hvr_create_edge_with_vertex(hvr_vertex_t *base,
        hvr_vertex_t *neighbor, hvr_edge_type_t edge, hvr_ctx_t in_ctx);

extern hvr_vertex_t *hvr_get_vertex(hvr_vertex_id_t id, hvr_ctx_t in_ctx);

extern void hvr_send_msg(hvr_vertex_id_t dst, hvr_vertex_t *msg,
        hvr_internal_ctx_t *ctx);

// Messages are returned in the opposite of the order in which they are received
extern int hvr_poll_msg(hvr_vertex_t *vert,
        hvr_vertex_t *out, hvr_internal_ctx_t *ctx);

// Not for application use
extern void send_updates_to_all_subscribed_pes(
        hvr_vertex_t *vert,
        hvr_partition_t part,
        uint8_t is_invalidation,
        int is_delete,
        process_perf_info_t *perf_info,
        unsigned long long *time_sending,
        hvr_internal_ctx_t *ctx);

static inline hvr_partition_t wrap_actor_to_partition(const hvr_vertex_t *vec,
        hvr_internal_ctx_t *ctx) {
    hvr_partition_t partition = ctx->actor_to_partition(vec, ctx);
    assert(partition < ctx->n_partitions);
    return partition;
}

static inline void mark_for_processing(hvr_vertex_t *vert,
        hvr_internal_ctx_t *ctx) {
    vert->needs_processing = 1;
    ctx->any_needs_processing = 1;
}

size_t hvr_n_local_vertices(hvr_ctx_t in_ctx);

#endif
