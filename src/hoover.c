/* For license: see LICENSE.txt file at top-level */

#define _BSD_SOURCE
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>
#include <math.h>
#include <limits.h>

#include <shmem.h>
#include <shmemx.h>

#include "hoover.h"
#include "hvr_vertex_iter.h"
#include "hvr_mailbox.h"

// #define DETAILED_PRINTS

#define MAX_INTERACTING_PARTITIONS 3000
#define N_SEND_ATTEMPTS 10

#define FINE_GRAIN_TIMING

// #define TRACK_VECTOR_GET_CACHE

static int print_profiling = 1;
// Purely for performance testing, should not be used live
static int dead_pe_processing = 1;
static FILE *profiling_fp = NULL;
static volatile int this_pe_has_exited = 0;

static const hvr_time_t check_producers_freq = 3;

typedef enum {
    CREATE, DELETE, UPDATE
} hvr_change_type_t;

static hvr_vertex_cache_node_t *handle_new_vertex(hvr_vertex_t *new_vert,
        unsigned long long *time_updating,
        unsigned long long *time_updating_edges,
        unsigned long long *time_creating_edges,
        unsigned *count_new_should_have_edges,
        unsigned long long *time_creating,
        hvr_internal_ctx_t *ctx);

static unsigned process_vertex_updates(hvr_internal_ctx_t *ctx,
        process_perf_info_t *perf_info);

#define USE_CSWAP_BITWISE_ATOMICS

#if SHMEM_MAJOR_VERSION == 1 && SHMEM_MINOR_VERSION >= 4 || SHMEM_MAJOR_VERSION >= 2

#define SHMEM_ULONGLONG_ATOMIC_OR shmem_ulonglong_atomic_or
#define SHMEM_UINT_ATOMIC_OR shmem_uint_atomic_or
#define SHMEM_UINT_ATOMIC_AND shmem_uint_atomic_and

#else
/*
 * Pre 1.4 some OpenSHMEM implementations offered a shmemx variant of atomic
 * bitwise functions. For others, we have to implement it on top of cswap.
 */

#ifdef USE_CSWAP_BITWISE_ATOMICS
#warning "Heads up! Using atomic compare-and-swap to implement atomic bitwise atomics!"

static void inline _shmem_ulonglong_atomic_or(unsigned long long *dst,
        unsigned long long val, int pe) {
    unsigned long long curr_val;
    shmem_getmem(&curr_val, dst, sizeof(curr_val), pe);

    while (1) {
        unsigned long long new_val = (curr_val | val);
        unsigned long long old_val = SHMEM_ULL_CSWAP(dst, curr_val, new_val, pe);
        if (old_val == curr_val) return;
        curr_val = old_val;
    }
}

static void inline _shmem_uint_atomic_or(unsigned int *dst, unsigned int val,
        int pe) {
    unsigned int curr_val;
    shmem_getmem(&curr_val, dst, sizeof(curr_val), pe);

    while (1) {
        unsigned int new_val = (curr_val | val);
        unsigned int old_val = SHMEM_UINT_CSWAP(dst, curr_val, new_val, pe);
        if (old_val == curr_val) return;
        curr_val = old_val;
    }
}

static void inline _shmem_uint_atomic_and(unsigned int *dst, unsigned int val,
        int pe) {
    unsigned int curr_val;
    shmem_getmem(&curr_val, dst, sizeof(curr_val), pe);

    while (1) {
        unsigned int new_val = (curr_val & val);
        unsigned int old_val = SHMEM_UINT_CSWAP(dst, curr_val, new_val, pe);
        if (old_val == curr_val) return;
        curr_val = old_val;
    }
}

#define SHMEM_ULONGLONG_ATOMIC_OR _shmem_ulonglong_atomic_or
#define SHMEM_UINT_ATOMIC_OR _shmem_uint_atomic_or
#define SHMEM_UINT_ATOMIC_AND _shmem_uint_atomic_or

#else

#define SHMEM_ULONGLONG_ATOMIC_OR shmemx_ulonglong_atomic_or
#define SHMEM_UINT_ATOMIC_OR shmemx_uint_atomic_or
#define SHMEM_UINT_ATOMIC_AND shmemx_uint_atomic_and

#endif
#endif

#define EDGE_GET_BUFFERING 4096

static inline void *shmem_malloc_wrapper(size_t nbytes) {
    static size_t total_nbytes = 0;
    if (nbytes == 0) {
        fprintf(stderr, "PE %d allocated %lu bytes. Exiting...\n",
                shmem_my_pe(), total_nbytes);
        exit(0);
    } else {
        void *ptr = shmem_malloc(nbytes);
        total_nbytes += nbytes;
        return ptr;
    }
}

static void flush_buffered_updates(hvr_internal_ctx_t *ctx) {
    for (int i = 0; i < ctx->npes; i++) {
        if (ctx->buffered_updates[i].len > 0) {
            hvr_mailbox_send(ctx->buffered_updates + i,
                    sizeof(hvr_vertex_update_t), i, -1,
                    &ctx->vertex_update_mailbox);
            ctx->buffered_updates[i].len = 0;
        }
        if (ctx->buffered_deletes[i].len > 0) {
            hvr_mailbox_send(ctx->buffered_deletes + i,
                    sizeof(hvr_vertex_update_t), i, -1,
                    &ctx->vertex_delete_mailbox);
            ctx->buffered_deletes[i].len = 0;
        }
    }
}

void hvr_ctx_create(hvr_ctx_t *out_ctx) {
    hvr_internal_ctx_t *new_ctx = (hvr_internal_ctx_t *)malloc(
            sizeof(*new_ctx));
    assert(new_ctx);
    memset(new_ctx, 0x00, sizeof(*new_ctx));

    new_ctx->pe = shmem_my_pe();
    new_ctx->npes = shmem_n_pes();
    /*
     * Reserve the top two bits of hvr_vertex_id/hvr_edge_info to store
     * information on which edge type is being stored.
     */
    assert(new_ctx->npes <= 0x3fffffffULL);

    size_t pool_nodes = 1024;
    if (getenv("HVR_SYMM_POOL_NNODES")) {
        pool_nodes = atoi(getenv("HVR_SYMM_POOL_NNODES"));
    }

    hvr_vertex_pool_create(get_symm_pool_nelements(), pool_nodes,
            &new_ctx->pool);
    new_ctx->vertex_partitions = (hvr_partition_t *)shmem_malloc(
            get_symm_pool_nelements() * sizeof(hvr_partition_t));

#ifdef VERBOSE
    int err = gethostname(new_ctx->my_hostname, 1024);
    assert(err == 0);

    printf("PE %d is on host %s.\n", new_ctx->pe, new_ctx->my_hostname);
#endif

    *out_ctx = new_ctx;
}

hvr_graph_id_t hvr_graph_create(hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;

    hvr_graph_id_t next_graph = (ctx->allocated_graphs)++;

    if (next_graph >= BITS_PER_BYTE * sizeof(hvr_graph_id_t)) {
        fprintf(stderr, "Ran out of graph IDs\n");
        abort();
    }

    return (1 << next_graph);
}

static hvr_partition_t wrap_actor_to_partition(hvr_vertex_t *vec,
        hvr_internal_ctx_t *ctx) {
    hvr_partition_t partition = ctx->actor_to_partition(vec, ctx);
    if (partition >= ctx->n_partitions) {
        char buf[1024];
        hvr_vertex_dump(vec, buf, 1024, ctx);

        fprintf(stderr, "Invalid partition %d (# partitions = %d) returned "
                "from actor_to_partition. Vector = {%s}\n", partition,
                ctx->n_partitions, buf);
        abort();
    }
    return partition;
}

static void update_vertex_partitions_for_vertex(hvr_vertex_t *curr,
        hvr_internal_ctx_t *ctx, hvr_vertex_t **partition_lists,
        uint8_t dist_from_local_vert) {
    assert(curr->id != HVR_INVALID_VERTEX_ID);
    hvr_partition_t partition = wrap_actor_to_partition(curr, ctx);

    // Prepend to appropriate partition list
    if (partition_lists[partition]) {
        curr->next_in_partition = partition_lists[partition];
        partition_lists[partition] = curr;
    } else {
        curr->next_in_partition = NULL;
        partition_lists[partition] = curr;
    }

    if (dist_from_local_vert <
            ctx->partition_min_dist_from_local_vert[partition]) {
        ctx->partition_min_dist_from_local_vert[partition] =
            dist_from_local_vert;
    }
}

/*
 * We also use this traversal to create lists of vertices in each partition
 * (partition_lists).
 */
static void update_actor_partitions(hvr_internal_ctx_t *ctx) {
    /*
     * Clear out existing partition lists TODO this might be wasteful to
     * recompute every iteration.
     */
    memset(ctx->local_partition_lists, 0x00,
            sizeof(hvr_vertex_t *) * ctx->n_partitions);
    memset(ctx->mirror_partition_lists, 0x00,
            sizeof(hvr_vertex_t *) * ctx->n_partitions);
    memset(ctx->partition_min_dist_from_local_vert, 0xff,
            sizeof(unsigned) * ctx->n_partitions);

    hvr_vertex_iter_t iter;
    hvr_vertex_iter_all_init(&iter, ctx);
    for (hvr_vertex_t *curr = hvr_vertex_iter_next(&iter); curr;
            curr = hvr_vertex_iter_next(&iter)) {
        update_vertex_partitions_for_vertex(curr, ctx,
                ctx->local_partition_lists, 0);
    }

    for (unsigned i = 0; i < HVR_MAP_BUCKETS; i++) {
        hvr_map_seg_t *seg = ctx->vec_cache.cache_map.buckets[i];
        while (seg) {
            for (unsigned j = 0; j < seg->nkeys; j++) {
                hvr_vertex_cache_node_t *node = seg->data[j].inline_vals[0].cached_vert;
                update_vertex_partitions_for_vertex(&node->vert, ctx,
                        ctx->mirror_partition_lists,
                        get_dist_from_local_vert(node, &ctx->vec_cache, ctx->pe));
            }
            seg = seg->next;
        }
    }
}

static inline void update_partition_info(hvr_partition_t p,
        hvr_set_t *local_new, hvr_set_t *local_old,
        hvr_dist_bitvec_t *registry, hvr_internal_ctx_t *ctx) {
    if (hvr_set_contains(p, local_new) != hvr_set_contains(p, local_old)) {
        /*
         * If a change in active partition set occurred, we notify any
         * partitions that p might interact with.
         */
        if (hvr_set_contains(p, local_new)) {
            // Became active
            hvr_dist_bitvec_set(p, ctx->pe, registry);
        } else {
            // Went inactive
            hvr_dist_bitvec_clear(p, ctx->pe, registry);
        }
    }
}

static void pull_vertices_from_dead_pe(int dead_pe, hvr_partition_t partition,
        hvr_internal_ctx_t *ctx) {
    hvr_vertex_t tmp_vert;

    /*
     * Fetch the vertices from PE 'dead_pe' in chunks, looking for vertices
     * in 'partition' and adding them to our local vec cache.
     */
    const size_t pool_size = ctx->pool.tracker.capacity;
    for (unsigned i = 0; i < pool_size; i += N_VERTICES_PER_BUF) {
        unsigned n_this_iter = pool_size - i;
        if (n_this_iter > N_VERTICES_PER_BUF) {
            n_this_iter = N_VERTICES_PER_BUF;
        }

        shmem_getmem(ctx->vert_partition_buf, ctx->vertex_partitions + i,
                n_this_iter * sizeof(ctx->vert_partition_buf[0]), dead_pe);

        for (unsigned j = 0; j < n_this_iter; j++) {
            if (ctx->vert_partition_buf[j] == partition) {
                shmem_getmem(&tmp_vert, ctx->pool.pool + (i + j),
                        sizeof(tmp_vert), dead_pe);

                unsigned long long unused; unsigned u_unused;
                handle_new_vertex(&tmp_vert, &unused, &unused, &unused,
                        &u_unused, &unused, ctx);
            }
        }
    }
}

/*
 * partition_time_window stores a set of the partitions that the local PE has
 * vertices inside of. This updates the partitions in that window set based on
 * the results of update_actor_partitions.
 */
static void update_partition_time_window(hvr_internal_ctx_t *ctx,
        unsigned long long *out_time_updating_partitions,
        unsigned long long *out_time_updating_producers,
        unsigned long long *out_time_updating_subscribers,
        unsigned long long *out_time_spent_processing_dead_pes) {
    unsigned long long time_spent_processing_dead_pes = 0;
    const unsigned long long start = hvr_current_time_us();

    hvr_set_wipe(ctx->tmp_subscriber_partition_time_window);
    hvr_set_wipe(ctx->tmp_producer_partition_time_window);

    // Update the set of partitions in a temporary buffer
    for (unsigned p = 0; p < ctx->n_partitions; p++) {
        /*
         * Update the global registry with which partitions this PE provides
         * updates on (producer) and which partitions it subscribes to updates
         * from (subscriber).
         */
        if (ctx->local_partition_lists[p]) {
            // Producer
            hvr_set_insert(p, ctx->tmp_producer_partition_time_window);
        }

        if ((ctx->local_partition_lists[p] || ctx->mirror_partition_lists[p])
                && ctx->partition_min_dist_from_local_vert[p] <=
                    ctx->max_graph_traverse_depth - 1) {
            // Subscriber to any interacting partitions with p
            hvr_partition_t interacting[MAX_INTERACTING_PARTITIONS];
            unsigned n_interacting;
            ctx->might_interact(p, interacting, &n_interacting,
                    MAX_INTERACTING_PARTITIONS, ctx);

            /*
             * Mark any partitions which this locally active partition might
             * interact with.
             */
            for (unsigned i = 0; i < n_interacting; i++) {
                hvr_set_insert(interacting[i],
                        ctx->tmp_subscriber_partition_time_window);
            }
        }
    }

    const unsigned long long after_part_updates = hvr_current_time_us();

    /*
     * For all partitions, check if our interest in them has changed on this
     * iteration. If it has, notify the global registry of that.
     */
    for (hvr_partition_t p = 0; p < ctx->n_partitions; p++) {
        update_partition_info(p, ctx->tmp_producer_partition_time_window,
                ctx->producer_partition_time_window,
                &ctx->partition_producers, ctx);
    }

    const unsigned long long after_producer_updates = hvr_current_time_us();

    // For each partition
    for (hvr_partition_t p = 0; p < ctx->n_partitions; p++) {
        const int old_sub = hvr_set_contains(p,
                ctx->subscriber_partition_time_window);
        if (hvr_set_contains(p, ctx->tmp_subscriber_partition_time_window)) {
            if (!old_sub) {
                // If this is a new subscription on this iteration:
                hvr_dist_bitvec_local_subcopy_init(&ctx->partition_producers,
                        ctx->producer_info + p);
                hvr_dist_bitvec_local_subcopy_init(&ctx->terminated_pes,
                        ctx->dead_info + p);

                // Download the list of producers for partition p.
                hvr_dist_bitvec_copy_locally(p, &ctx->partition_producers,
                        ctx->producer_info + p);

                /*
                 * notify all producers of this partition of our subscription
                 * (they will then send us a full update).
                 */
                hvr_partition_member_change_t change;
                change.pe = shmem_my_pe(); // The subscriber/unsubscriber
                change.partition = p;      // The partition (un)subscribed to
                change.entered = 1;        // new subscription
                for (int pe = 0; pe < ctx->npes; pe++) {
                    if (hvr_dist_bitvec_local_subcopy_contains(pe,
                                ctx->producer_info + p)) {
                        hvr_mailbox_send(&change, sizeof(change), pe,
                                -1, &ctx->forward_mailbox);
                    }
                }

                if (dead_pe_processing) {
                    const unsigned long long start = hvr_current_time_us();
                    hvr_dist_bitvec_copy_locally(p, &ctx->terminated_pes,
                            ctx->dead_info + p);
                    for (int pe = 0; pe < ctx->npes; pe++) {
                        if (hvr_dist_bitvec_local_subcopy_contains(pe,
                                    ctx->dead_info + p)) {
                            pull_vertices_from_dead_pe(pe, p, ctx);
                        }
                    }
                    time_spent_processing_dead_pes += (hvr_current_time_us() - start);
                }
            } else {
                /*
                 * If this is an existing subscription, copy down the current
                 * list of producers and check for changes.
                 *
                 * If a change is found, notify the new producer that we're 
                 * subscribed. TODO note that we're not doing anything when a
                 * producer leaves a partition, which doesn't matter
                 * semantically but could waste memory.
                 */

                if (ctx->iter % check_producers_freq == 0) {
                    hvr_dist_bitvec_copy_locally(p, &ctx->partition_producers,
                            &ctx->tmp_local_partition_membership);

                    hvr_partition_member_change_t change;
                    change.pe = shmem_my_pe(); // The subscriber/unsubscriber
                    change.partition = p;      // The partition (un)subscribed to
                    change.entered = 1;        // new subscription

                    /*
                     * Notify the new producer of this partition that I'm interested
                     * in their updates.
                     */
                    for (int pe = 0; pe < ctx->npes; pe++) {
                        if (!hvr_dist_bitvec_local_subcopy_contains(pe,
                                    ctx->producer_info + p) &&
                                hvr_dist_bitvec_local_subcopy_contains(pe,
                                    &ctx->tmp_local_partition_membership)) {
                            // New producer
                            hvr_mailbox_send(&change, sizeof(change), pe,
                                    -1, &ctx->forward_mailbox);
                        }
                    }

                    // Save for later
                    hvr_dist_bitvec_local_subcopy_copy(ctx->producer_info + p,
                            &ctx->tmp_local_partition_membership);

                    /*
                     * Look for newly terminated PEs that had local vertices in a
                     * given partition. Go grab their vertices if there's a new one.
                     */
                    hvr_dist_bitvec_copy_locally(p, &ctx->terminated_pes,
                            &ctx->tmp_local_partition_membership);

                    if (dead_pe_processing) {
                        const unsigned long long start = hvr_current_time_us();
                        for (int pe = 0; pe < ctx->npes; pe++) {
                            if (!hvr_dist_bitvec_local_subcopy_contains(pe,
                                        ctx->dead_info + p) &&
                                    hvr_dist_bitvec_local_subcopy_contains(pe,
                                        &ctx->tmp_local_partition_membership)) {
                                // New dead PE
                                pull_vertices_from_dead_pe(pe, p, ctx);
                            }
                        }
                        time_spent_processing_dead_pes += (hvr_current_time_us() -
                                start);
                    }

                    hvr_dist_bitvec_local_subcopy_copy(ctx->dead_info + p,
                            &ctx->tmp_local_partition_membership);
                }
            }
        } else {
            // If we are not subscribed to partition p
            if (old_sub) {
                /*
                 * If this is a new unsubscription notify all producers that we
                 * are no longer subscribed. They will remove us from the list
                 * of people they send msgs to.
                 */
                hvr_dist_bitvec_copy_locally(p, &ctx->partition_producers,
                        ctx->producer_info + p);

                hvr_partition_member_change_t change;
                change.pe = shmem_my_pe(); // The subscriber/unsubscriber
                change.partition = p;      // The partition (un)subscribed to
                change.entered = 0;        // unsubscription
                for (int pe = 0; pe < ctx->npes; pe++) {
                    if (hvr_dist_bitvec_local_subcopy_contains(pe,
                                ctx->producer_info + p)) {
                        hvr_mailbox_send(&change, sizeof(change), pe,
                                -1, &ctx->forward_mailbox);
                    }
                }

                hvr_dist_bitvec_local_subcopy_destroy(ctx->producer_info + p);
                hvr_dist_bitvec_local_subcopy_destroy(ctx->dead_info + p);
            }
        }
    }

    /*
     * Copy the newly computed partition windows for this PE over for next time
     * when we need to check for changes.
     */
    hvr_set_copy(ctx->subscriber_partition_time_window,
            ctx->tmp_subscriber_partition_time_window);

    hvr_set_copy(ctx->producer_partition_time_window,
            ctx->tmp_producer_partition_time_window);

    const unsigned long long end = hvr_current_time_us();

    *out_time_spent_processing_dead_pes = time_spent_processing_dead_pes;
    *out_time_updating_partitions = after_part_updates - start;
    *out_time_updating_producers = after_producer_updates - after_part_updates;
    *out_time_updating_subscribers = end - after_producer_updates;
}

void hvr_init(const hvr_partition_t n_partitions,
        hvr_update_metadata_func update_metadata,
        hvr_might_interact_func might_interact,
        hvr_update_coupled_val_func update_coupled_val,
        hvr_actor_to_partition actor_to_partition,
        hvr_start_time_step start_time_step,
        hvr_should_have_edge should_have_edge,
        hvr_should_terminate_func should_terminate,
        unsigned long long max_elapsed_seconds,
        unsigned max_graph_traverse_depth,
        hvr_ctx_t in_ctx) {
    assert(max_graph_traverse_depth < UINT8_MAX);

    hvr_internal_ctx_t *new_ctx = (hvr_internal_ctx_t *)in_ctx;

    assert(new_ctx->initialized == 0);
    new_ctx->initialized = 1;

    assert(max_graph_traverse_depth >= 1);

    new_ctx->p_wrk = (long long *)shmem_malloc_wrapper(
            SHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(long long));
    new_ctx->p_wrk_int = (int *)shmem_malloc_wrapper(
            SHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(int));
    new_ctx->p_sync = (long *)shmem_malloc_wrapper(
            SHMEM_REDUCE_SYNC_SIZE * sizeof(long));
    assert(new_ctx->p_wrk && new_ctx->p_sync && new_ctx->p_wrk_int);

    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        (new_ctx->p_sync)[i] = SHMEM_SYNC_VALUE;
    }

    new_ctx->subscriber_partition_time_window = hvr_create_empty_set(
            n_partitions);
    new_ctx->producer_partition_time_window = hvr_create_empty_set(
            n_partitions);
    new_ctx->tmp_subscriber_partition_time_window = hvr_create_empty_set(
            n_partitions);
    new_ctx->tmp_producer_partition_time_window = hvr_create_empty_set(
            n_partitions);

    assert(n_partitions <= HVR_INVALID_PARTITION);
    new_ctx->n_partitions = n_partitions;

    new_ctx->update_metadata = update_metadata;
    new_ctx->might_interact = might_interact;
    new_ctx->update_coupled_val = update_coupled_val;
    new_ctx->actor_to_partition = actor_to_partition;
    new_ctx->start_time_step = start_time_step;
    new_ctx->should_have_edge = should_have_edge;
    new_ctx->should_terminate = should_terminate;

    new_ctx->max_elapsed_seconds = max_elapsed_seconds;

    if (getenv("HVR_STRICT")) {
        if (new_ctx->pe == 0) {
            fprintf(stderr, "WARNING: Running in strict mode, this will lead "
                    "to degraded performance.\n");
        }
        new_ctx->strict_mode = 1;
        new_ctx->strict_counter_src = (int *)shmem_malloc_wrapper(sizeof(int));
        new_ctx->strict_counter_dest = (int *)shmem_malloc_wrapper(sizeof(int));
        assert(new_ctx->strict_counter_src && new_ctx->strict_counter_dest);
    }

    if (getenv("HVR_TRACE_DUMP")) {
        new_ctx->dump_mode = 1;

        char dump_file_name[256];
        sprintf(dump_file_name, "%d.csv", new_ctx->pe);
        new_ctx->dump_file = fopen(dump_file_name, "w");
        assert(new_ctx->dump_file);

        sprintf(dump_file_name, "%d.edges.csv", new_ctx->pe);
        new_ctx->edges_dump_file = fopen(dump_file_name, "w");
        assert(new_ctx->edges_dump_file);

        if (getenv("HVR_TRACE_DUMP_ONLY_LAST")) {
            new_ctx->only_last_iter_dump = 1;
        }
    }

    if (getenv("HVR_DISABLE_PROFILING_PRINTS")) {
        print_profiling = 0;
    } else {
        char profiling_filename[1024];
        sprintf(profiling_filename, "%d.prof", new_ctx->pe);
        profiling_fp = fopen(profiling_filename, "w");
        assert(profiling_fp);
    }

    if (getenv("HVR_DISABLE_DEAD_PE_PROCESSING")) {
        dead_pe_processing = 0;
    }

    new_ctx->prev_coupled_pes = hvr_create_empty_set(new_ctx->npes);
    new_ctx->coupled_pes = hvr_create_empty_set(new_ctx->npes);
    new_ctx->coupled_pes_received_from = hvr_create_empty_set(new_ctx->npes);
    hvr_set_insert(new_ctx->pe, new_ctx->coupled_pes);

    new_ctx->coupled_pes_values = (hvr_vertex_t *)malloc(
            new_ctx->npes * sizeof(hvr_vertex_t));
    assert(new_ctx->coupled_pes_values);
    for (unsigned i = 0; i < new_ctx->npes; i++) {
        hvr_vertex_init(&(new_ctx->coupled_pes_values)[i], new_ctx);
    }
    new_ctx->updates_on_this_iter = (int *)malloc(
            new_ctx->npes * sizeof(new_ctx->updates_on_this_iter[0]));
    assert(new_ctx->updates_on_this_iter);

    new_ctx->partitions_per_pe = (new_ctx->n_partitions + new_ctx->npes - 1) /
        new_ctx->npes;
    new_ctx->partitions_per_pe_vec_length_in_words =
        (new_ctx->npes + BITS_PER_WORD - 1) / BITS_PER_WORD;
    new_ctx->pes_per_partition = (unsigned *)shmem_malloc_wrapper(
            new_ctx->partitions_per_pe *
            new_ctx->partitions_per_pe_vec_length_in_words * sizeof(unsigned));
    assert(new_ctx->pes_per_partition);
    memset(new_ctx->pes_per_partition, 0x00, new_ctx->partitions_per_pe *
            new_ctx->partitions_per_pe_vec_length_in_words * sizeof(unsigned));

    new_ctx->local_pes_per_partition_buffer = (unsigned *)malloc(
            new_ctx->partitions_per_pe_vec_length_in_words * sizeof(unsigned));
    assert(new_ctx->local_pes_per_partition_buffer);

    new_ctx->local_partition_lists = (hvr_vertex_t **)malloc(
            sizeof(hvr_vertex_t *) * new_ctx->n_partitions);
    assert(new_ctx->local_partition_lists);

    new_ctx->mirror_partition_lists = (hvr_vertex_t **)malloc(
            sizeof(hvr_vertex_t *) * new_ctx->n_partitions);
    assert(new_ctx->mirror_partition_lists);

    new_ctx->partition_min_dist_from_local_vert = (unsigned *)malloc(
            sizeof(unsigned) * new_ctx->n_partitions);
    assert(new_ctx->partition_min_dist_from_local_vert);

    hvr_vertex_cache_init(&new_ctx->vec_cache, new_ctx->n_partitions);

    hvr_mailbox_init(&new_ctx->vertex_update_mailbox, 256 * 1024 * 1024);
    hvr_mailbox_init(&new_ctx->vertex_delete_mailbox, 256 * 1024 * 1024);
    hvr_mailbox_init(&new_ctx->forward_mailbox,       128 * 1024 * 1024);
    hvr_mailbox_init(&new_ctx->coupling_mailbox,      32 * 1024 * 1024);

    hvr_dist_bitvec_init(new_ctx->n_partitions, new_ctx->npes,
            &new_ctx->partition_producers);
    hvr_dist_bitvec_init(new_ctx->n_partitions, new_ctx->npes,
            &new_ctx->terminated_pes);

    hvr_dist_bitvec_local_subcopy_init(
            &new_ctx->partition_producers,
            &new_ctx->tmp_local_partition_membership);

    hvr_sparse_arr_init(&new_ctx->pe_subscription_info, new_ctx->n_partitions);

    new_ctx->max_graph_traverse_depth = max_graph_traverse_depth;

    new_ctx->producer_info = (hvr_dist_bitvec_local_subcopy_t *)malloc(
            new_ctx->n_partitions * sizeof(hvr_dist_bitvec_local_subcopy_t));
    assert(new_ctx->producer_info);
    new_ctx->dead_info = (hvr_dist_bitvec_local_subcopy_t *)malloc(
            new_ctx->n_partitions * sizeof(hvr_dist_bitvec_local_subcopy_t));
    assert(new_ctx->dead_info);

    size_t max_msg_len = sizeof(hvr_vertex_update_t);
    if (sizeof(hvr_partition_member_change_t) > max_msg_len) {
        max_msg_len = sizeof(hvr_partition_member_change_t);
    }
    if (sizeof(hvr_dead_pe_msg_t) > max_msg_len) {
        max_msg_len = sizeof(hvr_dead_pe_msg_t);
    }
    if (sizeof(hvr_coupling_msg_t) > max_msg_len) {
        max_msg_len = sizeof(hvr_coupling_msg_t);
    }
    new_ctx->msg_buf = malloc(max_msg_len);
    assert(new_ctx->msg_buf);
    new_ctx->msg_buf_capacity = max_msg_len;

    new_ctx->buffered_updates = (hvr_vertex_update_t *)malloc(
            new_ctx->npes * sizeof(hvr_vertex_update_t));
    assert(new_ctx->buffered_updates);
    memset(new_ctx->buffered_updates, 0x00,
            new_ctx->npes * sizeof(hvr_vertex_update_t));

    new_ctx->buffered_deletes = (hvr_vertex_update_t *)malloc(
            new_ctx->npes * sizeof(hvr_vertex_update_t));
    assert(new_ctx->buffered_deletes);
    memset(new_ctx->buffered_deletes, 0x00,
            new_ctx->npes * sizeof(hvr_vertex_update_t));

    new_ctx->vert_partition_buf = (hvr_partition_t *)malloc(
            N_VERTICES_PER_BUF * sizeof(hvr_partition_t));
    assert(new_ctx->vert_partition_buf);

    // Print the number of bytes allocated
    // shmem_malloc_wrapper(0);
    shmem_barrier_all();
}

static void *aborting_thread(void *user_data) {
    int nseconds = atoi(getenv("HVR_HANG_ABORT"));
    assert(nseconds > 0);

    const unsigned long long start = hvr_current_time_us();
    while (hvr_current_time_us() - start < nseconds * 1000000) {
        sleep(10);
    }

    if (!this_pe_has_exited) {
        fprintf(stderr, "ERROR: HOOVER forcibly aborting because "
                "HVR_HANG_ABORT was set.\n");
        abort(); // Get a core dump
    }
    return NULL;
}

static void process_neighbor_updates(hvr_internal_ctx_t *ctx,
        process_perf_info_t *perf_info) {
    size_t msg_len;
    int success = hvr_mailbox_recv(&ctx->msg_buf, &ctx->msg_buf_capacity,
            &msg_len, &ctx->forward_mailbox);
    while (success) {
        assert(msg_len == sizeof(hvr_partition_member_change_t));
        /*
         * Tells that a given PE has subscribed/unsubscribed to updates for a
         * given partition for which we are a producer for.
         */
        hvr_partition_member_change_t *change =
            (hvr_partition_member_change_t *)ctx->msg_buf;

        if (change->entered) {
            // Entered partition

            if (!hvr_sparse_arr_contains(change->partition, change->pe,
                        &ctx->pe_subscription_info)) {
                /*
                 * This is a new subscription from this PE for this
                 * partition. As we go through this, if we find new
                 * subscriptions, we need to transmit all vertex information we
                 * have for that partition to the PE.
                 */
                hvr_vertex_update_t msg;
                msg.len = 0;

                hvr_vertex_t *iter = ctx->local_partition_lists[
                    change->partition];
                while (iter) {
                    memcpy(&(msg.verts[msg.len]), iter, sizeof(*iter));
                    msg.len += 1;

                    if (msg.len == VERT_PER_UPDATE ||
                            (iter->next_in_partition == NULL && msg.len > 0)) {
                        int success = hvr_mailbox_send(&msg, sizeof(msg),
                                change->pe, N_SEND_ATTEMPTS,
                                &ctx->vertex_update_mailbox);
                        while (!success) {
                            perf_info->n_received_updates +=
                                process_vertex_updates(ctx, perf_info);
                            success = hvr_mailbox_send(&msg, sizeof(msg),
                                    change->pe, N_SEND_ATTEMPTS,
                                    &ctx->vertex_update_mailbox);
                        }
                        msg.len = 0;
                    }
                    iter = iter->next_in_partition;
                }
                hvr_sparse_arr_insert(change->partition, change->pe,
                        &ctx->pe_subscription_info);
            }
        } else {
            /*
             * Left partition. It is possible that we receive a left partition
             * message from a PE that we did not know about yet if the following
             * happens in quick succession:
             *   1. This PE becomes a producer of the partition, updating the
             *      global directory.
             *   2. The remote PE stops subscribing to this partition, before
             *      looping around to update_partition_time_window to realize
             *      this PE's change in status and send it a subscription msg.
             *   3. The remote PE loops around to update_partition_time_window
             *      and sends out notifications to all producers that it is
             *      leaving the partition.
             */
            if (hvr_sparse_arr_contains(change->partition, change->pe,
                        &ctx->pe_subscription_info)) {
                hvr_sparse_arr_remove(change->partition, change->pe,
                        &ctx->pe_subscription_info);
            }
        }
        success = hvr_mailbox_recv(&ctx->msg_buf, &ctx->msg_buf_capacity,
                &msg_len, &ctx->forward_mailbox);
    }
}

static inline int is_local_neighbor(hvr_vertex_id_t id,
        hvr_internal_ctx_t *ctx) {
    hvr_map_val_list_t neighbors;
    int n_neighbors = hvr_map_linearize(id, &ctx->edges.map,
            &neighbors);

    for (int n = 0; n < n_neighbors; n++) {
        hvr_edge_info_t edge = hvr_map_val_list_get(n, &neighbors).edge_info;
        hvr_vertex_id_t neighbor = EDGE_INFO_VERTEX(edge);
        if (VERTEX_ID_PE(neighbor) == ctx->pe) {
            return 1;
        }
    }

    return 0;
}

// The only place where edges between vertices are created/deleted
static inline void update_edge_info(hvr_vertex_id_t base_id,
        hvr_vertex_id_t neighbor_id,
        hvr_vertex_cache_node_t *base,
        hvr_vertex_cache_node_t *neighbor,
        hvr_edge_type_t new_edge, hvr_edge_type_t existing_edge,
        hvr_internal_ctx_t *ctx) {
    assert(new_edge != existing_edge);
    const int base_is_local = (VERTEX_ID_PE(base_id) == ctx->pe);
    const int neighbor_is_local = (VERTEX_ID_PE(neighbor_id) == ctx->pe);
    neighbor = (neighbor ? neighbor :
            hvr_vertex_cache_lookup(neighbor_id, &ctx->vec_cache));
    base = (base ? base :
            hvr_vertex_cache_lookup(base_id, &ctx->vec_cache));

    if (existing_edge != NO_EDGE) {
        hvr_remove_edge(base_id, neighbor_id, &ctx->edges);
        hvr_remove_edge(neighbor_id, base_id, &ctx->edges);

        // Decrement if condition holds tru
        base->n_local_neighbors -= neighbor_is_local;
        neighbor->n_local_neighbors -= base_is_local;
    }

    if (new_edge != NO_EDGE) {
        /*
         * Removing and then adding is less efficient than simply updating the
         * edge, but allows us to make more assertions that are helpful for
         * debugging.
         */
        hvr_add_edge(base_id, neighbor_id, new_edge, &ctx->edges);
        hvr_add_edge(neighbor_id, base_id, flip_edge_direction(new_edge),
                &ctx->edges);

        base->n_local_neighbors += neighbor_is_local;
        neighbor->n_local_neighbors += base_is_local;
    }

    /*
     * If this edge update involves one local vertex and one remote, the
     * remote may require a change in local neighbor list membership (either
     * addition or deletion).
     */
    if (base_is_local != neighbor_is_local &&
            (base_is_local || neighbor_is_local)) {
        hvr_vertex_cache_node_t *remote_node = (base_is_local ? neighbor : base);
        hvr_vertex_id_t remote = remote_node->vert.id;

        if (local_neighbor_list_contains(remote_node, &ctx->vec_cache)) {
            if (remote_node->n_local_neighbors == 0) {
                // Remove
                hvr_vertex_cache_remove_from_local_neighbor_list(remote_node,
                        &ctx->vec_cache);
            }
        } else {
            if (remote_node->n_local_neighbors > 0) {
                // Add
                hvr_vertex_cache_add_to_local_neighbor_list(remote_node,
                    &ctx->vec_cache);
            }
        }
    }

    /*
     * Only needs updating if this is an edge inbound in a given vertex (either
     * directed in or bidirectional).
     */
    if (base_is_local && new_edge != DIRECTED_OUT) {
        hvr_vertex_t *local = ctx->pool.pool + VERTEX_ID_OFFSET(base_id);
        local->needs_processing = 1;
    }

    if (neighbor_is_local && flip_edge_direction(new_edge) != DIRECTED_OUT) {
        hvr_vertex_t *local = ctx->pool.pool + VERTEX_ID_OFFSET(neighbor_id);
        local->needs_processing = 1;
    }
}

static int create_new_edges_helper(hvr_vertex_cache_node_t *vert,
        hvr_vertex_cache_node_t *updated_vert, hvr_internal_ctx_t *ctx) {
    hvr_edge_type_t edge = ctx->should_have_edge(&vert->vert,
            &updated_vert->vert, ctx);
    if (edge == NO_EDGE) {
        return 0;
    } else {
        update_edge_info(vert->vert.id, updated_vert->vert.id, vert,
                updated_vert, edge, NO_EDGE, ctx);
        return 1;
    }
}

/*
* Figure out what edges need to be added here, from should_have_edge and then
* insert them for the new vertex. Eventually, any local vertex which had a new
* edge inserted will need to be updated.
*/
static void create_new_edges(hvr_vertex_cache_node_t *updated,
        hvr_partition_t *interacting, unsigned n_interacting,
        hvr_internal_ctx_t *ctx,
        unsigned *count_new_should_have_edges) {
    unsigned local_count_new_should_have_edges = 0;

    for (unsigned i = 0; i < n_interacting; i++) {
        hvr_partition_t other_part = interacting[i];

        hvr_vertex_cache_node_t *cache_iter =
            ctx->vec_cache.partitions[other_part];
        while (cache_iter) {
            int edge_created = create_new_edges_helper(cache_iter, updated,
                    ctx);
            cache_iter = cache_iter->part_next;
            local_count_new_should_have_edges++;
        }
    }

    *count_new_should_have_edges += local_count_new_should_have_edges;
}

/*
 * When a vertex is deleted, we simply need to remove all of its
 * edges with local vertices and remove it from the cache.
 */
static void handle_deleted_vertex(hvr_vertex_t *dead_vert,
        hvr_internal_ctx_t *ctx) {
    static hvr_edge_info_t *edges_to_delete = NULL;
    static unsigned edges_to_delete_capacity = 0;

    hvr_vertex_cache_node_t *cached = hvr_vertex_cache_lookup(dead_vert->id,
            &ctx->vec_cache);

    // If we were caching this node, delete the mirrored version
    if (cached) {
        hvr_map_val_list_t neighbors;
        int n_neighbors = hvr_map_linearize(dead_vert->id, &ctx->edges.map,
                &neighbors);

        if (n_neighbors > 0 && edges_to_delete_capacity < n_neighbors) {
            edges_to_delete = (hvr_vertex_id_t *)realloc(edges_to_delete,
                    n_neighbors * sizeof(*edges_to_delete));
            assert(edges_to_delete);
            edges_to_delete_capacity = n_neighbors;
        }

        /*
         * Have to do this in two stages to avoid concurrently modifying the
         * neighbor list while iterating over it.
         */
        for (int n = 0; n < n_neighbors; n++) {
            edges_to_delete[n] = hvr_map_val_list_get(n, &neighbors).edge_info;
        }

        for (int n = 0; n < n_neighbors; n++) {
            hvr_vertex_id_t neighbor = EDGE_INFO_VERTEX(edges_to_delete[n]);
            update_edge_info(dead_vert->id, neighbor, cached, NULL, NO_EDGE,
                    EDGE_INFO_EDGE(edges_to_delete[n]), ctx);
        }

        hvr_vertex_cache_delete(dead_vert, &ctx->vec_cache);
    }
}

static hvr_vertex_cache_node_t *handle_new_vertex(hvr_vertex_t *new_vert,
        unsigned long long *time_updating,
        unsigned long long *time_updating_edges,
        unsigned long long *time_creating_edges,
        unsigned *count_new_should_have_edges,
        unsigned long long *time_creating,
        hvr_internal_ctx_t *ctx) {
    static hvr_edge_info_t *edges_to_update = NULL;
    static hvr_edge_type_t *new_edge_type = NULL;
    static unsigned edges_to_update_capacity = 0;

    hvr_partition_t partition = wrap_actor_to_partition(new_vert, ctx);
    hvr_vertex_id_t updated_vert_id = new_vert->id;

    hvr_partition_t interacting[MAX_INTERACTING_PARTITIONS];
    unsigned n_interacting;
    ctx->might_interact(partition, interacting, &n_interacting,
            MAX_INTERACTING_PARTITIONS, ctx);

    hvr_vertex_cache_node_t *updated = hvr_vertex_cache_lookup(
            updated_vert_id, &ctx->vec_cache);
    /*
     * If this is a vertex we already know about then we have
     * existing local edges that might need updating.
     */
    if (updated) {
        const unsigned long long start_update = hvr_current_time_us();
        /*
         * Update our local mirror with the information received in the
         * message.
         */
        memcpy(&updated->vert, new_vert, sizeof(*new_vert));

        /*
         * Look for existing edges and verify they should still exist with
         * this update to the local mirror.
         */
        hvr_map_val_list_t neighbors;
        int n_neighbors = hvr_map_linearize(updated_vert_id, &ctx->edges.map,
                &neighbors);

        if (n_neighbors > 0 && edges_to_update_capacity < n_neighbors) {
            edges_to_update = (hvr_edge_info_t *)realloc(edges_to_update,
                    n_neighbors * sizeof(*edges_to_update));
            assert(edges_to_update);
            new_edge_type = (hvr_edge_type_t *)realloc(new_edge_type,
                    n_neighbors * sizeof(*new_edge_type));
            edges_to_update_capacity = n_neighbors;
        }

        unsigned edges_to_update_len = 0;
        for (int n = 0; n < n_neighbors; n++) {
            hvr_edge_info_t edge_info = hvr_map_val_list_get(n,
                    &neighbors).edge_info;
            hvr_vertex_cache_node_t *cached_neighbor =
                hvr_vertex_cache_lookup(EDGE_INFO_VERTEX(edge_info),
                        &ctx->vec_cache);
            if (!cached_neighbor) {
                hvr_map_val_list_t vals_list;
                int n = hvr_map_linearize(EDGE_INFO_VERTEX(edge_info),
                        &ctx->vec_cache.cache_map,
                        &vals_list);
                hvr_vertex_id_t vert = EDGE_INFO_VERTEX(edge_info);
                fprintf(stderr, "PE %d Failed fetching vert for (%lu,%lu) n=%d\n",
                        ctx->pe, VERTEX_ID_PE(vert), VERTEX_ID_OFFSET(vert),
                        n);
                abort();
            }

            assert(cached_neighbor);

            // Check this edge should still exist
            hvr_edge_type_t new_edge = ctx->should_have_edge(
                    &updated->vert, &(cached_neighbor->vert), ctx);
            if (new_edge != EDGE_INFO_EDGE(edge_info)) {
                edges_to_update[edges_to_update_len] = edge_info;
                new_edge_type[edges_to_update_len++] = new_edge;
            }
        }

        for (unsigned i = 0; i < edges_to_update_len; i++) {
            update_edge_info(
                    updated->vert.id,
                    EDGE_INFO_VERTEX(edges_to_update[i]),
                    updated,
                    NULL,
                    new_edge_type[i],
                    EDGE_INFO_EDGE(edges_to_update[i]), ctx);
        }

        const unsigned long long done_updating_edges = hvr_current_time_us();

        create_new_edges(updated, interacting, n_interacting, ctx,
                count_new_should_have_edges);

        const unsigned long long done = hvr_current_time_us();

        *time_updating += (done - start_update);
        *time_updating_edges += (done_updating_edges - start_update);
        *time_creating_edges += (done - done_updating_edges);
    } else {
        const unsigned long long start_new = hvr_current_time_us();
        // A brand new vertex, or at least this is our first update on it
        updated = hvr_vertex_cache_add(new_vert, partition, &ctx->vec_cache);
        create_new_edges(updated, interacting, n_interacting, ctx,
                count_new_should_have_edges);
        *time_creating += (hvr_current_time_us() - start_new);
    }

    return updated;
}

static unsigned process_vertex_updates(hvr_internal_ctx_t *ctx,
        process_perf_info_t *perf_info) {
    unsigned n_updates = 0;
    size_t msg_len;

    const unsigned long long start = hvr_current_time_us();
    // Handle deletes, then updates
    int success = hvr_mailbox_recv(&ctx->msg_buf, &ctx->msg_buf_capacity,
            &msg_len, &ctx->vertex_delete_mailbox);
    while (success) {
        assert(msg_len == sizeof(hvr_vertex_update_t));
        hvr_vertex_update_t *msg = (hvr_vertex_update_t *)ctx->msg_buf;

        for (unsigned i = 0; i < msg->len; i++) {
            handle_deleted_vertex(&(msg->verts[i]), ctx);
            n_updates++;
        }

        success = hvr_mailbox_recv(&ctx->msg_buf, &ctx->msg_buf_capacity,
                &msg_len, &ctx->vertex_delete_mailbox);
    }

    const unsigned long long midpoint = hvr_current_time_us();
    success = hvr_mailbox_recv(&ctx->msg_buf, &ctx->msg_buf_capacity,
            &msg_len, &ctx->vertex_update_mailbox);
    while (success) {
        assert(msg_len == sizeof(hvr_vertex_update_t));
        hvr_vertex_update_t *msg = (hvr_vertex_update_t *)ctx->msg_buf;

        for (unsigned i = 0; i < msg->len; i++) {
            handle_new_vertex(&(msg->verts[i]),
                    &perf_info->time_updating,
                    &perf_info->time_updating_edges,
                    &perf_info->time_creating_edges,
                    &perf_info->count_new_should_have_edges,
                    &perf_info->time_creating,
                    ctx);
            n_updates++;
        }

        success = hvr_mailbox_recv(&ctx->msg_buf, &ctx->msg_buf_capacity,
                &msg_len, &ctx->vertex_update_mailbox);
    }

    const unsigned long long done = hvr_current_time_us();

    perf_info->time_handling_deletes += midpoint - start;
    perf_info->time_handling_news += done - midpoint;

    return n_updates;
}

static hvr_vertex_cache_node_t *add_neighbors_to_q(
        hvr_vertex_cache_node_t *node,
        hvr_vertex_cache_node_t *newq,
        hvr_internal_ctx_t *ctx) {
    hvr_map_val_list_t neighbors;
    int n_neighbors = hvr_map_linearize(node->vert.id, &ctx->edges.map,
            &neighbors);

    for (int n = 0; n < n_neighbors; n++) {
        hvr_edge_info_t edge_info = hvr_map_val_list_get(n,
                &neighbors).edge_info;
        hvr_vertex_id_t vert = EDGE_INFO_VERTEX(edge_info);
        hvr_vertex_cache_node_t *cached_neighbor =
            hvr_vertex_cache_lookup(vert, &ctx->vec_cache);
        if (!cached_neighbor) {
            hvr_map_val_list_t vals_list;
            int n = hvr_map_linearize(EDGE_INFO_VERTEX(edge_info),
                    &ctx->vec_cache.cache_map,
                    &vals_list);
            fprintf(stderr, "PE %d Failed fetching vert for (%lu,%lu) n=%d\n",
                    ctx->pe, VERTEX_ID_PE(vert), VERTEX_ID_OFFSET(vert), n);
            abort();
        }
        assert(cached_neighbor);

        if (get_dist_from_local_vert(cached_neighbor, &ctx->vec_cache,
                    ctx->pe) == UINT8_MAX) {
            set_dist_from_local_vert(cached_neighbor, UINT8_MAX - 1,
                    &ctx->vec_cache);
            cached_neighbor->tmp = newq;
            newq = cached_neighbor;
        }
    }

    node->tmp = NULL;
    return newq;
}

static void update_distances(hvr_internal_ctx_t *ctx) {
    /*
     * Clear all distances of mirrored vertices to an invalid value before
     * recomputing
     */
    memset(ctx->vec_cache.dist_from_local_vert, 0xff,
            ctx->vec_cache.pool_size *
            sizeof(ctx->vec_cache.dist_from_local_vert[0]));

    hvr_vertex_cache_node_t *newq = NULL;
    hvr_vertex_cache_node_t *q = ctx->vec_cache.local_neighbors_head;
    while (q) {
        set_dist_from_local_vert(q, 1, &ctx->vec_cache);
        newq = add_neighbors_to_q(q, newq, ctx);
        q = q->local_neighbors_next;
    }

    // Save distances for all vertices within the required graph depth
    for (unsigned l = 2; l <= ctx->max_graph_traverse_depth; l++) {
        newq = NULL;
        while (q) {
            hvr_vertex_cache_node_t *next_q = q->tmp;
            set_dist_from_local_vert(q, l, &ctx->vec_cache);
            newq = add_neighbors_to_q(q, newq, ctx);
            q = next_q;
        }

        q = newq;
    }

    // unsigned to_delete = 0;
    // unsigned zero_dist_verts = 0;
    // unsigned one_dist_verts = 0;
    // for (unsigned i = 0; i < HVR_MAP_BUCKETS; i++) {
    //     hvr_map_seg_t *seg = ctx->vec_cache.cache_map.buckets[i];
    //     while (seg) {
    //         for (unsigned j = 0; j < seg->nkeys; j++) {
    //             hvr_vertex_cache_node_t *node =
    //                 seg->data[j].inline_vals[0].cached_vert;
    //             uint8_t dist = get_dist_from_local_vert(node, &ctx->vec_cache,
    //                     ctx->pe);
    //             if (dist > ctx->max_graph_traverse_depth) {
    //                 to_delete++;
    //             } else if (dist == 0) {
    //                 zero_dist_verts++;
    //             } else if (dist == 1) {
    //                 one_dist_verts++;
    //             }
    //         }
    //         seg = seg->next;
    //     }
    // }
    // fprintf(stderr, "PE %d Want to delete %u / %lu : 0-dist=%u 1-dist=%u\n",
    //         shmem_my_pe(), to_delete, ctx->vec_cache.n_cached_vertices,
    //         zero_dist_verts, one_dist_verts);
}

void hvr_get_neighbors(hvr_vertex_t *vert, hvr_edge_info_t **out_neighbors,
        int *out_n_neighbors, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    assert(sizeof(hvr_edge_info_t) == sizeof(hvr_map_val_t));

    hvr_map_val_list_t neighbors;
    int n_neighbors = hvr_map_linearize(vert->id, &ctx->edges.map, &neighbors);

    if (n_neighbors <= 0) {
        *out_n_neighbors = 0;
        *out_neighbors = NULL;
    } else {
        *out_n_neighbors = n_neighbors;
        *out_neighbors = (hvr_edge_info_t *)malloc(
                n_neighbors * sizeof(hvr_edge_info_t));
        assert(*out_neighbors);
        for (unsigned n = 0; n < n_neighbors; n++) {
            (*out_neighbors)[n] = hvr_map_val_list_get(n, &neighbors).edge_info;
        }
    }
}

hvr_vertex_t *hvr_get_vertex(hvr_vertex_id_t vert_id, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;

    if (VERTEX_ID_PE(vert_id) == ctx->pe) {
        // Get it from the pool
        return ctx->pool.pool + VERTEX_ID_OFFSET(vert_id);
    } else {
        hvr_vertex_cache_node_t *cached = hvr_vertex_cache_lookup(vert_id,
                &ctx->vec_cache);
        if (!cached) {
            hvr_map_val_list_t vals_list;
            int n = hvr_map_linearize(vert_id, &ctx->vec_cache.cache_map,
                    &vals_list);
            fprintf(stderr, "PE %d failed getting (%lu,%lu) n=%d\n", ctx->pe,
                    VERTEX_ID_PE(vert_id), VERTEX_ID_OFFSET(vert_id), n);
            abort();
        }
        assert(cached);

        return &cached->vert;
    }
}

static int update_vertices(hvr_set_t *to_couple_with,
        hvr_internal_ctx_t *ctx) {
    if (ctx->update_metadata == NULL) {
        return 0;
    }

    int count = 0;
    hvr_vertex_iter_t iter;
    hvr_vertex_iter_init(&iter, ctx);
    for (hvr_vertex_t *curr = hvr_vertex_iter_next(&iter); curr;
            curr = hvr_vertex_iter_next(&iter)) {

        if (curr->needs_processing) {
            ctx->update_metadata(curr, to_couple_with, ctx);
            curr->needs_processing = 0;
            count++;
        }
    }

    // Update my local information on PEs I am coupled with.
    hvr_set_merge(ctx->coupled_pes, to_couple_with);
    return count;
}

void send_updates_to_all_subscribed_pes(hvr_vertex_t *vert,
        int is_delete,
        process_perf_info_t *perf_info,
        unsigned long long *time_sending,
        hvr_internal_ctx_t *ctx) {
    assert(VERTEX_ID_PE(vert->id) == ctx->pe);
    unsigned long long start = hvr_current_time_us();
    hvr_mailbox_t *mbox = (is_delete ? &ctx->vertex_delete_mailbox :
            &ctx->vertex_update_mailbox);

    hvr_partition_t part = wrap_actor_to_partition(vert, ctx);
    // Find subscribers to part and send message to them
    const unsigned n_subscribers = hvr_sparse_arr_row_length(part,
            &ctx->pe_subscription_info);
    if (n_subscribers > 0) {
        // Current neighbors for the updated vertex
        int *subscribers = NULL;
        unsigned capacity = 0;
        hvr_sparse_arr_linearize_row(part, &subscribers, &capacity,
                &ctx->pe_subscription_info);

        for (unsigned s = 0; s < n_subscribers; s++) {
            int sub = subscribers[s];

            hvr_vertex_update_t *msg = (is_delete ?
                    ctx->buffered_deletes + sub :
                    ctx->buffered_updates + sub);
            memcpy(&(msg->verts[msg->len]), vert, sizeof(*vert));
            msg->len += 1;

            if (msg->len == VERT_PER_UPDATE) {
                int success = hvr_mailbox_send(msg, sizeof(*msg), sub,
                        N_SEND_ATTEMPTS, mbox);
                while (!success) {
                    // Try processing some inbound messages, then re-sending
                    if (perf_info) {
                        *time_sending += (hvr_current_time_us() - start);
                        perf_info->n_received_updates += process_vertex_updates(
                                ctx, perf_info);
                        start = hvr_current_time_us();
                    }
                    success = hvr_mailbox_send(msg, sizeof(*msg), sub,
                            N_SEND_ATTEMPTS, mbox);
                }
                msg->len = 0;
            }
        }
        free(subscribers);
    }
    *time_sending += (hvr_current_time_us() - start);
}

static unsigned send_updates(hvr_internal_ctx_t *ctx,
        unsigned long long *time_sending,
        process_perf_info_t *perf_info) {
    unsigned n_updates_sent = 0;

    hvr_vertex_iter_t iter;
    hvr_vertex_iter_all_init(&iter, ctx);
    for (hvr_vertex_t *curr = hvr_vertex_iter_next(&iter); curr;
            curr = hvr_vertex_iter_next(&iter)) {
        // If this vertex was mutated on this iteration
        if (curr->needs_send) {
            send_updates_to_all_subscribed_pes(curr, 0, perf_info, time_sending,
                    ctx);
            n_updates_sent++;
            curr->needs_send = 0;
        }
    }

    flush_buffered_updates(ctx);

    return n_updates_sent;
}

static unsigned update_coupled_values(hvr_internal_ctx_t *ctx,
        hvr_vertex_t *coupled_metric, int count_updated) {
    // Copy the present value of the coupled metric locally
    memcpy(coupled_metric, ctx->coupled_pes_values + ctx->pe,
            sizeof(*coupled_metric));

    hvr_vertex_iter_t iter;
    hvr_vertex_iter_all_init(&iter, ctx);
    ctx->update_coupled_val(&iter, ctx, coupled_metric);

    /*
     * The coupling logic is a bit complicated because any PE can decide at any
     * time to become coupled with any other PE. The basic steps are as follows:
     *
     *   1. Send out messages to the mailboxes of any PEs which I am newly
     *      coupled with on this iteration, informing them that I am now
     *      coupled..
     *   3. Send out values to all PEs I am coupled to.
     *   2. Iteratively, check for:
     *      a. New messages from other PEs saying that I am now coupled with
     *         them. If this is a new coupling, forward it to everyone I am
     *         coupled with and then send that PE back a value from me.
     *         Otherwise, don't forward. The received message may itself have
     *         been forwarded.
     *      b. Messages from other PEs that I am coupled with, informing me of
     *         their coupled values for the current iteration. These have the
     *         potential to arrive out of order, in which case I receive a
     *         coupled value from another PE before I realize I am coupled with
     *         it. In that case, we should just resend it to myself. I may also
     *         receive a duplicate coupled value (a second value, when I've
     *         already received a first for the current iteration). This should
     *         simply be resent to myself and processed later.
     *
     * We will need to poll on these messages until we reach a point where we
     * have a coupled value for every PE that we know we are coupled to.
     */

    hvr_coupling_msg_t new_coupling_msg;
    new_coupling_msg.type = MSG_COUPLING_NEW;
    new_coupling_msg.pe = ctx->pe;

    hvr_coupling_msg_t coupled_val_msg;
    coupled_val_msg.type = MSG_COUPLING_VAL;
    coupled_val_msg.pe = ctx->pe;
    coupled_val_msg.updates_on_this_iter = count_updated;
    memcpy(&coupled_val_msg.val, coupled_metric, sizeof(*coupled_metric));

    for (unsigned p = 0; p < ctx->npes; p++) {
        if (hvr_set_contains(p, ctx->coupled_pes)) {
            if (!hvr_set_contains(p, ctx->prev_coupled_pes)) {
                // New coupling
                hvr_mailbox_send(&new_coupling_msg, sizeof(new_coupling_msg), p,
                        -1, &ctx->coupling_mailbox);
            }
            hvr_mailbox_send(&coupled_val_msg, sizeof(coupled_val_msg), p, -1,
                    &ctx->coupling_mailbox);
        }
    }
    hvr_set_wipe(ctx->coupled_pes_received_from);

    memcpy(ctx->coupled_pes_values + ctx->pe, coupled_metric,
            sizeof(*coupled_metric));
    hvr_set_insert(ctx->pe, ctx->coupled_pes_received_from);
    memset(ctx->updates_on_this_iter, 0x00,
            ctx->npes * sizeof(ctx->updates_on_this_iter[0]));

    size_t msg_len;
    while (hvr_set_count(ctx->coupled_pes_received_from) <
            hvr_set_count(ctx->coupled_pes)) {
        int success = hvr_mailbox_recv(&ctx->msg_buf, &ctx->msg_buf_capacity,
                &msg_len, &ctx->coupling_mailbox);
        if (success) {
            assert(msg_len == sizeof(hvr_coupling_msg_t));
            hvr_coupling_msg_t *msg = (hvr_coupling_msg_t *)ctx->msg_buf;
            if (msg->type == MSG_COUPLING_NEW) {
                const int new_pe = msg->pe;
                if (!hvr_set_contains(new_pe, ctx->coupled_pes)) {
                    hvr_set_insert(new_pe, ctx->coupled_pes);
                    // And forward
                    for (unsigned p = 0; p < ctx->npes; p++) {
                        if (hvr_set_contains(p, ctx->coupled_pes) &&
                                p != new_pe && p != ctx->pe) {
                            hvr_mailbox_send(msg, sizeof(*msg), p, -1,
                                    &ctx->coupling_mailbox);
                        }
                    }
                    hvr_mailbox_send(&coupled_val_msg, sizeof(coupled_val_msg),
                            new_pe, -1, &ctx->coupling_mailbox);
                }
            } else if (msg->type == MSG_COUPLING_VAL) {
                if (!hvr_set_contains(msg->pe, ctx->coupled_pes) ||
                        hvr_set_contains(msg->pe,
                            ctx->coupled_pes_received_from)) {
                    /*
                     * Don't know about this coupling yet, or already have a
                     * value for this PE. Re-send.
                     */
                    hvr_mailbox_send(msg, sizeof(*msg), ctx->pe, -1,
                            &ctx->coupling_mailbox);
                } else {
                    hvr_set_insert(msg->pe, ctx->coupled_pes_received_from);
                    ctx->updates_on_this_iter[msg->pe] = msg->updates_on_this_iter;
                    memcpy(ctx->coupled_pes_values + msg->pe, &msg->val,
                            sizeof(msg->val));
                }
            } else {
                abort();
            }
        }
    }

    hvr_set_copy(ctx->prev_coupled_pes, ctx->coupled_pes);

    /*
     * For each PE I know I'm coupled with, lock their coupled_timesteps
     * list and update my copy with any newer entries in my
     * coupled_timesteps list.
     */
    int ncoupled = 1; // include myself
    for (int p = 0; p < ctx->npes; p++) {
        if (p != ctx->pe && hvr_set_contains(p, ctx->coupled_pes)) {
            // No need to read lock here because I know I'm the only writer
            hvr_vertex_add(coupled_metric, ctx->coupled_pes_values + p, ctx);
            ncoupled++;
        }
    }

    hvr_vertex_iter_all_init(&iter, ctx);
    int should_abort = ctx->should_terminate(&iter, ctx,
            ctx->coupled_pes_values + ctx->pe, // Local coupled metric
            coupled_metric, // Global coupled metric
            ctx->coupled_pes, ncoupled, ctx->updates_on_this_iter);

    if (ncoupled > 1) {
        char buf[1024];
        hvr_vertex_dump(coupled_metric, buf, 1024, ctx);

        char coupled_pes_str[2048];
        hvr_set_to_string(ctx->coupled_pes, coupled_pes_str, 2048, NULL);

        printf("PE %d - computed coupled value {%s} from %d coupled PEs "
                "(%s)\n", ctx->pe, buf, ncoupled, coupled_pes_str);
    }

    return should_abort;
}

static size_t bytes_used_by_subcopy_arr(hvr_dist_bitvec_local_subcopy_t *arr,
        size_t N, hvr_set_t *window) {
    size_t bytes = N * sizeof(*arr);
    for (size_t i = 0; i < N; i++) {
        if (hvr_set_contains(i, window)) {
            bytes += hvr_dist_bitvec_local_subcopy_bytes(&arr[i]);
        }
    }
    return bytes;
}

static void print_profiling_info(
        unsigned long long start_iter,
        unsigned long long end_start_time_step,
        unsigned long long end_update_vertices,
        unsigned long long end_update_dist,
        unsigned long long end_update_partitions,
        unsigned long long end_partition_window,
        unsigned long long end_neighbor_updates,
        unsigned long long end_send_updates,
        unsigned long long end_vertex_updates,
        unsigned long long end_update_coupled,
        int count_updated,
        unsigned n_updates_sent,
        process_perf_info_t *perf_info,
        unsigned long long time_updating_partitions,
        unsigned long long time_updating_producers,
        unsigned long long time_updating_subscribers,
        unsigned long long dead_pe_time,
        unsigned long long time_sending,
        int should_abort,
        hvr_internal_ctx_t *ctx) {

    char partition_time_window_str[2048] = {'\0'};
// #ifdef DETAILED_PRINTS
//     hvr_set_to_string(ctx->producer_partition_time_window,
//             partition_time_window_str, 2048, ctx->partition_lists_lengths);
// #endif

    fprintf(profiling_fp, "PE %d - iter %d - total %f ms\n",
            ctx->pe, ctx->iter,
            (double)(end_update_coupled - start_iter) / 1000.0);
    fprintf(profiling_fp, "  start time step %f\n",
            (double)(end_start_time_step - start_iter) / 1000.0);
    fprintf(profiling_fp, "  update vertices %f - %d updates\n",
            (double)(end_update_vertices - end_start_time_step) / 1000.0,
            count_updated);
    fprintf(profiling_fp, "  update distances %f\n",
            (double)(end_update_dist - end_update_vertices) / 1000.0);
    fprintf(profiling_fp, "  update actor partitions %f\n",
            (double)(end_update_partitions - end_update_dist) / 1000.0);
    fprintf(profiling_fp, "  update partition window %f - update time = "
            "(parts=%f producers=%f subscribers=%f) dead PE time %f\n",
            (double)(end_partition_window - end_update_partitions) / 1000.0,
            (double)time_updating_partitions / 1000.0,
            (double)time_updating_producers / 1000.0,
            (double)time_updating_subscribers / 1000.0,
            (double)dead_pe_time / 1000.0);
    fprintf(profiling_fp, "  update neighbors %f\n",
            (double)(end_neighbor_updates - end_partition_window) / 1000.0);
    fprintf(profiling_fp, "  send updates %f - %u changes, %f ms sending\n",
            (double)(end_send_updates - end_neighbor_updates) / 1000.0,
            n_updates_sent, (double)time_sending / 1000.0);
    fprintf(profiling_fp, "  process vertex updates %f - %u received\n",
            (double)(end_vertex_updates - end_send_updates) / 1000.0,
            perf_info->n_received_updates);
    fprintf(profiling_fp, "    %f on deletes\n",
            (double)perf_info->time_handling_deletes / 1000.0);
    fprintf(profiling_fp, "    %f on news\n",
            (double)perf_info->time_handling_news / 1000.0);
    fprintf(profiling_fp, "      %f s on creating new\n",
            (double)perf_info->time_creating / 1000.0);
    fprintf(profiling_fp, "      %f s on updates - %f updating edges, "
            "%f creating edges - %u should_have_edges\n",
            (double)perf_info->time_updating / 1000.0,
            (double)perf_info->time_updating_edges / 1000.0,
            (double)perf_info->time_creating_edges / 1000.0,
            perf_info->count_new_should_have_edges);
    fprintf(profiling_fp, "  coupling %f\n",
            (double)(end_update_coupled - end_vertex_updates) / 1000.0);
    fprintf(profiling_fp, "  partition window = %s, %d / %d producer "
            "partitions and %d / %d subscriber partitions for %lu "
            "local vertices, %lu mirrored vertices\n", partition_time_window_str,
            hvr_set_count(ctx->producer_partition_time_window),
            ctx->n_partitions,
            hvr_set_count(ctx->subscriber_partition_time_window),
            ctx->n_partitions, hvr_n_allocated(ctx),
            ctx->vec_cache.n_cached_vertices);
    fprintf(profiling_fp, "  aborting? %d - remote "
            "cache hits=%llu misses=%llu, feature cache hits=%u misses=%u "
            "quiets=%llu\n",
            should_abort,
            ctx->vec_cache.cache_perf_info.nhits,
            ctx->vec_cache.cache_perf_info.nmisses,
            ctx->n_vector_cache_hits,
            ctx->n_vector_cache_misses,
            ctx->vec_cache.cache_perf_info.quiet_counter);

#ifdef DETAILED_PRINTS
    size_t edge_set_capacity, edge_set_used, vertex_cache_capacity,
           vertex_cache_used;
    double edge_set_val_capacity, edge_set_val_used, vertex_cache_val_capacity,
           vertex_cache_val_used;
    unsigned edge_set_max_len, vertex_cache_max_len;
    hvr_map_size_in_bytes(&ctx->edges.map, &edge_set_capacity, &edge_set_used,
            &edge_set_val_capacity, &edge_set_val_used, &edge_set_max_len);
    hvr_map_size_in_bytes(&ctx->vec_cache.cache_map, &vertex_cache_capacity,
            &vertex_cache_used, &vertex_cache_val_capacity,
            &vertex_cache_val_used, &vertex_cache_max_len);
    fprintf(profiling_fp, "  management data structure: vertex pool = %llu "
            "bytes, PE sub info = %llu bytes, producer info = %llu bytes, dead "
            "info = %llu bytes, edge set = %f MB (%f%% "
            "efficiency), vertex cache = %f MB (%f%% efficiency)\n",
            hvr_pool_size_in_bytes(ctx),
            hvr_sparse_arr_used_bytes(&ctx->pe_subscription_info),
            bytes_used_by_subcopy_arr(ctx->producer_info, ctx->n_partitions,
                ctx->subscriber_partition_time_window),
            bytes_used_by_subcopy_arr(ctx->dead_info, ctx->n_partitions,
                ctx->subscriber_partition_time_window),
            (double)edge_set_capacity / (1024.0 * 1024.0),
            100.0 * (double)edge_set_used / (double)edge_set_capacity,
            (double)vertex_cache_capacity / (1024.0 * 1024.0),
            100.0 * (double)vertex_cache_used / (double)vertex_cache_capacity);
    fprintf(profiling_fp, "    edge set value mean len=%f capacity=%f max "
            "len=%u\n", edge_set_val_used, edge_set_val_capacity,
            edge_set_max_len);
    fprintf(profiling_fp, "    vert cache value mean len=%f capacity=%f max "
            "len=%u\n", vertex_cache_val_used, vertex_cache_val_capacity,
            vertex_cache_max_len);
#endif
    fflush(profiling_fp);
}

static void save_current_state_to_dump_file(hvr_internal_ctx_t *ctx) {
    // Assume that all vertices have the same features.
    hvr_vertex_iter_t iter;
    hvr_vertex_iter_init(&iter, ctx);
    for (hvr_vertex_t *curr = hvr_vertex_iter_next(&iter); curr;
            curr = hvr_vertex_iter_next(&iter)) {
        fprintf(ctx->dump_file, "%u,%d,%lu,%u", ctx->iter, ctx->pe,
                curr->id, HVR_MAX_VECTOR_SIZE);
        for (unsigned f = 0; f < HVR_MAX_VECTOR_SIZE; f++) {
            fprintf(ctx->dump_file, ",%u,%f", f, hvr_vertex_get(f, curr, ctx));
        }
        fprintf(ctx->dump_file, "\n");

        hvr_edge_info_t *neighbors;
        int n_neighbors;
        hvr_get_neighbors(curr, &neighbors, &n_neighbors, ctx);

        fprintf(ctx->edges_dump_file, "%u,%d,%lu,%d,[", ctx->iter,
                ctx->pe, curr->id, n_neighbors);
        for (unsigned n = 0; n < n_neighbors; n++) {
            fprintf(ctx->edges_dump_file, " ");
            switch (EDGE_INFO_EDGE(neighbors[n])) {
                case DIRECTED_IN:
                    fprintf(ctx->edges_dump_file, "IN");
                    break;
                case DIRECTED_OUT:
                    fprintf(ctx->edges_dump_file, "OUT");
                    break;
                case BIDIRECTIONAL:
                    fprintf(ctx->edges_dump_file, "BI");
                    break;
                default:
                    abort();
            }
            fprintf(ctx->edges_dump_file, ":%lu",
                    EDGE_INFO_VERTEX(neighbors[n]));
        }
        fprintf(ctx->edges_dump_file, " ],,\n");
        free(neighbors);
    }
    fflush(ctx->dump_file);
    fflush(ctx->edges_dump_file);
}

hvr_exec_info hvr_body(hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    hvr_set_t *to_couple_with = hvr_create_empty_set(ctx->npes);

    if (getenv("HVR_HANG_ABORT")) {
        pthread_t aborting_pthread;
        const int pthread_err = pthread_create(&aborting_pthread, NULL,
                aborting_thread, NULL);
        assert(pthread_err == 0);
    }

    if (this_pe_has_exited) {
        // More throught needed to make sure this behavior would be safe
        fprintf(stderr, "[ERROR] HOOVER does not currently support re-entering "
                "hvr_body\n");
        abort();
    }

    shmem_barrier_all();

    // Initialize edges
    hvr_edge_set_init(&ctx->edges);

    const unsigned long long start_body = hvr_current_time_us();

    // Update distance of each mirrored vertex from a local vertex
    update_distances(ctx);

    const unsigned long long end_update_dist = hvr_current_time_us();

    /*
     * Find which partitions are locally active (either because a local vertex
     * is in them, or a locally mirrored vertex in them).
     */
    update_actor_partitions(ctx);
    const unsigned long long end_update_partitions = hvr_current_time_us();

    /*
     * Compute a set of all partitions which locally active partitions might
     * need notifications on. Subscribe to notifications about partitions from
     * update_partition_time_window in the global registry.
     */
    unsigned long long dead_pe_time, time_updating_partitions,
                  time_updating_producers, time_updating_subscribers;
    update_partition_time_window(ctx, &time_updating_partitions,
            &time_updating_producers, &time_updating_subscribers,
            &dead_pe_time);
    const unsigned long long end_partition_window = hvr_current_time_us();

    /*
     * Ensure everyone's partition windows are initialized before initializing
     * neighbors.
     */
    shmem_barrier_all();

    /*
     * Receive and process updates sent in update_pes_on_neighbors, and adjust
     * our local neighbors based on those updates. This stores a mapping from
     * partition to the PEs that are subscribed to updates in that partition
     * inside of ctx->pe_subscription_info.
     */
    process_perf_info_t perf_info;
    memset(&perf_info, 0x00, sizeof(perf_info));
    process_neighbor_updates(ctx, &perf_info);
    const unsigned long long end_neighbor_updates = hvr_current_time_us();

    /*
     * Process updates sent to us by neighbors via our main mailbox. Use these
     * updates to update edges for all vertices (both local and mirrored).
     */
    unsigned long long time_sending = 0;
    unsigned n_updates_sent = send_updates(ctx, &time_sending,
            &perf_info);

    // Ensure all updates are sent before processing them during initialization
    const unsigned long long end_send_updates = hvr_current_time_us();

    perf_info.n_received_updates += process_vertex_updates(ctx, &perf_info);
    const unsigned long long end_vertex_updates = hvr_current_time_us();

    hvr_vertex_t coupled_metric;
    int should_abort = update_coupled_values(ctx, &coupled_metric, 0);
    const unsigned long long end_update_coupled = hvr_current_time_us();

    if (print_profiling) {
        print_profiling_info(
                start_body,
                start_body,
                start_body,
                end_update_dist,
                end_update_partitions,
                end_partition_window,
                end_neighbor_updates,
                end_send_updates,
                end_vertex_updates,
                end_update_coupled,
                0, // count_updated
                n_updates_sent,
                &perf_info,
                time_updating_partitions,
                time_updating_producers,
                time_updating_subscribers,
                dead_pe_time,
                time_sending,
                should_abort,
                ctx);
    }

    ctx->iter += 1;

    while (!should_abort && hvr_current_time_us() - start_body <
            ctx->max_elapsed_seconds * 1000000ULL) {

        if (ctx->dump_mode && ctx->pool.tracker.used_list &&
                !ctx->only_last_iter_dump) {
            save_current_state_to_dump_file(ctx);
        }

        const unsigned long long start_iter = hvr_current_time_us();

        memset(&(ctx->vec_cache.cache_perf_info), 0x00,
                sizeof(ctx->vec_cache.cache_perf_info));

        hvr_set_wipe(to_couple_with);

        if (ctx->start_time_step) {
            hvr_vertex_iter_t iter;
            hvr_vertex_iter_init(&iter, ctx);
            ctx->start_time_step(&iter, ctx);
        }

        const unsigned long long end_start_time_step = hvr_current_time_us();

        // Must come before everything else
        int count_updated = update_vertices(to_couple_with, ctx);

        const unsigned long long end_update_vertices = hvr_current_time_us();

        update_distances(ctx);

        const unsigned long long end_update_dist = hvr_current_time_us();

        update_actor_partitions(ctx);

        const unsigned long long end_update_partitions = hvr_current_time_us();

        update_partition_time_window(ctx, &time_updating_partitions,
            &time_updating_producers, &time_updating_subscribers,
            &dead_pe_time);

        const unsigned long long end_partition_window = hvr_current_time_us();

        memset(&perf_info, 0x00, sizeof(perf_info));
        process_neighbor_updates(ctx, &perf_info);

        const unsigned long long end_neighbor_updates = hvr_current_time_us();

        n_updates_sent = send_updates(ctx, &time_sending, &perf_info);

        const unsigned long long end_send_updates = hvr_current_time_us();

        perf_info.n_received_updates += process_vertex_updates(ctx, &perf_info);

        const unsigned long long end_vertex_updates = hvr_current_time_us();

        should_abort = update_coupled_values(ctx, &coupled_metric,
                count_updated);

        const unsigned long long end_update_coupled = hvr_current_time_us();

        if (print_profiling) {
            print_profiling_info(
                    start_iter,
                    end_start_time_step,
                    end_update_vertices,
                    end_update_dist,
                    end_update_partitions,
                    end_partition_window,
                    end_neighbor_updates,
                    end_send_updates,
                    end_vertex_updates,
                    end_update_coupled,
                    count_updated,
                    n_updates_sent,
                    &perf_info,
                    time_updating_partitions,
                    time_updating_producers,
                    time_updating_subscribers,
                    dead_pe_time,
                    time_sending,
                    should_abort,
                    ctx);
        }

        if (ctx->strict_mode) {
            *(ctx->strict_counter_src) = 0;
            shmem_int_sum_to_all(ctx->strict_counter_dest,
                    ctx->strict_counter_src, 1, 0, 0, ctx->npes, ctx->p_wrk_int,
                    ctx->p_sync);
            shmem_barrier_all();
        }

        ctx->iter += 1;
    }

    shmem_quiet();

    hvr_set_destroy(to_couple_with);

    if (ctx->strict_mode) {
        while (1) {
            *(ctx->strict_counter_src) = 1;
            shmem_int_sum_to_all(ctx->strict_counter_dest,
                    ctx->strict_counter_src, 1, 0, 0, ctx->npes, ctx->p_wrk_int,
                    ctx->p_sync);
            shmem_barrier_all();
            if (*(ctx->strict_counter_dest) == ctx->npes) {
                break;
            }
        }
    }

    this_pe_has_exited = 1;

    for (unsigned i = 0; i < ctx->pool.tracker.capacity; i++) {
        hvr_vertex_t *vert = ctx->pool.pool + i;
        if (vert->id != HVR_INVALID_VERTEX_ID) {
            (ctx->vertex_partitions)[i] = wrap_actor_to_partition(vert,
                    ctx);
        } else {
            (ctx->vertex_partitions)[i] = HVR_INVALID_PARTITION;
        }
    }

    for (unsigned p = 0; p < ctx->n_partitions; p++) {
        if (ctx->local_partition_lists[p]) {
            hvr_dist_bitvec_set(p, ctx->pe, &ctx->terminated_pes);
        }
    }

    if (ctx->dump_mode && ctx->pool.tracker.used_list && ctx->only_last_iter_dump) {
        save_current_state_to_dump_file(ctx);
    }

    hvr_exec_info info;
    info.executed_iters = ctx->iter;
    return info;
}

void hvr_finalize(hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    if (ctx->dump_mode) {
        fclose(ctx->dump_file);
    }

    free(ctx);
}

int hvr_my_pe(hvr_ctx_t ctx) {
    return ctx->pe;
}

unsigned long long hvr_current_time_us() {
    struct timeval curr_time;
    gettimeofday(&curr_time, NULL);
    return curr_time.tv_sec * 1000000ULL + curr_time.tv_usec;
}
