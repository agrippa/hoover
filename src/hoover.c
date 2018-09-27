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
#include "shmem_rw_lock.h"
#include "hvr_vertex_iter.h"
#include "hvr_mailbox.h"

// #define DETAILED_PRINTS

#define CACHED_TIMESTEPS_TOLERANCE 2
#define MAX_INTERACTING_PARTITIONS 500
#define N_PARTITION_NODES_PREALLOC 3000

#define FINE_GRAIN_TIMING

// #define TRACK_VECTOR_GET_CACHE

static int print_profiling = 1;
static FILE *profiling_fp = NULL;
static volatile int this_pe_has_exited = 0;

typedef struct _hvr_partition_member_change_t {
    int pe;
    hvr_partition_t partition;
    int entered;
} hvr_partition_member_change_t;

typedef struct _hvr_dead_pe_msg_t {
    int pe;
    hvr_partition_t partition;
} hvr_dead_pe_msg_t;

typedef enum {
    CREATE, DELETE, UPDATE
} hvr_change_type_t;

typedef struct _hvr_vertex_update_t {
    hvr_vertex_t vert;
} hvr_vertex_update_t;

static hvr_vertex_cache_node_t *handle_new_vertex(hvr_vertex_t *new_vert,
        hvr_internal_ctx_t *ctx);

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

void hvr_ctx_create(hvr_ctx_t *out_ctx) {
    hvr_internal_ctx_t *new_ctx = (hvr_internal_ctx_t *)malloc(
            sizeof(*new_ctx));
    assert(new_ctx);
    memset(new_ctx, 0x00, sizeof(*new_ctx));

    new_ctx->pe = shmem_my_pe();
    new_ctx->npes = shmem_n_pes();

    new_ctx->pool = hvr_vertex_pool_create(get_symm_pool_nelements());

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
        unsigned *partition_lists_lengths, unsigned dist_from_local_vertex) {
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
    partition_lists_lengths[partition] += 1;

    if (dist_from_local_vertex <
            ctx->partition_min_dist_from_local_vertex[partition]) {
        ctx->partition_min_dist_from_local_vertex[partition] =
            dist_from_local_vertex;
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
    memset(ctx->local_partition_lists_lengths, 0x00,
            sizeof(unsigned) * ctx->n_partitions);
    memset(ctx->mirror_partition_lists, 0x00,
            sizeof(hvr_vertex_t *) * ctx->n_partitions);
    memset(ctx->mirror_partition_lists_lengths, 0x00,
            sizeof(unsigned) * ctx->n_partitions);
    memset(ctx->partition_min_dist_from_local_vertex, 0xff,
            sizeof(unsigned) * ctx->n_partitions);

    hvr_vertex_iter_t iter;
    hvr_vertex_iter_all_init(&iter, ctx);
    for (hvr_vertex_t *curr = hvr_vertex_iter_next(&iter); curr;
            curr = hvr_vertex_iter_next(&iter)) {
        update_vertex_partitions_for_vertex(curr, ctx,
                ctx->local_partition_lists, ctx->local_partition_lists_lengths,
                0);
    }

    for (unsigned i = 0; i < HVR_CACHE_BUCKETS; i++) {
        hvr_vertex_cache_node_t *iter = ctx->vec_cache.buckets[i];
        while (iter) {
            update_vertex_partitions_for_vertex(&iter->vert, ctx,
                    ctx->mirror_partition_lists,
                    ctx->mirror_partition_lists_lengths,
                    iter->min_dist_from_local_vertex);
            iter = iter->bucket_next;
        }
    }
}

static inline void update_partition_info(hvr_partition_t p,
        hvr_set_t *local_new, hvr_set_t *local_old,
        hvr_dist_bitvec_t *registry, hvr_internal_ctx_t *ctx, char *lbl) {
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
    const unsigned n_vertices_per_buf = 1024;
    static hvr_vertex_t *vert_buf = NULL;
    if (vert_buf == NULL) {
        vert_buf = (hvr_vertex_t *)malloc(n_vertices_per_buf *
                sizeof(*vert_buf));
        assert(vert_buf);
    }

    /*
     * Fetch the vertices from PE 'dead_pe' in chunks, looking for vertices
     * in 'partition' and adding them to our local vec cache.
     */
    for (unsigned i = 0; i < ctx->pool->pool_size; i += n_vertices_per_buf) {

        unsigned n_this_iter = ctx->pool->pool_size - i;
        if (n_this_iter > n_vertices_per_buf) {
            n_this_iter = n_vertices_per_buf;
        }
        hvr_vertex_t *src = ctx->pool->pool + i;
        shmem_getmem(vert_buf, src, n_this_iter * sizeof(*vert_buf),
                dead_pe);

        for (unsigned j = 0; j < n_this_iter; j++) {
            hvr_vertex_t *curr = vert_buf + j;
            if (curr->id != HVR_INVALID_VERTEX_ID) {
                hvr_partition_t part = wrap_actor_to_partition(curr, ctx);
                if (part == partition) {
                    handle_new_vertex(curr, ctx);
                }
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
        unsigned long long *out_time_spent_processing_dead_pes) {
    unsigned long long time_spent_processing_dead_pes = 0;
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
                && ctx->partition_min_dist_from_local_vertex[p] <=
                    ctx->max_graph_traverse_depth - 1) {
            // Subscriber to any interacting partitions with p
            hvr_partition_t interacting[MAX_INTERACTING_PARTITIONS];
            unsigned n_interacting;
            ctx->might_interact(p, ctx->full_partition_set, interacting,
                    &n_interacting, MAX_INTERACTING_PARTITIONS, ctx);

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

    /*
     * For all partitions, check if our interest in them has changed on this
     * iteration. If it has, notify the global registry of that.
     */
    for (hvr_partition_t p = 0; p < ctx->n_partitions; p++) {
        update_partition_info(p, ctx->tmp_subscriber_partition_time_window,
                ctx->subscriber_partition_time_window,
                &ctx->partition_subscribers, ctx, "subscriber");
        update_partition_info(p, ctx->tmp_producer_partition_time_window,
                ctx->producer_partition_time_window,
                &ctx->partition_producers, ctx, "producer");
    }

    // For each partition
    for (hvr_partition_t p = 0; p < ctx->n_partitions; p++) {
        // If we are subscribed to it:
        int old_sub = hvr_set_contains(p, ctx->subscriber_partition_time_window);
        if (hvr_set_contains(p, ctx->tmp_subscriber_partition_time_window)) {
            if (!old_sub) {
                // If this is a new subscription on this iteration:
                hvr_dist_bitvec_local_subcopy_init(&ctx->partition_producers,
                        ctx->producer_info + p);
                hvr_dist_bitvec_local_subcopy_init(&ctx->terminated_pes,
                        ctx->dead_info + p);

                /*
                 * copy and store the list of producers to check against next
                 * iter
                 */
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
                                &ctx->forward_mailbox);
                    }
                }

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

                hvr_dist_bitvec_copy_locally(p, &ctx->partition_producers,
                        &ctx->tmp_local_partition_membership2);

                hvr_partition_member_change_t change;
                change.pe = shmem_my_pe(); // The subscriber/unsubscriber
                change.partition = p;      // The partition (un)subscribed to
                change.entered = 1;        // new subscription

                for (int pe = 0; pe < ctx->npes; pe++) {
                    if (!hvr_dist_bitvec_local_subcopy_contains(pe,
                                ctx->producer_info + p) &&
                            hvr_dist_bitvec_local_subcopy_contains(pe,
                                &ctx->tmp_local_partition_membership2)) {
                        // New producer
                        hvr_mailbox_send(&change, sizeof(change), pe,
                                &ctx->forward_mailbox);
                    }
                }

                // Save for later
                hvr_dist_bitvec_local_subcopy_copy(ctx->producer_info + p,
                        &ctx->tmp_local_partition_membership2);

                hvr_dist_bitvec_copy_locally(p, &ctx->terminated_pes,
                        &ctx->tmp_local_partition_membership2);

                const unsigned long long start = hvr_current_time_us();
                for (int pe = 0; pe < ctx->npes; pe++) {
                    if (!hvr_dist_bitvec_local_subcopy_contains(pe,
                                ctx->dead_info + p) &&
                            hvr_dist_bitvec_local_subcopy_contains(pe,
                                &ctx->tmp_local_partition_membership2)) {
                        // New dead PE
                        pull_vertices_from_dead_pe(pe, p, ctx);
                    }
                }
                time_spent_processing_dead_pes += (hvr_current_time_us() - start);

                hvr_dist_bitvec_local_subcopy_copy(ctx->dead_info + p,
                        &ctx->tmp_local_partition_membership2);
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

                /*
                 * notify all producers of this partition of our subscription
                 * (they will then send us a full update).
                 */
                hvr_partition_member_change_t change;
                change.pe = shmem_my_pe(); // The subscriber/unsubscriber
                change.partition = p;      // The partition (un)subscribed to
                change.entered = 0;        // unsubscription
                for (int pe = 0; pe < ctx->npes; pe++) {
                    if (hvr_dist_bitvec_local_subcopy_contains(pe,
                                ctx->producer_info + p)) {
                        hvr_mailbox_send(&change, sizeof(change), pe,
                                &ctx->forward_mailbox);
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

    *out_time_spent_processing_dead_pes = time_spent_processing_dead_pes;
}

void hvr_init(const hvr_partition_t n_partitions,
        hvr_update_metadata_func update_metadata,
        hvr_might_interact_func might_interact,
        hvr_check_abort_func check_abort,
        hvr_actor_to_partition actor_to_partition,
        hvr_start_time_step start_time_step,
        hvr_should_have_edge should_have_edge,
        unsigned long long max_elapsed_seconds,
        unsigned max_graph_traverse_depth,
        hvr_ctx_t in_ctx) {
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

    hvr_partition_list_node_t *all_nodes = (hvr_partition_list_node_t *)malloc(
            N_PARTITION_NODES_PREALLOC * sizeof(*all_nodes));
    assert(all_nodes);

    assert(n_partitions <= HVR_INVALID_PARTITION);
    new_ctx->n_partitions = n_partitions;

    new_ctx->update_metadata = update_metadata;
    new_ctx->might_interact = might_interact;
    new_ctx->check_abort = check_abort;
    new_ctx->actor_to_partition = actor_to_partition;
    new_ctx->start_time_step = start_time_step;
    new_ctx->should_have_edge = should_have_edge;

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
    }

    if (getenv("HVR_DISABLE_PROFILING_PRINTS")) {
        print_profiling = 0;
    } else {
        char profiling_filename[1024];
        sprintf(profiling_filename, "%d.prof", new_ctx->pe);
        profiling_fp = fopen(profiling_filename, "w");
        assert(profiling_fp);
    }

    new_ctx->coupled_pes = hvr_create_empty_set_symmetric(new_ctx->npes);
    hvr_set_insert(new_ctx->pe, new_ctx->coupled_pes);

    new_ctx->coupled_pes_values = (hvr_vertex_t *)shmem_malloc_wrapper(
            new_ctx->npes * sizeof(hvr_vertex_t));
    assert(new_ctx->coupled_pes_values);
    for (unsigned i = 0; i < new_ctx->npes; i++) {
        hvr_vertex_init(&(new_ctx->coupled_pes_values)[i], new_ctx);
    }

    new_ctx->coupled_pes_values_buffer = (hvr_vertex_t *)malloc(
            new_ctx->npes * sizeof(hvr_vertex_t));
    assert(new_ctx->coupled_pes_values_buffer);

    new_ctx->coupled_lock = hvr_rwlock_create_n(1);
   
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
    new_ctx->local_partition_lists_lengths = (unsigned *)malloc(
            sizeof(unsigned) * new_ctx->n_partitions);
    assert(new_ctx->local_partition_lists_lengths);

    new_ctx->mirror_partition_lists = (hvr_vertex_t **)malloc(
            sizeof(hvr_vertex_t *) * new_ctx->n_partitions);
    assert(new_ctx->mirror_partition_lists);
    new_ctx->mirror_partition_lists_lengths = (unsigned *)malloc(
            sizeof(unsigned) * new_ctx->n_partitions);
    assert(new_ctx->mirror_partition_lists_lengths);

    new_ctx->partition_min_dist_from_local_vertex = (unsigned *)malloc(
            sizeof(unsigned) * new_ctx->n_partitions);
    assert(new_ctx->partition_min_dist_from_local_vertex);

    hvr_vertex_cache_init(&new_ctx->vec_cache, new_ctx->n_partitions);

    hvr_mailbox_init(&new_ctx->mailbox, 32 * 1024 * 1024);
    hvr_mailbox_init(&new_ctx->partition_mailbox, 32 * 1024 * 1024);
    hvr_mailbox_init(&new_ctx->forward_mailbox, 32 * 1024 * 1024);
    hvr_mailbox_init(&new_ctx->dead_mailbox, 16 * 1024 * 1024);

    hvr_dist_bitvec_init(new_ctx->n_partitions, new_ctx->npes,
            &new_ctx->partition_subscribers);
    hvr_dist_bitvec_init(new_ctx->n_partitions, new_ctx->npes,
            &new_ctx->partition_producers);
    hvr_dist_bitvec_init(new_ctx->n_partitions, new_ctx->npes,
            &new_ctx->terminated_pes);

    new_ctx->local_partition_subscribers =
        (hvr_dist_bitvec_local_subcopy_t *)malloc(
                new_ctx->partition_subscribers.dim0_per_pe *
                sizeof(*new_ctx->local_partition_subscribers));
    for (unsigned i = 0; i < new_ctx->partition_subscribers.dim0_per_pe; i++) {
        hvr_dist_bitvec_local_subcopy_init(
                &new_ctx->partition_subscribers,
                new_ctx->local_partition_subscribers + i);
    }

    new_ctx->local_partition_producers =
        (hvr_dist_bitvec_local_subcopy_t *)malloc(
                new_ctx->partition_subscribers.dim0_per_pe *
                sizeof(*new_ctx->local_partition_producers));
    for (unsigned i = 0; i < new_ctx->partition_subscribers.dim0_per_pe; i++) {
        hvr_dist_bitvec_local_subcopy_init(
                &new_ctx->partition_subscribers,
                new_ctx->local_partition_producers + i);
    }

    new_ctx->local_partition_terminated =
        (hvr_dist_bitvec_local_subcopy_t *)malloc(
                new_ctx->terminated_pes.dim0_per_pe *
                sizeof(*new_ctx->local_partition_terminated));
    for (unsigned i = 0; i < new_ctx->terminated_pes.dim0_per_pe; i++) {
        hvr_dist_bitvec_local_subcopy_init(
                &new_ctx->terminated_pes,
                new_ctx->local_partition_terminated + i);
    }

    hvr_dist_bitvec_local_subcopy_init(
            &new_ctx->partition_subscribers,
            &new_ctx->tmp_local_partition_membership);
    hvr_dist_bitvec_local_subcopy_init(
            &new_ctx->partition_producers,
            &new_ctx->tmp_local_partition_membership2);

    new_ctx->full_partition_set = hvr_create_full_set(new_ctx->n_partitions);

    new_ctx->pe_subscription_info = hvr_create_empty_edge_set();

    new_ctx->max_graph_traverse_depth = max_graph_traverse_depth;

    new_ctx->producer_info = (hvr_dist_bitvec_local_subcopy_t *)malloc(
            new_ctx->n_partitions * sizeof(hvr_dist_bitvec_local_subcopy_t));
    assert(new_ctx->producer_info);
    new_ctx->dead_info = (hvr_dist_bitvec_local_subcopy_t *)malloc(
            new_ctx->n_partitions * sizeof(hvr_dist_bitvec_local_subcopy_t));
    assert(new_ctx->dead_info);

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

static void process_neighbor_updates(hvr_internal_ctx_t *ctx) {
    void *buf = NULL;
    size_t buf_capacity = 0;
    size_t msg_len;
    int success = hvr_mailbox_recv(&buf, &buf_capacity,
            &msg_len, &ctx->forward_mailbox);
    while (success) {
        assert(msg_len == sizeof(hvr_partition_member_change_t));
        /*
         * Tells that a given PE has subscribed/unsubscribed to updates for a
         * given partition for which we are a producer for.
         */
        hvr_partition_member_change_t *change =
            (hvr_partition_member_change_t *)buf;

        if (change->entered) {
            // Entered partition

            if (hvr_have_edge(change->partition, change->pe,
                        ctx->pe_subscription_info) == NO_EDGE) {
                /*
                 * This is a new subscription from this PE for this
                 * partition. As we go through this, if we find new
                 * subscriptions, we need to transmit all vertex information we
                 * have for that partition to the PE.
                 */
                hvr_vertex_t *iter = ctx->local_partition_lists[
                    change->partition];
                while (iter) {
                    hvr_vertex_update_t msg;
                    memcpy(&msg.vert, iter, sizeof(*iter));
                    hvr_mailbox_send(&msg, sizeof(msg), change->pe,
                            &ctx->mailbox);
                    iter = iter->next_in_partition;
                }
                hvr_add_edge(change->partition, change->pe, BIDIRECTIONAL,
                        ctx->pe_subscription_info);
            }
        } else {
            // Left partition
            hvr_remove_edge(change->partition, change->pe,
                    ctx->pe_subscription_info);
        }
        success = hvr_mailbox_recv(&buf, &buf_capacity,
                &msg_len, &ctx->forward_mailbox);
    }
    free(buf);
}

static inline hvr_edge_type_t flip_edge_direction(hvr_edge_type_t dir) {
    switch (dir) {
        case (DIRECTED_IN):
            return DIRECTED_OUT;
        case (DIRECTED_OUT):
            return DIRECTED_IN;
        case (BIDIRECTIONAL):
            return BIDIRECTIONAL;
        default:
            assert(0);
    }
}

static void update_edge_info(hvr_vertex_t *base, hvr_vertex_t *neighbor,
        hvr_edge_type_t new_edge, hvr_edge_type_t existing_edge,
        hvr_internal_ctx_t *ctx) {
    assert(new_edge != existing_edge);

    if (existing_edge != NO_EDGE) {
        hvr_remove_edge(base->id, neighbor->id, ctx->edges);
        hvr_remove_edge(neighbor->id, base->id, ctx->edges);
    }

    if (new_edge != NO_EDGE) {
        /*
         * Removing and then adding is less efficient than simply updating the
         * edge, but allows us to make more assertions that are helpful for
         * debugging.
         */
        hvr_add_edge(base->id, neighbor->id, new_edge, ctx->edges);
        hvr_add_edge(neighbor->id, base->id, flip_edge_direction(new_edge),
                ctx->edges);
    }

    base->last_modify_iter = ctx->iter;
    neighbor->last_modify_iter = ctx->iter;
}

static int create_new_edges_helper(hvr_vertex_t *vert,
        hvr_vertex_t *updated_vert, hvr_internal_ctx_t *ctx) {
    hvr_edge_type_t edge = ctx->should_have_edge(vert, updated_vert, ctx);
    if (edge == NO_EDGE) {
        return 0;
    } else {
        update_edge_info(vert, updated_vert, edge, NO_EDGE, ctx);
        return 1;
    }
}

/*
* Figure out what edges need to be added here, from should_have_edge and then
* insert them for the new vertex. Eventually, any local vertex which had a new
* edge inserted will need to be updated.
*/
static unsigned create_new_edges(hvr_vertex_t *updated,
        hvr_partition_t *interacting, unsigned n_interacting,
        hvr_internal_ctx_t *ctx) {
    unsigned min_dist_from_local = UINT_MAX;
    for (unsigned i = 0; i < n_interacting; i++) {
        hvr_partition_t other_part = interacting[i];

        hvr_vertex_t *iter = ctx->local_partition_lists[other_part];
        while (iter) {
            int edge_created = create_new_edges_helper(iter, updated, ctx);
            if (edge_created) {
                min_dist_from_local = 1;
            }
            iter = iter->next_in_partition;
        }

        hvr_vertex_cache_node_t *cache_iter =
            ctx->vec_cache.partitions[other_part];
        while (cache_iter) {
            int edge_created = create_new_edges_helper(&cache_iter->vert,
                    updated, ctx);
            if (edge_created && cache_iter->min_dist_from_local_vertex <
                    min_dist_from_local) {
                min_dist_from_local = cache_iter->min_dist_from_local_vertex;
            }
            cache_iter = cache_iter->partition_next;
        }
    }

    return min_dist_from_local;
}

static hvr_vertex_cache_node_t *handle_new_vertex(hvr_vertex_t *new_vert,
        hvr_internal_ctx_t *ctx) {
    hvr_partition_t partition = wrap_actor_to_partition(new_vert, ctx);
    hvr_vertex_id_t updated_vert_id = new_vert->id;

    hvr_partition_t interacting[MAX_INTERACTING_PARTITIONS];
    unsigned n_interacting;
    ctx->might_interact(partition, ctx->full_partition_set,
            interacting, &n_interacting, MAX_INTERACTING_PARTITIONS, ctx);

    hvr_vertex_cache_node_t *updated = hvr_vertex_cache_lookup(
            updated_vert_id, &ctx->vec_cache);
    /*
     * If this is a vertex we already know about then we have
     * existing local edges that might need updating.
     */
    if (updated) {
        /*
         * Update our local mirror with the information received in the
         * message.
         */
        memcpy(&updated->vert, new_vert, sizeof(*new_vert));

        /*
         * Look for existing edges and verify they should still exist with
         * this update to the local mirror.
         */
        hvr_avl_tree_node_t *vertex_edge_tree = hvr_tree_find(
                ctx->edges->tree, updated_vert_id);
        if (vertex_edge_tree && vertex_edge_tree->linearized_length > 0) {
            hvr_vertex_id_t *neighbors = NULL;
            hvr_edge_type_t *edges = NULL;
            // Current neighbors for the updated vertex
            unsigned n_neighbors = hvr_tree_linearize(&neighbors, &edges,
                    vertex_edge_tree);

            for (unsigned n = 0; n < n_neighbors; n++) {
                hvr_vertex_t *neighbor = NULL;
                if (VERTEX_ID_PE(neighbors[n]) == ctx->pe) {
                    neighbor = ctx->pool->pool + VERTEX_ID_OFFSET(
                            neighbors[n]);
                } else {
                    hvr_vertex_cache_node_t *cached_neighbor =
                        hvr_vertex_cache_lookup(neighbors[n],
                                &ctx->vec_cache);
                    assert(cached_neighbor);
                    neighbor = &cached_neighbor->vert;
                }

                // Check this edge should still exist
                hvr_edge_type_t edge = ctx->should_have_edge(
                        &updated->vert, neighbor, ctx);
                int changed = (edge != edges[n]);
                if (changed) {
                    update_edge_info(&updated->vert, neighbor, edge, edges[n],
                            ctx);
                }
            }
        }

        create_new_edges(&updated->vert, interacting, n_interacting, ctx);
    } else {
        unsigned min_dist_from_local = create_new_edges(new_vert,
                interacting, n_interacting, ctx);
        updated = hvr_vertex_cache_add(new_vert, partition, min_dist_from_local,
                &ctx->vec_cache);
    }

    // So we know to rerun any neighbors on the next iteration
    updated->vert.last_modify_iter = ctx->iter;

    return updated;
}

static unsigned process_vertex_updates(hvr_internal_ctx_t *ctx) {
    unsigned n_updates = 0;
    void *buf = NULL;
    size_t buf_capacity = 0;
    size_t msg_len;
    int success = hvr_mailbox_recv(&buf, &buf_capacity,
            &msg_len, &ctx->mailbox);
    while (success) {
        assert(msg_len == sizeof(hvr_vertex_update_t));
        hvr_vertex_update_t *msg = (hvr_vertex_update_t *)buf;

        handle_new_vertex(&msg->vert, ctx);
        n_updates++;

        success = hvr_mailbox_recv(&buf, &buf_capacity,
                &msg_len, &ctx->mailbox);
    }
    free(buf);
    return n_updates;
}

void hvr_get_neighbors(hvr_vertex_t *vert, hvr_vertex_id_t **out_neighbors,
        hvr_edge_type_t **out_directions, size_t *out_n_neighbors,
        hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;

    hvr_avl_tree_node_t *vertex_edge_tree = hvr_tree_find(
            ctx->edges->tree, vert->id);
    if (vertex_edge_tree == NULL || vertex_edge_tree->linearized_length == 0) {
        *out_n_neighbors = 0;
    } else {
        *out_n_neighbors = hvr_tree_linearize(out_neighbors,
                out_directions, vertex_edge_tree);
    }
}

hvr_vertex_t *hvr_get_vertex(hvr_vertex_id_t vert_id, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;

    if (VERTEX_ID_PE(vert_id) == ctx->pe) {
        // Get it from the pool
        return ctx->pool->pool + VERTEX_ID_OFFSET(vert_id);
    } else {
        hvr_vertex_cache_node_t *cached = hvr_vertex_cache_lookup(vert_id,
                &ctx->vec_cache);
        assert(cached);

        return &cached->vert;
    }
}

static unsigned update_vertices(hvr_set_t *to_couple_with,
        hvr_internal_ctx_t *ctx) {
    unsigned n_local_verts = 0;

    hvr_vertex_iter_t iter;
    hvr_vertex_iter_init(&iter, ctx);
    for (hvr_vertex_t *curr = hvr_vertex_iter_next(&iter); curr;
            curr = hvr_vertex_iter_next(&iter)) {

        hvr_vertex_id_t *neighbors;
        hvr_edge_type_t *directions;
        size_t n_neighbors;
        hvr_get_neighbors(curr, &neighbors, &directions, &n_neighbors, ctx);

        int any_updates = (curr->last_modify_iter == ctx->iter - 1);
        for (unsigned n = 0; n < n_neighbors; n++) {
            hvr_vertex_t *neighbor = hvr_get_vertex(neighbors[n], ctx);
            if (neighbor->last_modify_iter == ctx->iter - 1) {
                any_updates = 1;
                break;
            }
        }

        if (any_updates) {
            ctx->update_metadata(curr, to_couple_with, ctx);
        }
        n_local_verts++;
    }
    return n_local_verts;
}

static unsigned send_updates(hvr_internal_ctx_t *ctx) {
    unsigned n_updates_sent = 0;

    hvr_vertex_iter_t iter;
    hvr_vertex_iter_all_init(&iter, ctx);
    for (hvr_vertex_t *curr = hvr_vertex_iter_next(&iter); curr;
            curr = hvr_vertex_iter_next(&iter)) {
        // If this vertex was mutated on this iteration
        if (curr->last_modify_iter == ctx->iter) {
            hvr_vertex_update_t msg;
            memcpy(&msg.vert, curr, sizeof(*curr));

            hvr_partition_t part = wrap_actor_to_partition(curr, ctx);
            // Find subscribers to part and send message to them
            hvr_avl_tree_node_t *pes_tree = hvr_tree_find(
                    ctx->pe_subscription_info->tree, part);
            if (pes_tree && pes_tree->linearized_length > 0) {
                hvr_vertex_id_t *subscribers = NULL;
                hvr_edge_type_t *edges = NULL;
                // Current neighbors for the updated vertex
                unsigned n_subscribers = hvr_tree_linearize(&subscribers,
                        &edges, pes_tree);

                for (unsigned s = 0; s < n_subscribers; s++) {
                    hvr_mailbox_send(&msg, sizeof(msg), subscribers[s],
                            &ctx->mailbox);
                }
            }

            n_updates_sent++;
        }
    }
    return n_updates_sent;
}

static unsigned update_coupled_values(hvr_internal_ctx_t *ctx,
        hvr_vertex_t *coupled_metric, int *should_abort,
        hvr_set_t *to_couple_with) {
    // Copy the present value of the coupled metric locally
    shmem_getmem(coupled_metric, ctx->coupled_pes_values + ctx->pe,
            sizeof(*coupled_metric), shmem_my_pe());

    // check_abort callback
    hvr_vertex_iter_t iter;
    hvr_vertex_iter_all_init(&iter, ctx);
    *should_abort = ctx->check_abort(&iter, ctx, to_couple_with,
            coupled_metric);

    // Update my local information on PEs I am coupled with.
    hvr_set_merge_atomic(ctx->coupled_pes, to_couple_with);

    /*
     * Atomically update other PEs that I am coupled with, telling them I am
     * coupled with them and who I am coupled with.
     */
    for (int p = 0; p < ctx->npes; p++) {
        if (p != ctx->pe && hvr_set_contains(p, ctx->coupled_pes)) {
            for (int i = 0; i < ctx->coupled_pes->nelements; i++) {
                unsigned long long remote_val = shmem_ulonglong_atomic_fetch(
                        ctx->coupled_pes->bit_vector + i, p);
                shmem_ulonglong_atomic_or(ctx->coupled_pes->bit_vector + i,
                        remote_val, shmem_my_pe());
                        
                // SHMEM_ULONGLONG_ATOMIC_OR(
                //         ctx->coupled_pes->bit_vector + i,
                //         (ctx->coupled_pes->bit_vector)[i], p);
            }
        }
    }

    // Update my symmetrically visible coupled metric
    hvr_rwlock_wlock((long *)ctx->coupled_lock, ctx->pe);
    shmem_putmem(ctx->coupled_pes_values + ctx->pe, coupled_metric,
            sizeof(*coupled_metric), ctx->pe);
    shmem_quiet();
    hvr_rwlock_wunlock((long *)ctx->coupled_lock, ctx->pe);

    /*
     * For each PE I know I'm coupled with, lock their coupled_timesteps
     * list and update my copy with any newer entries in my
     * coupled_timesteps list.
     */
    int ncoupled = 1; // include myself
    for (int p = 0; p < ctx->npes; p++) {
        if (p == ctx->pe) continue;

        if (hvr_set_contains_atomic(p, ctx->coupled_pes)) {
            // Pull in the latest coupled value from PE p.
            hvr_rwlock_rlock((long *)ctx->coupled_lock, p);
            shmem_getmem(ctx->coupled_pes_values_buffer,
                    ctx->coupled_pes_values,
                    ctx->npes * sizeof(hvr_vertex_t), p);
            hvr_rwlock_runlock((long *)ctx->coupled_lock, p);

            // Update my local copy with the latest values
            hvr_vertex_t *other = ctx->coupled_pes_values_buffer + p;
            hvr_vertex_t *mine = ctx->coupled_pes_values + p;

            hvr_rwlock_wlock((long *)ctx->coupled_lock, ctx->pe);
            memcpy(mine, other, sizeof(*mine));
            hvr_rwlock_wunlock((long *)ctx->coupled_lock, ctx->pe);

            // No need to read lock here because I know I'm the only writer
            hvr_vertex_add(coupled_metric, ctx->coupled_pes_values + p,
                    ctx);

            ncoupled++;
        }
    }

    return ncoupled;
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
    ctx->edges = hvr_create_empty_edge_set();

    /*
     * Find which partitions are locally active (either because a local vertex
     * is in them, or a locally mirrored vertex in them).
     */
    update_actor_partitions(ctx);

    /*
     * Compute a set of all partitions which locally active partitions might
     * need notifications on. Subscribe to notifications about partitions from
     * update_partition_time_window in the global registry.
     */
    unsigned long long dead_pe_time;
    update_partition_time_window(ctx, &dead_pe_time);

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
    process_neighbor_updates(ctx);

    /*
     * Process updates sent to us by neighbors via our main mailbox. Use these
     * updates to update edges for all vertices (both local and mirrored).
     */
    unsigned n_updates_sent = send_updates(ctx);

    unsigned n_received_updates = process_vertex_updates(ctx);

    hvr_vertex_t coupled_metric;
    int should_abort = 0;
    update_coupled_values(ctx, &coupled_metric, &should_abort, to_couple_with);

    ctx->iter += 1;

    const unsigned long long start_body = hvr_current_time_us();

    while (!should_abort && hvr_current_time_us() - start_body <
            ctx->max_elapsed_seconds * 1000000ULL) {

        if (ctx->dump_mode && ctx->pool->used_list) {
            // Assume that all vertices have the same features.
            unsigned nfeatures;
            unsigned features[HVR_MAX_VECTOR_SIZE];
            hvr_vertex_unique_features(
                    ctx->pool->pool + ctx->pool->used_list->start_index,
                    features, &nfeatures);

            hvr_vertex_iter_t iter;
            hvr_vertex_iter_init(&iter, ctx);
            for (hvr_vertex_t *curr = hvr_vertex_iter_next(&iter); curr;
                    curr = hvr_vertex_iter_next(&iter)) {
                fprintf(ctx->dump_file, "%u,%d,%lu,%u", ctx->iter, ctx->pe,
                        curr->id, nfeatures);
                for (unsigned f = 0; f < nfeatures; f++) {
                    fprintf(ctx->dump_file, ",%u,%f", features[f],
                            hvr_vertex_get(features[f], curr, ctx));
                }
                fprintf(ctx->dump_file, ",,\n");

                hvr_vertex_id_t *neighbors;
                hvr_edge_type_t *directions;
                size_t n_neighbors;
                hvr_get_neighbors(curr, &neighbors, &directions, &n_neighbors,
                        ctx);

                fprintf(ctx->edges_dump_file, "%u,%d,%lu,%lu,[", ctx->iter,
                        ctx->pe, curr->id, n_neighbors);
                for (unsigned n = 0; n < n_neighbors; n++) {
                    fprintf(ctx->edges_dump_file, " ");
                    switch (directions[n]) {
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
                            assert(0);
                    }
                    fprintf(ctx->edges_dump_file, ":%lu", neighbors[n]);
                }
                fprintf(ctx->edges_dump_file, " ],,\n");
            }
            fflush(ctx->dump_file);
            fflush(ctx->edges_dump_file);
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
        unsigned n_local_verts = update_vertices(to_couple_with, ctx);

        const unsigned long long end_update_vertices = hvr_current_time_us();

        update_actor_partitions(ctx);

        const unsigned long long end_update_partitions = hvr_current_time_us();

        update_partition_time_window(ctx, &dead_pe_time);

        const unsigned long long end_partition_window = hvr_current_time_us();

        process_neighbor_updates(ctx);

        const unsigned long long end_neighbor_updates = hvr_current_time_us();

        n_updates_sent = send_updates(ctx);

        const unsigned long long end_send_updates = hvr_current_time_us();

        n_received_updates = process_vertex_updates(ctx);

        const unsigned long long end_vertex_updates = hvr_current_time_us();

        unsigned ncoupled = update_coupled_values(ctx, &coupled_metric,
                &should_abort, to_couple_with);

        /*
         * TODO coupled_metric here contains the aggregate values over all
         * coupled PEs, including this one. Do we want to do anything with this,
         * other than print it?
         */
        if (ncoupled > 1) {
            char buf[1024];
            hvr_vertex_dump(&coupled_metric, buf, 1024, ctx);

            char coupled_pes_str[1024];
            hvr_set_to_string(ctx->coupled_pes, coupled_pes_str, 1024, NULL);

            printf("PE %d - computed coupled value {%s} from %d coupled PEs "
                    "(%s)\n", ctx->pe, buf, ncoupled, coupled_pes_str);
        }

        const unsigned long long end_update_coupled = hvr_current_time_us();

        char partition_time_window_str[2048] = {'\0'};
#ifdef DETAILED_PRINTS
        hvr_set_to_string(ctx->producer_partition_time_window,
                partition_time_window_str, 2048, ctx->partition_lists_lengths);
#endif

        if (print_profiling) {
            fprintf(profiling_fp, "PE %d - iter %d - total %f ms\n",
                    ctx->pe, ctx->iter,
                    (double)(end_update_coupled - start_iter) / 1000.0);
            fprintf(profiling_fp, "  start time step %f\n",
                    (double)(end_start_time_step - start_iter) / 1000.0);
            fprintf(profiling_fp, "  update actor partitions %f\n",
                    (double)(end_update_partitions - end_start_time_step) / 1000.0);
            fprintf(profiling_fp, "  update partition window %f - dead PE time %f\n",
                    (double)(end_partition_window - end_update_partitions) / 1000.0,
                    (double)dead_pe_time / 1000.0);
            fprintf(profiling_fp, "  update neighbors %f\n",
                    (double)(end_neighbor_updates - end_partition_window) / 1000.0);
            fprintf(profiling_fp, "  process vertex updates %f - %u received\n",
                    (double)(end_vertex_updates - end_neighbor_updates) / 1000.0,
                    n_received_updates);
            fprintf(profiling_fp, "  update vertices %f\n",
                    (double)(end_update_vertices - end_vertex_updates) / 1000.0);
            fprintf(profiling_fp, "  send updates %f - %u changes\n",
                    (double)(end_send_updates - end_update_vertices) / 1000.0,
                    n_updates_sent);
            fprintf(profiling_fp, "  coupling %f\n",
                    (double)(end_update_coupled - end_send_updates) / 1000.0);
            fprintf(profiling_fp, "  partition window = %s, %d / %d partitions "
                    "active for %u local vertices\n", partition_time_window_str,
                    hvr_set_count(ctx->producer_partition_time_window),
                    ctx->n_partitions, n_local_verts);
            fprintf(profiling_fp, "  aborting? %d - remote "
                    "cache hits=%llu misses=%llu, feature cache hits=%u misses=%u "
                    "quiets=%llu\n",
                    should_abort,
                    ctx->vec_cache.cache_perf_info.nhits,
                    ctx->vec_cache.cache_perf_info.nmisses,
                    ctx->n_vector_cache_hits,
                    ctx->n_vector_cache_misses,
                    ctx->vec_cache.cache_perf_info.quiet_counter);
            fflush(profiling_fp);
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

    fprintf(stderr, "PE %d leaving...\n", shmem_my_pe());

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

    for (unsigned p = 0; p < ctx->n_partitions; p++) {
        if (ctx->local_partition_lists[p]) {
            hvr_dist_bitvec_set(p, ctx->pe, &ctx->terminated_pes);
        }
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
