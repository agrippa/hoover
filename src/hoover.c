/* For license: see LICENSE.txt file at top-level */

#define _DEFAULT_SOURCE
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

#ifdef HVR_MULTITHREADED
#include <omp.h>
#define HVR_IS_MULTITHREADED 1
#else
#define HVR_IS_MULTITHREADED 0
#endif

#include "hoover.h"
#include "hvr_vertex_iter.h"
#include "hvr_mailbox.h"

// #define DETAILED_PRINTS
// #define COUPLING_PRINTS

// #define PRINT_PARTITIONS
#define MAX_INTERACTING_PARTITIONS 4000
#define N_SEND_ATTEMPTS 10
#define MAX_PROFILED_ITERS 1000
#define MS_PER_S 1000.0
#define MAX_MSGS_PROCESSED 1000

static int print_profiling = 1;
static int trace_shmem_malloc = 0;
static int max_producer_info_interval = 1;
// Purely for performance testing, should not be used live
static int dead_pe_processing = 1;
static FILE *profiling_fp = NULL;
static volatile int this_pe_has_exited = 0;

static const char *producer_info_segs_var_name = "HVR_PRODUCER_INFO_SEGS";

typedef struct _profiling_info_t {
    unsigned long long start_iter;
    unsigned long long end_start_time_step;
    unsigned long long end_update_vertices;
    unsigned long long end_update_dist;
    unsigned long long end_update_partitions;
    unsigned long long end_partition_window;
    unsigned long long end_neighbor_updates;
    unsigned long long end_send_updates;
    unsigned long long end_vertex_updates;
    unsigned long long end_update_coupled;
    int count_updated;
    unsigned n_updates_sent;
    process_perf_info_t perf_info;
    unsigned long long time_updating_partitions;
    unsigned long long time_updating_subscribers;
    unsigned long long time_1;
    unsigned long long time_2;
    unsigned long long time_3_4;
    unsigned long long time_5;
    unsigned long long time_sending;
    int should_abort;
    unsigned long long coupling_coupling;
    unsigned long long coupling_waiting;
    unsigned long long coupling_after;
    unsigned long long coupling_should_terminate;
    unsigned coupling_naborts;
    unsigned long long coupling_waiting_for_prev;
    unsigned long long coupling_processing_new_requests;
    unsigned long long coupling_sharing_info;
    unsigned long long coupling_waiting_for_info;
    unsigned long long coupling_negotiating;
    unsigned n_coupled_pes;
    int coupled_pes_root;
    hvr_time_t iter;
    int pe;
    hvr_partition_t n_producer_partitions;
    hvr_partition_t n_partitions;
    hvr_partition_t n_subscriber_partitions;
    uint64_t n_allocated_verts;
    unsigned long long n_mirrored_verts;

#ifdef PRINT_PARTITIONS
    char *subscriber_partitions_str;
    char *producer_partitions_str;
#endif

#ifdef DETAILED_PRINTS
    size_t remote_partition_subs_bytes;
    size_t remote_vert_subs_bytes;
    size_t my_vert_subs_bytes;
    size_t producer_info_bytes;
    size_t dead_info_bytes;
    size_t vertex_cache_allocated;
    size_t vertex_cache_used;
    size_t vertex_cache_symm_allocated;
    size_t vertex_cache_symm_used;
    unsigned vertex_cache_max_len;

    size_t edge_bytes_used;
    size_t edge_bytes_allocated;
    size_t edge_bytes_capacity;
    size_t max_edges;
    size_t max_edges_index;

    size_t mailbox_bytes_used;

    size_t msg_buf_pool_bytes_used;

    size_t buffered_msgs_bytes_used;

    size_t partition_producers_bytes_used;
    size_t terminated_pes_bytes_used;

    size_t mirror_partition_lists_bytes;
    size_t local_partition_lists_bytes;

    size_t active_partition_lists_bytes;
#endif
} profiling_info_t;

profiling_info_t saved_profiling_info[MAX_PROFILED_ITERS];
static volatile unsigned n_profiled_iters = 0;

// From https://stackoverflow.com/questions/1558402/memory-usage-of-current-process-in-c
typedef struct {
    unsigned long size,resident,share,text,lib,data,dt;
} statm_t;

void read_off_memory_status(statm_t *result) {
    unsigned long dummy;
    const char* statm_path = "/proc/self/statm";

    FILE *f = fopen(statm_path,"r");
    if(!f){
        perror(statm_path);
        abort();
    }
    if (7 != fscanf(f,"%ld %ld %ld %ld %ld %ld %ld",
                &result->size, &result->resident, &result->share, &result->text,
                &result->lib, &result->data, &result->dt)) {
        perror(statm_path);
        abort();
    }
    fclose(f);
}

static void print_memory_metrics(hvr_internal_ctx_t *ctx) {
    int pagesize = getpagesize();
    statm_t mem;
    read_off_memory_status(&mem);
    fprintf(stderr, "PE %d iter %u VM=%f MB resident=%f MB share=%f MB text=%f MB "
            "data=%f MB\n", ctx->pe, ctx->iter,
            (double)(mem.size * pagesize) / (1024.0 * 1024.0),
            (double)(mem.resident * pagesize) / (1024.0 * 1024.0),
            (double)(mem.share * pagesize) / (1024.0 * 1024.0),
            (double)(mem.text * pagesize) / (1024.0 * 1024.0),
            (double)(mem.data * pagesize) / (1024.0 * 1024.0));
}

typedef enum {
    CREATE, DELETE, UPDATE
} hvr_change_type_t;

static void handle_new_vertex(hvr_vertex_t *new_vert,
        process_perf_info_t *perf_info,
        hvr_internal_ctx_t *ctx);

static void handle_deleted_vertex(hvr_vertex_t *dead_vert,
        int expect_no_edges,
        hvr_internal_ctx_t *ctx);

static unsigned process_vertex_updates(hvr_internal_ctx_t *ctx,
        process_perf_info_t *perf_info);

static hvr_vertex_cache_node_t *set_up_vertex_subscription(hvr_vertex_id_t v,
        hvr_vertex_t *optional_body,
        hvr_internal_ctx_t *ctx);

void *malloc_helper(size_t nbytes) {
    static size_t count_bytes = 0;
    if (nbytes == 0) {
        fprintf(stderr, "PE %d allocated %lu bytes in the heap.\n",
                shmem_my_pe(), count_bytes);
        return NULL;
    }

    void *p = malloc(nbytes);
    count_bytes += nbytes;
    if (p) {
        memset(p, 0xff, nbytes);
    }
    return p;
}

void *shmem_malloc_wrapper(size_t nbytes) {
    static size_t total_nbytes = 0;
    static FILE *fp = NULL;

    if (trace_shmem_malloc) {
        if (fp == NULL) {
            char dump_file_name[256];
            sprintf(dump_file_name, "shmem_malloc_%d.txt", shmem_my_pe());
            fp = fopen(dump_file_name, "w");
            assert(fp);
        }

        fprintf(fp, "%d shmem_malloc %lu\n", shmem_my_pe(), nbytes);
        fflush(fp);
    }

    if (nbytes == 0) {
        fprintf(stderr, "PE %d allocated %lu bytes in the symmetric heap.\n",
                shmem_my_pe(), total_nbytes);
        return NULL;
    } else {
        void *ptr = shmem_malloc(nbytes);
        total_nbytes += nbytes;
        return ptr;
    }
}

void hvr_ctx_create(hvr_ctx_t *out_ctx) {
    hvr_internal_ctx_t *new_ctx = (hvr_internal_ctx_t *)malloc_helper(
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

    if (getenv("HVR_TRACE_SHMALLOC")) {
        trace_shmem_malloc = atoi(getenv("HVR_TRACE_SHMALLOC"));
    }

    hvr_vertex_cache_init(&new_ctx->vec_cache);

    if (dead_pe_processing) {
        new_ctx->vertex_partitions = (hvr_partition_t *)shmem_malloc_wrapper(
                new_ctx->vec_cache.pool_size * sizeof(hvr_partition_t));
    }

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

static void collect_impacted_vertices(hvr_vertex_cache_node_t *node,
        hvr_vertex_cache_node_t **head, hvr_internal_ctx_t *ctx) {
    if (node->tmp) {
        // Already traversed and in the list
        return;
    }

    node->tmp = *head;
    *head = node;

    hvr_edge_info_t *edge_buffer;
    unsigned n_neighbors = hvr_irr_matrix_linearize_zero_copy(
            CACHE_NODE_OFFSET(node, &ctx->vec_cache),
            &edge_buffer, &ctx->edges);

    for (unsigned n = 0; n < n_neighbors; n++) {
        hvr_vertex_id_t offset = EDGE_INFO_VERTEX(edge_buffer[n]);
        hvr_vertex_cache_node_t *neighbor = CACHE_NODE_BY_OFFSET(offset,
                &ctx->vec_cache);
        if (neighbor->dist_from_local_vert == node->dist_from_local_vert + 1) {
            collect_impacted_vertices(neighbor, head, ctx);
        }
    }

    node->dist_from_local_vert = UINT8_MAX;
}

static int update_dist_from_neighbors(hvr_vertex_cache_node_t *node,
        hvr_internal_ctx_t *ctx) {
    uint8_t new_dist = node->dist_from_local_vert;

    hvr_edge_info_t *edge_buffer;
    unsigned n_neighbors = hvr_irr_matrix_linearize_zero_copy(
            CACHE_NODE_OFFSET(node, &ctx->vec_cache),
            &edge_buffer, &ctx->edges);

    for (unsigned n = 0; n < n_neighbors; n++) {
        hvr_vertex_id_t offset = EDGE_INFO_VERTEX(edge_buffer[n]);
        hvr_vertex_cache_node_t *neighbor = CACHE_NODE_BY_OFFSET(offset,
                &ctx->vec_cache);
        uint8_t dist = neighbor->dist_from_local_vert;
        if (dist != UINT8_MAX && dist + 1 < new_dist) {
            new_dist = dist + 1;
        }
    }

    int changed = (node->dist_from_local_vert != new_dist);
    node->dist_from_local_vert = new_dist;
    return changed;
}

static void update_distance_after_edge_update(hvr_vertex_cache_node_t *node,
        hvr_internal_ctx_t *ctx) {
    hvr_vertex_cache_node_t *head = NULL;
    collect_impacted_vertices(node, &head, ctx);

    int changed;
    do {
        changed = 0;
        hvr_vertex_cache_node_t *iter = head;
        while (iter) {
            changed = (changed || update_dist_from_neighbors(iter, ctx));
            iter = iter->tmp;
        }
    } while (changed);

    // Clear our list
    while (head) {
        hvr_vertex_cache_node_t *next = head->tmp;
        head->tmp = NULL;
        head = next;
    }
}

// The only place where edges between vertices are created/deleted
static inline void update_edge_info(hvr_vertex_id_t base_id,
        hvr_vertex_id_t neighbor_id,
        hvr_vertex_cache_node_t *base,
        hvr_vertex_cache_node_t *neighbor,
        hvr_edge_type_t new_edge,
        const hvr_edge_type_t *known_existing_edge,
        const hvr_edge_create_type_t creation_type,
        hvr_internal_ctx_t *ctx) {
    assert(base && neighbor);

    const int base_is_local = (VERTEX_ID_PE(base_id) == ctx->pe);
    const int neighbor_is_local = (VERTEX_ID_PE(neighbor_id) == ctx->pe);

    hvr_vertex_id_t neighbor_offset = CACHE_NODE_OFFSET(neighbor,
            &ctx->vec_cache);
    hvr_vertex_id_t base_offset = CACHE_NODE_OFFSET(base, &ctx->vec_cache);

    hvr_edge_type_t existing_edge = (known_existing_edge ?
            *known_existing_edge :
            hvr_irr_matrix_get(CACHE_NODE_OFFSET(base, &ctx->vec_cache),
                CACHE_NODE_OFFSET(neighbor, &ctx->vec_cache), &ctx->edges));

    // Explicitly created edges should not pre-exist
    assert(creation_type == IMPLICIT_EDGE ||
            (new_edge != NO_EDGE && (existing_edge == NO_EDGE ||
                                     existing_edge == new_edge)));

    if (existing_edge == new_edge) return;

    if (existing_edge == NO_EDGE) {
        // new edge != NO_EDGE (creating a completely new edge)
        hvr_irr_matrix_set(base_offset, neighbor_offset, new_edge,
                creation_type, &ctx->edges, 1);
        hvr_irr_matrix_set(neighbor_offset, base_offset,
                flip_edge_direction(new_edge), creation_type, &ctx->edges, 1);

        base->n_local_neighbors += neighbor_is_local;
        neighbor->n_local_neighbors += base_is_local;

        /*
         * Find if either vertex's distance-to-local is < the other's
         * minus one. If so, update the other's distance to be the new lower
         * value, and cascade those updates through any neighbors whose
         * distances are == the old value + 1.
         */
        if (base->dist_from_local_vert < neighbor->dist_from_local_vert - 1) {
            // Need to update neighbor and cascade
            update_distance_after_edge_update(neighbor, ctx);
        } else if (neighbor->dist_from_local_vert <
                base->dist_from_local_vert - 1) {
            // Need to update base and cascade
            update_distance_after_edge_update(base, ctx);
        }
    } else if (new_edge == NO_EDGE) {
        // existing edge != NO_EDGE (deleting an existing edge)
        hvr_irr_matrix_set(base_offset, neighbor_offset, NO_EDGE, creation_type,
                &ctx->edges, 0);
        hvr_irr_matrix_set(neighbor_offset, base_offset, NO_EDGE, creation_type,
                &ctx->edges, 0);

        // Decrement if condition holds true
        base->n_local_neighbors -= neighbor_is_local;
        neighbor->n_local_neighbors -= base_is_local;

        /*
         * Check if either vertex's distance is equal to the other's
         * distance + 1. If so, its shortest path may be through that neighbor.
         * Re-compute and update its distances.
         */
        assert(abs(base->dist_from_local_vert -
                    neighbor->dist_from_local_vert) <= 1);
        if (base->dist_from_local_vert == neighbor->dist_from_local_vert + 1) {
            // Need to update base
            update_distance_after_edge_update(base, ctx);
        } else if (neighbor->dist_from_local_vert ==
                base->dist_from_local_vert + 1) {
            // Need to update neighbor
            update_distance_after_edge_update(neighbor, ctx);
        }
    } else {
        // Neither new or existing is NO_EDGE (updating existing edge)
        hvr_irr_matrix_set(base_offset, neighbor_offset, new_edge,
                creation_type, &ctx->edges, 0);
        hvr_irr_matrix_set(neighbor_offset, base_offset, new_edge,
                creation_type, &ctx->edges, 0);
    }

    if (creation_type == EXPLICIT_EDGE) {
        base->n_explicit_edges++;
        neighbor->n_explicit_edges++;
    }

    /*
     * If this edge update involves one local vertex and one remote, the
     * remote may require a change in local neighbor list membership (either
     * addition or deletion).
     */
    if (base_is_local != neighbor_is_local &&
            (base_is_local || neighbor_is_local)) {
        hvr_vertex_cache_node_t *remote_node = (base_is_local ? neighbor :
                base);

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
     * Vertex attributes only need updating if this is an edge inbound in a
     * given vertex (either directed in or bidirectional). new_edge direction
     * is expressed relative to base (i.e. DIRECTED_IN means it is an inbound
     * edge on base, and outbound on neighbor).
     */
    if (base_is_local && new_edge != DIRECTED_OUT) {
        mark_for_processing(&base->vert, ctx);
    }

    if (neighbor_is_local && flip_edge_direction(new_edge) != DIRECTED_OUT) {
        mark_for_processing(&neighbor->vert, ctx);
    }
}


static void mark_all_downstream_neighbors_for_processing(
        hvr_vertex_cache_node_t *modified, hvr_internal_ctx_t *ctx) {
    hvr_edge_info_t *edge_buffer;
    unsigned n_neighbors = hvr_irr_matrix_linearize_zero_copy(
            CACHE_NODE_OFFSET(modified, &ctx->vec_cache),
            &edge_buffer, &ctx->edges);

    for (unsigned n = 0; n < n_neighbors; n++) {
        hvr_vertex_id_t offset = EDGE_INFO_VERTEX(edge_buffer[n]);
        hvr_edge_type_t dir = EDGE_INFO_EDGE(edge_buffer[n]);
        hvr_vertex_cache_node_t *neighbor = CACHE_NODE_BY_OFFSET(offset,
                &ctx->vec_cache);

        int home_pe = hvr_vertex_get_owning_pe(&neighbor->vert);

        if (hvr_vertex_get_owning_pe(&neighbor->vert) == ctx->pe &&
                dir != DIRECTED_IN) {
            mark_for_processing(&neighbor->vert, ctx);
        }
    }
}

/*
* Figure out what edges need to be added here, from should_have_edge and then
* insert them for the new vertex. Eventually, any local vertex which had a new
* edge inserted will need to be updated.
*/
static unsigned create_new_edges(hvr_vertex_cache_node_t *updated,
        const hvr_partition_t *interacting, unsigned n_interacting,
        hvr_partition_list_t *partition_lists,
        hvr_internal_ctx_t *ctx) {
    assert(updated->vert.curr_part != HVR_INVALID_PARTITION);

    unsigned local_count_new_should_have_edges = 0;
    const hvr_edge_type_t no_edge = NO_EDGE;

    for (unsigned i = 0; i < n_interacting; i++) {
        hvr_partition_t other_part = interacting[i];

        hvr_vertex_t *cache_iter = hvr_partition_list_head(other_part,
            partition_lists);
        while (cache_iter) {
            hvr_vertex_cache_node_t *cache_node =
                (hvr_vertex_cache_node_t *)cache_iter;
            if (!cache_node->flag) {
                hvr_edge_type_t edge = ctx->should_have_edge(cache_iter,
                        &updated->vert, ctx);
                update_edge_info(cache_iter->id, updated->vert.id, cache_node,
                        updated, edge, &no_edge, IMPLICIT_EDGE, ctx);
                local_count_new_should_have_edges++;
            }

            cache_iter = cache_iter->next_in_partition;
        }
    }

    return local_count_new_should_have_edges;
}

static unsigned update_existing_edges(hvr_vertex_cache_node_t *updated,
        const hvr_partition_t *interacting,
        unsigned n_interacting, hvr_internal_ctx_t *ctx) {
    assert(updated->vert.curr_part != HVR_INVALID_PARTITION);

    unsigned local_count_new_should_have_edges = 0;

    /*
     * Look for existing edges and verify they should still exist with
     * this update to the local mirror.
     */
    unsigned n_neighbors = hvr_irr_matrix_linearize(
            CACHE_NODE_OFFSET(updated, &ctx->vec_cache),
            ctx->edge_buffer, MAX_MODIFICATIONS, &ctx->edges);

    for (unsigned n = 0; n < n_neighbors; n++) {
        hvr_vertex_cache_node_t *cached_neighbor = CACHE_NODE_BY_OFFSET(
                EDGE_INFO_VERTEX(ctx->edge_buffer[n]), &ctx->vec_cache);
        hvr_edge_type_t edge = EDGE_INFO_EDGE(ctx->edge_buffer[n]);

        // Check this edge should still exist
        hvr_edge_type_t new_edge = ctx->should_have_edge(
                &updated->vert, &(cached_neighbor->vert), ctx);
        update_edge_info(updated->vert.id, cached_neighbor->vert.id,
                updated, cached_neighbor, new_edge, &edge, IMPLICIT_EDGE, ctx);

        // Mark that we've already handled it
        cached_neighbor->flag = 1;
    }

    const unsigned long long done_updating_edges = hvr_current_time_us();

    local_count_new_should_have_edges += create_new_edges(updated, interacting,
            n_interacting, &ctx->mirror_partition_lists, ctx);
    local_count_new_should_have_edges += create_new_edges(updated, interacting,
            n_interacting, &ctx->local_partition_lists, ctx);

    // Clear the flag
    for (unsigned n = 0; n < n_neighbors; n++) {
        hvr_vertex_cache_node_t *cached_neighbor = CACHE_NODE_BY_OFFSET(
                EDGE_INFO_VERTEX(ctx->edge_buffer[n]), &ctx->vec_cache);
        cached_neighbor->flag = 0;
    }

    return local_count_new_should_have_edges;
}


static void insert_recently_created_in_partitions(hvr_internal_ctx_t *ctx) {
    hvr_partition_list_t *local_partition_lists = &ctx->local_partition_lists;

    while (ctx->recently_created) {
        hvr_vertex_t *curr = ctx->recently_created;

        ctx->recently_created = curr->next_in_partition;

        hvr_partition_t part = ctx->actor_to_partition(curr, ctx);
        curr->curr_part = part;
        curr->prev_part = HVR_INVALID_PARTITION;

        if (part != HVR_INVALID_PARTITION) {
            prepend_to_partition_list(curr, part, local_partition_lists, ctx);

            unsigned n_interacting = 0;
            ctx->might_interact(part, ctx->interacting, &n_interacting,
                    MAX_INTERACTING_PARTITIONS, ctx);

            update_existing_edges((hvr_vertex_cache_node_t *)curr,
                    ctx->interacting, n_interacting, ctx);
        } else {
            curr->next_in_partition = NULL;
            curr->prev_in_partition = NULL;
        }

        /*
         * No need for mark_all_downstream_neighbors_for_processing because the
         * creation of new edges above should do that for us.
         */
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
            hvr_dist_bitvec_set(p, ctx->pe, registry, HVR_IS_MULTITHREADED);
        } else {
            // Went inactive
            hvr_dist_bitvec_clear(p, ctx->pe, registry, HVR_IS_MULTITHREADED);
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
    const size_t pool_size = ctx->vec_cache.pool_size;
    for (unsigned i = 0; i < pool_size; i += N_VERTICES_PER_BUF) {
        unsigned n_this_iter = pool_size - i;
        if (n_this_iter > N_VERTICES_PER_BUF) {
            n_this_iter = N_VERTICES_PER_BUF;
        }

        shmem_getmem(ctx->vert_partition_buf, ctx->vertex_partitions + i,
                n_this_iter * sizeof(ctx->vert_partition_buf[0]), dead_pe);

        for (unsigned j = 0; j < n_this_iter; j++) {
            if (ctx->vert_partition_buf[j] == partition) {
                shmem_getmem(&tmp_vert,
                        ctx->vec_cache.pool_mem + (i + j),
                        sizeof(tmp_vert), dead_pe);

                process_perf_info_t dummy_perf_info;
                handle_new_vertex(&tmp_vert, &dummy_perf_info, ctx);
            }
        }
    }
}

static void handle_new_subscription(hvr_partition_t p,
        hvr_internal_ctx_t *ctx) {
    hvr_dist_bitvec_local_subcopy_t *p_dead_info =
        (hvr_dist_bitvec_local_subcopy_t *)malloc_helper(sizeof(*p_dead_info));
    assert(p_dead_info);
    hvr_dist_bitvec_local_subcopy_init(&ctx->terminated_pes,
            p_dead_info);
    hvr_map_add(p, p_dead_info, 0, &ctx->dead_info);

    hvr_dist_bitvec_local_subcopy_t *p_producer_info =
        (hvr_dist_bitvec_local_subcopy_t *)malloc_helper(
                sizeof(*p_producer_info));
    assert(p_producer_info);
    hvr_dist_bitvec_local_subcopy_init(&ctx->partition_producers,
            p_producer_info);
    hvr_map_add(p, p_producer_info, 0, &ctx->producer_info);


    // Download the list of producers for partition p.
    hvr_dist_bitvec_copy_locally(p, &ctx->partition_producers, p_producer_info);

    /*
     * notify all producers of this partition of our subscription
     * (they will then send us a full update).
     */
    hvr_partition_member_change_t change;
    change.pe = ctx->pe;  // The subscriber/unsubscriber
    change.partition = p; // The partition (un)subscribed to
    change.entered = 1;   // new subscription
    for (int pe = 0; pe < ctx->npes; pe++) {
        if (hvr_dist_bitvec_local_subcopy_contains(pe, p_producer_info)) {
            hvr_mailbox_send(&change, sizeof(change), pe,
                    -1, &ctx->forward_mailbox, HVR_IS_MULTITHREADED);
        }
    }

    if (dead_pe_processing) {
        const unsigned long long start = hvr_current_time_us();
        hvr_dist_bitvec_copy_locally(p, &ctx->terminated_pes,
                p_dead_info);
        for (int pe = 0; pe < ctx->npes; pe++) {
            if (hvr_dist_bitvec_local_subcopy_contains(pe,
                        p_dead_info)) {
#ifdef HVR_MULTITHREADED
#pragma omp critical
#endif
                pull_vertices_from_dead_pe(pe, p, ctx);
            }
        }
    }
}

static void handle_existing_subscription(hvr_partition_t p,
        hvr_internal_ctx_t *ctx) {
    if (ctx->iter < ctx->next_producer_info_check[p]) {
        return;
    }

    uint64_t curr_seq_no = hvr_dist_bitvec_get_seq_no(p,
            &ctx->partition_producers);

    hvr_dist_bitvec_local_subcopy_t *p_producer_info = hvr_map_get(p,
            &ctx->producer_info);
    assert(p_producer_info);

    if (curr_seq_no <= p_producer_info->seq_no) {
        // No change yet again, back off further
        hvr_time_t new_backoff = 2 * ctx->curr_producer_info_interval[p];
        if (new_backoff > max_producer_info_interval) {
            new_backoff = max_producer_info_interval;
        }
        ctx->next_producer_info_check[p] = ctx->iter + new_backoff;
        ctx->curr_producer_info_interval[p] = new_backoff;
        return;
    }

    // Reset
    ctx->next_producer_info_check[p] = ctx->iter + 1;
    ctx->curr_producer_info_interval[p] = 1;

    hvr_dist_bitvec_copy_locally(p, &ctx->partition_producers,
            &ctx->local_partition_producers);

    hvr_partition_member_change_t change;
    change.pe = ctx->pe;  // The subscriber/unsubscriber
    change.partition = p; // The partition (un)subscribed to
    change.entered = 1;   // new subscription

    /*
     * Notify any new producers of this partition that I'm
     * interested in their updates.
     */
    for (int pe = 0; pe < ctx->npes; pe++) {
        if (!hvr_dist_bitvec_local_subcopy_contains(pe, p_producer_info) &&
                hvr_dist_bitvec_local_subcopy_contains(pe,
                    &ctx->local_partition_producers)) {
            // New producer
            hvr_mailbox_send(&change, sizeof(change), pe,
                    -1, &ctx->forward_mailbox, HVR_IS_MULTITHREADED);
        }
    }

    // Save for later
    hvr_dist_bitvec_local_subcopy_copy(p_producer_info,
            &ctx->local_partition_producers);

    /*
     * Look for newly terminated PEs that had local vertices in a
     * given partition. Go grab their vertices if there's a new one.
     */

    if (dead_pe_processing) {
        hvr_dist_bitvec_copy_locally(p, &ctx->terminated_pes,
                &ctx->local_terminated_pes);

        hvr_dist_bitvec_local_subcopy_t *p_dead_info = hvr_map_get(p,
                &ctx->dead_info);
        assert(p_dead_info);

        for (int pe = 0; pe < ctx->npes; pe++) {
            if (!hvr_dist_bitvec_local_subcopy_contains(pe, p_dead_info) &&
                    hvr_dist_bitvec_local_subcopy_contains(pe,
                        &ctx->local_terminated_pes)) {
                // New dead PE
#ifdef HVR_MULTITHREADED
#pragma omp critical
#endif
                pull_vertices_from_dead_pe(pe, p, ctx);
            }
        }

        hvr_dist_bitvec_local_subcopy_copy(p_dead_info,
                &ctx->local_terminated_pes);
    }
}

static void handle_new_unsubscription(hvr_partition_t p,
        hvr_internal_ctx_t *ctx) {
    hvr_dist_bitvec_local_subcopy_t *p_producer_info = hvr_map_get(p,
            &ctx->producer_info);
    assert(p_producer_info);

    hvr_partition_member_change_t change;
    change.pe = ctx->pe;  // The subscriber/unsubscriber
    change.partition = p; // The partition (un)subscribed to
    change.entered = 0;   // unsubscription
    for (int pe = 0; pe < ctx->npes; pe++) {
        if (hvr_dist_bitvec_local_subcopy_contains(pe, p_producer_info)) {
            hvr_mailbox_send(&change, sizeof(change), pe,
                    -1, &ctx->forward_mailbox, HVR_IS_MULTITHREADED);
        }
    }

    /*
     * Invalidate all vertices cached in this partition because we won't be
     * getting new updates.
     */
#ifdef HVR_MULTITHREADED
#pragma omp critical
    {
#endif
    hvr_vertex_t *iter = hvr_partition_list_head(p,
            &ctx->mirror_partition_lists);
    while (iter) {
        hvr_vertex_t *next = iter->next_in_partition;
        if (hvr_vertex_get_owning_pe(iter) != ctx->pe) {
            handle_deleted_vertex(iter, 1, ctx);
        }
        iter = next;
    }
#ifdef HVR_MULTITHREADED
    }
#endif

    hvr_map_remove(p, p_producer_info, &ctx->producer_info);
    hvr_dist_bitvec_local_subcopy_destroy(&ctx->partition_producers,
            p_producer_info);

    hvr_dist_bitvec_local_subcopy_t *p_dead_info = hvr_map_get(p,
            &ctx->dead_info);
    assert(p_dead_info);
    hvr_map_remove(p, p_dead_info, &ctx->dead_info);
    hvr_dist_bitvec_local_subcopy_destroy(&ctx->terminated_pes, p_dead_info);

    free(p_producer_info);
    free(p_dead_info);
}

static inline void add_interacting_partitions(hvr_vertex_id_t p, hvr_set_t *s,
        hvr_internal_ctx_t *ctx, size_t *n_subscriber_partitions) {
    unsigned n_interacting = 0;
    hvr_partition_t *interacting = ctx->interacting;
    ctx->might_interact(p, interacting, &n_interacting,
            MAX_INTERACTING_PARTITIONS, ctx);

    /*
     * Mark any partitions which this locally active partition might
     * interact with.
     */
    for (unsigned i = 0; i < n_interacting; i++) {
        if (!hvr_set_contains(interacting[i], s)) {
            assert(*n_subscriber_partitions < ctx->max_active_partitions);
            ctx->new_subscriber_partitions_list[*n_subscriber_partitions] =
                interacting[i];
            *n_subscriber_partitions += 1;

            hvr_set_insert(interacting[i], s);
        }
    }
}

static void update_partition_window(hvr_internal_ctx_t *ctx,
        unsigned long long *out_time_updating_partitions,
        unsigned long long *out_time_updating_subscribers,
        unsigned long long *out_time_1,
        unsigned long long *out_time_2,
        unsigned long long *out_time_3_4,
        unsigned long long *out_time_5) {
    const unsigned long long start = hvr_current_time_us();

    hvr_set_t *new_subscribed_partitions = ctx->new_subscribed_partitions;
    hvr_set_t *new_produced_partitions = ctx->new_produced_partitions;
    hvr_set_wipe(new_subscribed_partitions);
    hvr_set_wipe(new_produced_partitions);

    size_t n_producer_partitions = 0;
    size_t n_subscriber_partitions = 0;

    for (int b = 0; b < HVR_MAP_BUCKETS; b++) {
        hvr_map_seg_t *iter = ctx->local_partition_lists.map.buckets[b];
        while (iter) {
            for (int i = 0; i < iter->nkeys; i++) {
                hvr_vertex_id_t p = iter->data[i].key;

                // Producer
                if (!hvr_set_contains(p, new_produced_partitions)) {
                    assert(n_producer_partitions < ctx->max_active_partitions);
                    ctx->new_producer_partitions_list[n_producer_partitions++] =
                        p;

                    hvr_set_insert(p, new_produced_partitions);
                }

                // Subscriber to any interacting partitions with p
                add_interacting_partitions(p, new_subscribed_partitions, ctx,
                        &n_subscriber_partitions);
            }
            iter = iter->next;
        }
    }

    for (int b = 0; b < HVR_MAP_BUCKETS; b++) {
        hvr_map_seg_t *iter = ctx->mirror_partition_lists.map.buckets[b];
        while (iter) {
            for (int i = 0; i < iter->nkeys; i++) {
                hvr_vertex_id_t p = iter->data[i].key;
                if (ctx->partition_min_dist_from_local_vert[p] <=
                        ctx->max_graph_traverse_depth - 1) {
                    // Subscriber to any interacting partitions with p
                    add_interacting_partitions(p,
                            new_subscribed_partitions, ctx,
                            &n_subscriber_partitions);
                }
            }
            iter = iter->next;
        }
    }

    const unsigned long long after_part_updates = hvr_current_time_us();

    /*
     * There is work to do in the following five cases:
     *
     *   1. A partition we stopped being a producer for.
     *   2. A partition we are now a producer for.
     *   3. A new subscription.
     *   4. A continuing subscription (from the previous iteration).
     *   5. A new unsubscription (that was a subscription on the previous
     *      iteration).
     */

    // ***** #1 (stopped producing) *****
    for (size_t i = 0; i < ctx->n_prev_producer_partitions; i++) {
        hvr_partition_t p = ctx->prev_producer_partitions_list[i];
        if (!hvr_set_contains(p, new_produced_partitions)) {
            hvr_dist_bitvec_clear(p, ctx->pe, &ctx->partition_producers,
                    HVR_IS_MULTITHREADED);
        }
    }

    const unsigned long long after_1 = hvr_current_time_us();

    // ***** #2 (new producer) *****
    for (size_t i = 0; i < n_producer_partitions; i++) {
        hvr_partition_t p = ctx->new_producer_partitions_list[i];
        if (!hvr_set_contains(p, ctx->produced_partitions)) {
            hvr_dist_bitvec_set(p, ctx->pe, &ctx->partition_producers,
                    HVR_IS_MULTITHREADED);
        }
    }

    const unsigned long long after_2 = hvr_current_time_us();

    // ***** #3 (new sub) and #4 (existing sub) *****
    for (size_t i = 0; i < n_subscriber_partitions; i++) {
        hvr_partition_t p = ctx->new_subscriber_partitions_list[i];
        const int old_sub = hvr_set_contains(p, ctx->subscribed_partitions);
        if (!old_sub) {
            /*
             * New subscription on this iteration. Copy down the set of
             * producers for this partition and send each a notification
             * that I'm now subscribed to that partition. Also copy down
             * the set of dead PEs that were producers of this partition,
             * and pull vertices from those PEs.
             */
            handle_new_subscription(p, ctx);
            ctx->next_producer_info_check[p] = ctx->iter + 1;
            ctx->curr_producer_info_interval[p] = 1;
        } else {
            /*
             * If this is an existing subscription, copy down the current
             * list of producers and check for changes. If a change is
             * found, notify the new producer that we're subscribed.
             */
            handle_existing_subscription(p, ctx);
        }
    }

    const unsigned long long after_3_4 = hvr_current_time_us();

    // ***** #5 (unsubscription) *****
    for (size_t i = 0; i < ctx->n_prev_subscriber_partitions; i++) {
        hvr_partition_t p = ctx->prev_subscriber_partitions_list[i];
        if (!hvr_set_contains(p, new_subscribed_partitions)) {
            /*
             * If this is a new unsubscription notify all producers that we
             * are no longer subscribed. They will remove us from the list
             * of people they send msgs to.
             */
            handle_new_unsubscription(p, ctx);
        }
    }

    const unsigned long long after_5 = hvr_current_time_us();

    /*
     * Copy the newly computed partition windows for this PE over for next time
     * when we need to check for changes.
     */
    hvr_set_copy(ctx->subscribed_partitions, new_subscribed_partitions);

    hvr_set_copy(ctx->produced_partitions, new_produced_partitions);

    hvr_partition_t *tmp_p = ctx->prev_producer_partitions_list;
    hvr_partition_t *tmp_s = ctx->prev_subscriber_partitions_list;

    ctx->prev_producer_partitions_list = ctx->new_producer_partitions_list;
    ctx->prev_subscriber_partitions_list = ctx->new_subscriber_partitions_list;
    ctx->n_prev_producer_partitions = n_producer_partitions;
    ctx->n_prev_subscriber_partitions = n_subscriber_partitions;

    ctx->new_producer_partitions_list = tmp_p;
    ctx->new_subscriber_partitions_list = tmp_s;

    const unsigned long long end = hvr_current_time_us();

    *out_time_updating_partitions = after_part_updates - start;
    *out_time_1 = after_1 - after_part_updates;
    *out_time_2 = after_2 - after_1;
    *out_time_3_4 = after_3_4 - after_2;
    *out_time_5 = after_5 - after_3_4;
    *out_time_updating_subscribers = end - after_part_updates;
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
    /*
     * For casting from hvr_vertex_t* in mirror_partition_lists to
     * hvr_vertex_cache_node_t*
     */
    assert(offsetof(hvr_vertex_cache_node_t, vert) == 0);

    hvr_internal_ctx_t *new_ctx = (hvr_internal_ctx_t *)in_ctx;

    assert(new_ctx->initialized == 0);
    new_ctx->initialized = 1;

    assert(max_graph_traverse_depth >= 1);

    new_ctx->thread_safe = (shmemx_query_thread() == SHMEM_THREAD_MULTIPLE);
#ifdef HVR_MULTITHREADED
    assert(new_ctx->thread_safe);

#pragma omp parallel
    {
        shmemx_thread_register();
#pragma omp single
        new_ctx->nthreads = omp_get_num_threads();
    }
#endif

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

    assert(n_partitions < HVR_INVALID_PARTITION);
    new_ctx->n_partitions = n_partitions;

    new_ctx->subscribed_partitions = hvr_create_empty_set(n_partitions);
    new_ctx->produced_partitions = hvr_create_empty_set(n_partitions);
    new_ctx->new_subscribed_partitions = hvr_create_empty_set(n_partitions);
    new_ctx->new_produced_partitions = hvr_create_empty_set(n_partitions);

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
        sprintf(dump_file_name, "%d.vertices.csv", new_ctx->pe);
        new_ctx->dump_file = fopen(dump_file_name, "w");
        assert(new_ctx->dump_file);

        sprintf(dump_file_name, "%d.edges.csv", new_ctx->pe);
        new_ctx->edges_dump_file = fopen(dump_file_name, "w");
        assert(new_ctx->edges_dump_file);

        if (getenv("HVR_TRACE_DUMP_ONLY_LAST")) {
            new_ctx->only_last_iter_dump = 1;
        }
    }

    if (getenv("HVR_CACHE_TRACE_DUMP")) {
        new_ctx->cache_dump_mode = 1;

        char dump_file_name[256];
        sprintf(dump_file_name, "%d.vertices.cached.csv", new_ctx->pe);
        new_ctx->cache_dump_file = fopen(dump_file_name, "w");
        assert(new_ctx->cache_dump_file);
    }

    if (getenv("HVR_MAX_PRODUCER_INFO_INTERVAL")) {
        max_producer_info_interval = atoi(getenv(
                    "HVR_MAX_PRODUCER_INFO_INTERVAL"));
        assert(max_producer_info_interval > 0);
    }

    if (getenv("HVR_DISABLE_PROFILING_PRINTS")) {
        print_profiling = 0;
    } else {
        char profiling_filename[1024];
        sprintf(profiling_filename, "%d.prof", new_ctx->pe);
        profiling_fp = fopen(profiling_filename, "w");
        assert(profiling_fp);
        if (new_ctx->pe == 0) {
            fprintf(stderr, "WARNING: Using %lu bytes of space for profiling "
                    "data structures\n",
                    MAX_PROFILED_ITERS * sizeof(profiling_info_t));
        }
    }

    if (getenv("HVR_DISABLE_DEAD_PE_PROCESSING")) {
        dead_pe_processing = 0;
    }

    // Set up our initial coupled cluster
    new_ctx->coupled_pes = hvr_create_empty_set(new_ctx->npes);
    hvr_set_insert(new_ctx->pe, new_ctx->coupled_pes);
    new_ctx->coupled_pes_root = new_ctx->pe;

    new_ctx->prev_coupled_pes = hvr_create_empty_set(new_ctx->npes);
    hvr_set_insert(new_ctx->pe, new_ctx->prev_coupled_pes);

    new_ctx->to_couple_with = hvr_create_empty_set(new_ctx->npes);
    new_ctx->other_to_couple_with = hvr_create_empty_set(new_ctx->npes);
    new_ctx->other_coupled_pes = hvr_create_empty_set(new_ctx->npes);
    new_ctx->received_from = hvr_create_empty_set(new_ctx->npes);
    new_ctx->already_coupled_with = hvr_create_empty_set(new_ctx->npes);

    new_ctx->all_terminated_pes = hvr_create_empty_set(new_ctx->npes);
    new_ctx->all_terminated_cluster_pes = hvr_create_empty_set(new_ctx->npes);
    new_ctx->prev_all_terminated_cluster_pes = hvr_create_empty_set(new_ctx->npes);
    new_ctx->new_all_terminated_cluster_pes = hvr_create_empty_set(new_ctx->npes);
    new_ctx->terminating_pes = hvr_create_empty_set(new_ctx->npes);
    new_ctx->other_terminating_pes = hvr_create_empty_set(new_ctx->npes);

    hvr_set_msg_init(new_ctx->coupled_pes, &new_ctx->coupled_pes_msg);
    hvr_set_msg_init(new_ctx->to_couple_with, &new_ctx->to_couple_with_msg);
    hvr_set_msg_init(new_ctx->terminating_pes, &new_ctx->terminating_pes_msg);
    hvr_set_msg_init(new_ctx->new_all_terminated_cluster_pes,
            &new_ctx->new_all_terminated_cluster_pes_msg);

    new_ctx->coupled_pes_values = (hvr_vertex_t *)malloc_helper(
            new_ctx->npes * sizeof(hvr_vertex_t));
    assert(new_ctx->coupled_pes_values);
    for (unsigned i = 0; i < new_ctx->npes; i++) {
        hvr_vertex_init(&(new_ctx->coupled_pes_values)[i],
                HVR_INVALID_VERTEX_ID, new_ctx->iter);
    }
    new_ctx->updates_on_this_iter = (int *)malloc_helper(
            new_ctx->npes * sizeof(new_ctx->updates_on_this_iter[0]));
    assert(new_ctx->updates_on_this_iter);

    hvr_partition_list_init(new_ctx->n_partitions,
            &new_ctx->local_partition_lists);

    hvr_partition_list_init(new_ctx->n_partitions,
            &new_ctx->mirror_partition_lists);

    new_ctx->partition_min_dist_from_local_vert = (uint8_t *)malloc_helper(
            sizeof(new_ctx->partition_min_dist_from_local_vert[0]) *
            new_ctx->n_partitions);
    assert(new_ctx->partition_min_dist_from_local_vert);

    hvr_mailbox_init(&new_ctx->vertex_update_mailbox,        128 * 1024 * 1024);
    hvr_mailbox_init(&new_ctx->vertex_delete_mailbox,        128 * 1024 * 1024);
    hvr_mailbox_init(&new_ctx->forward_mailbox,               32 * 1024 * 1024);
    hvr_mailbox_init(&new_ctx->vert_sub_mailbox,              32 * 1024 * 1024);
    hvr_mailbox_init(&new_ctx->edge_create_mailbox,           32 * 1024 * 1024);
    hvr_mailbox_init(&new_ctx->vertex_msg_mailbox,            32 * 1024 * 1024);
    hvr_mailbox_init(&new_ctx->coupling_mailbox,              8 * 1024 * 1024);
    hvr_mailbox_init(&new_ctx->coupling_ack_and_dead_mailbox, 8 * 1024 * 1024);
    hvr_mailbox_init(&new_ctx->coupling_val_mailbox,          8 * 1024 * 1024);
    hvr_mailbox_init(&new_ctx->to_couple_with_mailbox,        8 * 1024 * 1024);
    hvr_mailbox_init(&new_ctx->root_info_mailbox,             8 * 1024 * 1024);

    const unsigned n_to_buffer = 32;
    hvr_mailbox_buffer_init(&new_ctx->vert_sub_mailbox_buffer,
            &new_ctx->vert_sub_mailbox, new_ctx->npes,
            sizeof(hvr_vertex_subscription_t), n_to_buffer);
    hvr_mailbox_buffer_init(&new_ctx->edge_create_mailbox_buffer,
            &new_ctx->edge_create_mailbox, new_ctx->npes,
            sizeof(hvr_edge_create_msg_t), n_to_buffer);
    hvr_mailbox_buffer_init(&new_ctx->vertex_update_mailbox_buffer,
            &new_ctx->vertex_update_mailbox, new_ctx->npes,
            sizeof(hvr_vertex_update_t), n_to_buffer);
    hvr_mailbox_buffer_init(&new_ctx->vertex_delete_mailbox_buffer,
            &new_ctx->vertex_delete_mailbox, new_ctx->npes,
            sizeof(hvr_vertex_update_t), n_to_buffer);

    hvr_dist_bitvec_init(new_ctx->n_partitions, new_ctx->npes,
            &new_ctx->partition_producers);
    hvr_dist_bitvec_init(new_ctx->n_partitions, new_ctx->npes,
            &new_ctx->terminated_pes);

    hvr_dist_bitvec_local_subcopy_init(&new_ctx->partition_producers,
            &new_ctx->local_partition_producers);
    hvr_dist_bitvec_local_subcopy_init(&new_ctx->terminated_pes,
            &new_ctx->local_terminated_pes);

    hvr_sparse_arr_init(&new_ctx->remote_partition_subs, new_ctx->n_partitions);
    hvr_sparse_arr_init(&new_ctx->remote_vert_subs,
            new_ctx->vec_cache.pool_size);
    hvr_sparse_arr_init(&new_ctx->my_vert_subs, new_ctx->npes);

    new_ctx->max_graph_traverse_depth = max_graph_traverse_depth;

    unsigned prealloc_segs = 768;
    if (getenv(producer_info_segs_var_name)) {
        prealloc_segs = atoi(getenv(producer_info_segs_var_name));
    }
    hvr_map_init(&new_ctx->producer_info, prealloc_segs,
            producer_info_segs_var_name);
    hvr_map_init(&new_ctx->dead_info, prealloc_segs,
            producer_info_segs_var_name);

    new_ctx->next_producer_info_check = (hvr_time_t *)malloc_helper(
            new_ctx->n_partitions *
            sizeof(new_ctx->next_producer_info_check[0]));
    assert(new_ctx->next_producer_info_check);
    memset(new_ctx->next_producer_info_check, 0x00, new_ctx->n_partitions *
            sizeof(new_ctx->next_producer_info_check[0]));

    new_ctx->curr_producer_info_interval = (hvr_time_t *)malloc_helper(
            new_ctx->n_partitions *
            sizeof(new_ctx->curr_producer_info_interval[0]));
    assert(new_ctx->curr_producer_info_interval);
    for (hvr_partition_t p = 0; p < new_ctx->n_partitions; p++) {
        new_ctx->curr_producer_info_interval[p] = 1;
    }

    new_ctx->vert_partition_buf = (hvr_partition_t *)malloc_helper(
            N_VERTICES_PER_BUF * sizeof(hvr_partition_t));
    assert(new_ctx->vert_partition_buf);

#define MAX_MACRO(my_a, my_b) ((my_a) > (my_b) ? (my_a) : (my_b))
    size_t max_msg_len = sizeof(hvr_vertex_update_t);
    max_msg_len = MAX_MACRO(sizeof(hvr_partition_member_change_t), max_msg_len);
    max_msg_len = MAX_MACRO(sizeof(hvr_dead_pe_msg_t), max_msg_len);
    max_msg_len = MAX_MACRO(sizeof(hvr_coupling_msg_t), max_msg_len);
    max_msg_len = MAX_MACRO(sizeof(new_coupling_msg_t), max_msg_len);
    max_msg_len = MAX_MACRO(sizeof(new_coupling_msg_ack_t), max_msg_len);
    max_msg_len = MAX_MACRO(sizeof(inter_vert_msg_t), max_msg_len);
    max_msg_len = MAX_MACRO(new_ctx->coupled_pes_msg.msg_buf_len, max_msg_len);
    max_msg_len = MAX_MACRO(
            new_ctx->vert_sub_mailbox_buffer.buffer_size_per_pe *
            sizeof(hvr_vertex_subscription_t), max_msg_len);
    max_msg_len = MAX_MACRO(
            new_ctx->edge_create_mailbox_buffer.buffer_size_per_pe *
            sizeof(hvr_edge_create_msg_t), max_msg_len);
    max_msg_len = MAX_MACRO(
            new_ctx->vertex_update_mailbox_buffer.buffer_size_per_pe *
            sizeof(hvr_vertex_update_t), max_msg_len);
    max_msg_len = MAX_MACRO(
            new_ctx->vertex_delete_mailbox_buffer.buffer_size_per_pe *
            sizeof(hvr_vertex_update_t), max_msg_len);

    unsigned max_buffered_msgs = 1000;
    if (getenv("HVR_MAX_BUFFERED_MSGS")) {
        max_buffered_msgs = atoi(getenv("HVR_MAX_BUFFERED_MSGS"));
    }
    hvr_msg_buf_pool_init(&new_ctx->msg_buf_pool, max_msg_len,
            max_buffered_msgs);

    new_ctx->interacting = (hvr_partition_t *)malloc_helper(
            MAX_INTERACTING_PARTITIONS * sizeof(new_ctx->interacting[0]));
    assert(new_ctx->interacting);
#ifdef HVR_MULTITHREADED
    new_ctx->per_thread_interacting = (hvr_partition_t *)malloc_helper(
            new_ctx->nthreads * MAX_INTERACTING_PARTITIONS *
            sizeof(new_ctx->per_thread_interacting[0]));
    assert(new_ctx->per_thread_interacting);
#endif

    size_t buffered_msgs_pool_size = 1024ULL * 1024ULL;
    if (getenv("HVR_BUFFERED_MSGS_POOL_SIZE")) {
        buffered_msgs_pool_size = atoi(getenv("HVR_BUFFERED_MSGS_POOL_SIZE"));
    }
    hvr_buffered_msgs_init(new_ctx->vec_cache.pool_size,
            buffered_msgs_pool_size, &new_ctx->buffered_msgs);

    // Initialize edges
    size_t edges_pool_size = 1024ULL * 1024ULL * 1024ULL;
    if (getenv("HVR_EDGES_POOL_SIZE")) {
        edges_pool_size = atoi(getenv("HVR_EDGES_POOL_SIZE"));
    }
    hvr_irr_matrix_init(new_ctx->vec_cache.pool_size, edges_pool_size,
            &new_ctx->edges);

    size_t max_active_partitions = 100000;
    if (getenv("HVR_MAX_ACTIVE_PARTITIONS")) {
        max_active_partitions = atoi(getenv("HVR_MAX_ACTIVE_PARTITIONS"));
    }
    new_ctx->new_producer_partitions_list = (hvr_partition_t *)malloc_helper(
            max_active_partitions *
            sizeof(new_ctx->new_producer_partitions_list[0]));
    new_ctx->new_subscriber_partitions_list = (hvr_partition_t *)malloc_helper(
            max_active_partitions *
            sizeof(new_ctx->new_subscriber_partitions_list[0]));
    new_ctx->prev_producer_partitions_list = (hvr_partition_t *)malloc_helper(
            max_active_partitions *
            sizeof(new_ctx->prev_producer_partitions_list[0]));
    new_ctx->prev_subscriber_partitions_list = (hvr_partition_t *)malloc_helper(
            max_active_partitions *
            sizeof(new_ctx->prev_subscriber_partitions_list[0]));
    assert(new_ctx->new_producer_partitions_list &&
            new_ctx->new_subscriber_partitions_list &&
            new_ctx->prev_producer_partitions_list &&
            new_ctx->prev_subscriber_partitions_list);
    new_ctx->max_active_partitions = max_active_partitions;

    new_ctx->n_prev_producer_partitions = 0;
    new_ctx->n_prev_subscriber_partitions = 0;
    new_ctx->any_needs_processing = 1;

    new_ctx->edge_list_pool_size = 128 * 1024;
    if (getenv("HVR_EDGE_LIST_POOL_SIZE")) {
        new_ctx->edge_list_pool_size = atoi(getenv("HVR_EDGE_LIST_POOL_SIZE"));
    }
    new_ctx->edge_list_pool = malloc_helper(new_ctx->edge_list_pool_size);
    assert(new_ctx->edge_list_pool);
    new_ctx->edge_list_allocator = create_mspace_with_base(
            new_ctx->edge_list_pool, new_ctx->edge_list_pool_size, 0);
    assert(new_ctx->edge_list_allocator);

    // Print the number of bytes allocated
#ifdef DETAILED_PRINTS
    shmem_malloc_wrapper(0);
    malloc_helper(0);
    print_memory_metrics(new_ctx);
#endif

    shmem_barrier_all();
}

static void *aborting_thread(void *user_data) {
    int nseconds = atoi(getenv("HVR_HANG_ABORT"));
    assert(nseconds > 0);

    fprintf(stderr, "INFO: HOOVER will forcibly abort PE %d after %d "
            "seconds.\n", shmem_my_pe(), nseconds);

    const unsigned long long start = hvr_current_time_us();
    while (hvr_current_time_us() - start < nseconds * 1000000) {
        sleep(10);
    }

    if (!this_pe_has_exited) {
        fprintf(stderr, "INFO: HOOVER forcibly aborting PE %d after %d "
                "seconds because HVR_HANG_ABORT was set.\n", shmem_my_pe(),
                nseconds);
        abort(); // Get a core dump
    }
    return NULL;
}

static void send_all_vertices_in_partition(int pe, hvr_partition_t part,
        hvr_internal_ctx_t *ctx) {
    if (pe == ctx->pe) {
        return;
    }

    hvr_vertex_t *iter = hvr_partition_list_head(part,
            &ctx->local_partition_lists);
    while (iter) {
        hvr_vertex_update_t msg;
        hvr_vertex_update_init(&msg, iter, 0);
        hvr_mailbox_buffer_send(&msg, sizeof(msg), pe, -1,
                &ctx->vertex_update_mailbox_buffer, 0);
        iter = iter->next_in_partition;
    }
}

static void process_neighbor_updates(hvr_internal_ctx_t *ctx,
        process_perf_info_t *perf_info) {
    size_t msg_len;
    hvr_msg_buf_node_t *msg_buf_node = hvr_msg_buf_pool_acquire(
            &ctx->msg_buf_pool);
    int success = hvr_mailbox_recv(msg_buf_node->ptr, msg_buf_node->buf_size,
            &msg_len, &ctx->forward_mailbox, 0);
    while (success) {
        assert(msg_len == sizeof(hvr_partition_member_change_t));
        /*
         * Tells that a given PE has subscribed/unsubscribed to updates for a
         * given partition for which we are a producer for.
         */
        hvr_partition_member_change_t *change =
            (hvr_partition_member_change_t *)msg_buf_node->ptr;
        assert(change->pe >= 0 && change->pe < ctx->npes);
        assert(change->partition < ctx->n_partitions);
        assert(change->entered == 0 || change->entered == 1);

        if (change->entered) {
            // Entered partition

            if (!hvr_sparse_arr_contains(change->partition, change->pe,
                        &ctx->remote_partition_subs)) {
                /*
                 * This is a new subscription from a remote PE for this
                 * partition. If we find new subscriptions, we need to transmit
                 * all vertex information we have for that partition to the PE.
                 */
                send_all_vertices_in_partition(change->pe, change->partition,
                        ctx);
                hvr_sparse_arr_insert(change->partition, change->pe,
                        &ctx->remote_partition_subs);
            }
        } else {
            /*
             * Left partition. It is possible that we receive a left partition
             * message from a PE that we did not know about yet if the following
             * happens in quick succession:
             *   1. This PE becomes a producer of the partition, updating the
             *      global directory.
             *   2. The remote PE stops subscribing to this partition, before
             *      looping around to update_partition_window to realize
             *      this PE's change in status and send it a subscription msg.
             *   3. The remote PE loops around to update_partition_window
             *      and sends out notifications to all producers that it is
             *      leaving the partition.
             */
            if (hvr_sparse_arr_contains(change->partition, change->pe,
                        &ctx->remote_partition_subs)) {
                hvr_sparse_arr_remove(change->partition, change->pe,
                        &ctx->remote_partition_subs);
            }
        }
        success = hvr_mailbox_recv(msg_buf_node->ptr, msg_buf_node->buf_size,
                &msg_len, &ctx->forward_mailbox, 0);
    }

    // Poll for subscriptions to individual vertices
    success = hvr_mailbox_recv(msg_buf_node->ptr, msg_buf_node->buf_size,
            &msg_len, &ctx->vert_sub_mailbox, 0);
    while (success) {
        assert(msg_len % sizeof(hvr_vertex_subscription_t) == 0);
        hvr_vertex_subscription_t *changes =
            (hvr_vertex_subscription_t *)msg_buf_node->ptr;
        for (int i = 0; i < msg_len / sizeof(hvr_vertex_subscription_t); i++) {
            hvr_vertex_subscription_t *change = changes + i;
            assert(change->pe >= 0 && change->pe < ctx->npes);
            assert(VERTEX_ID_PE(change->vert) == ctx->pe);
            assert(change->entered == 0 || change->entered == 1);

            hvr_vertex_id_t offset = VERTEX_ID_OFFSET(change->vert);
            hvr_vertex_cache_node_t *local = ctx->vec_cache.pool_mem + offset;

            if (change->entered) {
                if (!hvr_sparse_arr_contains(offset, change->pe,
                            &ctx->remote_vert_subs)) {
                    /*
                     * Send current state of vertex
                     * TODO there is a problem here if a vertex subscription is sent
                     * and not processed until after the vertex is deleted and its
                     * cache slot re-used for something else.
                     */

                    hvr_vertex_update_t msg;
                    hvr_vertex_update_init(&msg, &local->vert, 0);
                    hvr_mailbox_buffer_send(&msg, sizeof(msg), change->pe, -1,
                            &ctx->vertex_update_mailbox_buffer, 0);

                    hvr_sparse_arr_insert(offset, change->pe,
                            &ctx->remote_vert_subs);
                }
            } else { // change->entered == 0
                if (hvr_sparse_arr_contains(offset, change->pe,
                            &ctx->remote_vert_subs)) {
                    hvr_sparse_arr_remove(offset, change->pe,
                            &ctx->remote_vert_subs);
                }
            }
        }

        success = hvr_mailbox_recv(msg_buf_node->ptr, msg_buf_node->buf_size,
                &msg_len, &ctx->vert_sub_mailbox, 0);
    }

    // Poll for remote edge creations
    success = hvr_mailbox_recv(msg_buf_node->ptr, msg_buf_node->buf_size,
            &msg_len, &ctx->edge_create_mailbox, 0);
    while (success) {
        assert(msg_len % sizeof(hvr_edge_create_msg_t) == 0);
        hvr_edge_create_msg_t *msgs = (hvr_edge_create_msg_t *)msg_buf_node->ptr;
        for (int i = 0; i < msg_len / sizeof(*msgs); i++) {
            hvr_edge_create_msg_t *msg = msgs + i;
            assert(VERTEX_ID_PE(msg->target) == ctx->pe &&
                    VERTEX_ID_PE(msg->src.id) != ctx->pe);

            // Insert the remote source into our cache
            hvr_vertex_cache_node_t *cached = set_up_vertex_subscription(
                    msg->src.id, &msg->src, ctx);

            // Insert the explcitly created edge in our local edge info
            update_edge_info(msg->target, msg->src.id,
                    cached,
                    ctx->vec_cache.pool_mem + VERTEX_ID_OFFSET(msg->target),
                    msg->edge, NULL, EXPLICIT_EDGE, ctx);
        }

        success = hvr_mailbox_recv(msg_buf_node->ptr, msg_buf_node->buf_size,
                &msg_len, &ctx->edge_create_mailbox, 0);
    }

    hvr_msg_buf_pool_release(msg_buf_node, &ctx->msg_buf_pool);
}


/*
 * When a vertex is deleted, we simply need to remove all of its
 * edges with local vertices and remove it from the cache.
 */
static void handle_deleted_vertex(hvr_vertex_t *dead_vert,
        int expect_no_edges,
        hvr_internal_ctx_t *ctx) {
    hvr_vertex_cache_node_t *cached = hvr_vertex_cache_lookup(dead_vert->id,
            &ctx->vec_cache);

    // If we were caching this node, delete the mirrored version
    if (cached) {
        unsigned n_neighbors = hvr_irr_matrix_linearize(
                CACHE_NODE_OFFSET(cached, &ctx->vec_cache),
                ctx->edge_buffer, MAX_MODIFICATIONS, &ctx->edges);

        if (expect_no_edges) {
            for (size_t n = 0; n < n_neighbors; n++) {
                hvr_vertex_id_t id = EDGE_INFO_VERTEX(ctx->edge_buffer[n]);
                assert(VERTEX_ID_PE(id) != ctx->pe);
            }
        }
        assert(cached->n_explicit_edges == 0);

        // Delete all edges
        for (size_t n = 0; n < n_neighbors; n++) {
            hvr_vertex_cache_node_t *cached_neighbor = CACHE_NODE_BY_OFFSET(
                    EDGE_INFO_VERTEX(ctx->edge_buffer[n]), &ctx->vec_cache);
            hvr_edge_type_t edge = EDGE_INFO_EDGE(ctx->edge_buffer[n]);
            update_edge_info(dead_vert->id, cached_neighbor->vert.id, cached,
                    cached_neighbor, NO_EDGE, &edge, IMPLICIT_EDGE, ctx);
        }

        hvr_vertex_t *cached_vert = &cached->vert;
        hvr_partition_t partition = cached_vert->curr_part;
        remove_from_partition_list_helper(cached_vert, partition,
                &ctx->mirror_partition_lists, ctx);

        hvr_vertex_cache_delete(cached, &ctx->vec_cache);
    }
}

static void handle_new_vertex(hvr_vertex_t *new_vert,
        process_perf_info_t *perf_info,
        hvr_internal_ctx_t *ctx) {
    unsigned local_count_new_should_have_edges = 0;

    hvr_partition_t new_partition = new_vert->curr_part;
    hvr_vertex_id_t updated_vert_id = new_vert->id;

    unsigned n_interacting = 0;
    if (new_partition != HVR_INVALID_PARTITION) {
        ctx->might_interact(new_partition, ctx->interacting, &n_interacting,
                MAX_INTERACTING_PARTITIONS, ctx);
    }

    hvr_vertex_cache_node_t *updated = hvr_vertex_cache_lookup(
            updated_vert_id, &ctx->vec_cache);

    /*
     * Am I subscribed to the partition the vertex is now in, or have explicit
     * edges on this vertex (meaning I am subscribed specifically to it).
     */
    int am_subscribed = (updated && updated->n_explicit_edges > 0) ||
        (new_partition != HVR_INVALID_PARTITION &&
         hvr_set_contains(new_partition, ctx->subscribed_partitions));

    /*
     * If this is a vertex we already know about then we have
     * existing local edges that might need updating.
     */
    if (updated) {
        // Assert that I was subscribed to updates from this partition
        hvr_partition_t old_partition = updated->vert.curr_part;

        if (am_subscribed) {
            const unsigned long long start_update = hvr_current_time_us();
            /*
             * Update our local mirror with the information received in the
             * message without overwriting partition list info.
             */
            memcpy(&updated->vert, new_vert,
                    offsetof(hvr_vertex_t, next_in_partition));
            updated->populated = 1;

            if (new_partition != HVR_INVALID_PARTITION) {
                update_partition_list_membership(&updated->vert, old_partition,
                        new_partition, &ctx->mirror_partition_lists, ctx);
                local_count_new_should_have_edges += update_existing_edges(
                        updated, ctx->interacting, n_interacting, ctx);
            }

            /*
             * Mark all downstream neighbors even if edges didn't change due to
             * change in attributes for this vertex.
             */
            mark_all_downstream_neighbors_for_processing(updated, ctx);

            const unsigned long long done_updating_edges = hvr_current_time_us();
            const unsigned long long done = hvr_current_time_us();

            if (perf_info) {
                perf_info->time_updating += (done - start_update);
                perf_info->time_updating_edges += (done_updating_edges -
                        start_update);
                perf_info->time_creating_edges += (done - done_updating_edges);
            }
        } else {
            // Not subscribed to the partition this vertex has moved to
            handle_deleted_vertex(new_vert, 0, ctx);
        }
    } else {
        /*
         * If this vertex is in a partition we aren't subscribed to, don't
         * accept it because we won't be getting updates for it (and the remote
         * PE might not know to update us).
         */

        if (am_subscribed) {
            const unsigned long long start_new = hvr_current_time_us();

            // A brand new vertex, or at least this is our first update on it
            updated = hvr_vertex_cache_add(new_vert, &ctx->vec_cache);
            prepend_to_partition_list(&updated->vert, new_partition,
                    &ctx->mirror_partition_lists, ctx);
            local_count_new_should_have_edges += create_new_edges(updated,
                    ctx->interacting, n_interacting,
                    &ctx->mirror_partition_lists, ctx);
            local_count_new_should_have_edges += create_new_edges(updated,
                    ctx->interacting, n_interacting,
                    &ctx->local_partition_lists, ctx);

            /*
             * Don't need to mark downstream because creation of new edges for
             * this new vertex will do that for us.
             */

            if (perf_info) {
                perf_info->time_creating += (hvr_current_time_us() - start_new);
            }
        }
    }

    if (perf_info) {
        perf_info->count_new_should_have_edges +=
            local_count_new_should_have_edges;
    }
}

void hvr_send_msg(hvr_vertex_id_t dst_id, hvr_vertex_t *payload,
        hvr_internal_ctx_t *ctx) {
    if (VERTEX_ID_PE(dst_id) == ctx->pe) {
        size_t offset = VERTEX_ID_OFFSET(dst_id);
        hvr_buffered_msgs_insert(offset, payload, &ctx->buffered_msgs);

        hvr_vertex_cache_node_t *local = ctx->vec_cache.pool_mem + offset;
        assert(local->vert.id != HVR_INVALID_VERTEX_ID); // Verify is allocated
        assert(local->vert.id == dst_id);
        mark_for_processing(&local->vert, ctx);
    } else {
        inter_vert_msg_t msg;
        msg.dst = dst_id;
        memcpy(&msg.payload, payload, sizeof(*payload));

        hvr_mailbox_send(&msg, sizeof(msg), VERTEX_ID_PE(dst_id), -1,
                &ctx->vertex_msg_mailbox, 0);
    }
}

int hvr_poll_msg(hvr_vertex_t *vert, hvr_vertex_t *out,
        hvr_internal_ctx_t *ctx) {
    assert(VERTEX_ID_PE(vert->id) == ctx->pe);
    return hvr_buffered_msgs_poll(VERTEX_ID_OFFSET(vert->id), out,
            &ctx->buffered_msgs);
}

static void process_incoming_messages(hvr_internal_ctx_t *ctx) {
    size_t msg_len;
    hvr_msg_buf_node_t *msg_buf_node = hvr_msg_buf_pool_acquire(
            &ctx->msg_buf_pool);
    int success = hvr_mailbox_recv(msg_buf_node->ptr, msg_buf_node->buf_size,
            &msg_len, &ctx->vertex_msg_mailbox, 0);
    while (success) {
        assert(msg_len == sizeof(inter_vert_msg_t));
        inter_vert_msg_t *msg = (inter_vert_msg_t *)msg_buf_node->ptr;
        assert(VERTEX_ID_PE(msg->dst) == ctx->pe);

        size_t offset = VERTEX_ID_OFFSET(msg->dst);
        hvr_buffered_msgs_insert(offset, &msg->payload, &ctx->buffered_msgs);

        hvr_vertex_cache_node_t *local = ctx->vec_cache.pool_mem + offset;
        assert(local->vert.id != HVR_INVALID_VERTEX_ID); // Verify is allocated
        mark_for_processing(&local->vert, ctx);

        success = hvr_mailbox_recv(msg_buf_node->ptr, msg_buf_node->buf_size,
                &msg_len, &ctx->vertex_msg_mailbox, 0);
    }

    hvr_msg_buf_pool_release(msg_buf_node, &ctx->msg_buf_pool);
}

static unsigned process_vertex_updates(hvr_internal_ctx_t *ctx,
        process_perf_info_t *perf_info) {
    unsigned n_updates = 0;
    size_t msg_len;

    const unsigned long long start = hvr_current_time_us();
    unsigned count_delete_msgs = 0;
    // Handle deletes, then updates
    hvr_msg_buf_node_t *msg_buf_node = hvr_msg_buf_pool_acquire(
            &ctx->msg_buf_pool);
    int success = hvr_mailbox_recv(msg_buf_node->ptr, msg_buf_node->buf_size,
            &msg_len, &ctx->vertex_delete_mailbox, 0);
    while (success) {
        assert(msg_len % sizeof(hvr_vertex_update_t) == 0);
        hvr_vertex_update_t *msgs = (hvr_vertex_update_t *)msg_buf_node->ptr;

        for (unsigned i = 0; i < msg_len / sizeof(*msgs); i++) {
            hvr_vertex_update_t *msg = msgs + i;
            handle_deleted_vertex(&(msg->vert), 0, ctx);
            n_updates++;
        }
        ctx->total_vertex_msgs_recvd += (msg_len / sizeof(*msgs));

        count_delete_msgs++;
        if (count_delete_msgs >= MAX_MSGS_PROCESSED) break;

        success = hvr_mailbox_recv(msg_buf_node->ptr, msg_buf_node->buf_size,
                &msg_len, &ctx->vertex_delete_mailbox, 0);
    }

    const unsigned long long midpoint = hvr_current_time_us();
    unsigned count_update_msgs = 0;
    success = hvr_mailbox_recv(msg_buf_node->ptr, msg_buf_node->buf_size,
            &msg_len, &ctx->vertex_update_mailbox, 0);
    while (success) {
        assert(msg_len % sizeof(hvr_vertex_update_t) == 0);
        hvr_vertex_update_t *msgs = (hvr_vertex_update_t *)msg_buf_node->ptr;

        for (unsigned i = 0; i < msg_len / sizeof(*msgs); i++) {
            hvr_vertex_update_t *msg = msgs + i;
            assert(VERTEX_ID_PE(msg->vert.id) != ctx->pe);

            if (msg->is_invalidation) {
                handle_deleted_vertex(&(msg->vert), 0, ctx);
            } else {
                handle_new_vertex(&(msg->vert), perf_info, ctx);
            }
            n_updates++;
        }
        ctx->total_vertex_msgs_recvd += (msg_len / sizeof(*msgs));

        count_update_msgs++;
        if (count_update_msgs >= MAX_MSGS_PROCESSED) break;

        success = hvr_mailbox_recv(msg_buf_node->ptr, msg_buf_node->buf_size,
                &msg_len, &ctx->vertex_update_mailbox, 0);
    }

    hvr_msg_buf_pool_release(msg_buf_node, &ctx->msg_buf_pool);

    const unsigned long long done = hvr_current_time_us();

    if (perf_info) {
        perf_info->time_handling_deletes += midpoint - start;
        perf_info->time_handling_news += done - midpoint;
    }

    return n_updates;
}

static inline void update_partition_distance(hvr_vertex_cache_node_t *node,
        uint8_t dist, uint8_t *partition_min_dist_from_local_vert) {
    const hvr_partition_t curr_part = node->vert.curr_part;
    if (curr_part != HVR_INVALID_PARTITION &&
            partition_min_dist_from_local_vert[curr_part] > dist) {
        partition_min_dist_from_local_vert[curr_part] = dist;
    }
}

static void update_distances(hvr_internal_ctx_t *ctx) {
    /*
     * Clear all distances of mirrored vertices to an invalid value before
     * recomputing
     */
    const unsigned max_graph_traverse_depth = ctx->max_graph_traverse_depth;
    hvr_vertex_cache_node_t *q = ctx->vec_cache.local_neighbors_head;

    uint8_t *partition_min_dist_from_local_vert =
        ctx->partition_min_dist_from_local_vert;
    memset(partition_min_dist_from_local_vert, 0xff,
            sizeof(*partition_min_dist_from_local_vert) * ctx->n_partitions);

    for (hvr_partition_t p = 0; p < ctx->n_partitions; p++) {
        hvr_vertex_t *iter = hvr_partition_list_head(p,
                &ctx->local_partition_lists);
        if (iter) {
            partition_min_dist_from_local_vert[p] = 0;
        } else {
            uint8_t min_dist = UINT8_MAX;
            iter = hvr_partition_list_head(p, &ctx->mirror_partition_lists);
            while (iter) {
                const uint8_t distance =
                    ((hvr_vertex_cache_node_t *)iter)->dist_from_local_vert;
                if (distance < min_dist) {
                    min_dist = distance;
                }
                iter = iter->next_in_partition;
            }
            partition_min_dist_from_local_vert[p] = min_dist;
        }
    }
}

void hvr_get_neighbors(hvr_vertex_t *vert, hvr_neighbors_t *neighbors,
        hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    hvr_vertex_cache_node_t *cached = hvr_vertex_cache_lookup(vert->id,
            &ctx->vec_cache);
    if (!cached) {
        /*
         * Might happen due to throttling of producer checking, we may have a
         * local vertex that we don't know we are a producer for yet.
         */
        hvr_neighbors_init(NULL, 0, &ctx->vec_cache, neighbors);
        return;
    }

    // Lookup edge information in ctx->edges
    hvr_edge_info_t *edge_buffer;
    unsigned n_neighbors = hvr_irr_matrix_linearize_zero_copy(
            CACHE_NODE_OFFSET(cached, &ctx->vec_cache),
            &edge_buffer, &ctx->edges);
    hvr_neighbors_init(edge_buffer, n_neighbors, &ctx->vec_cache, neighbors);

#if 0
    // Allocate buffer space to store the edges in before handing back to user
    void *tmp_buf = mspace_malloc(ctx->edge_list_allocator,
            n_neighbors * (sizeof(hvr_vertex_t *) + sizeof(hvr_edge_type_t)));
    if (!tmp_buf) {
        fprintf(stderr, "Ran out of edge pool space, consider increasing "
                "HVR_EDGE_LIST_POOL_SIZE (%lu)\n", ctx->edge_list_pool_size);
        exit(1);
    }
    hvr_vertex_t **tmp_vert_ptr_buffer = (hvr_vertex_t **)tmp_buf;
    hvr_edge_type_t *tmp_dir_buffer = (hvr_edge_type_t *)(tmp_vert_ptr_buffer +
            n_neighbors);

    unsigned n_populated_neighbors = 0;
    for (unsigned n = 0; n < n_neighbors; n++) {
        hvr_vertex_cache_node_t *cached_neighbor = CACHE_NODE_BY_OFFSET(
                EDGE_INFO_VERTEX(edge_buffer[n]), &ctx->vec_cache);
        if (cached_neighbor->populated) {
            tmp_vert_ptr_buffer[n_populated_neighbors] = &cached_neighbor->vert;
            tmp_dir_buffer[n_populated_neighbors++] =
                EDGE_INFO_EDGE(edge_buffer[n]);
        }
    }
        
    *out_verts = tmp_vert_ptr_buffer;
    *out_dirs = tmp_dir_buffer;
    return n_populated_neighbors;
#endif
}

void hvr_release_neighbors(hvr_vertex_t **out_verts, hvr_edge_type_t *out_dirs,
        int n_neighbors, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    mspace_free(ctx->edge_list_allocator, out_verts);
}

static hvr_vertex_cache_node_t *set_up_vertex_subscription(hvr_vertex_id_t vid,
        hvr_vertex_t *optional_body, hvr_internal_ctx_t *ctx) {
    /*
     * Reserve a cache slot for this remote vertex, pre-populate it with the
     * state passed to hvr_create_edge, notify that PE that we'll need
     * updates from it on this vertex, and then return.
     */
    int owning_pe = VERTEX_ID_PE(vid);

    hvr_vertex_cache_node_t *cached = hvr_vertex_cache_lookup(vid,
            &ctx->vec_cache);
    if (!cached) {
        hvr_vertex_t dummy;
        hvr_vertex_init(&dummy, vid, ctx->iter);
        cached = hvr_vertex_cache_add(&dummy, &ctx->vec_cache);
        cached->populated = 0;
    }

    if (optional_body) {
        /*
         * We already have a copy of valid state for this vertex that we can
         * pre-populate with.
         */
        memcpy(&cached->vert, optional_body, sizeof(*optional_body));
        cached->populated = 1;
    }

    if (owning_pe != ctx->pe) {
        if (hvr_set_contains(owning_pe, ctx->all_terminated_pes)) {
            // Already terminated, just pull this vertex's latest value.
            shmem_getmem(&cached->vert,
                    ctx->vec_cache.pool_mem + VERTEX_ID_OFFSET(vid),
                    sizeof(cached->vert), owning_pe);
            cached->populated = 1;
        } else {
            // Notify owning PE that we are subscribed to this vertex
            hvr_vertex_subscription_t msg;
            msg.pe = ctx->pe;
            msg.vert = vid;
            msg.entered = 1;
            hvr_mailbox_buffer_send(&msg, sizeof(msg), owning_pe, -1,
                    &ctx->vert_sub_mailbox_buffer, 0);
        }
    }

    hvr_sparse_arr_insert(owning_pe, VERTEX_ID_OFFSET(vid), &ctx->my_vert_subs);

    return cached;
}

// edge is relative to remote
static void signal_edge_creation(hvr_vertex_id_t remote, hvr_vertex_t *base,
        hvr_edge_type_t edge, hvr_internal_ctx_t *ctx) {
    assert(VERTEX_ID_PE(remote) != ctx->pe);

    hvr_edge_create_msg_t msg;
    memcpy(&msg.src, base, sizeof(*base));
    msg.target = remote;
    msg.edge = edge;

    hvr_mailbox_buffer_send(&msg, sizeof(msg), VERTEX_ID_PE(remote), -1,
            &ctx->edge_create_mailbox_buffer, 0);
}

/*
 * TODO today we don't support the inverse (hvr_delete_edge). When we do, we'll
 * need to figure out what we want to do with explicitly created vertex
 * subscriptions when their edges drop to zero (and the explicitly cached
 * vertex).
 */
static void hvr_create_edge_helper(hvr_vertex_t *local,
        hvr_vertex_id_t neighbor, hvr_vertex_t *optional_neighbor_body,
        hvr_edge_type_t edge, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    hvr_vertex_id_t local_id = local->id;

    // Assert that at least one of these vertices is locally owned
    assert(VERTEX_ID_PE(local_id) == ctx->pe);

    /*
     * Assert that both vertices are not in a valid partition, so that they do
     * not have any implicit edges created.
     */
    assert(local->curr_part == HVR_INVALID_PARTITION);

    hvr_vertex_cache_node_t *cached_local = hvr_vertex_cache_lookup(local_id,
            &ctx->vec_cache);
    hvr_vertex_cache_node_t *cached_neighbor = set_up_vertex_subscription(
            neighbor, optional_neighbor_body, ctx);
    assert(cached_local && cached_neighbor);

    // Create explicit edge
    update_edge_info(local_id, neighbor, cached_local, cached_neighbor, edge,
            NULL, EXPLICIT_EDGE, ctx);

    if (VERTEX_ID_PE(neighbor) != ctx->pe) {
        signal_edge_creation(neighbor, local, flip_edge_direction(edge), ctx);
    }
}

void hvr_create_edge_with_vertex_id(hvr_vertex_t *local,
        hvr_vertex_id_t neighbor, hvr_edge_type_t edge, hvr_ctx_t in_ctx) {
    hvr_create_edge_helper(local, neighbor, NULL, edge, in_ctx);
}

void hvr_create_edge_with_vertex(hvr_vertex_t *local,
        hvr_vertex_t *neighbor, hvr_edge_type_t edge, hvr_ctx_t in_ctx) {
    hvr_create_edge_helper(local, neighbor->id, neighbor, edge, in_ctx);
}

hvr_vertex_t *hvr_get_vertex(hvr_vertex_id_t id, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    hvr_vertex_cache_node_t *cached = hvr_vertex_cache_lookup(id,
            &ctx->vec_cache);
    if (cached) {
        return &cached->vert;
    } else {
        return NULL;
    }
}

static int update_vertices(hvr_set_t *to_couple_with,
        hvr_internal_ctx_t *ctx) {
    if (ctx->update_metadata == NULL || !ctx->any_needs_processing) {
        return 0;
    }

    ctx->any_needs_processing = 0;

    int count = 0;
    hvr_vertex_iter_t iter;
    hvr_vertex_iter_init(&iter, ctx);
    for (hvr_vertex_t *curr = hvr_vertex_iter_next(&iter); curr;
            curr = hvr_vertex_iter_next(&iter)) {

        if (curr->needs_processing) {
            curr->needs_processing = 0;
            const hvr_partition_t old_part = ctx->actor_to_partition(curr, ctx);

            ctx->update_metadata(curr, to_couple_with, ctx);

            hvr_partition_t new_partition = ctx->actor_to_partition(curr, ctx);
            curr->curr_part = new_partition;
            curr->prev_part = old_part;

            if (new_partition == HVR_INVALID_PARTITION) {
                assert(old_part == HVR_INVALID_PARTITION);
            } else {
                update_partition_list_membership(curr, old_part, new_partition,
                        &ctx->local_partition_lists, ctx);
                if (curr->needs_send) {
                    // Something changed
                    unsigned n_interacting = 0;
                    ctx->might_interact(new_partition, ctx->interacting,
                            &n_interacting, MAX_INTERACTING_PARTITIONS, ctx);
                    update_existing_edges((hvr_vertex_cache_node_t *)curr,
                            ctx->interacting, n_interacting, ctx);
                }
            }

            if (curr->needs_send) {
                /*
                 * Mark all downstream neighbors because attributes of this
                 * local vertex changed.
                 */
                mark_all_downstream_neighbors_for_processing(
                        (hvr_vertex_cache_node_t *)curr, ctx);
            }

            count++;
        }
    }

    return count;
}

static void send_updates_to_all_subscribed_pes_helper(hvr_vertex_t *vert,
        int *subscribers, unsigned n_subscribers, int is_delete,
        int is_invalidation, hvr_mailbox_buffer_t *mbox_buffer,
        hvr_internal_ctx_t *ctx,
        process_perf_info_t *perf_info, unsigned long long *start,
        unsigned long long *time_sending) {

    for (unsigned s = 0; s < n_subscribers; s++) {
        int sub_pe = subscribers[s];
        assert(sub_pe < ctx->npes);
        if (sub_pe == ctx->pe) {
            continue;
        }

        hvr_vertex_update_t msg;
        hvr_vertex_update_init(&msg, vert, is_invalidation);

        hvr_mailbox_buffer_send(&msg, sizeof(msg), sub_pe, -1,
                mbox_buffer, 0);
    }
}

void send_updates_to_all_subscribed_pes(
        hvr_vertex_t *vert,
        hvr_partition_t part,
        int is_invalidation,
        int is_delete,
        process_perf_info_t *perf_info,
        unsigned long long *time_sending,
        hvr_internal_ctx_t *ctx) {
    assert(part != HVR_INVALID_PARTITION);
    assert(vert->curr_part != HVR_INVALID_PARTITION);
    assert(VERTEX_ID_PE(vert->id) == ctx->pe);
    unsigned long long start = hvr_current_time_us();
    hvr_mailbox_buffer_t *mbox = (is_delete ?
            &ctx->vertex_delete_mailbox_buffer :
            &ctx->vertex_update_mailbox_buffer);

    // Find subscribers to part and send message to them
    int *subscribers = NULL;
    unsigned n_subscribers = hvr_sparse_arr_linearize_row(part,
            &subscribers, &ctx->remote_partition_subs);
    send_updates_to_all_subscribed_pes_helper(vert, subscribers, n_subscribers,
            is_delete, is_invalidation, mbox, ctx, perf_info, &start,
            time_sending);

    // Find subscribers to this particular vertex and send update to them
    n_subscribers = hvr_sparse_arr_linearize_row(VERTEX_ID_OFFSET(vert->id),
            &subscribers, &ctx->remote_vert_subs);
    send_updates_to_all_subscribed_pes_helper(vert, subscribers, n_subscribers,
            is_delete, is_invalidation, mbox, ctx, perf_info, &start,
            time_sending);

    *time_sending += (hvr_current_time_us() - start);
}

static unsigned send_vertex_updates(hvr_internal_ctx_t *ctx,
        unsigned long long *time_sending,
        process_perf_info_t *perf_info) {
    unsigned n_updates_sent = 0;

    hvr_vertex_iter_t iter;
    hvr_vertex_iter_all_init(&iter, ctx);
    for (hvr_vertex_t *curr = hvr_vertex_iter_next(&iter); curr;
            curr = hvr_vertex_iter_next(&iter)) {
        // If this vertex was mutated on this iteration
        if (curr->needs_send) {
            /*
             * If this vertex changed partitions, need to invalidate any cached
             * copies in the old partition.
             */
            hvr_partition_t new_partition = curr->curr_part;
            hvr_partition_t old_partition = curr->prev_part;

            if (new_partition == HVR_INVALID_PARTITION) {
                assert(old_partition == HVR_INVALID_PARTITION);
            } else {
                if (old_partition != HVR_INVALID_PARTITION &&
                        old_partition != new_partition) {
                    send_updates_to_all_subscribed_pes(curr, old_partition, 1,
                            0, perf_info, time_sending, ctx);
                }

                send_updates_to_all_subscribed_pes(curr, new_partition, 0, 0,
                        perf_info, time_sending, ctx);
                n_updates_sent++;
            }
            curr->needs_send = 0;
        }
    }

    hvr_mailbox_buffer_flush(&ctx->vertex_update_mailbox_buffer);
    hvr_mailbox_buffer_flush(&ctx->vertex_delete_mailbox_buffer);

    return n_updates_sent;
}

static void receive_coupled_val(hvr_coupling_msg_t *msg,
        hvr_internal_ctx_t *ctx) {
    assert(!hvr_set_contains(msg->pe, ctx->prev_all_terminated_cluster_pes));

    ctx->updates_on_this_iter[msg->pe] = msg->updates_on_this_iter;
    memcpy(ctx->coupled_pes_values + msg->pe, &msg->val, sizeof(msg->val));
}

static void wait_for_full_coupling_info(hvr_msg_buf_node_t *msg_buf_node,
        int expect_no_new_couplings_this_iter, hvr_internal_ctx_t *ctx) {
    while (1) {
        size_t msg_len;
        int success = hvr_mailbox_recv(msg_buf_node->ptr,
                msg_buf_node->buf_size, &msg_len,
                &ctx->coupling_mailbox, 0);
        if (success) {
            if (msg_len == sizeof(new_coupling_msg_t)) {
                new_coupling_msg_t *msg =
                    (new_coupling_msg_t *)msg_buf_node->ptr;
                if (expect_no_new_couplings_this_iter) {
                    /*
                     * If we are root and we're getting a new coupling request
                     * after completing the new coupling process, we don't
                     * process it until the next iteration so that we can
                     * proceed with exchanging coupled values with the current
                     * cluster. So resend to myself.
                     */
                    hvr_mailbox_send(msg, sizeof(*msg), ctx->pe, -1,
                            &ctx->coupling_mailbox, 0);
                } else {
                    // Reply, forwarding them to your root
                    new_coupling_msg_ack_t ack;
                    ack.pe = ctx->pe;
                    ack.root_pe = ctx->coupled_pes_root;
                    ack.abort = 0;
                    hvr_mailbox_send(&ack, sizeof(ack), msg->pe, -1,
                            &ctx->coupling_ack_and_dead_mailbox, 0);
                }
            } else if (msg_len == ctx->to_couple_with_msg.msg_buf_len) {
                hvr_internal_set_msg_t *msg =
                    (hvr_internal_set_msg_t *)msg_buf_node->ptr;
                if (msg->metadata == 42) {
                    // all terminated pes
                    hvr_set_copy(ctx->prev_all_terminated_cluster_pes,
                            ctx->all_terminated_cluster_pes);
                    hvr_set_msg_copy(ctx->all_terminated_cluster_pes, msg);
                    ctx->coupled_pes_root = msg->pe;
                } else if (msg->metadata == 43) {
                    // coupled pes
                    hvr_set_msg_copy(ctx->coupled_pes, msg);
                    ctx->coupled_pes_root = msg->pe;
                    break;
                } else {
                    abort();
                }
            } else {
                abort();
            }
        }
    }
}

static size_t blocking_recv(hvr_msg_buf_node_t *msg_buf_node, hvr_mailbox_t *mb,
        hvr_internal_ctx_t *ctx) {
    size_t msg_len;
    int success;
    do {
        success = hvr_mailbox_recv(msg_buf_node->ptr, msg_buf_node->buf_size,
                &msg_len, mb, 0);
    } while (!success);
    return msg_len;
}

static int negotiate_new_root(int other_root, hvr_msg_buf_node_t *msg_buf_node,
        hvr_internal_ctx_t *ctx) {
    size_t msg_len;
    // Found the true root PE of the other cluster
    int new_combined_root_pe = (ctx->pe < other_root ? ctx->pe : other_root);
    if (new_combined_root_pe == ctx->pe) {
        /*
         * I remain root, I'll receive info from the other root
         * including to_couple_with and coupled_pes.
         */
        msg_len = blocking_recv(msg_buf_node, &ctx->root_info_mailbox, ctx);
        assert(msg_len == ctx->to_couple_with_msg.msg_buf_len);
        hvr_internal_set_msg_t *msg =
            (hvr_internal_set_msg_t *)msg_buf_node->ptr;
        assert(msg->pe == other_root);
        assert(msg->metadata == 1);
        hvr_set_msg_copy(ctx->other_coupled_pes, msg);

        msg_len = blocking_recv(msg_buf_node, &ctx->root_info_mailbox, ctx);
        assert(msg_len == ctx->to_couple_with_msg.msg_buf_len);
        msg = (hvr_internal_set_msg_t *)msg_buf_node->ptr;
        assert(msg->pe == other_root);
        assert(msg->metadata == 2);
        hvr_set_msg_copy(ctx->other_to_couple_with, msg);

        msg_len = blocking_recv(msg_buf_node, &ctx->root_info_mailbox, ctx);
        assert(msg_len == ctx->to_couple_with_msg.msg_buf_len);
        msg = (hvr_internal_set_msg_t *)msg_buf_node->ptr;
        assert(msg->pe == other_root);
        assert(msg->metadata == 3);
        hvr_set_msg_copy(ctx->other_terminating_pes, msg);

        hvr_set_or(ctx->coupled_pes, ctx->other_coupled_pes);
        hvr_set_or(ctx->to_couple_with, ctx->other_to_couple_with);
        hvr_set_or(ctx->terminating_pes, ctx->other_terminating_pes);
    } else {
        /*
         * Other root is now the root, I need to send my info
         * and then become a normal PE.
         */
        hvr_set_msg_send(other_root, ctx->pe, ctx->iter, 1,
                &ctx->root_info_mailbox,
                &ctx->coupled_pes_msg);
        hvr_set_msg_send(other_root, ctx->pe, ctx->iter, 2,
                &ctx->root_info_mailbox,
                &ctx->to_couple_with_msg);
        hvr_set_msg_send(other_root, ctx->pe, ctx->iter, 3,
                &ctx->root_info_mailbox,
                &ctx->terminating_pes_msg);
    }

    return new_combined_root_pe;
}

static int check_incoming_coupling_requests(hvr_msg_buf_node_t *msg_buf_node,
        hvr_internal_ctx_t *ctx) {
    // Should only be called by root PEs
    assert(ctx->pe == ctx->coupled_pes_root);

    size_t msg_len;
    int success = hvr_mailbox_recv(msg_buf_node->ptr, msg_buf_node->buf_size,
            &msg_len, &ctx->coupling_mailbox, 0);
    if (success) {
        assert(msg_len == sizeof(new_coupling_msg_t));
        new_coupling_msg_t *msg =
            (new_coupling_msg_t *)msg_buf_node->ptr;
        int request_src_pe = msg->pe;

        // Acknowledge the new coupling request
        new_coupling_msg_ack_t ack;
        ack.pe = ctx->pe;
        ack.root_pe = ctx->coupled_pes_root;
        ack.abort = 0;
        hvr_mailbox_send(&ack, sizeof(ack), request_src_pe, -1,
                &ctx->coupling_ack_and_dead_mailbox, 0);

#ifdef COUPLING_PRINTS
        fprintf(stderr, "PE %d got coupling request from %d\n", ctx->pe,
                request_src_pe);
#endif

        // Must be another root, negotiate who will be the new root
        int new_root = negotiate_new_root(request_src_pe, msg_buf_node, ctx);
        if (new_root != ctx->pe) {
            ctx->coupled_pes_root = new_root;
        }
#ifdef COUPLING_PRINTS
        fprintf(stderr, "PE %d completing coupling request with %d, new root = "
                "%d\n", ctx->pe, request_src_pe, ctx->coupled_pes_root);
#endif
    }
    return success;
}

static int wait_for_coupling_ack(hvr_msg_buf_node_t *msg_buf_node,
        int pe_waiting_on, hvr_internal_ctx_t *ctx) {
    assert(sizeof(hvr_dead_pe_msg_t) != sizeof(new_coupling_msg_ack_t));

    int success;
    size_t msg_len;

    int got_ack = 0;
    do {
        success = hvr_mailbox_recv(msg_buf_node->ptr,
                msg_buf_node->buf_size, &msg_len,
                &ctx->coupling_mailbox, 0);
        if (success) {
            assert(msg_len == sizeof(new_coupling_msg_t));
            new_coupling_msg_t *remote_coupling_msg =
                (new_coupling_msg_t *)msg_buf_node->ptr;
            /*
             * If we get a message that someone is trying to couple with us
             * while we are trying to couple with someone else, we tell them to
             * abort to prevent deadlock (i.e. a loop of PEs each waiting on
             * each other to acknowledge each other's coupling requests).
             * There's still a chance of livelock with this scheme.
             */

            new_coupling_msg_ack_t ack;
            ack.pe = ctx->pe;
            ack.root_pe = ctx->coupled_pes_root;
            ack.abort = 1;
            hvr_mailbox_send(&ack, sizeof(ack), remote_coupling_msg->pe,
                    -1, &ctx->coupling_ack_and_dead_mailbox, 0);

        }

        success = hvr_mailbox_recv(msg_buf_node->ptr,
                msg_buf_node->buf_size, &msg_len,
                &ctx->coupling_ack_and_dead_mailbox, 0);
        if (success) {
            if (msg_len == sizeof(hvr_dead_pe_msg_t)) {
                // A PE has left the simulation, update our set of terminated PEs
                hvr_dead_pe_msg_t *msg = (hvr_dead_pe_msg_t *)msg_buf_node->ptr;
                assert(!hvr_set_contains(msg->pe, ctx->all_terminated_pes));
                hvr_set_insert(msg->pe, ctx->all_terminated_pes);

                /*
                 * Check if we were subscribed to any vertices on that PE, and
                 * if so go grab their final state.
                 */
                int *subscriptions = NULL;
                unsigned n_subscriptions = hvr_sparse_arr_linearize_row(msg->pe,
                        &subscriptions, &ctx->my_vert_subs);
                for (unsigned i = 0; i < n_subscriptions; i++) {
                    hvr_vertex_t local;
                    shmem_getmem(&local,
                            ctx->vec_cache.pool_mem + subscriptions[i],
                            sizeof(local), msg->pe);
                    hvr_vertex_id_t id = construct_vertex_id(msg->pe,
                            subscriptions[i]);
                    hvr_vertex_cache_node_t *cached = hvr_vertex_cache_lookup(
                            id, &ctx->vec_cache);
                    assert(cached);
                    memcpy(&cached->vert, &local, sizeof(local));
                    cached->populated = 1;
                }
            } else if (msg_len == sizeof(new_coupling_msg_ack_t)) {
                got_ack = 1;
            } else {
                abort();
            }
        }
    } while (!got_ack &&
            !hvr_set_contains(pe_waiting_on, ctx->all_terminated_pes));

    if (got_ack) {
        assert(!hvr_set_contains(pe_waiting_on, ctx->all_terminated_pes));
        assert(msg_len == sizeof(new_coupling_msg_ack_t));
        return 0;
    } else {
        assert(hvr_set_contains(pe_waiting_on, ctx->all_terminated_pes));
        return 1;
    }
}

static unsigned update_coupled_values(hvr_internal_ctx_t *ctx,
        hvr_vertex_t *coupled_metric, int count_updated,
        unsigned long long *time_coupling,
        unsigned long long *time_waiting,
        unsigned long long *time_after,
        unsigned long long *time_should_terminate,
        unsigned *out_naborts,
        unsigned long long *time_waiting_for_prev,
        unsigned long long *time_processing_new_requests,
        unsigned long long *time_sharing_info,
        unsigned long long *time_waiting_for_info,
        unsigned long long *time_negotiating,
        int terminating) {
    unsigned naborts = 0;
    *time_waiting_for_prev = *time_processing_new_requests =
        *time_sharing_info = *time_waiting_for_info = *time_negotiating = 0;
    unsigned long long start_coupling = hvr_current_time_us();

    hvr_msg_buf_node_t *msg_buf_node = hvr_msg_buf_pool_acquire(
            &ctx->msg_buf_pool);

    // Copy the present value of the coupled metric locally
    memcpy(coupled_metric, ctx->coupled_pes_values + ctx->pe,
            sizeof(*coupled_metric));

    hvr_vertex_iter_t iter;
    hvr_vertex_iter_all_init(&iter, ctx);
    ctx->update_coupled_val(&iter, ctx, coupled_metric);

    hvr_set_wipe(ctx->to_couple_with);
    for (int p = 0; p < ctx->npes; p++) {
        if (hvr_set_contains(p, ctx->coupled_pes) &&
                !hvr_set_contains(p, ctx->prev_coupled_pes)) {
            // fprintf(stderr, "PE %d wants to couple with %d\n", ctx->pe, p);
            hvr_set_insert(p, ctx->to_couple_with);
        }
    }

    hvr_set_msg_send(ctx->coupled_pes_root, ctx->pe, ctx->iter, terminating,
            &ctx->to_couple_with_mailbox, &ctx->to_couple_with_msg);

    if (ctx->pe == ctx->coupled_pes_root) {
        unsigned long long start_waiting_for_prev = hvr_current_time_us();
        // fprintf(stderr, "PE %d starting iter %u as root, terminating=%d\n", ctx->pe, ctx->iter, terminating);

        /*
         * Wait for all currently coupled PEs to tell me who they want to become
         * coupled with.
         */
        // unsigned long long start_prev_waiting = hvr_current_time_us();
        hvr_set_wipe(ctx->terminating_pes);
        hvr_set_wipe(ctx->to_couple_with);
        hvr_set_wipe(ctx->received_from);
        hvr_set_or(ctx->received_from, ctx->all_terminated_cluster_pes);
        while (!hvr_set_equal(ctx->received_from, ctx->prev_coupled_pes)) {
            size_t msg_len;
            int success = hvr_mailbox_recv(msg_buf_node->ptr,
                    msg_buf_node->buf_size, &msg_len,
                    &ctx->to_couple_with_mailbox, 0);
            if (success) {
                // unsigned long long elapsed = hvr_current_time_us() - start_prev_waiting;
                assert(msg_len == ctx->to_couple_with_msg.msg_buf_len);
                hvr_internal_set_msg_t *msg =
                    (hvr_internal_set_msg_t *)msg_buf_node->ptr;
                // fprintf(stderr, "PE %d received to_couple_with from %d\n",
                //         ctx->pe, msg->pe);
                hvr_set_msg_copy(ctx->other_to_couple_with, msg);
                hvr_set_or(ctx->to_couple_with, ctx->other_to_couple_with);
                hvr_set_insert(msg->pe, ctx->received_from);
                // fprintf(stderr, "PE %d on iter %d spent %f ms waiting for %d\n",
                //         ctx->pe, ctx->iter, (double)elapsed / 1000.0, msg->pe);


                if (msg->metadata /* terminating */ ) {
                    hvr_set_insert(msg->pe, ctx->terminating_pes);
                }
            }
        }

        // fprintf(stderr, "PE %d received all to_couple_with\n", ctx->pe);

        /*
         * prev_coupled_pes is the set of PEs that we know we are already
         * coupled with from the previous iteration.
         *
         * to_couple_with is the set of PEs that this root PE is now responsible
         * for becoming coupled with for its cluster.
         *
         * coupled_pes is relatively useless, as it is prev_coupled_pes and
         * whatever couplings this PE requested. So, here we clear it back to
         * prev_coupled_pes and use it to track who we are actually coupled
         * with.
         *
         * We need to iterate until the intersection of to_couple_with and
         * coupled_pes is equal to to_couple_with.
         *
         * already_coupled_with is used to detect this convergence.
         */
        hvr_set_copy(ctx->coupled_pes, ctx->prev_coupled_pes);

        // fprintf(stderr, "PE %d already coupled with equals to couple with? "
        //         "%d # already coupled with %lu # to coupled with %lu # coupled "
        //         "%lu\n",
        //         ctx->pe,
        //         hvr_set_equal(ctx->already_coupled_with, ctx->to_couple_with),
        //         ctx->already_coupled_with->n_contained,
        //         ctx->to_couple_with->n_contained, ctx->coupled_pes->n_contained);
        unsigned long long start_processing = hvr_current_time_us();

        int target_pe = 0;
        do {
            // Find a PE that we want to couple with but haven't yet.
            int found_pe = 0;
            for (int p = 0; p < ctx->npes; p++) {
                int new_target_pe = (target_pe + p) % ctx->npes;
                if (hvr_set_contains(new_target_pe, ctx->to_couple_with) &&
                        !hvr_set_contains(new_target_pe, ctx->coupled_pes)) {
                    target_pe = new_target_pe;
                    found_pe = 1;
                    break;
                }
            }

            // fprintf(stderr, "PE %d coupling with %d %d\n", ctx->pe, found_pe, target_pe);

            if (found_pe) {
                // fprintf(stderr, "PE %d trying to couple with %d\n", ctx->pe,
                //         target_pe);

                // Tell this PE we'd like to couple with it
                new_coupling_msg_t my_coupling_msg;
                my_coupling_msg.pe = ctx->pe;
                my_coupling_msg.iter = ctx->iter;

                new_coupling_msg_ack_t *ack = NULL;
                int expected_root_pe = target_pe;
                int is_dead = 0;
                do {
                    hvr_mailbox_send(&my_coupling_msg, sizeof(my_coupling_msg),
                            expected_root_pe, -1, &ctx->coupling_mailbox, 0);
#ifdef COUPLING_PRINTS
                    fprintf(stderr, "PE %d sending coupling request to %d\n",
                            ctx->pe, expected_root_pe);
#endif
                    /*
                     * Wait for it to acknowledge, either telling us it is the
                     * root or who it thinks the root is.
                     */
                    is_dead = wait_for_coupling_ack(msg_buf_node,
                            expected_root_pe, ctx);
#ifdef COUPLING_PRINTS
                    fprintf(stderr, "PE %d got back is_dead=%d\n", ctx->pe,
                            is_dead);
#endif
                    if (is_dead) {
                        /*
                         * Should only discover dead PEs that we are trying to
                         * directly couple with.
                         */
                        assert(expected_root_pe == target_pe);
                    } else { // !is_dead
                        ack = (new_coupling_msg_ack_t *)msg_buf_node->ptr;
#ifdef COUPLING_PRINTS
                        fprintf(stderr, "PE %d got ack from %d, root=%d, "
                                "abort=%d\n", ctx->pe, ack->pe, ack->root_pe,
                                ack->abort);
#endif
                        assert(ack->pe == expected_root_pe);
                        expected_root_pe = ack->root_pe;
                    }
                } while (!is_dead && !ack->abort && ack->root_pe != ack->pe);

                // fprintf(stderr, "PE %d finally coupling with %d (abort? %d)\n",
                //         ctx->pe, ack->root_pe, ack->abort);

                if (is_dead) {
                    /*
                     * This might lead to confusing user behavior down the line
                     * as they'll request a coupling that is never satisfied.
                     * TODO
                     */
                    hvr_set_clear(target_pe, ctx->to_couple_with);
                } else if (!ack->abort) {
                    // If we were not told to stop this coupling attempt
                    unsigned long long start_negotiate = hvr_current_time_us();
                    int new_combined_root_pe = negotiate_new_root(ack->root_pe,
                            msg_buf_node, ctx);
                    *time_negotiating += hvr_current_time_us() - start_negotiate;
                    if (new_combined_root_pe != ctx->pe) {
                        /*
                         * If we are no longer root, break out and start waiting
                         * for finalized cluster info from the new root.
                         */
                        ctx->coupled_pes_root = new_combined_root_pe;
                        break;
                    }
                } else {
                    /*
                     * Were told to abort by the other PE because they are
                     * trying to couple with someone else. The PE that told us
                     * to abort may not actually be the PE we were trying to
                     * couple with, as we may have been forwarded along to their
                     * root. However, we can remove our need to re-traverse that
                     * path here by removing the original target PE from the set
                     * and inserting the new root we found. This is okay,
                     * because we know that by coupling with the root we must
                     * also become coupled with the original target.
                     */
                    assert(ack->abort);
                    naborts++;

                    hvr_set_clear(target_pe, ctx->to_couple_with);
                    hvr_set_insert(ack->root_pe, ctx->to_couple_with);
                }
            }

            // Check to see if anyone tried to couple with me
            check_incoming_coupling_requests(msg_buf_node, ctx);
            if (ctx->coupled_pes_root != ctx->pe) {
                // Got a new coupling request, and now I'm no longer root
                break;
            }

            hvr_set_and(ctx->already_coupled_with, ctx->to_couple_with,
                    ctx->coupled_pes);
        } while (!hvr_set_equal(ctx->already_coupled_with, ctx->to_couple_with));

#ifdef COUPLING_PRINTS
        fprintf(stderr, "PE %d completed couplings, current root = %d\n",
                ctx->pe, ctx->coupled_pes_root);
#endif

        // fprintf(stderr, "AA PE %d finished coupling on iter %d with root=%d, %u coupled PEs, %u to couple with, %u already coupled with, equal %d\n", ctx->pe,
        //         ctx->iter, ctx->coupled_pes_root, hvr_set_count(ctx->coupled_pes), hvr_set_count(ctx->to_couple_with), hvr_set_count(ctx->already_coupled_with), hvr_set_equal(ctx->already_coupled_with, ctx->to_couple_with));
        unsigned long long start_sharing = hvr_current_time_us();

        if (ctx->coupled_pes_root == ctx->pe) {
            // If we're still root
            hvr_set_copy(ctx->new_all_terminated_cluster_pes,
                    ctx->all_terminated_cluster_pes);
            hvr_set_or(ctx->new_all_terminated_cluster_pes,
                    ctx->terminating_pes);

            int new_root = ctx->pe;
            if (hvr_set_contains(ctx->pe, ctx->terminating_pes)) {
                // Need to choose a new root
                new_root = -1;
                for (int p = 0; p < ctx->npes; p++) {
                    if (hvr_set_contains(p, ctx->coupled_pes) &&
                            !hvr_set_contains(p, ctx->new_all_terminated_cluster_pes)) {
                        new_root = p;
                    }
                }
                if (new_root < 0) {
                    /*
                     * We only can't select a new root if all coupled PEs are
                     * also terminated or terminating. In which case, we don't
                     * really care about the new root.
                     */
                    assert(hvr_set_equal(ctx->coupled_pes,
                                ctx->new_all_terminated_cluster_pes));
                }
            }

            for (int p = 0; p < ctx->npes; p++) {
                if (hvr_set_contains(p, ctx->coupled_pes)) {
                    hvr_set_msg_send(p, new_root, ctx->iter, 42,
                            &ctx->coupling_mailbox,
                            &ctx->new_all_terminated_cluster_pes_msg);
                    hvr_set_msg_send(p, new_root, ctx->iter, 43,
                            &ctx->coupling_mailbox,
                            &ctx->coupled_pes_msg);
                }
            }
        }

#ifdef COUPLING_PRINTS
        fprintf(stderr, "PE %d now waiting for finalized couplings on iter %u, "
                "root=%d\n", ctx->pe, ctx->iter, ctx->coupled_pes_root);
#endif

        unsigned long long start_waiting = hvr_current_time_us();
        wait_for_full_coupling_info(msg_buf_node,
                ctx->coupled_pes_root == ctx->pe, ctx);
        unsigned long long done_waiting = hvr_current_time_us();

        *time_waiting_for_prev = start_processing - start_waiting_for_prev;
        *time_processing_new_requests = start_sharing - start_processing;
        *time_sharing_info = start_waiting - start_sharing;
        *time_waiting_for_info = done_waiting - start_waiting;
#ifdef COUPLING_PRINTS
        fprintf(stderr, "PE %d done waiting for finalized couplings on iter "
                "%u, root=%d, terminating=%d\n", ctx->pe, ctx->iter,
                ctx->coupled_pes_root, terminating);
#endif
    } else {
        // fprintf(stderr, "PE %d waiting on root terminating? %d\n", ctx->pe, terminating);
        unsigned long long start_waiting = hvr_current_time_us();
        wait_for_full_coupling_info(msg_buf_node, 0, ctx);
        *time_waiting_for_info = hvr_current_time_us() - start_waiting;
        // fprintf(stderr, "PE %d done waiting on root terminating? %d\n", ctx->pe, terminating);
    }

    // fprintf(stderr, "BB PE %d finished coupling on iter %d with root=%d, %u coupled PEs, %u to couple with, %u already coupled with\n", ctx->pe,
    //         ctx->iter, ctx->coupled_pes_root, hvr_set_count(ctx->coupled_pes), hvr_set_count(ctx->to_couple_with), hvr_set_count(ctx->already_coupled_with));

    hvr_set_copy(ctx->prev_coupled_pes, ctx->coupled_pes);

    const unsigned long long interm = hvr_current_time_us();

    hvr_coupling_msg_t coupled_val_msg;
    coupled_val_msg.pe = ctx->pe;
    coupled_val_msg.updates_on_this_iter = count_updated;
    coupled_val_msg.iter = ctx->iter;
    memcpy(&coupled_val_msg.val, coupled_metric, sizeof(*coupled_metric));

    hvr_set_wipe(ctx->received_from);
    for (int p = 0; p < ctx->npes; p++) {
        if (hvr_set_contains(p, ctx->coupled_pes)) {
            if (hvr_set_contains(p, ctx->prev_all_terminated_cluster_pes)) {
                hvr_set_insert(p, ctx->received_from);
            } else {
                // fprintf(stderr, "PE %d sending coupled val to %d\n", ctx->pe, p);
                hvr_mailbox_send(&coupled_val_msg, sizeof(coupled_val_msg), p,
                        -1, &ctx->coupling_val_mailbox, 0);
            }
        }
    }

    memset(ctx->updates_on_this_iter, 0x00,
            ctx->npes * sizeof(ctx->updates_on_this_iter[0]));
    while (!hvr_set_equal(ctx->received_from, ctx->coupled_pes)) {
        size_t msg_len;
        int success = hvr_mailbox_recv(msg_buf_node->ptr,
                msg_buf_node->buf_size, &msg_len, &ctx->coupling_val_mailbox, 0);
        if (success) {
            assert(msg_len == sizeof(hvr_coupling_msg_t));
            hvr_coupling_msg_t *msg = (hvr_coupling_msg_t *)msg_buf_node->ptr;
            assert(hvr_set_contains(msg->pe, ctx->coupled_pes));
            assert(!hvr_set_contains(msg->pe, ctx->received_from));

            receive_coupled_val(msg, ctx);
            hvr_set_insert(msg->pe, ctx->received_from);
        }
    }

    // fprintf(stderr, "PE %d done waiting for coupled values, root=%d, # coupled=%d\n",
    //         ctx->pe, ctx->coupled_pes_root, hvr_set_count(ctx->coupled_pes));

    const unsigned long long done = hvr_current_time_us();

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

    const unsigned long long before_should_terminate = hvr_current_time_us();

    int should_abort = 1;
    if (!terminating) {
        if (ctx->should_terminate) {
            hvr_vertex_iter_all_init(&iter, ctx);
            should_abort = ctx->should_terminate(&iter, ctx,
                    ctx->coupled_pes_values + ctx->pe, // Local coupled metric
                    ctx->coupled_pes_values, // Coupled values for each PE
                    coupled_metric, // Global coupled metric (sum reduction)
                    ctx->coupled_pes, ncoupled, ctx->updates_on_this_iter,
                    ctx->all_terminated_cluster_pes);
        } else {
            should_abort = 0;
        }
    }

    hvr_msg_buf_pool_release(msg_buf_node, &ctx->msg_buf_pool);

    const unsigned long long end_func = hvr_current_time_us();

    *time_coupling = interm - start_coupling;
    *time_waiting = (done - interm);
    *time_after = (before_should_terminate - done);

    *time_should_terminate = (end_func - before_should_terminate);
    *out_naborts = naborts;

    return should_abort;
}

#ifdef DETAILED_PRINTS
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
#endif

static void save_profiling_info(
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
        unsigned long long time_updating_subscribers,
        unsigned long long time_1,
        unsigned long long time_2,
        unsigned long long time_3_4,
        unsigned long long time_5,
        unsigned long long time_sending,
        int should_abort,
        unsigned long long coupling_coupling,
        unsigned long long coupling_waiting,
        unsigned long long coupling_after,
        unsigned long long coupling_should_terminate,
        unsigned coupling_naborts,
        unsigned long long coupling_waiting_for_prev,
        unsigned long long coupling_processing_new_requests,
        unsigned long long coupling_sharing_info,
        unsigned long long coupling_waiting_for_info,
        unsigned long long coupling_negotiating,
        unsigned n_coupled_pes,
        hvr_internal_ctx_t *ctx) {
    static int printed_warning = 0;
    if (n_profiled_iters == MAX_PROFILED_ITERS) {
        if (!printed_warning) {
            fprintf(stderr, "WARNING: PE %d exceeded number of iters that can be "
                    "profiled. Remaining iterations will not be reported.\n",
                    ctx->pe);
            printed_warning = 1;
        }
        return;
    }

    saved_profiling_info[n_profiled_iters].start_iter = start_iter;
    saved_profiling_info[n_profiled_iters].end_start_time_step = end_start_time_step;
    saved_profiling_info[n_profiled_iters].end_update_vertices = end_update_vertices;
    saved_profiling_info[n_profiled_iters].end_update_dist = end_update_dist;
    saved_profiling_info[n_profiled_iters].end_update_partitions = end_update_partitions;
    saved_profiling_info[n_profiled_iters].end_partition_window = end_partition_window;
    saved_profiling_info[n_profiled_iters].end_neighbor_updates = end_neighbor_updates;
    saved_profiling_info[n_profiled_iters].end_send_updates = end_send_updates;
    saved_profiling_info[n_profiled_iters].end_vertex_updates = end_vertex_updates;
    saved_profiling_info[n_profiled_iters].end_update_coupled = end_update_coupled;
    saved_profiling_info[n_profiled_iters].count_updated = count_updated;
    saved_profiling_info[n_profiled_iters].n_updates_sent = n_updates_sent;
    memcpy(&saved_profiling_info[n_profiled_iters].perf_info,
            perf_info, sizeof(*perf_info));
    saved_profiling_info[n_profiled_iters].time_updating_partitions =
        time_updating_partitions;
    saved_profiling_info[n_profiled_iters].time_updating_subscribers =
        time_updating_subscribers;
    saved_profiling_info[n_profiled_iters].time_1 = time_1;
    saved_profiling_info[n_profiled_iters].time_2 = time_2;
    saved_profiling_info[n_profiled_iters].time_3_4 = time_3_4;
    saved_profiling_info[n_profiled_iters].time_5 = time_5;
    saved_profiling_info[n_profiled_iters].time_sending = time_sending;
    saved_profiling_info[n_profiled_iters].should_abort = should_abort;

    saved_profiling_info[n_profiled_iters].coupling_coupling = coupling_coupling;
    saved_profiling_info[n_profiled_iters].coupling_waiting = coupling_waiting;
    saved_profiling_info[n_profiled_iters].coupling_after = coupling_after;
    saved_profiling_info[n_profiled_iters].coupling_should_terminate =
        coupling_should_terminate;
    saved_profiling_info[n_profiled_iters].coupling_naborts =
        coupling_naborts;
    saved_profiling_info[n_profiled_iters].coupling_waiting_for_prev =
        coupling_waiting_for_prev;
    saved_profiling_info[n_profiled_iters].coupling_processing_new_requests =
        coupling_processing_new_requests;
    saved_profiling_info[n_profiled_iters].coupling_sharing_info =
        coupling_sharing_info;
    saved_profiling_info[n_profiled_iters].coupling_waiting_for_info =
        coupling_waiting_for_info;
    saved_profiling_info[n_profiled_iters].coupling_negotiating =
        coupling_negotiating;
    saved_profiling_info[n_profiled_iters].n_coupled_pes =
        hvr_set_count(ctx->coupled_pes);
    saved_profiling_info[n_profiled_iters].coupled_pes_root =
        ctx->coupled_pes_root;

    saved_profiling_info[n_profiled_iters].iter = ctx->iter;
    saved_profiling_info[n_profiled_iters].pe = ctx->pe;

    saved_profiling_info[n_profiled_iters].n_partitions = ctx->n_partitions;
    saved_profiling_info[n_profiled_iters].n_producer_partitions =
        hvr_set_count(ctx->produced_partitions);
    saved_profiling_info[n_profiled_iters].n_subscriber_partitions =
        hvr_set_count(ctx->subscribed_partitions);
    saved_profiling_info[n_profiled_iters].n_allocated_verts =
        hvr_n_local_vertices(ctx);
    saved_profiling_info[n_profiled_iters].n_mirrored_verts =
        ctx->vec_cache.n_cached_vertices;

#ifdef PRINT_PARTITIONS
#define PARTITIONS_STR_BUFSIZE (1024 * 1024)
    saved_profiling_info[n_profiled_iters].producer_partitions_str =
        (char *)malloc_helper(PARTITIONS_STR_BUFSIZE);
    saved_profiling_info[n_profiled_iters].subscriber_partitions_str =
        (char *)malloc_helper(PARTITIONS_STR_BUFSIZE);
    assert(saved_profiling_info[n_profiled_iters].producer_partitions_str &&
            saved_profiling_info[n_profiled_iters].subscriber_partitions_str);

    hvr_set_to_string(ctx->produced_partitions,
            saved_profiling_info[n_profiled_iters].producer_partitions_str,
            PARTITIONS_STR_BUFSIZE, NULL);
    hvr_set_to_string(ctx->subscribed_partitions,
            saved_profiling_info[n_profiled_iters].subscriber_partitions_str,
            PARTITIONS_STR_BUFSIZE, NULL);
#endif

#ifdef DETAILED_PRINTS
    size_t vertex_cache_allocated, vertex_cache_used,
           vertex_cache_symm_allocated, vertex_cache_symm_used;

    size_t edge_bytes_used, edge_bytes_allocated, edge_bytes_capacity,
           max_edges, max_edges_index;
    hvr_irr_matrix_usage(&edge_bytes_used, &edge_bytes_capacity,
            &edge_bytes_allocated, &max_edges, &max_edges_index, &ctx->edges);

    hvr_vertex_cache_mem_used(&vertex_cache_used, &vertex_cache_allocated,
            &vertex_cache_symm_used, &vertex_cache_symm_allocated,
            &ctx->vec_cache);

    saved_profiling_info[n_profiled_iters].remote_partition_subs_bytes =
        hvr_sparse_arr_used_bytes(&ctx->remote_partition_subs);
    saved_profiling_info[n_profiled_iters].remote_vert_subs_bytes =
        hvr_sparse_arr_used_bytes(&ctx->remote_vert_subs);
    saved_profiling_info[n_profiled_iters].my_vert_subs_bytes =
        hvr_sparse_arr_used_bytes(&ctx->my_vert_subs);

    size_t producer_info_capacity, producer_info_used;
    hvr_map_size_in_bytes(&ctx->producer_info, &producer_info_capacity,
            &producer_info_used, sizeof(hvr_dist_bitvec_local_subcopy_t));
    saved_profiling_info[n_profiled_iters].producer_info_bytes =
        producer_info_capacity;

    size_t dead_info_capacity, dead_info_used;
    hvr_map_size_in_bytes(&ctx->dead_info, &dead_info_capacity,
            &dead_info_used, sizeof(hvr_dist_bitvec_local_subcopy_t));
    saved_profiling_info[n_profiled_iters].dead_info_bytes =
        dead_info_capacity;

    saved_profiling_info[n_profiled_iters].vertex_cache_allocated =
        vertex_cache_allocated;
    saved_profiling_info[n_profiled_iters].vertex_cache_used =
        vertex_cache_used;
    saved_profiling_info[n_profiled_iters].vertex_cache_symm_allocated =
        vertex_cache_symm_allocated;
    saved_profiling_info[n_profiled_iters].vertex_cache_symm_used =
        vertex_cache_symm_used;
    saved_profiling_info[n_profiled_iters].edge_bytes_used =
        edge_bytes_used;
    saved_profiling_info[n_profiled_iters].edge_bytes_allocated =
        edge_bytes_allocated;
    saved_profiling_info[n_profiled_iters].edge_bytes_capacity =
        edge_bytes_capacity;
    saved_profiling_info[n_profiled_iters].max_edges = max_edges;
    saved_profiling_info[n_profiled_iters].max_edges_index = max_edges_index;

    // Mailboxes
    saved_profiling_info[n_profiled_iters].mailbox_bytes_used = 
        hvr_mailbox_mem_used(&ctx->vertex_update_mailbox) +
        hvr_mailbox_mem_used(&ctx->vertex_delete_mailbox) +
        hvr_mailbox_mem_used(&ctx->forward_mailbox) +
        hvr_mailbox_mem_used(&ctx->vert_sub_mailbox) +
        hvr_mailbox_mem_used(&ctx->edge_create_mailbox) +
        hvr_mailbox_mem_used(&ctx->vertex_msg_mailbox) +
        hvr_mailbox_mem_used(&ctx->coupling_mailbox) +
        hvr_mailbox_mem_used(&ctx->coupling_ack_and_dead_mailbox) +
        hvr_mailbox_mem_used(&ctx->coupling_val_mailbox) +
        hvr_mailbox_mem_used(&ctx->to_couple_with_mailbox) +
        hvr_mailbox_mem_used(&ctx->root_info_mailbox);

    saved_profiling_info[n_profiled_iters].msg_buf_pool_bytes_used =
        hvr_msg_buf_pool_mem_used(&ctx->msg_buf_pool);

    saved_profiling_info[n_profiled_iters].buffered_msgs_bytes_used =
        hvr_buffered_msgs_mem_used(&ctx->buffered_msgs);

    // Space used by my portion of the distributed bit vectors
    saved_profiling_info[n_profiled_iters].partition_producers_bytes_used =
        hvr_dist_bitvec_mem_used(&ctx->partition_producers);
    saved_profiling_info[n_profiled_iters].terminated_pes_bytes_used =
        hvr_dist_bitvec_mem_used(&ctx->terminated_pes);

    // Lists of vertices per partition
    saved_profiling_info[n_profiled_iters].mirror_partition_lists_bytes =
        hvr_partition_list_mem_used(&ctx->mirror_partition_lists);
    saved_profiling_info[n_profiled_iters].local_partition_lists_bytes =
        hvr_partition_list_mem_used(&ctx->local_partition_lists);

    // Active partition lists
    saved_profiling_info[n_profiled_iters].active_partition_lists_bytes =
        4 * ctx->max_active_partitions * sizeof(hvr_partition_t);
#endif

    n_profiled_iters++;
}

static void print_profiling_info(profiling_info_t *info) {

    fprintf(profiling_fp, "PE %d - iter %d - total %f ms\n",
            info->pe, info->iter,
            (double)(info->end_update_coupled - info->start_iter) / MS_PER_S);
    fprintf(profiling_fp, "  start time step %f\n",
            (double)(info->end_start_time_step - info->start_iter) / MS_PER_S);
    fprintf(profiling_fp, "  update vertices %f - %d updates\n",
            (double)(info->end_update_vertices - info->end_start_time_step) / MS_PER_S,
            info->count_updated);
    fprintf(profiling_fp, "  update distances %f\n",
            (double)(info->end_update_dist - info->end_update_vertices) / MS_PER_S);
    fprintf(profiling_fp, "  update actor partitions %f\n",
            (double)(info->end_update_partitions - info->end_update_dist) / MS_PER_S);
    fprintf(profiling_fp, "  update partition window %f - update time = "
            "(parts=%f subscribers=%f - %f %f %f %f)\n",
            (double)(info->end_partition_window - info->end_update_partitions) / MS_PER_S,
            (double)info->time_updating_partitions / MS_PER_S,
            (double)info->time_updating_subscribers / MS_PER_S,
            (double)info->time_1 / MS_PER_S,
            (double)info->time_2 / MS_PER_S,
            (double)info->time_3_4 / MS_PER_S,
            (double)info->time_5 / MS_PER_S);
    fprintf(profiling_fp, "  send updates %f - %u changes, %f ms sending\n",
            (double)(info->end_send_updates - info->end_partition_window) / MS_PER_S,
            info->n_updates_sent, (double)info->time_sending / MS_PER_S);

    fprintf(profiling_fp, "  update neighbors %f\n",
            (double)(info->end_neighbor_updates - info->end_send_updates) / MS_PER_S);
    fprintf(profiling_fp, "  process vertex updates %f - %u received\n",
            (double)(info->end_vertex_updates - info->end_neighbor_updates) / MS_PER_S,
            info->perf_info.n_received_updates);
    fprintf(profiling_fp, "    %f on deletes\n",
            (double)info->perf_info.time_handling_deletes / MS_PER_S);
    fprintf(profiling_fp, "    %f on news\n",
            (double)info->perf_info.time_handling_news / MS_PER_S);
    fprintf(profiling_fp, "      %f s on creating new\n",
            (double)info->perf_info.time_creating / MS_PER_S);
    fprintf(profiling_fp, "      %f s on updates - %f updating edges, "
            "%f creating edges - %u should_have_edges\n",
            (double)info->perf_info.time_updating / MS_PER_S,
            (double)info->perf_info.time_updating_edges / MS_PER_S,
            (double)info->perf_info.time_creating_edges / MS_PER_S,
            info->perf_info.count_new_should_have_edges);
    fprintf(profiling_fp, "  coupling %f - %f ms coupling, "
            "%f ms waiting, "
            "%f ms adding, "
            "%f ms on should terminate, "
            "%u aborts, "
            "%u coupled, "
            "root=%d, "
            "%f for prev, "
            "%f processing (%f negotiating), "
            "%f sharing, "
            "%f waiting\n",
            (double)(info->end_update_coupled - info->end_vertex_updates) / MS_PER_S,
            (double)info->coupling_coupling / MS_PER_S,
            (double)info->coupling_waiting / MS_PER_S,
            (double)info->coupling_after / MS_PER_S,
            (double)info->coupling_should_terminate / MS_PER_S,
            info->coupling_naborts,
            info->n_coupled_pes,
            info->coupled_pes_root,
            (double)info->coupling_waiting_for_prev / MS_PER_S,
            (double)info->coupling_processing_new_requests / MS_PER_S,
            (double)info->coupling_negotiating / MS_PER_S,
            (double)info->coupling_sharing_info / MS_PER_S,
            (double)info->coupling_waiting_for_info / MS_PER_S);
    fprintf(profiling_fp, "  %d / %d producer partitions and %d / %d "
            "subscriber partitions for %lu local vertices, %llu mirrored "
            "vertices\n",
            info->n_producer_partitions,
            info->n_partitions,
            info->n_subscriber_partitions,
            info->n_partitions,
            info->n_allocated_verts,
            info->n_mirrored_verts);
    fprintf(profiling_fp, "  aborting? %d\n", info->should_abort);

#ifdef PRINT_PARTITIONS
    fprintf(profiling_fp, "  producer partitions = %s\n",
            info->producer_partitions_str);
    fprintf(profiling_fp, "  subscriber partitions = %s\n",
            info->subscriber_partitions_str);
#endif

#ifdef DETAILED_PRINTS
    fprintf(profiling_fp, "  mem usage info:\n");
    fprintf(profiling_fp, "    PE sub info = %f MB\n",
            (double)info->remote_partition_subs_bytes / (1024.0 * 1024.0));
    fprintf(profiling_fp, "    Vert sub info = %f MB\n",
            (double)info->remote_vert_subs_bytes / (1024.0 * 1024.0));
    fprintf(profiling_fp, "    My vert sub info = %f MB\n",
            (double)info->my_vert_subs_bytes / (1024.0 * 1024.0));
    fprintf(profiling_fp, "    producer info = %f MB, dead info = %f MB\n",
            (double)info->producer_info_bytes / (1024.0 * 1024.0),
            (double)info->dead_info_bytes / (1024.0 * 1024.0));
    fprintf(profiling_fp, "    vertex cache = %f MB / %f MB (%f%%), symm = "
            "%f MB / %f MB (%f%%)\n",
            (double)info->vertex_cache_used / (1024.0 * 1024.0),
            (double)info->vertex_cache_allocated / (1024.0 * 1024.0),
            100.0 * (double)info->vertex_cache_used /
            (double)info->vertex_cache_allocated,
            (double)info->vertex_cache_symm_used / (1024.0 * 1024.0),
            (double)info->vertex_cache_symm_allocated / (1024.0 * 1024.0),
            100.0 * (double)info->vertex_cache_symm_used /
            (double)info->vertex_cache_symm_allocated);
    fprintf(profiling_fp, "    edge set: used=%f MB, allocated=%f MB, "
            "capacity=%f MB, max # edges=%lu, max edges index=%llu\n",
            (double)info->edge_bytes_used / (1024.0 * 1024.0),
            (double)info->edge_bytes_allocated / (1024.0 * 1024.0),
            (double)info->edge_bytes_capacity / (1024.0 * 1024.0),
            info->max_edges, info->max_edges_index);
    fprintf(profiling_fp, "    total bytes for all mailboxes = %f MB\n",
            (double)info->mailbox_bytes_used / (1024.0 * 1024.0));
    fprintf(profiling_fp, "    msg buf pool = %f MB\n",
            (double)info->msg_buf_pool_bytes_used / (1024.0 * 1024.0));
    fprintf(profiling_fp, "    buffered msgs = %f MB\n",
            (double)info->buffered_msgs_bytes_used / (1024.0 * 1024.0));
    fprintf(profiling_fp, "    partition producers dist. bitvec = %f MB\n",
            (double)info->partition_producers_bytes_used / (1024.0 * 1024.0));
    fprintf(profiling_fp, "    terminated PEs dist. bitvec = %f MB\n",
            (double)info->terminated_pes_bytes_used / (1024.0 * 1024.0));
    fprintf(profiling_fp, "    local part list = %f MB, mirror = %f MB\n",
            (double)info->local_partition_lists_bytes / (1024.0 * 1024.0),
            (double)info->mirror_partition_lists_bytes / (1024.0 * 1024.0));
    fprintf(profiling_fp, "    active partition lists bytes = %f MB\n",
            (double)info->active_partition_lists_bytes / (1024.0 * 1024.0));
#endif
}

static void write_vertex_to_file(FILE *fp, hvr_vertex_t *curr,
        hvr_internal_ctx_t *ctx) {
    fprintf(fp, "%u,%d,%lu,%u", ctx->iter, ctx->pe, curr->id,
            HVR_MAX_VECTOR_SIZE);
    for (unsigned f = 0; f < HVR_MAX_VECTOR_SIZE; f++) {
        fprintf(fp, ",%u,%f", f, hvr_vertex_get(f, curr, ctx));
    }
    fprintf(fp, "\n");
    fflush(fp);
}

static void save_cached_state_to_dump_file(hvr_internal_ctx_t *ctx) {
    hvr_vertex_cache_t *c = &ctx->vec_cache;
    for (hvr_partition_t p = 0; p < ctx->n_partitions; p++) {
        hvr_vertex_t *iter = hvr_partition_list_head(p,
                &ctx->mirror_partition_lists);
        while (iter) {
            write_vertex_to_file(ctx->cache_dump_file, iter, ctx);
            iter = iter->next_in_partition;
        }
    }
}

static void save_local_state_to_dump_file(hvr_internal_ctx_t *ctx) {
    hvr_vertex_iter_t iter;
    hvr_vertex_iter_init(&iter, ctx);
    for (hvr_vertex_t *curr = hvr_vertex_iter_next(&iter); curr;
            curr = hvr_vertex_iter_next(&iter)) {
        write_vertex_to_file(ctx->dump_file, curr, ctx);

        hvr_vertex_t **vertices;
        hvr_edge_type_t *edges;
        int n_neighbors = hvr_get_neighbors(curr, &vertices, &edges, ctx);

        fprintf(ctx->edges_dump_file, "%u,%d,%lu,%d", ctx->iter,
                ctx->pe, curr->id, n_neighbors);
        for (int n = 0; n < n_neighbors; n++) {
            hvr_vertex_t *vert = vertices[n];
            hvr_edge_type_t edge = edges[n];
            switch (edge) {
                case DIRECTED_IN:
                    fprintf(ctx->edges_dump_file, ",IN");
                    break;
                case DIRECTED_OUT:
                    fprintf(ctx->edges_dump_file, ",OUT");
                    break;
                case BIDIRECTIONAL:
                    fprintf(ctx->edges_dump_file, ",BI");
                    break;
                default:
                    abort();
            }
            fprintf(ctx->edges_dump_file, ":%lu:%lu", VERTEX_ID_PE(vert->id),
                    VERTEX_ID_OFFSET(vert->id));
        }
        fprintf(ctx->edges_dump_file, "\n");
    }
    fflush(ctx->dump_file);
    fflush(ctx->edges_dump_file);
}

hvr_exec_info hvr_body(hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    hvr_set_t *to_couple_with = hvr_create_empty_set(ctx->npes);
    if (getenv("HVR_HANG_ABORT")) {
        int i_abort = 1;
        if (getenv("HVR_HANG_ABORT_PE")) {
            int pe = atoi(getenv("HVR_HANG_ABORT_PE"));
            i_abort = (pe == shmem_my_pe());
        }

        if (i_abort) {
            pthread_t aborting_pthread;
            const int pthread_err = pthread_create(&aborting_pthread, NULL,
                    aborting_thread, NULL);
            assert(pthread_err == 0);
        }
    }

    if (this_pe_has_exited) {
        // More throught needed to make sure this behavior would be safe
        fprintf(stderr, "[ERROR] HOOVER does not currently support re-entering "
                "hvr_body\n");
        abort();
    }

    shmem_barrier_all();

    const unsigned long long start_body = hvr_current_time_us();

    /*
     * Find which partitions are locally active (either because a local vertex
     * is in them, or a locally mirrored vertex in them).
     */
    insert_recently_created_in_partitions(ctx);

    // Update distance of each mirrored vertex from a local vertex
    update_distances(ctx);

    const unsigned long long end_update_dist = hvr_current_time_us();

    const unsigned long long end_update_partitions = hvr_current_time_us();

    /*
     * Compute a set of all partitions which locally active partitions might
     * need notifications on. Subscribe to notifications about partitions from
     * update_partition_window in the global registry.
     */
    unsigned long long time_updating_partitions, time_updating_subscribers,
                  time_1, time_2, time_3_4, time_5;
    update_partition_window(ctx, &time_updating_partitions,
            &time_updating_subscribers, &time_1, &time_2, &time_3_4, &time_5);
    const unsigned long long end_partition_window = hvr_current_time_us();

    /*
     * Ensure everyone's partition windows are initialized before initializing
     * neighbors.
     */
    shmem_barrier_all();

    /*
     * Process updates sent to us by neighbors via our main mailbox. Use these
     * updates to update edges for all vertices (both local and mirrored).
     */
    process_perf_info_t perf_info;
    memset(&perf_info, 0x00, sizeof(perf_info));
    unsigned long long time_sending = 0;
    unsigned n_updates_sent = send_vertex_updates(ctx, &time_sending,
            &perf_info);

    // Ensure all updates are sent before processing them during initialization
    const unsigned long long end_send_updates = hvr_current_time_us();

    /*
     * Receive and process updates sent in update_pes_on_neighbors, and adjust
     * our local neighbors based on those updates. This stores a mapping from
     * partition to the PEs that are subscribed to updates in that partition
     * inside of ctx->remote_partition_subs.
     */
    process_neighbor_updates(ctx, &perf_info);
    const unsigned long long end_neighbor_updates = hvr_current_time_us();


    perf_info.n_received_updates += process_vertex_updates(ctx, &perf_info);
    process_incoming_messages(ctx);
    const unsigned long long end_vertex_updates = hvr_current_time_us();

    hvr_vertex_t coupled_metric;
    unsigned long long coupling_coupling,
                  coupling_waiting, coupling_after,
                  coupling_should_terminate, coupling_waiting_for_prev,
                  coupling_processing_new_requests, coupling_sharing_info,
                  coupling_waiting_for_info, coupling_negotiating;
    unsigned coupling_naborts;
    int should_abort = update_coupled_values(ctx, &coupled_metric,
            n_updates_sent,
            &coupling_coupling,
            &coupling_waiting, &coupling_after,
            &coupling_should_terminate, &coupling_naborts,
            &coupling_waiting_for_prev, &coupling_processing_new_requests,
            &coupling_sharing_info, &coupling_waiting_for_info,
            &coupling_negotiating, 0);
    const unsigned long long end_update_coupled = hvr_current_time_us();

    if (print_profiling) {
        save_profiling_info(
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
                time_updating_subscribers,
                time_1, time_2, time_3_4, time_5,
                time_sending,
                should_abort,
                coupling_coupling,
                coupling_waiting, coupling_after,
                coupling_should_terminate,
                coupling_naborts,
                coupling_waiting_for_prev,
                coupling_processing_new_requests,
                coupling_sharing_info,
                coupling_waiting_for_info,
                coupling_negotiating,
                0,
                ctx);
    }

    ctx->iter += 1;
    while (!should_abort && hvr_current_time_us() - start_body <
            ctx->max_elapsed_seconds * 1000000ULL) {

        // print_memory_metrics(ctx);

        if (ctx->dump_mode && !ctx->only_last_iter_dump) {
            save_local_state_to_dump_file(ctx);
        }

        const unsigned long long start_iter = hvr_current_time_us();

        hvr_set_wipe(to_couple_with);

        if (ctx->start_time_step) {
            hvr_vertex_iter_t iter;
            hvr_vertex_iter_init(&iter, ctx);
            ctx->start_time_step(&iter, to_couple_with, ctx);
        }

        const unsigned long long end_start_time_step = hvr_current_time_us();

        // Must come before everything else
        const int count_updated = update_vertices(to_couple_with, ctx);

        insert_recently_created_in_partitions(ctx);

        // Update my local information on PEs I am coupled with.
        hvr_set_merge(ctx->coupled_pes, to_couple_with);

        const unsigned long long end_update_vertices = hvr_current_time_us();

        update_distances(ctx);

        const unsigned long long end_update_dist = hvr_current_time_us();

        const unsigned long long end_update_partitions = hvr_current_time_us();

        update_partition_window(ctx, &time_updating_partitions,
            &time_updating_subscribers, &time_1, &time_2, &time_3_4, &time_5);

        const unsigned long long end_partition_window = hvr_current_time_us();

        time_sending = 0;
        memset(&perf_info, 0x00, sizeof(perf_info));
        n_updates_sent = send_vertex_updates(ctx, &time_sending, &perf_info);

        const unsigned long long end_send_updates = hvr_current_time_us();

        process_neighbor_updates(ctx, &perf_info);

        hvr_mailbox_buffer_flush(&ctx->vert_sub_mailbox_buffer);
        hvr_mailbox_buffer_flush(&ctx->edge_create_mailbox_buffer);

        const unsigned long long end_neighbor_updates = hvr_current_time_us();

        perf_info.n_received_updates += process_vertex_updates(ctx, &perf_info);
        process_incoming_messages(ctx);

        const unsigned long long end_vertex_updates = hvr_current_time_us();

        should_abort = update_coupled_values(ctx, &coupled_metric,
                count_updated, &coupling_coupling,
                &coupling_waiting, &coupling_after,
                &coupling_should_terminate, &coupling_naborts,
                &coupling_waiting_for_prev, &coupling_processing_new_requests,
                &coupling_sharing_info, &coupling_waiting_for_info,
                &coupling_negotiating, 0);

        const unsigned long long end_update_coupled = hvr_current_time_us();

        if (print_profiling) {
            save_profiling_info(
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
                    time_updating_subscribers,
                    time_1, time_2, time_3_4, time_5,
                    time_sending,
                    should_abort,
                    coupling_coupling,
                    coupling_waiting, coupling_after,
                    coupling_should_terminate,
                    coupling_naborts,
                    coupling_waiting_for_prev,
                    coupling_processing_new_requests,
                    coupling_sharing_info,
                    coupling_waiting_for_info,
                    coupling_negotiating,
                    hvr_set_count(ctx->coupled_pes),
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

    update_coupled_values(ctx, &coupled_metric,
            0, &coupling_coupling,
            &coupling_waiting, &coupling_after,
            &coupling_should_terminate, &coupling_naborts,
            &coupling_waiting_for_prev, &coupling_processing_new_requests,
            &coupling_sharing_info, &coupling_waiting_for_info,
            &coupling_negotiating, 1);

    // Notify all PEs that I have terminated
    hvr_dead_pe_msg_t dead_msg;
    dead_msg.pe = ctx->pe;
    for (int p = 0; p < ctx->npes; p++) {
        hvr_mailbox_send(&dead_msg, sizeof(dead_msg), p, -1,
                &ctx->coupling_ack_and_dead_mailbox, 0);
    }

    this_pe_has_exited = 1;

    /*
     * For each local vertex, persist its partition so that other PEs can come
     * look it up after I have terminated.
     */
    for (unsigned i = 0; i < ctx->vec_cache.pool_size; i++) {
        (ctx->vertex_partitions)[i] = HVR_INVALID_PARTITION;
    }
    hvr_vertex_iter_t iter;
    hvr_vertex_iter_all_init(&iter, ctx);
    for (hvr_vertex_t *curr = hvr_vertex_iter_next(&iter); curr;
            curr = hvr_vertex_iter_next(&iter)) {
        hvr_vertex_cache_node_t *curr_node = (hvr_vertex_cache_node_t *)curr;
        size_t offset = curr_node - ctx->vec_cache.pool_mem;
        (ctx->vertex_partitions)[offset] = curr_node->vert.curr_part;
    }

    /*
     * For each partition that I am a producer for, mark myself as a terminated
     * producer of that partition.
     */
    for (hvr_partition_t p = 0; p < ctx->n_partitions; p++) {
        if (hvr_partition_list_head(p, &ctx->local_partition_lists)) {
            hvr_dist_bitvec_set(p, ctx->pe, &ctx->terminated_pes, 0);
        }
    }

    if (ctx->dump_mode && ctx->only_last_iter_dump) {
        save_local_state_to_dump_file(ctx);
    }
    if (ctx->cache_dump_mode) {
        save_cached_state_to_dump_file(ctx);
    }

    hvr_set_destroy(to_couple_with);

    hvr_exec_info info;
    info.executed_iters = ctx->iter;
    return info;
}

void hvr_finalize(hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    if (print_profiling) {
        for (unsigned i = 0; i < n_profiled_iters; i++) {
            print_profiling_info(&saved_profiling_info[i]);
        }
        fclose(profiling_fp);
    }

    if (ctx->dump_mode) {
        fclose(ctx->dump_file);
    }
    if (ctx->cache_dump_mode) {
        fclose(ctx->cache_dump_file);
    }

    shmem_barrier_all();

    if (dead_pe_processing) {
        shmem_free(ctx->vertex_partitions);
    }

    free(ctx->coupled_pes_values);
    free(ctx->updates_on_this_iter);
    hvr_partition_list_destroy(&ctx->local_partition_lists);
    hvr_partition_list_destroy(&ctx->mirror_partition_lists);
    free(ctx->partition_min_dist_from_local_vert);

    hvr_vertex_cache_destroy(&ctx->vec_cache);

    hvr_mailbox_destroy(&ctx->vertex_update_mailbox);
    hvr_mailbox_destroy(&ctx->vertex_delete_mailbox);
    hvr_mailbox_destroy(&ctx->forward_mailbox);
    hvr_mailbox_destroy(&ctx->vert_sub_mailbox);
    hvr_mailbox_destroy(&ctx->edge_create_mailbox);
    hvr_mailbox_destroy(&ctx->coupling_mailbox);
    hvr_mailbox_destroy(&ctx->coupling_ack_and_dead_mailbox);
    hvr_mailbox_destroy(&ctx->coupling_val_mailbox);
    hvr_mailbox_destroy(&ctx->root_info_mailbox);

    hvr_sparse_arr_destroy(&ctx->remote_partition_subs);
    hvr_sparse_arr_destroy(&ctx->remote_vert_subs);
    hvr_sparse_arr_destroy(&ctx->my_vert_subs);

    hvr_map_destroy(&ctx->producer_info);
    hvr_map_destroy(&ctx->dead_info);

    free(ctx->vert_partition_buf);

    hvr_msg_buf_pool_destroy(&ctx->msg_buf_pool);

    free(ctx->interacting);

#ifdef HVR_MULTITHREADED
#pragma omp parallel
    {
        shmemx_thread_unregister();
    }
#endif

    free(ctx);
}

int hvr_my_pe(hvr_ctx_t ctx) {
    return ctx->pe;
}

unsigned long long hvr_current_time_us() {
    struct timespec monotime;
    clock_gettime(CLOCK_MONOTONIC, &monotime);
    return monotime.tv_sec * 1000000ULL + monotime.tv_nsec / 1000;

    // struct timeval curr_time;
    // gettimeofday(&curr_time, NULL);
    // return curr_time.tv_sec * 1000000ULL + curr_time.tv_usec;
}

size_t hvr_n_local_vertices(hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    return ctx->vec_cache.n_local_vertices;
}
