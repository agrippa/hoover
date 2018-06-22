/* For license: see LICENSE.txt file at top-level */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <shmem.h>
#include <math.h>

#include <hoover.h>

// Timing variables
static unsigned long long start_time = 0;
static unsigned long long time_limit_us = 0;
static long long elapsed_time = 0;
static long long max_elapsed, total_time;

// SHMEM variables
static int pe, npes;
long long p_wrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
long p_sync[SHMEM_REDUCE_SYNC_SIZE];

// HOOVER variables
hvr_graph_id_t graph = HVR_INVALID_GRAPH;

uint16_t actor_to_partition(hvr_sparse_vec_t *actor, hvr_ctx_t ctx) {
    // TODO
    return 0;
}

void start_time_step(hvr_vertex_iter_t *iter, hvr_ctx_t ctx) {
    /*
     * On each time step, calculate a random number of vertices to insert. Then,
     * use hvr_sparse_vec_create_n to insert them with 3 randomly generated
     * features which are designed to be most likely to just interact with
     * vertices on this node (but possibly with vertices on other nodes).
     */

    // TODO
}

void update_metadata(hvr_sparse_vec_t *vertex, hvr_sparse_vec_t *neighbors,
        const size_t n_neighbors, hvr_set_t *couple_with, hvr_ctx_t ctx) {
    /*
     * Calculate the K most common subgraphs in our current partition of the
     * graph, which may imply fetching vertices from remote nodes.
     */
}

int might_interact(const uint16_t partition, hvr_set_t *partitions,
        uint16_t *interacting_partitions,
        unsigned *n_interacting_partitions,
        unsigned interacting_partitions_capacity,
        hvr_ctx_t ctx) {
    /*
     * If a vertex in 'partition' may create an edge with any vertices in any
     * partition in 'partitions', they might interact (so return 1).
     */
    return 0; // TODO
}

int check_abort(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_sparse_vec_t *out_coupled_metric) {
    /*
     * May be a no-op for intrusion detection? Could use this callback to
     * iterate over the graph and check for anomalous patterns, then print that
     * info to STDOUT?
     */
    hvr_sparse_vec_set(0, 0.0, out_coupled_metric, ctx);

    unsigned long long time_so_far = hvr_current_time_us() - start_time;
    return (time_so_far > time_limit_us);
}

int main(int argc, char **argv) {
    hvr_ctx_t hvr_ctx;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <time-limit-in-seconds>\n", argv[0]);
        return 1;
    }

    time_limit_us = atoi(argv[1]) * 1000 * 1000;

    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        p_sync[i] = SHMEM_SYNC_VALUE;
    }

    shmem_init();
    hvr_ctx_create(&hvr_ctx);
    graph = hvr_graph_create(hvr_ctx);

    pe = shmem_my_pe();
    npes = shmem_n_pes();

    hvr_init(100, // # partitions
            update_metadata,
            might_interact,
            check_abort,
            actor_to_partition,
            start_time_step,
            graph,
            1.0, // Edge creation distance threshold
            0, // Min spatial feature inclusive
            1, // Max spatial feature inclusive
            MAX_TIMESTAMP,
            hvr_ctx);

    start_time = hvr_current_time_us();
    hvr_body(hvr_ctx);
    elapsed_time = hvr_current_time_us() - start_time;

    // Get a total wallclock time across all PEs
    shmem_longlong_sum_to_all(&total_time, &elapsed_time, 1, 0, 0, npes, p_wrk,
            p_sync);
    shmem_barrier_all();

    // Get a max wallclock time across all PEs
    shmem_longlong_max_to_all(&max_elapsed, &elapsed_time, 1, 0, 0, npes, p_wrk,
            p_sync);
    shmem_barrier_all();

    if (pe == 0) {
        fprintf(stderr, "%d PEs, total CPU time = %f ms, max elapsed = %f ms\n",
                npes, (double)total_time / 1000.0, (double)max_elapsed / 1000.0);
    }

    hvr_finalize(hvr_ctx);

    shmem_finalize();

    return 0;
}
