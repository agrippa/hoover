/* For license: see LICENSE.txt file at top-level */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <shmem.h>
#include <math.h>
#include <string.h>

#include <hoover.h>

static int npes, pe;
long long p_wrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
int int_p_wrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
long p_sync[SHMEM_REDUCE_SYNC_SIZE];

hvr_edge_type_t should_have_edge(hvr_vertex_t *base, hvr_vertex_t *neighbor,
        hvr_ctx_t ctx) {
    return BIDIRECTIONAL;
}

hvr_partition_t actor_to_partition(hvr_vertex_t *actor, hvr_ctx_t ctx) {
    return 0;
}

void start_time_step(hvr_vertex_iter_t *iter, hvr_set_t *couple_with,
        hvr_ctx_t ctx) {
    if (pe < npes) {
        for (int i = 0; i < npes; i++) {
            hvr_set_insert(i, couple_with);
        }
    }
}

void might_interact(const hvr_partition_t partition,
        hvr_partition_t *interacting_partitions,
        unsigned *n_interacting_partitions,
        unsigned interacting_partitions_capacity,
        hvr_ctx_t ctx) {
    interacting_partitions[0] = 0;
    *n_interacting_partitions = 1;
}

void update_coupled_val(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_vertex_t *out_coupled_metric) {
    hvr_vertex_set(0, pe, out_coupled_metric, ctx);
    hvr_vertex_set(1, ctx->iter, out_coupled_metric, ctx);
}

int should_terminate(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_vertex_t *local_coupled_metric, // coupled_pes[shmem_my_pe()]
        hvr_vertex_t *all_coupled_metrics, // Each PE's val
        hvr_vertex_t *global_coupled_metric, // Sum reduction of coupled_pes
        hvr_set_t *coupled_pes, // Set of coupled PEs
        int n_coupled_pes, int *updates_on_this_iter,
        hvr_set_t *terminated_coupled_pes) {

    int sum_updates = 0;
    for (int i = 0; i < npes; i++) {
        sum_updates += updates_on_this_iter[i];
    }

    static hvr_vertex_t *prev_values = NULL;
    assert(n_coupled_pes == 1 || n_coupled_pes == npes);
    if (n_coupled_pes == npes) {

        for (int i = 0; i < npes; i++) {
            assert((int)hvr_vertex_get(0, all_coupled_metrics + i, ctx) == i);

            if (prev_values) {
                assert(hvr_set_contains(i, terminated_coupled_pes) ||
                        (int)hvr_vertex_get(1, all_coupled_metrics + i, ctx) ==
                        (int)hvr_vertex_get(1, prev_values + i, ctx) + 1);
            }
        }

        if (prev_values == NULL) {
            prev_values = (hvr_vertex_t *)malloc(npes * sizeof(*prev_values));
            assert(prev_values);
        }
        memcpy(prev_values, all_coupled_metrics, npes * sizeof(*prev_values));
    }

    // printf("PE %d on iter %d did %d updates with %d coupled PEs.\n", pe,
    //         ctx->iter, sum_updates, n_coupled_pes);
    return 0;
}

int main(int argc, char **argv) {
    hvr_ctx_t hvr_ctx;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <time-limit>\n", argv[0]);
        return 1;
    }

    int time_limit = atoi(argv[1]);

    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        p_sync[i] = SHMEM_SYNC_VALUE;
    }

    shmem_init();
    pe = shmem_my_pe();
    npes = shmem_n_pes();

    if (pe == 0) {
        fprintf(stderr, "Running with %d\n", npes);
    }

    hvr_ctx_create(&hvr_ctx);

    hvr_init(1,
            NULL, // update_metadata
            might_interact,
            update_coupled_val,
            actor_to_partition,
            start_time_step,
            should_have_edge,
            should_terminate,
            time_limit, // max_elapsed_seconds
            1, // max_graph_traverse_depth
            hvr_ctx);

    hvr_exec_info info = hvr_body(hvr_ctx);
    printf("PE %d ran for %d iterations\n", shmem_my_pe(), info.executed_iters);

    shmem_barrier_all();

    hvr_finalize(hvr_ctx);

    shmem_finalize();

    return 0;
}
