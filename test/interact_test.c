/* For license: see LICENSE.txt file at top-level */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <shmem.h>
#include <math.h>

#include <hoover.h>

static int npes, pe;
long long p_wrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
long p_sync[SHMEM_REDUCE_SYNC_SIZE];
static unsigned long long start_time = 0;

hvr_partition_t actor_to_partition(hvr_vertex_t *actor, hvr_ctx_t ctx) {
    return 0;
}

hvr_edge_type_t should_have_edge(hvr_vertex_t *a, hvr_vertex_t *b,
        hvr_ctx_t ctx) {
    if (hvr_vertex_get_owning_pe(a) != hvr_vertex_get_owning_pe(b)) {
        if (hvr_vertex_get_owning_pe(a) % (npes / 2) ==
                hvr_vertex_get_owning_pe(b) % (npes / 2)) {
            if (hvr_vertex_get(1, a, ctx) > 0 && hvr_vertex_get(1, b, ctx) > 0) {
                return BIDIRECTIONAL;
            }
        }
    }
    return NO_EDGE;
}

static unsigned prev_n_neighbors = 0;

/*
 * Callback for the HOOVER runtime for updating positional or logical metadata
 * attached to each vertex based on the updated neighbors on each time step.
 */
void update_metadata(hvr_vertex_t *vertex, hvr_set_t *couple_with,
        hvr_ctx_t ctx) {
    hvr_edge_info_t *neighbors;
    size_t n_neighbors;
    hvr_get_neighbors(vertex, &neighbors, &n_neighbors, ctx);

    // printf("%llu\n", hvr_current_time_us() - start_time);
    if (hvr_current_time_us() - start_time > 10ULL * 1000ULL * 1000ULL) {
        hvr_vertex_set(1, 1, vertex, ctx);
    }

    if (prev_n_neighbors == 0) {
        assert(n_neighbors == 0 || n_neighbors == 1);
        prev_n_neighbors = n_neighbors;
    } else {
        assert((int)hvr_vertex_get(1, vertex, ctx) == 1);
        assert(prev_n_neighbors == 1);
        assert(n_neighbors == 1);
    }
    free(neighbors);
}

/*
 * Callback used to check if this PE might interact with another PE based on the
 * maximums and minimums of all vertices owned by each PE.
 */
void might_interact(const hvr_partition_t partition,
        hvr_partition_t *interacting_partitions,
        unsigned *n_interacting_partitions,
        unsigned interacting_partitions_capacity,
        hvr_ctx_t ctx) {
    assert(interacting_partitions_capacity > 0);
    interacting_partitions[0] = 0;
    *n_interacting_partitions = 1;
}

/*
 * Callback used by the HOOVER runtime to check if this PE can abort out of the
 * simulation.
 */
void update_coupled_val(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_vertex_t *out_coupled_metric) {
    // Force update_metadata to be called on all vertices on every iteration
    for (hvr_vertex_t *curr = hvr_vertex_iter_next(iter); curr;
            curr = hvr_vertex_iter_next(iter)) {
        hvr_vertex_trigger_update(curr, ctx);
    }
}

int should_terminate(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_vertex_t *local_coupled_metric, hvr_vertex_t *global_coupled_metric,
        hvr_set_t *coupled_pes, int n_coupled_pes) {
    return 0;
}

int main(int argc, char **argv) {
    hvr_ctx_t hvr_ctx;

    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        p_sync[i] = SHMEM_SYNC_VALUE;
    }

    shmem_init();
    hvr_ctx_create(&hvr_ctx);

    pe = shmem_my_pe();
    npes = shmem_n_pes();

    assert(npes % 2 == 0);

    hvr_vertex_t *vert = hvr_vertex_create(hvr_ctx);
    hvr_vertex_set(0, pe, vert, hvr_ctx);
    hvr_vertex_set(1, 0, vert, hvr_ctx);

    // Statically divide 2D grid into PARTITION_DIM x PARTITION_DIM partitions
    hvr_init(1,
            update_metadata,
            might_interact,
            update_coupled_val,
            actor_to_partition,
            NULL, // start_time_step
            should_have_edge,
            should_terminate,
            20, // max_elapsed_seconds
            1, // max_graph_traverse_depth
            hvr_ctx);

    start_time = hvr_current_time_us();
    hvr_body(hvr_ctx);

    hvr_edge_info_t *neighbors;
    size_t n_neighbors;
    hvr_get_neighbors(vert, &neighbors, &n_neighbors, hvr_ctx);
    assert(n_neighbors == 1);
    assert(VERTEX_ID_PE(neighbors[0].id) == (pe + (npes / 2)) % npes);

    char buf[2048];
    hvr_vertex_dump(vert, buf, 2048, hvr_ctx);
    printf("PE %d : %s\n", pe, buf);

    hvr_finalize(hvr_ctx);

    shmem_finalize();

    return 0;
}
