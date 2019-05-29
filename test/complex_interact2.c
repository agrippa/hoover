/* For license: see LICENSE.txt file at top-level */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <shmem.h>
#include <math.h>

#include <hoover.h>

#define ACTOR_ID 0
#define POS 1

unsigned N = 50000;
int pe, npes;

hvr_partition_t actor_to_partition(const hvr_vertex_t *vertex, hvr_ctx_t ctx) {
    return (hvr_partition_t)hvr_vertex_get(POS, vertex, ctx);
}

hvr_edge_type_t should_have_edge(const hvr_vertex_t *a, const hvr_vertex_t *b,
        hvr_ctx_t ctx) {
    if (hvr_vertex_get_owning_pe(a) != hvr_vertex_get_owning_pe(b)) {
        return BIDIRECTIONAL;
    } else {
        return NO_EDGE;
    }
}

void might_interact(const hvr_partition_t partition,
        hvr_partition_t *interacting_partitions,
        unsigned *n_interacting_partitions,
        unsigned interacting_partitions_capacity,
        hvr_ctx_t ctx) {
    *n_interacting_partitions = 0;
    if (partition > 0) {
        interacting_partitions[*n_interacting_partitions] = partition - 1;
        *n_interacting_partitions += 1;
    }
    if (partition < N - 1) {
        interacting_partitions[*n_interacting_partitions] = partition + 1;
        *n_interacting_partitions += 1;
    }
}

void update_metadata(hvr_vertex_t *vertex, hvr_set_t *couple_with,
        hvr_ctx_t ctx) {
    hvr_partition_t actor = (hvr_partition_t)hvr_vertex_get(ACTOR_ID, vertex,
            ctx);

    hvr_vertex_t **verts;
    hvr_edge_type_t *dirs;
    int n_neighbors = hvr_get_neighbors(vertex, &verts, &dirs, ctx);
    assert(n_neighbors == 0 || n_neighbors == 1);

    unsigned curr_pos = (unsigned)hvr_vertex_get(POS, vertex, ctx);
    if (actor == 0) {
        // shift +1 when we have zero neighbors, chasing actor 1
        if (n_neighbors == 0) {
            hvr_vertex_set(POS, curr_pos + 1, vertex, ctx);
        }
    } else if (actor == 1) {
        // shift +1 when we have one neighbor, running from actor 0
        if (n_neighbors == 1 && curr_pos < N - 1) {
            hvr_vertex_set(POS, curr_pos + 1, vertex, ctx);
        }
    } else {
        abort();
    }
}

void update_coupled_val(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_vertex_t *out_coupled_metric) {
    hvr_vertex_set(0, (double)0, out_coupled_metric, ctx);
}

int should_terminate(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_vertex_t *local_coupled_metric,
        hvr_vertex_t *all_coupled_metrics,
        hvr_vertex_t *global_coupled_metric,
        hvr_set_t *coupled_pes,
        int n_coupled_pes, int *updates_on_this_iter,
        hvr_set_t *terminated_coupled_pes) {
    return 0;
}

int main(int argc, char **argv) {
    hvr_ctx_t hvr_ctx;

    shmem_init();
    hvr_ctx_create(&hvr_ctx);

    pe = shmem_my_pe();
    npes = shmem_n_pes();
    assert(npes == 2);

    hvr_vertex_t *vertices = hvr_vertex_create(hvr_ctx);
    hvr_vertex_set(ACTOR_ID, pe, vertices, hvr_ctx);
    hvr_vertex_set(POS, pe, vertices, hvr_ctx);

    // Statically divide 2D grid into PARTITION_DIM x PARTITION_DIM partitions
    hvr_init(N, // # partitions
            update_metadata,
            might_interact,
            update_coupled_val,
            actor_to_partition,
            NULL, // start_time_step
            should_have_edge,
            should_terminate,
            60, // max_elapsed_seconds
            1, // max_graph_traverse_depth
            hvr_ctx);

    hvr_body(hvr_ctx);

    shmem_barrier_all();

    fprintf(stderr, "PE %d - (%u, %u) - iters=%d\n", pe,
            (unsigned)hvr_vertex_get(ACTOR_ID, vertices, hvr_ctx),
            (unsigned)hvr_vertex_get(POS, vertices, hvr_ctx),
            hvr_ctx->iter);
    assert((unsigned)hvr_vertex_get(ACTOR_ID, vertices, hvr_ctx) == pe);
    if (pe == 0) {
        assert((unsigned)hvr_vertex_get(POS, vertices, hvr_ctx) == N - 2);
    } else {
        assert((unsigned)hvr_vertex_get(POS, vertices, hvr_ctx) == N - 1);
    }
    printf("PE %d SUCCESS\n", pe);

    hvr_finalize(hvr_ctx);

    shmem_finalize();

    return 0;
}
