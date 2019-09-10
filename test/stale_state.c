/* For license: see LICENSE.txt file at top-level */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <shmem.h>
#include <math.h>

#include <hoover.h>

int pe, npes;

void start_time_step(hvr_vertex_iter_t *iter,
        hvr_set_t *couple_with, hvr_ctx_t ctx) {

    for (hvr_vertex_t *vertex = hvr_vertex_iter_next(iter); vertex;
            vertex = hvr_vertex_iter_next(iter)) {
        hvr_vertex_trigger_update(vertex, ctx);
    }
}

hvr_partition_t actor_to_partition(const hvr_vertex_t *vertex, hvr_ctx_t ctx) {
    return (hvr_partition_t)hvr_vertex_get(0, vertex, ctx);
}

hvr_edge_type_t should_have_edge(const hvr_vertex_t *a, const hvr_vertex_t *b,
        hvr_ctx_t ctx) {
    return BIDIRECTIONAL;
}

void update_metadata(hvr_vertex_t *vertex, hvr_set_t *couple_with,
        hvr_ctx_t ctx) {
    if (pe == 1 && ctx->iter > 1000) {
        hvr_vertex_set(0, 2, vertex, ctx);
    }
}

void might_interact(const hvr_partition_t partition,
        hvr_partition_t *interacting_partitions,
        unsigned *n_interacting_partitions,
        unsigned interacting_partitions_capacity,
        hvr_ctx_t ctx) {
    if (partition == 0) {
        interacting_partitions[0] = 1;
    } else if (partition == 1) {
        interacting_partitions[0] = 0;
    } else if (partition == 2) {
        interacting_partitions[0] = 1;
    } else {
        abort();
    }
        *n_interacting_partitions = 1;
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
    hvr_vertex_set(0, pe, &vertices[0], hvr_ctx);

    hvr_init(3, // # partitions
            update_metadata,
            might_interact,
            update_coupled_val,
            actor_to_partition,
            start_time_step,
            should_have_edge,
            should_terminate,
            30, // max_elapsed_seconds
            1, // max_graph_traverse_depth
            hvr_ctx);

    hvr_body(hvr_ctx);

    if (pe == 1) {
        fprintf(stderr, "PE 1 ran for %u iterations\n", hvr_ctx->iter);
    }

    shmem_barrier_all();

    if (pe == 0) {
        hvr_vertex_t **verts;
        hvr_edge_type_t *edges;

        hvr_vertex_t *neighbor;
        hvr_edge_type_t dir;
        hvr_neighbors_t neighbors;
        hvr_get_neighbors(&vertices[0], &neighbors, hvr_ctx);

        unsigned n_neighbors = 0;
        while (hvr_neighbors_next(&neighbors, &neighbor, &dir)) {
            n_neighbors++;
        }

        assert(n_neighbors == 0);
        hvr_release_neighbors(&neighbors, hvr_ctx);
        printf("SUCCESS\n");
    }

    hvr_finalize(hvr_ctx);

    shmem_finalize();

    return 0;
}
