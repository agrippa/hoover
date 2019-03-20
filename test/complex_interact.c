/* For license: see LICENSE.txt file at top-level */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <shmem.h>
#include <math.h>

#include <hoover.h>

unsigned N = 2000;
int pe, npes;

hvr_partition_t actor_to_partition(hvr_vertex_t *vertex, hvr_ctx_t ctx) {
    hvr_partition_t actor = (hvr_partition_t)hvr_vertex_get(0, vertex, ctx);
    if (actor < N) {
        return actor;
    } else {
        hvr_partition_t part = N + hvr_vertex_get(1, vertex, ctx);
        if (part >= 2 * N) part = 2 * N - 1;
        return part;
    }
}

hvr_edge_type_t should_have_edge(hvr_vertex_t *a, hvr_vertex_t *b,
        hvr_ctx_t ctx) {
    if (hvr_vertex_get_owning_pe(a) != hvr_vertex_get_owning_pe(b)) {
        // fprintf(stderr, "PE %d BIDIRECTIONAL (%f %f) (%f %f)\n", ctx->pe,
        //         hvr_vertex_get(0, a, ctx), hvr_vertex_get(1, a, ctx),
        //         hvr_vertex_get(0, b, ctx), hvr_vertex_get(1, b, ctx));
        return BIDIRECTIONAL;
    } else {
        // fprintf(stderr, "PE %d NO_EDGE (%f %f) (%f %f)\n", ctx->pe,
        //         hvr_vertex_get(0, a, ctx), hvr_vertex_get(1, a, ctx),
        //         hvr_vertex_get(0, b, ctx), hvr_vertex_get(1, b, ctx));
        return NO_EDGE;
    }
}

void update_metadata(hvr_vertex_t *vertex, hvr_set_t *couple_with,
        hvr_ctx_t ctx) {
    hvr_partition_t actor = (hvr_partition_t)hvr_vertex_get(0, vertex, ctx);
    if (actor == N) {
        assert(hvr_vertex_get_owning_pe(vertex) == 1);

        hvr_vertex_t **verts;
        hvr_edge_type_t *dirs;
        int n_neighbors = hvr_get_neighbors(vertex, &verts, &dirs, ctx);
        assert(n_neighbors >= 0 && n_neighbors <= N);

        fprintf(stderr, "PE %d iter %d, %u neighbors for actor %d\n",
                ctx->pe, ctx->iter, n_neighbors, N);

        for (int i = 0; i < n_neighbors; i++) {
            hvr_vertex_t *neighbor = verts[i];
            assert((unsigned)hvr_vertex_get(0, neighbor, ctx) < n_neighbors);
        }

        unsigned curr_val = (unsigned)hvr_vertex_get(1, vertex, ctx);
        assert(curr_val == n_neighbors || curr_val == n_neighbors - 1);

        if (curr_val < n_neighbors) {
            fprintf(stderr, "Transitioning from %u to %u\n", curr_val,
                    n_neighbors);
        }

        hvr_vertex_set(1, (double)n_neighbors, vertex, ctx);
    }
}

void might_interact(const hvr_partition_t partition,
        hvr_partition_t *interacting_partitions,
        unsigned *n_interacting_partitions,
        unsigned interacting_partitions_capacity,
        hvr_ctx_t ctx) {
    if (partition < N) {
        assert(partition >= 0);
        // For partitions on PE 0
        *n_interacting_partitions = partition + 1;
        assert(*n_interacting_partitions <= interacting_partitions_capacity);
        for (int i = 0; i < *n_interacting_partitions; i++) {
            interacting_partitions[i] = N + i;
        }
    } else {
        // Partition on PE 1
        assert(partition >= N && partition < 2 * N);
        *n_interacting_partitions = partition - N + 1;
        assert(*n_interacting_partitions <= interacting_partitions_capacity);
        for (int i = 0; i < *n_interacting_partitions; i++) {
            interacting_partitions[i] = i;
        }
    }
}

void update_coupled_val(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_vertex_t *out_coupled_metric) {
    hvr_vertex_set(0, (double)0, out_coupled_metric, ctx);
    hvr_vertex_set(1, (double)1, out_coupled_metric, ctx);
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

    hvr_vertex_t *vertices = NULL;
    if (pe == 0) {
        vertices = hvr_vertex_create_n(N, hvr_ctx);
        for (unsigned i = 0; i < N; i++) {
            hvr_vertex_set(0, i, &vertices[i], hvr_ctx);
            hvr_vertex_set(1, 0, &vertices[i], hvr_ctx);
        }
    } else {
        vertices = hvr_vertex_create_n(1, hvr_ctx);
        hvr_vertex_set(0, (double)N, vertices, hvr_ctx);
        hvr_vertex_set(1, (double)0, vertices, hvr_ctx);
    }

    // Statically divide 2D grid into PARTITION_DIM x PARTITION_DIM partitions
    hvr_init(2 * N, // # partitions
            update_metadata,
            might_interact,
            update_coupled_val,
            actor_to_partition,
            NULL, // start_time_step
            should_have_edge,
            should_terminate,
            30, // max_elapsed_seconds
            1, // max_graph_traverse_depth
            hvr_ctx);

    hvr_body(hvr_ctx);

    shmem_barrier_all();

    if (pe == 1) {
        assert((unsigned)hvr_vertex_get(0, vertices, hvr_ctx) == N);
        if ((unsigned)hvr_vertex_get(1, vertices, hvr_ctx) != N) {
            fprintf(stderr, "ERROR: Expected %u, got %u\n", N,
                    (unsigned)hvr_vertex_get(1, vertices, hvr_ctx));
        } else {
            printf("SUCCESS\n");
        }
    }

    hvr_finalize(hvr_ctx);

    shmem_finalize();

    return 0;
}
