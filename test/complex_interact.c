/* For license: see LICENSE.txt file at top-level */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <shmem.h>
#include <math.h>

#include <hoover.h>

#define ACTOR 0
#define CREATED_NEXT 1

unsigned N = 2000;
int pe, npes;

hvr_partition_t actor_to_partition(const hvr_vertex_t *vertex, hvr_ctx_t ctx) {
    return hvr_vertex_get_uint64(ACTOR, vertex, ctx);
}

hvr_edge_type_t should_have_edge(const hvr_vertex_t *a, const hvr_vertex_t *b,
        hvr_ctx_t ctx) {
    uint64_t actor_a = hvr_vertex_get_uint64(ACTOR, a, ctx);
    uint64_t actor_b = hvr_vertex_get_uint64(ACTOR, b, ctx);

    int delta = abs(actor_a - actor_b);
    assert(delta == 1);
    return BIDIRECTIONAL;
}

void might_interact(const hvr_partition_t partition,
        hvr_partition_t *interacting_partitions,
        unsigned *n_interacting_partitions,
        unsigned interacting_partitions_capacity,
        hvr_ctx_t ctx) {
    if (partition > 0) {
        interacting_partitions[*n_interacting_partitions] = partition - 1;
        *n_interacting_partitions += 1;
    }
    if (partition < N - 1) {
        interacting_partitions[*n_interacting_partitions] = partition + 1;
        *n_interacting_partitions += 1;
    }
}

void update_vertex(hvr_vertex_t *vertex, hvr_set_t *couple_with,
        hvr_ctx_t ctx) {
    uint64_t actor = hvr_vertex_get_uint64(ACTOR, vertex, ctx);
    uint64_t created_next = hvr_vertex_get_uint64(CREATED_NEXT, vertex, ctx);

    if (actor < N - 2 && !created_next) {
        hvr_vertex_t **verts;
        hvr_edge_type_t *dirs;
        int n_neighbors = hvr_get_neighbors(vertex, &verts, &dirs, ctx);
        assert(n_neighbors >= 0 && n_neighbors <= 2);

        // If we have a +1 neighbor, then create a +2 neighbor
        int found_plus_one = 0;
        for (int i = 0; i < n_neighbors; i++) {
            hvr_vertex_t *neighbor = verts[i];
            uint64_t actor_neighbor = hvr_vertex_get_uint64(ACTOR, neighbor, ctx);
            if (actor_neighbor == actor + 1) {
                assert(found_plus_one == 0);
                found_plus_one = 1;
            }
        }

        if (found_plus_one) {
            hvr_vertex_set(CREATED_NEXT, 1, vertex, ctx);
            hvr_vertex_t *next = hvr_vertex_create(ctx);
            hvr_vertex_set_uint64(ACTOR, actor + 2, next, ctx);
            hvr_vertex_set_uint64(CREATED_NEXT, 0, next, ctx);
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

    hvr_vertex_t *init_vertex = hvr_vertex_create(hvr_ctx);
    hvr_vertex_set_uint64(ACTOR, pe, init_vertex, hvr_ctx);
    hvr_vertex_set_uint64(CREATED_NEXT, 0, init_vertex, hvr_ctx);

    hvr_init(N, // # partitions
            update_vertex,
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

    unsigned count_actors = 0;
    hvr_vertex_iter_t iter;
    hvr_vertex_iter_init(&iter, hvr_ctx);
    for (hvr_vertex_t *curr = hvr_vertex_iter_next(&iter); curr;
            curr = hvr_vertex_iter_next(&iter)) {
        count_actors++;
    }
    assert(count_actors == N / 2);

    hvr_finalize(hvr_ctx);
    if (pe == 0) {
        printf("SUCCESS\n");
    }

    shmem_finalize();

    return 0;
}
