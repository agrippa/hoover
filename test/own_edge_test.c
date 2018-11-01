/* For license: see LICENSE.txt file at top-level */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <shmem.h>
#include <math.h>

#include <hoover.h>

hvr_partition_t actor_to_partition(hvr_vertex_t *actor, hvr_ctx_t ctx) {
    return 0;
}

hvr_edge_type_t should_have_edge(hvr_vertex_t *a, hvr_vertex_t *b,
        hvr_ctx_t ctx) {
    return BIDIRECTIONAL;
}

void update_metadata(hvr_vertex_t *vertex, hvr_set_t *couple_with,
        hvr_ctx_t ctx) {
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
}

int should_terminate(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_vertex_t *local_coupled_metric, hvr_vertex_t *global_coupled_metric,
        hvr_set_t *coupled_pes, int n_coupled_pes) {
    return 0;
}

int main(int argc, char **argv) {
    hvr_ctx_t hvr_ctx;

    shmem_init();
    hvr_ctx_create(&hvr_ctx);

    int pe = shmem_my_pe();

    if (pe == 0) {
        hvr_vertex_t *vert = hvr_vertex_create(hvr_ctx);
        hvr_vertex_set(0, 0, vert, hvr_ctx);
        hvr_vertex_set(1, 1, vert, hvr_ctx);
        hvr_vertex_set(2, 2, vert, hvr_ctx);
    }

    hvr_init(1,
            update_metadata,
            might_interact,
            update_coupled_val,
            actor_to_partition,
            NULL, // start_time_step
            should_have_edge,
            should_terminate,
            5, // max_elapsed_seconds
            1, // max_graph_traverse_depth
            hvr_ctx);

    hvr_body(hvr_ctx);

    unsigned count = 0;
    hvr_vertex_iter_t iter;
    hvr_vertex_iter_init(&iter, hvr_ctx);
    for (hvr_vertex_t *curr = hvr_vertex_iter_next(&iter); curr;
            curr = hvr_vertex_iter_next(&iter)) {
        count++;
    }

    if (pe == 0) {
        assert(count == 1);
        hvr_vertex_iter_init(&iter, hvr_ctx);
        hvr_vertex_t *only = hvr_vertex_iter_next(&iter);

        hvr_vertex_id_t *neighbors;
        hvr_edge_type_t *directions;
        size_t n_neighbors;
        hvr_get_neighbors(only, &neighbors, &directions, &n_neighbors, hvr_ctx);

        assert(n_neighbors == 1);
        assert(neighbors[0] == only->id);

        free(neighbors); free(directions);
    } else {
        assert(count == 0);
    }
    printf("%d: Success!\n", shmem_my_pe());

    hvr_finalize(hvr_ctx);

    shmem_finalize();

    return 0;
}
