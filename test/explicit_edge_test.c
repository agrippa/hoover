#include <shmem.h>
#include <stdio.h>
#include <hoover.h>

#define VERTEX_TYPE 0
#define VERTEX_ID 1
#define CREATED_LAYERED 2
#define LAYERED_VERTEX 3
#define SENT_MSG 4

#define BASE_GRAPH 0
#define LAYERED_GRAPH 1

static int pe, npes;

void start_time_step(hvr_vertex_iter_t *iter,
        hvr_set_t *couple_with, hvr_ctx_t ctx) {
}

static void update_vertex(hvr_vertex_t *vertex, hvr_set_t *couple_with,
        hvr_ctx_t ctx) {
    assert(hvr_vertex_get_uint64(VERTEX_ID, vertex, ctx) == pe);
    const int max_neighbors = (pe == 0 || pe == npes - 1) ? 1 : 2;

    if (hvr_vertex_get_uint64(VERTEX_TYPE, vertex, ctx) == BASE_GRAPH) {
        if (hvr_vertex_get_uint64(CREATED_LAYERED, vertex, ctx) == 0) {
            hvr_vertex_t *layered = hvr_vertex_create(ctx);
            hvr_vertex_set_uint64(VERTEX_TYPE,     LAYERED_GRAPH, layered, ctx);
            hvr_vertex_set_uint64(VERTEX_ID,       pe,         layered, ctx);
            hvr_vertex_set_uint64(CREATED_LAYERED, 0,          layered, ctx);
            hvr_vertex_set_uint64(LAYERED_VERTEX,  HVR_INVALID_VERTEX_ID,
                    layered, ctx);

            hvr_vertex_set_uint64(CREATED_LAYERED, 1, vertex, ctx);
            hvr_vertex_set_uint64(LAYERED_VERTEX, layered->id, vertex, ctx);
        }

        hvr_neighbors_t neighbors;
        hvr_get_neighbors(vertex, &neighbors, ctx);

        hvr_vertex_t *neighbor;
        hvr_edge_type_t dir;
        unsigned n_neighbors = 0;
        unsigned n_neighbors_with_layered = 0;
        while (hvr_neighbors_next(&neighbors, &neighbor, &dir)) {
            if (hvr_vertex_get_uint64(CREATED_LAYERED, neighbor, ctx)) {
                n_neighbors_with_layered++;
            }
            n_neighbors++;
        }
        assert(n_neighbors <= max_neighbors);

        if (n_neighbors == max_neighbors &&
                n_neighbors_with_layered == n_neighbors &&
                !hvr_vertex_get_uint64(SENT_MSG, vertex, ctx)) {
            hvr_vertex_t msg;
            hvr_vertex_init(&msg, HVR_INVALID_VERTEX_ID, ctx->iter);

            hvr_get_neighbors(vertex, &neighbors, ctx);

            hvr_vertex_set_uint64(0, n_neighbors, &msg, ctx);
            unsigned i = 0;
            while (hvr_neighbors_next(&neighbors, &neighbor, &dir)) {
                hvr_vertex_set_uint64(1 + i,
                        hvr_vertex_get_uint64(LAYERED_VERTEX, neighbor, ctx),
                        &msg, ctx);
                i++;
            }

            // fprintf(stderr, "PE %d sending message to layered vertex %llu "
            //         "containing %llu neighbors\n", pe,
            //         hvr_vertex_get_uint64(LAYERED_VERTEX, vertex, ctx),
            //         hvr_vertex_get_uint64(0, &msg, ctx));

            hvr_send_msg(hvr_vertex_get_uint64(LAYERED_VERTEX, vertex, ctx),
                    &msg, ctx);

            hvr_vertex_set_uint64(SENT_MSG, 1, vertex, ctx);
        }
    } else {
        assert(hvr_vertex_get_uint64(VERTEX_TYPE, vertex, ctx) ==
                LAYERED_GRAPH);

        // // fprintf(stderr, "PE %d running layered vertex %llu\n", pe,
        // //         hvr_vertex_get_id(vertex));

        hvr_vertex_t msg;
        while (hvr_poll_msg(vertex, &msg, ctx)) {
            int n_neighbors = hvr_vertex_get_uint64(0, &msg, ctx);
            // fprintf(stderr, "PE %d receiving msg at layered vertex containing "
            //         "%d neighbors\n", pe, n_neighbors);
            for (int i = 0; i < n_neighbors; i++) {
                hvr_create_edge_with_vertex_id(vertex,
                        hvr_vertex_get_uint64(1 + i, &msg, ctx), BIDIRECTIONAL,
                        ctx);
            }
        }
    }
}

static void might_interact(const hvr_partition_t partition,
        hvr_partition_t *interacting_partitions,
        unsigned *n_interacting_partitions,
        unsigned interacting_partitions_capacity,
        hvr_ctx_t ctx) {
    assert(partition != HVR_INVALID_PARTITION);
    assert(partition == 0);
    interacting_partitions[0] = 0;
    *n_interacting_partitions = 1;
}

void update_coupled_val(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_vertex_t *out_coupled_metric) {
    hvr_vertex_set_uint64(0, 0, out_coupled_metric, ctx);
}

hvr_partition_t actor_to_partition(const hvr_vertex_t *actor, hvr_ctx_t ctx) {
    if (hvr_vertex_get_uint64(VERTEX_TYPE, actor, ctx) == BASE_GRAPH) {
        return 0;
    } else {
        return HVR_INVALID_PARTITION;
    }
}

hvr_edge_type_t should_have_edge(const hvr_vertex_t *a, const hvr_vertex_t *b,
        hvr_ctx_t ctx) {
    assert(hvr_vertex_get_uint64(VERTEX_TYPE, a, ctx) == BASE_GRAPH);
    assert(hvr_vertex_get_uint64(VERTEX_TYPE, b, ctx) == BASE_GRAPH);

    if (abs(hvr_vertex_get_uint64(VERTEX_ID, a, ctx) -
                hvr_vertex_get_uint64(VERTEX_ID, b, ctx)) == 1) {
        return BIDIRECTIONAL;
    } else {
        return NO_EDGE;
    }
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
    shmem_init();
    pe = shmem_my_pe();
    npes = shmem_n_pes();

    hvr_ctx_t ctx;
    hvr_ctx_create(&ctx);

    hvr_vertex_t *vert = hvr_vertex_create(ctx);
    hvr_vertex_set_uint64(VERTEX_TYPE,     BASE_GRAPH, vert, ctx);
    hvr_vertex_set_uint64(VERTEX_ID,       pe,         vert, ctx);
    hvr_vertex_set_uint64(CREATED_LAYERED, 0,          vert, ctx);
    hvr_vertex_set_uint64(LAYERED_VERTEX,  HVR_INVALID_VERTEX_ID, vert, ctx);
    hvr_vertex_set_uint64(SENT_MSG,        0,          vert, ctx);

    hvr_init(1, // # partitions
            update_vertex,
            might_interact,
            update_coupled_val,
            actor_to_partition,
            start_time_step,
            should_have_edge,
            should_terminate,
            30, // max_elapsed_seconds
            1, // max_graph_traverse_depth
            ctx);

    hvr_body(ctx);
    assert(ctx->pe == pe);

    hvr_vertex_iter_t iter;
    hvr_vertex_iter_init(&iter, ctx);
    int count = 0;
    int count_base = 0;
    int count_layered = 0;
    for (hvr_vertex_t *vert = hvr_vertex_iter_next(&iter); vert;
            vert = hvr_vertex_iter_next(&iter)) {
        hvr_vertex_t *neighbor;
        hvr_edge_type_t dir;
        hvr_neighbors_t neighbors;
        hvr_get_neighbors(vert, &neighbors, ctx);

        unsigned n_neighbors = 0;
        const int max_neighbors = (pe == 0 || pe == npes - 1) ? 1 : 2;

        count++;
        if (hvr_vertex_get_uint64(VERTEX_TYPE, vert, ctx) == BASE_GRAPH) {
            unsigned n_neighbors = 0;
            while (hvr_neighbors_next(&neighbors, &neighbor, &dir)) {
                assert(hvr_vertex_get_uint64(VERTEX_TYPE, neighbor, ctx) ==
                        BASE_GRAPH);
                assert(abs(hvr_vertex_get_uint64(VERTEX_ID, neighbor, ctx) -
                            hvr_vertex_get_uint64(VERTEX_ID, vert, ctx)) == 1);
                n_neighbors++;
            }
            assert(n_neighbors == max_neighbors);

            count_base++;
        } else if (hvr_vertex_get_uint64(VERTEX_TYPE, vert, ctx) ==
                LAYERED_GRAPH) {
            unsigned n_neighbors = 0;
            while (hvr_neighbors_next(&neighbors, &neighbor, &dir)) {
                assert(hvr_vertex_get_uint64(VERTEX_TYPE, neighbor, ctx) ==
                        LAYERED_GRAPH);
                assert(abs(hvr_vertex_get_uint64(VERTEX_ID, neighbor, ctx) -
                            hvr_vertex_get_uint64(VERTEX_ID, vert, ctx)) == 1);
                n_neighbors++;
            }
            assert(n_neighbors == max_neighbors);

            count_layered++;
        } else {
            abort();
        }
    }
    assert(count == 2);
    assert(count_base == 1);
    assert(count_layered == 1);

    hvr_finalize(ctx);

    shmem_finalize();
    
    if (pe == 0) {
        printf("Success\n");
    }

    return 0;
}
