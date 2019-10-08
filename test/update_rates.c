#include <shmem.h>
#include <stdio.h>
#include <hoover.h>
#include "mmio.h"

/*
 * Tested graphs:
 *
 *  - https://sparse.tamu.edu/LAW/in-2004
 *  - https://www.cise.ufl.edu/research/sparse/matrices/SNAP/soc-LiveJournal1.html
 */

static int pe, npes;

static hvr_vertex_id_t *my_edges;
static uint64_t n_my_edges;
static uint64_t edges_so_far = 0;
static uint64_t batch_size = 16;

void start_time_step(hvr_vertex_iter_t *iter,
        hvr_set_t *couple_with, hvr_ctx_t ctx) {
    uint64_t limit = edges_so_far + batch_size;
    if (limit > n_my_edges) {
        limit = n_my_edges;
    }

    while (edges_so_far < limit) {
        hvr_vertex_id_t local_id = my_edges[2 * edges_so_far];
        hvr_vertex_id_t other_id = my_edges[2 * edges_so_far + 1];

        hvr_vertex_t *local = hvr_get_vertex(local_id, ctx);
        assert(local);

        hvr_create_edge_with_vertex_id(local, other_id, BIDIRECTIONAL, ctx);

        edges_so_far++;
    }
}

static void might_interact(const hvr_partition_t partition,
        hvr_partition_t *interacting_partitions,
        unsigned *n_interacting_partitions,
        unsigned interacting_partitions_capacity,
        hvr_ctx_t ctx) {
    abort(); // Should never be called
}

void update_coupled_val(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_vertex_t *out_coupled_metric) {
    hvr_vertex_set_uint64(0, 0, out_coupled_metric, ctx);
}

hvr_partition_t actor_to_partition(const hvr_vertex_t *actor, hvr_ctx_t ctx) {
    return HVR_INVALID_PARTITION;
}

hvr_edge_type_t should_have_edge(const hvr_vertex_t *a, const hvr_vertex_t *b,
        hvr_ctx_t ctx) {
    abort(); // Should never be called
}

int should_terminate(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_vertex_t *local_coupled_metric,
        hvr_vertex_t *all_coupled_metrics,
        hvr_vertex_t *global_coupled_metric,
        hvr_set_t *coupled_pes,
        int n_coupled_pes, int *updates_on_this_iter,
        hvr_set_t *terminated_coupled_pes,
        uint64_t n_msgs_this_iter) {
    return (edges_so_far == n_my_edges && n_msgs_this_iter == 0);
}

int main(int argc, char **argv) {
    int ret_code;

    if (argc != 3) {
        fprintf(stderr, "usage: %s <mat-file> <batch-size>\n", argv[0]);
        return 1;
    }

    shmem_init();
    pe = shmem_my_pe();
    npes = shmem_n_pes();

    hvr_ctx_t ctx;
    hvr_ctx_create(&ctx);

    const char *mat_filename = argv[1];
    batch_size = atoi(argv[2]);

    FILE *fp = fopen(mat_filename, "r");
    assert(fp);

    int M, N, nz;
    size_t n;
    n = fread(&M, sizeof(M), 1, fp);
    assert(n == 1);
    n = fread(&N, sizeof(N), 1, fp);
    assert(n == 1);
    n = fread(&nz, sizeof(nz), 1, fp);
    assert(n == 1);

    assert(M == N);

    if (pe == 0) {
        printf("Matrix %s is %d x %d with %d non-zeroes\n", mat_filename, M, N,
                nz);
    }

    int *I = (int *)malloc(nz * sizeof(*I));
    assert(I);
    int *J = (int *)malloc(nz * sizeof(*J));
    assert(J);

    n = fread(I, sizeof(int), nz, fp);
    assert(n == nz);
    n = fread(J, sizeof(int), nz, fp);
    assert(n == nz);

    fclose(fp);

    size_t vertices_per_pe = (M + npes - 1) / npes;
    size_t my_vertices_start = pe * vertices_per_pe;
    size_t my_vertices_end = (pe + 1) * vertices_per_pe;
    if (my_vertices_end > M) my_vertices_end = M;

    for (int i = my_vertices_start; i < my_vertices_end; i++) {
        hvr_vertex_t *vert = hvr_vertex_create(ctx);
    }

    size_t count_edges_to_insert = 0;
    for (int i = 0; i < nz; i++) {
        if (I[i] >= my_vertices_start && I[i] < my_vertices_end) {
            count_edges_to_insert++;
        }
    }

    my_edges = (hvr_vertex_id_t *)malloc(
            2 * count_edges_to_insert * sizeof(*my_edges));
    assert(my_edges);

    n_my_edges = 0;
    for (int i = 0; i < nz; i++) {
        assert(I[i] >= 0 && J[i] >= 0);

        if (I[i] >= my_vertices_start && I[i] < my_vertices_end) {
            assert(I[i] < M);
            size_t I_pe = I[i] / vertices_per_pe;
            assert(I_pe == pe);
            size_t I_offset = I[i] % vertices_per_pe;

            assert(J[i] < M);
            size_t J_pe = J[i] / vertices_per_pe;
            assert(J_pe < npes);
            size_t J_offset = J[i] % vertices_per_pe;

            my_edges[2 * n_my_edges] = construct_vertex_id(I_pe, I_offset);
            my_edges[2 * n_my_edges + 1] = construct_vertex_id(J_pe, J_offset);
            n_my_edges++;
        }
    }

    hvr_init(1, // # partitions
            NULL, // update_vertex
            might_interact,
            update_coupled_val,
            actor_to_partition,
            start_time_step, // start_time_step
            should_have_edge,
            should_terminate,
            30, // max_elapsed_seconds
            1, // max_graph_traverse_depth
            ctx);

    unsigned long long start_time = hvr_current_time_us();
    hvr_body(ctx);
    unsigned long long elapsed_time = hvr_current_time_us() - start_time;

    fprintf(stderr, "PE %d took %f ms to insert %lu edges with batch size %lu ( %f "
            "edges per s )\n",
            pe, (double)elapsed_time / 1000.0, n_my_edges, batch_size,
            (double)n_my_edges / ((double)elapsed_time / 1000000.0));

    hvr_finalize(ctx);

    shmem_finalize();
    
    if (pe == 0) {
        printf("Success\n");
    }

    return 0;
}
