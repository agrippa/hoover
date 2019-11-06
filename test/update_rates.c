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
    return (edges_so_far == n_my_edges); // && n_msgs_this_iter == 0);
}

int main(int argc, char **argv) {
    int ret_code;

    if (argc != 4) {
        fprintf(stderr, "usage: %s <mat-file> <batch-size> "
                "<max-elapsed-seconds>\n", argv[0]);
        return 1;
    }

    shmem_init();
    pe = shmem_my_pe();
    npes = shmem_n_pes();

    hvr_ctx_t ctx;
    hvr_ctx_create(&ctx);

    const char *mat_filename = argv[1];
    batch_size = atoi(argv[2]);
    int max_elapsed_seconds = atoi(argv[3]);

    char filename[2048];
    sprintf(filename, "%s.npes=%d.pe=%d_0", mat_filename, npes, pe);
    FILE *fp0 = fopen(filename, "r");
    assert(fp0);

    sprintf(filename, "%s.npes=%d.pe=%d_1", mat_filename, npes, pe);
    FILE *fp1 = fopen(filename, "r");
    assert(fp1);

    int64_t M0, N0, nz0, partition_nz0;
    int64_t M1, N1, nz1, partition_nz1;
    int64_t M, N, nz, partition_nz;
    size_t n;

    n = fread(&M0, sizeof(M0), 1, fp0);
    assert(n == 1);
    n = fread(&N0, sizeof(N0), 1, fp0);
    assert(n == 1);
    n = fread(&nz0, sizeof(nz0), 1, fp0);
    assert(n == 1);
    n = fread(&partition_nz0, sizeof(partition_nz0), 1, fp0);
    assert(n == 1);
    assert(M0 == N0);

    n = fread(&M1, sizeof(M1), 1, fp1);
    assert(n == 1);
    n = fread(&N1, sizeof(N1), 1, fp1);
    assert(n == 1);
    n = fread(&nz1, sizeof(nz1), 1, fp1);
    assert(n == 1);
    n = fread(&partition_nz1, sizeof(partition_nz1), 1, fp1);
    assert(n == 1);
    assert(M1 == N1);

    assert(M0 == M1);
    assert(N0 == N1);
    assert(nz0 == nz1);
    assert(partition_nz0 == partition_nz1);

    M = M0;
    N = N0;
    nz = nz0;
    partition_nz = partition_nz0;

    if (pe == 0) {
        fprintf(stderr, "Matrix %s is %ld x %ld with %ld non-zeroes\n",
                mat_filename, M, N, nz);
    }

    int64_t vertices_per_pe = (M + npes - 1) / npes;
    int64_t my_vertices_start = pe * vertices_per_pe;
    int64_t my_vertices_end = (pe + 1) * vertices_per_pe;
    if (my_vertices_end > M) my_vertices_end = M;

    for (int i = my_vertices_start; i < my_vertices_end; i++) {
        hvr_vertex_t *vert = hvr_vertex_create(ctx);
    }

    int64_t *edges0 = (int64_t *)malloc(partition_nz * sizeof(*edges0));
    assert(edges0);
    int64_t *edges1 = (int64_t *)malloc(partition_nz * sizeof(*edges1));
    assert(edges1);
    my_edges = (hvr_vertex_id_t *)malloc(2 * partition_nz * sizeof(*my_edges));
    assert(my_edges);

    n = fread(edges0, sizeof(*edges0), partition_nz, fp0);
    assert(n == partition_nz);
    n = fread(edges1, sizeof(*edges1), partition_nz, fp1);
    assert(n == partition_nz);
    fclose(fp0);
    fclose(fp1);

    for (int64_t i = 0; i < partition_nz; i++) {
        assert(edges0[i] < M);
        int64_t I_pe = edges0[i] / vertices_per_pe;
        assert(I_pe == pe);
        int64_t I_offset = edges0[i] % vertices_per_pe;

        assert(edges1[i] < M);
        int64_t J_pe = edges1[i] / vertices_per_pe;
        assert(J_pe < npes);
        int64_t J_offset = edges1[i] % vertices_per_pe;

        my_edges[2 * i] = construct_vertex_id(I_pe, I_offset);
        my_edges[2 * i + 1] = construct_vertex_id(J_pe, J_offset);
    }
    n_my_edges = partition_nz;

    free(edges0);
    free(edges1);

    if (pe == 0) {
        fprintf(stderr, "Done loading graph.\n");
    }

    hvr_init(1, // # partitions
            NULL, // update_vertex
            might_interact,
            update_coupled_val,
            actor_to_partition,
            start_time_step, // start_time_step
            should_have_edge,
            should_terminate,
            max_elapsed_seconds,
            1, // max_graph_traverse_depth
            0, // send_neighbor_updates_for_explicit_subs
            ctx);

    if (pe == 0) {
        fprintf(stderr, "HOOVER runtime initialized.\n");
    }

    unsigned long long start_time = hvr_current_time_us();
    hvr_body(ctx);
    unsigned long long elapsed_time = hvr_current_time_us() - start_time;

    fprintf(stderr, "PE %d took %f ms to insert %lu / %lu (%f%%) edges with "
            "batch size %lu ( %f "
            "edges per s )\n",
            pe, (double)elapsed_time / 1000.0, edges_so_far, n_my_edges,
            100.0 * (double)edges_so_far / (double)n_my_edges,
            batch_size,
            (double)n_my_edges / ((double)elapsed_time / 1000000.0));

    hvr_finalize(ctx);

    shmem_finalize();
    
    if (pe == 0) {
        printf("Success\n");
    }

    return 0;
}
