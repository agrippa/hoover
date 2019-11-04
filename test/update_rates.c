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

    FILE *fp = fopen(mat_filename, "r");
    assert(fp);

    int64_t M, N, nz;
    size_t n;
    n = fread(&M, sizeof(M), 1, fp);
    assert(n == 1);
    n = fread(&N, sizeof(N), 1, fp);
    assert(n == 1);
    n = fread(&nz, sizeof(nz), 1, fp);
    assert(n == 1);

    assert(M == N);

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

    const int tile_size = 64 * 1024 * 1024;
    int64_t *I = (int64_t *)malloc(tile_size * sizeof(*I));
    assert(I);
    int64_t *J = (int64_t *)malloc(tile_size * sizeof(*J));
    assert(J);

    if (pe == 0) {
        fprintf(stderr, "Counting edges...\n");
    }

    long I_offset = ftell(fp);

    int64_t count_edges_to_insert = 0;
    for (int64_t i = 0; i < nz; i += tile_size) {
        int64_t to_read = nz - i;
        if (to_read > tile_size) to_read = tile_size;

        n = fread(I, sizeof(*I), to_read, fp);
        assert(n == to_read);

        for (int64_t j = 0; j < to_read; j++) {
            if (I[j] >= my_vertices_start && I[j] < my_vertices_end) {
                count_edges_to_insert++;
            }
        }
    }

    if (pe == 0) {
        fprintf(stderr, "Loading edges...\n");
    }

    long J_offset = ftell(fp);

    my_edges = (hvr_vertex_id_t *)malloc(
            2 * count_edges_to_insert * sizeof(*my_edges));
    assert(my_edges);

#if 0
    n_my_edges = 0;
    for (int64_t i = 0; i < nz; i += tile_size) {
        int64_t to_read = nz - i;
        if (to_read > tile_size) to_read = tile_size;

        fseek(fp, I_offset + i * sizeof(*I), SEEK_SET);
        n = fread(I, sizeof(int64_t), to_read, fp);
        assert(n == to_read);

        fseek(fp, J_offset + i * sizeof(*J), SEEK_SET);
        n = fread(J, sizeof(int64_t), to_read, fp);
        assert(n == to_read);

        for (int64_t j = 0; j < to_read; j++) {
            if (I[j] >= my_vertices_start && I[j] < my_vertices_end) {
                assert(I[j] < M);
                int64_t I_pe = I[j] / vertices_per_pe;
                assert(I_pe == pe);
                int64_t I_offset = I[j] % vertices_per_pe;

                assert(J[j] < M);
                int64_t J_pe = J[j] / vertices_per_pe;
                assert(J_pe < npes);
                int64_t J_offset = J[j] % vertices_per_pe;

                my_edges[2 * n_my_edges] = construct_vertex_id(I_pe, I_offset);
                my_edges[2 * n_my_edges + 1] = construct_vertex_id(J_pe,
                        J_offset);
                n_my_edges++;
            }
        }
    }
    assert(n_my_edges == count_edges_to_insert);
#endif

    free(I);
    free(J);

    fclose(fp);

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

#if 0
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
#endif

    shmem_finalize();
    
    if (pe == 0) {
        printf("Success\n");
    }

    return 0;
}
