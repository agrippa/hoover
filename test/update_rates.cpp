#include <shmem.h>
#include <stdio.h>
#include <hoover.h>

extern "C" {
#include "mmio.h"
}

#define SKIP_COUPLING

/*
 * Tested graphs:
 *
 *  - https://sparse.tamu.edu/LAW/in-2004
 *  - https://www.cise.ufl.edu/research/sparse/matrices/SNAP/soc-LiveJournal1.html
 */

static int pe, npes;

static int64_t *edges_from_file_0;
static int64_t *edges_from_file_1;

static hvr_vertex_id_t *edges_0;
static hvr_vertex_id_t *edges_1;

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
        hvr_vertex_id_t local_id = edges_0[edges_so_far];
        hvr_vertex_id_t other_id = edges_1[edges_so_far];

        hvr_vertex_t *local = hvr_get_vertex(local_id, ctx);
        assert(local);

        hvr_create_edge_with_vertex_id(local, other_id, BIDIRECTIONAL, ctx);

        edges_so_far++;
    }

#ifndef SKIP_COUPLING
    if (edges_so_far == n_my_edges) {
        for (int p = 0; p < npes; p++) {
            hvr_set_insert(p, couple_with);
        }
    }
#endif
}

static void might_interact(const hvr_partition_t partition,
        hvr_partition_t *interacting_partitions,
        unsigned *n_interacting_partitions,
        unsigned interacting_partitions_capacity,
        hvr_ctx_t ctx) {
    abort(); // Should never be called
}

void update_coupled_val(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_vertex_t *out_coupled_metric, uint64_t n_msgs_recvd_this_iter,
        uint64_t n_msgs_sent_this_iter, uint64_t n_msgs_recvd_total,
        uint64_t n_msgs_sent_total) {
    hvr_vertex_set_uint64(0, n_my_edges - edges_so_far, out_coupled_metric, ctx);
    hvr_vertex_set_int64(1,
            (int64_t)n_msgs_sent_total - (int64_t)n_msgs_recvd_total,
            out_coupled_metric, ctx);
}

hvr_partition_t actor_to_partition(const hvr_vertex_t *actor, hvr_ctx_t ctx) {
    return HVR_INVALID_PARTITION;
}

hvr_edge_type_t should_have_edge(const hvr_vertex_t *a, const hvr_vertex_t *b,
        hvr_ctx_t ctx) {
    abort(); // Should never be called
}

int should_terminate(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_vertex_t *local_coupled_metric, // My metric
        hvr_vertex_t *all_coupled_metrics, // Array of each PE's metric
        hvr_set_t *coupled_pes, int n_coupled_pes,
        int *updates_on_this_iter,
        hvr_set_t *terminated_coupled_pes,
        uint64_t n_msgs_recvd_this_iter,
        uint64_t n_msgs_sent_this_iter,
        uint64_t n_msgs_recvd_total,
        uint64_t n_msgs_sent_total) {
#ifndef SKIP_COUPLING
    if (n_coupled_pes == npes) {
        uint64_t sum_remaining_edges = 0;
        int64_t sum_pending_msgs = 0;
        for (int p = 0; p < npes; p++) {
            sum_remaining_edges += hvr_vertex_get_uint64(0,
                    &all_coupled_metrics[p], ctx);
            sum_pending_msgs += hvr_vertex_get_int64(1,
                    &all_coupled_metrics[p], ctx);
        }

        assert(sum_pending_msgs >= 0);
        return sum_remaining_edges == 0 && sum_pending_msgs == 0;
    } else {
        return 0;
    }
#else
    return (edges_so_far == n_my_edges);
#endif
}

int main(int argc, char **argv) {
    assert(sizeof(hvr_vertex_id_t) == sizeof(int64_t));
    assert(HVR_MAX_VECTOR_SIZE == 2);

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
    FILE *fp0 = fopen(filename, "rb");
    assert(fp0);

    sprintf(filename, "%s.npes=%d.pe=%d_1", mat_filename, npes, pe);
    FILE *fp1 = fopen(filename, "rb");
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
    long offset = ftell(fp0);
    n = fread(&partition_nz0, sizeof(partition_nz0), 1, fp0);
    assert(n == 1);
    assert(M0 == N0);

    if (pe == 0) {
        printf("M=%ld N=%ld nz=%ld partition nz=%ld offset=%lu file=%s\n", M0,
                N0, nz0, partition_nz0, offset, filename);
    }

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
        fprintf(stderr, "%d PEs\n", npes);
    }

    int64_t vertices_per_pe = (M + npes - 1) / npes;
    int64_t my_vertices_start = pe * vertices_per_pe;
    int64_t my_vertices_end = (pe + 1) * vertices_per_pe;
    if (my_vertices_end > M) my_vertices_end = M;

    for (int i = my_vertices_start; i < my_vertices_end; i++) {
        hvr_vertex_t *vert = hvr_vertex_create(ctx);
    }

    edges_from_file_0 = (int64_t *)malloc(
            partition_nz * sizeof(*edges_from_file_0));
    assert(edges_from_file_0);
    edges_from_file_1 = (int64_t *)malloc(
            partition_nz * sizeof(*edges_from_file_1));
    assert(edges_from_file_1);

    edges_0 = (hvr_vertex_id_t *)edges_from_file_0;
    edges_1 = (hvr_vertex_id_t *)edges_from_file_1;

    n = fread(edges_from_file_0, sizeof(*edges_from_file_0), partition_nz, fp0);
    assert(n == partition_nz);
    n = fread(edges_from_file_1, sizeof(*edges_from_file_1), partition_nz, fp1);
    assert(n == partition_nz);
    fclose(fp0);
    fclose(fp1);

    for (int64_t i = 0; i < partition_nz; i++) {
        assert(edges_from_file_0[i] < M);
        int64_t I_pe = edges_from_file_0[i] / vertices_per_pe;
        assert(I_pe == pe);
        int64_t I_offset = edges_from_file_0[i] % vertices_per_pe;

        assert(edges_from_file_1[i] < M);
        int64_t J_pe = edges_from_file_1[i] / vertices_per_pe;
        assert(J_pe < npes);
        int64_t J_offset = edges_from_file_1[i] % vertices_per_pe;

        // Same memory as edges_from_file_0[i]
        edges_0[i] = construct_vertex_id(I_pe, I_offset);
        // Same memory as edges_from_file_1[i]
        edges_1[i] = construct_vertex_id(J_pe, J_offset);
    }
    n_my_edges = partition_nz;

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

    hvr_exec_info info = hvr_body(ctx);

    double elapsed_time_ms = (double)(info.start_hvr_body_wrapup_us -
            info.start_hvr_body_us) / 1000.0;
    fprintf(stderr, "PE %d took %f ms to insert %lu / %lu (%f%%) edges with "
            "batch size %lu ( %f edges per s ). %f %f %f. %d iters.\n",
            pe, elapsed_time_ms, edges_so_far, n_my_edges,
            100.0 * (double)edges_so_far / (double)n_my_edges,
            batch_size, (double)n_my_edges / (elapsed_time_ms / 1000.0),
            (double)(info.start_hvr_body_iterations_us - info.start_hvr_body_us) / 1000.0,
            (double)(info.start_hvr_body_wrapup_us - info.start_hvr_body_iterations_us) / 1000.0,
            (double)(info.end_hvr_body_us - info.start_hvr_body_wrapup_us) / 1000.0,
            info.executed_iters);

    hvr_finalize(ctx);

    shmem_finalize();
    
    if (pe == 0) {
        printf("Success\n");
    }

    return 0;
}
