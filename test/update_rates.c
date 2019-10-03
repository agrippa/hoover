#include <shmem.h>
#include <stdio.h>
#include <hoover.h>
#include "mmio.h"

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
        hvr_set_t *terminated_coupled_pes) {
    return (edges_so_far == n_my_edges);
}

int main(int argc, char **argv) {
    int ret_code;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <mat-file>\n", argv[0]);
        return 1;
    }

    shmem_init();
    pe = shmem_my_pe();
    npes = shmem_n_pes();

    const char *mat_filename = argv[1];
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

    if (pe == 0) {
        printf("Matrix %s is %d x %d with %d non-zeroes\n", mat_filename, M, N,
                nz);
    }

    int *I = (int *)malloc(nz * sizeof(*I));
    assert(I);
    int *J = (int *)malloc(nz * sizeof(*J));
    assert(J);
    double *val = (double *)malloc(nz * sizeof(*val));
    assert(val);

    n = fread(I, sizeof(int), nz, fp);
    assert(n == nz);
    n = fread(J, sizeof(int), nz, fp);
    assert(n == nz);
    n = fread(val, sizeof(double), nz, fp);
    assert(n == nz);

    fclose(fp);

    // TODO read graph into my_edges

#if 0
    hvr_ctx_t ctx;
    hvr_ctx_create(&ctx);

    hvr_vertex_t *vert = hvr_vertex_create(ctx);

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

    printf("PE %d took %f ms to insert %lu edges ( %f edges per ms )\n",
            pe, (double)elapsed_time / 1000.0, n_my_edges,
            (double)n_my_edges / ((double)elapsed_time / 1000.0));

    hvr_finalize(ctx);
#endif

    shmem_finalize();
    
    if (pe == 0) {
        printf("Success\n");
    }

    return 0;
}
