#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <shmem.h>
#include <math.h>

#include <hoover.h>

unsigned grid_dim = 3;
static int npes, pe;
long long total_time = 0;
long long max_elapsed = 0;
long long elapsed_time = 0;

long long p_wrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
long p_sync[SHMEM_REDUCE_SYNC_SIZE];

void vertex_owner(vertex_id_t vertex, unsigned *out_pe,
        size_t *out_local_offset) {
    const unsigned grid_size = grid_dim * grid_dim;
    const unsigned cells_per_pe = grid_size / npes;
    unsigned leftover = grid_size - (npes * cells_per_pe);

    if (vertex < (leftover * (cells_per_pe + 1))) {
        *out_pe = vertex / (cells_per_pe + 1);
        const unsigned base_pe_offset = *out_pe * (cells_per_pe + 1);
        *out_local_offset = vertex - base_pe_offset;
    } else {
        unsigned new_vertex = vertex - (leftover * (cells_per_pe + 1));
        *out_pe = leftover + (new_vertex / cells_per_pe);
        const unsigned base_pe_offset =
            (leftover * (cells_per_pe + 1)) +
            ((new_vertex / cells_per_pe) * cells_per_pe);
        *out_local_offset = vertex - base_pe_offset;
    }
}

void update_metadata(hvr_sparse_vec_t *vertex, hvr_sparse_vec_t *neighbors,
        const size_t n_neighbors, hvr_ctx_t ctx) {
    if (hvr_sparse_vec_get(2, vertex, ctx) == 0.0) {
        for (int i = 0; i < n_neighbors; i++) {
            if (hvr_sparse_vec_get(2, &neighbors[i], ctx)) {
                hvr_sparse_vec_set(2, 1.0, vertex, ctx);
                break;
            }
        }
    }
}

static unsigned long long last_time = 0;

int check_abort(hvr_sparse_vec_t *vertices, const size_t n_vertices,
        hvr_ctx_t ctx) {
    size_t nset = 0;
    for (int i = 0; i < n_vertices; i++) {
        if (hvr_sparse_vec_get(2, &vertices[i], ctx) > 0.0) {
            nset++;
        }
    }

    unsigned long long this_time = hvr_current_time_us();
    printf("PE %d - timestep %lu - set %lu / %lu - %f ms\n", pe,
            hvr_current_timestep(ctx), nset, n_vertices,
            last_time == 0 ? 0 : (double)(this_time - last_time) / 1000.0);
    last_time = this_time;

    // Only really makes sense when running on one PE for testing
    // printf("\nPE %d - timestep %lu:\n", pe, hvr_current_timestep(ctx));
    // for (int i = grid_dim - 1; i >= 0; i--) {
    //     printf("  ");
    //     for (int j = 0; j < grid_dim; j++) {
    //         if (hvr_sparse_vec_get(2, &vertices[i * grid_dim + j], ctx) > 0.0) {
    //             printf(" 1");
    //         } else {
    //             printf(" 0");
    //         }
    //     }
    //     printf("\n");
    // }

    if (nset == n_vertices) {
        return 1;
    } else {
        return 0;
    }
}

int main(int argc, char **argv) {
    hvr_ctx_t hvr_ctx;

    if (argc == 2) {
        grid_dim = atoi(argv[1]);
    }

    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        p_sync[i] = SHMEM_SYNC_VALUE;
    }

    shmem_init();
    hvr_ctx_create(&hvr_ctx);

    pe = shmem_my_pe();
    npes = shmem_n_pes();

    const unsigned grid_size = grid_dim * grid_dim;
    const unsigned cells_per_pe = grid_size / npes;
    unsigned leftover = grid_size - (npes * cells_per_pe);
    unsigned grid_cell_start, grid_cell_end;
    if (pe < leftover) {
        grid_cell_start = pe * (cells_per_pe + 1);
        grid_cell_end = grid_cell_start + cells_per_pe + 1;
    } else {
        unsigned base = leftover * (cells_per_pe + 1);
        grid_cell_start = base + (pe - leftover) * cells_per_pe;
        grid_cell_end = grid_cell_start + cells_per_pe;
    }
    const unsigned grid_cells_this_pe = grid_cell_end - grid_cell_start;

    if (pe == 0) {
        fprintf(stderr, "%d PEs, %u x %u = %u grid points, %u grid cells per "
                "PE\n", npes, grid_dim, grid_dim, grid_dim * grid_dim,
                cells_per_pe);
    }
    // fprintf(stderr, "PE %d has %u -> %u (%u grid rows)\n", pe, grid_row_start,
    //         grid_row_end, grid_rows_this_pe);
    if (grid_cells_this_pe == 0) {
        fprintf(stderr, "WARNING PE %d has no grid cells\n", pe);
    }

    hvr_sparse_vec_t *vertices = hvr_sparse_vec_create_n(cells_per_pe + 1);
    for (vertex_id_t vertex = grid_cell_start; vertex < grid_cell_end;
            vertex++) {
        const vertex_id_t row = vertex / grid_dim;
        const vertex_id_t col = vertex % grid_dim;

        vertices[vertex - grid_cell_start].id = vertex;
        hvr_sparse_vec_set(0, (double)row, &vertices[vertex - grid_cell_start], hvr_ctx);
        hvr_sparse_vec_set(1, (double)col, &vertices[vertex - grid_cell_start], hvr_ctx);
        hvr_sparse_vec_set(2, 0.0, &vertices[vertex - grid_cell_start], hvr_ctx);
    }
    if (pe == 0) {
        hvr_sparse_vec_set(2, 1.0, &vertices[0], hvr_ctx);
    }

#ifdef VERBOSE
    for (unsigned i = 0; i < n_local_grid_rows * grid_dim; i++) {
        char buf[1024];
        hvr_sparse_vec_dump(&vertices[i], buf, 1024);
        printf("%u - %s\n", i, buf);
    }
#endif

    hvr_pe_neighbors_set_t *neighbors = hvr_create_empty_pe_neighbors_set(
            hvr_ctx);
    hvr_pe_neighbors_set_insert(pe, neighbors);
    if (pe > 0) {
        hvr_pe_neighbors_set_insert(pe - 1, neighbors);
    }
    if (pe < npes - 1) {
        hvr_pe_neighbors_set_insert(pe + 1, neighbors);
    }

    hvr_init(grid_cells_this_pe, vertices,
            update_metadata, check_abort,
            vertex_owner, 1.1, 0, 1, hvr_ctx);

    const long long start_time = hvr_current_time_us();
    hvr_body(hvr_ctx);
    elapsed_time = hvr_current_time_us() - start_time;

    shmem_longlong_sum_to_all(&total_time, &elapsed_time, 1, 0, 0, npes, p_wrk,
            p_sync);
    shmem_barrier_all();
    shmem_longlong_max_to_all(&max_elapsed, &elapsed_time, 1, 0, 0, npes, p_wrk,
            p_sync);
    shmem_barrier_all();

    hvr_finalize(hvr_ctx);

    shmem_finalize();

    if (pe == 0) {
        printf("%d PEs, total CPU time = %f ms, max elapsed = %f ms, ~%u cells "
                "per PE\n", npes, (double)total_time / 1000.0,
                (double)max_elapsed / 1000.0, cells_per_pe);
    }

    return 0;
}
