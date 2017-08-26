#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <shmem.h>
#include <math.h>

#include <hoover.h>

unsigned grid_dim = 3;
static int npes, pe;

void vertex_owner(vertex_id_t vertex, unsigned *out_pe,
        size_t *out_local_offset) {
    const unsigned grid_points = grid_dim * grid_dim;
    const unsigned grid_points_per_pe = (grid_points + npes - 1) / npes;
    *out_pe = vertex / grid_points_per_pe;
    *out_local_offset = vertex % grid_points_per_pe;
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

double sparse_vec_distance_measure(hvr_sparse_vec_t *a,
        hvr_sparse_vec_t *b, hvr_ctx_t ctx) {
    const double row_delta = hvr_sparse_vec_get(0, b, ctx) -
        hvr_sparse_vec_get(0, a, ctx);
    const double col_delta = hvr_sparse_vec_get(1, b, ctx) -
        hvr_sparse_vec_get(1, a, ctx);
    return sqrt(row_delta * row_delta + col_delta * col_delta);
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

    shmem_init();
    hvr_ctx_create(&hvr_ctx);

    pe = shmem_my_pe();
    npes = shmem_n_pes();
    printf("Hello from PE %d / %d\n", pe + 1, npes);

    const unsigned grid_points = grid_dim * grid_dim;
    const unsigned grid_points_per_pe = (grid_points + npes - 1) / npes;
    const unsigned grid_point_start = pe * grid_points_per_pe;
    unsigned grid_point_end = (pe + 1) * grid_points_per_pe;
    if (grid_point_end > grid_points) grid_point_end = grid_points;
    const unsigned n_local_grid_points = grid_point_end - grid_point_start;

    hvr_sparse_vec_t *vertices = hvr_sparse_vec_create_n(grid_points_per_pe);
    for (unsigned i = 0; i < n_local_grid_points; i++) {
        const vertex_id_t vertex = pe * grid_points_per_pe + i;
        const vertex_id_t row = vertex / grid_dim;
        const vertex_id_t col = vertex % grid_dim;

        vertices[i].id = vertex;
        hvr_sparse_vec_set(0, (double)row, &vertices[i], hvr_ctx);
        hvr_sparse_vec_set(1, (double)col, &vertices[i], hvr_ctx);
        hvr_sparse_vec_set(2, 0.0, &vertices[i], hvr_ctx);
    }
    if (pe == 0) {
        hvr_sparse_vec_set(2, 1.0, &vertices[0], hvr_ctx);
    }

#ifdef VERBOSE
    for (unsigned i = 0; i < n_local_grid_points; i++) {
        char buf[1024];
        hvr_sparse_vec_dump(&vertices[i], buf, 1024);
        printf("%u - %s\n", i, buf);
    }
#endif

    hvr_edge_set_t *edges = hvr_create_empty_edge_set();
    for (unsigned i = 0; i < n_local_grid_points; i++) {
        const vertex_id_t vertex = vertices[i].id;
        const vertex_id_t row = vertex / grid_dim;
        const vertex_id_t col = vertex % grid_dim;

        if (row > 0) {
            const vertex_id_t down_vertex = (row - 1) * grid_dim + col;
            hvr_add_edge(vertex, down_vertex, edges);
        }
        if (row < grid_dim - 1) {
            const vertex_id_t up_vertex = (row + 1) * grid_dim + col;
            hvr_add_edge(vertex, up_vertex, edges);
        }
        if (col > 0) {
            const vertex_id_t left_vertex = row * grid_dim + (col - 1);
            hvr_add_edge(vertex, left_vertex, edges);
        }
        if (col < grid_dim - 1) {
            const vertex_id_t right_vertex = row * grid_dim + (col + 1);
            hvr_add_edge(vertex, right_vertex, edges);
        }
#ifdef  VERBOSE
        printf("PE %d - vertex %lu - # neighbors %lu\n", pe, vertex, hvr_count_edges(vertex, edges));
#endif
    }

    hvr_init(n_local_grid_points, vertices, edges,
            update_metadata, sparse_vec_distance_measure, check_abort,
            vertex_owner, 1.1, hvr_ctx);
    hvr_body(hvr_ctx);
    hvr_finalize(hvr_ctx);

    shmem_finalize();

    return 0;
}
