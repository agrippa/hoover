/* For license: see LICENSE.txt file at top-level */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <shmem.h>
#include <math.h>

#include <hoover.h>

/*
 * This simple example of the HOOVER framework creates a 2D grid of statically
 * placed actors.
 *
 * The grid has dimensions of grid_dim x grid_dim. We place actors at the center
 * of each grid cell, hence the total # of actors = grid_dim * grid_dim. Each
 * actors has a row and column coordinate, where both row and column start at 0
 * and go to grid_dim - 1 (row major). Actor IDs increase by 1 across each row
 * (hence, consecutive actor IDs are generally neighboring each other in the
 * same row). We partition actors across PEs as evenly as possible in
 * consecutive chunks. Hence, a single PE may own multiple rows and indeed may
 * own partial rows (if the number of rows is not evenly divisible by the number
 * of PEs).
 *
 * This code statically splits the domain of grid_dim x grid_dim into 10 x 10
 * partitions.
 */

#define CONNECTIVITY_THRESHOLD 1.1
#define PARTITION_DIM 40

unsigned grid_dim = 3;
static int npes, pe;
long long total_time = 0;
long long max_elapsed = 0;
long long elapsed_time = 0;

long long p_wrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
long p_sync[SHMEM_REDUCE_SYNC_SIZE];

static unsigned grid_cells_this_pe;

hvr_partition_t actor_to_partition(hvr_vertex_t *actor, hvr_ctx_t ctx) {
    const double row = hvr_vertex_get(0, actor, ctx);
    const double col = hvr_vertex_get(1, actor, ctx);

    const double partition_size = (double)grid_dim / (double)PARTITION_DIM;

    const int row_partition = (int)(row / partition_size);
    const int col_partition = (int)(col / partition_size);
    const hvr_partition_t partition = row_partition * PARTITION_DIM +
        col_partition;
    return partition;
}

hvr_edge_type_t should_have_edge(hvr_vertex_t *a, hvr_vertex_t *b,
        hvr_ctx_t ctx) {
    double delta0 = hvr_vertex_get(0, b, ctx) - hvr_vertex_get(0, a, ctx);
    double delta1 = hvr_vertex_get(1, b, ctx) - hvr_vertex_get(1, a, ctx);
    if (delta0 * delta0 + delta1 * delta1 <=
            CONNECTIVITY_THRESHOLD * CONNECTIVITY_THRESHOLD) {
        return BIDIRECTIONAL;
    } else {
        return NO_EDGE;
    }
}

/*
 * Callback for the HOOVER runtime for updating positional or logical metadata
 * attached to each vertex based on the updated neighbors on each time step.
 */
void update_metadata(hvr_vertex_t *vertex, hvr_set_t *couple_with,
        hvr_ctx_t ctx) {
    /*
     * If vertex is not already infected, update it to be infected if any of its
     * neighbors are.
     */
    if (hvr_vertex_get(2, vertex, ctx) == 0.0) {
        hvr_vertex_id_t *neighbors;
        hvr_edge_type_t *directions;
        size_t n_neighbors;
        hvr_get_neighbors(vertex, &neighbors, &directions, &n_neighbors, ctx);
        for (size_t i = 0; i < n_neighbors; i++) {
            hvr_vertex_t *neighbor = hvr_get_vertex(neighbors[i], ctx);
            if (hvr_vertex_get(2, neighbor, ctx)) {
                const int infected_by = hvr_vertex_get_owning_pe(neighbor);
                hvr_set_insert(infected_by, couple_with);
                // fprintf(stderr, "PE %d coupling with PE %d\n", shmem_my_pe(), infected_by);
                hvr_vertex_set(2, 1.0, vertex, ctx);
                break;
            }
        }
    }
}

/*
 * Callback used to check if this PE might interact with another PE based on the
 * maximums and minimums of all vertices owned by each PE.
 */
int might_interact(const hvr_partition_t partition, hvr_set_t *partitions,
        hvr_partition_t *interacting_partitions,
        unsigned *n_interacting_partitions,
        unsigned interacting_partitions_capacity,
        hvr_ctx_t ctx) {
    /*
     * If partition is neighboring any partition in partitions, they might
     * interact.
     */

    // Each partition has this length on each side
    double partition_dim = (double)grid_dim / (double)PARTITION_DIM;
    const int partition_row = partition / PARTITION_DIM;
    const int partition_col = partition % PARTITION_DIM;

    // Get bounding box of partitions in the grid coordinate system
    double min_row = (double)partition_row * partition_dim;
    double max_row = min_row + partition_dim;
    double min_col = (double)partition_col * partition_dim;
    double max_col = min_col + partition_dim;

    /*
     * Expand partition bounding box to include any possible points within
     * CONNECTIVITY_THRESHOLD distance.
     */
    min_row -= CONNECTIVITY_THRESHOLD;
    max_row += CONNECTIVITY_THRESHOLD;
    min_col -= CONNECTIVITY_THRESHOLD;
    max_col += CONNECTIVITY_THRESHOLD;

    int min_partition_row, min_partition_col, max_partition_row,
        max_partition_col;

    if (min_row < 0.0) min_partition_row = 0;
    else min_partition_row = (int)(min_row / partition_dim);

    if (min_col < 0.0) min_partition_col = 0;
    else min_partition_col = (int)(min_col / partition_dim);

    if (max_row >= (double)grid_dim) max_partition_row = PARTITION_DIM - 1;
    else max_partition_row = (int)(max_row / partition_dim);

    if (max_col >= (double)grid_dim) max_partition_col = PARTITION_DIM - 1;
    else max_partition_col = (int)(max_col / partition_dim);

    assert(min_partition_row <= max_partition_row);
    assert(min_partition_col <= max_partition_col);

    unsigned count_interacting_partitions = 0;
    for (int r = min_partition_row; r <= max_partition_row; r++) {
        for (int c = min_partition_col; c <= max_partition_col; c++) {
            const int part = r * PARTITION_DIM + c;
            if (hvr_set_contains(part, partitions)) {
                assert(count_interacting_partitions + 1 <=
                        interacting_partitions_capacity);
                interacting_partitions[count_interacting_partitions++] = part;
            }
        }
    }
    *n_interacting_partitions = count_interacting_partitions;
    return count_interacting_partitions > 0;
}

static unsigned long long last_time = 0;

/*
 * Callback used by the HOOVER runtime to check if this PE can abort out of the
 * simulation.
 */
int check_abort(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_set_t *to_couple_with,
        hvr_vertex_t *out_coupled_metric) {
    // Abort if all of my member vertices are infected
    size_t nset = 0;
    hvr_vertex_t *vert = hvr_vertex_iter_next(iter);
    while (vert) {
        if (hvr_vertex_get(2, vert, ctx) > 0.0) {
            nset++;
        }
        vert = hvr_vertex_iter_next(iter);
    }

    unsigned long long this_time = hvr_current_time_us();
    // fprintf(stderr, "PE %d - set %lu / %u - %f ms\n", pe, nset,
    //         grid_cells_this_pe,
    //         last_time == 0 ? 0 : (double)(this_time - last_time) / 1000.0);
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

    hvr_vertex_set(0, (double)nset, out_coupled_metric, ctx);
    hvr_vertex_set(1, (double)grid_cells_this_pe, out_coupled_metric, ctx);

    if (nset == grid_cells_this_pe) {
        return 1;
    // } else if (shmem_my_pe() == shmem_n_pes() - 1 &&
    //         nset == grid_dim * grid_dim) {
    //     return 1;
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

    // Partition cells of a 2D grid as evenly as posbible across all PEs
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
    grid_cells_this_pe = grid_cell_end - grid_cell_start;

    if (pe == 0) {
        fprintf(stderr, "%d PEs, %u x %u = %u grid points, %u grid cells per "
                "PE\n", npes, grid_dim, grid_dim, grid_dim * grid_dim,
                cells_per_pe);
    }
    fprintf(stderr, "PE %d owns actors %d -> %d\n", shmem_my_pe(),
            grid_cell_start, grid_cell_end);
    if (grid_cells_this_pe == 0) {
        fprintf(stderr, "WARNING PE %d has no grid cells\n", pe);
    }

    /*
     * Create each vertex owned by this PE. Each vertex has three attributes:
     *
     *  0: row of this cell
     *  1: column of this cell
     *  2: whether this cell has been "infected" by its neighbors
     */
    hvr_vertex_t *vertices = hvr_vertex_create_n(
            grid_cell_end - grid_cell_start, hvr_ctx);
    for (hvr_vertex_id_t vertex = grid_cell_start; vertex < grid_cell_end;
            vertex++) {
        const hvr_vertex_id_t row = vertex / grid_dim;
        const hvr_vertex_id_t col = vertex % grid_dim;

        hvr_vertex_set(0, (double)row, &vertices[vertex - grid_cell_start],
                hvr_ctx);
        hvr_vertex_set(1, (double)col, &vertices[vertex - grid_cell_start],
                hvr_ctx);
        if (pe == 0 && vertex == grid_cell_start) {
            // Initialze just the cell at (0, 0) as infected.
            hvr_vertex_set(2, 1.0, &vertices[vertex - grid_cell_start],
                    hvr_ctx);
        } else {
            hvr_vertex_set(2, 0.0, &vertices[vertex - grid_cell_start],
                    hvr_ctx);
        }
    }

    // Statically divide 2D grid into PARTITION_DIM x PARTITION_DIM partitions
    hvr_init(PARTITION_DIM * PARTITION_DIM,
            update_metadata,
            might_interact,
            check_abort,
            actor_to_partition,
            NULL, // start_time_step
            should_have_edge,
            20, // max_elapsed_seconds
            1, // max_graph_traverse_depth
            hvr_ctx);

    const long long start_time = hvr_current_time_us();
    hvr_body(hvr_ctx);
    elapsed_time = hvr_current_time_us() - start_time;

    shmem_longlong_sum_to_all(&total_time, &elapsed_time, 1, 0, 0, npes, p_wrk,
            p_sync);
    shmem_barrier_all();
    shmem_longlong_max_to_all(&max_elapsed, &elapsed_time, 1, 0, 0, npes, p_wrk,
            p_sync);
    shmem_barrier_all();

    if (pe == 0) {
        fprintf(stderr, "%d PEs, total CPU time = %f ms, max elapsed = %f ms, ~%u cells "
                "per PE\n", npes, (double)total_time / 1000.0,
                (double)max_elapsed / 1000.0, cells_per_pe);
    }

    hvr_finalize(hvr_ctx);

    shmem_finalize();

    return 0;
}
