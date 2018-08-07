/* For license: see LICENSE.txt file at top-level */

#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <shmem.h>
#include <math.h>

#include <hoover.h>

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

hvr_partition_t actor_to_partition(hvr_sparse_vec_t *actor, hvr_ctx_t ctx) {
    const double row = hvr_sparse_vec_get(0, actor, ctx);
    const double col = hvr_sparse_vec_get(1, actor, ctx);

    const double partition_size = (double)grid_dim / (double)PARTITION_DIM;

    const int row_partition = (int)(row / partition_size);
    const int col_partition = (int)(col / partition_size);
    const hvr_partition_t partition = row_partition * PARTITION_DIM + col_partition;
    return partition;
}

/*
 * Callback for the HOOVER runtime for updating positional or logical metadata
 * attached to each vertex based on the updated neighbors on each time step.
 */
void update_metadata(hvr_sparse_vec_t *vertex, hvr_sparse_vec_t *neighbors,
        const size_t n_neighbors, hvr_set_t *couple_with, hvr_ctx_t ctx) {
    /*
     * If vertex is not already infected, update it to be infected if any of its
     * neighbors are.
     */
    if (hvr_sparse_vec_get(2, vertex, ctx) == 0.0) {
        for (int i = 0; i < n_neighbors; i++) {
            if (hvr_sparse_vec_get(2, &neighbors[i], ctx)) {
                const int infected_by = hvr_sparse_vec_get_owning_pe(
                        &neighbors[i]);
                hvr_set_insert(infected_by, couple_with);
                hvr_sparse_vec_set(2, 1.0, vertex, ctx);
                break;
            }
        }
    }
}

int update_summary_data(void *summary, hvr_sparse_vec_t *actors,
        const int nactors, hvr_ctx_t ctx) {
    hvr_sparse_vec_t *mins = ((hvr_sparse_vec_t *)summary) + 0;
    hvr_sparse_vec_t *maxs = ((hvr_sparse_vec_t *)summary) + 1;
    double existing_minx, existing_miny, existing_maxx, existing_maxy;
    existing_minx = existing_miny = existing_maxx = existing_maxy = 0.0;

    const int first_timestep = (hvr_current_timestep(ctx) == 1);

    if (!first_timestep) {
        existing_minx = hvr_sparse_vec_get(0, mins, ctx);
        existing_miny = hvr_sparse_vec_get(1, mins, ctx);
        existing_maxx = hvr_sparse_vec_get(0, maxs, ctx);
        existing_maxy = hvr_sparse_vec_get(1, maxs, ctx);
    }

    assert(nactors > 0);
    double minx = hvr_sparse_vec_get(0, &actors[0], ctx);
    double maxx = minx;
    double miny = hvr_sparse_vec_get(1, &actors[0], ctx);
    double maxy = miny;

    // For each vertex on this PE
    for (unsigned i = 1; i < nactors; i++) {
        hvr_sparse_vec_t *curr = actors + i;
        double currx = hvr_sparse_vec_get(0, curr, ctx);
        double curry = hvr_sparse_vec_get(1, curr, ctx);

        if (currx < minx) minx = currx;
        if (currx > maxx) maxx = currx;
        if (curry < miny) miny = curry;
        if (curry > maxy) maxy = curry;
    }

    if (first_timestep || existing_minx != minx || existing_miny != miny ||
            existing_maxx != maxx || existing_maxy != maxy) {

        hvr_sparse_vec_init(mins, HVR_INVALID_GRAPH, ctx);
        hvr_sparse_vec_init(maxs, HVR_INVALID_GRAPH, ctx);
        hvr_sparse_vec_set(0, minx, mins, ctx);
        hvr_sparse_vec_set(1, miny, mins, ctx);
        hvr_sparse_vec_set(0, maxx, maxs, ctx);
        hvr_sparse_vec_set(1, maxy, maxs, ctx);
        return 1; // summary data changed
    } else {
        return 0; // no change
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
        hvr_sparse_vec_t *out_coupled_metric) {
    // Abort if all of my member vertices are infected
    size_t nset = 0;
    hvr_sparse_vec_t *vert = hvr_vertex_iter_next(iter);
    while (vert) {
        if (hvr_sparse_vec_get(2, vert, ctx) > 0.0) {
            nset++;
        }
        vert = hvr_vertex_iter_next(iter);
    }

    unsigned long long this_time = hvr_current_time_us();
    printf("PE %d - timestep %lu - set %lu / %u - %f ms\n", pe,
            (uint64_t)hvr_current_timestep(ctx), nset, grid_cells_this_pe,
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

    hvr_sparse_vec_set(0, (double)nset, out_coupled_metric, ctx);
    hvr_sparse_vec_set(1, (double)grid_cells_this_pe, out_coupled_metric, ctx);

    if (nset == grid_cells_this_pe) {
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
    hvr_graph_id_t graph = hvr_graph_create(hvr_ctx);

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
    unsigned const_features[] = {2, 3};
    double const_vals[] = {42.0, 43.0};
    hvr_sparse_vec_t *vertices = hvr_sparse_vec_create_n_with_const_attrs(
            grid_cell_end - grid_cell_start, graph, const_features, const_vals,
            2, hvr_ctx);
    for (hvr_vertex_id_t vertex = grid_cell_start; vertex < grid_cell_end;
            vertex++) {
        const hvr_vertex_id_t row = vertex / grid_dim;
        const hvr_vertex_id_t col = vertex % grid_dim;

        hvr_sparse_vec_set(0, (double)row, &vertices[vertex - grid_cell_start],
                hvr_ctx);
        hvr_sparse_vec_set(1, (double)col, &vertices[vertex - grid_cell_start],
                hvr_ctx);
        if (pe == 0 && vertex == grid_cell_start) {
            // Initialze just the cell at (0, 0) as infected.
            hvr_sparse_vec_set(2, 1.0, &vertices[vertex - grid_cell_start],
                    hvr_ctx);
        } else {
            hvr_sparse_vec_set(2, 0.0, &vertices[vertex - grid_cell_start],
                    hvr_ctx);
        }
    }

    for (hvr_vertex_id_t vertex = grid_cell_start; vertex < grid_cell_end;
            vertex++) {
        hvr_sparse_vec_t *vert = &vertices[vertex - grid_cell_start];
        finalize_actor_for_timestep(vert, hvr_ctx->timestep);

        hvr_ctx->timestep += 1;
        hvr_sparse_vec_t tmp;

        memcpy(&tmp, vert, sizeof(tmp));

        double val = hvr_sparse_vec_get(2, vert, hvr_ctx);
        assert(val == 42.0);
        val = hvr_sparse_vec_get(3, vert, hvr_ctx);
        assert(val == 43.0);
    }

    shmem_finalize();

    printf("PE %d done!\n", pe);

    return 0;
}