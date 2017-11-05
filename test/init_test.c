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

#define PARTITION_DIM 40

unsigned grid_dim = 3;
static int npes, pe;
long long total_time = 0;
long long max_elapsed = 0;
long long elapsed_time = 0;

long long p_wrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
long p_sync[SHMEM_REDUCE_SYNC_SIZE];

/*
 * Callback for the HOOVER runtime to use to determine the PE owning a given
 * vertex, and that vertex's local offset on the owner PE.
 */
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

uint16_t actor_to_partition(hvr_sparse_vec_t *actor, hvr_ctx_t ctx) {
    const double row = hvr_sparse_vec_get(0, actor, ctx);
    const double col = hvr_sparse_vec_get(1, actor, ctx);

    const double partition_size = (double)grid_dim / (double)PARTITION_DIM;

    // interaction distance, increase grid_dim if you hit this assertion
    assert(partition_size > 1.1);

    const int row_partition = (int)(row / partition_size);
    const int col_partition = (int)(col / partition_size);
    const uint16_t partition = row_partition * PARTITION_DIM + col_partition;
    return partition;
}

/*
 * Callback for the HOOVER runtime for updating positional or logical metadata
 * attached to each vertex based on the updated neighbors on each time step.
 */
void update_metadata(hvr_sparse_vec_t *vertex, hvr_sparse_vec_t *neighbors,
        const size_t n_neighbors, hvr_pe_set_t *couple_with, hvr_ctx_t ctx) {
    /*
     * If vertex is not already infected, update it to be infected if any of its
     * neighbors are.
     */
    if (hvr_sparse_vec_get(2, vertex, ctx) == 0.0) {
        for (int i = 0; i < n_neighbors; i++) {
            if (hvr_sparse_vec_get(2, &neighbors[i], ctx)) {
                const int infected_by = hvr_sparse_vec_get_owning_pe(
                        &neighbors[i]);
                hvr_pe_set_insert(infected_by, couple_with);
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

        hvr_sparse_vec_init(mins);
        hvr_sparse_vec_init(maxs);
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
int might_interact(const uint16_t partition, hvr_pe_set_t *partitions,
        hvr_ctx_t ctx) {
    /*
     * If partition is neighboring any partition in partitions, they might
     * interact.
     */
    const int partition_row = partition / PARTITION_DIM;
    const int partition_col = partition % PARTITION_DIM;

    for (int row = -1; row <= 1; row++) {
        for (int col = -1; col <= 1; col++) {
            const int part = (partition_row + row) * PARTITION_DIM +
                (partition_col + col);
            if (hvr_pe_set_contains(part, partitions)) {
                return 1;
            }
        }
    }
    return 0;
}

static unsigned long long last_time = 0;

/*
 * Callback used by the HOOVER runtime to check if this PE can abort out of the
 * simulation.
 */
int check_abort(hvr_sparse_vec_t *vertices, const size_t n_vertices,
        hvr_ctx_t ctx, hvr_sparse_vec_t *out_coupled_metric) {
    // Abort if all of my member vertices are infected
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

    hvr_sparse_vec_set(0, (double)nset, out_coupled_metric, ctx);
    hvr_sparse_vec_set(1, (double)n_vertices, out_coupled_metric, ctx);

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
    const unsigned grid_cells_this_pe = grid_cell_end - grid_cell_start;

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
    hvr_sparse_vec_t *vertices = hvr_sparse_vec_create_n(cells_per_pe + 1);
    for (vertex_id_t vertex = grid_cell_start; vertex < grid_cell_end;
            vertex++) {
        const vertex_id_t row = vertex / grid_dim;
        const vertex_id_t col = vertex % grid_dim;

        hvr_sparse_vec_set_id(vertex, &vertices[vertex - grid_cell_start]);
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

#ifdef VERBOSE
    for (unsigned i = 0; i < n_local_grid_rows * grid_dim; i++) {
        char buf[1024];
        hvr_sparse_vec_dump(&vertices[i], buf, 1024);
        printf("%u - %s\n", i, buf);
    }
#endif

    // Statically divide 2D grid into PARTITION_DIM x PARTITION_DIM partitions
    hvr_init(PARTITION_DIM * PARTITION_DIM, grid_cells_this_pe, vertices,
            update_metadata, might_interact, check_abort,
            vertex_owner, actor_to_partition, 1.1, 0, 1,
            INT64_MAX, hvr_ctx);

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
