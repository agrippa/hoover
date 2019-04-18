/* For license: see LICENSE.txt file at top-level */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <shmem.h>
#include <math.h>
#include <string.h>

#include <hoover.h>

#ifdef MULTITHREADED
#include <omp.h>
#include <shmemx.h>
#endif

#define TIME_STEP 0
#define ACTOR_ID 1
#define PX 2
#define PY 3
#define INFECTED 4
#define DST_X 5
#define DST_Y 6
#define NEXT_CREATED 7

#define PORTAL_CAPTURE_RADIUS 5.0
#define PE_ROW(this_pe) ((this_pe) / n_cells_x)
#define PE_COL(this_pe) ((this_pe) % n_cells_x)
#define PE_ROW_CELL_START(this_pe) ((double)PE_ROW(this_pe) * cell_dim_y)
#define PE_COL_CELL_START(this_pe) ((double)PE_COL(this_pe) * cell_dim_x)
#define CELL_ROW(y_coord) ((y_coord) / cell_dim_y)
#define CELL_COL(x_coord) ((x_coord) / cell_dim_x)
#define CELL_INDEX(cell_row, cell_col) ((cell_row) * n_cells_x + (cell_col))

int max_modeled_timestep = 0;
size_t n_local_actors = 0;
uint64_t total_n_actors = 0;
static unsigned n_time_partition = 0;
static unsigned n_y_partition = 0;
static unsigned n_x_partition = 0;
#ifdef MULTITHREADED
static int nthreads = 1;
#endif

typedef struct _portal_t {
    int pes[2];
    struct {
        double x, y;
    } locations[2];
} portal_t;

static int npes, pe;
static int n_cells_y, n_cells_x;
static double cell_dim_y;
static double cell_dim_x;
static double infection_radius;
static int max_num_timesteps;

/*
 * Construct a 2D grid, with one grid cell per PE. Build connections between
 * cells, having each cell connected to all eight neighbor cells plus some
 * randomly added long distance interactions.
 *
 * Inside of each cell of the 2D grid, generate a random number of actors that
 * perform random walks on the whole grid (possibly jumping to other cells) and
 * infect each other if within certain distances.
 */

long long p_wrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
int int_p_wrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
long p_sync[SHMEM_REDUCE_SYNC_SIZE];

static double distance(double x1, double y1, double x2, double y2) {
    double deltax = x2 - x1;
    double deltay = y2 - y1;
    return sqrt((deltay * deltay) + (deltax * deltax));
}

hvr_edge_type_t should_have_edge(hvr_vertex_t *base, hvr_vertex_t *neighbor,
        hvr_ctx_t ctx) {
    /*
     * Should always have an edge to the vertex that represents my state on the
     * next step.
     */
    int base_id = (int)hvr_vertex_get(ACTOR_ID, base, ctx);
    int neighbor_id = (int)hvr_vertex_get(ACTOR_ID, neighbor, ctx);

    int base_time = (int)hvr_vertex_get(TIME_STEP, base, ctx);
    int neighbor_time = (int)hvr_vertex_get(TIME_STEP, neighbor, ctx);

    if (base_id == neighbor_id) {
        if (base_time == neighbor_time + 1) {
            // fprintf(stderr, "Computing edges %d @ %d and %d @ %d | "
            //         "DIRECTED_IN\n", base_id, base_time, neighbor_id,
            //         neighbor_time);
            return DIRECTED_IN;
        } else if (neighbor_time == base_time + 1) {
            // fprintf(stderr, "Computing edges %d @ %d and %d @ %d | "
            //         "DIRECTED_OUT\n", base_id, base_time, neighbor_id,
            //         neighbor_time);
            return DIRECTED_OUT;
        }
    }

    int delta_time = abs(base_time - neighbor_time);
    if (delta_time == 1) {
        double deltax = hvr_vertex_get(PX, neighbor, ctx) -
            hvr_vertex_get(PX, base, ctx);
        double deltay = hvr_vertex_get(PY, neighbor, ctx) -
            hvr_vertex_get(PY, base, ctx);
        if (deltax * deltax + deltay * deltay <=
                infection_radius * infection_radius) {
            if (base_time < neighbor_time) {
                // fprintf(stderr, "Computing edges %d @ %d and %d @ %d | "
                //         "DIRECTED_OUT\n",
                //         base_id, base_time, neighbor_id, neighbor_time);
                return DIRECTED_OUT;
            } else {
                // fprintf(stderr, "Computing edges %d @ %d and %d @ %d | "
                //         "DIRECTED_IN\n",
                //         base_id, base_time, neighbor_id, neighbor_time);
                return DIRECTED_IN;
            }
        }
    }

    // fprintf(stderr, "Computing edges %d @ %d and %d @ %d | NO_EDGE\n",
    //         base_id, base_time, neighbor_id, neighbor_time);
    return NO_EDGE;
}

hvr_partition_t actor_to_partition(hvr_vertex_t *actor, hvr_ctx_t ctx) {
    const double timestep = hvr_vertex_get(TIME_STEP, actor, ctx);
    const double y = hvr_vertex_get(PY, actor, ctx);
    const double x = hvr_vertex_get(PX, actor, ctx);

    const double global_y_dim = (double)n_cells_y * cell_dim_y;
    const double global_x_dim = (double)n_cells_x * cell_dim_x;

    assert((int)timestep < max_num_timesteps);
    assert(x < global_x_dim);
    assert(y < global_y_dim);

    const double partition_time_step = (double)max_num_timesteps /
        (double)n_time_partition;
    const double partition_y_dim = global_y_dim / (double)n_y_partition;
    const double partition_x_dim = global_x_dim / (double)n_x_partition;

    const int time_step_partition = (int)(timestep / partition_time_step);
    const int y_partition = (int)(y / partition_y_dim);
    const int x_partition = (int)(x / partition_x_dim);

    assert(time_step_partition < n_time_partition);
    assert(x_partition < n_x_partition);
    assert(y_partition < n_y_partition);

    return time_step_partition * n_y_partition * n_x_partition +
        y_partition * n_x_partition + x_partition;
}

static void compute_next_pos(double p_x, double p_y,
        double dst_x, double dst_y,
        double *next_p_x, double *next_p_y) {
    if (fabs(p_x - dst_x) < 1e-9 || fabs(p_y - dst_y) < 1e-9) {
        /*
         * Seem to have reached destination, set new destination and start
         * moving there.
         */
        p_x = dst_x;
        p_y = dst_y;
    }

    double vx = dst_x - p_x;
    double vy = dst_y - p_y;
    const double mag = 5.0 * distance(p_x, p_y, dst_x, dst_y);
    const double normalized_vx = vx / mag;
    const double normalized_vy = vy / mag;
    if (fabs(vx) > fabs(normalized_vx)) vx = normalized_vx;
    if (fabs(vy) > fabs(normalized_vy)) vy = normalized_vy;

    double new_x = p_x + vx;
    double new_y = p_y + vy;

    const double global_x_dim = (double)n_cells_x * cell_dim_x;
    const double global_y_dim = (double)n_cells_y * cell_dim_y;
    if (new_x >= global_x_dim) new_x -= global_x_dim;
    if (new_y >= global_y_dim) new_y -= global_y_dim;
    if (new_x < 0.0) new_x += global_x_dim;
    if (new_y < 0.0) new_y += global_y_dim;

    assert(new_x >= 0.0 && new_x < global_x_dim);
    assert(new_y >= 0.0 && new_y < global_y_dim);

    *next_p_x = new_x;
    *next_p_y = new_y;
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
    hvr_vertex_t **verts;
    hvr_edge_type_t *dirs;
    int n_neighbors = hvr_get_neighbors(vertex, &verts, &dirs, ctx);

    const unsigned actor_id = (unsigned)hvr_vertex_get(ACTOR_ID, vertex, ctx);
    const unsigned timestep = (unsigned)hvr_vertex_get(TIME_STEP, vertex, ctx);

    /*
     * Scan over in edges to find the same actor on the previous timestep. Use
     * its infection state as our baseline (i.e. if it is uninfected, start with
     * the assumption that we are uninfected).
     */
    hvr_vertex_t *prev = NULL;
    hvr_vertex_t *next = NULL;
    for (int i = 0; i < n_neighbors; i++) {
        hvr_vertex_t *neighbor = verts[i];
        if ((int)hvr_vertex_get(ACTOR_ID, neighbor, ctx) == actor_id) {
            if (dirs[i] == DIRECTED_IN) {
                assert(prev == NULL);
                assert((int)hvr_vertex_get(TIME_STEP, neighbor, ctx) ==
                        timestep - 1);
                prev = neighbor;
            }
            if (dirs[i] == DIRECTED_OUT) {
                assert(next == NULL);
                assert((int)hvr_vertex_get(TIME_STEP, neighbor, ctx) ==
                        timestep + 1);
                next = neighbor;
            }
        }
    }

    if (prev && hvr_vertex_get(INFECTED, prev, ctx) > 0) {
        hvr_vertex_set(INFECTED, 1, vertex, ctx);
    }

    /*
     * If previous state was not infected, iterate over surroundings in previous
     * time step and check if any of those should infect us.
     */
    if ((int)hvr_vertex_get(INFECTED, vertex, ctx) == 0) {
        for (int i = 0; i < n_neighbors; i++) {
            if (dirs[i] == DIRECTED_IN) {
                hvr_vertex_t *neighbor = verts[i];
                if ((int)hvr_vertex_get(ACTOR_ID, neighbor, ctx) != actor_id) {
                    assert((int)hvr_vertex_get(TIME_STEP, neighbor, ctx) ==
                            timestep - 1);
                    int is_infected = hvr_vertex_get(INFECTED, neighbor, ctx);
                    if (is_infected) {
                        const int infected_by = hvr_vertex_get_owning_pe(neighbor);
                        assert(infected_by < ctx->npes);
                        hvr_set_insert(infected_by, couple_with);
                        hvr_vertex_set(INFECTED, 1, vertex, ctx);
                        break;
                    }
                }
            }
        }
    }

    // Update PX/PY, DST_X/DST_Y based on prev.
    if (prev) {
        double dst_x = hvr_vertex_get(DST_X, prev, ctx);
        double dst_y = hvr_vertex_get(DST_Y, prev, ctx);
        double p_x = hvr_vertex_get(PX, prev, ctx);
        double p_y = hvr_vertex_get(PY, prev, ctx);

        double new_x, new_y;
        compute_next_pos(p_x, p_y, dst_x, dst_y, &new_x, &new_y);

        hvr_vertex_set(PX, new_x, vertex, ctx);
        hvr_vertex_set(PY, new_y, vertex, ctx);
    }

    if (timestep < max_num_timesteps - 1 &&
            hvr_vertex_get(NEXT_CREATED, vertex, ctx) == 0) {
        // Add a next
        hvr_vertex_t *next = hvr_vertex_create_n(1, ctx);

        double x = hvr_vertex_get(PX, vertex, ctx);
        double y = hvr_vertex_get(PY, vertex, ctx);
        double dst_x = hvr_vertex_get(DST_X, vertex, ctx);
        double dst_y = hvr_vertex_get(DST_Y, vertex, ctx);
        int next_timestep = timestep + 1;
        if (next_timestep > max_modeled_timestep) {
            max_modeled_timestep = next_timestep;
        }

        hvr_vertex_set(INFECTED, hvr_vertex_get(INFECTED, vertex, ctx), next,
                ctx);
        hvr_vertex_set(DST_X, hvr_vertex_get(DST_X, vertex, ctx), next, ctx);
        hvr_vertex_set(DST_Y, hvr_vertex_get(DST_Y, vertex, ctx), next, ctx);
        hvr_vertex_set(TIME_STEP, next_timestep, next, ctx);
        hvr_vertex_set(ACTOR_ID, actor_id, next, ctx);

        double new_x, new_y;
        compute_next_pos(x, y, dst_x, dst_y, &new_x, &new_y);

        hvr_vertex_set(PX, new_x, next, ctx);
        hvr_vertex_set(PY, new_y, next, ctx);
        hvr_vertex_set(NEXT_CREATED, 0, next, ctx);

        hvr_vertex_set(NEXT_CREATED, 1, vertex, ctx);
    }
}

/*
 * Callback used to check if this PE might interact with another PE.
 *
 * If partition is neighboring any partition in partitions, they might
 * interact.
 */
void might_interact(const hvr_partition_t partition,
        hvr_partition_t *interacting_partitions,
        unsigned *n_interacting_partitions,
        unsigned interacting_partitions_capacity,
        hvr_ctx_t ctx) {

    // The global dimensions of the full simulation space
    const double global_x_dim = (double)n_cells_x * cell_dim_x;
    const double global_y_dim = (double)n_cells_y * cell_dim_y;

    // Dimension of each partition in the row, column, time directions
    double y_dim = global_y_dim / (double)n_y_partition;
    double x_dim = global_x_dim / (double)n_x_partition;
    double time_dim = (double)max_num_timesteps / (double)n_time_partition;

    /*
     * For the given partition, the (time, row, column) coordinate of this
     * partition in a 2D space.
     */
    unsigned partition_time = partition / (n_y_partition * n_x_partition);
    unsigned partition_y = (partition / n_x_partition) % n_y_partition;
    unsigned partition_x = partition % n_x_partition;

    // Get bounding box of partition in the grid coordinate system
    double min_y = (double)partition_y * y_dim;
    double max_y = min_y + y_dim;
    double min_x = (double)partition_x * x_dim;
    double max_x = min_x + x_dim;
    double min_time = (double)partition_time * time_dim;
    double max_time = min_time + time_dim;

    /*
     * Expand partition bounding box to include any possible points within
     * infection_radius distance.
     */
    min_time -= 1; // Only interact with previous and next timesteps
    max_time += 1;
    min_y -= infection_radius;
    max_y += infection_radius;
    min_x -= infection_radius;
    max_x += infection_radius;

    int min_partition_y, min_partition_x, max_partition_y,
        max_partition_x, min_partition_time, max_partition_time;

    if (min_y < 0.0) min_partition_y = 0;
    else min_partition_y = (int)(min_y / y_dim);

    if (min_x < 0.0) min_partition_x = 0;
    else min_partition_x = (int)(min_x / x_dim);
 
    if (min_time < 0.0) min_partition_time = 0;
    else min_partition_time = (int)(min_time / time_dim);

    if (max_y >= (double)global_y_dim) max_partition_y = n_y_partition - 1;
    else max_partition_y = (int)(max_y / y_dim);

    if (max_x >= (double)global_x_dim) max_partition_x = n_x_partition - 1;
    else max_partition_x = (int)(max_x / x_dim);
 
    if (max_time >= (double)max_num_timesteps) max_partition_time =
        n_time_partition - 1;
    else max_partition_time = (int)(max_time / time_dim);

    assert(min_partition_y <= max_partition_y);
    assert(min_partition_x <= max_partition_x);
    assert(min_partition_time <= max_partition_time);

    // fprintf(stderr, "computing interacting for part = (%u, %u, %u)\n",
    //         partition_time, partition_y, partition_x);

    unsigned count_interacting_partitions = 0;
    for (unsigned t = min_partition_time; t <= max_partition_time; t++) {
        for (unsigned r = min_partition_y; r <= max_partition_y; r++) {
            for (unsigned c = min_partition_x; c <= max_partition_x; c++) {
                // fprintf(stderr, "part = (%u, %u, %u) interacts with (%u, %u, "
                //         "%u)\n", partition_time, partition_y, partition_x,
                //         t, r, c);
                const unsigned part = t * n_y_partition * n_x_partition +
                    r * n_x_partition + c;
                if (count_interacting_partitions >= interacting_partitions_capacity) {
                    fprintf(stderr, "time = (%d, %d) y = (%d, %d) x = (%d, %d) "
                            "current count = %u, capacity = %u\n",
                            min_partition_time, max_partition_time,
                            min_partition_y, max_partition_y,
                            min_partition_x, max_partition_x,
                            count_interacting_partitions,
                            interacting_partitions_capacity);
                    abort();
                }
                interacting_partitions[count_interacting_partitions++] = part;
            }
        }
    }
    *n_interacting_partitions = count_interacting_partitions;
}

/*
 * Callback used by the HOOVER runtime to check if this PE can abort out of the
 * simulation.
 */
void update_coupled_val(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_vertex_t *out_coupled_metric) {
    // Abort if all of my member vertices are infected
    size_t nset = 0;
    hvr_vertex_t *vert = hvr_vertex_iter_next(iter);
    while (vert) {
        if ((int)hvr_vertex_get(TIME_STEP, vert, ctx) == max_num_timesteps - 1) {
            if (hvr_vertex_get(INFECTED, vert, ctx) > 0.0) {
                nset++;
            }
        }
        vert = hvr_vertex_iter_next(iter);
    }

    hvr_vertex_set(0, (double)nset, out_coupled_metric, ctx);
    hvr_vertex_set(1, (double)n_local_actors, out_coupled_metric, ctx);
}

int should_terminate(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_vertex_t *local_coupled_metric, // coupled_pes[shmem_my_pe()]
        hvr_vertex_t *all_coupled_metrics, // Each PE's val
        hvr_vertex_t *global_coupled_metric, // Sum reduction of coupled_pes
        hvr_set_t *coupled_pes, // An array of size npes, with each PE's val
        int n_coupled_pes,
        int *updates_on_this_iter, // An array of size npes, the number of vertex updates done on each coupled PE
        hvr_set_t *terminated_coupled_pes) {
    int sum_updates = 0;
    for (int i = 0; i < ctx->npes; i++) {
        sum_updates += updates_on_this_iter[i];
    }

    unsigned local_nset = (unsigned)hvr_vertex_get(0, local_coupled_metric,
            ctx);
    unsigned global_nset = (unsigned)hvr_vertex_get(0,
            global_coupled_metric, ctx);
    unsigned global_nverts = (unsigned)hvr_vertex_get(1,
            global_coupled_metric, ctx);
    if (local_nset > 0) {
        printf("PE %d - iter %lu - local set %u / %lu (%.2f%%)- # coupled = %d "
                "- global set %u / %u (%.2f%%) - %u vertex updates globally\n",
                pe,
                (uint64_t)ctx->iter,
                local_nset, n_local_actors,
                100.0 * (double)local_nset / (double)n_local_actors,
                n_coupled_pes,
                global_nset,
                global_nverts,
                100.0 * (double)global_nset / (double)global_nverts,
                sum_updates);
    }

    int aborting = 0;
    if (n_coupled_pes == ctx->npes) {

        if (sum_updates == 0) {
            int ninfected = (int)hvr_vertex_get(0, local_coupled_metric, ctx);
            double percent_infected = (double)ninfected /
                (double)n_local_actors;
            double global_percent_infected = 100.0 * (double)global_nset /
                (double)global_nverts;
            printf("PE %d leaving the simulation, %% local infected = %f "
                    "(%d / %lu), %% global infected = %f (%u / %u)\n",
                    shmem_my_pe(),
                    100.0 * percent_infected,
                    ninfected, n_local_actors,
                    global_percent_infected, global_nset, global_nverts);
            aborting = 1;
        }
    }

    return aborting;
}

static int safe_fread(double *buf, size_t n_to_read, FILE *fp) {
    size_t err = fread(buf, sizeof(*buf), n_to_read, fp);
    if (err == n_to_read) return 1;
    else {
        assert(feof(fp));
        return 0;
    }
}

int main(int argc, char **argv) {
    hvr_ctx_t hvr_ctx;

    if (argc != 9) {
        fprintf(stderr, "usage: %s <cell-dim-y> <cell-dim-x> "
                "<n-cells-y> <n-cells-x> "
                "<max-num-timesteps> <infection-radius> "
                "<time-limit> <input-file>\n",
                argv[0]);
        return 1;
    }

    cell_dim_y = atof(argv[1]);
    cell_dim_x = atof(argv[2]);
    n_cells_y = atoi(argv[3]);
    n_cells_x = atoi(argv[4]);
    max_num_timesteps = atoi(argv[5]);
    infection_radius = atof(argv[6]);
    int time_limit = atoi(argv[7]);
    char *input_filename = argv[8];

    n_time_partition = max_num_timesteps;
    n_y_partition = 200;
    n_x_partition = 200;
    hvr_partition_t npartitions = n_time_partition * n_y_partition *
        n_x_partition;

#ifdef MULTITHREADED
#pragma omp parallel
#pragma omp single
    nthreads = omp_get_num_threads();
#endif

    const double global_x_dim = (double)n_cells_x * cell_dim_x;
    const double global_y_dim = (double)n_cells_y * cell_dim_y;

    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        p_sync[i] = SHMEM_SYNC_VALUE;
    }

#ifdef MULTITHREADED
    int provided = shmemx_init_thread(SHMEM_THREAD_MULTIPLE);
    assert(provided == SHMEM_THREAD_MULTIPLE);
#else
    shmem_init();
#endif
    pe = shmem_my_pe();
    npes = shmem_n_pes();
    if (pe == 0) {
        fprintf(stderr, "Running with %d PEs\n", npes);
#ifdef MULTITHREADED
        fprintf(stderr, "Running with %d OMP threads\n", nthreads);
#endif
    }
    assert(npes == n_cells_y * n_cells_x);

    hvr_ctx_create(&hvr_ctx);

    const double my_cell_start_x = PE_COL_CELL_START(pe);
    const double my_cell_end_x = my_cell_start_x + cell_dim_x;
    const double my_cell_start_y = PE_ROW_CELL_START(pe);
    const double my_cell_end_y = my_cell_start_y + cell_dim_y;

    // unsigned long long start_count_local = hvr_current_time_us();
    FILE *input = fopen(input_filename, "rb");
    assert(input);
    /*
     * 0: actor id
     * 1: px
     * 2: py
     * 3: dst_x
     * 4: dst_y
     * 5: infected
     */
    double buf[6];
    while (safe_fread(buf, 6, input)) {
        if (buf[1] >= my_cell_start_x && buf[1] < my_cell_end_x &&
                buf[2] >= my_cell_start_y && buf[2] < my_cell_end_y) {
            n_local_actors++;
        }
    }
    fclose(input);
    // unsigned long long elapsed_count_local = hvr_current_time_us() -
    //     start_count_local;
    // fprintf(stderr, "PE %d took %f ms to count local, %lu local actors\n", pe,
    //         (double)elapsed_count_local / 1000.0, n_local_actors);

    // unsigned long long start_pop_local = hvr_current_time_us();
    hvr_vertex_t *actors = hvr_vertex_create_n(n_local_actors, hvr_ctx);

    size_t index = 0;
    input = fopen(input_filename, "rb");
    while (safe_fread(buf, 6, input)) {
        if (buf[1] >= my_cell_start_x && buf[1] < my_cell_end_x &&
                buf[2] >= my_cell_start_y && buf[2] < my_cell_end_y) {
            double actor_id = buf[0];
            double x = buf[1];
            double y = buf[2];
            double dst_x = buf[3];
            double dst_y = buf[4];
            double infected = buf[5];

            if (infected > 0.0) {
                fprintf(stderr, "PE %d - actor %lu infected\n", pe,
                        (unsigned long)actor_id);
            }

            hvr_vertex_set(PX, x, &actors[index], hvr_ctx);
            hvr_vertex_set(PY, y, &actors[index], hvr_ctx);
            hvr_vertex_set(INFECTED, infected, &actors[index], hvr_ctx);
            hvr_vertex_set(DST_X, dst_x, &actors[index], hvr_ctx);
            hvr_vertex_set(DST_Y, dst_y, &actors[index], hvr_ctx);
            hvr_vertex_set(TIME_STEP, 0, &actors[index], hvr_ctx);
            hvr_vertex_set(ACTOR_ID, actor_id, &actors[index], hvr_ctx);
            hvr_vertex_set(NEXT_CREATED, 0, &actors[index], hvr_ctx);

            index++;
        }
    }
    assert(index == n_local_actors);
    fclose(input);

    size_t *actors_per_pe = (size_t *)shmem_malloc(npes * sizeof(*actors_per_pe));
    assert(actors_per_pe);
    for (int p = 0; p < npes; p++) {
        shmem_putmem(actors_per_pe + pe, &n_local_actors,
                sizeof(n_local_actors), p);
    }
    shmem_barrier_all();
    for (int p = 0; p < npes; p++) {
        total_n_actors += actors_per_pe[p];
    }

    // unsigned long long elapsed_pop_local = hvr_current_time_us() -
    //     start_pop_local;
    // fprintf(stderr, "PE %d took %f ms to populate local\n", pe,
    //         (double)elapsed_pop_local / 1000.0);

    if (pe == 0) {
        fprintf(stderr, "Running for at most %d seconds\n", time_limit);
        fprintf(stderr, "Using %u partitions (%u time partitions * %u y "
                "partitions * %u x partitions)\n", npartitions,
                n_time_partition, n_y_partition, n_x_partition);
        fprintf(stderr, "Loading input from %s\n", input_filename);
        fprintf(stderr, "~%lu actors per PE x %d PEs x %u timesteps = %lu "
                "vertices across all PEs (~%f vertices per PE)\n",
                n_local_actors,
                npes,
                max_num_timesteps,
                total_n_actors * max_num_timesteps,
                (double)(total_n_actors * max_num_timesteps) / (double)npes);
        fprintf(stderr, "%d timesteps, y=%f x x=%f grid\n", max_num_timesteps,
                global_y_dim, global_x_dim);
    }
    shmem_barrier_all();

    hvr_init(npartitions,
            update_metadata,
            might_interact,
            update_coupled_val,
            actor_to_partition,
            NULL, // start_time_step
            should_have_edge,
            should_terminate,
            time_limit, // max_elapsed_seconds
            1, // max_graph_traverse_depth
            hvr_ctx);

    const long long start_time = hvr_current_time_us();
    hvr_body(hvr_ctx);
    const long long elapsed_time = hvr_current_time_us() - start_time;

    long long *elapsed_times = (long long *)shmem_malloc(
            npes * sizeof(*elapsed_times));
    assert(elapsed_times);
    shmem_longlong_put(elapsed_times + pe, &elapsed_time, 1, 0);
    shmem_barrier_all();

    long long total_time = 0;
    for (int p = 0; p < npes; p++) {
        total_time += elapsed_times[p];
    }
    long long max_elapsed = 0;
    for (int p = 0; p < npes; p++) {
        if (elapsed_times[p] > max_elapsed) {
            max_elapsed = elapsed_times[p];
        }
    }

    uint64_t *msgs_sent = (uint64_t *)shmem_malloc(
            npes * sizeof(*msgs_sent));
    assert(msgs_sent);
    uint64_t *msgs_recv = (uint64_t *)shmem_malloc(
            npes * sizeof(*msgs_recv));
    assert(msgs_recv);
    int *modeled_timesteps = (int *)shmem_malloc(
            npes * sizeof(*modeled_timesteps));
    assert(modeled_timesteps);

    shmem_int_put(modeled_timesteps + pe, &max_modeled_timestep, 1, 0);
    shmem_uint64_put(msgs_sent + pe, &(hvr_ctx->total_vertex_msgs_sent), 1, 0);
    shmem_uint64_put(msgs_recv + pe, &(hvr_ctx->total_vertex_msgs_recvd), 1, 0);

    shmem_barrier_all();

    uint64_t total_msgs_sent = msgs_sent[0];
    uint64_t total_msgs_recv = msgs_recv[0];
    int all_max_modeled_timestep = modeled_timesteps[0];
    for (int p = 1; p < npes; p++) {
        if (modeled_timesteps[p] < all_max_modeled_timestep) {
            all_max_modeled_timestep = modeled_timesteps[p];
        }
        total_msgs_sent += msgs_sent[p];
        total_msgs_recv += msgs_recv[p];
    }

    if (pe == 0) {
        printf("%d PEs, %d timesteps, infection radius of %f, total CPU time = "
                "%f ms, max elapsed = %f ms, ~%lu actors per PE, completed %d "
                "iters\n", npes, max_num_timesteps, infection_radius,
                (double)total_time / 1000.0, (double)max_elapsed / 1000.0,
                n_local_actors, hvr_ctx->iter);
        printf("In total %lu msgs sent, %lu msgs received\n", total_msgs_sent,
                total_msgs_recv);
        printf("Max modeled timestep across all PEs = %d, # vertices on PE 0 = "
                "%lu\n", all_max_modeled_timestep, hvr_n_allocated(hvr_ctx));
        for (int p = 0; p < npes; p++) {
            printf("  PE %d got to timestep %d\n", p, modeled_timesteps[p]);
        }
    }

    hvr_finalize(hvr_ctx);

    shmem_finalize();

    return 0;
}
