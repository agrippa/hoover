/* For license: see LICENSE.txt file at top-level */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <shmem.h>
#include <math.h>
#include <string.h>

#include <hoover.h>

#define TIME_PARTITION_DIM 270
#define Y_PARTITION_DIM 150
#define X_PARTITION_DIM 150

#define PX 0
#define PY 1
#define INFECTED 2
#define HOME_X 3
#define HOME_Y 4
#define DST_X 5
#define DST_Y 6
#define TIME_STEP 7
#define ACTOR_ID 8

#define MAX_V 0.5

#define PORTAL_CAPTURE_RADIUS 5.0
#define PE_ROW(this_pe) ((this_pe) / pe_cols)
#define PE_COL(this_pe) ((this_pe) % pe_cols)
#define PE_ROW_CELL_START(this_pe) ((double)PE_ROW(this_pe) * cell_dim)
#define PE_COL_CELL_START(this_pe) ((double)PE_COL(this_pe) * cell_dim)
#define CELL_ROW(y_coord) ((y_coord) / cell_dim)
#define CELL_COL(x_coord) ((x_coord) / cell_dim)
#define CELL_INDEX(cell_row, cell_col) ((cell_row) * pe_cols + (cell_col))

#define MAX_DST_DELTA 10.0

typedef struct _portal_t {
    int pes[2];
    struct {
        double x, y;
    } locations[2];
} portal_t;

static int npes, pe;
static int pe_rows, pe_cols;
static portal_t *portals = NULL;
static int n_global_portals = 0;
static int actors_per_cell;
static double cell_dim;
long long total_time = 0;
long long max_elapsed = 0;
long long elapsed_time = 0;
static double max_delta_velocity;
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
long p_sync[SHMEM_REDUCE_SYNC_SIZE];

static double distance(double x1, double y1, double x2, double y2) {
    double deltax = x2 - x1;
    double deltay = y2 - y1;
    return sqrt((deltay * deltay) + (deltax * deltax));
}

static double random_double_in_range(double min_val_inclusive,
        double max_val_exclusive) {
    return min_val_inclusive + (((double)rand() / (double)RAND_MAX) *
            (max_val_exclusive - min_val_inclusive));
}

static int random_int_in_range(int max_val) {
    return (rand() % max_val);
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

    if (base_id == neighbor_id && base_time + 1 == neighbor_time) {
        return DIRECTED_OUT;
    }

    if (base_time > 0 && base_time - 1 == neighbor_time) {
        double deltax = hvr_vertex_get(PX, neighbor, ctx) -
            hvr_vertex_get(PX, base, ctx);
        double deltay = hvr_vertex_get(PY, neighbor, ctx) -
            hvr_vertex_get(PY, base, ctx);
        if (deltax * deltax + deltay * deltay <=
                infection_radius * infection_radius) {
            return DIRECTED_IN;
        }
    }

    return NO_EDGE;
}

hvr_partition_t actor_to_partition(hvr_vertex_t *actor, hvr_ctx_t ctx) {
    const double timestep = hvr_vertex_get(TIME_STEP, actor, ctx);
    const double y = hvr_vertex_get(PY, actor, ctx);
    const double x = hvr_vertex_get(PX, actor, ctx);

    const double global_y_dim = (double)pe_rows * cell_dim;
    const double global_x_dim = (double)pe_cols * cell_dim;

    assert((int)timestep < max_num_timesteps);
    assert(x < global_x_dim);
    assert(y < global_y_dim);

    const double partition_time_step = (double)max_num_timesteps /
        (double)TIME_PARTITION_DIM;
    const double partition_y_dim = global_y_dim / (double)Y_PARTITION_DIM;
    const double partition_x_dim = global_x_dim / (double)X_PARTITION_DIM;

    const int time_step_partition = (int)(timestep / partition_time_step);
    const int y_partition = (int)(y / partition_y_dim);
    const int x_partition = (int)(x / partition_x_dim);

    assert(time_step_partition < TIME_PARTITION_DIM);
    assert(x_partition < X_PARTITION_DIM);
    assert(y_partition < Y_PARTITION_DIM);

    return time_step_partition * Y_PARTITION_DIM * X_PARTITION_DIM +
        y_partition * X_PARTITION_DIM + x_partition;
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
    hvr_vertex_id_t *neighbors = NULL;
    hvr_edge_type_t *directions = NULL;
    size_t n_neighbors = 0;
    hvr_get_neighbors(vertex, &neighbors, &directions, &n_neighbors, ctx);

    /*
     * Scan over in edges to find the same actor on the previous timestep. Use
     * its infection state as our baseline (i.e. if it is uninfected, start with
     * the assumption that we are uninfected).
     */
    hvr_vertex_t *prev = NULL;
    if ((int)hvr_vertex_get(TIME_STEP, vertex, ctx) > 0) {
        for (int i = 0; i < n_neighbors; i++) {
            if (directions[i] == DIRECTED_IN) {
                hvr_vertex_t *neighbor = hvr_get_vertex(neighbors[i], ctx);
                if ((int)hvr_vertex_get(ACTOR_ID, neighbor, ctx) ==
                        (int)hvr_vertex_get(ACTOR_ID, vertex, ctx)) {
                    assert(prev == NULL);
                    assert((int)hvr_vertex_get(TIME_STEP, neighbor, ctx) ==
                            (int)hvr_vertex_get(TIME_STEP, vertex, ctx) - 1);
                    prev = neighbor;
                }
            }
        }
        assert(prev);
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
            if (directions[i] == DIRECTED_IN) {
                hvr_vertex_t *neighbor = hvr_get_vertex(neighbors[i], ctx);
                if ((int)hvr_vertex_get(ACTOR_ID, neighbor, ctx) !=
                        (int)hvr_vertex_get(ACTOR_ID, vertex, ctx)) {
                    assert((int)hvr_vertex_get(TIME_STEP, neighbor, ctx) ==
                            (int)hvr_vertex_get(TIME_STEP, vertex, ctx) - 1);
                    int is_infected = hvr_vertex_get(INFECTED, neighbor, ctx);
                    if (is_infected) {
                        const int infected_by = hvr_vertex_get_owning_pe(neighbor);
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
        const double global_x_dim = (double)pe_cols * cell_dim;
        const double global_y_dim = (double)pe_rows * cell_dim;
        const double expected_max_radius = sqrt(
                MAX_DST_DELTA * MAX_DST_DELTA + MAX_DST_DELTA * MAX_DST_DELTA);

        double dst_x = hvr_vertex_get(DST_X, prev, ctx);
        double dst_y = hvr_vertex_get(DST_Y, prev, ctx);
        double p_x = hvr_vertex_get(PX, prev, ctx);
        double p_y = hvr_vertex_get(PY, prev, ctx);
        const double home_x = hvr_vertex_get(HOME_X, prev, ctx);
        const double home_y = hvr_vertex_get(HOME_Y, prev, ctx);

        const double home_distance = distance(home_x, home_y, p_x, p_y);
        assert(home_distance <= expected_max_radius);

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

        for (int p = 0; p < n_global_portals; p++) {
            if (distance(new_x, new_y, portals[p].locations[0].x,
                        portals[p].locations[0].y) < PORTAL_CAPTURE_RADIUS) {
                new_x = portals[p].locations[1].x +
                    random_double_in_range(-10.0, 10.0);
                new_y = portals[p].locations[1].y +
                    random_double_in_range(-10.0, 10.0);
                break;
            }

            if (distance(new_x, new_y, portals[p].locations[1].x,
                        portals[p].locations[1].y) < PORTAL_CAPTURE_RADIUS) {
                new_x = portals[p].locations[0].x +
                    random_double_in_range(-10.0, 10.0);
                new_y = portals[p].locations[0].y +
                    random_double_in_range(-10.0, 10.0);
                break;
            }
        }

        if (new_x >= global_x_dim) new_x -= global_x_dim;
        if (new_y >= global_y_dim) new_y -= global_y_dim;
        if (new_x < 0.0) new_x += global_x_dim;
        if (new_y < 0.0) new_y += global_y_dim;

        assert(new_x >= 0.0 && new_x < global_x_dim);
        assert(new_y >= 0.0 && new_y < global_y_dim);

        hvr_vertex_set(PX, new_x, vertex, ctx);
        hvr_vertex_set(PY, new_y, vertex, ctx);
    }
        free(neighbors); free(directions);
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
    const double global_x_dim = (double)pe_cols * cell_dim;
    const double global_y_dim = (double)pe_rows * cell_dim;

    // Dimension of each partition in the row, column, time directions
    double y_dim = global_y_dim / (double)Y_PARTITION_DIM;
    double x_dim = global_x_dim / (double)X_PARTITION_DIM;
    double time_dim = (double)max_num_timesteps / (double)TIME_PARTITION_DIM;

    /*
     * For the given partition, the (time, row, column) coordinate of this
     * partition in a 2D space.
     */
    const int partition_time = partition / (Y_PARTITION_DIM * X_PARTITION_DIM);
    const int partition_y = (partition / X_PARTITION_DIM) % Y_PARTITION_DIM;
    const int partition_x = partition % X_PARTITION_DIM;

    double min_time = (double)partition_time * time_dim;
    double max_time = min_time + time_dim;

    // Get bounding box of partition in the grid coordinate system
    double min_y = (double)partition_y * y_dim;
    double max_y = min_y + y_dim;
    double min_x = (double)partition_x * x_dim;
    double max_x = min_x + x_dim;

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

    if (max_y >= (double)global_y_dim) max_partition_y = Y_PARTITION_DIM - 1;
    else max_partition_y = (int)(max_y / y_dim);

    if (max_x >= (double)global_x_dim) max_partition_x = X_PARTITION_DIM - 1;
    else max_partition_x = (int)(max_x / x_dim);
 
    if (max_time >= (double)max_num_timesteps) max_partition_time = TIME_PARTITION_DIM - 1;
    else max_partition_time = (int)(max_time / time_dim);

    assert(min_partition_y <= max_partition_y);
    assert(min_partition_x <= max_partition_x);
    assert(min_partition_time <= max_partition_time);

    unsigned count_interacting_partitions = 0;
    for (int t = min_partition_time; t <= max_partition_time; t++) {
        for (int r = min_partition_y; r <= max_partition_y; r++) {
            for (int c = min_partition_x; c <= max_partition_x; c++) {
                const int part = t * Y_PARTITION_DIM * X_PARTITION_DIM +
                    r * X_PARTITION_DIM + c;
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
                assert(count_interacting_partitions <
                        interacting_partitions_capacity);
                interacting_partitions[count_interacting_partitions++] = part;
            }
        }
    }
    *n_interacting_partitions = count_interacting_partitions;
}

static unsigned long long last_time = 0;

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
        if ((int)hvr_vertex_get(TIME_STEP, vert, ctx) == max_num_timesteps - 1 &&
                hvr_vertex_get(INFECTED, vert, ctx) > 0.0) {
            nset++;
        }
        vert = hvr_vertex_iter_next(iter);
    }

    unsigned long long this_time = hvr_current_time_us();
    if (nset > 0) {
        printf("PE %d - iter %lu - set %lu / %u\n", pe, (uint64_t)ctx->iter,
                nset, actors_per_cell);
    }
    last_time = this_time;

    hvr_vertex_set(0, (double)nset, out_coupled_metric, ctx);
}

int should_terminate(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_vertex_t *local_coupled_metric, hvr_vertex_t *global_coupled_metric,
        hvr_set_t *coupled_pes, int n_coupled_pes) {
    if ((int)hvr_vertex_get(0, local_coupled_metric, ctx) == actors_per_cell) {
        // return 0;
        printf("PE %d leaving the simulation\n", shmem_my_pe());
        return 1;
    } else {
        return 0;
    }
}

int main(int argc, char **argv) {
    hvr_ctx_t hvr_ctx;

    if (argc != 11) {
        fprintf(stderr, "usage: %s <cell-dim> <# global portals> "
                "<actors per cell> <pe-rows> <pe-cols> <n-initial-infected> "
                "<max-num-timesteps> <infection-radius> <max-delta-velocity> "
                "<time-limit>\n",
                argv[0]);
        return 1;
    }

    cell_dim = atof(argv[1]);
    n_global_portals = atoi(argv[2]);
    actors_per_cell = atoi(argv[3]);
    pe_rows = atoi(argv[4]);
    pe_cols = atoi(argv[5]);
    const int n_initial_infected = atoi(argv[6]);
    max_num_timesteps = atoi(argv[7]);
    infection_radius = atof(argv[8]);
    max_delta_velocity = atof(argv[9]);
    int time_limit = atoi(argv[10]);

    const double global_x_dim = (double)pe_cols * cell_dim;
    const double global_y_dim = (double)pe_rows * cell_dim;

    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        p_sync[i] = SHMEM_SYNC_VALUE;
    }

    shmem_init();
    pe = shmem_my_pe();
    npes = shmem_n_pes();
    assert(npes == pe_rows * pe_cols);

    hvr_ctx_create(&hvr_ctx);

    srand(123 + pe);

    /*
     * Every cell/PE allows direct transitions of actors out of it to any of its
     * eight neighboring PEs. Each cell/PE also optionally contains what we call
     * "portals" which are essentially points in the cell which are connected to
     * points in farther away cells to model longer distance interactions (e.g.
     * plane, train routes). If an actor hits one of these portals, they are
     * "teleported" to the other end of the portal.
     */
    portals = (portal_t *)shmem_malloc(n_global_portals * sizeof(*portals));
    assert((n_global_portals == 0) || portals);
    if (pe == 0) {
        fprintf(stderr, "Creating %d portals\n", n_global_portals);
        fprintf(stderr, "%d actors per cell, %d actors in total\n",
                actors_per_cell, actors_per_cell * npes);
        fprintf(stderr, "%d timesteps, %f x %f grid\n", max_num_timesteps,
                global_y_dim, global_x_dim);
        for (int p = 0; p < n_global_portals; p++) {
            const int pe0 = (rand() % npes);
            const int pe1 = (rand() % npes);

            portals[p].pes[0] = pe0;
            portals[p].pes[1] = pe1;
            portals[p].locations[0].x = random_double_in_range(
                    PE_COL_CELL_START(pe0), PE_COL_CELL_START(pe0) + cell_dim);
            portals[p].locations[0].y = random_double_in_range(
                    PE_ROW_CELL_START(pe0), PE_ROW_CELL_START(pe0) + cell_dim);
            portals[p].locations[1].x = random_double_in_range(
                    PE_COL_CELL_START(pe1), PE_COL_CELL_START(pe1) + cell_dim);
            portals[p].locations[1].y = random_double_in_range(
                    PE_ROW_CELL_START(pe1), PE_ROW_CELL_START(pe1) + cell_dim);
            fprintf(stderr, "  %d - %d : (%f, %f) - (%f, %f)\n", pe0, pe1,
                    portals[p].locations[0].x, portals[p].locations[0].y,
                    portals[p].locations[1].x, portals[p].locations[1].y);
        }
    }
    assert((n_global_portals * sizeof(*portals)) % 4 == 0);
    if (n_global_portals > 0) {
        shmem_broadcast32(portals, portals,
                (n_global_portals * sizeof(*portals)) / 4, 0, 0, 0, npes,
                p_sync);
    }
    shmem_barrier_all();

    // Seed the initial infected actors.
    int *initial_infected = (int *)shmem_malloc(
            n_initial_infected * sizeof(*initial_infected));
    assert(initial_infected);
    if (pe == 0) {
        for (int i = 0; i < n_initial_infected; i++) {
            initial_infected[i] = random_int_in_range(npes * actors_per_cell);
        }
    }
    assert(sizeof(int) == 4);
    shmem_broadcast32(initial_infected, initial_infected, n_initial_infected, 0,
            0, 0, npes, p_sync);
    shmem_barrier_all();

    // Seed the location of local actors.
    hvr_vertex_t *actors = hvr_vertex_create_n(
            actors_per_cell * max_num_timesteps, hvr_ctx);
    for (int a = 0; a < actors_per_cell; a++) {
        const double x = random_double_in_range(PE_COL_CELL_START(pe),
                PE_COL_CELL_START(pe) + cell_dim);
        const double y = random_double_in_range(PE_ROW_CELL_START(pe),
                PE_ROW_CELL_START(pe) + cell_dim);

        double dst_x = x + random_double_in_range(-MAX_DST_DELTA,
                MAX_DST_DELTA);
        double dst_y = y + random_double_in_range(-MAX_DST_DELTA,
                MAX_DST_DELTA);
        if (dst_x > global_x_dim) dst_x = global_x_dim - 1.0;
        if (dst_y > global_y_dim) dst_y = global_y_dim - 1.0;
        if (dst_x < 0.0) dst_x = 0.0;
        if (dst_y < 0.0) dst_y = 0.0;

        int is_infected = 0;
        for (int i = 0; i < n_initial_infected; i++) {
            int owner_pe = initial_infected[i] / actors_per_cell;
            if (owner_pe == pe) {
                int local_offset = initial_infected[i] % actors_per_cell;
                assert(local_offset < actors_per_cell);
                if (local_offset == a)  {
                    is_infected = 1;
                    break;
                }
            }
        }

        if (is_infected) {
            fprintf(stderr, "PE %d - local offset %d infected\n", pe, a);
        }

        for (int t = 0; t < max_num_timesteps; t++) {
            int index = a * max_num_timesteps + t;

            hvr_vertex_set(PX, x, &actors[index], hvr_ctx);
            hvr_vertex_set(PY, y, &actors[index], hvr_ctx);

            if (is_infected) {
                hvr_vertex_set(INFECTED, 1, &actors[index], hvr_ctx);
            } else {
                hvr_vertex_set(INFECTED, 0, &actors[index], hvr_ctx);
            }

            hvr_vertex_set(HOME_X, x, &actors[index], hvr_ctx);
            hvr_vertex_set(HOME_Y, y, &actors[index], hvr_ctx);

            hvr_vertex_set(DST_X, dst_x, &actors[index], hvr_ctx);
            hvr_vertex_set(DST_Y, dst_y, &actors[index], hvr_ctx);

            hvr_vertex_set(TIME_STEP, t, &actors[index], hvr_ctx);
            hvr_vertex_set(ACTOR_ID, shmem_my_pe() * actors_per_cell + a,
                    &actors[index], hvr_ctx);
        }
    }

    hvr_init(TIME_PARTITION_DIM * Y_PARTITION_DIM * X_PARTITION_DIM,
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
    elapsed_time = hvr_current_time_us() - start_time;

    shmem_longlong_sum_to_all(&total_time, &elapsed_time, 1, 0, 0, npes, p_wrk,
            p_sync);
    shmem_barrier_all();
    shmem_longlong_max_to_all(&max_elapsed, &elapsed_time, 1, 0, 0, npes, p_wrk,
            p_sync);
    shmem_barrier_all();

    if (pe == 0) {
        printf("%d PEs, %d timesteps, infection radius of %f, total CPU time = "
                "%f ms, max elapsed = %f ms, ~%u actors per PE, completed %d "
                "iters\n", npes, max_num_timesteps, infection_radius,
                (double)total_time / 1000.0, (double)max_elapsed / 1000.0,
                actors_per_cell, hvr_ctx->iter);
    }

    hvr_finalize(hvr_ctx);

    shmem_finalize();

    return 0;
}
