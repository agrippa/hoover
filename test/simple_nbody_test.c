/* For license: see LICENSE.txt file at top-level */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <shmem.h>
#include <math.h>
#include <string.h>

#include <hoover.h>

#define MAX_ACCEL 1.0

#define TIMESTEP 0
#define PARTICLE_ID 1
#define PX 2
#define PY 3
#define VX 4
#define VY 5
#define NEXT_CREATED 6
#define NEXT_ID 7
#define PREV_PX 9
#define PREV_PY 10
#define PREV_VX 11
#define PREV_VY 12

static int npes, pe;
static double distance_threshold = 10.0;

static const double domain_dim = 100.0;
static int timesteps = 20;
static const int partitions_per_dim = 10;

static FILE *fp = NULL;

#define PARTITION_DIM (domain_dim / partitions_per_dim)

#define TIMESTEP_PARTITION(my_part) ((hvr_partition_t)((my_part) / \
            (partitions_per_dim * partitions_per_dim)))
#define Y_PARTITION(my_part) ((hvr_partition_t)(((my_part) / \
                partitions_per_dim) % partitions_per_dim))
#define X_PARTITION(my_part) ((hvr_partition_t)((my_part) % partitions_per_dim))

long long p_wrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
int int_p_wrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
long p_sync[SHMEM_REDUCE_SYNC_SIZE];

static inline double distance(double x1, double y1, double x2, double y2) {
    double deltax = x2 - x1;
    double deltay = y2 - y1;
    return sqrt((deltay * deltay) + (deltax * deltax));
}

hvr_edge_type_t should_have_edge(hvr_vertex_t *base, hvr_vertex_t *neighbor,
        hvr_ctx_t ctx) {
    int base_time = (int)hvr_vertex_get(TIMESTEP, base, ctx);
    int neighbor_time = (int)hvr_vertex_get(TIMESTEP, neighbor, ctx);

    if (abs(base_time - neighbor_time) == 1) {
        double base_x = hvr_vertex_get(PREV_PX, base, ctx);
        double base_y = hvr_vertex_get(PREV_PY, base, ctx);
        double neighbor_x = hvr_vertex_get(PREV_PX, neighbor, ctx);
        double neighbor_y = hvr_vertex_get(PREV_PY, neighbor, ctx);
        if (distance(base_x, base_y, neighbor_x, neighbor_y) <
                distance_threshold) {
            if (base_time < neighbor_time) {
                return DIRECTED_OUT;
            } else {
                return DIRECTED_IN;
            }
        }
    }
    return NO_EDGE;
}

hvr_partition_t actor_to_partition(hvr_vertex_t *actor, hvr_ctx_t ctx) {
    const int timestep = hvr_vertex_get(TIMESTEP, actor, ctx);
    const double y = hvr_vertex_get(PREV_PY, actor, ctx);
    const double x = hvr_vertex_get(PREV_PX, actor, ctx);

    const hvr_partition_t timestep_partition = timestep;
    const hvr_partition_t y_partition = (hvr_partition_t)(y / PARTITION_DIM);
    const hvr_partition_t x_partition = (hvr_partition_t)(x / PARTITION_DIM);

    assert(timestep_partition < timesteps);
    assert(y_partition < partitions_per_dim);
    assert(x_partition < partitions_per_dim);

    return timestep_partition * partitions_per_dim * partitions_per_dim +
        y_partition * partitions_per_dim + x_partition;
}

static void compute_accel(double px, double py, int timestep, int id,
        hvr_vertex_t **neighbors, hvr_edge_type_t *dirs, int n_neighbors,
        double *accel_x_out, double *accel_y_out, hvr_ctx_t ctx) {
    double accel_x = 0.0;
    double accel_y = 0.0;
    double mass = 1.0;
    double max_force = 0.1;
    for (int i = 0; i < n_neighbors; i++) {
        hvr_vertex_t *neighbor = neighbors[i];
        if ((int)hvr_vertex_get(TIMESTEP, neighbor, ctx) == timestep - 1 &&
                (int)hvr_vertex_get(PARTICLE_ID, neighbor, ctx) != id) {
            double x_delta = hvr_vertex_get(PX, neighbor, ctx) - px;
            double y_delta = hvr_vertex_get(PY, neighbor, ctx) - py;
            assert(x_delta != 0.0 && y_delta != 0.0);

            double x_force = 1.0 / x_delta;
            double y_force = 1.0 / y_delta;
            assert(!isinf(x_force) && !isinf(y_force));

            accel_x += x_force;
            accel_y += y_force;
        }
    }

    double norm = distance(0.0, 0.0, accel_x, accel_y);
    if (norm > MAX_ACCEL) {
        double new_accel_x = MAX_ACCEL * cos(atan(accel_y / accel_x));
        double new_accel_y = MAX_ACCEL * sin(atan(accel_y / accel_x));

        double sign_x = (accel_x >= 0.0 ? 1.0 : -1.0);
        double sign_y = (accel_y >= 0.0 ? 1.0 : -1.0);

        accel_x = sign_x * fabs(new_accel_x);
        accel_y = sign_y * fabs(new_accel_y);
    }

    *accel_x_out = accel_x;
    *accel_y_out = accel_y;
}

void update_metadata(hvr_vertex_t *vertex, hvr_set_t *couple_with,
        hvr_ctx_t ctx) {
    int vertex_id = (int)hvr_vertex_get(PARTICLE_ID, vertex, ctx);
    int vertex_timestep = (int)hvr_vertex_get(TIMESTEP, vertex, ctx);

    double px = hvr_vertex_get(PX, vertex, ctx);
    double py = hvr_vertex_get(PY, vertex, ctx);
    double vx = hvr_vertex_get(VX, vertex, ctx);
    double vy = hvr_vertex_get(VY, vertex, ctx);

    hvr_vertex_t **verts;
    hvr_edge_type_t *dirs;
    int n_neighbors = hvr_get_neighbors(vertex, &verts, &dirs, ctx);

    hvr_vertex_t prev;
    int have_msg = hvr_poll_msg(vertex, &prev, ctx);
    if (have_msg) {
        // Messages are sorted most recent to least recent
        assert((int)hvr_vertex_get(TIMESTEP, &prev, ctx) == vertex_timestep - 1);
        assert((int)hvr_vertex_get(PARTICLE_ID, &prev, ctx) == vertex_id);

        hvr_vertex_set(HAVE_PREV, 1, vertex, ctx);
        hvr_vertex_set(PREV_PX, hvr_vertex_get(PX, &prev, ctx), vertex, ctx);
        hvr_vertex_set(PREV_PY, hvr_vertex_get(PY, &prev, ctx), vertex, ctx);
        hvr_vertex_set(PREV_VX, hvr_vertex_get(VX, &prev, ctx), vertex, ctx);
        hvr_vertex_set(PREV_VY, hvr_vertex_get(VY, &prev, ctx), vertex, ctx);

        // Flush less recent messages to this vertex
        do {
            have_msg = hvr_poll_msg(vertex, &prev, ctx);
        } while (have_msg);
    }

    fprintf(stderr, "PE %d updating particle %d on iter %d\n", ctx->pe,
            vertex_id, ctx->iter);
    fprintf(fp, "PE %d updating particle %d on timestep %d, iter %d. "
            "p=(%f, %f), v=(%f, %f). # neighbors = %d. have_msg=%d\n", ctx->pe,
            vertex_id, vertex_timestep, ctx->iter, px, py, vx, vy,
            n_neighbors, have_msg);

    for (int i = 0; i < n_neighbors; i++) {
        hvr_vertex_t *neighbor = verts[i];
        int neighbor_id = (int)hvr_vertex_get(PARTICLE_ID, neighbor, ctx);
        int neighbor_timestep = (int)hvr_vertex_get(TIMESTEP, neighbor, ctx);
        fprintf(fp, "  %d: direction=%d. particle %d, timestep %d. "
                "pos=(%f, %f) vel=(%f %f)\n", i, dirs[i], neighbor_id,
                neighbor_timestep,
                hvr_vertex_get(PX, neighbor, ctx),
                hvr_vertex_get(PY, neighbor, ctx),
                hvr_vertex_get(VX, neighbor, ctx),
                hvr_vertex_get(VY, neighbor, ctx));
    }

    fflush(fp);

    if (hvr_vertex_get(HAVE_PREV, vertex, ctx) > 0) {
        const double px = hvr_vertex_get(PREV_PX, vertex, ctx);
        const double py = hvr_vertex_get(PREV_PY, vertex, ctx);
        double accel_x, accel_y;
        compute_accel(hvr_vertex_get(PREV_PX, vertex, ctx),
                hvr_vertex_get(PREV_PY, vertex, ctx),
                vertex_timestep, vertex_id, verts, dirs,
                n_neighbors, &accel_x, &accel_y, ctx);
        assert(!isinf(accel_x));
        assert(!isinf(accel_y));

        const double vx = hvr_vertex_get(PREV_VX, vertex, ctx) + accel_x;
        const double vy = hvr_vertex_get(PREV_VY, vertex, ctx) + accel_y;
        hvr_vertex_set(VX, vx, vertex, ctx);
        hvr_vertex_set(VY, vy, vertex, ctx);
        assert(!isinf(vx));
        assert(!isinf(vy));

        double new_px = px + vx;
        double new_py = py + vy;
        while (new_px < 0.0) new_px += domain_dim;
        while (new_px >= domain_dim) new_px -= domain_dim;
        while (new_py < 0.0) new_py += domain_dim;
        while (new_py >= domain_dim) new_py -= domain_dim;
        assert(!isinf(new_px));
        assert(!isinf(new_py));

        hvr_vertex_set(PX, new_px, vertex, ctx);
        hvr_vertex_set(PY, new_py, vertex, ctx);

        fprintf(fp, "  PE %d done updating particle %d on timestep %d. "
                "p=(%f, %f), v=(%f, %f). # neighbors = %d. changed? %d (%d "
                "%d %d %d)\n", ctx->pe, vertex_id, vertex_timestep,
                hvr_vertex_get(PX, vertex, ctx),
                hvr_vertex_get(PY, vertex, ctx),
                hvr_vertex_get(VX, vertex, ctx),
                hvr_vertex_get(VY, vertex, ctx),
                n_neighbors, vertex->needs_send,
                px != hvr_vertex_get(PX, vertex, ctx),
                py != hvr_vertex_get(PY, vertex, ctx),
                vx != hvr_vertex_get(VX, vertex, ctx),
                vy != hvr_vertex_get(VY, vertex, ctx)
                );
        fflush(fp);
    }

    if (vertex_timestep < timesteps - 1) {
        if (hvr_vertex_get(NEXT_CREATED, vertex, ctx) == 0.0) {
            hvr_vertex_t *next = hvr_vertex_create_n(1, ctx);
            assert(next);

            hvr_vertex_set(TIMESTEP, vertex_timestep + 1, next, ctx);
            hvr_vertex_set(PARTICLE_ID, vertex_id, next, ctx);
            hvr_vertex_set(PX, hvr_vertex_get(PX, vertex, ctx), next, ctx);
            hvr_vertex_set(PY, hvr_vertex_get(PY, vertex, ctx), next, ctx);
            hvr_vertex_set(VX, hvr_vertex_get(VX, vertex, ctx), next, ctx);
            hvr_vertex_set(VY, hvr_vertex_get(VY, vertex, ctx), next, ctx);
            hvr_vertex_set(NEXT_CREATED, 0, next, ctx);
            hvr_vertex_set_uint64(NEXT_ID, 0, next, ctx);
            hvr_vertex_set(HAVE_PREV, 0, next, ctx);
            hvr_vertex_set(PREV_PX, hvr_vertex_get(PX, vertex, ctx), next, ctx);
            hvr_vertex_set(PREV_PY, hvr_vertex_get(PY, vertex, ctx), next, ctx);
            hvr_vertex_set(PREV_VX, hvr_vertex_get(VX, vertex, ctx), next, ctx);
            hvr_vertex_set(PREV_VY, hvr_vertex_get(VY, vertex, ctx), next, ctx);

            hvr_vertex_set_uint64(NEXT_ID, hvr_vertex_get_id(next), vertex, ctx);
            hvr_vertex_set(NEXT_CREATED, 1, vertex, ctx);
        }

        uint64_t next_id = hvr_vertex_get_uint64(NEXT_ID, vertex, ctx);
        hvr_send_msg(next_id, vertex, ctx);
    }
}

static void might_interact_helper(hvr_partition_t t, hvr_partition_t y_min,
        hvr_partition_t y_max, hvr_partition_t x_min, hvr_partition_t x_max,
        hvr_partition_t *interacting_partitions,
        unsigned *n_interacting_partitions,
        unsigned interacting_partitions_capacity) {
    for (hvr_partition_t y = y_min; y <= y_max; y++) {
        for (hvr_partition_t x = x_min; x <= x_max; x++) {
            hvr_partition_t p = t * partitions_per_dim * partitions_per_dim +
                y * partitions_per_dim + x;
            assert(*n_interacting_partitions < interacting_partitions_capacity);
            interacting_partitions[*n_interacting_partitions] = p;
            *n_interacting_partitions += 1;
        }
    }
}

void might_interact(const hvr_partition_t partition,
        hvr_partition_t *interacting_partitions,
        unsigned *n_interacting_partitions,
        unsigned interacting_partitions_capacity,
        hvr_ctx_t ctx) {
    const hvr_partition_t timestep_partition = TIMESTEP_PARTITION(partition);
    const hvr_partition_t y_partition = Y_PARTITION(partition);
    const hvr_partition_t x_partition = X_PARTITION(partition);
    assert(timestep_partition >= 0 && timestep_partition < timesteps);
    assert(y_partition >= 0 && y_partition < partitions_per_dim);
    assert(x_partition >= 0 && x_partition < partitions_per_dim);

    double y_partition_min = (y_partition * PARTITION_DIM) - distance_threshold;
    double y_partition_max = (y_partition + 1) * PARTITION_DIM +
            distance_threshold;
    double x_partition_min = (x_partition * PARTITION_DIM) - distance_threshold;
    double x_partition_max = (x_partition + 1) * PARTITION_DIM +
        distance_threshold;

    hvr_partition_t min_y_part, max_y_part, min_x_part, max_x_part;

    if (y_partition_min < 0.0) min_y_part = 0;
    else min_y_part = (hvr_partition_t)(y_partition_min / PARTITION_DIM);

    if (x_partition_min < 0.0) min_x_part = 0;
    else min_x_part = (hvr_partition_t)(x_partition_min / PARTITION_DIM);

    if (y_partition_max >= domain_dim) max_y_part = partitions_per_dim - 1;
    else max_y_part = (hvr_partition_t)(y_partition_max / PARTITION_DIM);

    if (x_partition_max >= domain_dim) max_x_part = partitions_per_dim - 1;
    else max_x_part = (hvr_partition_t)(x_partition_max / PARTITION_DIM);

    *n_interacting_partitions = 0;
    if (timestep_partition > 0) {
        might_interact_helper(timestep_partition - 1,
                min_y_part, max_y_part, min_x_part, max_x_part,
                interacting_partitions, n_interacting_partitions,
                interacting_partitions_capacity);
    }

    if (timestep_partition < timesteps - 1) {
        might_interact_helper(timestep_partition + 1,
                min_y_part, max_y_part, min_x_part, max_x_part,
                interacting_partitions, n_interacting_partitions,
                interacting_partitions_capacity);
    }
}

void update_coupled_val(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_vertex_t *out_coupled_metric) {
    // Abort if all of my member vertices are infected
    hvr_vertex_set(0, 0, out_coupled_metric, ctx);
}

int should_terminate(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_vertex_t *local_coupled_metric, // coupled_pes[shmem_my_pe()]
        hvr_vertex_t *all_coupled_metrics, // Each PE's val
        hvr_vertex_t *global_coupled_metric, // Sum reduction of coupled_pes
        hvr_set_t *coupled_pes, // An array of size npes, with each PE's val
        int n_coupled_pes,
        int *updates_on_this_iter, // An array of size npes, the number of vertex updates done on each coupled PE
        hvr_set_t *terminated_coupled_pes) {
    return 0;
}

static double random_double_in_range(double min_val_inclusive,
        double max_val_exclusive) {
    return min_val_inclusive + (((double)rand() / (double)RAND_MAX) *
            (max_val_exclusive - min_val_inclusive));
}

int main(int argc, char **argv) {
    hvr_ctx_t hvr_ctx;

    if (argc != 3) {
        fprintf(stderr, "usage: %s <# simulation timesteps> "
                "<time limit in s>\n", argv[0]);
        return 1;
    }


    timesteps = atoi(argv[1]);
    int time_limit_in_seconds = atoi(argv[2]);

    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        p_sync[i] = SHMEM_SYNC_VALUE;
    }

    shmem_init();
    pe = shmem_my_pe();
    npes = shmem_n_pes();

    char filename[1024];
    sprintf(filename, "%d.out", pe);
    fp = fopen(filename, "w");
    assert(fp);

    hvr_ctx_create(&hvr_ctx);

    srand(123 + pe);

    const int nparticles_per_pe = 1;
    hvr_vertex_t *particles = hvr_vertex_create_n(nparticles_per_pe, hvr_ctx);
    for (int i = 0; i < nparticles_per_pe; i++) {
        hvr_vertex_set(TIMESTEP, 0, &particles[i], hvr_ctx);
        hvr_vertex_set(PARTICLE_ID, pe * nparticles_per_pe + i, &particles[i],
                hvr_ctx);

        hvr_vertex_set(PREV_PX, random_double_in_range(0.0, domain_dim),
                &particles[i], hvr_ctx);
        hvr_vertex_set(PREV_PY, random_double_in_range(0.0, domain_dim),
                &particles[i], hvr_ctx);
        hvr_vertex_set(PREV_VX, 0.0, &particles[i], hvr_ctx);
        hvr_vertex_set(PREV_VY, 0.0, &particles[i], hvr_ctx);

        hvr_vertex_set(PX, hvr_vertex_get(PREV_PX, &particles[i], hvr_ctx),
                &particles[i], hvr_ctx);
        hvr_vertex_set(PY, hvr_vertex_get(PREV_PY, &particles[i], hvr_ctx),
                &particles[i], hvr_ctx);
        hvr_vertex_set(VX, 0.0, &particles[i], hvr_ctx);
        hvr_vertex_set(VY, 0.0, &particles[i], hvr_ctx);

        hvr_vertex_set(NEXT_CREATED, 0, &particles[i], hvr_ctx);
        hvr_vertex_set_uint64(NEXT_ID, 0, &particles[i], hvr_ctx);
    }

    hvr_init(timesteps * partitions_per_dim * partitions_per_dim,
            update_metadata,
            might_interact,
            update_coupled_val,
            actor_to_partition,
            NULL, // start_time_step
            should_have_edge,
            should_terminate,
            time_limit_in_seconds, // max_elapsed_seconds
            1, // max_graph_traverse_depth
            hvr_ctx);

    const long long start_time = hvr_current_time_us();
    hvr_body(hvr_ctx);
    const long long elapsed_time = hvr_current_time_us() - start_time;

    shmem_barrier_all();

    if (pe == 0) {
        printf("%d PEs done after %d iterations on PE 0\n", npes,
                hvr_ctx->iter);
    }

    hvr_finalize(hvr_ctx);

    shmem_finalize();

    return 0;
}
