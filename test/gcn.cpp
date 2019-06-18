/* For license: see LICENSE.txt file at top-level */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <shmem.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <limits.h>

#include <map>
#include <set>
#include <vector>
#include <algorithm>

#include <hoover.h>
#include <shmem_rw_lock.h>

/* 
 * A proof of concept for deploying a GCN onto HOOVER.
 */

#define TYPE 0
#define ATTR0 1
#define ATTR1 2
#define ATTR2 3
#define HAVE_PARENT 4
#define PARENT 5

typedef enum {
    NODE_TYPE = 0,
    SUPERNODE_TYPE
} vertex_type_t;

#define UNIFORM_DIST
// #define POINT_DIST
// #define GAUSSIAN_DIST

#define MAX(a, b) (((a) > (b)) ? a : b)
#define MIN(a, b) (((a) < (b)) ? a : b)

// Timing variables
static long long elapsed_time = 0;
static long long max_elapsed, total_time;

// SHMEM variables
static int pe, npes;
long long p_wrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
long p_sync[SHMEM_REDUCE_SYNC_SIZE];

// Application variables
static double node_distance_threshold = 1.0;
static double supernode_distance_threshold = 10.0;
static unsigned domain_dim[3] = {0, 0, 0}; // Size of the global domain
static unsigned pe_chunks_by_dim[3] = {0, 0, 0}; // Number of chunks per dim
static unsigned pe_chunk_size[3] = {0, 0, 0}; // Length of each chunk in each dim
static unsigned partitions_by_dim[3] = {0, 0, 0}; // # of partitions in each dim
static unsigned partitions_size[3] = {0, 0, 0}; // Size of partition in each dim

// GCN variables
#define MAX_NEIGHBORS 10
static double _adjacency[MAX_NEIGHBORS + 1][MAX_NEIGHBORS + 1];
static double _features[MAX_NEIGHBORS + 1][3];
static double _weights[3][2] = {{0.1, 0.02},
                         {0.3, 0.4},
                         {0.2, 0.1}};
static double _adjacency_times_features[MAX_NEIGHBORS + 1][3];
static double _times_weights[MAX_NEIGHBORS + 1][2];

#define PE_COORD_0(my_pe) ((my_pe) / (pe_chunks_by_dim[1] * \
            pe_chunks_by_dim[2]))
#define PE_COORD_1(my_pe) (((my_pe) / pe_chunks_by_dim[2]) % \
        pe_chunks_by_dim[1])
#define PE_COORD_2(my_pe) ((my_pe) % pe_chunks_by_dim[2])
#define TOTAL_PARTITIONS (partitions_by_dim[0] * partitions_by_dim[1] * \
        partitions_by_dim[2])

static unsigned min_point[3] = {0, 0, 0};
static unsigned max_point[3] = {0, 0, 0};

static unsigned min_n_vertices_to_add = 100;
static unsigned max_n_vertices_to_add = 500;

static unsigned n_local_vertices = 0;

#ifdef GAUSSIAN_DIST
// Taken from https://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/
static double randn(double mu, double sigma) {
    double U1, U2, W, mult;
    static double X1, X2;
    static int call = 0;

    if (call == 1)
    {
        call = !call;
        return (mu + sigma * (double) X2);
    }

    do
    {
        U1 = -1 + ((double) rand () / RAND_MAX) * 2;
        U2 = -1 + ((double) rand () / RAND_MAX) * 2;
        W = pow (U1, 2) + pow (U2, 2);
    }
    while (W >= 1 || W == 0);

    mult = sqrt ((-2 * log (W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;

    call = !call;

    return (mu + sigma * (double) X1);
}
#endif // GAUSSIAN_DIST

static unsigned int g_seed;

// Used to seed the generator.
inline void fast_srand(int seed) {
    g_seed = seed;
}

// Compute a pseudorandom integer.
// Output value in range [0, 32767]
inline uint64_t fast_rand(void) {
    g_seed = (214013*g_seed+2531011);
    int lower = (g_seed>>16)&0x7FFF;

    g_seed = (214013*g_seed+2531011);
    int upper = (g_seed>>16)&0x7FFF;

    return (((uint64_t)upper) << 32) + lower;
}

static void rand_point(unsigned *f0, unsigned *f1, unsigned *f2) {
#ifdef GAUSSIAN_DIST
    *f0 = (int)randn((double)(min_point[0] + max_point[0]) / 2.0,
            (max_point[0] - min_point[0]) / 3.0);
    *f1 = (unsigned)randn((double)(min_point[1] + max_point[1]) / 2.0,
            (max_point[1] - min_point[1]) / 3.0);
    *f2 = (unsigned)randn((double)(min_point[2] + max_point[2]) / 2.0,
            (max_point[2] - min_point[2]) / 3.0);
#elif defined(UNIFORM_DIST)
    *f0 = min_point[0] + (fast_rand() % (max_point[0] - min_point[0]));
    *f1 = min_point[1] + (fast_rand() % (max_point[1] - min_point[1]));
    *f2 = min_point[2] + (fast_rand() % (max_point[2] - min_point[2]));
#elif defined(POINT_DIST)
    *f0 = min_point[0] + (max_point[0] - min_point[0]) / 2;
    *f1 = min_point[1] + (max_point[1] - min_point[1]) / 2;
    *f2 = min_point[2] + (max_point[2] - min_point[2]) / 2;
#else
#error No distributed specified
#endif
}

hvr_partition_t actor_to_partition(const hvr_vertex_t *actor, hvr_ctx_t ctx) {
    unsigned feat1_partition = (unsigned)hvr_vertex_get(ATTR0, actor, ctx) /
        partitions_size[0];
    unsigned feat2_partition = (unsigned)hvr_vertex_get(ATTR1, actor, ctx) /
        partitions_size[1];
    unsigned feat3_partition = (unsigned)hvr_vertex_get(ATTR2, actor, ctx) /
        partitions_size[2];

    feat1_partition = feat1_partition % partitions_by_dim[0];
    feat2_partition = feat2_partition % partitions_by_dim[1];
    feat3_partition = feat3_partition % partitions_by_dim[2];

    return feat1_partition * partitions_by_dim[1] * partitions_by_dim[2] +
        feat2_partition * partitions_by_dim[2] + feat3_partition;
}

hvr_edge_type_t should_have_edge(const hvr_vertex_t *a, const hvr_vertex_t *b,
        hvr_ctx_t ctx) {
    if (hvr_vertex_get_uint64(TYPE, a, ctx) !=
            hvr_vertex_get_uint64(TYPE, b, ctx) ||
            hvr_vertex_get_id(a) == hvr_vertex_get_id(b)) {
        return NO_EDGE;
    }

    double distance_threshold = 
        (hvr_vertex_get_uint64(TYPE, a, ctx) == NODE_TYPE) ?
        node_distance_threshold : supernode_distance_threshold;

    // Same vertex type
    const double delta0 = hvr_vertex_get(ATTR0, b, ctx) -
        hvr_vertex_get(ATTR0, a, ctx);
    const double delta1 = hvr_vertex_get(ATTR1, b, ctx) -
        hvr_vertex_get(ATTR1, a, ctx);
    const double delta2 = (hvr_vertex_get_uint64(TYPE, a, ctx) == NODE_TYPE) ?
        hvr_vertex_get(ATTR2, b, ctx) - hvr_vertex_get(ATTR2, a, ctx) : 0.0;
    if (delta0 * delta0 + delta1 * delta1 + delta2 * delta2 <=
            distance_threshold * distance_threshold) {
        return BIDIRECTIONAL;
    } else {
        return NO_EDGE;
    }
}

void start_time_step(hvr_vertex_iter_t *iter, hvr_set_t *couple_with,
        hvr_ctx_t ctx) {
#ifdef VERBOSE
    const unsigned long long start_callback = hvr_current_time_us();
#endif

    unsigned n_vertices_to_add;
    if (min_n_vertices_to_add == max_n_vertices_to_add) {
        n_vertices_to_add = min_n_vertices_to_add;
    } else {
        n_vertices_to_add = min_n_vertices_to_add +
            (fast_rand() % (max_n_vertices_to_add - min_n_vertices_to_add));
    }

    n_local_vertices += n_vertices_to_add;

    for (unsigned i = 0; i < n_vertices_to_add; i++) {
        hvr_vertex_t *new_vertex = hvr_vertex_create(ctx);
        unsigned feat1, feat2, feat3;
        rand_point(&feat1, &feat2, &feat3);

        feat1 = MAX(feat1, 0);
        feat2 = MAX(feat2, 0);
        feat3 = MAX(feat3, 0);

        feat1 = MIN(feat1, domain_dim[0] - 1);
        feat2 = MIN(feat2, domain_dim[1] - 1);
        feat3 = MIN(feat3, domain_dim[2] - 1);

        hvr_vertex_set_uint64(TYPE, NODE_TYPE, new_vertex, ctx);
        hvr_vertex_set(ATTR0, feat1, new_vertex, ctx);
        hvr_vertex_set(ATTR1, feat2, new_vertex, ctx);
        hvr_vertex_set(ATTR2, feat3, new_vertex, ctx);
        hvr_vertex_set_uint64(HAVE_PARENT, 0, new_vertex, ctx);
    }
}

void update_vertex(hvr_vertex_t *vertex, hvr_set_t *couple_with,
        hvr_ctx_t ctx) {
    /*
     * If a vertex, apply a convolution to its neighborhood to compute a
     * 2-feature vector that represents it. Create a parent with this state if
     * none exists yet, otherwise send a message to the parent with that state.
     *
     * If a supernode, pull out my latest message from my child and update my
     * state based on that. If I have zero neighbors, print myself as a
     * potential anomaly.
     */
    if (hvr_vertex_get_uint64(TYPE, vertex, ctx) == NODE_TYPE) {
        hvr_vertex_t **neighbors;
        hvr_edge_type_t *neighbor_dirs;
        int n_neighbors = hvr_get_neighbors(vertex, &neighbors,
                &neighbor_dirs, ctx);
        assert(n_neighbors <= MAX_NEIGHBORS);

        _features[0][0] = hvr_vertex_get(ATTR0, vertex, ctx);
        _features[0][1] = hvr_vertex_get(ATTR1, vertex, ctx);
        _features[0][2] = hvr_vertex_get(ATTR2, vertex, ctx);

        // self edge
        _adjacency[0][0] = 1;

        for (int i = 0; i < n_neighbors; i++) {
            _features[i + 1][0] = hvr_vertex_get(ATTR0, neighbors[i], ctx);
            _features[i + 1][1] = hvr_vertex_get(ATTR1, neighbors[i], ctx);
            _features[i + 1][2] = hvr_vertex_get(ATTR2, neighbors[i], ctx);

            // 'vertex' has an edge with each neighbor
            _adjacency[0][i + 1] = 1;
            _adjacency[i + 1][0] = 1;

            // self edge for each neighbor
            _adjacency[i + 1][i + 1] = 1;

            // Find anyone else in neighbors list
            hvr_vertex_t **other_neighbors;
            hvr_edge_type_t *other_neighbor_dirs;
            int n_other_neighbors = hvr_get_neighbors(neighbors[i],
                    &other_neighbors, &other_neighbor_dirs, ctx);
            for (int j = 0; j < n_other_neighbors; j++) {
                int found = -1;
                for (int k = 0; k < n_neighbors && found < 0; k++) {
                    if (hvr_vertex_get_id(other_neighbors[j]) ==
                            hvr_vertex_get_id(neighbors[k])) {
                        found = k;
                    }
                }
                if (found >= 0) {
                    _adjacency[i + 1][found + 1] = 1;
                    _adjacency[found + 1][i + 1] = 1;
                }
            }
            hvr_release_neighbors(other_neighbors, other_neighbor_dirs,
                    n_other_neighbors, ctx);
        }

        hvr_release_neighbors(neighbors, neighbor_dirs, n_neighbors,
                ctx);

        // Perform _adjacency * _features * _weights followed by max pooling
        for (int i = 0; i < n_neighbors + 1; i++) {
            for (int j = 0; j < 3; j++) {
                // ith row of _adjacency * jth column of _features
                double sum = 0.0;
                for (int k = 0; k < n_neighbors + 1; k++) {
                    sum += _adjacency[i][k] * _features[k][j];
                }
                _adjacency_times_features[i][j] = sum;
            }
        }

        for (int i = 0; i < n_neighbors + 1; i++) {
            for (int j = 0; j < 2; j++) {
                double sum = 0.0;
                for (int k = 0; k < n_neighbors + 1; k++) {
                    sum += _adjacency_times_features[i][k] * _weights[k][j];
                }
                _times_weights[i][j] = sum;
            }
        }

        double maximum0 = _times_weights[0][0];
        double maximum1 = _times_weights[0][1];
        for (int i = 1; i < n_neighbors + 1; i++) {
            if (_times_weights[i][0] > maximum0) {
                maximum0 = _times_weights[i][0];
            }
            if (_times_weights[i][1] > maximum1) {
                maximum1 = _times_weights[i][1];
            }
        }

        if (hvr_vertex_get_uint64(HAVE_PARENT, vertex, ctx)) {
            // Send message
            hvr_vertex_t msg;
            hvr_vertex_init(&msg, HVR_INVALID_VERTEX_ID, 0);
            hvr_vertex_set(0, maximum0, &msg, ctx);
            hvr_vertex_set(1, maximum1, &msg, ctx);
            hvr_send_msg(hvr_vertex_get_uint64(PARENT, vertex, ctx), &msg, ctx);
        } else {
            // Create parent
            hvr_vertex_t *parent = hvr_vertex_create(ctx);
            hvr_vertex_set_uint64(TYPE, SUPERNODE_TYPE, parent, ctx);
            hvr_vertex_set(ATTR0, maximum0, parent, ctx);
            hvr_vertex_set(ATTR1, maximum1, parent, ctx);
            hvr_vertex_set(ATTR2, 0.0, parent, ctx);

            hvr_vertex_set_uint64(HAVE_PARENT, 1, vertex, ctx);
            hvr_vertex_set_uint64(PARENT, hvr_vertex_get_id(parent), vertex,
                    ctx);
        }
    } else {
        assert(hvr_vertex_get_uint64(TYPE, vertex, ctx) == SUPERNODE_TYPE);

        hvr_vertex_t msg;
        int have_msg = hvr_poll_msg(vertex, &msg, ctx);
        if (have_msg) {
            // Update my state
            hvr_vertex_set(ATTR0, hvr_vertex_get(0, &msg, ctx), vertex, ctx);
            hvr_vertex_set(ATTR1, hvr_vertex_get(1, &msg, ctx), vertex, ctx);
        }

        // Flush older messages
        while (have_msg) {
            have_msg = hvr_poll_msg(vertex, &msg, ctx);
        }

        hvr_vertex_t **neighbors;
        hvr_edge_type_t *neighbor_dirs;
        int n_neighbors = hvr_get_neighbors(vertex, &neighbors,
                &neighbor_dirs, ctx);
        if (n_neighbors == 0) {
            printf("PE %d Vertex %llu has no neighbors, possible anomaly\n",
                    pe, hvr_vertex_get_id(vertex));
        }
        hvr_release_neighbors(neighbors, neighbor_dirs, n_neighbors,
                ctx);
    }
}

void might_interact(const hvr_partition_t partition,
        hvr_partition_t *interacting_partitions,
        unsigned *out_n_interacting_partitions,
        unsigned interacting_partitions_capacity,
        hvr_ctx_t ctx) {
    assert(partition != HVR_INVALID_PARTITION);

    /*
     * If a vertex in 'partition' may create an edge with any vertices in any
     * partition in 'partitions', they might interact (so return 1).
     */
    hvr_partition_t feat1_partition = partition /
        (partitions_by_dim[1] * partitions_by_dim[2]);
    hvr_partition_t feat2_partition = (partition / partitions_by_dim[2]) %
        partitions_by_dim[1];
    hvr_partition_t feat3_partition = partition % partitions_by_dim[2];

    int64_t feat1_min = feat1_partition * partitions_size[0];
    int64_t feat1_max = (feat1_partition + 1) * partitions_size[0];

    int64_t feat2_min = feat2_partition * partitions_size[1];
    int64_t feat2_max = (feat2_partition + 1) * partitions_size[1];

    int64_t feat3_min = feat3_partition * partitions_size[2];
    int64_t feat3_max = (feat3_partition + 1) * partitions_size[2];

    feat1_min -= MAX(node_distance_threshold, supernode_distance_threshold);
    feat1_max += MAX(node_distance_threshold, supernode_distance_threshold);
    feat2_min -= MAX(node_distance_threshold, supernode_distance_threshold);
    feat2_max += MAX(node_distance_threshold, supernode_distance_threshold);
    feat3_min -= MAX(node_distance_threshold, supernode_distance_threshold);
    feat3_max += MAX(node_distance_threshold, supernode_distance_threshold);

#if 0
    feat1_min = MAX(feat1_min, 0);
    feat2_min = MAX(feat2_min, 0);
    feat3_min = MAX(feat3_min, 0);

    feat1_max = MIN(feat1_max, domain_dim[0] - 1);
    feat2_max = MIN(feat2_max, domain_dim[1] - 1);
    feat3_max = MIN(feat3_max, domain_dim[2] - 1);
#endif

    unsigned n_interacting_partitions = 0;

    for (int64_t other_feat1_partition = feat1_min / partitions_size[0];
            other_feat1_partition <= feat1_max / partitions_size[0];
            other_feat1_partition++) {
        for (int64_t other_feat2_partition = feat2_min / partitions_size[1];
                other_feat2_partition <= feat2_max / partitions_size[1];
                other_feat2_partition++) {
            for (int64_t other_feat3_partition = feat3_min / partitions_size[2];
                    other_feat3_partition <= feat3_max / partitions_size[2];
                    other_feat3_partition++) {
                /*
                 * other_feat*_partition may each be negative or
                 * > partitions_by_dim[*]
                 */

                int64_t feat1 = other_feat1_partition;
                int64_t feat2 = other_feat2_partition;
                int64_t feat3 = other_feat3_partition;

                while (feat1 < 0) feat1 += partitions_by_dim[0];
                while (feat2 < 0) feat2 += partitions_by_dim[1];
                while (feat3 < 0) feat3 += partitions_by_dim[2];

                feat1 = (feat1 % partitions_by_dim[0]);
                feat2 = (feat2 % partitions_by_dim[1]);
                feat3 = (feat3 % partitions_by_dim[2]);

                int64_t this_part =
                    feat1 * partitions_by_dim[1] * partitions_by_dim[2] +
                    feat2 * partitions_by_dim[2] +
                    feat3;
                assert(n_interacting_partitions + 1 <=
                        interacting_partitions_capacity);
                interacting_partitions[n_interacting_partitions++] = this_part;
            }
        }
    }
    *out_n_interacting_partitions = n_interacting_partitions;
}

void update_coupled_val(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_vertex_t *out_coupled_metric) {
    hvr_vertex_set(0, 0.0, out_coupled_metric, ctx);
}

int main(int argc, char **argv) {
    hvr_ctx_t hvr_ctx;

    if (argc != 15) {
        fprintf(stderr, "usage: %s <time-limit-in-seconds> "
                "<distance-threshold> <domain-size-0> <domain-size-1> "
                "<domain-size-2> <pe-dim-0> <pe-dim-1> <pe-dim-2> "
                "<partition-dim-0> <partition-dim-1> <partition-dim-2>"
                "<min-vertices-to-add> <max-vertices-to-add>\n",
                argv[0]);
        return 1;
    }

    unsigned long long time_limit_s = atoi(argv[1]);
    node_distance_threshold = atof(argv[2]);
    supernode_distance_threshold = atof(argv[3]);
    domain_dim[0] = atoi(argv[4]);
    domain_dim[1] = atoi(argv[5]);
    domain_dim[2] = atoi(argv[6]);
    pe_chunks_by_dim[0] = atoi(argv[7]);
    pe_chunks_by_dim[1] = atoi(argv[8]);
    pe_chunks_by_dim[2] = atoi(argv[9]);
    partitions_by_dim[0] = atoi(argv[10]);
    partitions_by_dim[1] = atoi(argv[11]);
    partitions_by_dim[2] = atoi(argv[12]);
    min_n_vertices_to_add = atoi(argv[13]);
    max_n_vertices_to_add = atoi(argv[14]);

    if (min_n_vertices_to_add > max_n_vertices_to_add) {
        fprintf(stderr, "Minimum # vertices to add must be less than the "
                "maximum. Minimum = %u, maximum = %u.\n", min_n_vertices_to_add,
                max_n_vertices_to_add);
        return 1;
    }

    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        p_sync[i] = SHMEM_SYNC_VALUE;
    }
    shmem_init();

    pe = shmem_my_pe();
    npes = shmem_n_pes();

    if (pe == 0) {
        printf("Domain size = %u x %u x %u\n", domain_dim[0], domain_dim[1],
                domain_dim[2]);
        printf("PE cube = %u x %u x %u\n", pe_chunks_by_dim[0],
                pe_chunks_by_dim[1], pe_chunks_by_dim[2]);
        printf("Partitions cube = %u x %u x %u\n", partitions_by_dim[0],
                partitions_by_dim[1], partitions_by_dim[2]);
#ifdef MULTITHREADED
        printf("# OMP threads = %d\n", nthreads);
#endif
    }

    assert(pe_chunks_by_dim[0] * pe_chunks_by_dim[1] * pe_chunks_by_dim[2] ==
            (unsigned)npes);
    assert(domain_dim[0] % pe_chunks_by_dim[0] == 0);
    assert(domain_dim[1] % pe_chunks_by_dim[1] == 0);
    assert(domain_dim[2] % pe_chunks_by_dim[2] == 0);

    pe_chunk_size[0] = domain_dim[0] / pe_chunks_by_dim[0];
    pe_chunk_size[1] = domain_dim[1] / pe_chunks_by_dim[1];
    pe_chunk_size[2] = domain_dim[2] / pe_chunks_by_dim[2];

    min_point[0] = PE_COORD_0(pe) * pe_chunk_size[0];
    min_point[1] = PE_COORD_1(pe) * pe_chunk_size[1];
    min_point[2] = PE_COORD_2(pe) * pe_chunk_size[2];

    max_point[0] = min_point[0] + pe_chunk_size[0];
    max_point[1] = min_point[1] + pe_chunk_size[1];
    max_point[2] = min_point[2] + pe_chunk_size[2];

    assert(domain_dim[0] % partitions_by_dim[0] == 0);
    assert(domain_dim[1] % partitions_by_dim[1] == 0);
    assert(domain_dim[2] % partitions_by_dim[2] == 0);
    partitions_size[0] = domain_dim[0] / partitions_by_dim[0];
    partitions_size[1] = domain_dim[1] / partitions_by_dim[1];
    partitions_size[2] = domain_dim[2] / partitions_by_dim[2];

    if (pe == 0) {
        printf("%d PE(s) running...\n", npes);
    }
    printf("PE %d responsible for (%u, %u, %u) -> (%u, %u, %u)\n", pe,
            min_point[0], min_point[1], min_point[2], max_point[0],
            max_point[1], max_point[2]);
    fflush(stdout);

    shmem_barrier_all();

    hvr_ctx_create(&hvr_ctx);

    fast_srand(123 + pe);

    hvr_init(TOTAL_PARTITIONS, // # partitions
            update_vertex,
            might_interact,
            update_coupled_val,
            actor_to_partition,
            start_time_step,
            should_have_edge,
            NULL, // should_terminate
            time_limit_s,
            1,
            hvr_ctx);

    shmem_barrier_all();

    unsigned long long start_time = hvr_current_time_us();
    hvr_exec_info info = hvr_body(hvr_ctx);
    elapsed_time = hvr_current_time_us() - start_time;

    // Get a total wallclock time across all PEs
    shmem_longlong_sum_to_all(&total_time, &elapsed_time, 1, 0, 0, npes, p_wrk,
            p_sync);
    shmem_barrier_all();

    // Get a max wallclock time across all PEs
    shmem_longlong_max_to_all(&max_elapsed, &elapsed_time, 1, 0, 0, npes, p_wrk,
            p_sync);
    shmem_barrier_all();

    unsigned count_nodes = 0;
    unsigned count_supernodes = 0;
    double avg_supernode_edges = 0.0;
    unsigned n_prints = 0;
    hvr_vertex_iter_t iter;
    hvr_vertex_iter_init(&iter, hvr_ctx);
    for (hvr_vertex_t *vertex = hvr_vertex_iter_next(&iter); vertex;
            vertex = hvr_vertex_iter_next(&iter)) {
        if (hvr_vertex_get_uint64(TYPE, vertex, hvr_ctx) == NODE_TYPE) {
            count_nodes++;
        } else if (hvr_vertex_get_uint64(TYPE, vertex, hvr_ctx) ==
                SUPERNODE_TYPE) {

            if (pe == 0 && n_prints < 10) {
                printf("PE %d vertex %llu attrs (%f, %f)\n", pe,
                        hvr_vertex_get_id(vertex),
                        hvr_vertex_get(ATTR0, vertex, hvr_ctx),
                        hvr_vertex_get(ATTR1, vertex, hvr_ctx));
                n_prints++;
            }

            hvr_vertex_t **neighbors;
            hvr_edge_type_t *neighbor_dirs;
            int n_neighbors = hvr_get_neighbors(vertex, &neighbors,
                    &neighbor_dirs, hvr_ctx);
            avg_supernode_edges += n_neighbors;
            hvr_release_neighbors(neighbors, neighbor_dirs, n_neighbors,
                    hvr_ctx);
            count_supernodes++;
        } else {
            abort();
        }
    }

    printf("PE %d has %u nodes, %u supernodes, %f edges per supernode\n", pe,
            count_nodes, count_supernodes,
            avg_supernode_edges / (double)count_supernodes);

    if (pe == 0) {
        printf("%d PEs, total CPU time = %f ms, max elapsed = %f ms, "
                "%d iterations completed on PE 0\n", npes,
                (double)total_time / 1000.0, (double)max_elapsed / 1000.0,
                info.executed_iters);
    }

    hvr_finalize(hvr_ctx);

    shmem_finalize();

    return 0;
}
