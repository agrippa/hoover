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
 * This implementation borrows code from the existing distributed GBAD
 * implemetation (instruction_detection.cpp) in HOOVER to implement a simple
 * community detection example.
 *
 * At a high level, it does this by combining brute force k-clique detection
 * with the Clique percolation method
 * (https://en.wikipedia.org/wiki/Clique_percolation_method) to identify
 * overlapping communities.
 */

#define K 4

// #define VERBOSE

#define TYPE 0
#define ATTR0 1
#define ATTR1 2
#define ATTR2 3
#define VERT_ID 4
#define SUPERNODE_ID (K + 1)
#define SUPERNODE_LBL (K + 2)

typedef enum {
    NODE_TYPE = 0,
    SUPERNODE_TYPE
} vertex_type_t;

#define UNIFORM_DIST
// #define POINT_DIST
// #define GAUSSIAN_DIST

#define MAX(a, b) (((a) > (b)) ? a : b)
#define MIN(a, b) (((a) < (b)) ? a : b)

typedef struct _clique_t {
    hvr_vertex_id_t vertices[K];
} clique_t;

clique_t *local_cliques = NULL;
unsigned n_local_cliques = 0;

// Timing variables
static unsigned long long start_time = 0;
static unsigned long long time_limit_s = 0;
static long long elapsed_time = 0;
static long long max_elapsed, total_time;

// SHMEM variables
static int pe, npes;
long long p_wrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
long p_sync[SHMEM_REDUCE_SYNC_SIZE];

// Application variables
static double distance_threshold = 1.0;
static unsigned domain_dim[3] = {0, 0, 0}; // Size of the global domain
static unsigned pe_chunks_by_dim[3] = {0, 0, 0}; // Number of chunks per dim
static unsigned pe_chunk_size[3] = {0, 0, 0}; // Length of each chunk in each dim
static unsigned partitions_by_dim[3] = {0, 0, 0}; // # of partitions in each dim
static unsigned partitions_size[3] = {0, 0, 0}; // Size of partition in each dim

#define PE_COORD_0(my_pe) ((my_pe) / (pe_chunks_by_dim[1] * \
            pe_chunks_by_dim[2]))
#define PE_COORD_1(my_pe) (((my_pe) / pe_chunks_by_dim[2]) % \
        pe_chunks_by_dim[1])
#define PE_COORD_2(my_pe) ((my_pe) % pe_chunks_by_dim[2])
#define TOTAL_PARTITIONS (partitions_by_dim[0] * partitions_by_dim[1] * \
        partitions_by_dim[2])

static unsigned min_point[3] = {0, 0, 0};
static unsigned max_point[3] = {0, 0, 0};

static unsigned min_n_vertices_to_add = 300;
static unsigned max_n_vertices_to_add = 300;

static unsigned n_local_vertices = 0;

static FILE *pe_anomalies_fp = NULL;

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
    if (hvr_vertex_get_uint64(TYPE, actor, ctx) == SUPERNODE_TYPE) {
        // Not in a partition, does not need implicit discovery
        return HVR_INVALID_PARTITION;
    }

    unsigned feat1_partition = (unsigned)hvr_vertex_get(ATTR0, actor, ctx) /
        partitions_size[0];
    unsigned feat2_partition = (unsigned)hvr_vertex_get(ATTR1, actor, ctx) /
        partitions_size[1];
    unsigned feat3_partition = (unsigned)hvr_vertex_get(ATTR2, actor, ctx) /
        partitions_size[2];
    return feat1_partition * partitions_by_dim[1] * partitions_by_dim[2] +
        feat2_partition * partitions_by_dim[2] + feat3_partition;
}

static int supernodes_overlapping(const hvr_vertex_t *a, const hvr_vertex_t *b,
        hvr_ctx_t ctx) {
    int count_overlapping = 0;
    for (unsigned i = 0; i < K; i++) {
        hvr_vertex_id_t v = hvr_vertex_get_uint64(1 + i, a, ctx);

        int found = 0;
        for (unsigned j = 0; j < K && !found; j++) {
            if (hvr_vertex_get_uint64(1 + j, b, ctx) == v) {
                found = 1;
            }
        }

        if (found) count_overlapping++;
    }
    return count_overlapping;
}

hvr_edge_type_t should_have_edge(const hvr_vertex_t *a, const hvr_vertex_t *b,
        hvr_ctx_t ctx) {
    if (hvr_vertex_get_uint64(TYPE, a, ctx) == NODE_TYPE &&
            hvr_vertex_get_uint64(TYPE, b, ctx) == NODE_TYPE) {
        if (hvr_vertex_get_id(a) == hvr_vertex_get_id(b)) {
            return NO_EDGE;
        }

        const double delta0 = hvr_vertex_get(ATTR0, b, ctx) -
            hvr_vertex_get(ATTR0, a, ctx);
        const double delta1 = hvr_vertex_get(ATTR1, b, ctx) -
            hvr_vertex_get(ATTR1, a, ctx);
        const double delta2 = hvr_vertex_get(ATTR2, b, ctx) -
            hvr_vertex_get(ATTR2, a, ctx);
        if (delta0 * delta0 + delta1 * delta1 + delta2 * delta2 <=
                distance_threshold * distance_threshold) {
            return BIDIRECTIONAL;
        }
        return NO_EDGE;
    } else {
        abort();
    }
}

static inline int clique_contains(hvr_vertex_id_t v, clique_t *clique,
        unsigned n) {
    for (unsigned i = 0; i < n; i++) {
        if (clique->vertices[i] == v) return 1;
    }
    return 0;
}

static inline int neighbors_contains(hvr_vertex_id_t v,
        hvr_vertex_t **neighbors, unsigned n) {
    for (unsigned i = 0; i < n; i++) {
        if (neighbors[i]->id == v) return 1;
    }
    return 0;
}

static inline int list_of_clique_contains(clique_t *clique, clique_t *cliques,
        unsigned n) {
    for (unsigned i = 0; i < n; i++) {

        int match = 1;
        for (unsigned j = 0; j < K && match; j++) {
            if (!clique_contains(clique->vertices[j], &cliques[i], K)) {
                match = 0;
            }
        }

        if (match) return 1;
    }
    return 0;
}

static void find_cliques(clique_t *clique, unsigned n_inserted,
        clique_t **cliques, unsigned *ncliques, hvr_vertex_t **candidates,
        unsigned ncandidates, hvr_ctx_t ctx) {
    if (n_inserted == K) {
        /*
         * If we get here, we have a clique. Cross-PE cliques will be
         * represented on both PEs.
         */
        unsigned current_n = *ncliques;
        if (!list_of_clique_contains(clique, *cliques, current_n)) {
            *cliques = (clique_t *)realloc(*cliques,
                    (current_n + 1) * sizeof(clique_t));
            assert(*cliques);
            memcpy((*cliques) + current_n, clique, sizeof(*clique));
            *ncliques = current_n + 1;
        }
    } else {
        // Add another one
        for (unsigned i = 0; i < ncandidates; i++) {
            hvr_vertex_id_t id = candidates[i]->id;
            if (clique_contains(id, clique, n_inserted)) continue;

            hvr_neighbors_t neighbors;
            hvr_get_neighbors(candidates[i], &neighbors, ctx);

            int edges_with_all = 1;
            hvr_vertex_t *neighbor;
            hvr_edge_type_t neighbor_dir;
            while (hvr_neighbors_next(&neighbors, &neighbor, &neighbor_dir)) {
                int found = 0;
                for (unsigned j = 0; j < n_inserted && !found; j++) {
                    hvr_vertex_id_t to_find = clique->vertices[j];
                    if (neighbor->id == to_find) {
                        found = 1;
                    }
                }
                if (!found) {
                    edges_with_all = 0;
                    break;
                }
            }

            hvr_release_neighbors(&neighbors, ctx);

            if (edges_with_all) {
                clique->vertices[n_inserted] = id;
                find_cliques(clique, n_inserted + 1, cliques, ncliques,
                        candidates, ncandidates, ctx);
            }
        }
    }
}

void start_time_step(hvr_vertex_iter_t *iter, hvr_set_t *couple_with,
        hvr_ctx_t ctx) {
#ifdef VERBOSE
    const unsigned long long start_callback = hvr_current_time_us();
#endif

    /*
     * On each time step, calculate a random number of vertices to insert. Then,
     * use hvr_sparse_vec_create_n to insert them with 3 randomly generated
     * features which are designed to be most likely to just interact with
     * vertices on this node (but possibly with vertices on other nodes).
     */
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
        hvr_vertex_set(VERT_ID, fast_rand(), new_vertex, ctx);
    }

    const unsigned long long start_search = hvr_current_time_us();

    /*
     * For each local vertex, grab its neighbors. Recursively try all
     * permutations of them to see if any combination leads to a k-clique.
     */
    unsigned previous_n_cliques = n_local_cliques;
    unsigned n_local_nodes = 0;
    unsigned n_local_supernodes = 0;
    unsigned min_supernode_edges = UINT_MAX;
    unsigned max_supernode_edges = 0;
    for (hvr_vertex_t *vertex = hvr_vertex_iter_next(iter); vertex;
            vertex = hvr_vertex_iter_next(iter)) {
        hvr_vertex_t *neighbor;
        hvr_edge_type_t neighbor_dir;
        hvr_neighbors_t neighbors;
        hvr_get_neighbors(vertex, &neighbors, ctx);

        if (hvr_vertex_get_uint64(TYPE, vertex, ctx) == NODE_TYPE) {
            clique_t clique;
            clique.vertices[0] = vertex->id;
            unsigned n_inserted = 1;

            static hvr_vertex_t *packed_neighbors[1024];
            unsigned n_neighbors = 0;
            while (hvr_neighbors_next(&neighbors, &neighbor, &neighbor_dir)) {
                assert(n_neighbors < 1024);
                packed_neighbors[n_neighbors++] = neighbor;
            }

            find_cliques(&clique, n_inserted, &local_cliques, &n_local_cliques,
                    packed_neighbors, n_neighbors, ctx);
            n_local_nodes++;
        } else {
            assert(hvr_vertex_get_uint64(TYPE, vertex, ctx) == SUPERNODE_TYPE);
            n_local_supernodes++;

            unsigned n_neighbors = 0;
            while (hvr_neighbors_next(&neighbors, &neighbor, &neighbor_dir)) {
                n_neighbors++;
            }

            if (n_neighbors < min_supernode_edges) {
                min_supernode_edges = n_neighbors;
            }
            if (n_neighbors > max_supernode_edges) {
                max_supernode_edges = n_neighbors;
            }
        }
        hvr_release_neighbors(&neighbors, ctx);
    }
    assert(n_local_supernodes == previous_n_cliques);

    /*
     * All cliques between (previous_n_cliques, n_local_cliques] are new. Need
     * to insert a node in the super node graph for them and notify each member
     * vertex of its new parent clique.
     */

    for (unsigned i = previous_n_cliques; i < n_local_cliques; i++) {
        clique_t *new_clique = &local_cliques[i];
        hvr_vertex_t *new_supernode = hvr_vertex_create(ctx);

        uint64_t supernode_id = 0;
        hvr_vertex_set_uint64(TYPE, SUPERNODE_TYPE, new_supernode, ctx);
        for (unsigned j = 0; j < K; j++) {
            hvr_vertex_set_uint64(1 + j, new_clique->vertices[j], new_supernode,
                    ctx);
            supernode_id += new_clique->vertices[j];
        }
        hvr_vertex_set_uint64(SUPERNODE_ID, supernode_id, new_supernode, ctx);
        hvr_vertex_set_uint64(SUPERNODE_LBL, hvr_vertex_get_id(new_supernode),
                new_supernode, ctx);

        for (unsigned j = 0; j < K; j++) {
            hvr_send_msg(new_clique->vertices[j], new_supernode, ctx);
        }
    }

#if 0
    if (n_local_cliques > 0) {
        printf("PE %d has %d k-cliques/supernodes, %u nodes on iter %d. Min / "
                "max supernode edges = %u / %u\n", ctx->pe, n_local_cliques,
                n_local_nodes, ctx->iter, min_supernode_edges,
                max_supernode_edges);
    }
#endif
}

void update_vertex(hvr_vertex_t *vertex, hvr_set_t *couple_with,
        hvr_ctx_t ctx) {
    static std::map<hvr_vertex_id_t, std::set<hvr_vertex_id_t> > parents_map;

    if (hvr_vertex_get_uint64(TYPE, vertex, ctx) == NODE_TYPE) {
        if (parents_map.find(vertex->id) == parents_map.end()) {
            parents_map.insert(
                    std::pair<hvr_vertex_id_t, std::set<hvr_vertex_id_t> >(
                        vertex->id, std::set<hvr_vertex_id_t>()));
        }

        std::set<hvr_vertex_id_t>& parents = parents_map.at(vertex->id);

        hvr_vertex_t clique;
        int have_msg = hvr_poll_msg(vertex, &clique, ctx);
        while (have_msg) {
            /*
             * As a node receives notifications of wrapping cliques, notify all
             * existing cliques of it.
             */
            for (std::set<hvr_vertex_id_t>::iterator i = parents.begin(),
                    e = parents.end(); i != e; i++) {
                hvr_send_msg(*i, &clique, ctx);
            }

            parents.insert(clique.id);

            have_msg = hvr_poll_msg(vertex, &clique, ctx);
        }
    } else {
        assert(hvr_vertex_get_uint64(TYPE, vertex, ctx) == SUPERNODE_TYPE);

        hvr_vertex_t clique;
        int have_msg = hvr_poll_msg(vertex, &clique, ctx);
        while (have_msg) {
            // New potential neighbor supernode
            int count_overlapping = supernodes_overlapping(vertex, &clique,
                    ctx);
            if (count_overlapping >= K - 1) {
                hvr_create_edge_with_vertex(vertex, &clique, BIDIRECTIONAL,
                        ctx);
                hvr_set_insert(hvr_vertex_get_owning_pe(&clique), couple_with);
            }

            have_msg = hvr_poll_msg(vertex, &clique, ctx);
        }

        // Find connected components in supernode graph via label propagation
        hvr_neighbors_t neighbors;
        hvr_get_neighbors(vertex, &neighbors, ctx);

        uint64_t min_supernode_lbl = hvr_vertex_get_uint64(SUPERNODE_LBL, vertex,
                ctx);
        hvr_vertex_t *neighbor;
        hvr_edge_type_t neighbor_dir;
        while (hvr_neighbors_next(&neighbors, &neighbor, &neighbor_dir)) {
            uint64_t neighbor_lbl = hvr_vertex_get_uint64(SUPERNODE_LBL,
                    neighbor, ctx);
            if (neighbor_lbl < min_supernode_lbl) {
                min_supernode_lbl = neighbor_lbl;
            }
        }

        hvr_vertex_set_uint64(SUPERNODE_LBL, min_supernode_lbl, vertex,
                ctx);

        hvr_release_neighbors(&neighbors, ctx);
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

    feat1_min -= distance_threshold;
    feat1_max += distance_threshold;
    feat2_min -= distance_threshold;
    feat2_max += distance_threshold;
    feat3_min -= distance_threshold;
    feat3_max += distance_threshold;

    feat1_min = MAX(feat1_min, 0);
    feat2_min = MAX(feat2_min, 0);
    feat3_min = MAX(feat3_min, 0);

    feat1_max = MIN(feat1_max, domain_dim[0] - 1);
    feat2_max = MIN(feat2_max, domain_dim[1] - 1);
    feat3_max = MIN(feat3_max, domain_dim[2] - 1);

    unsigned n_interacting_partitions = 0;

    for (unsigned other_feat1_partition = feat1_min / partitions_size[0];
            other_feat1_partition <= feat1_max / partitions_size[0];
            other_feat1_partition++) {
        for (unsigned other_feat2_partition = feat2_min / partitions_size[1];
                other_feat2_partition <= feat2_max / partitions_size[1];
                other_feat2_partition++) {
            for (unsigned other_feat3_partition = feat3_min / partitions_size[2];
                    other_feat3_partition <= feat3_max / partitions_size[2];
                    other_feat3_partition++) {

                unsigned this_part = other_feat1_partition *
                    partitions_by_dim[1] * partitions_by_dim[2] +
                    other_feat2_partition * partitions_by_dim[2] +
                    other_feat3_partition;
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

    if (argc != 14 && argc != 13) {
        fprintf(stderr, "usage: %s <time-limit-in-seconds> "
                "<distance-threshold> <domain-size-0> <domain-size-1> "
                "<domain-size-2> <pe-dim-0> <pe-dim-1> <pe-dim-2> "
                "<partition-dim-0> <partition-dim-1> <partition-dim-2>"
                "[<min-vertices-to-add> <max-vertices-to-add> | <test-file>]\n",
                argv[0]);
        return 1;
    }

    time_limit_s = atoi(argv[1]);
    distance_threshold = atof(argv[2]);
    domain_dim[0] = atoi(argv[3]);
    domain_dim[1] = atoi(argv[4]);
    domain_dim[2] = atoi(argv[5]);
    pe_chunks_by_dim[0] = atoi(argv[6]);
    pe_chunks_by_dim[1] = atoi(argv[7]);
    pe_chunks_by_dim[2] = atoi(argv[8]);
    partitions_by_dim[0] = atoi(argv[9]);
    partitions_by_dim[1] = atoi(argv[10]);
    partitions_by_dim[2] = atoi(argv[11]);

    char *test_filename = NULL;
    if (argc == 13) {
        test_filename = argv[12];
        min_n_vertices_to_add = 0;
        max_n_vertices_to_add = 0;
    } else {
        min_n_vertices_to_add = atoi(argv[12]);
        max_n_vertices_to_add = atoi(argv[13]);
    }

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

    if (test_filename) {
        FILE *fp = fopen(test_filename, "r");
        if (!fp) {
            fprintf(stderr, "Failed opening file \"%s\"\n", test_filename);
            abort();
        }

        char *line = NULL;
        size_t len = 0; // length of buffer, so strlen + 1
        ssize_t read;
        unsigned line_no = 0;
        while ((read = getline(&line, &len, fp)) != -1) {
            if (line_no % npes == pe) {
                hvr_vertex_t *vert = hvr_vertex_create(hvr_ctx);
                hvr_vertex_set_uint64(TYPE, NODE_TYPE, vert, hvr_ctx);
                unsigned feat = 1;

                unsigned start_index = 0;
                unsigned end_index = 1;
                while (line[end_index] != '\0' && line[end_index] != '\n') {
                    if (line[end_index] == ',') {
                        // Parse digit from start_index
                        line[end_index] = '\0';
                        hvr_vertex_set(feat++, atof(line + start_index), vert,
                                hvr_ctx);
                        start_index = end_index + 1;
                        end_index = start_index;
                    } else {
                        end_index++;
                    }
                }

                assert(start_index != end_index);
                line[end_index] = '\0';
                hvr_vertex_set(feat, atof(line + start_index), vert, hvr_ctx);
                assert(feat == 4);

                printf("PE %d creating vertex %f at (%f, %f, %f)\n",
                        pe,
                        hvr_vertex_get(4, vert, hvr_ctx),
                        hvr_vertex_get(1, vert, hvr_ctx),
                        hvr_vertex_get(2, vert, hvr_ctx),
                        hvr_vertex_get(3, vert, hvr_ctx));

                n_local_vertices++;
            }
            line_no++;
        }

        if (line) free(line);

        fclose(fp);
        shmem_barrier_all();
    }

    start_time = hvr_current_time_us();
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

    hvr_vertex_iter_t iter;
    hvr_vertex_iter_init(&iter, hvr_ctx);
    unsigned n_local_nodes = 0;
    unsigned n_local_supernodes = 0;
    unsigned min_node_edges = UINT_MAX;
    unsigned max_node_edges = 0;
    unsigned min_supernode_edges = UINT_MAX;
    unsigned max_supernode_edges = 0;
    for (int p = 0; p < npes; p++) {
        if (p == pe) {
#if 0
            FILE *dump_file = (pe == 0 ? fopen("supernodes.txt", "w") :
                    fopen("supernodes.txt", "a"));
            assert(dump_file);
#endif

            for (hvr_vertex_t *vertex = hvr_vertex_iter_next(&iter); vertex;
                    vertex = hvr_vertex_iter_next(&iter)) {
#if 0
                hvr_vertex_t **neighbors;
                hvr_edge_type_t *neighbor_dirs;
                int n_neighbors = hvr_get_neighbors(vertex, &neighbors,
                        &neighbor_dirs, hvr_ctx);
#endif

                if (hvr_vertex_get_uint64(TYPE, vertex, hvr_ctx) == NODE_TYPE) {
#if 0
                    min_node_edges = (n_neighbors < min_node_edges ?
                            n_neighbors : min_node_edges);
                    max_node_edges = (n_neighbors > max_node_edges ?
                            n_neighbors : max_node_edges);
                    if (test_filename) {
                        printf("Vertex %f, neighbors",
                                hvr_vertex_get(VERT_ID, vertex, hvr_ctx));
                        for (int i = 0; i < n_neighbors; i++) {
                            printf(" %f", hvr_vertex_get(VERT_ID, neighbors[i],
                                        hvr_ctx));
                        }
                        printf("\n");
                    }
#endif
                    n_local_nodes++;
                } else {
#if 0
                    min_supernode_edges = (n_neighbors < min_supernode_edges ?
                            n_neighbors : min_supernode_edges);
                    max_supernode_edges = (n_neighbors > max_supernode_edges ?
                            n_neighbors : max_supernode_edges);
                    if (test_filename) {
                        printf("Supernode %lu, %d neighbors, children",
                                hvr_vertex_get_uint64(SUPERNODE_ID, vertex,
                                    hvr_ctx), n_neighbors);
                        for (int i = 0; i < K; i++) {
                            printf(" %lu", hvr_vertex_get_uint64(1 + i, vertex,
                                        hvr_ctx));
                        }
                        printf("\n");
                    }

                    fprintf(dump_file, "Supernode %lu : community %lu :",
                            hvr_vertex_get_uint64(SUPERNODE_ID, vertex, hvr_ctx),
                            hvr_vertex_get_uint64(SUPERNODE_LBL, vertex, hvr_ctx));
                    for (int i = 0; i < K; i++) {
                        fprintf(dump_file, " %lu", hvr_vertex_get_uint64(1 + i,
                                    vertex, hvr_ctx));
                    }
                    fprintf(dump_file, "\n");
#endif

                    n_local_supernodes++;
                }

#if 0
                hvr_release_neighbors(neighbors, neighbor_dirs, n_neighbors,
                        hvr_ctx);
#endif
            }
#if 0
            fclose(dump_file);
#endif
        }

#if 0
        fflush(stdout);
#endif
        shmem_barrier_all();
    }

    fflush(stdout);
    shmem_barrier_all();

    printf("PE %d, %u nodes, %u supernodes. min, max node edges = %u, %u. min, "
            "max supernode edges = %u, %u. Coupled to %lu PEs.\n", pe,
            n_local_nodes, n_local_supernodes, min_node_edges, max_node_edges,
            min_supernode_edges, max_supernode_edges,
            hvr_set_count(hvr_ctx->coupled_pes));
    fflush(stdout);

    shmem_barrier_all();

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
