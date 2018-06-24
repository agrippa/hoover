/* For license: see LICENSE.txt file at top-level */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <shmem.h>
#include <math.h>
#include <string.h>

#include <set>

#include <hoover.h>

#define MAX(a, b) (((a) > (b)) ? a : b)
#define MIN(a, b) (((a) < (b)) ? a : b)

#define MAX_SUBGRAPH_VERTICES 3
#define TOP_N 3

const unsigned PARTITION_DIM = 40U;
static double distance_threshold = 1.0;

typedef struct _adjacency_matrix_t {
    unsigned n_vertices;
    unsigned char matrix[MAX_SUBGRAPH_VERTICES][MAX_SUBGRAPH_VERTICES];
} adjacency_matrix_t;

typedef struct _subgraph_t {
    hvr_vertex_id_t vertices[MAX_SUBGRAPH_VERTICES];
    adjacency_matrix_t adjacency_matrix;
} subgraph_t;

typedef struct _pattern_count_t {
    adjacency_matrix_t matrix;
    unsigned count;
} pattern_count_t;

// Timing variables
static unsigned long long start_time = 0;
static unsigned long long time_limit_us = 0;
static long long elapsed_time = 0;
static long long max_elapsed, total_time;

// SHMEM variables
static int pe, npes;
long long p_wrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
long p_sync[SHMEM_REDUCE_SYNC_SIZE];

// HOOVER variables
static hvr_graph_id_t graph = HVR_INVALID_GRAPH;

// Application variables
static double max_feat_val = 0.0;
static double feat_range_per_partition = 0.0;
static double range_per_pe = 100;

#include <math.h>
#include <stdlib.h>

static int same_pattern(adjacency_matrix_t *a, adjacency_matrix_t *b) {
    if (a->n_vertices != b->n_vertices) return 0;

    for (unsigned i = 0; i < a->n_vertices; i++) {
        for (unsigned j = 0; j < a->n_vertices; j++) {
            if (a->matrix[i][j] != b->matrix[i][j]) {
                return 0;
            }
        }
    }
    return 1;
}

// static void adjacency_matrix_to_string(adjacency_matrix_t *a, char *buf,
//         size_t buf_size) {
//     unsigned buf_index = 0;
// 
//     int nwritten = snprintf(buf + buf_index, buf_size - buf_index,
//             "[# vertices = %d]\n", a->n_vertices);
//     assert(nwritten > 0 && (unsigned)nwritten < buf_size - buf_index);
//     buf_index += nwritten;
// 
//     for (int i = 0; i < MAX_SUBGRAPH_VERTICES; i++) {
//         int nwritten = snprintf(buf + buf_index, buf_size - buf_index,
//                 "[");
//         assert(nwritten > 0 && (unsigned)nwritten < buf_size - buf_index);
//         buf_index += nwritten;
// 
//         for (int j = 0; j < MAX_SUBGRAPH_VERTICES; j++) {
//             int nwritten = snprintf(buf + buf_index, buf_size - buf_index,
//                     " %d", a->matrix[i][j]);
//             assert(nwritten > 0 && (unsigned)nwritten < buf_size - buf_index);
//             buf_index += nwritten;
//         }
// 
//         nwritten = snprintf(buf + buf_index, buf_size - buf_index, "]\n");
//         assert(nwritten > 0 && (unsigned)nwritten < buf_size - buf_index);
//         buf_index += nwritten;
//     }
// }

static unsigned adjacency_matrix_n_edges(adjacency_matrix_t *a) {
    unsigned count_edges = 0;
    for (unsigned i = 0; i < MAX_SUBGRAPH_VERTICES; i++) {
        for (unsigned j = 0; j <= i; j++) {
            if (a->matrix[i][j]) {
                count_edges++;
            }
        }
    }
    return count_edges;
}

static int index_in_subgraph(hvr_vertex_id_t vertex, subgraph_t *graph) {
    for (unsigned i = 0; i < graph->adjacency_matrix.n_vertices; i++) {
        if (graph->vertices[i] == vertex) return i;
    }
    return -1;
}

static int already_in_subgraph(hvr_vertex_id_t vertex, subgraph_t *graph) {
    return index_in_subgraph(vertex, graph) >= 0;
}

static int subgraph_has_edge(hvr_vertex_id_t a, hvr_vertex_id_t b,
        subgraph_t *graph) {
    int a_index = index_in_subgraph(a, graph);
    int b_index = index_in_subgraph(b, graph);
    assert(a_index >= 0 && a_index < MAX_SUBGRAPH_VERTICES);
    assert(b_index >= 0 && b_index < MAX_SUBGRAPH_VERTICES);

    return graph->adjacency_matrix.matrix[a_index][b_index] ||
        graph->adjacency_matrix.matrix[b_index][a_index];
}

static void subgraph_add_edge(hvr_vertex_id_t a, hvr_vertex_id_t b,
        subgraph_t *graph) {
    int a_index = index_in_subgraph(a, graph);
    int b_index = index_in_subgraph(b, graph);
    assert(a_index >= 0 && a_index < MAX_SUBGRAPH_VERTICES);
    assert(b_index >= 0 && b_index < MAX_SUBGRAPH_VERTICES);

    // Bi directional for now
    graph->adjacency_matrix.matrix[a_index][b_index] = 1;
    graph->adjacency_matrix.matrix[b_index][a_index] = 1;
}

static void explore_subgraphs(subgraph_t *curr_state,
        pattern_count_t **known_patterns, unsigned *n_known_patterns,
        hvr_ctx_t ctx, std::set<hvr_vertex_id_t> &visited, unsigned *n_explores,
        unsigned max_explores) {
    if (*n_explores >= max_explores) return;
    *n_explores += 1;

    /*
     * Track the existence of this state so we can find the most common
     * states after this traversal
     */
    int found = 0;
    for (unsigned i = 0; i < *n_known_patterns && !found; i++) {
        if (same_pattern(&(curr_state->adjacency_matrix),
                    &((*known_patterns)[i].matrix))) {
            // Increment count for this pattern
            (*known_patterns)[i].count += 1;
            found = 1;
        }
    }
    if (!found) {
        *known_patterns = (pattern_count_t *)realloc(*known_patterns,
                (*n_known_patterns + 1) * sizeof(pattern_count_t));
        assert(*known_patterns);

        pattern_count_t *new_pattern = (*known_patterns) + *n_known_patterns;
        memcpy(&(new_pattern->matrix), &(curr_state->adjacency_matrix),
                sizeof(new_pattern->matrix));
        new_pattern->count = 1;
        *n_known_patterns += 1;
    }

    // For each of the vertices already in this subgraph
    for (unsigned i = 0; i < curr_state->adjacency_matrix.n_vertices; i++) {
        hvr_vertex_id_t existing_vertex = curr_state->vertices[i];

        /*
         * Find the neighbors (i.e. halo regions) of the current vertex in the
         * subgraph. Check if adding any of them results in a change in graph
         * structure (addition of a vertex and/or edge). If it does, make that
         * change and explore further.
         */
        hvr_vertex_id_t *neighbors;
        unsigned n_neighbors;
        hvr_sparse_vec_get_neighbors(existing_vertex, ctx, &neighbors,
                &n_neighbors);

        // if (n_neighbors > 0) {
        //     fprintf(stderr, "Found %u neighbors for vertex %lu\n", n_neighbors,
        //             existing_vertex);
        // }

        for (unsigned j = 0; j < n_neighbors; j++) {
            hvr_vertex_id_t neighbor = neighbors[j];

            if (already_in_subgraph(neighbor, curr_state)) {
                /*
                 * 'neighbor' is already in the subgraph (as is
                 * 'existing_vertex'). If the edge
                 * 'neighbor'<->'existing_vertex' is already in the subgraph, do
                 * nothing. Otherwise, add it and iterate.
                 */
                if (!subgraph_has_edge(neighbor, existing_vertex, curr_state)) {
                    subgraph_t new_state;
                    memcpy(&new_state, curr_state, sizeof(new_state));
                    subgraph_add_edge(neighbor, existing_vertex, &new_state);
                    explore_subgraphs(&new_state, known_patterns,
                            n_known_patterns, ctx, visited, n_explores,
                            max_explores);
                }
            } else {
                // Can only add a new vertex to the graph if we have space to.
                if (curr_state->adjacency_matrix.n_vertices < MAX_SUBGRAPH_VERTICES) {
                    // Add the new vertex and an edge to it.
                    subgraph_t new_state;
                    memcpy(&new_state, curr_state, sizeof(new_state));
                    new_state.vertices[new_state.adjacency_matrix.n_vertices] =
                        neighbor;
                    new_state.adjacency_matrix.n_vertices += 1;
                    subgraph_add_edge(neighbor, existing_vertex, &new_state);
                    explore_subgraphs(&new_state, known_patterns,
                            n_known_patterns, ctx, visited, n_explores,
                            max_explores);
                }
            }
        }
        free(neighbors);
    }
}

static unsigned score_pattern(pattern_count_t *pattern) {
    return pattern->count * adjacency_matrix_n_edges(&pattern->matrix);
}

// Taken from https://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/
double randn(double mu, double sigma) {
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

uint16_t actor_to_partition(hvr_sparse_vec_t *actor, hvr_ctx_t ctx) {
    int feat1_partition = hvr_sparse_vec_get(0, actor, ctx) /
        feat_range_per_partition;
    int feat2_partition = hvr_sparse_vec_get(1, actor, ctx) /
        feat_range_per_partition;
    int feat3_partition = hvr_sparse_vec_get(2, actor, ctx) /
        feat_range_per_partition;
    return feat1_partition * PARTITION_DIM * PARTITION_DIM +
        feat2_partition * PARTITION_DIM + feat3_partition;
}

void start_time_step(hvr_vertex_iter_t *iter, hvr_ctx_t ctx) {
    /*
     * On each time step, calculate a random number of vertices to insert. Then,
     * use hvr_sparse_vec_create_n to insert them with 3 randomly generated
     * features which are designed to be most likely to just interact with
     * vertices on this node (but possibly with vertices on other nodes).
     */
    int n_vertices_to_add = rand() % 100;

    double mean = (pe * range_per_pe) + (range_per_pe / 2.0);
    double sigma = range_per_pe / 2.0;

    hvr_sparse_vec_t *new_vertices = hvr_sparse_vec_create_n(n_vertices_to_add,
            graph, ctx);

    for (int i = 0; i < n_vertices_to_add; i++) {
        /*
         * For each PE, have each feature have a gaussian distribution with a
         * mean of 'pe + 1' and a standard deviation of 0.5.
         */
        double feat1 = floor(randn(mean, sigma));
        double feat2 = floor(randn(mean, sigma));
        double feat3 = floor(randn(mean, sigma));

        feat1 = MAX(feat1, 0);
        feat2 = MAX(feat2, 0);
        feat3 = MAX(feat3, 0);

        feat1 = MIN(feat1, max_feat_val - 1);
        feat2 = MIN(feat2, max_feat_val - 1);
        feat3 = MIN(feat3, max_feat_val - 1);

        assert(feat1 >= 0 && feat2 >= 0 && feat3 >= 0);

        hvr_sparse_vec_set(0, feat1, &new_vertices[i], ctx);
        hvr_sparse_vec_set(1, feat2, &new_vertices[i], ctx);
        hvr_sparse_vec_set(2, feat3, &new_vertices[i], ctx);
    }

}

void update_metadata(hvr_sparse_vec_t *vertex, hvr_sparse_vec_t *neighbors,
        const size_t n_neighbors, hvr_set_t *couple_with, hvr_ctx_t ctx) {
    /*
     * NOOP. Vertices never move, and so we never need to use this to update
     * their features.
     *
     * Conceivably, HOOVER could use the fact that update_metadata is never used
     * as an optimization to avoid re-checking vertices which haven't been
     * updated.
     */
}

int might_interact(const uint16_t partition, hvr_set_t *partitions,
        uint16_t *interacting_partitions,
        unsigned *n_interacting_partitions,
        unsigned interacting_partitions_capacity,
        hvr_ctx_t ctx) {
    /*
     * If a vertex in 'partition' may create an edge with any vertices in any
     * partition in 'partitions', they might interact (so return 1).
     */
    uint16_t feat1_partition = partition / (PARTITION_DIM * PARTITION_DIM);
    uint16_t feat2_partition = (partition / PARTITION_DIM) % PARTITION_DIM;
    uint16_t feat3_partition = partition % PARTITION_DIM;

    double feat1_min = feat1_partition * feat_range_per_partition;
    double feat1_max = (feat1_partition + 1) * feat_range_per_partition;

    double feat2_min = feat2_partition * feat_range_per_partition;
    double feat2_max = (feat2_partition + 1) * feat_range_per_partition;

    double feat3_min = feat3_partition * feat_range_per_partition;
    double feat3_max = (feat3_partition + 1) * feat_range_per_partition;

    feat1_min -= distance_threshold;
    feat1_max += distance_threshold;
    feat2_min -= distance_threshold;
    feat2_max += distance_threshold;
    feat3_min -= distance_threshold;
    feat3_max += distance_threshold;

    feat1_min = MAX(feat1_min, 0.0);
    feat2_min = MAX(feat2_min, 0.0);
    feat3_min = MAX(feat3_min, 0.0);

    feat1_max = MIN(feat1_max, max_feat_val - 1.0);
    feat2_max = MIN(feat2_max, max_feat_val - 1.0);
    feat3_max = MIN(feat3_max, max_feat_val - 1.0);

    *n_interacting_partitions = 0;

    for (int other_feat1_partition = feat1_min / feat_range_per_partition;
            other_feat1_partition < feat1_max / feat_range_per_partition;
            other_feat1_partition++) {
        for (int other_feat2_partition = feat2_min / feat_range_per_partition;
                other_feat2_partition < feat2_max / feat_range_per_partition;
                other_feat2_partition++) {
            for (int other_feat3_partition = feat3_min / feat_range_per_partition;
                    other_feat3_partition < feat3_max / feat_range_per_partition;
                    other_feat3_partition++) {

                unsigned this_part = other_feat1_partition * PARTITION_DIM *
                    PARTITION_DIM + other_feat2_partition * PARTITION_DIM +
                    other_feat3_partition;
                if (hvr_set_contains(this_part, partitions)) {
                    assert(*n_interacting_partitions + 1 <=
                            interacting_partitions_capacity);
                    interacting_partitions[*n_interacting_partitions] =
                        this_part;
                    *n_interacting_partitions += 1;
                }
            }
        }
    }

    return (*n_interacting_partitions > 0);
}

int check_abort(hvr_vertex_iter_t *iter, hvr_ctx_t ctx,
        hvr_sparse_vec_t *out_coupled_metric) {
    /*
     * Calculate the K most common subgraphs in our current partition of the
     * graph, which may imply fetching vertices from remote nodes.
     *
     * The best candidate subgraph is the one that minimizes the sum of the
     * descriptive length of the subgraph and the overall graph when compressed
     * by the subgraph. For simplicity, we define the descriptive length of a
     * graph as the number of edges.
     */

    pattern_count_t *known_patterns = NULL;
    unsigned n_known_patterns = 0;

    unsigned n_explores = 0;
    unsigned max_explores = 2048;
    for (hvr_sparse_vec_t *vertex = hvr_vertex_iter_next(iter); vertex;
            vertex = hvr_vertex_iter_next(iter)) {
        assert(vertex->id != HVR_INVALID_VERTEX_ID);

        subgraph_t sub;
        sub.adjacency_matrix.n_vertices = 1;
        sub.vertices[0] = vertex->id;
        memset(sub.adjacency_matrix.matrix, 0x00, MAX_SUBGRAPH_VERTICES *
                MAX_SUBGRAPH_VERTICES * sizeof(unsigned char));

        std::set<hvr_vertex_id_t> visited;
        explore_subgraphs(&sub, &known_patterns, &n_known_patterns, ctx,
                visited, &n_explores, max_explores);
    }

    // Sort known patterns by score, highest to lowest
    for (unsigned i = 0; i < n_known_patterns; i++) {
        unsigned best_score = score_pattern(known_patterns + i);
        unsigned best_score_index = i;
        for (unsigned j = i + 1; j < n_known_patterns; j++) {
            unsigned this_score = score_pattern(known_patterns + j);
            if (best_score < this_score) {
                best_score = this_score;
                best_score_index = j;
            }
        }

        pattern_count_t tmp;
        memcpy(&tmp, known_patterns + i, sizeof(tmp));
        memcpy(known_patterns + i, known_patterns + best_score_index,
                sizeof(tmp));
        memcpy(known_patterns + best_score_index, &tmp, sizeof(tmp));
    }

    if (n_known_patterns > 0) {
        fprintf(stderr, "PE %d found %u patterns on timestep %d using %d "
                "visits. Best score = %u, vertex count = %u, edge count = %u\n",
                pe, n_known_patterns, hvr_current_timestep(ctx), max_explores,
                score_pattern(known_patterns + 0),
                known_patterns[0].matrix.n_vertices,
                adjacency_matrix_n_edges(&known_patterns[0].matrix));
    }

    /*
     * TODO collapse the top N patterns into a 64-bit value and share with
     * everyone through coupled_metric.
     */
    hvr_sparse_vec_set(0, 0.0, out_coupled_metric, ctx);

    unsigned long long time_so_far = hvr_current_time_us() - start_time;
    return (time_so_far > time_limit_us);
}

int main(int argc, char **argv) {
    hvr_ctx_t hvr_ctx;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <time-limit-in-seconds> "
                "[distance-threshold]\n", argv[0]);
        return 1;
    }

    time_limit_us = atoi(argv[1]) * 1000 * 1000;
    if (argc > 2) {
        distance_threshold = atof(argv[2]);
    }

    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        p_sync[i] = SHMEM_SYNC_VALUE;
    }

    shmem_init();

    pe = shmem_my_pe();
    npes = shmem_n_pes();

    if (pe == 0) {
        fprintf(stderr, "%d PE(s) running...\n", npes);
    }

    hvr_ctx_create(&hvr_ctx);
    graph = hvr_graph_create(hvr_ctx);

    max_feat_val = npes * range_per_pe;
    feat_range_per_partition = max_feat_val / (double)PARTITION_DIM;

    if (pe == 0) {
        fprintf(stderr, "Maximum feature value = %f, feature range per "
                "partition = %f\n", max_feat_val, feat_range_per_partition);
    }

    hvr_init(PARTITION_DIM * PARTITION_DIM * PARTITION_DIM, // # partitions
            update_metadata,
            might_interact,
            check_abort,
            actor_to_partition,
            start_time_step,
            graph,
            distance_threshold, // Edge creation distance threshold
            0, // Min spatial feature inclusive
            1, // Max spatial feature inclusive
            MAX_TIMESTAMP,
            hvr_ctx);

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

    if (pe == 0) {
        fprintf(stderr, "%d PEs, total CPU time = %f ms, max elapsed = %f ms, "
                "%d iterations completed on PE 0\n", npes,
                (double)total_time / 1000.0, (double)max_elapsed / 1000.0,
                info.executed_timesteps);
    }

    hvr_finalize(hvr_ctx);

    shmem_finalize();

    return 0;
}
