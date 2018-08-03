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

#include <set>
#include <vector>
#include <algorithm>

#include <hoover.h>
#include <shmem_rw_lock.h>

#define MAX(a, b) (((a) > (b)) ? a : b)
#define MIN(a, b) (((a) < (b)) ? a : b)

// Maximum # of vertices allowed in a subgraph
#define MAX_SUBGRAPH_VERTICES 5

// Number of patterns to share with other PEs
#define N_PATTERNS_SHARED 5

#define N_PATTERNS_TO_CONSIDER 6

/*
 * Maximum distance between two patterns for them to be considered anomalous
 * relative to each other. Must be >= 1.
 */
#define MAX_DISTANCE_FOR_ANOMALY 1

// const unsigned PARTITION_DIM = 70U;
const unsigned PARTITION_DIM = 120U;
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

typedef struct _timestamped_pattern_count_t {
    hvr_time_t timestamp;
    unsigned n_patterns;
    pattern_count_t patterns[N_PATTERNS_SHARED];
} timestamped_pattern_count_t;

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
static timestamped_pattern_count_t *best_patterns = NULL;
static timestamped_pattern_count_t *best_patterns_buffer = NULL;
static long *best_patterns_lock = NULL;
static timestamped_pattern_count_t *neighbor_patterns_buffer = NULL;
static pattern_count_t *sorted_best_patterns = NULL;
static unsigned n_sorted_best_patterns = 0;

#define MAX_LOCAL_PATTERNS 200
static pattern_count_t *known_local_patterns = NULL;
static std::vector<int> **pes_sharing_local_patterns = NULL;
static unsigned n_known_local_patterns = 0;

static unsigned n_local_vertices = 0;

static FILE *pe_anomalies_fp = NULL;

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

static unsigned pattern_distance(adjacency_matrix_t *a, adjacency_matrix_t *b) {
    unsigned max_vertices = MAX(a->n_vertices, b->n_vertices);
    unsigned count_differences = 0;
    for (unsigned i = 0; i < max_vertices; i++) {
        for (unsigned j = 0; j <= i; j++) {
            if (a->matrix[i][j] != b->matrix[i][j]) {
                count_differences++;
            }
        }
    }
    return count_differences + abs((long int)(a->n_vertices - b->n_vertices));
}

static void adjacency_matrix_to_string(adjacency_matrix_t *a, char *buf,
        size_t buf_size) {
    unsigned buf_index = 0;

    int nwritten = snprintf(buf + buf_index, buf_size - buf_index,
            "[# vertices = %d]\n", a->n_vertices);
    assert(nwritten > 0 && (unsigned)nwritten < buf_size - buf_index);
    buf_index += nwritten;

    for (int i = 0; i < MAX_SUBGRAPH_VERTICES; i++) {
        int nwritten = snprintf(buf + buf_index, buf_size - buf_index,
                "[");
        assert(nwritten > 0 && (unsigned)nwritten < buf_size - buf_index);
        buf_index += nwritten;

        for (int j = 0; j < MAX_SUBGRAPH_VERTICES; j++) {
            int nwritten = snprintf(buf + buf_index, buf_size - buf_index,
                    " %d", a->matrix[i][j]);
            assert(nwritten > 0 && (unsigned)nwritten < buf_size - buf_index);
            buf_index += nwritten;
        }

        nwritten = snprintf(buf + buf_index, buf_size - buf_index, "]\n");
        assert(nwritten > 0 && (unsigned)nwritten < buf_size - buf_index);
        buf_index += nwritten;
    }
}

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

static void explore_subgraphs(hvr_vertex_id_t last_added,
        subgraph_t *curr_state,
        pattern_count_t *known_patterns,
        std::vector<int> **pes_sharing_local_patterns,
        unsigned *n_known_patterns, hvr_ctx_t ctx, unsigned *n_explores,
        unsigned curr_depth, unsigned max_depth, unsigned *count_local_gets,
        unsigned *count_remote_gets,
        unsigned long long *accum_get_neighbors_time) {
    static hvr_vertex_id_t *neighbors = NULL;
    static unsigned neighbors_capacity = 0;

    if (curr_depth >= max_depth) {
        return;
    }
    *n_explores += 1;

    /*
     * Track the existence of this state so we can find the most common
     * states after this traversal
     */
    int found = 0;
    for (unsigned i = 0; i < *n_known_patterns && !found; i++) {
        if (pattern_distance(&(curr_state->adjacency_matrix),
                    &(known_patterns[i].matrix)) == 0) {
            // Increment count for this pattern
            std::vector<int> *other_pes = pes_sharing_local_patterns[i];
            for (unsigned v = 0; v < curr_state->adjacency_matrix.n_vertices;
                    v++) {
                int owning_pe = VERTEX_ID_PE(curr_state->vertices[v]);
                if (owning_pe != pe && std::find(other_pes->begin(),
                            other_pes->end(), owning_pe) == other_pes->end()) {
                    other_pes->push_back(owning_pe);
                }
            }
            known_patterns[i].count += 1;
            found = 1;
        }
    }
    if (!found && curr_state->adjacency_matrix.n_vertices > 1) {
        if (*n_known_patterns >= MAX_LOCAL_PATTERNS) {
            fprintf(stderr, "ERROR: # patterns (%d) has exceeded maximum "
                    "(%d)\n", *n_known_patterns, MAX_LOCAL_PATTERNS);
            abort();
        }

        pattern_count_t *new_pattern = known_patterns + *n_known_patterns;
        memcpy(&(new_pattern->matrix), &(curr_state->adjacency_matrix),
                sizeof(new_pattern->matrix));
        new_pattern->count = 1;

        std::vector<int> *new_set = pes_sharing_local_patterns[*n_known_patterns];
        new_set->clear();
        for (unsigned v = 0; v < curr_state->adjacency_matrix.n_vertices; v++) {
            int owning_pe = VERTEX_ID_PE(curr_state->vertices[v]);
            if (owning_pe != pe && std::find(new_set->begin(), new_set->end(),
                        owning_pe) == new_set->end()) {
                new_set->push_back(owning_pe);
            }
        }

        *n_known_patterns += 1;
    }

    // For the most recently added vertex
    hvr_vertex_id_t existing_vertex = last_added;

    /*
     * Find the neighbors (i.e. halo regions) of the current vertex in the
     * subgraph. Check if adding any of them results in a change in graph
     * structure (addition of a vertex and/or edge). If it does, make that
     * change and explore further.
     */
    const unsigned long long start_get_neighbors = hvr_current_time_us();
    unsigned n_neighbors;
    hvr_sparse_vec_get_neighbors_with_metrics(existing_vertex, ctx,
            &neighbors, &n_neighbors, &neighbors_capacity, count_local_gets,
            count_remote_gets);
    *accum_get_neighbors_time += (hvr_current_time_us() - start_get_neighbors);

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
                explore_subgraphs(neighbor, &new_state, known_patterns,
                        pes_sharing_local_patterns, n_known_patterns, ctx,
                        n_explores, curr_depth + 1, max_depth, count_local_gets,
                        count_remote_gets, accum_get_neighbors_time);
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
                explore_subgraphs(neighbor, &new_state, known_patterns,
                        pes_sharing_local_patterns, n_known_patterns, ctx,
                        n_explores, curr_depth + 1, max_depth, count_local_gets,
                        count_remote_gets, accum_get_neighbors_time);
            }
        }
    }
    free(neighbors);
}

static unsigned score_pattern(pattern_count_t *pattern) {
    return pattern->count * adjacency_matrix_n_edges(&pattern->matrix);
}

hvr_partition_t actor_to_partition(hvr_sparse_vec_t *actor, hvr_ctx_t ctx) {
    int feat1_partition = hvr_sparse_vec_get(0, actor, ctx) /
        feat_range_per_partition;
    int feat2_partition = hvr_sparse_vec_get(1, actor, ctx) /
        feat_range_per_partition;
    int feat3_partition = hvr_sparse_vec_get(2, actor, ctx) /
        feat_range_per_partition;
    return feat1_partition * PARTITION_DIM * PARTITION_DIM +
        feat2_partition * PARTITION_DIM + feat3_partition;
}

// Assumes we already hold the write lock on best_patterns_lock for the local PE
static void update_patterns_from(timestamped_pattern_count_t *tmp_buffer,
        int target_pe) {
    hvr_rwlock_rlock(best_patterns_lock, target_pe);
    shmem_getmem(neighbor_patterns_buffer, best_patterns,
            npes * sizeof(timestamped_pattern_count_t), target_pe);
    hvr_rwlock_runlock(best_patterns_lock, target_pe);

    /*
     * Look to see if any of the patterns stored in our left neighbor are
     * more recent than ours
     */
    for (int i = 0; i < npes; i++) {
        if (neighbor_patterns_buffer[i].timestamp >
                tmp_buffer[i].timestamp) {
            memcpy(&tmp_buffer[i], &neighbor_patterns_buffer[i],
                    sizeof(timestamped_pattern_count_t));
        }
    }
}

static void sort_patterns_by_score(pattern_count_t *patterns,
        unsigned n_patterns) {
    for (unsigned i = 0; i < n_patterns; i++) {
        unsigned best_score = score_pattern(patterns + i);
        unsigned best_score_index = i;
        for (unsigned j = i + 1; j < n_patterns; j++) {
            unsigned this_score = score_pattern(patterns + j);
            if (best_score < this_score) {
                best_score = this_score;
                best_score_index = j;
            }
        }

        pattern_count_t tmp;
        memcpy(&tmp, patterns + i, sizeof(tmp));
        memcpy(patterns + i, patterns + best_score_index,
                sizeof(tmp));
        memcpy(patterns + best_score_index, &tmp, sizeof(tmp));
    }
}

void start_time_step(hvr_vertex_iter_t *iter, hvr_ctx_t ctx) {
    /*
     * On each time step, calculate a random number of vertices to insert. Then,
     * use hvr_sparse_vec_create_n to insert them with 3 randomly generated
     * features which are designed to be most likely to just interact with
     * vertices on this node (but possibly with vertices on other nodes).
     */
    const int n_vertices_to_add = rand() % 300;

    double mean = (pe * range_per_pe) + (range_per_pe / 2.0);
    double sigma = range_per_pe / 4.0;

    hvr_sparse_vec_t *new_vertices = hvr_sparse_vec_create_n(n_vertices_to_add,
            graph, ctx);
    n_local_vertices += n_vertices_to_add;

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

    /*
     * Calculate the K most common subgraphs in our current partition of the
     * graph, which may imply fetching vertices from remote nodes.
     *
     * The best candidate subgraph is the one that minimizes the sum of the
     * descriptive length of the subgraph and the overall graph when compressed
     * by the subgraph.
     */
    n_known_local_patterns = 0;
    unsigned n_explores = 0;
    unsigned n_local_gets = 0;
    unsigned n_remote_gets = 0;
    const unsigned long long start_search = hvr_current_time_us();
    unsigned long long accum_get_neighbors_time = 0;
    for (hvr_sparse_vec_t *vertex = hvr_vertex_iter_next(iter); vertex;
            vertex = hvr_vertex_iter_next(iter)) {
        assert(vertex->id != HVR_INVALID_VERTEX_ID);

        subgraph_t sub;
        sub.adjacency_matrix.n_vertices = 1;
        sub.vertices[0] = vertex->id;
        memset(sub.adjacency_matrix.matrix, 0x00, MAX_SUBGRAPH_VERTICES *
                MAX_SUBGRAPH_VERTICES * sizeof(unsigned char));

        unsigned n_this_explores = 0;
        unsigned max_depth = 5;
        explore_subgraphs(vertex->id, &sub, known_local_patterns,
            pes_sharing_local_patterns, &n_known_local_patterns, ctx,
            &n_this_explores, 0, max_depth, &n_local_gets, &n_remote_gets,
            &accum_get_neighbors_time);
        n_explores += n_this_explores;
    }

    sort_patterns_by_score(known_local_patterns, n_known_local_patterns);
    const unsigned long long end_search = hvr_current_time_us();

    if (n_known_local_patterns > 0) {
        fprintf(stderr, "PE %d found %u patterns on timestep %d using %d "
                "visits, %f ms to search (%f ms spent on neighbor fetching), "
                "%u local vertices in total. Best "
                "score = %u, vertex count = %u, edge count = %u. # local gets "
                "= %u, # remote gets = %u\n",
                pe, n_known_local_patterns, hvr_current_timestep(ctx),
                n_explores, (double)(end_search - start_search) / 1000.0,
                (double)accum_get_neighbors_time / 1000.0,
                n_local_vertices,
                score_pattern(known_local_patterns + 0),
                known_local_patterns[0].matrix.n_vertices,
                adjacency_matrix_n_edges(&known_local_patterns[0].matrix),
                n_local_gets, n_remote_gets);
    }

    /*
     * Update my remotely accessible best patterns from the patterns I just
     * computed locally and my neighbors' patterns
     */

    memcpy(best_patterns_buffer, best_patterns, npes * sizeof(*best_patterns));
    best_patterns_buffer[pe].timestamp = hvr_current_timestep(ctx);
    best_patterns_buffer[pe].n_patterns = MIN(N_PATTERNS_SHARED, n_known_local_patterns);
    memcpy(best_patterns_buffer[pe].patterns, known_local_patterns,
            best_patterns_buffer[pe].n_patterns * sizeof(pattern_count_t));

    if (pe > 0) {
        // Left neighbor
        update_patterns_from(best_patterns_buffer, pe - 1);
    }

    if (pe < npes - 1) {
        // Left neighbor
        update_patterns_from(best_patterns_buffer, pe + 1);
    }

    hvr_rwlock_wlock(best_patterns_lock, pe);
    memcpy(best_patterns, best_patterns_buffer, npes * sizeof(*best_patterns));
    hvr_rwlock_wunlock(best_patterns_lock, pe);

    /*
     * Compute the globally most common patterns based on best_patterns from all
     * PEs.
     */
    n_sorted_best_patterns = 0;
    for (int p = 0; p < npes; p++) {
        // Know that I'm the only writer, so no need to rlock best_patterns here
        for (unsigned i = 0; i < best_patterns[p].n_patterns; i++) {

            int found = 0;
            for (unsigned j = 0; j < n_sorted_best_patterns; j++) {
                if (pattern_distance(&sorted_best_patterns[j].matrix,
                            &best_patterns[p].patterns[i].matrix) == 0) {
                    sorted_best_patterns[j].count +=
                        best_patterns[p].patterns[i].count;
                    found = 1;
                }
            }

            if (!found) {
                memcpy(&sorted_best_patterns[n_sorted_best_patterns],
                        &best_patterns[p].patterns[i], sizeof(pattern_count_t));
                n_sorted_best_patterns++;
            }
        }
    }

    sort_patterns_by_score(sorted_best_patterns, n_sorted_best_patterns);
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

int might_interact(const hvr_partition_t partition, hvr_set_t *partitions,
        hvr_partition_t *interacting_partitions,
        unsigned *n_interacting_partitions,
        unsigned interacting_partitions_capacity,
        hvr_ctx_t ctx) {
    /*
     * If a vertex in 'partition' may create an edge with any vertices in any
     * partition in 'partitions', they might interact (so return 1).
     */
    hvr_partition_t feat1_partition = partition / (PARTITION_DIM * PARTITION_DIM);
    hvr_partition_t feat2_partition = (partition / PARTITION_DIM) % PARTITION_DIM;
    hvr_partition_t feat3_partition = partition % PARTITION_DIM;

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
        hvr_set_t *to_couple_with, hvr_sparse_vec_t *out_coupled_metric) {

    /*
     * Find anomalies based on the patterns in sorted_best_patterns. If
     * those anomalies have edges with vertices in other nodes, couple with
     * those other nodes and produce a report on the anomalies.
     *
     * We do this by looking for patterns known_local_patterns that have a low
     * pattern_distance to the top common patterns but are not in the top common
     * patterns.
     */

    unsigned count_frequent_pattern_matches = 0;
    unsigned count_distance_too_high = 0;
    double avg_distance_too_high = 0.0;
    for (unsigned j = 0; j < n_known_local_patterns; j++) {
        pattern_count_t *local_pattern = known_local_patterns + j;

        int is_a_frequent_pattern = 0;
        int is_similar_to_a_frequent_pattern = -1;
        unsigned lowest_distance = 0;

        for (unsigned i = 0; i < MIN(n_sorted_best_patterns,
                    N_PATTERNS_TO_CONSIDER); i++) {
            pattern_count_t *frequent_pattern = sorted_best_patterns + i;

            unsigned dist = pattern_distance(&(frequent_pattern->matrix),
                    &(local_pattern->matrix));
            if (dist == 0) {
                // Identical to a frequent pattern
                is_a_frequent_pattern = 1;
                break;
            } else if (dist > 0 && dist <= MAX_DISTANCE_FOR_ANOMALY) {
                /*
                 * This defines an anomaly as a local pattern that is similar
                 * but not identical to one of the top globally identified
                 * subgraph patterns but is not itself in the top patterns.
                 */
                if (is_similar_to_a_frequent_pattern < 0) {
                    is_similar_to_a_frequent_pattern = i;
                }
            } else {
                // dist > MAX_DISTANCE_FOR_ANOMALY
                if (lowest_distance == 0 || dist < lowest_distance) {
                    lowest_distance = dist;
                }
            }
        }

        if (is_a_frequent_pattern) {
            count_frequent_pattern_matches++;
        } else if (is_similar_to_a_frequent_pattern >= 0) {
            char buf[1024];
            fprintf(stderr, "PE %d found potentially anomalous pattern on "
                    "timestep %d!\n", pe, hvr_current_timestep(ctx));

            // if (pe_anomalies_fp == NULL) {
            //     sprintf(buf, "pe_%d.anomalies.txt", pe);
            //     pe_anomalies_fp = fopen(buf, "w");
            //     assert(pe_anomalies_fp);
            // }

            // fprintf(pe_anomalies_fp, "Found anomaly on timestep %d\n",
            //         hvr_current_timestep(ctx));
            // fprintf(pe_anomalies_fp, "Anomaly (count=%u):\n",
            //         local_pattern->count);
            // adjacency_matrix_to_string(&local_pattern->matrix, buf, 1024);
            // fprintf(pe_anomalies_fp, buf);
            // fprintf(pe_anomalies_fp, "Regular Pattern (count=%u):\n",
            //         sorted_best_patterns[is_similar_to_a_frequent_pattern].count);
            // adjacency_matrix_to_string(
            //         &sorted_best_patterns[is_similar_to_a_frequent_pattern].matrix, buf,
            //         1024);
            // fprintf(pe_anomalies_fp, buf);
            // fprintf(pe_anomalies_fp, "\n");

            // fflush(pe_anomalies_fp);

            /*
             * Become coupled with other PEs whose vertices are parts of this
             * anomalous pattern.
             */
            std::vector<int> *other_pes = pes_sharing_local_patterns[j];
            for (std::vector<int>::iterator i = other_pes->begin(),
                    e = other_pes->end(); i != e; i++) {
                int other_pe = *i;
                hvr_set_insert(other_pe, to_couple_with);
            }
        } else {
            avg_distance_too_high += lowest_distance;
            count_distance_too_high++;
        }
    }

    // TODO anything useful we'd like to use the coupled metric for?
    hvr_sparse_vec_set(0, 0.0, out_coupled_metric, ctx);

    unsigned long long time_so_far = hvr_current_time_us() - start_time;
    fprintf(stderr, "PE %d - check_abort - elapsed time = %f s, time limit = "
            "%f s, # local vertices = %u, # local patterns = %u, # frequent patterns = %u, # exact pattern matches = %u, # "
            "times distance too high = %u, avg distance too high = %f, aborting? %d\n", pe,
            (double)time_so_far / 1000000.0, (double)time_limit_us / 1000000.0,
            n_local_vertices, n_known_local_patterns, n_sorted_best_patterns, count_frequent_pattern_matches,
            count_distance_too_high,
            avg_distance_too_high / (double)count_distance_too_high,
            (time_so_far > time_limit_us));
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

    srand(123 + pe);

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
            &graph, 1,
            distance_threshold, // Edge creation distance threshold
            0, // Min spatial feature inclusive
            2, // Max spatial feature inclusive
            MAX_TIMESTAMP,
            hvr_ctx);

    best_patterns = (timestamped_pattern_count_t *)shmem_malloc(
            npes * sizeof(*best_patterns));
    assert(best_patterns);
    best_patterns_buffer = (timestamped_pattern_count_t *)malloc(
            npes * sizeof(*best_patterns_buffer));
    assert(best_patterns_buffer);
    neighbor_patterns_buffer = (timestamped_pattern_count_t *)malloc(
            npes * sizeof(*neighbor_patterns_buffer));
    assert(neighbor_patterns_buffer);
    best_patterns_lock = (long *)hvr_rwlock_create_n(1);
    assert(best_patterns_lock);

    sorted_best_patterns = (pattern_count_t *)malloc(
            npes * N_PATTERNS_SHARED * sizeof(*sorted_best_patterns));
    assert(sorted_best_patterns);

    known_local_patterns = (pattern_count_t *)malloc(
            MAX_LOCAL_PATTERNS * sizeof(*known_local_patterns));
    assert(known_local_patterns);
    pes_sharing_local_patterns = (std::vector<int> **)malloc(
            MAX_LOCAL_PATTERNS * sizeof(*pes_sharing_local_patterns));
    assert(pes_sharing_local_patterns);
    for (int i = 0; i < MAX_LOCAL_PATTERNS; i++) {
        pes_sharing_local_patterns[i] = new std::vector<int>();
        pes_sharing_local_patterns[i]->reserve(npes);
    }

    *best_patterns_lock = 0;
    for (int i = 0; i < npes; i++) {
        best_patterns[i].timestamp = -1;
    }

    shmem_barrier_all();

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
