#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <shmem.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#include "hoover.h"

hvr_sparse_vec_t *hvr_sparse_vec_create_n(const size_t nvecs) {
    hvr_sparse_vec_t *new_vecs = (hvr_sparse_vec_t *)shmem_malloc(
            nvecs * sizeof(*new_vecs));
    assert(new_vecs);
    for (size_t i = 0; i < nvecs; i++) {
        new_vecs[i].nfeatures = 0;
    }
    return new_vecs;
}

void hvr_sparse_vec_dump(hvr_sparse_vec_t *vec, char *buf,
        const size_t buf_size) {
    char *iter = buf;
    int first = 1;

    for (unsigned i = 0; i < vec->nfeatures; i++) {
        unsigned feat = vec->features[i];
        uint64_t timestamp = vec->timestamp[i];

        int is_max = 1;
        for (unsigned j = 0; j < vec->nfeatures; j++) {
            if (vec->features[j] == feat && vec->timestamp[j] > timestamp) {
                is_max = 0;
                break;
            }
        }

        if (is_max) {
            const int capacity = buf_size - (iter - buf);
            int written;
            if (first) { 
                written = snprintf(iter, capacity, "%u: %f", feat,
                        vec->values[i]);
            } else {
                written = snprintf(iter, capacity, ", %u: %f", feat,
                        vec->values[i]);
            }
            if (written <= 0 || written > capacity) {
                assert(0);
            }

            iter += written;
            first = 0;
        }
    }
}

static void hvr_sparse_vec_set_internal(const unsigned feature, const double val,
        hvr_sparse_vec_t *vec, const uint64_t timestep) {
    for (int i = 0; i < vec->nfeatures; i++) {
        if (vec->features[i] == feature && vec->timestamp[i] == timestep) {
            // Only one element per timestep ever present
            vec->values[i] = val;
            return;
        }
    }

    const unsigned nfeatures = vec->nfeatures;
    if (nfeatures < HVR_MAX_SPARSE_VEC_CAPACITY) {
        vec->features[nfeatures] = feature;
        vec->values[nfeatures] = val;
        vec->timestamp[nfeatures] = timestep;
        vec->nfeatures = nfeatures + 1;
    } else {
        // Find an old value of this feature to replace
        uint64_t oldest_timestamp = 0;
        int index_of_oldest = -1;
        for (int i = 0; i < HVR_MAX_SPARSE_VEC_CAPACITY; i++) {
            if (vec->features[i] == feature) {
                if (index_of_oldest < 0 ||
                        oldest_timestamp > vec->timestamp[i]) {
                    index_of_oldest = i;
                    oldest_timestamp = vec->timestamp[i];
                }
            }
        }
        /*
         * TODO For now we just always assert that we've found at least one item
         * to replace. Realistically, this should probably be tunable behavior
         * as it may cause problems. For example, what if you only have one
         * item to replace and you replace an old value that somewhere will
         * later request? If nothing else, the user might want an error emitted.
         */
        assert(index_of_oldest > 0);
        vec->values[index_of_oldest] = val;
        vec->timestamp[index_of_oldest] = timestep;
    }
}

void hvr_sparse_vec_set(const unsigned feature, const double val,
        hvr_sparse_vec_t *vec, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    hvr_sparse_vec_set_internal(feature, val, vec, ctx->timestep);
}

static double hvr_sparse_vec_get_internal(const unsigned feature,
        const hvr_sparse_vec_t *vec, const uint64_t curr_timestamp) {
    uint64_t best_timestamp_delta = 0;
    double value = 0.0;
    int found = 0;

    for (unsigned i = 0; i < vec->nfeatures; i++) {
        if (vec->features[i] == feature && vec->timestamp[i] < curr_timestamp) {
            uint64_t timestamp_delta = curr_timestamp - vec->timestamp[i];
            if (found == 0 || timestamp_delta < best_timestamp_delta) {
                value = vec->values[i];
                best_timestamp_delta = timestamp_delta;
                found = 1;
            }
        }
    }

    if (found == 0) {
        /*
         * TODO What to do in this situation should probably be also tunable.
         * i.e., do we grab whatever value we can even if its timestamp is
         * newer? Do we throw an error?
         */
        return 0.0;
    } else {
        return value;
    }
}

double hvr_sparse_vec_get(const unsigned feature, const hvr_sparse_vec_t *vec,
        hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    return hvr_sparse_vec_get_internal(feature, vec, ctx->timestep);
}

static int hvr_sparse_vec_contains(const unsigned feature,
        hvr_sparse_vec_t *vec) {
    for (unsigned i = 0; i < vec->nfeatures; i++) {
        if (vec->features[i] == feature) {
            return 1;
        }
    }
    return 0;
}

static int unsigned_int_cmp(const void *a_in, const void *b_in) {
    int *a = (int *)a_in;
    int *b = (int *)b_in;
    return *a - *b;
}

// features must be an array of at least length HVR_MAX_SPARSE_VEC_CAPACITY
static void hvr_sparse_vec_unique_features(hvr_sparse_vec_t *vec,
        const uint64_t timestep, unsigned *nfeatures_out, unsigned *features) {
    unsigned nfeatures = 0;
    for (unsigned i = 0; i < vec->nfeatures; i++) {
        int already_have = 0;
        for (unsigned j = 0; j < nfeatures; j++) {
            if (vec->timestamp[i] < timestep &&
                    vec->features[i] == features[j]) {
                already_have = 1;
                break;
            }
        }

        if (!already_have) {
            features[nfeatures++] = vec->features[i];
        }
    }

    qsort(features, nfeatures, sizeof(*features), unsigned_int_cmp);
    *nfeatures_out = nfeatures;
}

hvr_pe_neighbors_set_t *hvr_create_empty_pe_neighbors_set(hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    hvr_pe_neighbors_set_t *set = (hvr_pe_neighbors_set_t *)malloc(
            sizeof(*set));
    assert(set);

    const int nbytes = (ctx->npes + 8 - 1) / 8;
    set->bit_vector = (unsigned char *)malloc(nbytes);
    assert(set->bit_vector);
    memset(set->bit_vector, 0x00, nbytes);
    set->nbytes = nbytes;

    return set;
}

static void hvr_pe_neighbors_set_insert_internal(int pe,
        unsigned char *bit_vector) {
    const int byte = pe / 8;
    const int bit = pe % 8;
    const unsigned char old_val = bit_vector[byte];
    bit_vector[byte] = (old_val | (1 << bit));
}

void hvr_pe_neighbors_set_insert(int pe, hvr_pe_neighbors_set_t *set) {
    hvr_pe_neighbors_set_insert_internal(pe, set->bit_vector);
}

static void hvr_pe_neighbors_set_clear_internal(int pe,
        unsigned char *bit_vector) {
    const int byte = pe / 8;
    const int bit = pe % 8;
    const unsigned char old_val = bit_vector[byte];
    bit_vector[byte] = (old_val & ~(1 << bit));
}

void hvr_pe_neighbors_set_clear(int pe, hvr_pe_neighbors_set_t *set) {
    hvr_pe_neighbors_set_clear_internal(pe, set->bit_vector);
}

int hvr_pe_neighbors_set_contains_internal(int pe,
        unsigned char *bit_vector) {
    const int byte = pe / 8;
    const int bit = pe % 8;
    const unsigned char old_val = bit_vector[byte];
    if (old_val & (1 << bit)) {
        return 1;
    } else {
        return 0;
    }
}

int hvr_pe_neighbors_set_contains(int pe, hvr_pe_neighbors_set_t *set) {
    return hvr_pe_neighbors_set_contains_internal(pe, set->bit_vector);
}

static unsigned hvr_pe_neighbor_set_count_internal(unsigned char *bit_vector,
        unsigned nbytes) {
    unsigned count = 0;
    for (int byte = 0; byte < nbytes; byte++) {
        for (int bit = 0; bit < 8; bit++) {
            if (bit_vector[byte] & (1 << bit)) {
                count++;
            }
        }
    }
    return count;
}

unsigned hvr_pe_neighbor_set_count(hvr_pe_neighbors_set_t *set) {
    return hvr_pe_neighbor_set_count_internal(set->bit_vector, set->nbytes);
}

void hvr_pe_neighbor_set_destroy(hvr_pe_neighbors_set_t *set) {
    free(set->bit_vector);
    free(set);
}

static void hvr_pe_neighbor_set_wipe(hvr_pe_neighbors_set_t *set) {
    memset(set->bit_vector, 0x00, set->nbytes);
}

hvr_edge_set_t *hvr_create_empty_edge_set() {
    hvr_edge_set_t *new_set = (hvr_edge_set_t *)malloc(sizeof(*new_set));
    assert(new_set);
    new_set->tree = NULL;
    return new_set;
}

void hvr_add_edge(const vertex_id_t local_vertex_id,
        const vertex_id_t global_vertex_id, hvr_edge_set_t *set) {
    // If it already exists, just returns existing node in tree
    set->tree = hvr_tree_insert(set->tree, local_vertex_id);
    hvr_avl_tree_node_t *inserted = hvr_tree_find(set->tree, local_vertex_id);
    inserted->subtree = hvr_tree_insert(inserted->subtree, global_vertex_id);
}

int hvr_have_edge(const vertex_id_t local_vertex_id,
        const vertex_id_t global_vertex_id, hvr_edge_set_t *set) {
    hvr_avl_tree_node_t *inserted = hvr_tree_find(set->tree, local_vertex_id);
    if (inserted == NULL) {
        return 0;
    }
    return hvr_tree_find(inserted->subtree, global_vertex_id) != NULL;
}

size_t hvr_count_edges(const vertex_id_t local_vertex_id, hvr_edge_set_t *set) {
    hvr_avl_tree_node_t *found = hvr_tree_find(set->tree, local_vertex_id);
    if (found == NULL) return 0;
    else return hvr_tree_size(found->subtree);
}

void hvr_clear_edge_set(hvr_edge_set_t *set) {
    hvr_tree_destroy(set->tree);
    set->tree = NULL;
}

void hvr_release_edge_set(hvr_edge_set_t *set) {
    hvr_tree_destroy(set->tree);
    free(set);
}

static void hvr_print_edge_set_helper(hvr_avl_tree_node_t *tree,
        const int print_colon) {
    if (tree == NULL) {
        return;
    }
    hvr_print_edge_set_helper(tree->left, print_colon);
    hvr_print_edge_set_helper(tree->right, print_colon);
    if (print_colon) {
        printf("%lu: ", tree->key);
        hvr_print_edge_set_helper(tree->subtree, 0);
        printf("\n");
    } else {
        printf("%lu ", tree->key);
    }
}

void hvr_print_edge_set(hvr_edge_set_t *set) {
    if (set->tree == NULL) {
        printf("Empty set\n");
    } else {
        hvr_print_edge_set_helper(set->tree, 1);
    }
}

void hvr_ctx_create(hvr_ctx_t *out_ctx) {
    hvr_internal_ctx_t *new_ctx = (hvr_internal_ctx_t *)malloc(
            sizeof(*new_ctx));
    assert(new_ctx);
    memset(new_ctx, 0x00, sizeof(*new_ctx));

    new_ctx->pe = shmem_my_pe();
    new_ctx->npes = shmem_n_pes();

    *out_ctx = new_ctx;
}

static int boxes_within_range(hvr_sparse_vec_t *min1, hvr_sparse_vec_t *max1,
        hvr_sparse_vec_t *min2, hvr_sparse_vec_t *max2, const double range,
        const uint64_t timestep, const unsigned min_spatial_feature,
        const unsigned max_spatial_feature) {
    unsigned min1_features[HVR_MAX_SPARSE_VEC_CAPACITY];
    unsigned max1_features[HVR_MAX_SPARSE_VEC_CAPACITY];
    unsigned min2_features[HVR_MAX_SPARSE_VEC_CAPACITY];
    unsigned max2_features[HVR_MAX_SPARSE_VEC_CAPACITY];
    unsigned min1_nfeatures, max1_nfeatures, min2_nfeatures, max2_nfeatures;

    hvr_sparse_vec_unique_features(min1, timestep, &min1_nfeatures,
            min1_features);
    hvr_sparse_vec_unique_features(max1, timestep, &max1_nfeatures,
            max1_features);
    hvr_sparse_vec_unique_features(min2, timestep, &min2_nfeatures,
            min2_features);
    hvr_sparse_vec_unique_features(max2, timestep, &max2_nfeatures,
            max2_features);

    /*
     * Assert that all bounding boxes have the same dimensionality across the
     * same dimensions.
     */
    assert(min1_nfeatures == max1_nfeatures);
    assert(min1_nfeatures == min2_nfeatures);
    assert(min1_nfeatures == max2_nfeatures);

    for (unsigned i = 0; i < min1_nfeatures; i++) {
        assert(min1_features[i] == max1_features[i]);
        assert(min1_features[i] == min2_features[i]);
        assert(min1_features[i] == max2_features[i]);
    }

    for (unsigned feature = min_spatial_feature; feature <= max_spatial_feature;
            feature++) {
        const int overlapping = 
            (hvr_sparse_vec_get_internal(feature, max1, timestep) + range) >=
            (hvr_sparse_vec_get_internal(feature, min2, timestep)) &&
            (hvr_sparse_vec_get_internal(feature, max2, timestep)) >=
            (hvr_sparse_vec_get_internal(feature, min1, timestep) - range);
        if (!overlapping) {
            return 0;
        }
    }
    return 1;
}

static void lock_bounding_box_list(const int pe, hvr_internal_ctx_t *ctx) {
    int old_val;
    do {
        old_val = shmem_int_cswap(ctx->bounding_boxes_lock, 0, 1, pe);
    } while (old_val != 0);
}

static void unlock_bounding_box_list(const int pe, hvr_internal_ctx_t *ctx) {
    shmem_int_cswap(ctx->bounding_boxes_lock, 1, 0, pe);
}

static void update_neighbors_based_on_bounding_boxes(const uint64_t timestep,
        hvr_internal_ctx_t *ctx) {
    hvr_sparse_vec_t *my_mins = ctx->bounding_boxes + (2 * ctx->pe);
    hvr_sparse_vec_t *my_maxs = ctx->bounding_boxes + (2 * ctx->pe + 1);

    hvr_pe_neighbor_set_wipe(ctx->my_neighbors);

    lock_bounding_box_list(ctx->pe, ctx);

    for (int p = 0; p < ctx->npes; p++) {
        hvr_sparse_vec_t *mins = ctx->bounding_boxes + (2 * p);
        hvr_sparse_vec_t *maxs = ctx->bounding_boxes + (2 * p + 1);

        if (boxes_within_range(my_mins, my_maxs, mins, maxs,
                    ctx->connectivity_threshold, timestep,
                    ctx->min_spatial_feature, ctx->max_spatial_feature)) {
            hvr_pe_neighbors_set_insert(p, ctx->my_neighbors);
        }
    }

    unlock_bounding_box_list(ctx->pe, ctx);

    printf("PE %d is talking to %d other PEs\n", ctx->pe,
            hvr_pe_neighbor_set_count(ctx->my_neighbors));
}

static void update_bounding_box(hvr_sparse_vec_t *mins, hvr_sparse_vec_t *maxs,
        hvr_internal_ctx_t *ctx) {
    mins->nfeatures = 0;
    maxs->nfeatures = 0;

    for (unsigned i = 0; i < ctx->n_local_vertices; i++) {
        hvr_sparse_vec_t *curr = ctx->vertices + i;

        for (unsigned j = 0; j < curr->nfeatures; j++) {
            if (!hvr_sparse_vec_contains(curr->features[j], mins) ||
                    curr->values[j] < hvr_sparse_vec_get_internal(
                        curr->features[j], mins, 1)) {
                hvr_sparse_vec_set_internal(curr->features[j], curr->values[j],
                        mins, 0);
            }

            if (!hvr_sparse_vec_contains(curr->features[j], maxs) ||
                    curr->values[j] > hvr_sparse_vec_get_internal(
                        curr->features[j], maxs, 1)) {
                hvr_sparse_vec_set_internal(curr->features[j], curr->values[j],
                        maxs, 0);
            }
        }
    }

    ctx->bounding_boxes_timestamps[ctx->pe] = ctx->timestep - 1;
}

void hvr_init(const vertex_id_t n_local_vertices, hvr_sparse_vec_t *vertices,
        hvr_edge_set_t *edges,
        hvr_update_metadata_func update_metadata,
        hvr_check_abort_func check_abort, hvr_vertex_owner_func vertex_owner,
        const double connectivity_threshold, const unsigned min_spatial_feature,
        const unsigned max_spatial_feature, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *new_ctx = (hvr_internal_ctx_t *)in_ctx;

    assert(new_ctx->initialized == 0);
    new_ctx->initialized = 1;

    new_ctx->p_wrk = (long long *)shmem_malloc(
            SHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(long long));
    new_ctx->p_wrk_int = (int *)shmem_malloc(
            SHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(int));
    new_ctx->p_sync = (long *)shmem_malloc(
            SHMEM_REDUCE_SYNC_SIZE * sizeof(long));
    assert(new_ctx->p_wrk && new_ctx->p_sync && new_ctx->p_wrk_int);

    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        (new_ctx->p_sync)[i] = SHMEM_SYNC_VALUE;
    }

    new_ctx->buffer = (hvr_sparse_vec_t *)shmem_malloc(
            sizeof(*(new_ctx->buffer)));
    assert(new_ctx->buffer);

    new_ctx->n_local_vertices = n_local_vertices;
    new_ctx->vertices_per_pe = (long long *)shmem_malloc(
            new_ctx->npes * sizeof(long long ));
    assert(new_ctx->vertices_per_pe);
    for (unsigned p = 0; p < new_ctx->npes; p++) {
        shmem_longlong_p(&(new_ctx->vertices_per_pe[new_ctx->pe]),
                n_local_vertices, p);
    }

    /*
     * Need a barrier to ensure everyone has done their puts before summing into
     * n_global_vertices.
     */
    shmem_barrier_all();

    new_ctx->n_global_vertices = 0;
    for (unsigned p = 0; p < new_ctx->npes; p++) {
        new_ctx->n_global_vertices += new_ctx->vertices_per_pe[p];
    }

    new_ctx->vertices = vertices;

    new_ctx->edges = edges;

    new_ctx->update_metadata = update_metadata;
    new_ctx->check_abort = check_abort;
    new_ctx->vertex_owner = vertex_owner;
    new_ctx->connectivity_threshold = connectivity_threshold;
    assert(min_spatial_feature <= max_spatial_feature);
    new_ctx->min_spatial_feature = min_spatial_feature;
    new_ctx->max_spatial_feature = max_spatial_feature;

    if (getenv("HVR_STRICT")) {
        if (new_ctx->pe == 0) {
            fprintf(stderr, "WARNING: Running in strict mode, this will lead "
                    "to degraded performance.\n");
        }
        new_ctx->strict_mode = 1;
        new_ctx->strict_counter_src = (int *)shmem_malloc(sizeof(int));
        new_ctx->strict_counter_dest = (int *)shmem_malloc(sizeof(int));
        assert(new_ctx->strict_counter_src && new_ctx->strict_counter_dest);
    }

    new_ctx->my_neighbors = hvr_create_empty_pe_neighbors_set(new_ctx);
    new_ctx->bounding_boxes_lock = (int *)shmem_malloc(sizeof(int));
    assert(new_ctx->bounding_boxes_lock);
    *(new_ctx->bounding_boxes_lock) = 0;
    new_ctx->bounding_boxes = hvr_sparse_vec_create_n(new_ctx->npes * 2);
    new_ctx->bounding_boxes_buffer = hvr_sparse_vec_create_n(new_ctx->npes * 2);
    new_ctx->bounding_boxes_timestamps = (long long *)shmem_malloc(
            new_ctx->npes * sizeof(long long));
    new_ctx->bounding_boxes_timestamps_buffer = (long long *)shmem_malloc(
            new_ctx->npes * sizeof(long long));
    assert(new_ctx->bounding_boxes_timestamps &&
            new_ctx->bounding_boxes_timestamps_buffer);
    update_bounding_box(new_ctx->bounding_boxes + (2 * new_ctx->pe),
            new_ctx->bounding_boxes  + (2 * new_ctx->pe + 1), new_ctx);

    for (int p = 0; p < new_ctx->npes; p++) {
        if (p == new_ctx->pe) continue;
        shmem_putmem(new_ctx->bounding_boxes + (2 * new_ctx->pe),
                new_ctx->bounding_boxes + (2 * new_ctx->pe),
                2 * sizeof(hvr_sparse_vec_t), p);
        (new_ctx->bounding_boxes_timestamps)[p] = 0;
    }
    shmem_barrier_all();

    update_neighbors_based_on_bounding_boxes(1, new_ctx);
}

static double sparse_vec_distance_measure(hvr_sparse_vec_t *a,
        hvr_sparse_vec_t *b, hvr_internal_ctx_t *ctx) {
    double acc = 0.0;
    for (unsigned f = ctx->min_spatial_feature; f <= ctx->max_spatial_feature;
            f++) {
        const double delta = hvr_sparse_vec_get(f, b, ctx) -
            hvr_sparse_vec_get(f, a, ctx);
        acc = acc + (delta * delta);
    }
    return sqrt(acc);
}

void hvr_body(hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;

    shmem_barrier_all();

    size_t neighbors_capacity = 256;
    vertex_id_t *neighbors = (vertex_id_t *)malloc(
            neighbors_capacity * sizeof(*neighbors));
    hvr_sparse_vec_t *neighbor_data = (hvr_sparse_vec_t *)malloc(
            neighbors_capacity * sizeof(*neighbor_data));
    assert(neighbors && neighbor_data);

    ctx->timestep = 1;
    int abort = 0;
    while (!abort) {
        const unsigned long long start_iter = hvr_current_time_us();

        for (vertex_id_t i = 0; i < ctx->n_local_vertices; i++) {
            hvr_avl_tree_node_t *vertex_edge_tree = hvr_tree_find(
                    ctx->edges->tree, ctx->vertices[i].id);

            const size_t old_neighbors_capacity = neighbors_capacity;
            const size_t n_neighbors = hvr_tree_linearize(&neighbors,
                    &neighbors_capacity, vertex_edge_tree->subtree);
            if (neighbors_capacity != old_neighbors_capacity) {
                free(neighbor_data);
                neighbor_data = (hvr_sparse_vec_t *)malloc(
                        neighbors_capacity * sizeof(*neighbor_data));
                assert(neighbor_data);
            }

            for (unsigned n = 0; n < n_neighbors; n++) {
                unsigned other_pe;
                size_t local_offset;
                ctx->vertex_owner(neighbors[n], &other_pe, &local_offset);

                hvr_sparse_vec_t *other = &(ctx->vertices[local_offset]);
                shmem_getmem(ctx->buffer, other, sizeof(*other), other_pe);

                memcpy(neighbor_data + n, ctx->buffer, sizeof(*neighbor_data));
            }

            ctx->update_metadata(&(ctx->vertices[i]), neighbor_data,
                    n_neighbors, ctx);
        }

        const unsigned long long finished_updates = hvr_current_time_us();

        hvr_clear_edge_set(ctx->edges);

        ctx->timestep += 1;

        // Update my bounding box
        update_bounding_box(ctx->bounding_boxes + (2 * ctx->pe),
                ctx->bounding_boxes  + (2 * ctx->pe + 1), ctx);

        // Update who I think my neighbors are
        update_neighbors_based_on_bounding_boxes(ctx->timestep, ctx);

        // Share my updates with my neighbors
        for (unsigned p = 0; p < ctx->npes; p++) {
            if (hvr_pe_neighbors_set_contains(p, ctx->my_neighbors)) {
                // Lock the other PE's bounding box list, update my entry in it
                lock_bounding_box_list(p, ctx);

                shmem_getmem(ctx->bounding_boxes_buffer,
                        ctx->bounding_boxes,
                        2 * ctx->npes * sizeof(hvr_sparse_vec_t), p);
                shmem_getmem(ctx->bounding_boxes_timestamps_buffer,
                        ctx->bounding_boxes_timestamps,
                        ctx->npes * sizeof(long long), p);

                for (unsigned pp = 0; pp < ctx->npes; pp++) {
                    if (ctx->bounding_boxes_timestamps[pp] >
                            ctx->bounding_boxes_timestamps_buffer[pp]) {
                        shmem_putmem(ctx->bounding_boxes + (2 * pp),
                                ctx->bounding_boxes + (2 * pp),
                                2 * sizeof(hvr_sparse_vec_t), p);
                    }
                }

                shmem_fence();

                unlock_bounding_box_list(p, ctx);
            }
        }

        // For each PE
        for (unsigned p = 0; p < ctx->npes; p++) {
            const unsigned target_pe = (ctx->pe + p) % ctx->npes;
            if (!hvr_pe_neighbors_set_contains(target_pe, ctx->my_neighbors)) {
                continue;
            }

            // For each vertex stored on the other PE
            for (vertex_id_t j = 0; j < ctx->vertices_per_pe[target_pe]; j++) {
                hvr_sparse_vec_t *other = &(ctx->vertices[j]);

                // Fetch this vertex using getmem
                shmem_getmem(ctx->buffer, other, sizeof(*other), target_pe);

                /*
                 * For each local vertex, check if we want to add an edge from
                 * other to this.
                 */
                for (vertex_id_t i = 0; i < ctx->n_local_vertices; i++) {
                    /*
                     * Never want to add an edge from a node to itself (at
                     * least, we don't have a use case for this yet).
                     */
                    if (target_pe == ctx->pe && i == j) continue;

                    hvr_sparse_vec_t *curr = &(ctx->vertices[i]);

                    const double distance = sparse_vec_distance_measure(curr,
                            ctx->buffer, ctx);
#ifdef VERBOSE
                    char buf1[1024];
                    char buf2[1024];
                    hvr_sparse_vec_dump(curr, buf1, 1024);
                    hvr_sparse_vec_dump(ctx->buffer, buf2, 1024);

                    printf("%s edge from %lu (%s) -> %lu (%s), dist = %f "
                            "threshold = %f, timestep %lu\n",
                            (distance < ctx->connectivity_threshold) ?
                            "Adding" : "Not adding", curr->id, buf1,
                            ctx->buffer->id, buf2, distance,
                            ctx->connectivity_threshold, ctx->timestep);
#endif
                    if (distance < ctx->connectivity_threshold) {
                        // Add edge
                        hvr_add_edge(curr->id, ctx->buffer->id, ctx->edges);
                    }
                }
            }
        }

        const unsigned long long finished_edge_adds = hvr_current_time_us();

        abort = ctx->check_abort(ctx->vertices, ctx->n_local_vertices,
                ctx);

        const unsigned long long finished_check_abort = hvr_current_time_us();

        printf("PE %d - total %f ms - metadata %f ms - edges %f ms - abort %f "
                "ms\n", ctx->pe,
                (double)(finished_check_abort - start_iter) / 1000.0,
                (double)(finished_updates - start_iter) / 1000.0,
                (double)(finished_edge_adds - finished_updates) / 1000.0,
                (double)(finished_check_abort - finished_edge_adds) / 1000.0);

        if (ctx->strict_mode) {
            *(ctx->strict_counter_src) = 0;
            shmem_int_sum_to_all(ctx->strict_counter_dest,
                    ctx->strict_counter_src, 1, 0, 0, ctx->npes, ctx->p_wrk_int,
                    ctx->p_sync);
            shmem_barrier_all();
        }
    }

    free(neighbors);

    if (ctx->strict_mode) {
        while (1) {
            *(ctx->strict_counter_src) = 1;
            shmem_int_sum_to_all(ctx->strict_counter_dest,
                    ctx->strict_counter_src, 1, 0, 0, ctx->npes, ctx->p_wrk_int,
                    ctx->p_sync);
            shmem_barrier_all();
            if (*(ctx->strict_counter_dest) == ctx->npes) {
                break;
            }
        }
    }
}

void hvr_finalize(hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    free(ctx);
}

uint64_t hvr_current_timestep(hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    return ctx->timestep;
}

unsigned long long hvr_current_time_us() {
    struct timeval curr_time;
    gettimeofday(&curr_time, NULL);
    return curr_time.tv_sec * 1000000ULL + curr_time.tv_usec;
}
