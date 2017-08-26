#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <shmem.h>
#include <time.h>
#include <sys/time.h>

#include "hoover.h"

static long long *p_wrk = NULL;
static int *p_wrk_int = NULL;
static long *p_sync = NULL;

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

void hvr_sparse_vec_set(const unsigned feature, const double val,
        hvr_sparse_vec_t *vec, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;

    for (int i = 0; i < vec->nfeatures; i++) {
        if (vec->features[i] == feature && vec->timestamp[i] == ctx->timestep) {
            // Only one element per timestep ever present
            vec->values[i] = val;
            return;
        }
    }

    const unsigned nfeatures = vec->nfeatures;
    if (nfeatures < HVR_MAX_SPARSE_VEC_CAPACITY) {
        vec->features[nfeatures] = feature;
        vec->values[nfeatures] = val;
        vec->timestamp[nfeatures] = ctx->timestep;
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
        vec->timestamp[index_of_oldest] = ctx->timestep;
    }
}

double hvr_sparse_vec_get(const unsigned feature, const hvr_sparse_vec_t *vec,
        hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    const uint64_t curr_timestamp = ctx->timestep;

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
    *out_ctx = new_ctx;
}

void hvr_init(const vertex_id_t n_local_vertices, hvr_sparse_vec_t *vertices,
        hvr_edge_set_t *edges,
        hvr_update_metadata_func update_metadata,
        hvr_sparse_vec_distance_measure_func distance_measure,
        hvr_check_abort_func check_abort, hvr_vertex_owner_func vertex_owner,
        const double connectivity_threshold, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *new_ctx = (hvr_internal_ctx_t *)in_ctx;

    assert(new_ctx->initialized == 0);
    new_ctx->initialized = 1;

    p_wrk = (long long *)shmem_malloc(
            SHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(*p_wrk));
    p_wrk_int = (int *)shmem_malloc(
            SHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(*p_wrk_int));
    p_sync = (long *)shmem_malloc(
            SHMEM_REDUCE_SYNC_SIZE * sizeof(*p_sync));
    assert(p_wrk && p_sync && p_wrk_int);

    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        p_sync[i] = SHMEM_SYNC_VALUE;
    }

    new_ctx->pe = shmem_my_pe();
    new_ctx->npes = shmem_n_pes();

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
    shmem_barrier_all();
    new_ctx->n_global_vertices = 0;
    for (unsigned p = 0; p < new_ctx->npes; p++) {
        new_ctx->n_global_vertices += new_ctx->vertices_per_pe[p];
    }

    new_ctx->vertices = vertices;

    new_ctx->edges = edges;

    new_ctx->update_metadata = update_metadata;
    new_ctx->distance_measure = distance_measure;
    new_ctx->check_abort = check_abort;
    new_ctx->vertex_owner = vertex_owner;
    new_ctx->connectivity_threshold = connectivity_threshold;

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

        hvr_clear_edge_set(ctx->edges);

        ctx->timestep += 1;

        for (unsigned p = 0; p < ctx->npes; p++) {
            const unsigned target_pe = (ctx->pe + p) % ctx->npes;

            for (vertex_id_t j = 0; j < ctx->vertices_per_pe[target_pe]; j++) {
                hvr_sparse_vec_t *other = &(ctx->vertices[j]);

                shmem_getmem(ctx->buffer, other, sizeof(*other), target_pe);

                for (vertex_id_t i = 0; i < ctx->n_local_vertices; i++) {
                    /*
                     * Never want to add an edge from a node to itself (at
                     * least, we don't have a use case for this yet).
                     */
                    if (target_pe == ctx->pe && i == j) continue;

                    hvr_sparse_vec_t *curr = &(ctx->vertices[i]);
                    const double distance = ctx->distance_measure(curr,
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

        abort = ctx->check_abort(ctx->vertices, ctx->n_local_vertices,
                ctx);

        if (ctx->strict_mode) {
            *(ctx->strict_counter_src) = 0;
            shmem_int_sum_to_all(ctx->strict_counter_dest,
                    ctx->strict_counter_src, 1, 0, 0, ctx->npes, p_wrk_int,
                    p_sync);
            shmem_barrier_all();
        }
    }

    free(neighbors);

    if (ctx->strict_mode) {
        while (1) {
            *(ctx->strict_counter_src) = 1;
            shmem_int_sum_to_all(ctx->strict_counter_dest,
                    ctx->strict_counter_src, 1, 0, 0, ctx->npes, p_wrk_int,
                    p_sync);
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
