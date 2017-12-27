#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <limits.h>

#include <shmem.h>
#include <shmemx.h>

#include "hoover.h"

#define EDGE_GET_BUFFERING 1024

static int have_default_sparse_vec_val = 0;
static double default_sparse_vec_val = 0.0;

static void lock_actor_to_partition(const int pe, hvr_internal_ctx_t *ctx) {
    shmem_set_lock(ctx->actor_to_partition_locks + pe);
}

static void unlock_actor_to_partition(const int pe, hvr_internal_ctx_t *ctx) {
    shmem_clear_lock(ctx->actor_to_partition_locks + pe);
}

static void lock_partition_time_window(const int pe, hvr_internal_ctx_t *ctx) {
    shmem_set_lock(ctx->partition_time_window_locks + pe);
}

static void unlock_partition_time_window(const int pe, hvr_internal_ctx_t *ctx) {
    shmem_clear_lock(ctx->partition_time_window_locks + pe);
}

hvr_sparse_vec_t *hvr_sparse_vec_create_n(const size_t nvecs) {
    hvr_sparse_vec_t *new_vecs = (hvr_sparse_vec_t *)shmem_malloc(
            nvecs * sizeof(*new_vecs));
    assert(new_vecs);
    for (size_t i = 0; i < nvecs; i++) {
        hvr_sparse_vec_init(&new_vecs[i]);
    }
    return new_vecs;
}

void hvr_sparse_vec_init(hvr_sparse_vec_t *vec) {
    memset(vec, 0x00, sizeof(*vec));
    vec->pe = shmem_my_pe();
    vec->cached_timestamp = -1;
}

static inline unsigned prev_bucket(const unsigned bucket) {
    return (bucket == 0) ? (HVR_BUCKETS - 1) : (bucket - 1);
}

static int uint_compare(const void *_a, const void *_b) {
    unsigned a = *((unsigned *)_a);
    unsigned b = *((unsigned *)_b);
    if (a < b) {
        return -1;
    } else if (a > b) {
        return 1;
    } else {
        return 0;
    }
}

/*
 * out_features must be at least of size HVR_MAX_FEATURES. This is not a cheap
 * call.
 */
static void hvr_sparse_vec_unique_features(hvr_sparse_vec_t *vec,
        unsigned *out_features, unsigned *n_out_features) {
    *n_out_features = 0;

    for (unsigned i = 0; i < HVR_BUCKETS; i++) {
        for (unsigned j = 0; j < vec->bucket_size[i]; j++) {
            const unsigned curr_feature = vec->features[i][j];
            int already_have = 0;

            for (unsigned k = 0; k < *n_out_features; k++) {
                if (out_features[k] == curr_feature) {
                    already_have = 1;
                    break;
                }
            }

            if (!already_have) {
                out_features[*n_out_features] = curr_feature;
                *n_out_features += 1;
            }
        }
    }

    qsort(out_features, *n_out_features, sizeof(*out_features), uint_compare);
}

static void set_helper(hvr_sparse_vec_t *vec, const unsigned curr_bucket,
        const unsigned feature, const double val) {
    // Handle an existing bucket for this timestep
    const unsigned bucket_size = vec->bucket_size[curr_bucket];
    for (unsigned i = 0; i < bucket_size; i++) {
        if (vec->features[curr_bucket][i] == feature) {
            // Replace
            vec->values[curr_bucket][i] = val;
            return;
        }
    }
    assert(bucket_size < HVR_BUCKET_SIZE); // Can hold all features
    vec->features[curr_bucket][bucket_size] = feature;
    vec->values[curr_bucket][bucket_size] = val;
    vec->bucket_size[curr_bucket] = bucket_size + 1;
}

/*
 * If set is called multiple times with the same timestep and feature, previous
 * values are wiped out.
 */
static void hvr_sparse_vec_set_internal(const unsigned feature,
        const double val, hvr_sparse_vec_t *vec, const int64_t timestep) {
    unsigned initial_bucket = prev_bucket(vec->next_bucket);

    unsigned curr_bucket = initial_bucket;
    do {
        // bucket size == 0 indicates an invalid bucket
        if (vec->bucket_size[curr_bucket] > 0) {
            if (vec->timestamps[curr_bucket] == timestep) {
                // Handle an existing bucket for this timestep
                set_helper(vec, curr_bucket, feature, val);
                return;
            } else if (vec->timestamps[curr_bucket] < timestep) {
                /*
                 * No need to iterate further, as buckets are sorted in
                 * descending order so we won't find the bucket for timestep
                 * below here.
                 */
                break;
            }
        }

        // Move to the next bucket
        curr_bucket = prev_bucket(curr_bucket);
    } while (curr_bucket != initial_bucket);

    // Create a new bucket for this timestep
    const unsigned bucket_to_replace = vec->next_bucket;

    vec->timestamps[bucket_to_replace] = -1;

    __sync_synchronize();

    vec->bucket_size[bucket_to_replace] = 0;

    if (vec->bucket_size[initial_bucket] > 0) {
        /*
         * If we have an existing bucket at initial_bucket, copy its contents
         * over and then update.
         */
        const unsigned initial_bucket_size = vec->bucket_size[initial_bucket];
        memcpy(vec->values[bucket_to_replace], vec->values[initial_bucket],
                initial_bucket_size * sizeof(double));
        memcpy(vec->features[bucket_to_replace], vec->features[initial_bucket],
                initial_bucket_size * sizeof(unsigned));
        vec->bucket_size[bucket_to_replace] = initial_bucket_size;
    }

    set_helper(vec, bucket_to_replace, feature, val);

    __sync_synchronize();

    vec->timestamps[bucket_to_replace] = timestep;
    __sync_synchronize();
    vec->next_bucket = (bucket_to_replace + 1) % HVR_BUCKETS;
}

void hvr_sparse_vec_set(const unsigned feature, const double val,
        hvr_sparse_vec_t *vec, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    hvr_sparse_vec_set_internal(feature, val, vec, ctx->timestep);
}

static int find_feature_in_bucket(const hvr_sparse_vec_t *vec,
        const unsigned curr_bucket, const unsigned feature,
        double *out_val) {
    const unsigned bucket_size = vec->bucket_size[curr_bucket];
    for (unsigned i = 0; i < bucket_size; i++) {
        if (vec->features[curr_bucket][i] == feature) {
            *out_val = vec->values[curr_bucket][i];
            return 1;
        }
    }
    return 0;
}

static int hvr_sparse_vec_get_internal(const unsigned feature,
        hvr_sparse_vec_t *vec, const int64_t curr_timestamp,
        double *out_val) {

    if (vec->cached_timestamp == curr_timestamp - 1 &&
            vec->cached_timestamp_index < HVR_BUCKETS &&
            vec->timestamps[vec->cached_timestamp_index] ==
                curr_timestamp - 1) {
        /*
         * If we might have the location of this timestamp in this vector
         * cached, double check and then use that information to do an O(1)
         * lookup if possible.
         */

        if (find_feature_in_bucket(vec, vec->cached_timestamp_index, feature,
                    out_val)) {
            return 1;
        }
    }

    unsigned initial_bucket = prev_bucket(vec->next_bucket);

    unsigned curr_bucket = initial_bucket;
    do {
        // bucket size == 0 indicates an invalid bucket
        if (vec->bucket_size[curr_bucket] == 0) break;

        if (vec->timestamps[curr_bucket] >= 0 &&
                vec->timestamps[curr_bucket] < curr_timestamp) {
            // Handle finding an existing bucket for this timestep
            if (find_feature_in_bucket(vec, curr_bucket, feature, out_val)) {
                vec->cached_timestamp = vec->timestamps[curr_bucket];
                vec->cached_timestamp_index = curr_bucket;
                return 1;
            }

            // const unsigned bucket_size = vec->bucket_size[curr_bucket];
            // for (unsigned i = 0; i < bucket_size; i++) {
            //     if (vec->features[curr_bucket][i] == feature) {
            //         *out_val = vec->values[curr_bucket][i];
            //         return 1;
            //     }
            // }
            break;
        }

        // Move to the next bucket
        curr_bucket = prev_bucket(curr_bucket);
    } while (curr_bucket != initial_bucket);

    if (have_default_sparse_vec_val) {
        *out_val = default_sparse_vec_val;
        return 1;
    }  else {
        return 0;
    }
}

double hvr_sparse_vec_get(const unsigned feature, hvr_sparse_vec_t *vec,
        hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    double result;
    if (hvr_sparse_vec_get_internal(feature, vec, ctx->timestep, &result)) {
        return result;
    } else {
        /*
         * TODO What to do in this situation should probably be also tunable.
         * i.e., do we grab whatever value we can even if its timestamp is
         * newer? Do we throw an error? Do we return 0.0?
         */
        assert(0);
    }
}

static void hvr_sparse_vec_dump_internal(hvr_sparse_vec_t *vec, char *buf,
        const size_t buf_size, const int64_t timestep) {
    char *iter = buf;
    int first = 1;

    unsigned n_features;
    unsigned features[HVR_BUCKET_SIZE];
    hvr_sparse_vec_unique_features(vec, features, &n_features);

    for (unsigned i = 0; i < n_features; i++) {
        const unsigned feat = features[i];
        double val;

        const int err = hvr_sparse_vec_get_internal(feat, vec, timestep, &val);
        assert(err == 1);

        const int capacity = buf_size - (iter - buf);
        int written;
        if (first) {
            written = snprintf(iter, capacity, "%u: %f", feat, val);
        } else {
            written = snprintf(iter, capacity, ", %u: %f", feat, val);
        }
        if (written <= 0 || written > capacity) {
            assert(0);
        }

        iter += written;
        first = 0;
    }
}

void hvr_sparse_vec_dump(hvr_sparse_vec_t *vec, char *buf,
        const size_t buf_size, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    hvr_sparse_vec_dump_internal(vec, buf, buf_size, ctx->timestep);
}


void hvr_sparse_vec_set_id(const vertex_id_t id, hvr_sparse_vec_t *vec) {
    vec->id = id;
}

vertex_id_t hvr_sparse_vec_get_id(hvr_sparse_vec_t *vec) {
    return vec->id;
}

int hvr_sparse_vec_get_owning_pe(hvr_sparse_vec_t *vec) {
    return vec->pe;
}

// static int hvr_sparse_vec_timestamp_bounds(hvr_sparse_vec_t *vec,
//         uint64_t *out_min, uint64_t *out_max) {
//     unsigned initial_bucket;
//     if (vec->next_bucket == 0) {
//         initial_bucket = HVR_BUCKETS - 1;
//     } else {
//         initial_bucket = vec->next_bucket - 1;
//     }
// 
//     if (vec->bucket_size[initial_bucket] == 0) {
//         // No values;
//         return 0;
//     }
// 
//     uint64_t max_timestamp = vec->timestamps[initial_bucket];
//     uint64_t min_timestamp;
//     unsigned curr_bucket = initial_bucket;
//     do {
//         min_timestamp = vec->timestamps[curr_bucket];
// 
//         // Move to the next bucket
//         if (curr_bucket == 0) curr_bucket = HVR_BUCKETS - 1;
//         else curr_bucket -= 1;
//     } while (curr_bucket != initial_bucket && vec->bucket_size[curr_bucket] > 0);
// 
//     *out_min = min_timestamp;
//     *out_max = max_timestamp;
//     return 1;
// 
// }

#define HAS_TIMESTAMP -1
#define NEVER_TIMESTAMP -2

// static int hvr_sparse_vec_has_timestamp(hvr_sparse_vec_t *vec,
//         const uint64_t timestamp) {
//     unsigned initial_bucket;
//     if (vec->next_bucket == 0) initial_bucket = HVR_BUCKETS - 1;
//     else initial_bucket = vec->next_bucket - 1;
// 
//     if (vec->bucket_size[initial_bucket] == 0) {
//         // No values;
//         return 0;
//     }
// 
//     uint64_t min_timestamp;
//     unsigned curr_bucket = initial_bucket;
//     do {
//         if (vec->timestamps[curr_bucket] == timestamp ||
//                 vec->timestamps[curr_bucket] == INT64_MAX) {
//             return HAS_TIMESTAMP;
//         }
//         min_timestamp = vec->timestamps[curr_bucket];
// 
//         // Move to the next bucket
//         if (curr_bucket == 0) curr_bucket = HVR_BUCKETS - 1;
//         else curr_bucket -= 1;
//     } while (curr_bucket != initial_bucket && vec->bucket_size[curr_bucket] > 0);
// 
//     if (min_timestamp > timestamp) {
//         return NEVER_TIMESTAMP;
//     } else {
//         return min_timestamp;
//     }
// }

// static int find_time_slot_for(hvr_sparse_vec_t *vec,
//         const uint64_t target_timestamp) {
//     unsigned initial_bucket;
//     if (vec->next_bucket == 0) initial_bucket = HVR_BUCKETS - 1;
//     else initial_bucket = vec->next_bucket - 1;
// 
//     assert(vec->bucket_size[initial_bucket] > 0);
// 
//     int time_index = -1;
//     unsigned curr_bucket = initial_bucket;
//     do {
//         if (vec->timestamps[curr_bucket] < target_timestamp) {
//             break;
//         } else if (vec->timestamps[curr_bucket] == target_timestamp) {
//             time_index = curr_bucket;
//         } else if (vec->timestamps[curr_bucket] == INT64_MAX) {
//             time_index = curr_bucket;
//             break;
//         }
// 
//         // Move to the next bucket
//         if (curr_bucket == 0) curr_bucket = HVR_BUCKETS - 1;
//         else curr_bucket -= 1;
//     } while (curr_bucket != initial_bucket && vec->bucket_size[curr_bucket] > 0);
// 
//     return time_index;
// }

// static void hvr_sparse_vec_add_internal(hvr_sparse_vec_t *dst,
//         hvr_sparse_vec_t *src, const uint64_t target_timestamp) {
//     int dst_time_index = find_time_slot_for(dst, target_timestamp);
//     assert(dst_time_index >= 0);
// 
//     int src_time_index = find_time_slot_for(src, target_timestamp);
//     assert(src_time_index >= 0);
// 
//     const unsigned n_dst_features = dst->bucket_size[dst_time_index];
//     const unsigned n_src_features = src->bucket_size[src_time_index];
//     assert(n_dst_features == n_src_features);
// 
//     for (unsigned i = 0; i < n_dst_features; i++) {
//         unsigned feature = dst->features[dst_time_index][i];
//         unsigned dst_feature_index = i;
// 
//         int src_feature_index = -1;
//         for (unsigned j = 0; j < n_src_features; j++) {
//             if (src->features[src_time_index][j] == feature) {
//                 src_feature_index = j;
//                 break;
//             }
//         }
//         assert(src_feature_index >= 0);
// 
//         dst->values[dst_time_index][dst_feature_index] +=
//             src->values[src_time_index][src_feature_index];
//     }
// }

static int get_newest_timestamp(hvr_sparse_vec_t *vec, int64_t *out_timestamp) {
    const unsigned newest_bucket = prev_bucket(vec->next_bucket);

    if (vec->bucket_size[newest_bucket] == 0) {
        return 0;
    } else {
        *out_timestamp = vec->timestamps[newest_bucket];
        return 1;
    }
}

void hvr_sparse_vec_cache_clear(hvr_sparse_vec_cache_t *cache) {
    // Assumes already initialized, so must free any existing entries
    for (unsigned i = 0; i < HVR_CACHE_BUCKETS; i++) {
        hvr_sparse_vec_cache_node_t *iter = cache->buckets[i];
        while (iter) {
            hvr_sparse_vec_cache_node_t *next = iter->next;
            iter->next = cache->pool;
            cache->pool = iter;
            iter = next;
        }
        cache->buckets[i] = NULL;
        cache->bucket_size[i] = 0;
    }
}

void hvr_sparse_vec_cache_init(hvr_sparse_vec_cache_t *cache) {
    memset(cache, 0x00, sizeof(*cache));
}

hvr_sparse_vec_t *hvr_sparse_vec_cache_lookup(vertex_id_t vert,
        hvr_sparse_vec_cache_t *cache, int64_t timestep) {
    const unsigned bucket = vert % HVR_CACHE_BUCKETS;
    hvr_sparse_vec_cache_node_t *head = cache->buckets[bucket];
    hvr_sparse_vec_cache_node_t *iter = head;
    hvr_sparse_vec_cache_node_t *prev = NULL;
    while (iter) {
        if (iter->vec.id == vert) break;
        prev = iter;
        iter = iter->next;
    }

    if (iter == NULL) {
        return NULL;
    } else {
        // Decide whether this cached entry is new enough to be useful
        int64_t newest_timestamp;
        int success = get_newest_timestamp(&(iter->vec), &newest_timestamp);

        if (success && (newest_timestamp >= timestep ||
                    timestep - newest_timestamp <= 5)) {
            // Can use
            if (prev) {
                prev->next = iter->next;
                iter->next = cache->buckets[bucket];
                cache->buckets[bucket] = iter;
            }
            return &(iter->vec);
        } else {
            // Otherwise, evict
            if (prev) {
                prev->next = iter->next;
            } else {
                cache->buckets[bucket] = iter->next;
            }
            cache->bucket_size[bucket] -= 1;

            iter->next = cache->pool;
            cache->pool = iter;

            return NULL;
        }
    }
}

void hvr_sparse_vec_cache_insert(hvr_sparse_vec_t *vec,
        hvr_sparse_vec_cache_t *cache) {
    // Assume that vec is not already in the cache, but don't enforce this
    const unsigned bucket = vec->id % HVR_CACHE_BUCKETS;
    if (cache->bucket_size[bucket] == HVR_CACHE_MAX_BUCKET_SIZE) {
        // Evict and re-use
        hvr_sparse_vec_cache_node_t *iter = cache->buckets[bucket];
        hvr_sparse_vec_cache_node_t *prev = NULL;
        while (iter->next) {
            prev = iter;
            iter = iter->next;
        }
        // Re-use iter and place it at head
        prev->next = NULL;
        memcpy(&(iter->vec), vec, sizeof(*vec));
        iter->next = cache->buckets[bucket];
        cache->buckets[bucket] = iter;
    } else {
        // Allocate and insert
        hvr_sparse_vec_cache_node_t *new_node;
        if (cache->pool) {
            new_node = cache->pool;
            cache->pool = new_node->next;
        } else {
            new_node = (hvr_sparse_vec_cache_node_t *)malloc(sizeof(*new_node));
            assert(new_node);
        }
        memcpy(&(new_node->vec), vec, sizeof(*vec));
        new_node->next = cache->buckets[bucket];
        cache->buckets[bucket] = new_node;
        cache->bucket_size[bucket] += 1;
    }
}

static hvr_pe_set_t *hvr_create_empty_pe_set_helper(hvr_internal_ctx_t *ctx,
        const int nelements, bit_vec_element_type *bit_vector) {
    hvr_pe_set_t *set = (hvr_pe_set_t *)malloc(sizeof(*set));
    assert(set);

    set->bit_vector = bit_vector;
    set->nelements = nelements;

    memset(set->bit_vector, 0x00, nelements * sizeof(bit_vec_element_type));

    return set;
}

hvr_pe_set_t *hvr_create_empty_pe_set_symmetric_custom(const unsigned nvals,
        hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    const int nelements = (nvals + (sizeof(bit_vec_element_type) *
                BITS_PER_BYTE) - 1) / (sizeof(bit_vec_element_type) *
                BITS_PER_BYTE);
    bit_vec_element_type *bit_vector = (bit_vec_element_type *)shmem_malloc(
            nelements * sizeof(bit_vec_element_type));
    assert(bit_vector);
    return hvr_create_empty_pe_set_helper(ctx, nelements, bit_vector);
}

hvr_pe_set_t *hvr_create_empty_pe_set_symmetric(hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    return hvr_create_empty_pe_set_symmetric_custom(ctx->npes, ctx);
}

hvr_pe_set_t *hvr_create_empty_pe_set_custom(const unsigned nvals,
        hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    const int nelements = (nvals + (sizeof(bit_vec_element_type) *
                BITS_PER_BYTE) - 1) / (sizeof(bit_vec_element_type) *
                BITS_PER_BYTE);
    bit_vec_element_type *bit_vector = (bit_vec_element_type *)malloc(
            nelements * sizeof(bit_vec_element_type));
    assert(bit_vector);
    return hvr_create_empty_pe_set_helper(ctx, nelements, bit_vector);
}

hvr_pe_set_t *hvr_create_empty_pe_set(hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    return hvr_create_empty_pe_set_custom(ctx->npes, ctx);
}

static void hvr_pe_set_insert_internal(int pe,
        bit_vec_element_type *bit_vector) {
    const int element = pe / (sizeof(*bit_vector) * BITS_PER_BYTE);
    const int bit = pe % (sizeof(*bit_vector) * BITS_PER_BYTE);
    const bit_vec_element_type old_val = bit_vector[element];
    bit_vector[element] = (old_val | ((bit_vec_element_type)1 << bit));
}

void hvr_pe_set_insert(int pe, hvr_pe_set_t *set) {
    hvr_pe_set_insert_internal(pe, set->bit_vector);
}

static void hvr_pe_set_clear_internal(int pe,
        bit_vec_element_type *bit_vector) {
    const int element = pe / (sizeof(*bit_vector) * BITS_PER_BYTE);
    const int bit = pe % (sizeof(*bit_vector) * BITS_PER_BYTE);
    const bit_vec_element_type old_val = bit_vector[element];
    bit_vector[element] = (old_val & ~((bit_vec_element_type)1 << bit));
}

void hvr_pe_set_clear(int pe, hvr_pe_set_t *set) {
    hvr_pe_set_clear_internal(pe, set->bit_vector);
}

void hvr_pe_set_wipe(hvr_pe_set_t *set) {
    memset(set->bit_vector, 0x00, set->nelements * sizeof(*(set->bit_vector)));
}

int hvr_pe_set_contains_internal(int pe, bit_vec_element_type *bit_vector) {
    const int element = pe / (sizeof(*bit_vector) * BITS_PER_BYTE);
    const int bit = pe % (sizeof(*bit_vector) * BITS_PER_BYTE);
    const bit_vec_element_type old_val = bit_vector[element];
    if (old_val & ((bit_vec_element_type)1 << bit)) {
        return 1;
    } else {
        return 0;
    }
}

int hvr_pe_set_contains(int pe, hvr_pe_set_t *set) {
    return hvr_pe_set_contains_internal(pe, set->bit_vector);
}

static unsigned hvr_pe_set_count_internal(bit_vec_element_type *bit_vector,
        const unsigned nelements) {
    unsigned count = 0;
    for (int element = 0; element < nelements; element++) {
        for (int bit = 0; bit < sizeof(*bit_vector) * BITS_PER_BYTE; bit++) {
            if (bit_vector[element] & ((bit_vec_element_type)1 << bit)) {
                count++;
            }
        }
    }
    return count;
}

unsigned hvr_pe_set_count(hvr_pe_set_t *set) {
    return hvr_pe_set_count_internal(set->bit_vector, set->nelements);
}

void hvr_pe_set_destroy(hvr_pe_set_t *set) {
    free(set->bit_vector);
    free(set);
}

void hvr_pe_set_to_string(hvr_pe_set_t *set, char *buf, unsigned buflen) {
    int offset = snprintf(buf, buflen, "{");

    const size_t nvals = set->nelements * sizeof(bit_vec_element_type) *
        BITS_PER_BYTE;
    for (unsigned i = 0; i < nvals; i++) {
        if (hvr_pe_set_contains(i, set)) {
            offset += snprintf(buf + offset, buflen - offset - 1, " %u", i);
            assert(offset < buflen);
        }
    }

    snprintf(buf + offset, buflen - offset - 1, " }");
}

void hvr_pe_set_merge(hvr_pe_set_t *set, hvr_pe_set_t *other) {
    assert(set->nelements == other->nelements);

    for (int i = 0; i < set->nelements; i++) {
        (set->bit_vector)[i] |= (other->bit_vector)[i];
    }
}

void hvr_pe_set_merge_atomic(hvr_pe_set_t *set, hvr_pe_set_t *other) {
    assert(set->nelements == other->nelements);
    // Assert that we can use the long long atomics
    assert(sizeof(unsigned long long) == sizeof(bit_vec_element_type));

    for (int i = 0; i < set->nelements; i++) {
        shmemx_ulonglong_atomic_or(set->bit_vector + i, (other->bit_vector)[i],
                shmem_my_pe());
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

    new_ctx->pe = shmem_my_pe();
    new_ctx->npes = shmem_n_pes();

    if (getenv("HVR_DEFAULT_SPARSE_VEC_VAL")) {
        have_default_sparse_vec_val = 1;
        default_sparse_vec_val = atof(getenv("HVR_DEFAULT_SPARSE_VEC_VAL"));
    }

    *out_ctx = new_ctx;
}

/*
 * For every other PE in this simulation, reach out and take the bit vector of
 * partitions that they have had actors in during a recent time window.
 *
 * Then, for each partition set in their bit vector check locally if an actor in
 * that partition might interact with any actor in any partition we have actors
 * in locally. If so, add that PE as a neighbor.
 */
static void update_neighbors_based_on_partitions(hvr_internal_ctx_t *ctx) {
        hvr_pe_set_wipe(ctx->my_neighbors);

        for (unsigned p = 0; p < ctx->npes; p++) {

            lock_partition_time_window(p, ctx);
            shmem_getmem(ctx->other_pe_partition_time_window->bit_vector,
                    ctx->partition_time_window->bit_vector,
                    ctx->other_pe_partition_time_window->nelements * sizeof(bit_vec_element_type), p);
            unlock_partition_time_window(p, ctx);

            for (unsigned part = 0; part < ctx->n_partitions; part++) {
                if (hvr_pe_set_contains(part, ctx->other_pe_partition_time_window)) {
                    if (ctx->might_interact(part, ctx->partition_time_window, ctx)) {
                        hvr_pe_set_insert(p, ctx->my_neighbors);
                        break;
                    }
                }
            }
        }

#ifdef VERBOSE
    printf("PE %d is talking to %d other PEs\n", ctx->pe,
            hvr_pe_set_count(ctx->my_neighbors));
#endif
}

static double sparse_vec_distance_measure(hvr_sparse_vec_t *a,
        hvr_sparse_vec_t *b, const int64_t a_max_timestep,
        const int64_t b_max_timestep, const unsigned min_spatial_feature,
        const unsigned max_spatial_feature) {
    double acc = 0.0;
    for (unsigned f = min_spatial_feature; f <= max_spatial_feature; f++) {

        double a_val, b_val;
        const int a_err = hvr_sparse_vec_get_internal(f, a, a_max_timestep + 1,
                &a_val);
        assert(a_err == 1);
        const int b_err = hvr_sparse_vec_get_internal(f, b, b_max_timestep + 1,
                &b_val);
        assert(b_err == 1);

        const double delta = b_val - a_val;
        acc += (delta * delta);
    }
    return sqrt(acc);
}

/*
 * Check if an edge should be added from any locally stored actors to the jth
 * actor on PE target_pe (whose data is stored in vec).
 */
static void check_for_edge_to_add(hvr_sparse_vec_t *vec,
        const int target_pe, const int j, unsigned long long *update_edge_time,
        const int64_t other_pes_timestep, hvr_internal_ctx_t *ctx) {
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

        const unsigned long long start_update_time = hvr_current_time_us();
        const double distance = sparse_vec_distance_measure(curr,
                vec, ctx->timestep - 1, other_pes_timestep,
                ctx->min_spatial_feature, ctx->max_spatial_feature);
        const unsigned long long end_update_time = hvr_current_time_us();
        *update_edge_time += (end_update_time - start_update_time);

        if (distance < ctx->connectivity_threshold) {
            // Add edge
            hvr_add_edge(curr->id, vec->id, ctx->edges);
        }
    }
}

// GET the remote vector in a way that we don't see any partial updates.
static void get_remote_vec_nbi(hvr_sparse_vec_t *dst, hvr_sparse_vec_t *src,
        const int src_pe) {
    shmem_getmem_nbi(&(dst->next_bucket), &(src->next_bucket),
            sizeof(src->next_bucket), src_pe);

    shmem_fence();

    shmem_getmem_nbi(&(dst->timestamps[0]), &(src->timestamps[0]),
            HVR_BUCKETS * sizeof(src->timestamps[0]), src_pe);

    shmem_fence();

    shmem_getmem_nbi(dst, src,
            offsetof(hvr_sparse_vec_t, timestamps), src_pe);

    dst->cached_timestamp = -1;
    dst->cached_timestamp_index = 0;
}

static void update_edges(hvr_internal_ctx_t *ctx,
        unsigned long long *getmem_time, unsigned long long *update_edge_time) {
    // For each PE
    uint16_t *other_actor_to_partition_map = (uint16_t *)malloc(
            ctx->max_n_local_vertices * sizeof(uint16_t));
    assert(other_actor_to_partition_map);

    for (unsigned p = 0; p < ctx->npes; p++) {
        const unsigned target_pe = (ctx->pe + p) % ctx->npes;
        if (!hvr_pe_set_contains(target_pe, ctx->my_neighbors)) {
            continue;
        }

        // Grab the target PEs mapping from actors to partitions
        lock_actor_to_partition(target_pe, ctx);
        shmem_getmem(other_actor_to_partition_map, ctx->actor_to_partition_map,
                ctx->max_n_local_vertices * sizeof(uint16_t), target_pe);
        unlock_actor_to_partition(target_pe, ctx);

        int64_t other_pes_timestep;
        shmem_getmem(&other_pes_timestep, (int64_t *)ctx->symm_timestep,
                sizeof(other_pes_timestep), target_pe);
        /*
         * Prevent this PE from seeing into the future, if the other PE is ahead
         * of us.
         */
        if (other_pes_timestep > ctx->timestep - 1) {
            other_pes_timestep = ctx->timestep - 1;
        }

#define EDGE_CHECK_CHUNKING 8
        int filled = 0;
        vertex_id_t vecs_local_offset[EDGE_CHECK_CHUNKING];
        hvr_sparse_vec_t vecs[EDGE_CHECK_CHUNKING];

        // For each vertex on the remote PE
        for (vertex_id_t j = 0; j < ctx->vertices_per_pe[target_pe]; j++) {
            hvr_sparse_vec_t *other = &(ctx->vertices[j]);
            const unsigned long long start_time = hvr_current_time_us();
            const uint64_t actor_partition = other_actor_to_partition_map[j];

            /*
             * If actor j on the remote PE might interact with anything in our
             * local PE.
             */
            if (ctx->might_interact(actor_partition, ctx->partition_time_window,
                        ctx)) {
                get_remote_vec_nbi(vecs + filled, other, target_pe);

                vecs_local_offset[filled] = j;
                filled++;

                const unsigned long long end_getmem_time = hvr_current_time_us();
                *getmem_time += (end_getmem_time - start_time);

                if (filled == EDGE_CHECK_CHUNKING) {
                    shmem_quiet();

                    for (int f = 0; f < filled; f++) {
                        check_for_edge_to_add(&vecs[f], target_pe,
                                vecs_local_offset[f], update_edge_time,
                                other_pes_timestep, ctx);
                    }
                    filled = 0;
                }
            }
        }

        if (filled > 0) {
            shmem_quiet();

            for (int f = 0; f < filled; f++) {
                check_for_edge_to_add(&vecs[f], target_pe, vecs_local_offset[f],
                        update_edge_time, other_pes_timestep, ctx);
            }
        }
    }

    free(other_actor_to_partition_map);
}

/*
 * Update the mapping from each local actor to the partition it belongs to
 * (actor_to_partition_map) as well as information on the last timestep that
 * used each partition (last_timestep_using_partition). This mapping is stored
 * per-timestep, in a circular buffer.
 */
static void update_actor_partitions(hvr_internal_ctx_t *ctx) {
    lock_actor_to_partition(ctx->pe, ctx);

    for (unsigned a = 0; a < ctx->n_local_vertices; a++) {
        const uint16_t partition = ctx->actor_to_partition(
                ctx->vertices + a, ctx);
        assert(partition < ctx->n_partitions);

        // Update a mapping from local actor to the partition it belongs to
        (ctx->actor_to_partition_map)[a] = partition;

        /*
         * This doesn't necessarily need to be in the critical section, but to
         * avoid multiple traversal over all actors we stick it here.
         */
        (ctx->last_timestep_using_partition)[partition] = ctx->timestep;
    }

    unlock_actor_to_partition(ctx->pe, ctx);
}

/*
 * partition_time_window stores a list of the partitions that the local PE has
 * had actors inside during some window of recent timesteps. This updates the
 * partitions in that window set based on the results of
 * update_actor_partitions.
 */
static void update_partition_time_window(hvr_internal_ctx_t *ctx) {
    lock_partition_time_window(ctx->pe, ctx);

    hvr_pe_set_wipe(ctx->partition_time_window);

    for (unsigned p = 0; p < ctx->n_partitions; p++) {
        const int64_t last_use = ctx->last_timestep_using_partition[p];
        if (last_use >= 0) {
            assert(last_use <= ctx->timestep);
            if (ctx->timestep - last_use < HVR_BUCKETS) {
                hvr_pe_set_insert(p, ctx->partition_time_window);
            }
        }
    }

    unlock_partition_time_window(ctx->pe, ctx);
}

static void update_all_pe_timesteps_helper(const int target_pe,
        hvr_internal_ctx_t *ctx) {
    shmem_set_lock(ctx->all_pe_timesteps_locks + target_pe);

    shmem_getmem(ctx->all_pe_timesteps_buffer, ctx->all_pe_timesteps,
            ctx->npes * sizeof(int64_t), target_pe);
    int any_remote_updates = 0;
    for (int i = 0; i < ctx->npes; i++) {
        if (ctx->all_pe_timesteps[i] > ctx->all_pe_timesteps_buffer[i]) {
            // Update remote with newer timestep
            ctx->all_pe_timesteps_buffer[i] = ctx->all_pe_timesteps[i];
            any_remote_updates = 1;
        } else {
            /*
             * Update local with newer timestep, with either same or greater
             * timestep.
             */
            ctx->all_pe_timesteps[i] = ctx->all_pe_timesteps_buffer[i];
        }
    }

    if (any_remote_updates) {
        shmem_putmem(ctx->all_pe_timesteps, ctx->all_pe_timesteps_buffer,
                ctx->npes * sizeof(int64_t), target_pe);
    }

    shmem_clear_lock(ctx->all_pe_timesteps_locks + target_pe);
}

static void update_all_pe_timesteps(hvr_internal_ctx_t *ctx,
        const int pe_stencil) {
    // Just look at my pe + 1 and pe - 1 neighbors

    int start_pe = ctx->pe - pe_stencil;
    if (start_pe < 0) start_pe = 0;
    int end_pe = ctx->pe + pe_stencil;
    if (end_pe > ctx->npes - 1) end_pe = ctx->npes - 1;

    for (int target_pe = start_pe; target_pe <= end_pe; target_pe++) {
        if (target_pe == ctx->pe) continue;
        update_all_pe_timesteps_helper(target_pe, ctx);
    }
}

static void update_my_timestep(hvr_internal_ctx_t *ctx,
        const int64_t set_timestep) {
    shmem_set_lock(ctx->all_pe_timesteps_locks + ctx->pe);
    (ctx->all_pe_timesteps)[ctx->pe] = set_timestep;
    shmem_clear_lock(ctx->all_pe_timesteps_locks + ctx->pe);
}


static void oldest_pe_timestep(hvr_internal_ctx_t *ctx,
        int64_t *out_oldest_timestep, unsigned *out_oldest_pe) {
    shmem_set_lock(ctx->all_pe_timesteps_locks + ctx->pe);

    int64_t oldest_timestep = ctx->all_pe_timesteps[0];
    unsigned oldest_pe = 0;
    for (int i = 1; i < ctx->npes; i++) {
        if (ctx->all_pe_timesteps[i] < oldest_timestep) {
            oldest_timestep = ctx->all_pe_timesteps[i];
            oldest_pe = i;
        }
    }

    shmem_clear_lock(ctx->all_pe_timesteps_locks + ctx->pe);

    *out_oldest_timestep = oldest_timestep;
    *out_oldest_pe = oldest_pe;
}

void hvr_init(const uint16_t n_partitions, const vertex_id_t n_local_vertices,
        hvr_sparse_vec_t *vertices, hvr_update_metadata_func update_metadata,
        hvr_might_interact_func might_interact,
        hvr_check_abort_func check_abort, hvr_vertex_owner_func vertex_owner,
        hvr_actor_to_partition actor_to_partition,
        const double connectivity_threshold, const unsigned min_spatial_feature,
        const unsigned max_spatial_feature, const int64_t max_timestep,
        hvr_ctx_t in_ctx) {
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
            EDGE_GET_BUFFERING * sizeof(*(new_ctx->buffer)));
    assert(new_ctx->buffer);

    new_ctx->symm_timestep = (volatile int64_t *)shmem_malloc(
            sizeof(*(new_ctx->symm_timestep)));
    assert(new_ctx->symm_timestep);
    *(new_ctx->symm_timestep) = -1;

    new_ctx->all_pe_timesteps = (int64_t *)shmem_malloc(
            new_ctx->npes * sizeof(*(new_ctx->all_pe_timesteps)));
    assert(new_ctx->all_pe_timesteps);
    memset(new_ctx->all_pe_timesteps, 0x00,
        new_ctx->npes * sizeof(*(new_ctx->all_pe_timesteps)));

    new_ctx->all_pe_timesteps_buffer = (int64_t *)shmem_malloc(
            new_ctx->npes * sizeof(*(new_ctx->all_pe_timesteps_buffer)));
    assert(new_ctx->all_pe_timesteps_buffer);
    memset(new_ctx->all_pe_timesteps_buffer, 0x00,
        new_ctx->npes * sizeof(*(new_ctx->all_pe_timesteps_buffer)));

    new_ctx->all_pe_timesteps_locks = (long *)shmem_malloc(
            new_ctx->npes * sizeof(*(new_ctx->all_pe_timesteps_locks)));
    assert(new_ctx->all_pe_timesteps_locks);
    memset(new_ctx->all_pe_timesteps_locks, 0x00,
            new_ctx->npes * sizeof(*(new_ctx->all_pe_timesteps_locks)));

    new_ctx->last_timestep_using_partition = (int64_t *)malloc(
            n_partitions * sizeof(int64_t));
    assert(new_ctx->last_timestep_using_partition);
    for (unsigned i = 0; i < n_partitions; i++) {
        (new_ctx->last_timestep_using_partition)[i] = -1;
    }
    new_ctx->partition_time_window = hvr_create_empty_pe_set_symmetric_custom(
            n_partitions, new_ctx);

    new_ctx->actor_to_partition_locks = (long *)shmem_malloc(new_ctx->npes *
            sizeof(long));
    assert(new_ctx->actor_to_partition_locks);
    memset(new_ctx->actor_to_partition_locks, 0x00, new_ctx->npes *
            sizeof(long));

    new_ctx->partition_time_window_locks = (long *)shmem_malloc(new_ctx->npes *
            sizeof(long));
    assert(new_ctx->partition_time_window_locks);
    memset(new_ctx->partition_time_window_locks, 0x00,
            new_ctx->npes * sizeof(long));

    new_ctx->n_partitions = n_partitions;
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

    new_ctx->max_n_local_vertices = new_ctx->vertices_per_pe[0];
    new_ctx->n_global_vertices = 0;
    for (unsigned p = 0; p < new_ctx->npes; p++) {
        new_ctx->n_global_vertices += new_ctx->vertices_per_pe[p];
        if (new_ctx->vertices_per_pe[p] > new_ctx->max_n_local_vertices) {
            new_ctx->max_n_local_vertices = new_ctx->vertices_per_pe[p];
        }
    }

    new_ctx->vertices = vertices;

    new_ctx->actor_to_partition_map = (uint16_t *)shmem_malloc(HVR_BUCKETS *
            new_ctx->max_n_local_vertices * sizeof(*(new_ctx->actor_to_partition_map)));
    assert(new_ctx->actor_to_partition_map);

    new_ctx->update_metadata = update_metadata;
    new_ctx->might_interact = might_interact;
    new_ctx->check_abort = check_abort;
    new_ctx->vertex_owner = vertex_owner;
    new_ctx->actor_to_partition = actor_to_partition;
    new_ctx->connectivity_threshold = connectivity_threshold;
    assert(min_spatial_feature <= max_spatial_feature);
    new_ctx->min_spatial_feature = min_spatial_feature;
    new_ctx->max_spatial_feature = max_spatial_feature;
    new_ctx->max_timestep = max_timestep;

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

    if (getenv("HVR_TRACE_DUMP")) {
        char dump_file_name[256];
        sprintf(dump_file_name, "%d.csv", new_ctx->pe);

        new_ctx->dump_mode = 1;
        new_ctx->dump_file = fopen(dump_file_name, "w");
        assert(new_ctx->dump_file);
    }

    new_ctx->my_neighbors = hvr_create_empty_pe_set(new_ctx);

    new_ctx->coupled_pes = hvr_create_empty_pe_set_symmetric(new_ctx);
    hvr_pe_set_insert(new_ctx->pe, new_ctx->coupled_pes);
    new_ctx->coupled_pes_values = hvr_sparse_vec_create_n(new_ctx->npes);
    new_ctx->coupled_pes_values_buffer = hvr_sparse_vec_create_n(new_ctx->npes);
    new_ctx->coupled_locks = (long *)shmem_malloc(
            new_ctx->npes * sizeof(*(new_ctx->coupled_locks)));
    assert(new_ctx->coupled_locks);
    memset((long *)new_ctx->coupled_locks, 0x00,
            new_ctx->npes * sizeof(*(new_ctx->coupled_locks)));

    shmem_barrier_all();
}

static void update_local_actor_metadata(const vertex_id_t actor,
        hvr_pe_set_t *to_couple_with, unsigned long long *fetch_neighbors_time,
       unsigned long long *update_metadata_time, hvr_internal_ctx_t *ctx) {
    static size_t neighbors_capacity = 0;
    static vertex_id_t *neighbors = NULL;
    if (neighbors == NULL) {
        neighbors_capacity = 256;
        neighbors = (vertex_id_t *)malloc(neighbors_capacity *
                sizeof(*neighbors));
        assert(neighbors);
    }

    // The list of edges for local actor i
    hvr_avl_tree_node_t *vertex_edge_tree = hvr_tree_find(
            ctx->edges->tree, ctx->vertices[actor].id);

    // Update the metadata for actor i
    if (vertex_edge_tree != NULL) {
        // This vertex has edges
        const size_t n_neighbors = hvr_tree_linearize(&neighbors,
                &neighbors_capacity, vertex_edge_tree->subtree);

        // Simplifying assumption for now
        assert(n_neighbors < EDGE_GET_BUFFERING);

        const unsigned long long start_single_update = hvr_current_time_us();
        for (unsigned n = 0; n < n_neighbors; n++) {
            const vertex_id_t neighbor = neighbors[n];

            unsigned other_pe;
            size_t local_offset;
            ctx->vertex_owner(neighbor, &other_pe, &local_offset);

            hvr_sparse_vec_t *other = &(ctx->vertices[local_offset]);

            get_remote_vec_nbi(ctx->buffer + n, other, other_pe);
        }

        shmem_quiet();

        const unsigned long long finish_neighbor_fetch= hvr_current_time_us();
        *fetch_neighbors_time += (finish_neighbor_fetch - start_single_update);

        ctx->update_metadata(&(ctx->vertices[actor]), ctx->buffer, n_neighbors,
                to_couple_with, ctx);
        update_metadata_time += (hvr_current_time_us() - finish_neighbor_fetch);
    } else {
        const unsigned long long start_single_update = hvr_current_time_us();
        // This vertex has no edges
        ctx->update_metadata(&(ctx->vertices[actor]), NULL, 0, to_couple_with,
                ctx);
        update_metadata_time += (hvr_current_time_us() - start_single_update);
    }
}


void hvr_body(hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;

    shmem_barrier_all();

    ctx->other_pe_partition_time_window = hvr_create_empty_pe_set_custom(
            ctx->n_partitions, ctx);

    *(ctx->symm_timestep) = 0;
    ctx->timestep = 1;

    update_actor_partitions(ctx);
    update_partition_time_window(ctx);

    /*
     * Ensure everyone's partition windows are initialized before initializing
     * neighbors.
     */
    shmem_barrier_all();

    update_neighbors_based_on_partitions(ctx);

    // Initialize edges
    ctx->edges = hvr_create_empty_edge_set();
    unsigned long long unused;
    update_edges(ctx, &unused, &unused);

    hvr_pe_set_t *to_couple_with = hvr_create_empty_pe_set(ctx);

    int abort = 0;
    while (!abort && ctx->timestep < ctx->max_timestep) {
        const unsigned long long start_iter = hvr_current_time_us();

        unsigned long long fetch_neighbors_time = 0;
        unsigned long long update_metadata_time = 0;

        hvr_pe_set_wipe(to_couple_with);

        // Update each actor's metadata
        for (vertex_id_t i = 0; i < ctx->n_local_vertices; i++) {
            update_local_actor_metadata(i, to_couple_with,
                    &fetch_neighbors_time, &update_metadata_time, ctx);
        }

        const unsigned long long finished_updates = hvr_current_time_us();

        *(ctx->symm_timestep) = ctx->timestep;
        ctx->timestep += 1;

        // Update mapping from actors to partitions
        update_actor_partitions(ctx);

        const unsigned long long finished_actor_partitions = hvr_current_time_us();

        /*
         * Update a fuzzy window of partitions that have recently had local
         * actors in them.
         */
        update_partition_time_window(ctx);

        const unsigned long long finished_time_window = hvr_current_time_us();

        // Update neighboring PEs based on fuzzy partition windows
        update_neighbors_based_on_partitions(ctx);

        const unsigned long long finished_summary_update = hvr_current_time_us();

        // Update edges with actors in neighboring PEs
        hvr_clear_edge_set(ctx->edges);
        unsigned long long getmem_time = 0;
        unsigned long long update_edge_time = 0;
        update_edges(ctx, &getmem_time, &update_edge_time);

        const unsigned long long finished_edge_adds = hvr_current_time_us();

        hvr_sparse_vec_t coupled_metric;
        memcpy(&coupled_metric, ctx->coupled_pes_values + ctx->pe,
                sizeof(coupled_metric));
        abort = ctx->check_abort(ctx->vertices, ctx->n_local_vertices,
                ctx, &coupled_metric);

        // Update my local information on PEs I am coupled with.
        hvr_pe_set_merge_atomic(ctx->coupled_pes, to_couple_with);

        // Atomically update other PEs that I am coupled with.
        for (int p = 0; p < ctx->npes; p++) {
            if (p != ctx->pe && hvr_pe_set_contains(p, ctx->coupled_pes)) {
                for (int i = 0; i < ctx->coupled_pes->nelements; i++) {
                    shmemx_ulonglong_atomic_or(
                            ctx->coupled_pes->bit_vector + i,
                            (ctx->coupled_pes->bit_vector)[i], p);
                }
            }
        }

        const unsigned long long finished_neighbor_updates =
            hvr_current_time_us();

        // shmem_set_lock(ctx->coupled_locks + ctx->pe);
        // shmem_putmem(ctx->coupled_pes_values + ctx->pe, &coupled_metric,
        //         sizeof(coupled_metric), ctx->pe);
        // shmem_quiet();
        // shmem_clear_lock(ctx->coupled_locks + ctx->pe);

        // /*
        //  * For each PE I know I'm coupled with, lock their coupled_timesteps
        //  * list and update my copy with any newer entries in my
        //  * coupled_timesteps list.
        //  */
        // int ncoupled = 1; // include myself
        // for (int p = 0; p < ctx->npes; p++) {
        //     if (p == ctx->pe) continue;

        //     if (hvr_pe_set_contains(p, ctx->coupled_pes)) {

        //         /*
        //          * Wait until we've found an update to p's coupled value that is
        //          * for this timestep.
        //          */
        //         assert(hvr_sparse_vec_has_timestamp(&coupled_metric,
        //                     ctx->timestep) == HAS_TIMESTAMP);
        //         int other_has_timestamp = hvr_sparse_vec_has_timestamp(
        //                 ctx->coupled_pes_values + p, ctx->timestep);
        //         while (other_has_timestamp != HAS_TIMESTAMP) {
        //             shmem_set_lock(ctx->coupled_locks + p);

        //             // Pull in the coupled PEs current values
        //             shmem_getmem(ctx->coupled_pes_values_buffer,
        //                     ctx->coupled_pes_values,
        //                     ctx->npes * sizeof(hvr_sparse_vec_t), p);

        //             shmem_clear_lock(ctx->coupled_locks + p);

        //             // int i = p;
        //             for (int i = 0; i < ctx->npes; i++) {
        //                 hvr_sparse_vec_t *other =
        //                     ctx->coupled_pes_values_buffer + i;
        //                 hvr_sparse_vec_t *mine = ctx->coupled_pes_values + i;

        //                 uint64_t other_min_timestamp, other_max_timestamp;
        //                 uint64_t mine_min_timestamp, mine_max_timestamp;
        //                 const int other_success =
        //                     hvr_sparse_vec_timestamp_bounds(other,
        //                             &other_min_timestamp, &other_max_timestamp);
        //                 const int mine_success =
        //                     hvr_sparse_vec_timestamp_bounds(mine,
        //                             &mine_min_timestamp, &mine_max_timestamp);

        //                 if (other_success) {
        //                     if (!mine_success) {
        //                         memcpy(mine, other, sizeof(*mine));
        //                     } else if (other_max_timestamp > mine_max_timestamp) {
        //                         memcpy(mine, other, sizeof(*mine));
        //                     }
        //                 }
        //             }

        //             other_has_timestamp = hvr_sparse_vec_has_timestamp(
        //                     ctx->coupled_pes_values + p, ctx->timestep);
        //         }
        //         assert(other_has_timestamp != NEVER_TIMESTAMP);

        //         hvr_sparse_vec_add_internal(&coupled_metric,
        //                 ctx->coupled_pes_values + p, ctx->timestep);

        //         ncoupled++;
        //     }
        // }

        // /*
        //  * TODO coupled_metric here contains the aggregate values over all
        //  * coupled PEs, including this one.
        //  */
        // if (ncoupled > 1) {
        //     char buf[1024];
        //     hvr_sparse_vec_dump_internal(&coupled_metric, buf, 1024,
        //             ctx->timestep + 1);
        //     fprintf(stderr, "PE %d - computed coupled value {%s} from %d "
        //             "coupled PEs on timestep %lu\n", ctx->pe, buf, ncoupled,
        //             ctx->timestep - 1);
        // }

        /*
         * Throttle the progress of much faster PEs to ensure we don't get out
         * of range of timesteps on other PEs.
         */
        update_my_timestep(ctx, ctx->timestep);
        update_all_pe_timesteps(ctx, 1);
        unsigned nspins = 0;

        int64_t oldest_timestep;
        unsigned oldest_pe;
        oldest_pe_timestep(ctx, &oldest_timestep, &oldest_pe);

        while (ctx->timestep - oldest_timestep > HVR_BUCKETS / 2) {
            update_all_pe_timesteps(ctx, nspins + 1);
            oldest_pe_timestep(ctx, &oldest_timestep, &oldest_pe);
            nspins++;
        }

        const unsigned long long finished_check_abort = hvr_current_time_us();

        if (ctx->dump_mode) {
            // Assume that all vertices have the same features.
            unsigned nfeatures;
            unsigned features[HVR_BUCKET_SIZE];
            hvr_sparse_vec_unique_features(ctx->vertices, features, &nfeatures);

            for (unsigned v = 0; v < ctx->n_local_vertices; v++) {
                hvr_sparse_vec_t *vertex = ctx->vertices + v;
                fprintf(ctx->dump_file, "%lu,%u,%ld,%d", vertex->id, nfeatures,
                        ctx->timestep, ctx->pe);
                for (unsigned f = 0; f < nfeatures; f++) {
                    fprintf(ctx->dump_file, ",%u,%f", features[f],
                            hvr_sparse_vec_get(features[f], vertex, ctx));
                }
                fprintf(ctx->dump_file, ",,\n");
            }
        }


#ifdef VERBOSE
        char neighbors_str[1024];
        hvr_pe_set_to_string(ctx->my_neighbors, neighbors_str, 1024);

        char partition_time_window_str[1024];
        hvr_pe_set_to_string(ctx->partition_time_window,
                partition_time_window_str, 1024);
#endif

        printf("PE %d - total %f ms - metadata %f ms (%f %f) - summary %f ms (%f %f %f) - "
                "edges %f ms (%f %f) - neighbor updates %f ms - abort %f ms - "
                "%u spins - %u / %u PE neighbors %s - partition window = %s - "
                "aborting? %d - last step? %d\n", ctx->pe,
                (double)(finished_check_abort - start_iter) / 1000.0,
                (double)(finished_updates - start_iter) / 1000.0,
                (double)fetch_neighbors_time / 1000.0,
                (double)update_metadata_time / 1000.0,
                (double)(finished_summary_update - finished_updates) / 1000.0,
                (double)(finished_actor_partitions - finished_updates) / 1000.0,
                (double)(finished_time_window - finished_actor_partitions) / 1000.0,
                (double)(finished_summary_update - finished_time_window) / 1000.0,
                (double)(finished_edge_adds - finished_summary_update) / 1000.0,
                (double)getmem_time / 1000.0, (double)update_edge_time / 1000.0,
                (double)(finished_neighbor_updates - finished_edge_adds) / 1000.0,
                (double)(finished_check_abort - finished_neighbor_updates) / 1000.0,
                nspins, hvr_pe_set_count(ctx->my_neighbors), ctx->npes,
#ifdef VERBOSE
                neighbors_str, partition_time_window_str,
#else
                "", "",
#endif
                abort, ctx->timestep >= ctx->max_timestep);

        if (ctx->strict_mode) {
            *(ctx->strict_counter_src) = 0;
            shmem_int_sum_to_all(ctx->strict_counter_dest,
                    ctx->strict_counter_src, 1, 0, 0, ctx->npes, ctx->p_wrk_int,
                    ctx->p_sync);
            shmem_barrier_all();
        }
    }

    /*
     * As I'm exiting, ensure that any PEs that are coupled with me (or may make
     * themselves coupled with me in the future) never block on me.
     */
    shmem_set_lock(ctx->coupled_locks + ctx->pe);
    ctx->timestep += 1;
    unsigned n_features;
    unsigned features[HVR_BUCKET_SIZE];
    hvr_sparse_vec_unique_features(ctx->coupled_pes_values + ctx->pe, features,
            &n_features);
    for (unsigned f = 0; f < n_features; f++) {
        double last_val;
        int success = hvr_sparse_vec_get_internal(features[f],
                ctx->coupled_pes_values + ctx->pe, ctx->timestep + 1,
                &last_val);
        assert(success == 1);

        hvr_sparse_vec_set_internal(features[f], last_val,
                ctx->coupled_pes_values + ctx->pe, INT64_MAX);
    }

    shmem_clear_lock(ctx->coupled_locks + ctx->pe);

    update_my_timestep(ctx, INT64_MAX);

    shmem_quiet(); // Make sure the timestep updates complete

    hvr_pe_set_destroy(to_couple_with);

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
    if (ctx->dump_mode) {
        fclose(ctx->dump_file);
    }
    free(ctx);
}

int64_t hvr_current_timestep(hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    return ctx->timestep;
}

unsigned long long hvr_current_time_us() {
    struct timeval curr_time;
    gettimeofday(&curr_time, NULL);
    return curr_time.tv_sec * 1000000ULL + curr_time.tv_usec;
}
