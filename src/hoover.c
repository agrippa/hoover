/* For license: see LICENSE.txt file at top-level */

#define _BSD_SOURCE
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>
#include <math.h>
#include <limits.h>

#include <shmem.h>
#include <shmemx.h>

#include "hoover.h"
#include "hoover_internal.h"
#include "shmem_rw_lock.h"
#include "hvr_vertex_iter.h"

#define CACHED_TIMESTEPS_TOLERANCE 5
#define MAX_INTERACTING_PARTITIONS 100

// #define FINE_GRAIN_TIMING

// #define TRACK_VECTOR_GET_CACHE

static int print_profiling = 1;

#define USE_CSWAP_BITWISE_ATOMICS

#if SHMEM_MAJOR_VERSION == 1 && SHMEM_MINOR_VERSION >= 4 || SHMEM_MAJOR_VERSION >= 2

#define SHMEM_ULONGLONG_ATOMIC_OR shmem_ulonglong_atomic_or
#define SHMEM_UINT_ATOMIC_OR shmem_uint_atomic_or
#define SHMEM_UINT_ATOMIC_AND shmem_uint_atomic_and

#else
/*
 * Pre 1.4 some OpenSHMEM implementations offered a shmemx variant of atomic
 * bitwise functions. For others, we have to implement it on top of cswap.
 */

#ifdef USE_CSWAP_BITWISE_ATOMICS
#warning "Heads up! Using atomic compare-and-swap to implement atomic bitwise atomics!"

static void inline _shmem_ulonglong_atomic_or(unsigned long long *dst,
        unsigned long long val, int pe) {
    unsigned long long curr_val;
    shmem_getmem(&curr_val, dst, sizeof(curr_val), pe);

    while (1) {
        unsigned long long new_val = (curr_val | val);
        unsigned long long old_val = SHMEM_ULL_CSWAP(dst, curr_val, new_val, pe);
        if (old_val == curr_val) return;
        curr_val = old_val;
    }
}

static void inline _shmem_uint_atomic_or(unsigned int *dst, unsigned int val,
        int pe) {
    unsigned int curr_val;
    shmem_getmem(&curr_val, dst, sizeof(curr_val), pe);

    while (1) {
        unsigned int new_val = (curr_val | val);
        unsigned int old_val = SHMEM_UINT_CSWAP(dst, curr_val, new_val, pe);
        if (old_val == curr_val) return;
        curr_val = old_val;
    }
}

static void inline _shmem_uint_atomic_and(unsigned int *dst, unsigned int val,
        int pe) {
    unsigned int curr_val;
    shmem_getmem(&curr_val, dst, sizeof(curr_val), pe);

    while (1) {
        unsigned int new_val = (curr_val & val);
        unsigned int old_val = SHMEM_UINT_CSWAP(dst, curr_val, new_val, pe);
        if (old_val == curr_val) return;
        curr_val = old_val;
    }
}

#define SHMEM_ULONGLONG_ATOMIC_OR _shmem_ulonglong_atomic_or
#define SHMEM_UINT_ATOMIC_OR _shmem_uint_atomic_or
#define SHMEM_UINT_ATOMIC_AND _shmem_uint_atomic_or

#else

#define SHMEM_ULONGLONG_ATOMIC_OR shmemx_ulonglong_atomic_or
#define SHMEM_UINT_ATOMIC_OR shmemx_uint_atomic_or
#define SHMEM_UINT_ATOMIC_AND shmemx_uint_atomic_and

#endif
#endif

#define EDGE_GET_BUFFERING 2048

static int have_default_sparse_vec_val = 0;
static double default_sparse_vec_val = 0.0;

static inline int hvr_sparse_vec_find_bucket(hvr_sparse_vec_t *vec,
        const hvr_time_t curr_timestamp, unsigned *nhits, unsigned *nmisses);

static inline void *shmem_malloc_wrapper(size_t nbytes) {
    static size_t total_nbytes = 0;
    if (nbytes == 0) {
        fprintf(stderr, "PE %d allocated %lu bytes. Exiting...\n",
                shmem_my_pe(), total_nbytes);
        exit(0);
    } else {
        void *ptr = shmem_malloc(nbytes);
        total_nbytes += nbytes;
        return ptr;
    }
}

static void rlock_actor_to_partition(const int pe, hvr_internal_ctx_t *ctx) {
    assert(pe < ctx->npes);
    hvr_rwlock_rlock(ctx->actor_to_partition_lock, pe);
}

static void runlock_actor_to_partition(const int pe, hvr_internal_ctx_t *ctx) {
    assert(pe < ctx->npes);
    hvr_rwlock_runlock(ctx->actor_to_partition_lock, pe);
}

static void wlock_actor_to_partition(const int pe, hvr_internal_ctx_t *ctx) {
    assert(pe < ctx->npes);
    hvr_rwlock_wlock(ctx->actor_to_partition_lock, pe);
}

static void wunlock_actor_to_partition(const int pe, hvr_internal_ctx_t *ctx) {
    assert(pe < ctx->npes);
    hvr_rwlock_wunlock(ctx->actor_to_partition_lock, pe);
}

hvr_sparse_vec_t *hvr_sparse_vec_create_n(const size_t nvecs,
        hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;

    return hvr_alloc_sparse_vecs(nvecs, ctx);
}

void hvr_sparse_vec_init(hvr_sparse_vec_t *vec, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;

    memset(vec, 0x00, sizeof(*vec));
    vec->cached_timestamp = -1;
    vec->created_timestamp = ctx->timestep;

    for (unsigned i = 0; i < HVR_BUCKETS; i++) {
        vec->timestamps[i] = -1;
        vec->finalized[i] = -1;
    }

    hvr_sparse_vec_pool_t *pool = ctx->pool;
    if (vec >= pool->pool && vec < pool->pool + pool->pool_size) {
        /*
         * This is a remotely accessible vector that we're initializing, set its
         * ID to encode the PE it sets in as well as its offset in the pool of
         * that PE.
         *
         * Use the top 32 bits of the ID to store the PE ID, the bottom 32 bits
         * to store the offset of this vector in the pool.
         */
        vec->id = construct_vertex_id(ctx->pe, vec - pool->pool);
    } else {
        // Some locally allocated PE, give it a non-unique ID
        vec->id = HVR_INVALID_VERTEX_ID;
    }
}

static inline unsigned prev_bucket(const unsigned bucket) {
    return (bucket == 0) ? (HVR_BUCKETS - 1) : (bucket - 1);
}

static inline unsigned next_bucket(const unsigned bucket) {
    return (bucket + 1) % HVR_BUCKETS;
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
        const hvr_time_t curr_timestep, unsigned *out_features,
        unsigned *n_out_features) {
    *n_out_features = 0;

    unsigned unused;
    const int bucket = hvr_sparse_vec_find_bucket(vec, curr_timestep, &unused,
            &unused);
    assert(bucket >= 0);

    for (unsigned i = 0; i < vec->bucket_size[bucket]; i++) {
        out_features[*n_out_features] = vec->features[bucket][i];
        *n_out_features += 1;
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
 *
 * Assume that timestep is always >= the largest timestep value in this sparse
 * vec.
 */
static void hvr_sparse_vec_set_internal(const unsigned feature,
        const double val, hvr_sparse_vec_t *vec, const hvr_time_t timestep) {
    assert(timestep <= MAX_TIMESTAMP);
    unsigned initial_bucket = prev_bucket(vec->next_bucket);

    if (vec->timestamps[initial_bucket] == timestep) {
        // Doing an update on my current timestep, not finalized yet
        assert(vec->finalized[initial_bucket] != timestep);
        set_helper(vec, initial_bucket, feature, val);
    } else {
        // Don't have a bucket for my latest timestep yet
        assert(vec->timestamps[initial_bucket] < timestep);

        // Create a new bucket for this timestep
        const unsigned bucket_to_replace = vec->next_bucket;

        vec->timestamps[bucket_to_replace] = timestep;

        __sync_synchronize();

        if (vec->bucket_size[initial_bucket] > 0) {
            /*
             * If we have an existing bucket at initial_bucket, copy its
             * contents over and then update so that we are making updates on
             * top of initial state.
             */
            unsigned initial_bucket_size = vec->bucket_size[initial_bucket];
            memcpy(vec->values[bucket_to_replace], vec->values[initial_bucket],
                    initial_bucket_size * sizeof(double));
            memcpy(vec->features[bucket_to_replace],
                    vec->features[initial_bucket],
                    initial_bucket_size * sizeof(unsigned));
            vec->bucket_size[bucket_to_replace] = initial_bucket_size;
        } else {
            vec->bucket_size[bucket_to_replace] = 0;
        }

        set_helper(vec, bucket_to_replace, feature, val);

        vec->next_bucket = (bucket_to_replace + 1) % HVR_BUCKETS;
    }
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

static inline int hvr_sparse_vec_find_bucket(hvr_sparse_vec_t *vec,
        const hvr_time_t curr_timestamp, unsigned *nhits, unsigned *nmisses) {
    if (vec->cached_timestamp == curr_timestamp - 1 &&
            vec->cached_timestamp_index < HVR_BUCKETS &&
            vec->timestamps[vec->cached_timestamp_index] == curr_timestamp - 1 &&
            vec->finalized[vec->cached_timestamp_index] == curr_timestamp - 1) {
        /*
         * If we might have the location of this timestamp in this vector
         * cached, double check and then use that information to do an O(1)
         * lookup if possible.
         */
#ifdef TRACK_VECTOR_GET_CACHE
        (*nhits)++;
#endif
        return vec->cached_timestamp_index;
    }
#ifdef TRACK_VECTOR_GET_CACHE
    (*nmisses)++;
#endif

    unsigned initial_bucket = prev_bucket(vec->next_bucket);

    unsigned curr_bucket = initial_bucket;
    do {
        // No valid entries
        if (vec->timestamps[curr_bucket] < 0) break;

        if (vec->timestamps[curr_bucket] != vec->finalized[curr_bucket]) {
            // Ignore partial entry
        } else {
            if (vec->timestamps[curr_bucket] < curr_timestamp ||
                    vec->timestamps[curr_bucket] == MAX_TIMESTAMP) {
                // Handle finding an existing bucket for this timestep
                vec->cached_timestamp = vec->timestamps[curr_bucket];
                vec->cached_timestamp_index = curr_bucket;
                return curr_bucket;
            }
        }

        // Move to the next bucket
        curr_bucket = prev_bucket(curr_bucket);
    } while (curr_bucket != initial_bucket);

    return -1;
}

static int hvr_sparse_vec_get_internal(const unsigned feature,
        hvr_sparse_vec_t *vec, const hvr_time_t curr_timestamp,
        double *out_val, unsigned *nhits, unsigned *nmisses) {

    int target_bucket = hvr_sparse_vec_find_bucket(vec, curr_timestamp, nhits,
            nmisses);

    if (target_bucket >= 0 && find_feature_in_bucket(vec, target_bucket,
                feature, out_val)) {
        return 1;
    } else if (have_default_sparse_vec_val) {
        *out_val = default_sparse_vec_val;
        return 1;
    } else {
        return 0;
    }
}

double hvr_sparse_vec_get(const unsigned feature, hvr_sparse_vec_t *vec,
        hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    double result;
    if (hvr_sparse_vec_get_internal(feature, vec, ctx->timestep, &result,
                &ctx->n_vector_cache_hits, &ctx->n_vector_cache_misses)) {
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
        const size_t buf_size, const hvr_time_t timestep, unsigned *nhits,
        unsigned *nmisses) {
    char *iter = buf;
    int first = 1;

    unsigned n_features;
    unsigned features[HVR_BUCKET_SIZE];
    hvr_sparse_vec_unique_features(vec, timestep, features, &n_features);

    for (unsigned i = 0; i < n_features; i++) {
        const unsigned feat = features[i];
        double val;

        const int err = hvr_sparse_vec_get_internal(feat, vec, timestep, &val,
                nhits, nmisses);
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
    hvr_sparse_vec_dump_internal(vec, buf, buf_size, ctx->timestep,
            &ctx->n_vector_cache_hits, &ctx->n_vector_cache_misses);
}

int hvr_sparse_vec_get_owning_pe(hvr_sparse_vec_t *vec) {
    assert(vec->id != HVR_INVALID_VERTEX_ID);
    return VERTEX_ID_PE(vec->id);
}

static hvr_time_t hvr_sparse_vec_max_timestamp(hvr_sparse_vec_t *vec) {
    unsigned unused;
    int highest_bucket = hvr_sparse_vec_find_bucket(vec, MAX_TIMESTAMP + 1,
            &unused, &unused);
    if (highest_bucket < 0) {
        return -1;
    } else {
        return vec->timestamps[highest_bucket];
    }
}

static hvr_time_t hvr_sparse_vec_min_timestamp(hvr_sparse_vec_t *vec) {
    const unsigned initial_bucket = vec->next_bucket;
    unsigned bucket = initial_bucket;
    while (vec->timestamps[bucket] < 0) {
        bucket = next_bucket(bucket);
        if (bucket == initial_bucket) break;
    }

    if (vec->timestamps[bucket] >= 0 && vec->timestamps[bucket] == vec->finalized[bucket]) {
        return vec->timestamps[bucket];
    } else {
        return -1;
    }
}

#define HAS_TIMESTAMP -1
#define NOT_HAVE_TIMESTAMP -2
#define NEVER_HAVE_TIMESTAMP -3

static int hvr_sparse_vec_has_timestamp(hvr_sparse_vec_t *vec,
        const hvr_time_t timestamp) {
    unsigned unused;
    const int target_bucket = hvr_sparse_vec_find_bucket(vec, timestamp + 1,
            &unused, &unused);
    if (target_bucket >= 0 && (vec->timestamps[target_bucket] == timestamp ||
            vec->timestamps[target_bucket] == MAX_TIMESTAMP)) {
        return HAS_TIMESTAMP;
    } else {
        hvr_time_t min_timestamp = hvr_sparse_vec_min_timestamp(vec);
        if (min_timestamp > timestamp) {
            return NEVER_HAVE_TIMESTAMP;
        } else {
            return NOT_HAVE_TIMESTAMP;
        }
    }
}

static void hvr_sparse_vec_add_internal(hvr_sparse_vec_t *dst,
        hvr_sparse_vec_t *src, const uint64_t target_timestamp) {
    unsigned unused;
    const int dst_bucket = hvr_sparse_vec_find_bucket(dst, target_timestamp + 1,
            &unused, &unused);
    assert(dst_bucket >= 0);

    const int src_bucket = hvr_sparse_vec_find_bucket(src, target_timestamp + 1,
            &unused, &unused);
    assert(src_bucket >= 0);

    const unsigned n_dst_features = dst->bucket_size[dst_bucket];
    const unsigned n_src_features = src->bucket_size[src_bucket];
    if (n_dst_features != n_src_features) {
        fprintf(stderr, "ERROR: n_dst_features=%u n_src_features=%u "
                "dst_bucket=%d src_bucket=%d target_timestamp=%llu "
                "dst_timestamp=%llu src_timestamp=%llu dst=[%u %u] "
                "src=[%u %u]\n", n_dst_features, n_src_features,
                dst_bucket, src_bucket, (unsigned long long)target_timestamp,
                (unsigned long long)dst->timestamps[dst_bucket],
                (unsigned long long)src->timestamps[src_bucket],
                dst->features[dst_bucket][0], dst->features[dst_bucket][1],
                src->features[src_bucket][0], src->features[src_bucket][1]);
        abort();
    }

    for (unsigned i = 0; i < n_dst_features; i++) {
        unsigned feature = dst->features[dst_bucket][i];
        unsigned dst_feature_index = i;

        int src_feature_index = -1;
        for (unsigned j = 0; j < n_src_features; j++) {
            if (src->features[src_bucket][j] == feature) {
                src_feature_index = j;
                break;
            }
        }
        if (src_feature_index < 0) {
            fprintf(stderr, "ERROR: n_dst_features=%u n_src_features=%u "
                    "dst_bucket=%d src_bucket=%d target_timestamp=%llu "
                    "dst_timestamp=%llu src_timestamp=%llu dst=[%u %u] "
                    "src=[%u %u] dst=[%f %f] src=[%f %f] dst-finalized=%u "
                    "src-finalized=%u dst-pe=%lu src-pe=%lu\n", n_dst_features,
                    n_src_features, dst_bucket, src_bucket,
                    (unsigned long long)target_timestamp,
                    (unsigned long long)dst->timestamps[dst_bucket],
                    (unsigned long long)src->timestamps[src_bucket],
                    dst->features[dst_bucket][0], dst->features[dst_bucket][1],
                    src->features[src_bucket][0], src->features[src_bucket][1],
                    dst->values[dst_bucket][0], dst->values[dst_bucket][1],
                    src->values[src_bucket][0], src->values[src_bucket][1],
                    dst->finalized[dst_bucket], src->finalized[src_bucket],
                    VERTEX_ID_PE(dst->id), VERTEX_ID_PE(src->id));

            abort();
        }

        dst->values[dst_bucket][dst_feature_index] +=
            src->values[src_bucket][src_feature_index];
    }
}

static int get_newest_timestamp(hvr_sparse_vec_t *vec,
        hvr_time_t *out_timestamp) {
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
    cache->nhits = 0;
    cache->nmisses = 0;
    cache->nmisses_due_to_age = 0;
}

void hvr_sparse_vec_cache_init(hvr_sparse_vec_cache_t *cache) {
    memset(cache, 0x00, sizeof(*cache));
    if (getenv("HVR_CACHE_MAX_BUCKET_SIZE")) {
        cache->hvr_cache_max_bucket_size = atoi(
                getenv("HVR_CACHE_MAX_BUCKET_SIZE"));
    } else {
        cache->hvr_cache_max_bucket_size = 1024;
    }
}

hvr_sparse_vec_cache_node_t *hvr_sparse_vec_cache_lookup(unsigned offset,
        hvr_sparse_vec_cache_t *cache, hvr_time_t target_timestep) {
    const unsigned bucket = offset % HVR_CACHE_BUCKETS;
    hvr_sparse_vec_cache_node_t *head = cache->buckets[bucket];
    hvr_sparse_vec_cache_node_t *iter = head;
    hvr_sparse_vec_cache_node_t *prev = NULL;
    while (iter) {
        if (iter->offset == offset) break;
        prev = iter;
        iter = iter->next;
    }

    if (iter == NULL) {
        cache->nmisses++;
        return NULL;
    } else {
        // Decide whether this cached entry is new enough to be useful
        hvr_time_t newest_timestamp;
        int success = get_newest_timestamp(&(iter->vec), &newest_timestamp);

        if (success && (newest_timestamp >= target_timestep ||
                    target_timestep - newest_timestamp <= CACHED_TIMESTEPS_TOLERANCE ||
                    iter->pending_comm)) {
            // Can use, update the cache to make this most recently used
            if (prev) {
                prev->next = iter->next;
                iter->next = cache->buckets[bucket];
                cache->buckets[bucket] = iter;
            }
            cache->nhits++;
            return iter;
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

            cache->nmisses_due_to_age++;
            return NULL;
        }
    }
}

void hvr_sparse_vec_cache_quiet(hvr_sparse_vec_cache_t *cache,
        unsigned long long *counter) {
    for (unsigned b = 0; b < HVR_CACHE_BUCKETS; b++) {
        hvr_sparse_vec_cache_node_t *iter = cache->buckets[b];
        while (iter && iter->pending_comm) {
            iter->pending_comm = 0;
            iter = iter->next;
        }
    }
    *counter = *counter + 1;
}

hvr_sparse_vec_cache_node_t *hvr_sparse_vec_cache_reserve(
        unsigned offset, hvr_sparse_vec_cache_t *cache) {
    // Assume that vec is not already in the cache, but don't enforce this
    const unsigned bucket = offset % HVR_CACHE_BUCKETS;
    if (cache->bucket_size[bucket] == cache->hvr_cache_max_bucket_size) {
        // Evict and re-use
        hvr_sparse_vec_cache_node_t *iter = cache->buckets[bucket];
        hvr_sparse_vec_cache_node_t *prev = NULL;
        while (iter->next) {
            prev = iter;
            iter = iter->next;
        }
        // Re-use iter and place it at head
        prev->next = NULL;
        assert(iter->pending_comm == 0);
        iter->offset = offset;
        iter->pending_comm = 1;
        iter->next = cache->buckets[bucket];
        cache->buckets[bucket] = iter;
        return iter;
    } else {
        // Allocate and insert
        hvr_sparse_vec_cache_node_t *new_node;
        if (cache->pool) {
            new_node = cache->pool;
            cache->pool = new_node->next;
        } else {
            new_node = (hvr_sparse_vec_cache_node_t *)malloc(sizeof(*new_node));
            assert(new_node);
            memset(new_node, 0x00, sizeof(*new_node));
        }
        assert(new_node->pending_comm == 0);
        new_node->offset = offset;
        new_node->pending_comm = 1;
        new_node->next = cache->buckets[bucket];
        cache->buckets[bucket] = new_node;
        cache->bucket_size[bucket] += 1;
        return new_node;
    }
}

void hvr_sparse_vec_cache_insert(unsigned offset, hvr_sparse_vec_t *vec,
        hvr_sparse_vec_cache_t *cache) {
    hvr_sparse_vec_cache_node_t *node = hvr_sparse_vec_cache_reserve(offset,
            cache);
    node->pending_comm = 0;
    memcpy(&(node->vec), vec, sizeof(*vec));
}

static void sum_hits_and_misses(hvr_sparse_vec_cache_t *vec_caches,
        const unsigned ncaches, unsigned *out_nhits, unsigned *out_nmisses,
        unsigned *out_nmisses_due_to_age) {
    *out_nhits = 0; *out_nmisses = 0; *out_nmisses_due_to_age = 0;

    for (unsigned i = 0; i < ncaches; i++) {
        *out_nhits += vec_caches[i].nhits;
        *out_nmisses += vec_caches[i].nmisses;
        *out_nmisses_due_to_age += vec_caches[i].nmisses_due_to_age;
    }
}

static hvr_set_t *hvr_create_empty_set_helper(hvr_internal_ctx_t *ctx,
        const int nelements, hvr_set_t *set, bit_vec_element_type *bit_vector) {
    set->bit_vector = bit_vector;
    set->nelements = nelements;
    set->n_contained = 0;

    memset(set->bit_vector, 0x00, nelements * sizeof(bit_vec_element_type));

    return set;
}

hvr_set_t *hvr_create_empty_set_symmetric_custom(const unsigned nvals,
        hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    const size_t bits_per_ele = sizeof(bit_vec_element_type) *
                BITS_PER_BYTE;
    const int nelements = (nvals + bits_per_ele - 1) / bits_per_ele;
    hvr_set_t *set = (hvr_set_t *)shmem_malloc(sizeof(*set));
    assert(set);
    bit_vec_element_type *bit_vector = (bit_vec_element_type *)shmem_malloc(
            nelements * sizeof(bit_vec_element_type));
    assert(bit_vector);
    return hvr_create_empty_set_helper(ctx, nelements, set, bit_vector);
}

hvr_set_t *hvr_create_empty_set_symmetric(hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    return hvr_create_empty_set_symmetric_custom(ctx->npes, ctx);
}

hvr_set_t *hvr_create_empty_set(const unsigned nvals,
        hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    const size_t bits_per_ele = sizeof(bit_vec_element_type) *
                BITS_PER_BYTE;
    const int nelements = (nvals + bits_per_ele - 1) / bits_per_ele;
    hvr_set_t *set = (hvr_set_t *)malloc(sizeof(*set));
    assert(set);
    bit_vec_element_type *bit_vector = (bit_vec_element_type *)malloc(
            nelements * sizeof(bit_vec_element_type));
    assert(bit_vector);
    return hvr_create_empty_set_helper(ctx, nelements, set, bit_vector);
}

static int hvr_set_insert_internal(int pe,
        bit_vec_element_type *bit_vector) {
    const int element = pe / (sizeof(*bit_vector) * BITS_PER_BYTE);
    const int bit = pe % (sizeof(*bit_vector) * BITS_PER_BYTE);
    const bit_vec_element_type old_val = bit_vector[element];
    const bit_vec_element_type new_val =
        (old_val | ((bit_vec_element_type)1 << bit));
    bit_vector[element] = new_val;
    return old_val != new_val;
}

int hvr_set_insert(int pe, hvr_set_t *set) {
    const int changed = hvr_set_insert_internal(pe, set->bit_vector);
    if (changed) {
        if (set->n_contained < PE_SET_CACHE_SIZE) {
            (set->cache)[set->n_contained] = pe;
        }
        set->n_contained++;
    }
    return changed;
}

void hvr_set_wipe(hvr_set_t *set) {
    memset(set->bit_vector, 0x00, set->nelements * sizeof(*(set->bit_vector)));
    set->n_contained = 0;
}

int hvr_set_contains_internal(int pe, bit_vec_element_type *bit_vector) {
    const int element = pe / (sizeof(*bit_vector) * BITS_PER_BYTE);
    const int bit = pe % (sizeof(*bit_vector) * BITS_PER_BYTE);
    const bit_vec_element_type old_val = bit_vector[element];
    if (old_val & ((bit_vec_element_type)1 << bit)) {
        return 1;
    } else {
        return 0;
    }
}

int hvr_set_contains(int pe, hvr_set_t *set) {
    return hvr_set_contains_internal(pe, set->bit_vector);
}

unsigned hvr_set_count(hvr_set_t *set) {
    return set->n_contained;
}

void hvr_set_destroy(hvr_set_t *set) {
    free(set->bit_vector);
    free(set);
}

void hvr_set_to_string(hvr_set_t *set, char *buf, unsigned buflen) {
    int offset = snprintf(buf, buflen, "{");

    const size_t nvals = set->nelements * sizeof(bit_vec_element_type) *
        BITS_PER_BYTE;
    for (unsigned i = 0; i < nvals; i++) {
        if (hvr_set_contains(i, set)) {
            offset += snprintf(buf + offset, buflen - offset - 1, " %u", i);
            assert(offset < buflen);
        }
    }

    snprintf(buf + offset, buflen - offset - 1, " }");
}

static void hvr_set_merge_atomic(hvr_set_t *set, hvr_set_t *other) {
    assert(set->nelements == other->nelements);
    // Assert that we can use the long long atomics
    assert(sizeof(unsigned long long) == sizeof(bit_vec_element_type));

    for (int i = 0; i < set->nelements; i++) {
        SHMEM_ULONGLONG_ATOMIC_OR(set->bit_vector + i, (other->bit_vector)[i],
                shmem_my_pe());
    }
}

unsigned *hvr_set_non_zeros(hvr_set_t *set,
        unsigned *n_non_zeros, int *user_must_free) {
    *n_non_zeros = set->n_contained;
    if (set->n_contained <= PE_SET_CACHE_SIZE) {
        *user_must_free = 0;
        return set->cache;
    } else {
        unsigned count_non_zeros = 0;
        unsigned *non_zeros = (unsigned *)malloc(
                set->n_contained * sizeof(*non_zeros));
        assert(non_zeros);
        *user_must_free = 1;

        for (unsigned i = 0; i < set->nelements; i++) {
            const bit_vec_element_type ele = set->bit_vector[i];
            for (unsigned bit = 0; bit < sizeof(ele) * BITS_PER_BYTE; bit++) {
                if ((ele & ((bit_vec_element_type)1 << bit)) != 0) {
                    non_zeros[count_non_zeros++] =
                        i * sizeof(ele) * BITS_PER_BYTE + bit;
                }
            }
        }

        if (count_non_zeros != set->n_contained) {
            fprintf(stderr, "Unexpected nnz: count_non_zeros = %u, "
                    "n_contained = %u\n", count_non_zeros, set->n_contained);
            abort();
        }

        return non_zeros;
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

    size_t symm_pool_nelements = 1024UL * 1024UL; // Default
    if (getenv("HVR_SYMM_POOL_SIZE")) {
        symm_pool_nelements = atoi(getenv("HVR_SYMM_POOL_SIZE"));
    }

    new_ctx->pool = hvr_sparse_vec_pool_create(symm_pool_nelements);

#ifdef VERBOSE
    int err = gethostname(new_ctx->my_hostname, 1024);
    assert(err == 0);

    printf("PE %d is on host %s.\n", new_ctx->pe, new_ctx->my_hostname);
#endif

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
    uint16_t interacting_partitions[MAX_INTERACTING_PARTITIONS];
    unsigned n_interacting_partitions;
    hvr_set_wipe(ctx->my_neighbors);

    for (unsigned part = 0; part < ctx->n_partitions; part++) {
        if (ctx->might_interact(part, ctx->partition_time_window,
                    interacting_partitions, &n_interacting_partitions,
                    MAX_INTERACTING_PARTITIONS, ctx)) {
            /*
             * If this is a partition we might interact with, we go find its PE
             * bit vector on the owning PE, grab it, and update neighbors to be
             * anyone in that bit vector.
             */
            const int partition_owner_pe = part / ctx->partitions_per_pe;
            const int partition_offset = part % ctx->partitions_per_pe;

            shmem_getmem(ctx->local_pes_per_partition_buffer,
                    ctx->pes_per_partition + (partition_offset *
                        ctx->partitions_per_pe_vec_length_in_words),
                    ctx->partitions_per_pe_vec_length_in_words *
                        sizeof(unsigned),
                    partition_owner_pe);

            /*
             * Iterate over the local_pes_per_partition_buffer bit vector, and
             * add as a neighbor any PEs in it.
             */
            for (unsigned p = 0; p < ctx->npes; p++) {
                unsigned word = p / BITS_PER_WORD;
                unsigned bit = p % BITS_PER_WORD;
                unsigned bit_mask = (1U << bit);

                unsigned word_val = (ctx->local_pes_per_partition_buffer)[word];
                if (word_val & bit_mask) {
                    hvr_set_insert(p, ctx->my_neighbors);
                }
            }
        }
    }

#ifdef VERBOSE
    printf("PE %d is talking to %d other PEs\n", ctx->pe,
            hvr_set_count(ctx->my_neighbors));
#endif
}

static double sparse_vec_distance_measure(hvr_sparse_vec_t *a,
        hvr_sparse_vec_t *b, const hvr_time_t a_max_timestep,
        const hvr_time_t b_max_timestep, const unsigned min_spatial_feature,
        const unsigned max_spatial_feature, unsigned *nhits,
        unsigned *nmisses) {
    const int a_bucket = hvr_sparse_vec_find_bucket(a, a_max_timestep + 1,
            nhits, nmisses);
    assert(a_bucket >= 0);
    const int b_bucket = hvr_sparse_vec_find_bucket(b, b_max_timestep + 1,
            nhits, nmisses);
    assert(b_bucket >= 0);

    double acc = 0.0;
    for (unsigned f = min_spatial_feature; f <= max_spatial_feature; f++) {
        double a_val, b_val;
        const int a_err = find_feature_in_bucket(a, a_bucket, f, &a_val);
        assert(a_err == 1);
        const int b_err = find_feature_in_bucket(b, b_bucket, f, &b_val);
        assert(b_err == 1);

        const double delta = b_val - a_val;
        acc += (delta * delta);
    }
    return acc;
}

static void get_remote_vec_nbi_uncached(hvr_sparse_vec_t *dst,
        hvr_sparse_vec_t *src, const int src_pe) {
    shmem_getmem_nbi((void *)&(dst->next_bucket),
            (void *)&(src->next_bucket),
            sizeof(src->next_bucket), src_pe);

    shmem_fence();

    shmem_getmem_nbi(&(dst->finalized[0]), &(src->finalized[0]),
            HVR_BUCKETS * sizeof(src->finalized[0]), src_pe);

    shmem_fence();

    shmem_getmem_nbi(dst, src,
            offsetof(hvr_sparse_vec_t, finalized), src_pe);

    /*
     * No need to set next_in_partition here since we'll never use it for a
     * remote actor.
     */
    dst->cached_timestamp = -1;
    dst->cached_timestamp_index = 0;
}

static hvr_sparse_vec_cache_node_t *get_remote_vec_nbi(const uint32_t offset,
        const int src_pe, hvr_internal_ctx_t *ctx,
        hvr_sparse_vec_cache_t *vec_caches) {
    hvr_sparse_vec_cache_t *cache = vec_caches + src_pe;
    hvr_sparse_vec_cache_node_t *cached = hvr_sparse_vec_cache_lookup(offset,
            cache, ctx->timestep - 1);

    if (cached) {
        // May still be pending
        return cached;
    } else {
        hvr_sparse_vec_t *src = &(ctx->pool->pool[offset]);
        hvr_sparse_vec_cache_node_t *node = hvr_sparse_vec_cache_reserve(offset,
                cache);
        get_remote_vec_nbi_uncached(&(node->vec), src, src_pe);
        return node;
    }
}

static void check_edges_to_add(hvr_sparse_vec_t *remote_vec,
        uint16_t *interacting_partitions, unsigned n_interacting_partitions,
        hvr_time_t other_pes_timestep, hvr_internal_ctx_t *ctx,
        unsigned long long *n_distance_measures) {
    unsigned long long local_n_distance_measures = 0;

    if (remote_vec->created_timestamp >= ctx->timestep) {
        /*
         * Early abort on remote actors that didn't exist until our current
         * timestep (so we'll have no past state to look back at).
         */
        return;
    }

    for (unsigned p = 0; p < n_interacting_partitions; p++) {
        hvr_sparse_vec_t *partition_list =
            ctx->partition_lists[interacting_partitions[p]];
        while (partition_list) {
            /*
             * Never want to add an edge from a node to itself (at
             * least, we don't have a use case for this yet).
             */
            if (partition_list->id != remote_vec->id) {
                const double distance = sparse_vec_distance_measure(
                        partition_list, remote_vec, 
                        ctx->timestep - 1, other_pes_timestep,
                        ctx->min_spatial_feature,
                        ctx->max_spatial_feature,
                        &ctx->n_vector_cache_hits,
                        &ctx->n_vector_cache_misses);
                if (distance < ctx->connectivity_threshold *
                        ctx->connectivity_threshold) {
                    // Add edge
                    hvr_add_edge(partition_list->id, remote_vec->id,
                            ctx->edges);
                }
                local_n_distance_measures++;
            }
            partition_list = partition_list->next_in_partition;
        }
    }
    *n_distance_measures += local_n_distance_measures;
}

/*
 * For each local actor, update the edges it has to other PEs in the simulation
 * based on its current position.
 */
static void update_edges(hvr_internal_ctx_t *ctx,
        hvr_sparse_vec_cache_t *vec_caches,
        unsigned long long *getmem_time, unsigned long long *update_edge_time,
        unsigned long long *out_n_edge_checks,
        unsigned long long *out_partition_checks,
        unsigned long long *quiet_counter,
        unsigned long long *out_n_distance_measures) {
    uint16_t interacting_partitions[MAX_INTERACTING_PARTITIONS];
    unsigned n_interacting_partitions;

    unsigned long long n_distance_measures = 0;
    unsigned long long total_partition_checks = 0;

    unsigned long long n_edge_checks = 0;
    uint16_t *other_actor_to_partition_map = (uint16_t *)malloc(
            ctx->pool->pool_size * sizeof(uint16_t));
    assert(other_actor_to_partition_map);

    // For each PE
    for (unsigned p = 0; p < ctx->npes; p++) {
        const unsigned target_pe = (ctx->pe + p) % ctx->npes;

        /*
         * If this PE is not in my neighbor set (i.e. has no chance of
         * interaction based on partitions of the problem domain), skip it when
         * looking for new edges.
         */
        if (!hvr_set_contains(target_pe, ctx->my_neighbors)) {
            continue;
        }

        // Grab the target PEs mapping from actors to partitions
        rlock_actor_to_partition(target_pe, ctx);
        shmem_getmem(other_actor_to_partition_map, ctx->actor_to_partition_map,
                ctx->pool->pool_size * sizeof(uint16_t), target_pe);
        runlock_actor_to_partition(target_pe, ctx);

        // Check what timestep the remote PE is on currently.
        hvr_time_t other_pes_timestep;
        shmem_getmem(&other_pes_timestep, (hvr_time_t *)ctx->symm_timestep,
                sizeof(other_pes_timestep), target_pe);
        /*
         * Prevent this PE from seeing into the future, if the other PE is ahead
         * of us.
         */
        if (other_pes_timestep > ctx->timestep - 1) {
            other_pes_timestep = ctx->timestep - 1;
        }

#define BUFFERING 512
        struct {
            hvr_sparse_vec_cache_node_t *node;
            uint16_t interacting_partitions[MAX_INTERACTING_PARTITIONS];
            unsigned n_interacting_partitions;
        } buffered[BUFFERING];
        unsigned nbuffered = 0;

        // For each vertex on the remote PE
        for (vertex_id_t j = 0; j < ctx->pool->pool_size; j++) {
#ifdef FINE_GRAIN_TIMING
            const unsigned long long start_time = hvr_current_time_us();
#endif
            const uint16_t actor_partition = other_actor_to_partition_map[j];

            // If remote actor j might interact with anything in our local PE.
            if (actor_partition != HVR_INVALID_PARTITION &&
                    ctx->might_interact(actor_partition,
                        ctx->partition_time_window, interacting_partitions,
                        &n_interacting_partitions, MAX_INTERACTING_PARTITIONS,
                        ctx)) {
                buffered[nbuffered].node = get_remote_vec_nbi(j, target_pe, ctx,
                        vec_caches);
                memcpy(buffered[nbuffered].interacting_partitions,
                        interacting_partitions,
                        MAX_INTERACTING_PARTITIONS * sizeof(uint16_t));
                buffered[nbuffered].n_interacting_partitions =
                    n_interacting_partitions;
                nbuffered++;

                if (nbuffered == BUFFERING) {
                    // Process buffered
                    shmem_quiet();
                    hvr_sparse_vec_cache_quiet(vec_caches + target_pe,
                            quiet_counter);
#ifdef FINE_GRAIN_TIMING
                    *getmem_time += (hvr_current_time_us() - start_time);
                    const unsigned long long start_time = hvr_current_time_us();
#endif
                    for (unsigned i = 0; i < nbuffered; i++) {
                        check_edges_to_add(&(buffered[i].node->vec),
                                buffered[i].interacting_partitions,
                                buffered[i].n_interacting_partitions,
                                other_pes_timestep, ctx,
                                &n_distance_measures);
                        total_partition_checks +=
                            buffered[i].n_interacting_partitions;
                    }
#ifdef FINE_GRAIN_TIMING
                    *update_edge_time += (hvr_current_time_us() - start_time);
#endif
                    nbuffered = 0;
                } else {
#ifdef FINE_GRAIN_TIMING
                    *getmem_time += (hvr_current_time_us() - start_time);
#endif
                }

                n_edge_checks++;
            }
        }

        if (nbuffered > 0) {
            // Process buffered
#ifdef FINE_GRAIN_TIMING
            unsigned long long start_time = hvr_current_time_us();
#endif
            shmem_quiet();
            hvr_sparse_vec_cache_quiet(vec_caches + target_pe, quiet_counter);
#ifdef FINE_GRAIN_TIMING
            *getmem_time += (hvr_current_time_us() - start_time);
#endif

#ifdef FINE_GRAIN_TIMING
            start_time = hvr_current_time_us();
#endif
            for (unsigned i = 0; i < nbuffered; i++) {
                check_edges_to_add(&(buffered[i].node->vec),
                        buffered[i].interacting_partitions,
                        buffered[i].n_interacting_partitions,
                        other_pes_timestep, ctx,
                        &n_distance_measures);
                total_partition_checks +=
                    buffered[i].n_interacting_partitions;
            }
#ifdef FINE_GRAIN_TIMING
            *update_edge_time += (hvr_current_time_us() - start_time);
#endif
        }
    }

    free(other_actor_to_partition_map);
    *out_n_edge_checks = n_edge_checks;
    *out_partition_checks = total_partition_checks;
    *out_n_distance_measures = n_distance_measures;
}

/*
 * Update the mapping from each local actor to the partition it belongs to
 * (actor_to_partition_map) as well as information on the last timestep that
 * used each partition (last_timestep_using_partition). This mapping is stored
 * per-timestep, in a circular buffer.
 */
static void update_actor_partitions(hvr_internal_ctx_t *ctx) {
    hvr_sparse_vec_t **partition_lists = ctx->partition_lists;
    memset(partition_lists, 0x00,
            sizeof(hvr_sparse_vec_t *) * ctx->n_partitions);

    wlock_actor_to_partition(ctx->pe, ctx);

    hvr_vertex_iter_t iter;
    hvr_vertex_iter_init(&iter, ctx);
    hvr_sparse_vec_t *curr = hvr_vertex_iter_next(&iter);
    while (curr) {
        assert(curr->id != HVR_INVALID_VERTEX_ID);

        uint16_t partition;
        partition = ctx->actor_to_partition(curr, ctx);
        assert(partition < ctx->n_partitions);

        // Update a mapping from local actor to the partition it belongs to
        (ctx->actor_to_partition_map)[VERTEX_ID_OFFSET(curr->id)] =
            partition;

        /*
         * This doesn't necessarily need to be in the critical section,
         * but to avoid multiple traversal over all actors (i.e. a
         * second loop over n_local_vertices outside of the critical
         * section) we stick it here.
         */
        (ctx->last_timestep_using_partition)[partition] = ctx->timestep;

        if (partition_lists[partition]) {
            curr->next_in_partition = partition_lists[partition];
            partition_lists[partition] = curr;
        } else {
            curr->next_in_partition = NULL;
            partition_lists[partition] = curr;
        }
        curr = hvr_vertex_iter_next(&iter);
    }

    wunlock_actor_to_partition(ctx->pe, ctx);
}

/*
 * partition_time_window stores a list of the partitions that the local PE has
 * had actors inside during some window of recent timesteps. This updates the
 * partitions in that window set based on the results of
 * update_actor_partitions.
 */
static void update_partition_time_window(hvr_internal_ctx_t *ctx) {
    hvr_set_wipe(ctx->tmp_partition_time_window);

    // Update the set of partitions in a temporary buffer
    for (unsigned p = 0; p < ctx->n_partitions; p++) {
        const hvr_time_t last_use = ctx->last_timestep_using_partition[p];
        if (last_use >= 0) {
            assert(last_use <= ctx->timestep);
            if (ctx->timestep - last_use < HVR_BUCKETS) {
                hvr_set_insert(p, ctx->tmp_partition_time_window);
            }
        }
    }

    for (unsigned p = 0; p < ctx->n_partitions; p++) {
        if (hvr_set_contains(p, ctx->tmp_partition_time_window) !=
                hvr_set_contains(p, ctx->partition_time_window)) {
            // A change in active partitions, must update the partition owner
            const int partition_owner_pe = p / ctx->partitions_per_pe;
            const int partition_offset = p % ctx->partitions_per_pe;

            const unsigned pe_word = ctx->pe / BITS_PER_WORD;
            const unsigned pe_bit = ctx->pe % BITS_PER_WORD;
            unsigned pe_mask = (1U << pe_bit);

            if (hvr_set_contains(p, ctx->tmp_partition_time_window)) {
                // Changed from being inactive to active, set bit
                SHMEM_UINT_ATOMIC_OR(
                        ctx->pes_per_partition + (partition_offset *
                            ctx->partitions_per_pe_vec_length_in_words) + pe_word,
                        pe_mask, partition_owner_pe);

            } else {
                // Changed from being active to inactive, clear bit
                pe_mask = ~pe_mask;
                SHMEM_UINT_ATOMIC_AND(
                        ctx->pes_per_partition + (partition_offset *
                            ctx->partitions_per_pe_vec_length_in_words) + pe_word,
                        pe_mask, partition_owner_pe);
            }
        }
    }

    // Copy the newly computed partition window over
    memcpy(ctx->partition_time_window->bit_vector,
        ctx->tmp_partition_time_window->bit_vector,
        ctx->tmp_partition_time_window->nelements *
        sizeof(bit_vec_element_type));
    memcpy(ctx->partition_time_window, ctx->tmp_partition_time_window,
        offsetof(hvr_set_t, bit_vector));
}

/*
 * This method grabs the current timestep table from a neighboring PE and uses
 * information in it to update our local table.
 *
 * At one point, we also pushed updates to the remote PE as well but found that
 * this putmem operation became a bottleneck, with a pronounced long tail for
 * the occasional iteration that triggered this logic. It was removed.
 */
static void update_all_pe_timesteps_helper(const int target_pe,
        hvr_internal_ctx_t *ctx) {
    assert(target_pe < ctx->npes);
    shmem_set_lock(ctx->all_pe_timesteps_locks + target_pe);
    shmem_getmem(ctx->all_pe_timesteps_buffer, ctx->all_pe_timesteps,
            ctx->npes * sizeof(hvr_time_t), target_pe);
    shmem_clear_lock(ctx->all_pe_timesteps_locks + target_pe);

    for (int i = 0; i < ctx->npes; i++) {
        if (ctx->all_pe_timesteps[i] < ctx->all_pe_timesteps_buffer[i]) {
            /*
             * Update local with newer timestep, with either same or greater
             * timestep.
             */
            ctx->all_pe_timesteps[i] = ctx->all_pe_timesteps_buffer[i];
        }
    }

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
        const hvr_time_t set_timestep) {
    shmem_set_lock(ctx->all_pe_timesteps_locks + ctx->pe);
    shmem_putmem(ctx->all_pe_timesteps + ctx->pe, &set_timestep,
            sizeof(set_timestep), ctx->pe);
    shmem_clear_lock(ctx->all_pe_timesteps_locks + ctx->pe);
}

static void oldest_pe_timestep(hvr_internal_ctx_t *ctx,
        hvr_time_t *out_oldest_timestep, unsigned *out_oldest_pe) {
    shmem_set_lock(ctx->all_pe_timesteps_locks + ctx->pe);

    hvr_time_t oldest_timestep = ctx->all_pe_timesteps[0];
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

void hvr_init(const uint16_t n_partitions,
        hvr_update_metadata_func update_metadata,
        hvr_might_interact_func might_interact,
        hvr_check_abort_func check_abort,
        hvr_actor_to_partition actor_to_partition,
        hvr_start_time_step start_time_step,
        const double connectivity_threshold, const unsigned min_spatial_feature,
        const unsigned max_spatial_feature, const hvr_time_t max_timestep,
        hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *new_ctx = (hvr_internal_ctx_t *)in_ctx;

    assert(new_ctx->initialized == 0);
    new_ctx->initialized = 1;

    new_ctx->p_wrk = (long long *)shmem_malloc_wrapper(
            SHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(long long));
    new_ctx->p_wrk_int = (int *)shmem_malloc_wrapper(
            SHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(int));
    new_ctx->p_sync = (long *)shmem_malloc_wrapper(
            SHMEM_REDUCE_SYNC_SIZE * sizeof(long));
    assert(new_ctx->p_wrk && new_ctx->p_sync && new_ctx->p_wrk_int);

    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        (new_ctx->p_sync)[i] = SHMEM_SYNC_VALUE;
    }

    new_ctx->neighbor_buffer = (hvr_sparse_vec_cache_node_t **)malloc(
            EDGE_GET_BUFFERING * sizeof(hvr_sparse_vec_cache_node_t *));
    assert(new_ctx->neighbor_buffer);
    new_ctx->buffered_neighbors = (hvr_sparse_vec_t *)malloc(
            EDGE_GET_BUFFERING * sizeof(hvr_sparse_vec_t));
    assert(new_ctx->buffered_neighbors);
    new_ctx->buffered_neighbors_pes = (int *)malloc(
            EDGE_GET_BUFFERING * sizeof(int));
    assert(new_ctx->buffered_neighbors_pes);

    new_ctx->symm_timestep = (volatile hvr_time_t *)shmem_malloc_wrapper(
            sizeof(*(new_ctx->symm_timestep)));
    assert(new_ctx->symm_timestep);
    *(new_ctx->symm_timestep) = -1;

    new_ctx->all_pe_timesteps = (hvr_time_t *)shmem_malloc_wrapper(
            new_ctx->npes * sizeof(hvr_time_t));
    assert(new_ctx->all_pe_timesteps);
    memset(new_ctx->all_pe_timesteps, 0x00,
        new_ctx->npes * sizeof(hvr_time_t));

    new_ctx->all_pe_timesteps_buffer = (hvr_time_t *)shmem_malloc_wrapper(
            new_ctx->npes * sizeof(hvr_time_t));
    assert(new_ctx->all_pe_timesteps_buffer);
    memset(new_ctx->all_pe_timesteps_buffer, 0x00,
        new_ctx->npes * sizeof(hvr_time_t));

    new_ctx->all_pe_timesteps_locks = (long *)shmem_malloc_wrapper(
            new_ctx->npes * sizeof(long));
    assert(new_ctx->all_pe_timesteps_locks);
    memset(new_ctx->all_pe_timesteps_locks, 0x00,
            new_ctx->npes * sizeof(long));

    new_ctx->last_timestep_using_partition = (hvr_time_t *)malloc(
            n_partitions * sizeof(hvr_time_t));
    assert(new_ctx->last_timestep_using_partition);
    for (unsigned i = 0; i < n_partitions; i++) {
        (new_ctx->last_timestep_using_partition)[i] = -1;
    }

    new_ctx->partition_time_window = hvr_create_empty_set_symmetric_custom(
            n_partitions, new_ctx);
    new_ctx->tmp_partition_time_window = hvr_create_empty_set(
            n_partitions, new_ctx);

    new_ctx->actor_to_partition_lock = hvr_rwlock_create_n(1);

    new_ctx->partition_time_window_lock = hvr_rwlock_create_n(1);

    assert(n_partitions <= HVR_INVALID_PARTITION);
    new_ctx->n_partitions = n_partitions;

    new_ctx->actor_to_partition_map = (uint16_t *)shmem_malloc_wrapper(
            new_ctx->pool->pool_size * sizeof(uint16_t));
    assert(new_ctx->actor_to_partition_map);
    for (unsigned i = 0; i < new_ctx->pool->pool_size; i++) {
        (new_ctx->actor_to_partition_map)[i] = HVR_INVALID_PARTITION;
    }

    new_ctx->update_metadata = update_metadata;
    new_ctx->might_interact = might_interact;
    new_ctx->check_abort = check_abort;
    new_ctx->actor_to_partition = actor_to_partition;
    new_ctx->start_time_step = start_time_step;

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
        new_ctx->strict_counter_src = (int *)shmem_malloc_wrapper(sizeof(int));
        new_ctx->strict_counter_dest = (int *)shmem_malloc_wrapper(sizeof(int));
        assert(new_ctx->strict_counter_src && new_ctx->strict_counter_dest);
    }

    if (getenv("HVR_TRACE_DUMP")) {
        char dump_file_name[256];
        sprintf(dump_file_name, "%d.csv", new_ctx->pe);

        new_ctx->dump_mode = 1;
        new_ctx->dump_file = fopen(dump_file_name, "w");
        assert(new_ctx->dump_file);
    }

    if (getenv("HVR_DISABLE_PROFILING_PRINTS")) {
        print_profiling = 0;
    }

    new_ctx->my_neighbors = hvr_create_empty_set(new_ctx->npes, new_ctx);

    new_ctx->coupled_pes = hvr_create_empty_set_symmetric(new_ctx);
    hvr_set_insert(new_ctx->pe, new_ctx->coupled_pes);

    new_ctx->coupled_pes_values = (hvr_sparse_vec_t *)shmem_malloc_wrapper(
            new_ctx->npes * sizeof(hvr_sparse_vec_t));
    assert(new_ctx->coupled_pes_values);
    for (unsigned i = 0; i < new_ctx->npes; i++) {
        hvr_sparse_vec_init(&(new_ctx->coupled_pes_values)[i], new_ctx);
    }

    new_ctx->coupled_pes_values_buffer = (hvr_sparse_vec_t *)malloc(
            new_ctx->npes * sizeof(hvr_sparse_vec_t));
    assert(new_ctx->coupled_pes_values_buffer);

    new_ctx->coupled_lock = hvr_rwlock_create_n(1);
   
    new_ctx->partitions_per_pe = (new_ctx->n_partitions + new_ctx->npes - 1) /
        new_ctx->npes;
    new_ctx->partitions_per_pe_vec_length_in_words =
        (new_ctx->npes + BITS_PER_WORD - 1) / BITS_PER_WORD;
    new_ctx->pes_per_partition = (unsigned *)shmem_malloc_wrapper(
            new_ctx->partitions_per_pe *
            new_ctx->partitions_per_pe_vec_length_in_words * sizeof(unsigned));
    assert(new_ctx->pes_per_partition);
    memset(new_ctx->pes_per_partition, 0x00, new_ctx->partitions_per_pe *
            new_ctx->partitions_per_pe_vec_length_in_words * sizeof(unsigned));

    new_ctx->local_pes_per_partition_buffer = (unsigned *)malloc(
            new_ctx->partitions_per_pe_vec_length_in_words * sizeof(unsigned));
    assert(new_ctx->local_pes_per_partition_buffer);

    new_ctx->partition_lists = (hvr_sparse_vec_t **)malloc(
            sizeof(hvr_sparse_vec_t *) * new_ctx->n_partitions);
    assert(new_ctx->partition_lists);

    // Print the number of bytes allocated
    // shmem_malloc_wrapper(0);

    shmem_barrier_all();
}

/*
 * Given the local ID of an actor, fetch the latest information on each of its
 * neighbors from their owning PEs, and then call the user-defined
 * update_metadata function to update information on this actor.
 */
static unsigned update_local_actor_metadata(hvr_sparse_vec_t *vertex,
        hvr_set_t *to_couple_with, unsigned long long *fetch_neighbors_time,
        unsigned long long *quiet_neighbors_time,
       unsigned long long *update_metadata_time, hvr_internal_ctx_t *ctx,
       hvr_sparse_vec_cache_t *vec_caches, unsigned long long *quiet_counter) {
    // Buffer used to linearize neighbors list into
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
            ctx->edges->tree, vertex->id);

    // Update the metadata for actor i
    if (vertex_edge_tree != NULL) {
        // This vertex has edges
        const size_t n_neighbors = hvr_tree_linearize(&neighbors,
                &neighbors_capacity, vertex_edge_tree->subtree);

        // Simplifying assumption for now
        if (n_neighbors > EDGE_GET_BUFFERING) {
            fprintf(stderr, "Invalid # neighbors - %lu > %u\n", n_neighbors,
                    EDGE_GET_BUFFERING);
            abort();
        }

        // Fetch all neighbors of this vertex
        const unsigned long long start_single_update = hvr_current_time_us();
        for (unsigned n = 0; n < n_neighbors; n++) {
            const vertex_id_t neighbor = neighbors[n];

            uint64_t other_pe = VERTEX_ID_PE(neighbor);
            uint64_t local_offset = VERTEX_ID_OFFSET(neighbor);

            hvr_sparse_vec_cache_node_t *cache_node = get_remote_vec_nbi(
                    local_offset, other_pe, ctx, vec_caches);
            (ctx->neighbor_buffer)[n] = cache_node;
            (ctx->buffered_neighbors_pes)[n] = other_pe;
        }

        const unsigned long long finish_neighbor_fetch = hvr_current_time_us();

        // Quiet any caches that were hit by the above fetches
        shmem_quiet();
        for (unsigned n = 0; n < n_neighbors; n++) {
            hvr_sparse_vec_cache_quiet(
                    vec_caches + (ctx->buffered_neighbors_pes)[n],
                    quiet_counter);
        }

        /*
         * Copy from the cache nodes into a contiguous buffer before passing to
         * the user
         */
        for (unsigned n = 0; n < n_neighbors; n++) {
            memcpy(ctx->buffered_neighbors + n,
                    &((ctx->neighbor_buffer)[n]->vec),
                    sizeof(hvr_sparse_vec_t));
        }

        const unsigned long long finish_neighbor_quiet = hvr_current_time_us();
        *fetch_neighbors_time += (finish_neighbor_fetch - start_single_update);
        *quiet_neighbors_time += (finish_neighbor_quiet - finish_neighbor_fetch);

        ctx->update_metadata(vertex, ctx->buffered_neighbors,
                n_neighbors, to_couple_with, ctx);
        update_metadata_time += (hvr_current_time_us() - finish_neighbor_fetch);
        return n_neighbors;
    } else {
        const unsigned long long start_single_update = hvr_current_time_us();
        // This vertex has no edges
        ctx->update_metadata(vertex, NULL, 0, to_couple_with,
                ctx);
        update_metadata_time += (hvr_current_time_us() - start_single_update);
        return 0;
    }
}

static inline void finalize_actor_for_timestep(hvr_sparse_vec_t *actor,
        hvr_internal_ctx_t *ctx, const hvr_time_t timestep) {
    unsigned latest_bucket = prev_bucket(actor->next_bucket);
    if (actor->timestamps[latest_bucket] == timestep) {
        assert(actor->finalized[latest_bucket] < timestep);
        actor->finalized[latest_bucket] = timestep;
    }
}

static void finalize_actors_for_timestep(hvr_internal_ctx_t *ctx,
        const hvr_time_t timestep) {
    hvr_vertex_iter_t iter;
    hvr_vertex_iter_init(&iter, ctx);
    hvr_sparse_vec_t *curr = hvr_vertex_iter_next(&iter);
    while (curr) {
        finalize_actor_for_timestep(curr, ctx, timestep);
        curr = hvr_vertex_iter_next(&iter);
    }
}

static void *aborting_thread(void *user_data) {
    const unsigned long long start = hvr_current_time_us();
    while (hvr_current_time_us() - start < 120 * 1000000) {
        sleep(10);
    }
    abort(); // Get a core dump
}

void hvr_body(hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;

    hvr_sparse_vec_cache_t *vec_caches =
        (hvr_sparse_vec_cache_t *)malloc(ctx->npes * sizeof(*vec_caches));
    assert(vec_caches);
    for (int i = 0; i < ctx->npes; i++) {
        hvr_sparse_vec_cache_init(vec_caches + i);
    }

    shmem_barrier_all();

    ctx->other_pe_partition_time_window = hvr_create_empty_set(
            ctx->n_partitions, ctx);

    *(ctx->symm_timestep) = 0;
    ctx->timestep = 1;

    finalize_actors_for_timestep(ctx, 0);

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
    update_edges(ctx, vec_caches, &unused, &unused, &unused, &unused, &unused,
            &unused);

    hvr_set_t *to_couple_with = hvr_create_empty_set(ctx->npes, ctx);

    if (getenv("HVR_HANG_ABORT")) {
        pthread_t aborting_pthread;
        const int pthread_err = pthread_create(&aborting_pthread, NULL,
                aborting_thread, NULL);
        assert(pthread_err == 0);
    }

    if (ctx->timestep >= ctx->max_timestep) {
        fprintf(stderr, "Invalid number of timesteps entered, must be > %u\n",
                ctx->timestep);
        exit(1);
    }

    int should_abort = 0;
    while (!should_abort && ctx->timestep < ctx->max_timestep) {
        const unsigned long long start_iter = hvr_current_time_us();

        unsigned long long quiet_counter = 0;
        unsigned long long fetch_neighbors_time = 0;
        unsigned long long quiet_neighbors_time = 0;
        unsigned long long update_metadata_time = 0;

        if (ctx->start_time_step) {
            hvr_vertex_iter_t iter;
            hvr_vertex_iter_init(&iter, ctx);
            ctx->start_time_step(&iter, ctx);
        }

        hvr_set_wipe(to_couple_with);

        unsigned long long sum_n_neighbors = 0;

        // Update each actor's metadata
        size_t count_vertices = 0;
        hvr_vertex_iter_t iter;
        hvr_vertex_iter_init(&iter, ctx);
        hvr_sparse_vec_t *curr = hvr_vertex_iter_next(&iter);
        while (curr) {
            sum_n_neighbors += update_local_actor_metadata(curr,
                    to_couple_with,
                    &fetch_neighbors_time, &quiet_neighbors_time,
                    &update_metadata_time, ctx,
                    vec_caches, &quiet_counter);
            count_vertices++;
            curr = hvr_vertex_iter_next(&iter);
        }
        const double avg_n_neighbors = (double)sum_n_neighbors /
            (double)count_vertices;

        const unsigned long long finished_updates = hvr_current_time_us();

        *(ctx->symm_timestep) = ctx->timestep;
        ctx->timestep += 1;

        __sync_synchronize();

        // Finalize the updates we just made
        finalize_actors_for_timestep(ctx, ctx->timestep - 1);

        __sync_synchronize();

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
        unsigned long long n_edge_checks = 0;
        unsigned long long partition_checks = 0;
        unsigned long long n_distance_measures = 0;
        update_edges(ctx, vec_caches, &getmem_time, &update_edge_time,
                &n_edge_checks, &partition_checks, &quiet_counter,
                &n_distance_measures);

        const unsigned long long finished_edge_adds = hvr_current_time_us();

        hvr_sparse_vec_t coupled_metric;
        memcpy(&coupled_metric, ctx->coupled_pes_values + ctx->pe,
                sizeof(coupled_metric));

        hvr_vertex_iter_init(&iter, ctx);
        should_abort = ctx->check_abort(&iter, ctx, &coupled_metric);
        finalize_actor_for_timestep(
                &coupled_metric, ctx, ctx->timestep);

        // Update my local information on PEs I am coupled with.
        hvr_set_merge_atomic(ctx->coupled_pes, to_couple_with);

        // Atomically update other PEs that I am coupled with.
        for (int p = 0; p < ctx->npes; p++) {
            if (p != ctx->pe && hvr_set_contains(p, ctx->coupled_pes)) {
                for (int i = 0; i < ctx->coupled_pes->nelements; i++) {
                    SHMEM_ULONGLONG_ATOMIC_OR(
                            ctx->coupled_pes->bit_vector + i,
                            (ctx->coupled_pes->bit_vector)[i], p);
                }
            }
        }

        const unsigned long long finished_neighbor_updates =
            hvr_current_time_us();

        hvr_rwlock_wlock((long *)ctx->coupled_lock, ctx->pe);
        shmem_putmem(ctx->coupled_pes_values + ctx->pe, &coupled_metric,
                sizeof(coupled_metric), ctx->pe);
        shmem_quiet();
        hvr_rwlock_wunlock((long *)ctx->coupled_lock, ctx->pe);

        const unsigned long long finished_coupled_values = hvr_current_time_us();

        /*
         * For each PE I know I'm coupled with, lock their coupled_timesteps
         * list and update my copy with any newer entries in my
         * coupled_timesteps list.
         */
        unsigned n_coupled_spins = 0;
        int ncoupled = 1; // include myself
        for (int p = 0; p < ctx->npes; p++) {
            if (p == ctx->pe) continue;

            if (hvr_set_contains(p, ctx->coupled_pes)) {
                /*
                 * Wait until we've found an update to p's coupled value that is
                 * for this timestep.
                 */
                assert(hvr_sparse_vec_has_timestamp(&coupled_metric,
                            ctx->timestep) == HAS_TIMESTAMP);
                int other_has_timestamp = hvr_sparse_vec_has_timestamp(
                        ctx->coupled_pes_values + p, ctx->timestep);
                assert(other_has_timestamp != NEVER_HAVE_TIMESTAMP);

                while (other_has_timestamp != HAS_TIMESTAMP) {
                    hvr_rwlock_rlock((long *)ctx->coupled_lock, p);
                    
                    shmem_getmem(ctx->coupled_pes_values_buffer,
                            ctx->coupled_pes_values,
                            ctx->npes * sizeof(hvr_sparse_vec_t), p);

                    hvr_rwlock_runlock((long *)ctx->coupled_lock, p);

                    for (int i = 0; i < ctx->npes; i++) {
                        (ctx->coupled_pes_values_buffer)[i].cached_timestamp = -1;
                        (ctx->coupled_pes_values_buffer)[i].cached_timestamp_index = 0;
                    }

                    hvr_rwlock_wlock((long *)ctx->coupled_lock, ctx->pe);
                    for (int i = 0; i < ctx->npes; i++) {
                        hvr_sparse_vec_t *other =
                            ctx->coupled_pes_values_buffer + i;
                        hvr_sparse_vec_t *mine = ctx->coupled_pes_values + i;

                        const hvr_time_t other_max_timestamp =
                            hvr_sparse_vec_max_timestamp(other);
                        const hvr_time_t mine_max_timestamp =
                            hvr_sparse_vec_max_timestamp(mine);

                        if (other_max_timestamp >= 0) {
                            if (mine_max_timestamp < 0) {
                                memcpy(mine, other, sizeof(*mine));
                            } else if (other_max_timestamp > mine_max_timestamp) {
                                memcpy(mine, other, sizeof(*mine));
                            }
                        }
                    }
                    hvr_rwlock_wunlock((long *)ctx->coupled_lock, ctx->pe);

                    other_has_timestamp = hvr_sparse_vec_has_timestamp(
                            ctx->coupled_pes_values + p, ctx->timestep);
                    assert(other_has_timestamp != NEVER_HAVE_TIMESTAMP);
                    n_coupled_spins++;
                }

                hvr_sparse_vec_add_internal(&coupled_metric,
                        ctx->coupled_pes_values + p, ctx->timestep);

                ncoupled++;
            }
        }

        /*
         * TODO coupled_metric here contains the aggregate values over all
         * coupled PEs, including this one. Do we want to do anything with this,
         * other than print it?
         */
        if (ncoupled > 1) {
            char buf[1024];
            unsigned unused;
            hvr_sparse_vec_dump_internal(&coupled_metric, buf, 1024,
                    ctx->timestep + 1, &unused, &unused);
            printf("PE %d - computed coupled value {%s} from %d "
                    "coupled PEs on timestep %d\n", ctx->pe, buf, ncoupled,
                    ctx->timestep);
        }

        const unsigned long long finished_coupling = hvr_current_time_us();

        /*
         * Throttle the progress of much faster PEs to ensure we don't get out
         * of range of timesteps on other PEs.
         */
        update_my_timestep(ctx, ctx->timestep);
        update_all_pe_timesteps(ctx, 1);

        hvr_time_t oldest_timestep;
        unsigned oldest_pe;
        oldest_pe_timestep(ctx, &oldest_timestep, &oldest_pe);

        unsigned nspins = 0;
        while (ctx->timestep - oldest_timestep > HVR_BUCKETS / 2) {
            update_all_pe_timesteps(ctx, nspins + 1);
            oldest_pe_timestep(ctx, &oldest_timestep, &oldest_pe);
            nspins++;
        }

        const unsigned long long finished_throttling = hvr_current_time_us();

        if (ctx->dump_mode && ctx->pool->used_list) {
            // Assume that all vertices have the same features.
            unsigned nfeatures;
            unsigned features[HVR_BUCKET_SIZE];
            hvr_sparse_vec_unique_features(
                    ctx->pool->pool + ctx->pool->used_list->start_index,
                    ctx->timestep, features, &nfeatures);

            hvr_vertex_iter_t iter;
            hvr_vertex_iter_init(&iter, ctx);
            hvr_sparse_vec_t *curr = hvr_vertex_iter_next(&iter);
            while (curr) {
                fprintf(ctx->dump_file, "%lu,%u,%ld,%d", curr->id,
                        nfeatures, (int64_t)ctx->timestep, ctx->pe);
                for (unsigned f = 0; f < nfeatures; f++) {
                    fprintf(ctx->dump_file, ",%u,%f", features[f],
                            hvr_sparse_vec_get(features[f], curr, ctx));
                }
                fprintf(ctx->dump_file, ",,\n");

                curr = hvr_vertex_iter_next(&iter);
            }
        }


#ifdef VERBOSE
        char neighbors_str[1024];
        hvr_set_to_string(ctx->my_neighbors, neighbors_str, 1024);

        char partition_time_window_str[1024];
        hvr_set_to_string(ctx->partition_time_window,
                partition_time_window_str, 1024);
#endif

        unsigned nhits, nmisses, nmisses_due_to_age;
        sum_hits_and_misses(vec_caches, ctx->npes, &nhits, &nmisses,
                &nmisses_due_to_age);

        if (print_profiling) {
            printf("PE %d - timestep %d - total %f ms - metadata %f ms (%f %f "
                    "%f) - summary %f ms (%f %f %f) - edges %f ms (%f %f %llu "
                    "%llu %llu) - neighbor updates %f ms - coupled values %f "
                    "ms - coupling %f ms (%u) - throttling %f ms - %u spins - "
                    "%u / %u PE neighbors %s - partition window = %s, %d / %d "
                    "active - aborting? %d - last step? %d - remote cache "
                    "hits=%u misses=%u age misses=%u, feature cache hits=%u "
                    "misses=%u quiets=%llu, avg # edges=%f\n", ctx->pe,
                    ctx->timestep,
                    (double)(finished_throttling - start_iter) / 1000.0,
                    (double)(finished_updates - start_iter) / 1000.0,
                    (double)fetch_neighbors_time / 1000.0,
                    (double)quiet_neighbors_time / 1000.0,
                    (double)update_metadata_time / 1000.0,
                    (double)(finished_summary_update - finished_updates) / 1000.0,
                    (double)(finished_actor_partitions - finished_updates) / 1000.0,
                    (double)(finished_time_window - finished_actor_partitions) / 1000.0,
                    (double)(finished_summary_update - finished_time_window) / 1000.0,
                    (double)(finished_edge_adds - finished_summary_update) / 1000.0,
                    (double)update_edge_time / 1000.0, (double)getmem_time / 1000.0,
                    n_edge_checks, partition_checks, n_distance_measures,
                    (double)(finished_neighbor_updates - finished_edge_adds) / 1000.0,
                    (double)(finished_coupled_values - finished_neighbor_updates) / 1000.0,
                    (double)(finished_coupling - finished_coupled_values) / 1000.0, n_coupled_spins,
                    (double)(finished_throttling - finished_coupling) / 1000.0,
                    nspins, hvr_set_count(ctx->my_neighbors), ctx->npes,
#ifdef VERBOSE
                    neighbors_str, partition_time_window_str,
#else
                    "", "",
#endif
                    hvr_set_count(ctx->partition_time_window),
                    ctx->n_partitions, should_abort,
                    ctx->timestep >= ctx->max_timestep, nhits, nmisses,
                    nmisses_due_to_age, ctx->n_vector_cache_hits,
                    ctx->n_vector_cache_misses, quiet_counter, avg_n_neighbors);
        }

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
    hvr_rwlock_wlock((long *)ctx->coupled_lock, ctx->pe);
    ctx->timestep += 1;
    unsigned n_features;
    unsigned features[HVR_BUCKET_SIZE];
    hvr_sparse_vec_unique_features(ctx->coupled_pes_values + ctx->pe,
            ctx->timestep + 1, features, &n_features);
    for (unsigned f = 0; f < n_features; f++) {
        double last_val;
        int success = hvr_sparse_vec_get_internal(features[f],
                ctx->coupled_pes_values + ctx->pe, ctx->timestep + 1,
                &last_val, &ctx->n_vector_cache_hits,
                &ctx->n_vector_cache_misses);
        assert(success == 1);

        hvr_sparse_vec_set_internal(features[f], last_val,
                ctx->coupled_pes_values + ctx->pe, MAX_TIMESTAMP);
    }
    finalize_actor_for_timestep(ctx->coupled_pes_values + ctx->pe, ctx,
            MAX_TIMESTAMP);

    hvr_rwlock_wunlock((long *)ctx->coupled_lock, ctx->pe);

    update_my_timestep(ctx, MAX_TIMESTAMP);

    shmem_quiet(); // Make sure the timestep updates complete

    hvr_set_destroy(to_couple_with);

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

hvr_time_t hvr_current_timestep(hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    return ctx->timestep;
}

int hvr_my_pe(hvr_ctx_t ctx) {
    return ctx->pe;
}

unsigned long long hvr_current_time_us() {
    struct timeval curr_time;
    gettimeofday(&curr_time, NULL);
    return curr_time.tv_sec * 1000000ULL + curr_time.tv_usec;
}
