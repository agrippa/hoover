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

// #define DETAILED_PRINTS

#define CACHED_TIMESTEPS_TOLERANCE 2
#define MAX_INTERACTING_PARTITIONS 100
#define N_PARTITION_NODES_PREALLOC 3000
#define CACHE_BUCKET(vert_id) ((vert_id) % HVR_CACHE_BUCKETS)

// #define FINE_GRAIN_TIMING

// #define TRACK_VECTOR_GET_CACHE

static int print_profiling = 1;
static FILE *profiling_fp = NULL;
static volatile int this_pe_has_exited = 0;

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

#define EDGE_GET_BUFFERING 4096

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

hvr_sparse_vec_t *hvr_sparse_vec_create_n_with_const_attrs(const size_t nvecs,
        hvr_graph_id_t graph, unsigned *const_features, double *const_vals, 
        unsigned n_consts, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    return hvr_alloc_sparse_vecs(nvecs, graph, const_features, const_vals,
            n_consts, ctx);
}

hvr_sparse_vec_t *hvr_sparse_vec_create_n(const size_t nvecs,
        hvr_graph_id_t graph, hvr_ctx_t in_ctx) {
    return hvr_sparse_vec_create_n_with_const_attrs(nvecs, graph, NULL, NULL, 0,
            in_ctx);
}

void hvr_sparse_vec_delete_n(hvr_sparse_vec_t *vecs,
        const size_t nvecs, hvr_ctx_t in_ctx) {
    // Mark for deletion, but don't actually free them yet
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    for (unsigned i = 0; i < nvecs; i++) {
        vecs[i].deleted_timestamp = ctx->timestep;
        finalize_actor_for_timestep(vecs + i, ctx->timestep);
    }
}

void hvr_sparse_vec_init_with_const_attrs(hvr_sparse_vec_t *vec,
        hvr_graph_id_t graph, unsigned *const_attr_features,
        double *const_attr_values, unsigned n_const_attrs, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;

    memset(vec, 0x00, sizeof(*vec));
    vec->cached_timestamp = -1;
    vec->created_timestamp = ctx->timestep;
    vec->deleted_timestamp = MAX_TIMESTAMP;
    vec->graph = graph;

    for (unsigned i = 0; i < HVR_BUCKETS; i++) {
        vec->timestamps[i] = -1;
        vec->finalized[i] = -1;
    }

    // Initialize constant attributes on this vertex
    assert(n_const_attrs <= HVR_MAX_CONSTANT_ATTRS);
    vec->n_const_features = n_const_attrs;
    memcpy(vec->const_features, const_attr_features,
            n_const_attrs * sizeof(*const_attr_features));
    memcpy(vec->const_values, const_attr_values,
            n_const_attrs * sizeof(*const_attr_values));

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

void hvr_sparse_vec_init(hvr_sparse_vec_t *vec, hvr_graph_id_t graph,
        hvr_ctx_t in_ctx) {
    hvr_sparse_vec_init_with_const_attrs(vec, graph, NULL, NULL, 0, in_ctx);
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
    for (unsigned i = 0; i < vec->n_const_features; i++) {
        out_features[*n_out_features] = vec->const_features[i];
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
    if (vec->deleted_timestamp != MAX_TIMESTAMP) {
        fprintf(stderr, "ERROR Setting value on deleted vertex? "
                "ctx->timestep=%d vec->deleted_timestamp=%d\n", timestep,
                vec->deleted_timestamp);
        abort();
    }
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

static int find_feature_in_const_attrs(const hvr_sparse_vec_t *vec,
        const unsigned feature, double *out_val) {
    for (unsigned i = 0; i < vec->n_const_features; i++) {
        if ((vec->const_features)[i] == feature) {
            *out_val = (vec->const_values)[i];
            return 1;
        }
    }
    return 0;
}

/*
 * Find the finalized bucket in vec that is closest to but less than
 * curr_timestamp. This may also return buckets for MAX_TIMESTAMP, which were
 * created as a PE exited the simulation.
 */
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
    // First check to see if this is a constant feature.
    for (unsigned i = 0; i < vec->n_const_features; i++) {
        if ((vec->const_features)[i] == feature) {
            *out_val = (vec->const_values)[i];
            return 1;
        }
    }

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

/*
 * Note that this only adds together mutable attributes. Constant attributes
 * on the dst are untouched.
 */
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
    assert(n_dst_features == n_src_features);

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
        assert(src_feature_index >= 0);

        dst->values[dst_bucket][dst_feature_index] +=
            src->values[src_bucket][src_feature_index];
    }
}

static int get_newest_timestamp(hvr_sparse_vec_t *vec,
        hvr_time_t *out_timestamp) {
    unsigned unused;
    int bucket = hvr_sparse_vec_find_bucket(vec, MAX_TIMESTAMP, &unused,
            &unused);

    if (bucket == -1) {
        return 0;
    } else {
        *out_timestamp = vec->timestamps[bucket];
        return 1;
    }
}

void hvr_sparse_vec_cache_init(hvr_sparse_vec_cache_t *cache) {
    memset(cache, 0x00, sizeof(*cache));

    unsigned n_preallocs = 1024;
    if (getenv("HVR_VEC_CACHE_PREALLOCS")) {
        n_preallocs = atoi(getenv("HVR_VEC_CACHE_PREALLOCS"));
    }

    hvr_sparse_vec_cache_node_t *prealloc =
        (hvr_sparse_vec_cache_node_t *)malloc(n_preallocs * sizeof(*prealloc));
    assert(prealloc);
    memset(prealloc, 0x00, n_preallocs * sizeof(*prealloc));

    prealloc[0].next = prealloc + 1;
    prealloc[0].prev = NULL;
    prealloc[n_preallocs - 1].next = NULL;
    prealloc[n_preallocs - 1].prev = prealloc + (n_preallocs - 2);
    for (unsigned i = 1; i < n_preallocs - 1; i++) {
        prealloc[i].next = prealloc + (i + 1);
        prealloc[i].prev = prealloc + (i - 1);
    }
    cache->pool_head = prealloc + 0;
    cache->pool_size = n_preallocs;
}

static void remove_node_from_cache(hvr_sparse_vec_cache_node_t *iter,
        hvr_sparse_vec_cache_t *cache) {
    const unsigned bucket = CACHE_BUCKET(iter->vert);
    assert(iter->pending_comm == 0);

    // Need to fix the prev and next elements in this
    if (iter->prev == NULL && iter->next == NULL) {
        assert(cache->buckets[bucket] == iter);
        cache->buckets[bucket] = NULL;
    } else if (iter->next == NULL) {
        iter->prev->next = NULL;
    } else if (iter->prev == NULL) {
        cache->buckets[bucket] = iter->next;
        iter->next->prev = NULL;
    } else {
        iter->prev->next = iter->next;
        iter->next->prev = iter->prev;
    }

    // Then remove from the LRU list
    if (iter->lru_prev == NULL && iter->lru_next == NULL) {
        assert(cache->lru_head == iter && cache->lru_tail == iter);
        cache->lru_head = NULL;
        cache->lru_tail = NULL;
    } else if (iter->lru_next == NULL) {
        assert(cache->lru_tail == iter);
        cache->lru_tail = iter->lru_prev;
        iter->lru_prev->lru_next = NULL;
    } else if (iter->lru_prev == NULL) {
        assert(cache->lru_head == iter);
        cache->lru_head = iter->lru_next;
        iter->lru_next->lru_prev = NULL;
    } else {
        iter->lru_prev->lru_next = iter->lru_next;
        iter->lru_next->lru_prev = iter->lru_prev;
    }
}

void hvr_sparse_vec_cache_clear_old(hvr_sparse_vec_cache_t *cache,
        hvr_time_t timestep, int pe) {
    for (unsigned bucket = 0; bucket < HVR_CACHE_BUCKETS; bucket++) {
        hvr_sparse_vec_cache_node_t *iter = cache->buckets[bucket];

        while (iter) {
            // Always have to keep pending nodes
            if (iter->pending_comm) {
                iter = iter->next;
                continue;
            }

            hvr_time_t newest_timestamp;
            int success = get_newest_timestamp(&(iter->vec), &newest_timestamp);

            if (!success || (newest_timestamp <= timestep &&
                    timestep - newest_timestamp > CACHED_TIMESTEPS_TOLERANCE)) {

                // Removes node from bucket and LRU lists
                remove_node_from_cache(iter, cache);

                // Finally, insert into our list of free nodes
                hvr_sparse_vec_cache_node_t *tmp = iter->next;
                iter->next = cache->pool_head;
                iter->prev = NULL;
                if (cache->pool_head) {
                    cache->pool_head->prev = iter;
                }
                cache->pool_head = iter;
                iter = tmp;
            } else {
                // Keep in the cache
                iter = iter->next;
            }
        }
    }
}

hvr_sparse_vec_cache_node_t *hvr_sparse_vec_cache_lookup(hvr_vertex_id_t vert,
        hvr_sparse_vec_cache_t *cache, hvr_time_t target_timestep) {
    const unsigned bucket = CACHE_BUCKET(vert);

    hvr_sparse_vec_cache_node_t *iter = cache->buckets[bucket];
    while (iter) {
        if (iter->vert == vert) break;
        iter = iter->next;
    }

    if (iter == NULL) {
        cache->nmisses++;
    } else {
        cache->nhits++;
    }
    return iter;
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

    shmem_quiet();

    *counter = *counter + 1;
}

static void get_cache_metrics(hvr_sparse_vec_cache_t *cache,
        hvr_time_t curr_timestep,
        hvr_time_t *max_timestep_out, hvr_time_t *min_timestep_out,
        double *avg_timestep_out,
        unsigned *count_requested_on_this_timestep_out) {
    hvr_time_t max_timestep = -1;
    hvr_time_t min_timestep = MAX_TIMESTAMP;
    double avg_timestep = 0.0;
    unsigned avg_timestep_count = 0;
    unsigned count_requested_on_this_timestep = 0;
    for (unsigned b = 0; b < HVR_CACHE_BUCKETS; b++) {
        hvr_sparse_vec_cache_node_t *iter = cache->buckets[b];
        while (iter) {
            if (iter->requested_on == curr_timestep) {
                count_requested_on_this_timestep++;
            } else {
                /*
                 * Only get metrics on those actors requested in earlier
                 * timesteps
                 */
                assert(!iter->pending_comm);
                hvr_time_t newest_timestamp;
                int success = get_newest_timestamp(&(iter->vec),
                        &newest_timestamp);
                if (success) {
                    if (newest_timestamp > max_timestep) {
                        max_timestep = newest_timestamp;
                    }
                    if (newest_timestamp < min_timestep) {
                        min_timestep = newest_timestamp;
                    }
                    avg_timestep += newest_timestamp;
                    avg_timestep_count++;
                }
            }
            iter = iter->next;
        }
    }
    avg_timestep /= (double)avg_timestep_count;

    *max_timestep_out = max_timestep;
    *min_timestep_out = min_timestep;
    *avg_timestep_out = avg_timestep;
    *count_requested_on_this_timestep_out = count_requested_on_this_timestep;
}

hvr_sparse_vec_cache_node_t *hvr_sparse_vec_cache_reserve(
        hvr_vertex_id_t vert, hvr_sparse_vec_cache_t *cache,
        hvr_time_t curr_timestep, int pe) {
    // Assume that vec is not already in the cache, but don't enforce this
    hvr_sparse_vec_cache_node_t *new_node = NULL;
    if (cache->pool_head) {
        // Look for an already free node
        new_node = cache->pool_head;
        cache->pool_head = new_node->next;
        if (cache->pool_head) {
            cache->pool_head->prev = NULL;
        }
    } else if (cache->lru_tail && !cache->lru_tail->pending_comm) { 
        /*
         * Find the oldest requested node, check it isn't pending communication,
         * and if so use it.
         */
        new_node = cache->lru_tail;

        // Removes node from bucket and LRU lists
        remove_node_from_cache(new_node, cache);
    } else {
        // No valid node found, print an error
        hvr_time_t max_timestep, min_timestep;
        double avg_timestep;
        unsigned count_requested_on_this_timestep;

        get_cache_metrics(cache, curr_timestep, &max_timestep, &min_timestep,
                &avg_timestep, &count_requested_on_this_timestep);

        fprintf(stderr, "ERROR: PE %d exhausted %u cache slots, curr timestep "
                "= %d, max timestep = %d, min timestep = %d, average timestep "
                "= %f, # requested on this timestep = %u\n", pe,
                cache->pool_size, curr_timestep, max_timestep, min_timestep,
                avg_timestep, count_requested_on_this_timestep);
        abort();
    }

    const unsigned bucket = CACHE_BUCKET(vert);
    assert(new_node->pending_comm == 0);
    new_node->vert = vert;
    new_node->pending_comm = 1;
    new_node->requested_on = curr_timestep;

    // Insert into the appropriate bucket
    if (cache->buckets[bucket]) {
        cache->buckets[bucket]->prev = new_node;
    }
    new_node->next = cache->buckets[bucket];
    new_node->prev = NULL;
    cache->buckets[bucket] = new_node;

    // Insert into the LRU list
    new_node->lru_prev = NULL;
    new_node->lru_next = cache->lru_head;
    if (cache->lru_head) {
        cache->lru_head->lru_prev = new_node;
        cache->lru_head = new_node;
    } else {
        assert(cache->lru_tail == NULL);
        cache->lru_head = cache->lru_tail = new_node;
    }

    return new_node;
}

static hvr_set_t *hvr_create_empty_set_helper(hvr_internal_ctx_t *ctx,
        const int nelements, hvr_set_t *set, bit_vec_element_type *bit_vector) {
    set->bit_vector = bit_vector;
    set->nelements = nelements;
    set->n_contained = 0;

    memset(set->bit_vector, 0x00, nelements * sizeof(bit_vec_element_type));

    return set;
}

hvr_set_t *hvr_create_empty_set_symmetric(const unsigned nvals,
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

void hvr_fill_set(hvr_set_t *s) {
    memset(s->bit_vector, 0xff, s->nelements * sizeof(*(s->bit_vector)));
}

hvr_set_t *hvr_create_full_set(const unsigned nvals, hvr_ctx_t in_ctx) {
    hvr_set_t *empty_set = hvr_create_empty_set(nvals, in_ctx);
    for (unsigned i = 0; i < nvals; i++) {
        hvr_set_insert(i, empty_set);
    }
    return empty_set;
}

static int hvr_set_insert_internal(int val,
        bit_vec_element_type *bit_vector) {
    const int element = val / (sizeof(*bit_vector) * BITS_PER_BYTE);
    const int bit = val % (sizeof(*bit_vector) * BITS_PER_BYTE);
    const bit_vec_element_type old_val = bit_vector[element];
    const bit_vec_element_type new_val =
        (old_val | ((bit_vec_element_type)1 << bit));
    bit_vector[element] = new_val;
    return old_val != new_val;
}

int hvr_set_insert(int val, hvr_set_t *set) {
    const int changed = hvr_set_insert_internal(val, set->bit_vector);
    if (changed) {
        if (set->n_contained < PE_SET_CACHE_SIZE) {
            (set->cache)[set->n_contained] = val;
        }
        set->n_contained++;
    }
    return changed;
}

void hvr_set_wipe(hvr_set_t *set) {
    memset(set->bit_vector, 0x00, set->nelements * sizeof(*(set->bit_vector)));
    set->n_contained = 0;
}

int hvr_set_contains_internal(int val, bit_vec_element_type *bit_vector) {
    const int element = val / (sizeof(*bit_vector) * BITS_PER_BYTE);
    const int bit = val % (sizeof(*bit_vector) * BITS_PER_BYTE);
    const bit_vec_element_type old_val = bit_vector[element];
    if (old_val & ((bit_vec_element_type)1 << bit)) {
        return 1;
    } else {
        return 0;
    }
}

int hvr_set_contains(int val, hvr_set_t *set) {
    return hvr_set_contains_internal(val, set->bit_vector);
}

unsigned hvr_set_count(hvr_set_t *set) {
    return set->n_contained;
}

void hvr_set_destroy(hvr_set_t *set) {
    free(set->bit_vector);
    free(set);
}

void hvr_set_to_string(hvr_set_t *set, char *buf, unsigned buflen,
        unsigned *values) {
    int offset = snprintf(buf, buflen, "{");

    const size_t nvals = set->nelements * sizeof(bit_vec_element_type) *
        BITS_PER_BYTE;
    for (unsigned i = 0; i < nvals; i++) {
        if (hvr_set_contains(i, set)) {
            offset += snprintf(buf + offset, buflen - offset - 1, " %u", i);
            assert(offset < buflen);
            if (values) {
                offset += snprintf(buf + offset, buflen - offset - 1, ": %u",
                        values[i]);
                assert(offset < buflen);
            }
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

/*
 * Add an edge from a local vertex local_vertex_id to another vertex (possibly
 * local or remote) global_vertex_id.
 */
void hvr_add_edge(const hvr_vertex_id_t local_vertex_id,
        const hvr_vertex_id_t global_vertex_id, hvr_edge_set_t *set) {
    // If it already exists, just returns existing node in tree
    set->tree = hvr_tree_insert(set->tree, local_vertex_id);
    hvr_avl_tree_node_t *inserted = hvr_tree_find(set->tree, local_vertex_id);
    inserted->subtree = hvr_tree_insert(inserted->subtree, global_vertex_id);
}

int hvr_have_edge(const hvr_vertex_id_t local_vertex_id,
        const hvr_vertex_id_t global_vertex_id, hvr_edge_set_t *set) {
    hvr_avl_tree_node_t *inserted = hvr_tree_find(set->tree, local_vertex_id);
    if (inserted == NULL) {
        return 0;
    }
    return hvr_tree_find(inserted->subtree, global_vertex_id) != NULL;
}

size_t hvr_count_edges(const hvr_vertex_id_t local_vertex_id,
        hvr_edge_set_t *set) {
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

hvr_graph_id_t hvr_graph_create(hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;

    hvr_graph_id_t next_graph = (ctx->allocated_graphs)++;

    if (next_graph >= BITS_PER_BYTE * sizeof(hvr_graph_id_t)) {
        fprintf(stderr, "Ran out of graph IDs\n");
        abort();
    }

    return (1 << next_graph);
}

/*
 * For every other PE in this simulation, reach out and take the bit vector of
 * partitions that they have had actors in during a recent time window.
 *
 * Then, for each partition set in their bit vector check locally if an actor in
 * that partition might interact with any actor in any partition we have actors
 * in locally. If so, add that PE as a neighbor.
 *
 * The time complexity is generally linear with the number of partitions.
 */
static void update_neighbors_based_on_partitions(hvr_internal_ctx_t *ctx) {
    static hvr_set_t *full_part_set = NULL;
    if (full_part_set == NULL) {
        full_part_set = hvr_create_full_set(ctx->n_partitions, ctx);
    }

    hvr_partition_t interacting_partitions[MAX_INTERACTING_PARTITIONS];
    unsigned n_interacting_partitions;
    hvr_set_wipe(ctx->my_neighbors);

    for (hvr_partition_list_node_t *curr = ctx->active_partitions_list; curr;
            curr = curr->next) {
        /*
         * Find all of the partitions that any locally active partition might
         * interact with
         */
        if (ctx->might_interact(curr->part, full_part_set,
                    interacting_partitions, &n_interacting_partitions,
                    MAX_INTERACTING_PARTITIONS, ctx)) {
            for (unsigned i = 0; i < n_interacting_partitions; i++) {
                unsigned part = interacting_partitions[i];

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
    }

#ifdef VERBOSE
    printf("PE %d is talking to %d other PEs on timestep %d\n", ctx->pe,
            hvr_set_count(ctx->my_neighbors), ctx->timestep);
#endif
}

static double sparse_vec_distance_measure(hvr_sparse_vec_t *a,
        hvr_sparse_vec_t *b,
        const hvr_time_t a_max_timestep,
        const hvr_time_t b_max_timestep,
        const unsigned min_spatial_feature,
        const unsigned max_spatial_feature,
        unsigned *nhits,
        unsigned *nmisses) {
    const int a_bucket = hvr_sparse_vec_find_bucket(a, a_max_timestep,
            nhits, nmisses);
    assert(a_bucket >= 0);
    const int b_bucket = hvr_sparse_vec_find_bucket(b, b_max_timestep,
            nhits, nmisses);
    assert(b_bucket >= 0);

    double acc = 0.0;
    for (unsigned f = min_spatial_feature; f <= max_spatial_feature; f++) {
        int success;
        double a_val, b_val;

        success = find_feature_in_const_attrs(a, f, &a_val);
        if (!success) {
            success = find_feature_in_bucket(a, a_bucket, f, &a_val);
            assert(success == 1);
        }

        success = find_feature_in_const_attrs(b, f, &b_val);
        if (!success) {
            success = find_feature_in_bucket(b, b_bucket, f, &b_val);
            assert(success == 1);
        }

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
        hvr_sparse_vec_cache_t *cache, int *is_cached) {
    hvr_sparse_vec_cache_node_t *cached = hvr_sparse_vec_cache_lookup(
            construct_vertex_id(src_pe, offset), cache, ctx->timestep - 1);

    if (cached) {
        // May still be pending
        if (is_cached) *is_cached = 1;
        return cached;
    } else {
        if (is_cached) *is_cached = 0;
        hvr_sparse_vec_t *src = &(ctx->pool->pool[offset]);
        hvr_sparse_vec_cache_node_t *node = hvr_sparse_vec_cache_reserve(
                construct_vertex_id(src_pe, offset), cache, ctx->timestep,
                ctx->pe);
        get_remote_vec_nbi_uncached(&(node->vec), src, src_pe);
        return node;
    }
}

static void get_remote_vec_blocking(hvr_vertex_id_t vertex,
        hvr_sparse_vec_t *dst, hvr_internal_ctx_t *ctx) {
    hvr_sparse_vec_cache_node_t *placeholder = get_remote_vec_nbi(
            VERTEX_ID_OFFSET(vertex), VERTEX_ID_PE(vertex), ctx,
            &ctx->vec_cache, NULL);
    hvr_sparse_vec_cache_quiet(&ctx->vec_cache,
            &(ctx->cache_perf_info.quiet_counter));
    memcpy(dst, &(placeholder->vec), sizeof(*dst));
}

static void check_edges_to_add(hvr_sparse_vec_t *remote_vec,
        hvr_partition_t *interacting_partitions,
        unsigned n_interacting_partitions,
        hvr_time_t other_pes_timestep, hvr_internal_ctx_t *ctx,
        unsigned long long *n_distance_measures) {
    unsigned long long local_n_distance_measures = 0;
    assert(remote_vec->id != HVR_INVALID_VERTEX_ID);

    if (remote_vec->created_timestamp >= ctx->timestep) {
        /*
         * Early abort on remote actors that didn't exist until our current
         * timestep (so we'll have no past state to look back at).
         */
        return;
    }

    if (remote_vec->deleted_timestamp < ctx->timestep) {
        /*
         * Early abort on remote actors that were deleted before our current
         * timestep.
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
                        ctx->timestep, other_pes_timestep,
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
        hvr_sparse_vec_cache_t *vec_cache,
        unsigned long long *getmem_time,
        unsigned long long *update_edge_time,
        unsigned long long *out_n_edge_checks,
        unsigned long long *out_partition_checks,
        unsigned long long *quiet_counter,
        unsigned long long *out_n_distance_measures,
        unsigned *n_cached_remote_fetches,
        unsigned *n_uncached_remote_fetches) {
    hvr_partition_t interacting_partitions[MAX_INTERACTING_PARTITIONS];
    unsigned n_interacting_partitions;

    unsigned long long n_distance_measures = 0;
    unsigned long long total_partition_checks = 0;

    unsigned long long n_edge_checks = 0;

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
        shmem_getmem(ctx->other_actor_to_partition_map,
                ctx->actor_to_partition_map,
                ctx->pool->pool_size * sizeof(hvr_partition_t), target_pe);
        runlock_actor_to_partition(target_pe, ctx);

        // Check what timestep the remote PE is on currently.
        hvr_time_t other_pes_timestep;
        shmem_getmem(&other_pes_timestep, (hvr_time_t *)ctx->symm_timestep,
                sizeof(other_pes_timestep), target_pe);
        /*
         * Prevent this PE from seeing into the future, if the other PE is ahead
         * of us.
         */
        if (other_pes_timestep > ctx->timestep) {
            other_pes_timestep = ctx->timestep;
        }

#define BUFFERING 512
        struct {
            hvr_sparse_vec_cache_node_t *node;
            hvr_partition_t interacting_partitions[MAX_INTERACTING_PARTITIONS];
            unsigned n_interacting_partitions;
        } buffered[BUFFERING];
        unsigned nbuffered = 0;

#ifdef FINE_GRAIN_TIMING
        unsigned long long start_time = hvr_current_time_us();
#endif
        // For each vertex on the remote PE
        for (hvr_vertex_id_t j = 0; j < ctx->pool->pool_size; j++) {
            const hvr_partition_t actor_partition =
                (ctx->other_actor_to_partition_map)[j];

            // If remote actor j might interact with anything in our local PE.
            if (actor_partition != HVR_INVALID_PARTITION &&
                    ctx->might_interact(actor_partition,
                        ctx->partition_time_window, interacting_partitions,
                        &n_interacting_partitions, MAX_INTERACTING_PARTITIONS,
                        ctx)) {
                int is_cached;
                buffered[nbuffered].node = get_remote_vec_nbi(j, target_pe, ctx,
                        vec_cache, &is_cached);
                memcpy(buffered[nbuffered].interacting_partitions,
                        interacting_partitions,
                        MAX_INTERACTING_PARTITIONS * sizeof(hvr_partition_t));
                buffered[nbuffered].n_interacting_partitions =
                    n_interacting_partitions;
                nbuffered++;
                if (is_cached) {
                    *n_cached_remote_fetches += 1;
                } else {
                    *n_uncached_remote_fetches += 1;
                }

                if (nbuffered == BUFFERING) {
                    // Process buffered
                    hvr_sparse_vec_cache_quiet(vec_cache, quiet_counter);
#ifdef FINE_GRAIN_TIMING
                    *getmem_time += (hvr_current_time_us() - start_time);
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
                    start_time = hvr_current_time_us();
#endif
                    nbuffered = 0;
                }

                n_edge_checks++;
             }
         }
 
         if (nbuffered > 0) {
             // Process buffered
             hvr_sparse_vec_cache_quiet(vec_cache, quiet_counter);
 #ifdef FINE_GRAIN_TIMING
             *getmem_time += (hvr_current_time_us() - start_time);
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

    *out_n_edge_checks = n_edge_checks;
    *out_partition_checks = total_partition_checks;
    *out_n_distance_measures = n_distance_measures;
}

static hvr_partition_t wrap_actor_to_partition(hvr_sparse_vec_t *vec,
        hvr_internal_ctx_t *ctx) {
    hvr_partition_t partition = ctx->actor_to_partition(vec, ctx);
    if (partition >= ctx->n_partitions) {
        char buf[1024];
        hvr_sparse_vec_dump(vec, buf, 1024, ctx);

        fprintf(stderr, "Invalid partition %d (# partitions = %d) returned "
                "from actor_to_partition. Vector = {%s}\n", partition,
                ctx->n_partitions, buf);
        abort();
    }
    return partition;
}

/*
 * Update the mapping from each local actor to the partition it belongs to
 * (actor_to_partition_map) as well as information on the last timestep that
 * used each partition (last_timestep_using_partition). This mapping is stored
 * per-timestep, in a circular buffer.
 */
static void update_actor_partitions(hvr_internal_ctx_t *ctx) {
    hvr_sparse_vec_t **partition_lists = ctx->partition_lists;
    unsigned *partition_lists_lengths = ctx->partition_lists_lengths;
    memset(partition_lists, 0x00,
            sizeof(hvr_sparse_vec_t *) * ctx->n_partitions);
    memset(partition_lists_lengths, 0x00, sizeof(unsigned) * ctx->n_partitions);

    wlock_actor_to_partition(ctx->pe, ctx);

    hvr_vertex_iter_t iter;
    hvr_vertex_iter_init_with_dead_vertices(&iter, ctx->interacting_graphs_mask,
            ctx);
    for (hvr_sparse_vec_t *curr = hvr_vertex_iter_next(&iter); curr;
            curr = hvr_vertex_iter_next(&iter)) {
        assert(curr->id != HVR_INVALID_VERTEX_ID);

        hvr_partition_t partition = wrap_actor_to_partition(curr, ctx);

        // Update a mapping from local actor to the partition it belongs to
        (ctx->actor_to_partition_map)[VERTEX_ID_OFFSET(curr->id)] =
            partition;

        /*
         * This doesn't necessarily need to be in the critical section,
         * but to avoid multiple traversal over all actors (i.e. a
         * second loop over # local vertices outside of the critical
         * section) we stick it here.
         */
        if (curr->deleted_timestamp != MAX_TIMESTAMP) {
            if (curr->deleted_timestamp >
                    (ctx->last_timestep_using_partition)[partition]) {
                (ctx->last_timestep_using_partition)[partition] =
                    curr->deleted_timestamp;
            }
        } else {
            (ctx->last_timestep_using_partition)[partition] = ctx->timestep;
        }

        if (partition_lists[partition]) {
            curr->next_in_partition = partition_lists[partition];
            partition_lists[partition] = curr;
        } else {
            curr->next_in_partition = NULL;
            partition_lists[partition] = curr;
        }
        partition_lists_lengths[partition] += 1;
    }

    wunlock_actor_to_partition(ctx->pe, ctx);
}

static unsigned free_old_actors(hvr_time_t timestep, hvr_internal_ctx_t *ctx) {
    hvr_vertex_iter_t iter;
    hvr_vertex_iter_init_with_dead_vertices(&iter, HVR_ALL_GRAPHS, ctx);
    unsigned count_vertices = 0;
    for (hvr_sparse_vec_t *curr = hvr_vertex_iter_next(&iter); curr;
            curr = hvr_vertex_iter_next(&iter)) {
        if (curr->deleted_timestamp != MAX_TIMESTAMP &&
                timestep - curr->deleted_timestamp >= HVR_BUCKETS) {
            // Release back into the pool
            hvr_free_sparse_vecs(curr, 1, ctx);
        } else {
            count_vertices++;
        }
    }
    return count_vertices;
}

/*
 * partition_time_window stores a list of the partitions that the local PE has
 * had actors inside during some window of recent timesteps. This updates the
 * partitions in that window set based on the results of
 * update_actor_partitions.
 */
static void update_partition_time_window(hvr_internal_ctx_t *ctx) {
    hvr_set_wipe(ctx->tmp_partition_time_window);

    while (ctx->active_partitions_list) {
        hvr_partition_list_node_t *head = ctx->active_partitions_list;
        ctx->active_partitions_list = head->next;
        head->next = ctx->partitions_list_pool;
        ctx->partitions_list_pool = head;
    }

    // Update the set of partitions in a temporary buffer
    for (unsigned p = 0; p < ctx->n_partitions; p++) {
        const hvr_time_t last_use = ctx->last_timestep_using_partition[p];
        if (last_use >= 0) {
            assert(last_use <= ctx->timestep);
            if (ctx->timestep - last_use < HVR_BUCKETS) {
                hvr_set_insert(p, ctx->tmp_partition_time_window);

                hvr_partition_list_node_t *head = ctx->partitions_list_pool;
                assert(head);
                ctx->partitions_list_pool = head->next;
                head->part = p;
                head->next = ctx->active_partitions_list;
                ctx->active_partitions_list = head;
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

void hvr_init(const hvr_partition_t n_partitions,
        hvr_update_metadata_func update_metadata,
        hvr_might_interact_func might_interact,
        hvr_check_abort_func check_abort,
        hvr_actor_to_partition actor_to_partition,
        hvr_start_time_step start_time_step,
        hvr_graph_id_t *interacting_graphs, unsigned n_interacting_graphs,
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

    new_ctx->partition_time_window = hvr_create_empty_set_symmetric(
            n_partitions, new_ctx);
    new_ctx->tmp_partition_time_window = hvr_create_empty_set(
            n_partitions, new_ctx);

    new_ctx->actor_to_partition_lock = hvr_rwlock_create_n(1);

    new_ctx->partition_time_window_lock = hvr_rwlock_create_n(1);

    hvr_partition_list_node_t *all_nodes = (hvr_partition_list_node_t *)malloc(
            N_PARTITION_NODES_PREALLOC * sizeof(*all_nodes));
    assert(all_nodes);
    new_ctx->partitions_list_pool = NULL;
    new_ctx->active_partitions_list = NULL;
    for (unsigned i = 0; i < N_PARTITION_NODES_PREALLOC; i++) {
        hvr_partition_list_node_t *new_node = all_nodes + i;
        new_node->next = new_ctx->partitions_list_pool;
        new_ctx->partitions_list_pool = new_node;
    }

    assert(n_partitions <= HVR_INVALID_PARTITION);
    new_ctx->n_partitions = n_partitions;

    new_ctx->actor_to_partition_map = (hvr_partition_t *)shmem_malloc_wrapper(
            new_ctx->pool->pool_size * sizeof(hvr_partition_t));
    assert(new_ctx->actor_to_partition_map);
    for (unsigned i = 0; i < new_ctx->pool->pool_size; i++) {
        (new_ctx->actor_to_partition_map)[i] = HVR_INVALID_PARTITION;
    }
    new_ctx->other_actor_to_partition_map = (hvr_partition_t *)malloc(
            new_ctx->pool->pool_size * sizeof(hvr_partition_t));
    assert(new_ctx->other_actor_to_partition_map);

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
    } else {
        char profiling_filename[1024];
        sprintf(profiling_filename, "%d.prof", new_ctx->pe);
        profiling_fp = fopen(profiling_filename, "w");
        assert(profiling_fp);
    }

    new_ctx->my_neighbors = hvr_create_empty_set(new_ctx->npes, new_ctx);

    new_ctx->coupled_pes = hvr_create_empty_set_symmetric(new_ctx->npes,
            new_ctx);
    hvr_set_insert(new_ctx->pe, new_ctx->coupled_pes);

    new_ctx->coupled_pes_values = (hvr_sparse_vec_t *)shmem_malloc_wrapper(
            new_ctx->npes * sizeof(hvr_sparse_vec_t));
    assert(new_ctx->coupled_pes_values);
    for (unsigned i = 0; i < new_ctx->npes; i++) {
        hvr_sparse_vec_init(&(new_ctx->coupled_pes_values)[i],
                HVR_INVALID_GRAPH, new_ctx);
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
    new_ctx->partition_lists_lengths = (unsigned *)malloc(
            sizeof(unsigned) * new_ctx->n_partitions);
    assert(new_ctx->partition_lists_lengths);

    new_ctx->interacting_graphs = (hvr_graph_id_t *)malloc(
            n_interacting_graphs * sizeof(hvr_graph_id_t));
    assert(new_ctx->interacting_graphs);
    memcpy(new_ctx->interacting_graphs, interacting_graphs,
            n_interacting_graphs * sizeof(hvr_graph_id_t));
    new_ctx->n_interacting_graphs = n_interacting_graphs;
    new_ctx->interacting_graphs_mask = 0;
    for (unsigned i = 0; i < n_interacting_graphs; i++) {
        new_ctx->interacting_graphs_mask = (new_ctx->interacting_graphs_mask |
                interacting_graphs[i]);
    }

    hvr_sparse_vec_cache_init(&new_ctx->vec_cache);

    // Print the number of bytes allocated
    // shmem_malloc_wrapper(0);

    shmem_barrier_all();
}

void hvr_sparse_vec_get_neighbors(hvr_vertex_id_t vertex,
        hvr_ctx_t in_ctx, hvr_vertex_id_t **neighbors_out,
        unsigned *n_neighbors_out) {
    unsigned remote_gets, local_gets, remote_fetches;
    hvr_sparse_vec_get_neighbors_with_metrics(vertex, in_ctx, neighbors_out,
            n_neighbors_out, &local_gets, &remote_gets, &remote_fetches,
            &remote_fetches);
}

static void handle_buffered_nodes(unsigned n_nodes_buffered,
        hvr_sparse_vec_t *remote_vec,
        hvr_internal_ctx_t *ctx,
        unsigned *n_neighbors_out,
        unsigned *neighbors_buf_len,
        hvr_vertex_id_t **neighbors_buf) {
    unsigned long long dummy;
    hvr_sparse_vec_cache_quiet(&ctx->vec_cache, &dummy);

    for (unsigned i = 0; i < n_nodes_buffered; i++) {
        hvr_sparse_vec_t *remote_remote_vec =
            &((ctx->neighbor_buffer)[i]->vec);
        if (remote_remote_vec->created_timestamp < ctx->timestep &&
                remote_remote_vec->deleted_timestamp >= ctx->timestep) {
            const double distance = sparse_vec_distance_measure(
                    remote_vec, remote_remote_vec,
                    ctx->timestep, ctx->timestep,
                    ctx->min_spatial_feature,
                    ctx->max_spatial_feature,
                    &ctx->n_vector_cache_hits,
                    &ctx->n_vector_cache_misses);
            if (distance < ctx->connectivity_threshold *
                    ctx->connectivity_threshold) {
                // Add as a neighbor
                if (*n_neighbors_out == *neighbors_buf_len) {
                    // Resize
                    *neighbors_buf_len *= 2;
                    *neighbors_buf = (hvr_vertex_id_t *)realloc(
                            *neighbors_buf,
                            *neighbors_buf_len * sizeof(*neighbors_buf));
                    assert(*neighbors_buf);
                }
                (*neighbors_buf)[*n_neighbors_out] = remote_remote_vec->id;

                *n_neighbors_out += 1;
            }
        }
    }

}

/*
 * Retrieve a list of vertices that have edges with the specified vertex
 * (whether remote or local).
 */
int hvr_sparse_vec_get_neighbors_with_metrics(hvr_vertex_id_t vertex,
        hvr_ctx_t in_ctx, hvr_vertex_id_t **neighbors_out,
        unsigned *n_neighbors_out,
        unsigned *count_local_gets,
        unsigned *count_remote_gets,
        unsigned *count_cached_remote_fetches,
        unsigned *count_uncached_remote_fetches) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    static hvr_set_t *full_partition_set = NULL;
    static hvr_vertex_id_t *neighbors_buf = NULL;
    static unsigned neighbors_buf_len = 0;
    if (full_partition_set == NULL) {
        full_partition_set = hvr_create_empty_set(ctx->n_partitions, ctx);

        neighbors_buf_len = 256;
        neighbors_buf = (hvr_vertex_id_t *)malloc(
                neighbors_buf_len * sizeof(*neighbors_buf));
        assert(neighbors_buf);
    }

    int owning_pe = VERTEX_ID_PE(vertex);
    *n_neighbors_out = 0;

    if (owning_pe == ctx->pe) {
        // I already have the neighbors of this vertex, as it is local
        hvr_avl_tree_node_t *vertex_edge_tree = hvr_tree_find(
                ctx->edges->tree, vertex);

        // If vertex_edge_tree is NULL, this node has no edges
        if (vertex_edge_tree) {
            *n_neighbors_out = hvr_tree_linearize(neighbors_out,
                    vertex_edge_tree->subtree);
        }
        *count_local_gets += 1;
        return 1;
    } else {
        // Must figure out the edges on a remote vertex
        unsigned n_nodes_buffered = 0;
        hvr_fill_set(full_partition_set);

        hvr_sparse_vec_t remote_vec;
        get_remote_vec_blocking(vertex, &remote_vec,ctx);

        *count_remote_gets += 1;

        // Compute partition for this vertex
        hvr_partition_t partition = wrap_actor_to_partition(&remote_vec, ctx);

        hvr_partition_t interacting_partitions[MAX_INTERACTING_PARTITIONS];
        unsigned n_interacting_partitions;
        // Figure out all partitions it might interact with
        if (ctx->might_interact(partition, full_partition_set,
                    interacting_partitions, &n_interacting_partitions,
                    MAX_INTERACTING_PARTITIONS, ctx)) {

            for (int interacting_part_index = 0;
                    interacting_part_index < n_interacting_partitions;
                    interacting_part_index++) {
                hvr_partition_t interacting_partition =
                    interacting_partitions[interacting_part_index];

                const int partition_owner_pe = interacting_partition /
                    ctx->partitions_per_pe;
                const int partition_offset = interacting_partition %
                    ctx->partitions_per_pe;

                shmem_getmem(ctx->local_pes_per_partition_buffer,
                        ctx->pes_per_partition + (partition_offset *
                            ctx->partitions_per_pe_vec_length_in_words),
                        ctx->partitions_per_pe_vec_length_in_words *
                            sizeof(unsigned),
                        partition_owner_pe);

                for (unsigned p = 0; p < ctx->npes; p++) {
                    unsigned word = p / BITS_PER_WORD;
                    unsigned bit = p % BITS_PER_WORD;
                    unsigned bit_mask = (1U << bit);

                    unsigned word_val =
                        (ctx->local_pes_per_partition_buffer)[word];
                    if (word_val & bit_mask) {
                        // PE 'p' may have neighbors of this vertex

                        // Grab the target PEs mapping from actors to partitions
                        rlock_actor_to_partition(p, ctx);
                        shmem_getmem(ctx->other_actor_to_partition_map,
                                ctx->actor_to_partition_map,
                                ctx->pool->pool_size * sizeof(hvr_partition_t),
                                p);
                        runlock_actor_to_partition(p, ctx);

                        /*
                         * For each vertex on the remote PE that may be a
                         * neighbor of vertex
                         */
                        for (hvr_vertex_id_t j = 0; j < ctx->pool->pool_size;
                                j++) {
                            const hvr_partition_t remote_actor_partition =
                                (ctx->other_actor_to_partition_map)[j];

                            int is_interacting = 0;
                            for (unsigned k = 0; k < n_interacting_partitions;
                                    k++) {
                                if (interacting_partitions[k] ==
                                        remote_actor_partition) {
                                    is_interacting = 1;
                                    break;
                                }
                            }

                            if (is_interacting) {
                                /*
                                 * Get this vector and check if an edge should
                                 * be added
                                 */
                                int is_cached;
                                hvr_sparse_vec_cache_node_t *node = get_remote_vec_nbi(
                                        j, p, ctx, &ctx->vec_cache, &is_cached);
                                if (is_cached) {
                                    *count_cached_remote_fetches += 1;
                                } else {
                                    *count_uncached_remote_fetches += 1;
                                }
                                if (n_nodes_buffered == EDGE_GET_BUFFERING) {
                                    handle_buffered_nodes(n_nodes_buffered,
                                            &remote_vec, ctx,
                                            n_neighbors_out, &neighbors_buf_len,
                                            &neighbors_buf);
                                    n_nodes_buffered = 0;
                                }
                                assert(n_nodes_buffered < EDGE_GET_BUFFERING);
                                (ctx->neighbor_buffer)[n_nodes_buffered++] = node;
                            }
                        }

                    }
                }
            }
        }

        if (n_nodes_buffered > 0) {
            handle_buffered_nodes(n_nodes_buffered, &remote_vec, ctx,
                    n_neighbors_out, &neighbors_buf_len, &neighbors_buf);
        }

        *neighbors_out = neighbors_buf;

        return 0;
    }
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
       hvr_sparse_vec_cache_t *vec_cache, unsigned long long *quiet_counter) {
    // Buffer used to linearize neighbors list into

    // The list of edges for local actor i
    hvr_avl_tree_node_t *vertex_edge_tree = hvr_tree_find(
            ctx->edges->tree, vertex->id);

    // Update the metadata for actor i
    if (vertex_edge_tree != NULL) {
        // This vertex has edges
        hvr_vertex_id_t *neighbors;
        const size_t n_neighbors = hvr_tree_linearize(&neighbors,
                vertex_edge_tree->subtree);

        // Simplifying assumption for now
        if (n_neighbors > EDGE_GET_BUFFERING) {
            fprintf(stderr, "Invalid # neighbors - %lu > %u\n", n_neighbors,
                    EDGE_GET_BUFFERING);
            abort();
        }

        // Fetch all neighbors of this vertex
        const unsigned long long start_single_update = hvr_current_time_us();
        for (unsigned n = 0; n < n_neighbors; n++) {
            const hvr_vertex_id_t neighbor = neighbors[n];

            uint64_t other_pe = VERTEX_ID_PE(neighbor);
            uint64_t local_offset = VERTEX_ID_OFFSET(neighbor);

            hvr_sparse_vec_cache_node_t *cache_node = get_remote_vec_nbi(
                    local_offset, other_pe, ctx, vec_cache, NULL);
            (ctx->neighbor_buffer)[n] = cache_node;
        }

        const unsigned long long finish_neighbor_fetch = hvr_current_time_us();

        // Quiet any caches that were hit by the above fetches
        hvr_sparse_vec_cache_quiet(vec_cache, quiet_counter);

        /*
         * Copy from the cache nodes into a contiguous buffer before passing to
         * the user
         */
        unsigned actual_n_neighbors = 0;
        for (unsigned n = 0; n < n_neighbors; n++) {
            hvr_sparse_vec_t *vec = &((ctx->neighbor_buffer)[n]->vec);
            assert(vec->id != HVR_INVALID_VERTEX_ID);
            if (vec->deleted_timestamp < ctx->timestep) {
                continue;
            }

            memcpy(ctx->buffered_neighbors + actual_n_neighbors, vec,
                    sizeof(hvr_sparse_vec_t));
            actual_n_neighbors++;
        }

        const unsigned long long finish_neighbor_quiet = hvr_current_time_us();
        *fetch_neighbors_time += (finish_neighbor_fetch - start_single_update);
        *quiet_neighbors_time += (finish_neighbor_quiet - finish_neighbor_fetch);

        ctx->update_metadata(vertex, ctx->buffered_neighbors,
                actual_n_neighbors, to_couple_with, ctx);
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

void finalize_actor_for_timestep(hvr_sparse_vec_t *actor,
        const hvr_time_t timestep) {
    unsigned latest_bucket = prev_bucket(actor->next_bucket);
    if (actor->timestamps[latest_bucket] == timestep) {
        assert(actor->finalized[latest_bucket] < timestep);
        actor->finalized[latest_bucket] = timestep;
    }
}

static void finalize_actors_for_timestep(hvr_internal_ctx_t *ctx,
        hvr_graph_id_t target_graph, const hvr_time_t timestep) {
    hvr_vertex_iter_t iter;
    hvr_vertex_iter_init(&iter, target_graph, ctx);
    for (hvr_sparse_vec_t *curr = hvr_vertex_iter_next(&iter); curr;
            curr = hvr_vertex_iter_next(&iter)) {
        finalize_actor_for_timestep(curr, timestep);
    }
}

static void *aborting_thread(void *user_data) {
    int nseconds = atoi(getenv("HVR_HANG_ABORT"));
    assert(nseconds > 0);

    const unsigned long long start = hvr_current_time_us();
    while (hvr_current_time_us() - start < nseconds * 1000000) {
        sleep(10);
    }

    if (!this_pe_has_exited) {
        fprintf(stderr, "ERROR: HOOVER forcibly aborting because "
                "HVR_HANG_ABORT was set.\n");
        abort(); // Get a core dump
    }
    return NULL;
}

hvr_exec_info hvr_body(hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;

    shmem_barrier_all();
    ctx->other_pe_partition_time_window = hvr_create_empty_set(
            ctx->n_partitions, ctx);

    ctx->timestep = 1;
    *(ctx->symm_timestep) = ctx->timestep;

    finalize_actors_for_timestep(ctx, HVR_ALL_GRAPHS, 0);

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
    unsigned long long unused; unsigned u_unused;
    update_edges(ctx, &ctx->vec_cache, &unused, &unused, &unused, &unused,
            &unused, &unused, &u_unused, &u_unused);

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

        memset(&(ctx->cache_perf_info), 0x00, sizeof(ctx->cache_perf_info));

        if (ctx->start_time_step) {
            hvr_vertex_iter_t iter;
            hvr_vertex_iter_init(&iter, ctx->interacting_graphs_mask, ctx);
            ctx->start_time_step(&iter, ctx);
        }

        const unsigned long long finished_start = hvr_current_time_us();

        hvr_set_wipe(to_couple_with);

        unsigned long long sum_n_neighbors = 0;

        // Update each actor's metadata
        size_t count_vertices = 0;
        for (unsigned g = 0; g < ctx->n_interacting_graphs; g++) {
            hvr_graph_id_t target_graph = ctx->interacting_graphs[g];

            hvr_vertex_iter_t iter;
            hvr_vertex_iter_init(&iter, target_graph, ctx);
            for (hvr_sparse_vec_t *curr = hvr_vertex_iter_next(&iter); curr;
                    curr = hvr_vertex_iter_next(&iter)) {
                sum_n_neighbors += update_local_actor_metadata(curr,
                        to_couple_with,
                        &ctx->cache_perf_info.fetch_neighbors_time,
                        &ctx->cache_perf_info.quiet_neighbors_time,
                        &ctx->cache_perf_info.update_metadata_time, ctx,
                        &ctx->vec_cache, &ctx->cache_perf_info.quiet_counter);
                count_vertices++;
            }
        }
        const double avg_n_neighbors = (double)sum_n_neighbors /
            (double)count_vertices;

        const unsigned long long finished_updates = hvr_current_time_us();

        ctx->timestep += 1;
        *(ctx->symm_timestep) = ctx->timestep;

        __sync_synchronize();

        // Finalize the updates we just made
        finalize_actors_for_timestep(ctx, HVR_ALL_GRAPHS, ctx->timestep - 1);

        __sync_synchronize();

        // Update mapping from actors to partitions
        update_actor_partitions(ctx);

        /*
         * Remove cached remote vecs that are old relative to our next timestep.
         */
        hvr_sparse_vec_cache_clear_old(&ctx->vec_cache, ctx->timestep, ctx->pe);

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
        unsigned n_cached_remote_fetches_for_update_edges = 0;
        unsigned n_uncached_remote_fetches_for_update_edges = 0;
        update_edges(ctx, &ctx->vec_cache, &getmem_time, &update_edge_time,
                &n_edge_checks, &partition_checks,
                &ctx->cache_perf_info.quiet_counter,
                &n_distance_measures, &n_cached_remote_fetches_for_update_edges,
                &n_uncached_remote_fetches_for_update_edges);

        const unsigned long long finished_edge_adds = hvr_current_time_us();

        hvr_sparse_vec_t coupled_metric;
        memcpy(&coupled_metric, ctx->coupled_pes_values + ctx->pe,
                sizeof(coupled_metric));

        hvr_vertex_iter_t iter;
        hvr_vertex_iter_init(&iter, ctx->interacting_graphs_mask, ctx);
        should_abort = ctx->check_abort(&iter, ctx, to_couple_with,
                &coupled_metric);
        finalize_actor_for_timestep(
                &coupled_metric, ctx->timestep);

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

            char coupled_pes_str[1024];
            hvr_set_to_string(ctx->coupled_pes, coupled_pes_str, 1024, NULL);

            printf("PE %d - computed coupled value {%s} from %d "
                    "coupled PEs (%s) on timestep %d\n", ctx->pe, buf, ncoupled,
                    coupled_pes_str, ctx->timestep);
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

        /*
         * De-allocate any actors in the pool that have become old enough that
         * no PE will access them.
         */
        const unsigned n_local_verts = free_old_actors(oldest_timestep, ctx);

        const unsigned long long finished_throttling = hvr_current_time_us();

        if (ctx->dump_mode && ctx->pool->used_list) {
            // Assume that all vertices have the same features.
            unsigned nfeatures;
            unsigned features[HVR_BUCKET_SIZE];
            hvr_sparse_vec_unique_features(
                    ctx->pool->pool + ctx->pool->used_list->start_index,
                    ctx->timestep, features, &nfeatures);

            hvr_vertex_iter_t iter;
            hvr_vertex_iter_init(&iter, ctx->interacting_graphs_mask, ctx);
            for (hvr_sparse_vec_t *curr = hvr_vertex_iter_next(&iter); curr;
                    curr = hvr_vertex_iter_next(&iter)) {
                fprintf(ctx->dump_file, "%lu,%u,%ld,%d", curr->id,
                        nfeatures, (int64_t)ctx->timestep, ctx->pe);
                for (unsigned f = 0; f < nfeatures; f++) {
                    fprintf(ctx->dump_file, ",%u,%f", features[f],
                            hvr_sparse_vec_get(features[f], curr, ctx));
                }
                fprintf(ctx->dump_file, ",,\n");
            }
        }


        char neighbors_str[1024] = {'\0'};
        char partition_time_window_str[2048] = {'\0'};
#ifdef DETAILED_PRINTS
        hvr_set_to_string(ctx->my_neighbors, neighbors_str, 1024, NULL);

        hvr_set_to_string(ctx->partition_time_window, partition_time_window_str,
                2048, ctx->partition_lists_lengths);
#endif

        if (print_profiling) {
            fprintf(profiling_fp, "PE %d - timestep %d - total %f ms\n",
                    ctx->pe, ctx->timestep,
                    (double)(finished_throttling - start_iter) / 1000.0);
            fprintf(profiling_fp, "  start_time_step %f ms\n", 
                    (double)(finished_start - start_iter) / 1000.0);
            fprintf(profiling_fp, "  metadata %f ms (%f %f %f)\n",
                    (double)(finished_updates - finished_start) / 1000.0,
                    (double)ctx->cache_perf_info.fetch_neighbors_time / 1000.0,
                    (double)ctx->cache_perf_info.quiet_neighbors_time / 1000.0,
                    (double)ctx->cache_perf_info.update_metadata_time / 1000.0);
            fprintf(profiling_fp, "  summary %f ms (%f %f %f)\n",
                    (double)(finished_summary_update - finished_updates) / 1000.0,
                    (double)(finished_actor_partitions - finished_updates) / 1000.0,
                    (double)(finished_time_window - finished_actor_partitions) / 1000.0,
                    (double)(finished_summary_update - finished_time_window) / 1000.0);
            fprintf(profiling_fp, "  edges %f ms (update=%f getmem=%f # edge "
                    "checks=%llu # part checks=%llu # dist measures=%llu # "
                    "(cached|uncached) remote fetches=(%u|%u))\n",
                    (double)(finished_edge_adds - finished_summary_update) / 1000.0,
                    (double)update_edge_time / 1000.0,
                    (double)getmem_time / 1000.0,
                    n_edge_checks,
                    partition_checks,
                    n_distance_measures,
                    n_cached_remote_fetches_for_update_edges,
                    n_uncached_remote_fetches_for_update_edges);
            fprintf(profiling_fp, "  neighbor updates %f ms\n",
                    (double)(finished_neighbor_updates - finished_edge_adds) / 1000.0);
            fprintf(profiling_fp, "  coupled values %f ms\n",
                    (double)(finished_coupled_values - finished_neighbor_updates) / 1000.0);
            fprintf(profiling_fp, "  coupling %f ms (%u)\n",
                    (double)(finished_coupling - finished_coupled_values) / 1000.0,
                    n_coupled_spins);
            fprintf(profiling_fp, "  throttling %f ms (%u spins)\n",
                    (double)(finished_throttling - finished_coupling) / 1000.0,
                    nspins);
            fprintf(profiling_fp, "  %u / %u PE neighbors %s\n",
                    hvr_set_count(ctx->my_neighbors), ctx->npes,
                    neighbors_str);
            fprintf(profiling_fp, "  partition window = %s, %d / %d partitions "
                    "active for %u local vertices\n", partition_time_window_str,
                    hvr_set_count(ctx->partition_time_window),
                    ctx->n_partitions, n_local_verts);
            fprintf(profiling_fp, "  aborting? %d - last step? %d - remote "
                    "cache hits=%u misses=%u, feature cache hits=%u misses=%u "
                    "quiets=%llu, avg # edges=%f\n",
                    should_abort,
                    ctx->timestep >= ctx->max_timestep,
                    ctx->vec_cache.nhits,
                    ctx->vec_cache.nmisses,
                    ctx->n_vector_cache_hits,
                    ctx->n_vector_cache_misses,
                    ctx->cache_perf_info.quiet_counter,
                    avg_n_neighbors);
            fflush(profiling_fp);
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
    finalize_actor_for_timestep(ctx->coupled_pes_values + ctx->pe,
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

    this_pe_has_exited = 1;

    hvr_exec_info info;
    info.executed_timesteps = ctx->timestep;
    return info;
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
