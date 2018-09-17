#include <string.h>
#include <stdio.h>
#include <shmem.h>

#include "hvr_vertex_cache.h"

void hvr_vertex_cache_init(hvr_vertex_cache_t *cache) {
    memset(cache, 0x00, sizeof(*cache));

    unsigned n_preallocs = 1024;
    if (getenv("HVR_VEC_CACHE_PREALLOCS")) {
        n_preallocs = atoi(getenv("HVR_VEC_CACHE_PREALLOCS"));
    }

    hvr_vertex_cache_node_t *prealloc =
        (hvr_vertex_cache_node_t *)malloc(n_preallocs * sizeof(*prealloc));
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

/*
 * Given a vertex ID on a remote PE, look up that vertex in our local cache.
 * Only returns an entry if the newest timestep stored in that entry is new
 * enough given target_timestep. If no matching entry is found, returns NULL.
 *
 * May lead to evictions of very old cache entries that we now consider unusable
 * because of their age, as judged by CACHED_TIMESTEPS_TOLERANCE.
 */
hvr_vertex_cache_node_t *hvr_vertex_cache_lookup(hvr_vertex_id_t vert,
        hvr_vertex_cache_t *cache) {
    const unsigned bucket = CACHE_BUCKET(vert);

    hvr_vertex_cache_node_t *iter = cache->buckets[bucket];
    while (iter) {
        if (iter->vert.id == vert) break;
        iter = iter->next;
    }

    if (iter == NULL) {
        cache->cache_perf_info.nmisses++;
    } else {
        cache->cache_perf_info.nhits++;
    }
    return iter;
}

// static void remove_node_from_cache(hvr_sparse_vec_cache_node_t *iter,
//         hvr_sparse_vec_cache_t *cache) {
//     const unsigned bucket = CACHE_BUCKET(iter->vert);
//     assert(iter->pending_comm == 0);
// 
//     // Need to fix the prev and next elements in this
//     if (iter->prev == NULL && iter->next == NULL) {
//         assert(cache->buckets[bucket] == iter);
//         cache->buckets[bucket] = NULL;
//     } else if (iter->next == NULL) {
//         iter->prev->next = NULL;
//     } else if (iter->prev == NULL) {
//         cache->buckets[bucket] = iter->next;
//         iter->next->prev = NULL;
//     } else {
//         iter->prev->next = iter->next;
//         iter->next->prev = iter->prev;
//     }
// 
//     // Then remove from the LRU list
//     if (iter->lru_prev == NULL && iter->lru_next == NULL) {
//         assert(cache->lru_head == iter && cache->lru_tail == iter);
//         cache->lru_head = NULL;
//         cache->lru_tail = NULL;
//     } else if (iter->lru_next == NULL) {
//         assert(cache->lru_tail == iter);
//         cache->lru_tail = iter->lru_prev;
//         iter->lru_prev->lru_next = NULL;
//     } else if (iter->lru_prev == NULL) {
//         assert(cache->lru_head == iter);
//         cache->lru_head = iter->lru_next;
//         iter->lru_next->lru_prev = NULL;
//     } else {
//         iter->lru_prev->lru_next = iter->lru_next;
//         iter->lru_next->lru_prev = iter->lru_prev;
//     }
// }

hvr_vertex_cache_node_t *hvr_vertex_cache_add(hvr_vertex_t *vert,
        unsigned min_dist_from_local_vertex, hvr_vertex_cache_t *cache) {
    // Assume that vec is not already in the cache, but don't enforce this
    hvr_vertex_cache_node_t *new_node = NULL;
    if (cache->pool_head) {
        // Look for an already free node
        new_node = cache->pool_head;
        cache->pool_head = new_node->next;
        if (cache->pool_head) {
            cache->pool_head->prev = NULL;
        }
    // } else if (cache->lru_tail && !cache->lru_tail->pending_comm) { 
    //     /*
    //      * Find the oldest requested node, check it isn't pending communication,
    //      * and if so use it.
    //      */
    //     new_node = cache->lru_tail;

    //     // Removes node from bucket and LRU lists
    //     remove_node_from_cache(new_node, cache);
    } else {
        // No valid node found, print an error
        fprintf(stderr, "ERROR: PE %d exhausted %u cache slots\n",
                shmem_my_pe(), cache->pool_size);
        abort();
    }

    const unsigned bucket = CACHE_BUCKET(vert->id);
    memcpy(&new_node->vert, vert, sizeof(*vert));
    new_node->min_dist_from_local_vertex = min_dist_from_local_vertex;

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
