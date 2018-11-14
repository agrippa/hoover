#include <string.h>
#include <stdio.h>
#include <shmem.h>

#include "hvr_vertex_cache.h"

void hvr_vertex_cache_init(hvr_vertex_cache_t *cache,
        hvr_partition_t npartitions) {
    memset(cache, 0x00, sizeof(*cache));

    hvr_map_init(&cache->cache_map, 1);

    cache->partitions = (hvr_vertex_cache_node_t **)malloc(
            npartitions * sizeof(hvr_vertex_cache_node_t *));
    assert(cache->partitions);
    memset(cache->partitions, 0x00,
            npartitions * sizeof(hvr_vertex_cache_node_t *));
    cache->npartitions = npartitions;

    unsigned n_preallocs = 1024;
    if (getenv("HVR_VEC_CACHE_PREALLOCS")) {
        n_preallocs = atoi(getenv("HVR_VEC_CACHE_PREALLOCS"));
    }

    hvr_vertex_cache_node_t *prealloc =
        (hvr_vertex_cache_node_t *)malloc(n_preallocs * sizeof(*prealloc));
    assert(prealloc);
    memset(prealloc, 0x00, n_preallocs * sizeof(*prealloc));

    prealloc[0].bucket_next = prealloc + 1;
    prealloc[0].bucket_prev = NULL;
    prealloc[n_preallocs - 1].bucket_next = NULL;
    prealloc[n_preallocs - 1].bucket_prev = prealloc + (n_preallocs - 2);
    for (unsigned i = 1; i < n_preallocs - 1; i++) {
        prealloc[i].bucket_next = prealloc + (i + 1);
        prealloc[i].bucket_prev = prealloc + (i - 1);
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
    static hvr_map_val_t *nodes = NULL;
    if (nodes == NULL) {
        nodes = (hvr_map_val_t *)malloc(sizeof(*nodes));
        assert(nodes);
    }
    unsigned capacity = 1;
    int n = hvr_map_linearize(vert, &nodes, &capacity, &cache->cache_map);
    assert(n == -1 || n == 1);

    if (n == -1) {
        cache->cache_perf_info.nmisses++;
        return NULL;
    } else {
        cache->cache_perf_info.nhits++;
        return nodes[0].cached_vert;
    }
}

static void linked_list_remove_helper(hvr_vertex_cache_node_t *to_remove,
        hvr_vertex_cache_node_t *prev, hvr_vertex_cache_node_t *next,
        hvr_vertex_cache_node_t **prev_next,
        hvr_vertex_cache_node_t **next_prev,
        hvr_vertex_cache_node_t **head) {
    if (prev == NULL && next == NULL) {
        // Only element in the list
        assert(*head == to_remove);
        *head = NULL;
    } else if (prev == NULL) {
        // Only next is non-null, first element in list
        assert(*head == to_remove);
        *next_prev = NULL;
        *head = next;
    } else if (next == NULL) {
        // Only prev is non-null, last element in list
        *prev_next = NULL;
    } else {
        assert(prev && next);
        assert(*prev_next == *next_prev && *prev_next == to_remove &&
            *next_prev == to_remove);
        *prev_next = next;
        *next_prev = prev;
    }
}

void hvr_vertex_cache_delete(hvr_vertex_t *vert, hvr_vertex_cache_t *cache) {
    hvr_vertex_cache_node_t *node = hvr_vertex_cache_lookup(vert->id, cache);
    assert(node);

    hvr_map_val_t to_remove = {.cached_vert = node};
    hvr_map_remove(vert->id, to_remove, CACHED_VERT_INFO, &cache->cache_map);

    // Remove from buckets list
    unsigned bucket = CACHE_BUCKET(vert->id);
    linked_list_remove_helper(node, node->bucket_prev, node->bucket_next,
            node->bucket_prev ? &(node->bucket_prev->bucket_next) : NULL,
            node->bucket_next ? &(node->bucket_next->bucket_prev) : NULL,
            &(cache->buckets[bucket]));

    // Remove from partitions list
    linked_list_remove_helper(node, node->part_prev, node->part_next,
            node->part_prev ? &(node->part_prev->part_next) : NULL,
            node->part_next ? &(node->part_next->part_prev) : NULL,
            &(cache->partitions[node->part]));

    // Insert into pool using bucket pointers
    if (cache->pool_head) {
        cache->pool_head->bucket_prev = node;
    }
    node->bucket_next = cache->pool_head;
    node->bucket_prev = NULL;
    cache->pool_head = node;
}

hvr_vertex_cache_node_t *hvr_vertex_cache_add(hvr_vertex_t *vert,
        hvr_partition_t part, unsigned min_dist_from_local_vertex,
        hvr_vertex_cache_t *cache) {
    // Assume that vec is not already in the cache, but don't enforce this
    hvr_vertex_cache_node_t *new_node = NULL;
    if (cache->pool_head) {
        // Look for an already free node
        new_node = cache->pool_head;
        cache->pool_head = new_node->bucket_next;
        if (cache->pool_head) {
            cache->pool_head->bucket_prev = NULL;
        }
    } else {
        // No valid node found, print an error
        fprintf(stderr, "ERROR: PE %d exhausted %u cache slots\n",
                shmem_my_pe(), cache->pool_size);
        abort();
    }

    const unsigned bucket = CACHE_BUCKET(vert->id);
    memcpy(&new_node->vert, vert, sizeof(*vert));
    new_node->min_dist_from_local_vertex = min_dist_from_local_vertex;
    new_node->part = part;

    // Insert into the appropriate bucket
    if (cache->buckets[bucket]) {
        cache->buckets[bucket]->bucket_prev = new_node;
    }
    new_node->bucket_next = cache->buckets[bucket];
    new_node->bucket_prev = NULL;
    cache->buckets[bucket] = new_node;

    // Insert into the appropriate partition list
    if (cache->partitions[part]) {
        cache->partitions[part]->part_prev = new_node;
    }
    new_node->part_next = cache->partitions[part];
    new_node->part_prev = NULL;
    cache->partitions[part] = new_node;

    hvr_map_val_t to_insert = {.cached_vert = new_node};
    hvr_map_add(vert->id, to_insert, CACHED_VERT_INFO, &cache->cache_map);

    return new_node;
}

void hvr_vertex_cache_metrics(hvr_vertex_cache_t *cache, size_t *min_bucket_len,
        size_t *max_bucket_len) {
    for (unsigned i = 0; i < HVR_CACHE_BUCKETS; i++) {
        size_t count = 0;
        hvr_vertex_cache_node_t *iter = cache->buckets[i];
        while (iter) {
            iter = iter->bucket_next;
            count++;
        }

        if (i == 0) {
            *min_bucket_len = count;
            *max_bucket_len = count;
        } else {
            if (count < *min_bucket_len) *min_bucket_len = count;
            if (count > *max_bucket_len) *max_bucket_len = count;
        }
    }
}
