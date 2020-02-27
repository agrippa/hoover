#ifndef _HVR_MAP_H
#define _HVR_MAP_H

#include "hvr_common.h"
#include "dlmalloc.h"
#include "sparsehash/sparse_hash_map"

using google::sparse_hash_map;

#define HVR_MAP_SEG_SIZE 128
#define HVR_MAP_BUCKETS 16384

typedef struct _hvr_map_val_t {
    hvr_vertex_id_t key;
    void *data;
} hvr_map_val_t;

typedef struct _hvr_map_seg_t {
    hvr_map_val_t data[HVR_MAP_SEG_SIZE];

    // Number of keys in this map segment
    unsigned nkeys;

    // Next map segment in this bucket
    struct _hvr_map_seg_t *next;
} hvr_map_seg_t;

typedef struct _hvr_map_t {
    hvr_map_seg_t *buckets[HVR_MAP_BUCKETS];
    hvr_map_seg_t *bucket_tails[HVR_MAP_BUCKETS];

    hvr_map_seg_t *seg_pool;
    hvr_map_seg_t *prealloc_seg_pool;
    unsigned n_prealloc;

    const char *seg_env_var;
} hvr_map_t;

extern void hvr_map_init(hvr_map_t *m, unsigned n_segs,
        const char *seg_env_var);

extern void hvr_map_destroy(hvr_map_t *m);

/*
 * Add a new key-value pair mapping from key -> to_insert to the map. If
 * replace == 1, then the new value will be inserted whether there is an
 * existing value for that key or not. If replace == 0 and there is an existing
 * value, we assert that the new and existing values are the same.
 */
extern void hvr_map_add(hvr_vertex_id_t key, void *to_insert,
        int replace, hvr_map_t *m);

/*
 * Remove the key-value pair for key from the map, asserting that the current
 * value is val.
 */
extern void hvr_map_remove(hvr_vertex_id_t key, void *val,
        hvr_map_t *m);

/*
 * Get the current value for key from the map, returning NULL if none is found.
 * Note that this assumes we never add a mapping to NULL to any map.
 */
extern void *hvr_map_get(hvr_vertex_id_t key, hvr_map_t *m);

extern void hvr_map_clear(hvr_map_t *m);

extern void hvr_map_size_in_bytes(hvr_map_t *m, size_t *capacity, size_t *used,
        size_t bytes_per_value);

#endif
