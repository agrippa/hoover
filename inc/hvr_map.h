#ifndef _HVR_MAP_H
#define _HVR_MAP_H

#include "hvr_common.h"
#include "hvr_vertex_pool.h"
#include "dlmalloc.h"

#define HVR_MAP_SEG_SIZE 128
#define HVR_MAP_BUCKETS 8192

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
} hvr_map_t;

extern void hvr_map_init(hvr_map_t *m, unsigned n_segs);

extern void hvr_map_destroy(hvr_map_t *m);

extern void hvr_map_add(hvr_vertex_id_t key, void *to_insert,
        hvr_map_t *m);

extern void hvr_map_remove(hvr_vertex_id_t key, void *val,
        hvr_map_t *m);

extern void *hvr_map_get(hvr_vertex_id_t key, hvr_map_t *m);

extern void hvr_map_clear(hvr_map_t *m);

extern void hvr_map_size_in_bytes(hvr_map_t *m, size_t *capacity, size_t *used,
        size_t bytes_per_value);

#endif
