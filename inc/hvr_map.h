#ifndef _HVR_MAP_H
#define _HVR_MAP_H

#include "hvr_common.h"

#define HVR_MAP_SEG_SIZE 16384
#define HVR_MAP_BUCKETS 2048

typedef union _hvr_map_val_t {
    hvr_edge_info_t edge_info;
    void *cached_vert;
} hvr_map_val_t;

typedef struct _hvr_map_seg_t {
    hvr_vertex_id_t keys[HVR_MAP_SEG_SIZE];
    hvr_map_val_t *vals[HVR_MAP_SEG_SIZE];

    // Allocated capacity of each key's values array
    unsigned capacity[HVR_MAP_SEG_SIZE];
    // Actual length of each values array
    unsigned length[HVR_MAP_SEG_SIZE];
    // Number of keys in this map segment
    unsigned nkeys;

    // Next map segment in this bucket
    struct _hvr_map_seg_t *next;
} hvr_map_seg_t;

typedef struct _hvr_map_t {
    hvr_map_seg_t *buckets[HVR_MAP_BUCKETS];
    unsigned init_val_capacity;
} hvr_map_t;

extern void hvr_map_init(hvr_map_t *m, unsigned init_val_capacity);

extern void hvr_map_add(hvr_vertex_id_t key, hvr_map_val_t to_insert,
        int is_edge_info, hvr_map_t *m);

extern void hvr_map_remove(hvr_vertex_id_t key, hvr_map_val_t val,
        int is_edge_info, hvr_map_t *m);

extern hvr_edge_type_t hvr_map_contains(hvr_vertex_id_t key,
        hvr_vertex_id_t val, hvr_map_t *m);

extern unsigned hvr_map_linearize(hvr_vertex_id_t key,
        hvr_map_val_t **vals, unsigned *capacity, hvr_map_t *m);

extern void hvr_map_clear(hvr_map_t *m);

extern size_t hvr_map_count_values(hvr_vertex_id_t key, hvr_map_t *m);

#endif
