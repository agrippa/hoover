#ifndef _HVR_MAP_H
#define _HVR_MAP_H

#include "hvr_common.h"

#define HVR_MAP_SEG_SIZE 2048
#define HVR_MAP_BUCKETS 2048

typedef struct _hvr_map_seg_t {
    hvr_vertex_id_t keys[HVR_MAP_SEG_SIZE];
    hvr_vertex_id_t *values[HVR_MAP_SEG_SIZE];
    hvr_edge_type_t *edge_types[HVR_MAP_SEG_SIZE];

    unsigned capacity[HVR_MAP_SEG_SIZE];
    unsigned length[HVR_MAP_SEG_SIZE];
    unsigned nkeys;

    struct _hvr_map_seg_t *next;
} hvr_map_seg_t;

typedef struct _hvr_map_t {
    hvr_map_seg_t *buckets[HVR_MAP_BUCKETS];
} hvr_map_t;

extern void hvr_map_init(hvr_map_t *m);

extern void hvr_map_add(hvr_vertex_id_t key, hvr_vertex_id_t val,
        hvr_edge_type_t edge_type, hvr_map_t *m);

extern void hvr_map_remove(hvr_vertex_id_t key, hvr_vertex_id_t val,
        hvr_map_t *m);

extern hvr_edge_type_t hvr_map_contains(hvr_vertex_id_t key,
        hvr_vertex_id_t val, hvr_map_t *m);

extern unsigned hvr_map_linearize(hvr_vertex_id_t key,
        hvr_vertex_id_t **vertices, hvr_edge_type_t **directions,
        unsigned *capacity, hvr_map_t *m);

extern void hvr_map_clear(hvr_map_t *m);

extern size_t hvr_map_count_values(hvr_vertex_id_t key, hvr_map_t *m);

#endif
