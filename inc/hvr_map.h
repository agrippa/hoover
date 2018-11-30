#ifndef _HVR_MAP_H
#define _HVR_MAP_H

#include "hvr_common.h"

#define HVR_MAP_SEG_SIZE 32768
#define HVR_MAP_BUCKETS 2048
#define HVR_MAP_N_INLINE_VALS 1

typedef enum {
    INTERACT_INFO,
    EDGE_INFO,
    CACHED_VERT_INFO
} hvr_map_type_t;

typedef union _hvr_map_val_t {
    hvr_edge_info_t edge_info;
    void *cached_vert;
    hvr_partition_t interact;
} hvr_map_val_t;

typedef struct _hvr_map_val_list_t {
    hvr_map_val_t *inline_vals;
    hvr_map_val_t *ext_vals;
    int nvals;
} hvr_map_val_list_t;

static inline int hvr_map_val_list_nvals(hvr_map_val_list_t *l) {
    return l->nvals;
}

static inline hvr_map_val_t hvr_map_val_list_get(const unsigned i,
        hvr_map_val_list_t *l) {
    if (i < HVR_MAP_N_INLINE_VALS) {
        return (l->inline_vals)[i];
    } else {
        return (l->ext_vals)[i - HVR_MAP_N_INLINE_VALS];
    }
}

typedef struct _hvr_map_seg_t {
    hvr_vertex_id_t keys[HVR_MAP_SEG_SIZE];
    hvr_map_val_t inline_vals[HVR_MAP_SEG_SIZE][HVR_MAP_N_INLINE_VALS];
    hvr_map_val_t *ext_vals[HVR_MAP_SEG_SIZE];

    // Allocated capacity of each key's values array
    unsigned ext_capacity[HVR_MAP_SEG_SIZE];
    // Actual length of each values array
    unsigned length[HVR_MAP_SEG_SIZE];
    // Number of keys in this map segment
    unsigned nkeys;

    // Next map segment in this bucket
    struct _hvr_map_seg_t *next;
} hvr_map_seg_t;

typedef struct _hvr_map_t {
    hvr_map_seg_t *buckets[HVR_MAP_BUCKETS];
    hvr_map_seg_t *bucket_tails[HVR_MAP_BUCKETS];
    hvr_map_type_t type;
    unsigned init_val_capacity;
} hvr_map_t;

extern void hvr_map_init(hvr_map_t *m, unsigned init_val_capacity,
        hvr_map_type_t type);

extern void hvr_map_add(hvr_vertex_id_t key, hvr_map_val_t to_insert,
        hvr_map_t *m);

extern void hvr_map_remove(hvr_vertex_id_t key, hvr_map_val_t val,
        hvr_map_t *m);

extern hvr_edge_type_t hvr_map_contains(hvr_vertex_id_t key,
        hvr_vertex_id_t val, hvr_map_t *m);

extern int hvr_map_linearize(hvr_vertex_id_t key,
        hvr_map_val_t **vals, unsigned *capacity, hvr_map_t *m);

extern void hvr_map_clear(hvr_map_t *m);

extern size_t hvr_map_count_values(hvr_vertex_id_t key, hvr_map_t *m);

extern void hvr_map_size_in_bytes(hvr_map_t *m, size_t *capacity, size_t *used,
        double *avg_val_capacity, double *avg_val_length);

#endif
