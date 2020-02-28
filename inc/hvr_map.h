#ifndef _HVR_MAP_H
#define _HVR_MAP_H

#include "hvr_common.h"
#ifdef __cplusplus
extern "C" {
#endif
#include "dlmalloc.h"
#ifdef __cplusplus
}
#endif

#define HVR_MAP_SEG_SIZE 128
#define HVR_MAP_BUCKETS 16384
#define HVR_MAP_BUCKET(my_key) ((my_key) % HVR_MAP_BUCKETS)

typedef struct _hvr_map_seg_t {
    hvr_vertex_id_t data_key[HVR_MAP_SEG_SIZE];
    void *data_data[HVR_MAP_SEG_SIZE];

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

void hvr_map_init(hvr_map_t *m, unsigned n_segs,
        const char *seg_env_var);

void hvr_map_destroy(hvr_map_t *m);

/*
 * Add a new key-value pair mapping from key -> to_insert to the map. If
 * replace == 1, then the new value will be inserted whether there is an
 * existing value for that key or not. If replace == 0 and there is an existing
 * value, we assert that the new and existing values are the same.
 */
void hvr_map_add(hvr_vertex_id_t key, void *to_insert,
        int replace, hvr_map_t *m);

/*
 * Remove the key-value pair for key from the map, asserting that the current
 * value is val.
 */
void hvr_map_remove(hvr_vertex_id_t key, void *val,
        hvr_map_t *m);

/*
 * Get the current value for key from the map, returning NULL if none is found.
 * Note that this assumes we never add a mapping to NULL to any map.
 */
// void *hvr_map_get(hvr_vertex_id_t key, hvr_map_t *m);

static inline int binarySearch(const hvr_vertex_id_t *arr,
        const hvr_vertex_id_t x) 
{
    int l = 0;
    int r = HVR_MAP_SEG_SIZE - 1;
    while (r >= l) {
        const int mid = l + (r - l)/2; 

        // If the element is present at the middle  
        // itself 
        if (arr[mid] == x) {
            return mid;
        } else if (arr[mid] > x) {
            r = mid - 1;
        } else {
            l = mid + 1;
        }
    } 

    // We reach here when element is not  
    // present in array 
    return -1; 
}

static inline int hvr_map_find(hvr_vertex_id_t key, hvr_map_t *m,
        hvr_map_seg_t **out_seg, unsigned *out_index) {
    unsigned bucket = HVR_MAP_BUCKET(key);

    hvr_map_seg_t *seg = m->buckets[bucket];
    while (seg) {
        const unsigned nkeys = seg->nkeys;
        if (nkeys == HVR_MAP_SEG_SIZE) {
            int index = binarySearch(seg->data_key, key);
            if (index >= 0) {
                *out_seg = seg;
                *out_index = index;
                return 1;
            }
        } else {
            for (unsigned i = 0; i < nkeys; i++) {
                if (seg->data_key[i] == key) {
                    *out_seg = seg;
                    *out_index = i;
                    return 1;
                }
            }
        }
        seg = seg->next;
    }
    return 0;
}

static inline void *hvr_map_get(hvr_vertex_id_t key, hvr_map_t *m) {
    hvr_map_seg_t *seg;
    unsigned seg_index;

    const int success = hvr_map_find(key, m, &seg, &seg_index);

    if (success) {
        return seg->data_data[seg_index];
    } else {
        return NULL;
    }
}

void hvr_map_clear(hvr_map_t *m);

void hvr_map_size_in_bytes(hvr_map_t *m, size_t *capacity, size_t *used,
        size_t bytes_per_value);

#endif
