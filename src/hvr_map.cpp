#include "hvr_map.h"

#include <stdio.h>
#include <string.h>

static inline void swap(hvr_vertex_id_t *a, hvr_vertex_id_t *b, void **co_a, void **co_b) {
    hvr_vertex_id_t tmp = *a;
    *a = *b;
    *b = tmp;

    void *tmp2 = *co_a;
    *co_a = *co_b;
    *co_b = tmp2;
}

/* This function takes last element as pivot, places 
 *    the pivot element at its correct position in sorted 
 *        array, and places all smaller (smaller than pivot) 
 *           to left of pivot and all greater elements to right 
 *              of pivot */
static int partition (hvr_vertex_id_t *arr, void **co_arr, int low, int high) 
{
    hvr_vertex_id_t pivot = arr[high];    // pivot 
    int i = (low - 1);  // Index of smaller element 

    for (int j = low; j <= high- 1; j++) 
    { 
        // If current element is smaller than the pivot 
        if (arr[j] < pivot) 
        { 
            i++;    // increment index of smaller element 
            swap(&arr[i], &arr[j], &co_arr[i], &co_arr[j]); 
        } 
    } 
    swap(&arr[i + 1], &arr[high], &co_arr[i + 1], &co_arr[high]); 
    return (i + 1); 
} 

/* The main function that implements QuickSort 
   arr[] --> Array to be sorted, 
   low  --> Starting index, 
   high  --> Ending index */
static void quickSort(hvr_vertex_id_t *arr, void **co_arr, int low, int high) 
{ 
    if (low < high) 
    { 
        /* pi is partitioning index, arr[p] is now 
           at right place */
        int pi = partition(arr, co_arr, low, high); 

        // Separately sort elements before 
        // partition and after partition 
        quickSort(arr, co_arr, low, pi - 1); 
        quickSort(arr, co_arr, pi + 1, high); 
    } 
}

// Add a new key with one initial value
static void hvr_map_seg_add(hvr_vertex_id_t key, void *data,
        hvr_map_seg_t *s) {
    const unsigned insert_index = s->nkeys;
    assert(insert_index < HVR_MAP_SEG_SIZE);
    s->data_key[insert_index] = key;
    s->data_data[insert_index] = data;

    s->nkeys++;

    if (s->nkeys == HVR_MAP_SEG_SIZE) {
        quickSort(&(s->data_key[0]), &(s->data_data[0]), 0, HVR_MAP_SEG_SIZE - 1);
    }
}

static inline void hvr_map_seg_init(hvr_map_seg_t *seg) {
    seg->nkeys = 0;
    seg->next = NULL;
}

void hvr_map_init(hvr_map_t *m, unsigned n_segs, const char *seg_env_var) {
    memset(m, 0x00, sizeof(*m));

    hvr_map_seg_t *prealloc = (hvr_map_seg_t *)malloc_helper(
            n_segs * sizeof(*prealloc));
    assert(prealloc);
    for (unsigned i = 0; i < n_segs - 1; i++) {
        prealloc[i].next = prealloc + (i + 1);
    }
    prealloc[n_segs - 1].next = NULL;
    m->seg_pool = prealloc;
    m->prealloc_seg_pool = prealloc;
    m->n_prealloc = n_segs;
    m->seg_env_var = seg_env_var;
}

void hvr_map_destroy(hvr_map_t *m) {
    free(m->prealloc_seg_pool);
}

void hvr_map_add(hvr_vertex_id_t key, void *to_insert, int replace,
        hvr_map_t *m) {
    hvr_map_seg_t *seg;
    unsigned seg_index;
    int success = hvr_map_find(key, m, &seg, &seg_index);

    if (success) {
        // Key already exists
        if (replace) {
            seg->data_data[seg_index] = to_insert;
        } else {
            assert(seg->data_data[seg_index] == to_insert);
        }
    } else {
        const unsigned bucket = HVR_MAP_BUCKET(key);

        // Have to add as new key
        if (m->buckets[bucket] == NULL) {
            // First segment created
            hvr_map_seg_t *new_seg = m->seg_pool;
            if (!new_seg) {
                fprintf(stderr, "ERROR> Ran out of map segments (%u "
                        "pre-allocated) Consider increasing %s.\n",
                        m->n_prealloc, m->seg_env_var);
                abort();
            }
            m->seg_pool = new_seg->next;

            hvr_map_seg_init(new_seg);

            hvr_map_seg_add(key, to_insert, new_seg);
            assert(m->buckets[bucket] == NULL);
            m->buckets[bucket] = new_seg;
            m->bucket_tails[bucket] = new_seg;
        } else {
            hvr_map_seg_t *last_seg_in_bucket = m->bucket_tails[bucket];

            if (last_seg_in_bucket->nkeys == HVR_MAP_SEG_SIZE) {
                // Have to append new segment
                hvr_map_seg_t *new_seg = m->seg_pool;
                if (!new_seg) {
                    fprintf(stderr, "ERROR> Ran out of map segments (%u "
                            "pre-allocated) Consider increasing %s.\n",
                            m->n_prealloc, m->seg_env_var);
                    abort();
                }
                m->seg_pool = new_seg->next;
                hvr_map_seg_init(new_seg);

                hvr_map_seg_add(key, to_insert, new_seg);
                assert(last_seg_in_bucket->next == NULL);
                last_seg_in_bucket->next = new_seg;
                m->bucket_tails[bucket] = new_seg;
            } else {
                // Insert in existing segment
                hvr_map_seg_add(key, to_insert, last_seg_in_bucket);
            }
        }
    }
}

void hvr_map_remove(hvr_vertex_id_t key, void *val, hvr_map_t *m) {
    hvr_map_seg_t *seg;
    unsigned seg_index;

    const int success = hvr_map_find(key, m, &seg, &seg_index);

    if (success) {
        assert(seg->data_data[seg_index] == val);

        unsigned copy_to = seg_index;
        unsigned copy_from = seg->nkeys - 1;

        seg->data_key[copy_to] = seg->data_key[copy_from];
        seg->data_data[copy_to] = seg->data_data[copy_from];
        seg->nkeys -= 1;
    }
}

void hvr_map_clear(hvr_map_t *m) {
    for (unsigned i = 0; i < HVR_MAP_BUCKETS; i++) {
        hvr_map_seg_t *seg = m->buckets[i];
        while (seg) {
            hvr_map_seg_t *next = seg->next;
            free(seg);
            seg = next;
        }
        m->buckets[i] = NULL;
        m->bucket_tails[i] = NULL;
    }
}

void hvr_map_size_in_bytes(hvr_map_t *m, size_t *out_capacity,
        size_t *out_used, size_t bytes_per_value) {
    size_t allocated = sizeof(*m) + m->n_prealloc * sizeof(hvr_map_seg_t);

    size_t used = sizeof(*m);
    for (unsigned b = 0; b < HVR_MAP_BUCKETS; b++) {
        hvr_map_seg_t *bucket = m->buckets[b];
        while (bucket) {
            unsigned n_unused_keys = HVR_MAP_SEG_SIZE - bucket->nkeys;
            used += sizeof(hvr_map_seg_t) - (n_unused_keys * sizeof(hvr_vertex_id_t) + sizeof(void*));

            used += bucket->nkeys * bytes_per_value;
            allocated += bucket->nkeys * bytes_per_value;

            bucket = bucket->next;
        }
    }

    *out_capacity = allocated;
    *out_used = used;
}
