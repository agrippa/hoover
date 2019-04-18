#include "hvr_map.h"

#include <stdio.h>
#include <string.h>

#define HVR_MAP_BUCKET(my_key) ((my_key) % HVR_MAP_BUCKETS)
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static int comp(const void *_a, const void *_b) {
    hvr_map_val_t *a = (hvr_map_val_t *)_a;
    hvr_map_val_t *b = (hvr_map_val_t *)_b;

    if (a->key < b->key) return -1;
    else if (a->key > b->key) return 1;
    else return 0;
}

// Add a new key with one initial value
static void hvr_map_seg_add(hvr_vertex_id_t key, void *cached_vert,
        hvr_map_seg_t *s, unsigned init_val_capacity) {
    const unsigned insert_index = s->nkeys;
    assert(insert_index < HVR_MAP_SEG_SIZE);
    s->data[insert_index].key = key;
    s->data[insert_index].cached_vert = cached_vert;

    s->nkeys++;

    if (s->nkeys == HVR_MAP_SEG_SIZE) {
        qsort(&(s->data[0]), HVR_MAP_SEG_SIZE, sizeof(s->data[0]), comp);
    }
}

static inline int binarySearch(const hvr_map_val_t *arr,
        const hvr_vertex_id_t x) 
{
    int l = 0;
    int r = HVR_MAP_SEG_SIZE - 1;
    while (r >= l) {
        const int mid = l + (r - l)/2; 

        // If the element is present at the middle  
        // itself 
        if (arr[mid].key == x) {
            return mid;
        } else if (arr[mid].key > x) {
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
            int index = binarySearch(seg->data, key);
            if (index >= 0) {
                *out_seg = seg;
                *out_index = index;
                return 1;
            }
        } else {
            for (unsigned i = 0; i < nkeys; i++) {
                if (seg->data[i].key == key) {
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

void hvr_map_init(hvr_map_t *m, unsigned n_segs, size_t vals_pool_size,
        unsigned vals_pool_nodes, unsigned init_val_capacity) {
    memset(m, 0x00, sizeof(*m));
    m->init_val_capacity = init_val_capacity;

    hvr_map_seg_t *prealloc = (hvr_map_seg_t *)malloc(
            n_segs * sizeof(*prealloc));
    assert(prealloc);
    for (unsigned i = 0; i < n_segs - 1; i++) {
        prealloc[i].next = prealloc + (i + 1);
    }
    prealloc[n_segs - 1].next = NULL;
    m->seg_pool = prealloc;
    m->prealloc_seg_pool = prealloc;
    m->n_prealloc = n_segs;

    /*
     * vals_pool_size may be zero if we know we will never need dynamically
     * allocated values from this map (i.e. all entries are guaranteed to have a
     * small enough number of values that they fit in the statically allocated
     * portion of the value space.
     */
    m->val_pool = malloc(vals_pool_size * sizeof(hvr_map_val_t));
    assert(m->val_pool || vals_pool_size == 0);
    if (vals_pool_size > 0) {
        m->tracker = create_mspace_with_base(m->val_pool,
                vals_pool_size * sizeof(hvr_map_val_t), 0);
        assert(m->tracker);
    } else {
        m->tracker = 0;
    }
}

void hvr_map_destroy(hvr_map_t *m) {
    free(m->prealloc_seg_pool);
    if (m->tracker) {
        destroy_mspace(m->tracker);
    }
    free(m->val_pool);
}

void hvr_map_add(hvr_vertex_id_t key, void *to_insert, hvr_map_t *m) {
    hvr_map_seg_t *seg;
    unsigned seg_index;
    int success = hvr_map_find(key, m, &seg, &seg_index);

    if (success) {
        // Key already exists
        assert(seg->data[seg_index].cached_vert == to_insert);
    } else {
        const unsigned bucket = HVR_MAP_BUCKET(key);

        // Have to add as new key
        if (m->buckets[bucket] == NULL) {
            // First segment created
            hvr_map_seg_t *new_seg = m->seg_pool;
            assert(new_seg);
            m->seg_pool = new_seg->next;
            memset(new_seg, 0x00, sizeof(*new_seg));

            hvr_map_seg_add(key, to_insert, new_seg, m->init_val_capacity);
            assert(m->buckets[bucket] == NULL);
            m->buckets[bucket] = new_seg;
            m->bucket_tails[bucket] = new_seg;
        } else {
            hvr_map_seg_t *last_seg_in_bucket= m->bucket_tails[bucket];

            if (last_seg_in_bucket->nkeys == HVR_MAP_SEG_SIZE) {
                // Have to append new segment
                hvr_map_seg_t *new_seg = m->seg_pool;
                assert(new_seg);
                m->seg_pool = new_seg->next;
                memset(new_seg, 0x00, sizeof(*new_seg));

                hvr_map_seg_add(key, to_insert, new_seg, m->init_val_capacity);
                assert(last_seg_in_bucket->next == NULL);
                last_seg_in_bucket->next = new_seg;
                m->bucket_tails[bucket] = new_seg;
            } else {
                // Insert in existing segment
                hvr_map_seg_add(key, to_insert, last_seg_in_bucket,
                        m->init_val_capacity);
            }
        }
    }
}

// Remove function for edge info
void hvr_map_remove(hvr_vertex_id_t key, void *val, hvr_map_t *m) {
    hvr_map_seg_t *seg;
    unsigned seg_index;

    const int success = hvr_map_find(key, m, &seg, &seg_index);

    if (success) {
        assert(seg->data[seg_index].cached_vert == val);

        unsigned copy_to = seg_index;
        unsigned copy_from = seg->nkeys - 1;

        memcpy(&(seg->data[copy_to]), &(seg->data[copy_from]),
                sizeof(seg->data[0]));
        seg->nkeys -= 1;
    }
}

void *hvr_map_get(hvr_vertex_id_t key, hvr_map_t *m) {
    hvr_map_seg_t *seg;
    unsigned seg_index;

    const int success = hvr_map_find(key, m, &seg, &seg_index);

    if (success) {
        return seg->data[seg_index].cached_vert;
    } else {
        return NULL;
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
        size_t *out_used) {
    size_t allocated = sizeof(*m) + m->n_prealloc * sizeof(hvr_map_seg_t);

    size_t used = sizeof(*m);
    for (unsigned b = 0; b < HVR_MAP_BUCKETS; b++) {
        hvr_map_seg_t *bucket = m->buckets[b];
        while (bucket) {
            unsigned n_unused_keys = HVR_MAP_SEG_SIZE - bucket->nkeys;
            used += sizeof(hvr_map_seg_t) - (n_unused_keys *
                    (sizeof(hvr_map_val_t) + sizeof(hvr_vertex_id_t)));
            bucket = bucket->next;
        }
    }

    *out_capacity = allocated;
    *out_used = used;
}
