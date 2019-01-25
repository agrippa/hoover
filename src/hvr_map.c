#include "hvr_map.h"

#include <stdio.h>
#include <string.h>

#define HVR_MAP_BUCKET(my_key) ((my_key) % HVR_MAP_BUCKETS)
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static int comp(const void *_a, const void *_b) {
    hvr_map_entry_t *a = (hvr_map_entry_t *)_a;
    hvr_map_entry_t *b = (hvr_map_entry_t *)_b;

    if (a->key < b->key) return -1;
    else if (a->key > b->key) return 1;
    else return 0;
}

// Add a new key with one initial value
static void hvr_map_seg_add(hvr_vertex_id_t key, hvr_map_val_t val,
        hvr_map_seg_t *s, unsigned init_val_capacity) {
    const unsigned insert_index = s->nkeys;
    assert(insert_index < HVR_MAP_SEG_SIZE);
    s->data[insert_index].key = key;
    s->keys[insert_index] = key;

    assert(HVR_MAP_N_INLINE_VALS > 0);

    s->data[insert_index].inline_vals[0] = val;
    s->data[insert_index].ext_vals = NULL;
    s->data[insert_index].ext_capacity = 0;
    s->data[insert_index].length = 1;

    s->nkeys++;

    if (s->nkeys == HVR_MAP_SEG_SIZE) {
        qsort(&(s->data[0]), HVR_MAP_SEG_SIZE, sizeof(s->data[0]), comp);
        for (unsigned i = 0; i < HVR_MAP_SEG_SIZE; i++) {
            s->keys[i] = s->data[i].key;
        }
    }
}

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
            int index = binarySearch(seg->keys, key);
            if (index >= 0) {
                *out_seg = seg;
                *out_index = index;
                return 1;
            }
        } else {
            for (unsigned i = 0; i < nkeys; i++) {
                if (seg->keys[i] == key) {
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
        unsigned vals_pool_nodes, unsigned init_val_capacity,
        hvr_map_type_t type) {
    memset(m, 0x00, sizeof(*m));
    m->init_val_capacity = init_val_capacity;
    m->type = type;

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

void hvr_map_add(hvr_vertex_id_t key, hvr_map_val_t to_insert,
        hvr_map_t *m) {
    hvr_map_seg_t *seg;
    unsigned seg_index;
    int success = hvr_map_find(key, m, &seg, &seg_index);

    if (success) {
        // Key already exists, add it to existing values
        unsigned i = 0;
        unsigned nvals = seg->data[seg_index].length;
        hvr_map_val_t *inline_vals = &(seg->data[seg_index].inline_vals[0]);
        hvr_map_val_t *ext_vals = seg->data[seg_index].ext_vals;

        /*
         * Check if this value is already present for the given key, and if so
         * abort early.
         */
        switch (m->type) {
            case (EDGE_INFO):
                while (i < nvals && i < HVR_MAP_N_INLINE_VALS) {
                    if (EDGE_INFO_VERTEX(inline_vals[i].edge_info) ==
                            EDGE_INFO_VERTEX(to_insert.edge_info)) {
                        assert(EDGE_INFO_EDGE(inline_vals[i].edge_info) ==
                                EDGE_INFO_EDGE(to_insert.edge_info));
                        return;
                    }
                    i++;
                }
                while (i < nvals) {
                    if (EDGE_INFO_VERTEX(ext_vals[i - HVR_MAP_N_INLINE_VALS].edge_info) ==
                            EDGE_INFO_VERTEX(to_insert.edge_info)) {
                        assert(EDGE_INFO_EDGE(ext_vals[i - HVR_MAP_N_INLINE_VALS].edge_info) ==
                                EDGE_INFO_EDGE(to_insert.edge_info));
                        return;
                    }
                    i++;
                }
                break;
            case (CACHED_VERT_INFO):
                while (i < nvals && i < HVR_MAP_N_INLINE_VALS) {
                    if (inline_vals[i].cached_vert == to_insert.cached_vert) {
                        return;
                    }
                    i++;
                }
                while (i < nvals) {
                    if (ext_vals[i - HVR_MAP_N_INLINE_VALS].cached_vert ==
                            to_insert.cached_vert) {
                        return;
                    }
                    i++;
                }
                break;
            case (INTERACT_INFO):
                while (i < nvals && i < HVR_MAP_N_INLINE_VALS) {
                    if (inline_vals[i].interact == to_insert.interact) {
                        return;
                    }
                    i++;
                }
                while (i < nvals) {
                    if (ext_vals[i - HVR_MAP_N_INLINE_VALS].interact ==
                            to_insert.interact) {
                        return;
                    }
                    i++;
                }
                break;
            default:
                abort();
        }

        if (nvals < HVR_MAP_N_INLINE_VALS) {
            // Can immediately insert into inline values.
            inline_vals[nvals] = to_insert;
        } else {
            // Must insert into extended values
            if (nvals - HVR_MAP_N_INLINE_VALS == seg->data[seg_index].ext_capacity) {
                // Need to resize
                unsigned curr_capacity = seg->data[seg_index].ext_capacity;
                unsigned new_capacity = (curr_capacity == 0 ?
                        m->init_val_capacity : 2 * curr_capacity);

                hvr_map_val_t *new_mem = mspace_realloc(m->tracker, ext_vals,
                        new_capacity * sizeof(*new_mem));
                assert(new_mem);
                seg->data[seg_index].ext_vals = new_mem;
                seg->data[seg_index].ext_capacity = new_capacity;
            }
            seg->data[seg_index].ext_vals[nvals - HVR_MAP_N_INLINE_VALS] =
                to_insert;
        }

        seg->data[seg_index].length += 1;
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
void hvr_map_remove(hvr_vertex_id_t key, hvr_map_val_t val,
        hvr_map_t *m) {
    hvr_map_seg_t *seg;
    unsigned seg_index;

    const int success = hvr_map_find(key, m, &seg, &seg_index);

    if (success) {
        const unsigned nvals = seg->data[seg_index].length;
        hvr_map_val_t *inline_vals = &(seg->data[seg_index].inline_vals[0]);
        hvr_map_val_t *ext_vals = seg->data[seg_index].ext_vals;
        unsigned j = 0;
        int found = 0;

        // Clear out the value we are removing
        switch (m->type) {
            case (EDGE_INFO):
                while (!found && j < HVR_MAP_N_INLINE_VALS && j < nvals) {
                    if (EDGE_INFO_VERTEX(inline_vals[j].edge_info) ==
                            EDGE_INFO_VERTEX(val.edge_info)) {
                        found = 1;
                    } else {
                        j++;
                    }
                }
                while (!found && j < nvals) {
                    if (EDGE_INFO_VERTEX(ext_vals[j - HVR_MAP_N_INLINE_VALS].edge_info) ==
                            EDGE_INFO_VERTEX(val.edge_info)) {
                        found = 1;
                    } else {
                        j++;
                    }
                }
                break;
            case (CACHED_VERT_INFO):
                while (!found && j < HVR_MAP_N_INLINE_VALS && j < nvals) {
                    if (inline_vals[j].cached_vert == val.cached_vert) {
                        found = 1;
                    } else {
                        j++;
                    }
                }
                while (!found && j < nvals) {
                    if (ext_vals[j - HVR_MAP_N_INLINE_VALS].cached_vert ==
                            val.cached_vert) {
                        found = 1;
                    } else {
                        j++;
                    }
                }
                break;
            case (INTERACT_INFO):
                while (!found && j < HVR_MAP_N_INLINE_VALS && j < nvals) {
                    if (inline_vals[j].interact == val.interact) {
                        found = 1;
                    } else {
                        j++;
                    }
                }
                while (!found && j < nvals) {
                    if (ext_vals[j - HVR_MAP_N_INLINE_VALS].interact ==
                            val.interact) {
                        found = 1;
                    } else {
                        j++;
                    }
                }
                break;
            default:
                abort();
        }

        if (found) {
            // Found it
            assert(j < nvals);
            hvr_map_val_t *copy_to, *copy_from;

            if (j < HVR_MAP_N_INLINE_VALS) {
                copy_to = &inline_vals[j];
            } else {
                copy_to = &ext_vals[j - HVR_MAP_N_INLINE_VALS];
            }

            if (nvals - 1 < HVR_MAP_N_INLINE_VALS) {
                copy_from = &inline_vals[nvals - 1];
            } else {
                copy_from = &ext_vals[nvals - 1 - HVR_MAP_N_INLINE_VALS];
            }

            *copy_to = *copy_from;
            seg->data[seg_index].length = nvals - 1;
        }
    }
}

int hvr_map_linearize(hvr_vertex_id_t key, hvr_map_t *m,
        hvr_map_val_list_t *out_vals) {
    hvr_map_seg_t *seg;
    unsigned seg_index;

    const int success = hvr_map_find(key, m, &seg, &seg_index);

    if (success) {
        out_vals->inline_vals = &(seg->data[seg_index].inline_vals[0]);
        out_vals->ext_vals = seg->data[seg_index].ext_vals;

        return seg->data[seg_index].length;
    } else {
        return -1;
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

size_t hvr_map_count_values(hvr_vertex_id_t key, hvr_map_t *m) {
    hvr_map_seg_t *seg;
    unsigned seg_index;

    const int success = hvr_map_find(key, m, &seg, &seg_index);

    if (success) {
        return seg->data[seg_index].length;
    } else {
        return 0;
    }
}

void hvr_map_size_in_bytes(hvr_map_t *m, size_t *out_capacity,
        size_t *out_used, double *avg_val_capacity, double *avg_val_length,
        unsigned *out_max_val_length) {
    size_t allocated = sizeof(*m);
    size_t used = sizeof(*m);

    size_t sum_val_lengths = 0;
    size_t count_keys = 0;
    unsigned max_val_length = 0;

    allocated += m->n_prealloc * sizeof(hvr_map_seg_t);
    for (unsigned b = 0; b < HVR_MAP_BUCKETS; b++) {
        hvr_map_seg_t *bucket = m->buckets[b];
        while (bucket) {
            for (unsigned i = 0; i < bucket->nkeys; i++) {
                allocated += bucket->data[i].ext_capacity *
                    sizeof(hvr_map_val_t);
                used += (bucket->data[i].length - 1) * sizeof(hvr_map_val_t);
                sum_val_lengths += bucket->data[i].length;
                if (bucket->data[i].length > max_val_length) {
                    max_val_length = bucket->data[i].length;
                }
                count_keys++;
            }
            unsigned n_unused_keys = HVR_MAP_SEG_SIZE - bucket->nkeys;
            used += sizeof(hvr_map_seg_t) - (n_unused_keys *
                    sizeof(hvr_map_entry_t) + sizeof(hvr_vertex_id_t));
            bucket = bucket->next;
        }
    }

    *out_capacity = allocated;
    *out_used = used;
    *avg_val_capacity = 0;
    *avg_val_length = (double)sum_val_lengths / (double)count_keys;
    *out_max_val_length = max_val_length;
}
