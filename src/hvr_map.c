#include "hvr_map.h"

#include <string.h>

#define HVR_MAP_BUCKET(my_key) ((my_key) % HVR_MAP_BUCKETS)
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Add a new key with one initial value
static void hvr_map_seg_add(hvr_vertex_id_t key, hvr_map_val_t val,
        hvr_map_seg_t *s, unsigned init_val_capacity) {
    const unsigned insert_index = s->nkeys;
    assert(insert_index < HVR_MAP_SEG_SIZE);
    s->keys[insert_index] = key;

    assert(HVR_MAP_N_INLINE_VALS > 0);

    s->inline_vals[insert_index][0] = val;
    s->ext_vals[insert_index] = NULL;
    s->ext_capacity[insert_index] = 0;

    s->length[insert_index] = 1;
    s->nkeys++;
}

static inline int hvr_map_find(hvr_vertex_id_t key, hvr_map_t *m,
        hvr_map_seg_t **out_seg, unsigned *out_index) {
    unsigned bucket = HVR_MAP_BUCKET(key);

    hvr_map_seg_t *seg = m->buckets[bucket];
    while (seg) {
        for (unsigned i = 0; i < seg->nkeys; i++) {
            if (seg->keys[i] == key) {
                *out_seg = seg;
                *out_index = i;
                return 1;
            }
        }
        seg = seg->next;
    }
    return 0;
}

void hvr_map_init(hvr_map_t *m, unsigned init_val_capacity,
        hvr_map_type_t type) {
    memset(m, 0x00, sizeof(*m));
    m->init_val_capacity = init_val_capacity;
    m->type = type;
}

void hvr_map_add(hvr_vertex_id_t key, hvr_map_val_t to_insert,
        hvr_map_t *m) {
    hvr_map_seg_t *seg;
    unsigned seg_index;
    int success = hvr_map_find(key, m, &seg, &seg_index);

    if (success) {
        // Key already exists, add it to existing values
        unsigned i = 0;
        unsigned nvals = seg->length[seg_index];
        hvr_map_val_t *inline_vals = &(seg->inline_vals[seg_index][0]);
        hvr_map_val_t *ext_vals = seg->ext_vals[seg_index];

        /*
         * Check if this value is already present for the given key, and if so
         * abort early.
         */
        switch (m->type) {
            case (EDGE_INFO):
                while (i < nvals && i < HVR_MAP_N_INLINE_VALS) {
                    if (EDGE_INFO_VERTEX(inline_vals[i].edge_info) == EDGE_INFO_VERTEX(to_insert.edge_info)) {
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
            if (nvals - HVR_MAP_N_INLINE_VALS == seg->ext_capacity[seg_index]) {
                // Need to resize
                unsigned curr_capacity = seg->ext_capacity[seg_index];
                unsigned new_capacity = (curr_capacity == 0 ?
                        m->init_val_capacity : 2 * curr_capacity);
                seg->ext_vals[seg_index] = (hvr_map_val_t *)realloc(ext_vals,
                        new_capacity * sizeof(hvr_map_val_t));
                assert(seg->ext_vals[seg_index]);
                seg->ext_capacity[seg_index] = new_capacity;
            }
            seg->ext_vals[seg_index][nvals - HVR_MAP_N_INLINE_VALS] = to_insert;
        }

        seg->length[seg_index] += 1;
    } else {
        const unsigned bucket = HVR_MAP_BUCKET(key);

        // Have to add as new key
        if (m->buckets[bucket] == NULL) {
            // First segment created
            hvr_map_seg_t *new_seg = (hvr_map_seg_t *)calloc(1,
                    sizeof(*new_seg));
            assert(new_seg);

            hvr_map_seg_add(key, to_insert, new_seg, m->init_val_capacity);
            assert(m->buckets[bucket] == NULL);
            m->buckets[bucket] = new_seg;
            m->bucket_tails[bucket] = new_seg;
        } else {
            hvr_map_seg_t *last_seg_in_bucket= m->bucket_tails[bucket];

            if (last_seg_in_bucket->nkeys == HVR_MAP_SEG_SIZE) {
                // Have to append new segment
                hvr_map_seg_t *new_seg = (hvr_map_seg_t *)calloc(1,
                        sizeof(*new_seg));
                assert(new_seg);
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
        const unsigned nvals = seg->length[seg_index];
        hvr_map_val_t *inline_vals = &(seg->inline_vals[seg_index][0]);
        hvr_map_val_t *ext_vals = seg->ext_vals[seg_index];
        unsigned j = 0;
        int found = 0;

        // Clear out the value we are removing
        switch (m->type) {
            case (EDGE_INFO):
                while (!found && j < HVR_MAP_N_INLINE_VALS && j < nvals) {
                    if (EDGE_INFO_VERTEX(inline_vals[j].edge_info) == EDGE_INFO_VERTEX(val.edge_info)) {
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
            seg->length[seg_index] = nvals - 1;
        }
    }
}

hvr_edge_type_t hvr_map_contains(hvr_vertex_id_t key,
        hvr_vertex_id_t val, hvr_map_t *m) {
    assert(m->type == EDGE_INFO);

    hvr_map_seg_t *seg;
    unsigned seg_index;

    const int success = hvr_map_find(key, m, &seg, &seg_index);

    if (success) {
        const unsigned nvals = seg->length[seg_index];
        hvr_map_val_t *inline_vals = &(seg->inline_vals[seg_index][0]);
        hvr_map_val_t *ext_vals = seg->ext_vals[seg_index];
        unsigned j = 0;
        while (j < MIN(HVR_MAP_N_INLINE_VALS, nvals)) {
            if (EDGE_INFO_VERTEX(inline_vals[j].edge_info) == val) {
                return EDGE_INFO_EDGE(inline_vals[j].edge_info);
            }
            j++;
        }
        while (j < nvals) {
            if (EDGE_INFO_VERTEX(ext_vals[j - HVR_MAP_N_INLINE_VALS].edge_info) == val) {
                return EDGE_INFO_EDGE(ext_vals[j - HVR_MAP_N_INLINE_VALS].edge_info);
            }
            j++;
        }
    }
    return NO_EDGE;
}

int hvr_map_linearize(hvr_vertex_id_t key,
        hvr_map_val_t **out_vals, unsigned *capacity, hvr_map_t *m) {
    hvr_map_seg_t *seg;
    unsigned seg_index;

    const int success = hvr_map_find(key, m, &seg, &seg_index);

    if (success) {
        const unsigned nvals = seg->length[seg_index];
        hvr_map_val_t *inline_vals = &(seg->inline_vals[seg_index][0]);
        hvr_map_val_t *ext_vals = seg->ext_vals[seg_index];

        if (*capacity < nvals) {
            *out_vals = (hvr_map_val_t *)realloc(*out_vals,
                    nvals * sizeof(hvr_map_val_t));
            assert(*out_vals);
            *capacity = nvals;
        }

        memcpy(*out_vals, inline_vals,
                MIN(nvals, HVR_MAP_N_INLINE_VALS) * sizeof(*inline_vals));
        if (nvals > HVR_MAP_N_INLINE_VALS) {
            memcpy((*out_vals) + HVR_MAP_N_INLINE_VALS, ext_vals,
                    (nvals - HVR_MAP_N_INLINE_VALS) * sizeof(*ext_vals));
        }
        return nvals;
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
        return seg->length[seg_index];
    } else {
        return 0;
    }
}

void hvr_map_size_in_bytes(hvr_map_t *m, size_t *out_capacity,
        size_t *out_used, double *avg_val_capacity, double *avg_val_length) {
    size_t capacity = sizeof(*m);
    size_t used = sizeof(*m);
    uint64_t sum_val_capacity = 0;
    uint64_t sum_val_length = 0;
    uint64_t count_vals = 0;

    for (unsigned b = 0; b < HVR_MAP_BUCKETS; b++) {
        hvr_map_seg_t *bucket = m->buckets[b];
        while (bucket) {
            capacity += sizeof(*bucket);
            used += sizeof(*bucket);

            for (unsigned v = 0; v < HVR_MAP_SEG_SIZE; v++) {
                capacity += (HVR_MAP_N_INLINE_VALS + bucket->ext_capacity[v]) *
                    sizeof(hvr_map_val_t);
                used += bucket->length[v] * sizeof(hvr_map_val_t);

                if (bucket->length[v] > 0) {
                    sum_val_capacity += HVR_MAP_N_INLINE_VALS +
                        bucket->ext_capacity[v];
                    sum_val_length += bucket->length[v];
                    count_vals++;
                }
            }

            bucket = bucket->next;
        }
    }

    *out_capacity = capacity;
    *out_used = used;
    *avg_val_capacity = (double)sum_val_capacity / (double)count_vals;
    *avg_val_length = (double)sum_val_length / (double)count_vals;
}
