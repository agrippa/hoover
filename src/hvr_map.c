#include "hvr_map.h"

#include <string.h>

#define HVR_MAP_BUCKET(my_key) ((my_key) % HVR_MAP_BUCKETS)

// Add a new key with one initial value
static void hvr_map_seg_add(hvr_vertex_id_t key, hvr_map_val_t val,
        hvr_map_seg_t *s, unsigned init_val_capacity) {
    const unsigned insert_index = s->nkeys;
    assert(insert_index < HVR_MAP_SEG_SIZE);
    s->keys[insert_index] = key;
    s->vals[insert_index] = (hvr_map_val_t *)malloc(
            init_val_capacity * sizeof(hvr_map_val_t));
    assert(s->vals[insert_index]);

    s->vals[insert_index][0] = val;

    s->capacity[insert_index] = init_val_capacity;
    s->length[insert_index] = 1;
    s->nkeys++;
}

static int hvr_map_find(hvr_vertex_id_t key, hvr_map_t *m,
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

void hvr_map_init(hvr_map_t *m, unsigned init_val_capacity) {
    memset(m, 0x00, sizeof(*m));
    m->init_val_capacity = init_val_capacity;
}

void hvr_map_add(hvr_vertex_id_t key, hvr_map_val_t to_insert, int is_edge_info,
        hvr_map_t *m) {
    hvr_map_seg_t *seg;
    unsigned seg_index;
    int success = hvr_map_find(key, m, &seg, &seg_index);

    if (success) {
        // Key already exists, add it to existing values
        unsigned nvals = seg->length[seg_index];
        hvr_map_val_t *vals = seg->vals[seg_index];

        if (is_edge_info) {
            for (unsigned i = 0; i < nvals; i++) {
                if (vals[i].edge_info.id == to_insert.edge_info.id) {
                    assert(vals[i].edge_info.edge == to_insert.edge_info.edge);
                    return;
                }
            }
        } else {
            for (unsigned i = 0; i < nvals; i++) {
                if (vals[i].cached_vert == to_insert.cached_vert) {
                    return;
                }
            }
        }

        if (nvals == seg->capacity[seg_index]) {
            // Need to resize
            unsigned new_capacity = 2 * seg->capacity[seg_index];
            seg->vals[seg_index] = (hvr_map_val_t *)realloc(vals,
                    new_capacity * sizeof(hvr_map_val_t));
            assert(seg->vals[seg_index]);
            seg->capacity[seg_index] = new_capacity;
        }

        seg->vals[seg_index][nvals] = to_insert;
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
        } else {
            hvr_map_seg_t *last_seg_in_bucket = m->buckets[bucket];
            while (last_seg_in_bucket->next) {
                last_seg_in_bucket = last_seg_in_bucket->next;
            }

            if (last_seg_in_bucket->nkeys == HVR_MAP_SEG_SIZE) {
                // Have to append new segment
                hvr_map_seg_t *new_seg = (hvr_map_seg_t *)calloc(1,
                        sizeof(*new_seg));
                assert(new_seg);
                hvr_map_seg_add(key, to_insert, new_seg, m->init_val_capacity);
                assert(last_seg_in_bucket->next == NULL);
                last_seg_in_bucket->next = new_seg;
            } else {
                // Insert in existing segment
                hvr_map_seg_add(key, to_insert, last_seg_in_bucket, m->init_val_capacity);
            }
        }
    }
}

// Remove function for edge info
void hvr_map_remove(hvr_vertex_id_t key, hvr_map_val_t val, int is_edge_info,
        hvr_map_t *m) {
    hvr_map_seg_t *seg;
    unsigned seg_index;

    const int success = hvr_map_find(key, m, &seg, &seg_index);

    if (success) {
        const unsigned nvals = seg->length[seg_index];
        hvr_map_val_t *vals = seg->vals[seg_index];

        // Clear out the value we are removing
        if (is_edge_info) {
            for (unsigned j = 0; j < nvals; j++) {
                if (vals[j].edge_info.id == val.edge_info.id) {
                    vals[j] = vals[nvals - 1];
                    seg->length[seg_index] = nvals - 1;
                    return;
                }
            }

        } else {
            for (unsigned j = 0; j < nvals; j++) {
                if (vals[j].cached_vert == val.cached_vert) {
                    vals[j] = vals[nvals - 1];
                    seg->length[seg_index] = nvals - 1;
                    return;
                }
            }
        }
    }
}

hvr_edge_type_t hvr_map_contains(hvr_vertex_id_t key,
        hvr_vertex_id_t val, hvr_map_t *m) {
    hvr_map_seg_t *seg;
    unsigned seg_index;

    const int success = hvr_map_find(key, m, &seg, &seg_index);

    if (success) {
        const unsigned nvals = seg->length[seg_index];
        hvr_map_val_t *vals = seg->vals[seg_index];
        for (unsigned j = 0; j < nvals; j++) {
            if (vals[j].edge_info.id == val) {
                return vals[j].edge_info.edge;
            }
        }
    }
    return NO_EDGE;
}

unsigned hvr_map_linearize(hvr_vertex_id_t key,
        hvr_map_val_t **out_vals, unsigned *capacity, hvr_map_t *m) {
    hvr_map_seg_t *seg;
    unsigned seg_index;

    const int success = hvr_map_find(key, m, &seg, &seg_index);

    if (success) {
        const unsigned nvals = seg->length[seg_index];
        hvr_map_val_t *vals = seg->vals[seg_index];

        if (*capacity < nvals) {
            *out_vals = (hvr_map_val_t *)realloc(*out_vals,
                    nvals * sizeof(hvr_map_val_t));
            assert(*out_vals);
            *capacity = nvals;
        }

        memcpy(*out_vals, vals, nvals * sizeof(*vals));
        return nvals;
    } else {
        return 0;
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
