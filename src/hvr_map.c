#include "hvr_map.h"

#define INIT_VAL_CAPACITY 16

static void hvr_map_seg_init(hvr_map_seg_t *s) {
    memset(s, 0x00, sizeof(*s));
}

// Add a new key with one initial value
static void hvr_map_seg_add(hvr_vertex_id key, hvr_vertex_id_t val,
        hvr_edge_type_t edge_type, hvr_map_seg_t *s) {
    const unsigned insert_index = s->nkeys;
    assert(insert_index < HVR_MAP_SEG_SIZE);
    s->keys[insert_index] = key;
    s->values[insert_index] = (hvr_vertex_id_t *)malloc(
            INIT_VAL_CAPACITY * sizeof(hvr_vertex_id_t));
    assert(s->values[insert_index]);
    s->edge_types[insert_index] = (hvr_edge_type_t *)malloc(
            INIT_VAL_CAPACITY * sizeof(hvr_edge_type_t));
    assert(s->edge_types[insert_index]);

    s->values[insert_index][0] = val;
    s->edge_Types[insert_index][0] = edge_type;

    s->capacity[insert_index] = INIT_VAL_CAPACITY;
    s->length[insert_index] = 1;
    s->nkeys++;
}

void hvr_map_init(hvr_map_t *m) {
    memset(m, 0x00, sizeof(*m));
}

void hvr_map_add(hvr_vertex_id_t key, hvr_vertex_id_t val,
        hvr_edge_type_t edge_type, hvr_map_t *m) {
    unsigned bucket = key % HVR_MAP_BUCKETS;

    hvr_map_seg_t *prev = NULL;
    hvr_map_seg_t *seg = m->buckets[bucket];
    while (seg) {
        for (unsigned i = 0; i < seg->nkeys; i++) {
            if (seg->keys[i] == key) {
                if (seg->length[i] == seg->capacity[i]) {
                    unsigned new_capacity = 2 * seg->capacity[i];
                    seg->values[i] = (hvr_vertex_id_t *)realloc(seg->values[i],
                            new_capacity * sizeof(hvr_vertex_id_t));
                    assert(seg->values[i]);

                    seg->edge_types[i] = (hvr_edge_type_t *)realloc(
                            seg->edge_types[i],
                            new_capacity * sizeof(hvr_edge_type_t));
                    seg->capacity[i] = new_capacity;
                }
                seg->values[i][seg->length[i]] = val;
                seg->edge_types[i][seg->length[i]] = edge_type;
                seg->length[i] += 1;
                return;
            }
        }
        prev = seg;
        seg = seg->next;
    }

    // Have to add as new key
    if (prev == NULL) {
        // First segment created
        hvr_map_seg_t *new_seg = (hvr_map_seg_t *)malloc(sizeof(*new_seg));
        assert(new_seg);
        hvr_map_seg_init(new_seg);

        hvr_map_seg_add(key, val, edge_type, new_seg);
        assert(m->buckets[bucket] == NULL);
        m->buckets[bucket] = new_seg;
    } else {
        if (prev->nkeys == HVR_MAP_SEG_SIZE) {
            // Have to append new segment
            hvr_map_seg_t *new_seg = (hvr_map_seg_t *)malloc(sizeof(*new_seg));
            assert(new_seg);
            hvr_map_seg_init(new_seg);
            hvr_map_seg_add(key, val, edge_type, new_seg);
            assert(prev->next == NULL);
            prev->next = new_seg;
        } else {
            // Insert in existing segment
            hvr_map_seg_add(key, val, edge_type, prev);
        }
    }
}

static int hvr_map_find(hvr_vertex_id_t key, hvr_map_t *m,
        hvr_map_seg_t **out_seg, unsigned *out_index) {
    unsigned bucket = key % HVR_MAP_BUCKETS;

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

void hvr_map_remove(hvr_vertex_id_t key, hvr_vertex_id_t val,
        hvr_map_t *m) {
    hvr_map_seg_t *seg;
    unsigned seg_index;

    const int success = hvr_map_find(key, m, &seg, &seg_index);

    if (success) {
        const unsigned nvals = seg->length[seg_index];
        hvr_vertex_id_t *vals = seg->values[seg_index];
        hvr_edge_type_t *edges = seg->edge_types[seg_index];
        for (unsigned j = 0; j < nvals; j++) {
            if (vals[j] == val) {
                // Clear out the value we are removing
                vals[j] = vals[nvals - 1];
                edges[j] = edges[nvals- 1];
                seg->length[seg_index] = nvals - 1;
                return;
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
        hvr_vertex_id_t *vals = seg->values[seg_index];
        hvr_edge_type_t *edges = seg->edge_types[seg_index];
        for (unsigned j = 0; j < nvals; j++) {
            if (vals[j] == val) {
                return edges[j];
            }
        }
    }
    return NO_EDGE;
}

unsigned hvr_map_linearize(hvr_vertex_id_t key,
        hvr_vertex_id_t **vertices, hvr_edge_type_t **directions,
        unsigned *capacity, hvr_map_t *m) {
    hvr_map_seg_t *seg;
    unsigned seg_index;

    const int success = hvr_map_find(key, m, &seg, &seg_index);

    if (success) {
        const unsigned nvals = seg->length[seg_index];
        hvr_vertex_id_t *vals = seg->values[seg_index];
        hvr_edge_type_t *edges = seg->edge_types[seg_index];

        if (*capacity < nvals) {
            *vertices = (hvr_vertex_id_t *)realloc(*vertices,
                    nvals * sizeof(hvr_vertex_id_t));
            assert(*vertices);
            *directions = (hvr_edge_type_t *)realloc(*directions,
                    nvals * sizeof(hvr_edge_type_t));
            assert(*directions);
            *capacity = nvals;
        }

        memcpy(*vertices, vals, nvals * sizeof(hvr_vertex_id_t));
        memcpy(*directions, edges, nvals * sizeof(hvr_edge_type_t));
        return nvals;
    } else {
        assert(0);
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

void hvr_map_count_values(hvr_vertex_id_t key, hvr_map_t *m) {
    hvr_map_seg_t *seg;
    unsigned seg_index;

    const int success = hvr_map_find(key, m, &seg, &seg_index);

    if (success) {
        return seg->length[seg_index];
    } else {
        return 0;
    }
}
