#ifndef _HVR_VERTEX_H
#define _HVR_VERTEX_H

#include <string.h>

#include "hvr_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HVR_MAX_VECTOR_SIZE 12

typedef struct _hvr_vertex_t {
    hvr_vertex_id_t id;

    double values[HVR_MAX_VECTOR_SIZE];

    hvr_time_t creation_iter;
    unsigned char needs_send;
    unsigned char needs_processing;
    hvr_partition_t curr_part;

    // These two fields must be the last in the vertex struct, in this order
    struct _hvr_vertex_t *next_in_partition;
    struct _hvr_vertex_t *prev_in_partition;
} hvr_vertex_t;

/*
 * Create nvecs new, empty vectors.
 */
extern hvr_vertex_t *hvr_vertex_create(hvr_ctx_t ctx);
extern hvr_vertex_t *hvr_vertex_create_n(size_t n, hvr_ctx_t ctx);

/*
 * Remove these vertices from the graph.
 */
extern void hvr_vertex_delete(hvr_vertex_t *vert, hvr_ctx_t ctx);

/*
 * Initialize an empty sparse vector.
 */
extern void hvr_vertex_init(hvr_vertex_t *vert, hvr_ctx_t ctx);

/*
 * Get the value for the specified feature in the provided vector.
 */
static inline double hvr_vertex_get(const unsigned feature,
        const hvr_vertex_t *vert, hvr_ctx_t in_ctx) {
    assert(feature < HVR_MAX_VECTOR_SIZE);
    return vert->values[feature];
}

static inline uint64_t hvr_vertex_get_uint64(const unsigned feature,
        const hvr_vertex_t *vert, hvr_ctx_t in_ctx) {
    assert(sizeof(uint64_t) == sizeof(double));
    assert(feature < HVR_MAX_VECTOR_SIZE);

    uint64_t result;
    memcpy(&result, &(vert->values[feature]), sizeof(result));
    return result;
}

/*
 * Set the specified feature to the specified value in the provided vector.
 */
static inline void hvr_vertex_set(const unsigned feature, const double val,
        hvr_vertex_t *vert, hvr_ctx_t in_ctx) {
    assert(feature < HVR_MAX_VECTOR_SIZE);
    if (val != vert->values[feature]) {
        vert->values[feature] = val;
        vert->needs_send = 1;
    }
}

static inline void hvr_vertex_set_uint64(const unsigned feature,
        const uint64_t val, hvr_vertex_t *vert, hvr_ctx_t in_ctx) {
    assert(sizeof(uint64_t) == sizeof(double));
    assert(feature < HVR_MAX_VECTOR_SIZE);

    uint64_t old = hvr_vertex_get_uint64(feature, vert, in_ctx);
    if (val != old) {
        memcpy(&(vert->values[feature]), &val, sizeof(val));
        vert->needs_send = 1;
    }
}

/*
 * Write a string representation of the passed vertex to the provided character
 * buffer.
 */
extern void hvr_vertex_dump(hvr_vertex_t *vert, char *buf,
        const size_t buf_size, hvr_ctx_t ctx);

/*
 * Get the PE that allocated and owns this vertex.
 */
extern int hvr_vertex_get_owning_pe(hvr_vertex_t *vec);

/*
 * Add the contents of two vertices together (dst and src) and store the result
 * in dst.
 */
extern void hvr_vertex_add(hvr_vertex_t *dst, hvr_vertex_t *src,
        hvr_ctx_t in_ctx);

/*
 * Check if two vertices are identical. To be identical, they must have the same
 * number of attributes, the same attributes, and the same value for each
 * attribute.
 */
extern int hvr_vertex_equal(hvr_vertex_t *a, hvr_vertex_t *b, hvr_ctx_t in_ctx);

/*
 * Force update_metadata to be called on this vertex and any neighbors on the
 * next iteration. This can be used as a way to trigger work, even if an update
 * hasn't been made to the vertices attributes.
 */
extern void hvr_vertex_trigger_update(hvr_vertex_t *vert, hvr_ctx_t in_ctx);

static inline hvr_vertex_id_t hvr_vertex_get_id(hvr_vertex_t *vert) {
    return vert->id;
}

#ifdef __cplusplus
}
#endif

#endif // _HVR_VERTEX_H
