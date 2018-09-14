#ifndef _HVR_VERTEX_H
#define _HVR_VERTEX_H

#include "hvr_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HVR_MAX_VECTOR_SIZE 7

typedef struct _hvr_vertex_t {
    hvr_vertex_id_t id;

    double values[HVR_MAX_VECTOR_SIZE];
    unsigned features[HVR_MAX_VECTOR_SIZE];
    unsigned size;

    struct _hvr_vertex_t *next_in_partition;

    hvr_time_t creation_iter;
    hvr_time_t last_modify_iter;
} hvr_vertex_t;

/*
 * Create nvecs new, empty vectors.
 */
hvr_vertex_t *hvr_vertex_create(hvr_ctx_t ctx);

/*
 * Remove these vertices from the graph.
 */
void hvr_vertex_delete(hvr_vertex_t *vert, hvr_ctx_t ctx);

/*
 * Initialize an empty sparse vector.
 */
void hvr_vertex_init(hvr_vertex_t *vert, hvr_ctx_t ctx);

/*
 * Set the specified feature to the specified value in the provided vector.
 */
void hvr_vertex_set(const unsigned feature, const double val,
        hvr_vertex_t *vert, hvr_ctx_t in_ctx);

/*
 * Get the value for the specified feature in the provided vector.
 */
double hvr_vertex_get(const unsigned feature, hvr_vertex_t *vert,
        hvr_ctx_t in_ctx);

#ifdef __cplusplus
}
#endif

#endif // _HVR_VERTEX_H
