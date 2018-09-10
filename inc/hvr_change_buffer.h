#ifndef _HVR_CHANGE_BUFFER_H
#define _HVR_CHANGE_BUFFER_H

#include "hvr_common.h"
#include "hvr_sparse_vec.h"

typedef enum {
    CREATE,
    DELETE,
    UPDATE
} hvr_change_type_t;

typedef struct _hvr_change_t {
    hvr_vertex_id_t id;
    hvr_time_t timestamp;
    hvr_change_type_t type;
    hvr_graph_t graph;

    double values[HVR_BUCKET_SIZE];
    unsigned features[HVR_BUCKET_SIZE];
    unsigned size;

    double const_values[HVR_MAX_CONSTANT_ATTRS];
    unsigned const_features[HVR_MAX_CONSTANT_ATTRS];
    unsigned n_const_features;
} hvr_change_t;

typedef struct _hvr_timestep_changes_node_t {
    hvr_change_t change;
    struct _hvr_timestep_changes_node_t *next;
} hvr_timestep_changes_node_t;

typedef struct _hvr_timestep_changes_t {
    hvr_time_t timestamp;
    hvr_timestep_changes_node_t *changes_list;
    struct _hvr_timestep_changes_t *next;
} hvr_timestep_changes_t;

typedef struct _hvr_buffered_changes_t {
    hvr_timestep_changes_t *timestep_list;
} hvr_buffered_changes_t;

void hvr_buffered_changes_init(hvr_buffered_changes_t *changes);

void hvr_buffered_changes_add(hvr_change_t *change,
        hvr_buffered_changes_t *changes);

hvr_timestep_changes_t *hvr_buffered_changes_remove_any(hvr_time_t max_timestep,
        hvr_buffered_changes_t *changes);

void hvr_buffered_changes_free(hvr_timestep_changes_t *changes);

void hvr_vertex_from_change(hvr_change_t *change, hvr_sparse_vec_t *vertex,
        hvr_internal_ctx_t *ctx);

#endif // _HVR_CHANGE_BUFFER_H
