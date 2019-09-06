#ifndef _HVR_BUFFERED_CHANGES_H
#define _HVR_BUFFERED_CHANGES_H

#include "hvr_common.h"

typedef struct _hvr_buffered_edge_create_t {
    hvr_vertex_id_t base_id;
    hvr_vertex_id_t neighbor_id;
    hvr_edge_type_t edge;
} hvr_buffered_edge_create_t;

typedef struct _hvr_buffered_vertex_delete_t {
    hvr_vertex_id_t to_delete;
} hvr_buffered_vertex_delete_t;

typedef struct _hvr_buffered_change_t {
    int is_edge_create;
    union {
        hvr_buffered_edge_create_t edge;
        hvr_buffered_vertex_delete_t del;
    } change;
    struct _hvr_buffered_change_t *next;
} hvr_buffered_change_t;

typedef struct _hvr_buffered_changes_t {
    hvr_buffered_change_t *head;
    hvr_buffered_change_t *pool;
    hvr_buffered_change_t *pool_mem;
    size_t nallocated;
} hvr_buffered_changes_t;

void hvr_buffered_changes_init(size_t nallocated,
        hvr_buffered_changes_t *changes);

void hvr_buffered_changes_edge_create(hvr_vertex_id_t base_id,
        hvr_vertex_id_t neighbor_id, hvr_edge_type_t edge,
        hvr_buffered_changes_t *changes);

void hvr_buffered_changes_delete_vertex(hvr_vertex_id_t to_delete,
        hvr_buffered_changes_t *changes);

int hvr_buffered_changes_poll(hvr_buffered_changes_t *changes,
        hvr_buffered_change_t *out_change);

void hvr_buffered_changes_destroy(hvr_buffered_changes_t *changes);

#endif
