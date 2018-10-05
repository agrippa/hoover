#ifndef _HVR_VERTEX_LL_H
#define _HVR_VERTEX_LL_H

#include "hvr_vertex.h"

typedef struct _hvr_vertex_ll_node_t {
    hvr_vertex_t *vert;
    struct _hvr_vertex_ll_node_t *next;
} hvr_vertex_ll_node_t;

typedef struct _hvr_vertex_ll_t {
    hvr_vertex_ll_node_t *pool;
    hvr_vertex_ll_node_t *head;
    size_t capacity;
    size_t length;
    hvr_vertex_ll_node_t *allocated;
} hvr_vertex_ll_t;

void hvr_vertex_ll_init(hvr_vertex_ll_t *l, size_t capacity);
void hvr_vertex_ll_destroy(hvr_vertex_ll_t *l);

void hvr_vertex_ll_push(hvr_vertex_t *vert, hvr_vertex_ll_t *l);
hvr_vertex_t *hvr_vertex_ll_pop(hvr_vertex_ll_t *l);

#endif
