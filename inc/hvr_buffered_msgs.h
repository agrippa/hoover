#ifndef _HVR_BUFFERED_MSGS_H
#define _HVR_BUFFERED_MSGS_H

#include "hvr_vertex.h"
#ifdef __cplusplus
extern "C" {
#endif
#include "dlmalloc.h"
#ifdef __cplusplus
}
#endif

typedef struct _hvr_buffered_msgs_node_t {
    hvr_vertex_t vert;
    struct _hvr_buffered_msgs_node_t *next;
} hvr_buffered_msgs_node_t;

typedef struct _hvr_buffered_msgs_t {
    hvr_buffered_msgs_node_t **buffered;
    size_t nvertices;

    void *pool;
    size_t pool_size;
    mspace allocator;
} hvr_buffered_msgs_t;

void hvr_buffered_msgs_init(size_t nvertices, size_t pool_size,
        hvr_buffered_msgs_t *b);

void hvr_buffered_msgs_insert(size_t i, hvr_vertex_t *payload,
        hvr_buffered_msgs_t *b);

int hvr_buffered_msgs_poll(size_t i, hvr_vertex_t *out, hvr_buffered_msgs_t *b);

size_t hvr_buffered_msgs_mem_used(hvr_buffered_msgs_t *b);

#endif
