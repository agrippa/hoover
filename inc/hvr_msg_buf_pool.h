#ifndef _HVR_MSG_BUF_POOL_H
#define _HVR_MSG_BUF_POOL_H

#include <stdlib.h>

typedef struct _hvr_msg_buf_node_t {
    void *ptr;
    size_t buf_size;
    struct _hvr_msg_buf_node_t *next;
} hvr_msg_buf_node_t;

typedef struct _hvr_msg_buf_pool_t {
    hvr_msg_buf_node_t *head;

    void *buf_mem;
    hvr_msg_buf_node_t *node_mem;

    size_t buf_size;
    size_t pool_size;
} hvr_msg_buf_pool_t;

void hvr_msg_buf_pool_init(hvr_msg_buf_pool_t *pool, size_t buf_size,
        size_t pool_size);

hvr_msg_buf_node_t *hvr_msg_buf_pool_acquire(hvr_msg_buf_pool_t *pool);

void hvr_msg_buf_pool_release(hvr_msg_buf_node_t *node,
        hvr_msg_buf_pool_t *pool);

void hvr_msg_buf_pool_destroy(hvr_msg_buf_pool_t *pool);

size_t hvr_msg_buf_pool_mem_used(hvr_msg_buf_pool_t *pool);

#endif
