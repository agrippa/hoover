#include <assert.h>

#include "hvr_msg_buf_pool.h"

void hvr_msg_buf_pool_init(hvr_msg_buf_pool_t *pool, size_t buf_size,
        size_t pool_size) {
    pool->buf_mem = malloc(pool_size * buf_size);
    assert(pool->buf_mem);
    pool->node_mem = (hvr_msg_buf_node_t *)malloc(
            pool_size * sizeof(pool->node_mem[0]));
    assert(pool->node_mem);
    pool->buf_size = buf_size;
    pool->pool_size = pool_size;

    for (size_t i = 0; i < pool_size; i++) {
        hvr_msg_buf_node_t *curr = pool->node_mem + i;

        curr->ptr = ((char *)pool->buf_mem) + (i * buf_size);
        curr->buf_size = buf_size;
        if (i < pool_size - 1) {
            curr->next = pool->node_mem + (i + 1);
        } else {
            curr->next = NULL;
        }
    }

    pool->head = pool->node_mem;
}

hvr_msg_buf_node_t *hvr_msg_buf_pool_acquire(hvr_msg_buf_pool_t *pool) {
    hvr_msg_buf_node_t *result = pool->head;
    assert(result);
    pool->head = result->next;
    result->next = NULL;
    return result;
}

void hvr_msg_buf_pool_release(hvr_msg_buf_node_t *node,
        hvr_msg_buf_pool_t *pool) {
    assert(node->next == NULL);
    node->next = pool->head;
    pool->head = node;
}

void hvr_msg_buf_pool_destroy(hvr_msg_buf_pool_t *pool) {
    free(pool->buf_mem);
    free(pool->node_mem);
}

size_t hvr_msg_buf_pool_mem_used(hvr_msg_buf_pool_t *pool) {
    return pool->pool_size * pool->buf_size +
        pool->pool_size * sizeof(pool->node_mem[0]);
}
