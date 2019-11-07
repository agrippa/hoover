#ifndef _HVR_AVL_TREE_H
#define _HVR_AVL_TREE_H

#include <stdint.h>
#include "hvr_common.h"

struct hvr_avl_node {
	struct hvr_avl_node * kid[2];
    uint64_t value;
	uint32_t key;
	int16_t height;
};

extern struct hvr_avl_node *nnil;

typedef struct _hvr_avl_node_allocator {
    struct hvr_avl_node *head;
    struct hvr_avl_node *mem;
    size_t pool_size;
    char *envvar;
} hvr_avl_node_allocator;

void hvr_avl_insert(struct hvr_avl_node **rootp, uint32_t key,
        uint64_t value, hvr_avl_node_allocator *tracker);

int hvr_avl_delete(struct hvr_avl_node **rootp, uint32_t key,
        hvr_avl_node_allocator * tracker);

void hvr_avl_delete_all(struct hvr_avl_node *rootp,
        hvr_avl_node_allocator *tracker);

struct hvr_avl_node *hvr_avl_find(struct hvr_avl_node *root,
        uint32_t key);

int hvr_avl_serialize(struct hvr_avl_node *root, uint64_t *values,
        int arr_capacity);

unsigned hvr_avl_size(struct hvr_avl_node *root);

void hvr_avl_node_allocator_init(hvr_avl_node_allocator *allocator,
        size_t pool_size, const char *envvar);

struct hvr_avl_node *hvr_avl_node_allocator_alloc(
        hvr_avl_node_allocator *allocator);

void hvr_avl_node_allocator_free(struct hvr_avl_node *node,
        hvr_avl_node_allocator *allocator);

size_t hvr_avl_node_allocator_bytes_allocated(
        hvr_avl_node_allocator *allocator);

#endif
