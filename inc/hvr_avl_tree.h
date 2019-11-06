#ifndef _HVR_AVL_TREE_H
#define _HVR_AVL_TREE_H

#include "dlmalloc.h"

struct hvr_avl_node {
	int payload;
	int height;
	struct hvr_avl_node * kid[2];
};

extern void hvr_avl_insert(struct hvr_avl_node **rootp, int value,
        mspace tracker);

extern int hvr_avl_delete(struct hvr_avl_node **rootp, int value,
        mspace tracker);

extern void hvr_avl_delete_all(struct hvr_avl_node *rootp, mspace tracker);

extern struct hvr_avl_node *hvr_avl_find(struct hvr_avl_node *root, int target);

extern void hvr_avl_serialize(struct hvr_avl_node *root, int *arr,
        int arr_capacity);

extern struct hvr_avl_node *nnil;

#endif
