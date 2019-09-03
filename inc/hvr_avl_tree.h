#ifndef _HVR_AVL_TREE_H
#define _HVR_AVL_TREE_H

#include "dlmalloc.h"

struct node {
	int payload;
	int height;
	struct node * kid[2];
};

extern void hvr_avl_insert(struct node **rootp, int value, mspace tracker);

extern int hvr_avl_delete(struct node **rootp, int value, mspace tracker);

extern void hvr_avl_delete_all(struct node *rootp, mspace tracker);

extern struct node *hvr_avl_find(struct node *root, int target);

extern void hvr_avl_serialize(struct node *root, int *arr, int arr_capacity);

extern struct node *nnil;

#endif
