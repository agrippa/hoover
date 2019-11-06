#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include "hvr_avl_tree.h"
 
struct hvr_avl_node dummy = { 0, 0, {&dummy, &dummy} }, *nnil = &dummy;
// internally, nnil is the new nul
 
static struct hvr_avl_node *new_node(int value, mspace tracker)
{
    struct hvr_avl_node *n = (struct hvr_avl_node *)mspace_malloc(tracker,
            sizeof(*n));
    if (!n) {
        fprintf(stderr, "ERROR Failed allocating an AVL node. Consider "
                "increasing HVR_SPARSE_ARR_POOL.\n");
        abort();
    }
	*n = (struct hvr_avl_node) { value, 1, {nnil, nnil} };
	return n;
}
 
static inline int max(int a, int b) { return a > b ? a : b; }

static inline void set_height(struct hvr_avl_node *n) {
	n->height = 1 + max(n->kid[0]->height, n->kid[1]->height);
}
 
static inline int balance(struct hvr_avl_node *n) {
	return n->kid[0]->height - n->kid[1]->height;
}
 
// rotate a subtree according to dir; if new root is nil, old root is freed
static struct hvr_avl_node * rotate(struct hvr_avl_node **rootp, int dir,
        mspace tracker)
{
	struct hvr_avl_node *old_r = *rootp, *new_r = old_r->kid[dir];
 
	if (nnil == (*rootp = new_r))
		mspace_free(tracker, old_r);
	else {
		old_r->kid[dir] = new_r->kid[!dir];
		set_height(old_r);
		new_r->kid[!dir] = old_r;
	}
	return new_r;
}
 
static void adjust_balance(struct hvr_avl_node **rootp, mspace tracker)
{
	struct hvr_avl_node *root = *rootp;
	int b = balance(root)/2;
	if (b) {
		int dir = (1 - b)/2;
		if (balance(root->kid[dir]) == -b)
			rotate(&root->kid[dir], !dir, tracker);
		root = rotate(rootp, dir, tracker);
	}
	if (root != nnil) set_height(root);
}
 
// find the node that contains value as payload; or returns 0
static struct hvr_avl_node *query(struct hvr_avl_node *root, int value)
{
	return root == nnil
		? 0
		: root->payload == value
			? root
			: query(root->kid[value > root->payload], value);
}
 
void hvr_avl_insert(struct hvr_avl_node **rootp, int value, mspace tracker)
{
	struct hvr_avl_node *root = *rootp;
 
	if (root == nnil)
		*rootp = new_node(value, tracker);
	else if (value != root->payload) { // don't allow dup keys
		hvr_avl_insert(&root->kid[value > root->payload], value, tracker);
		adjust_balance(rootp, tracker);
	}
}
 
int hvr_avl_delete(struct hvr_avl_node **rootp, int value, mspace tracker)
{
	struct hvr_avl_node *root = *rootp;
	if (root == nnil) return 0; // not found
 
	// if this is the node we want, rotate until off the tree
	if (root->payload == value)
		if (nnil == (root = rotate(rootp, balance(root) < 0, tracker)))
			return 1;
 
	int success = hvr_avl_delete(&root->kid[value > root->payload], value,
            tracker);
	adjust_balance(rootp, tracker);
    return success;
}

void hvr_avl_delete_all(struct hvr_avl_node *root, mspace tracker) {
    if (root == nnil) return;

    hvr_avl_delete_all(root->kid[0], tracker);
    hvr_avl_delete_all(root->kid[1], tracker);
    mspace_free(tracker, root);
}

static void hvr_avl_serialize_helper(struct hvr_avl_node *root, int *arr,
        int arr_capacity, int *index) {
    if (root == nnil) return;

    assert(*index < arr_capacity);
    arr[*index] = root->payload;
    *index += 1;

    hvr_avl_serialize_helper(root->kid[0], arr, arr_capacity, index);
    hvr_avl_serialize_helper(root->kid[1], arr, arr_capacity, index);
}

void hvr_avl_serialize(struct hvr_avl_node *root, int *arr, int arr_capacity) {
    int index = 0;
    hvr_avl_serialize_helper(root, arr, arr_capacity, &index);
}

struct hvr_avl_node *hvr_avl_find(struct hvr_avl_node *root, int target) {
    if (root == nnil) {
        return nnil;
    } else if (root->payload == target) {
        return root;
    } else {
        return hvr_avl_find(root->kid[target > root->payload], target);
    }
}
