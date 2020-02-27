#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>

#include "hvr_avl_tree.h"

struct hvr_avl_node dummy = { {&dummy, &dummy}, 0, 0, 0 }, *nnil = &dummy;
 
static struct hvr_avl_node *new_node(uint32_t key, uint64_t value,
        hvr_avl_node_allocator *tracker) {
    struct hvr_avl_node *n = hvr_avl_node_allocator_alloc(tracker);
    n->kid[0] = nnil;
    n->kid[1] = nnil;
    n->value = value;
    n->key = key;
    n->height = 1;
	return n;
}
 
static inline int16_t max(int16_t a, int16_t b) { return a > b ? a : b; }

static inline void set_height(struct hvr_avl_node *n) {
	n->height = 1 + max(n->kid[0]->height, n->kid[1]->height);
}
 
static inline int balance(struct hvr_avl_node *n) {
	return n->kid[0]->height - n->kid[1]->height;
}
 
// rotate a subtree according to dir; if new root is nil, old root is freed
static struct hvr_avl_node * rotate(struct hvr_avl_node **rootp, int dir,
        hvr_avl_node_allocator *tracker)
{
	struct hvr_avl_node *old_r = *rootp, *new_r = old_r->kid[dir];
 
	if (nnil == (*rootp = new_r))
        hvr_avl_node_allocator_free(old_r, tracker);
	else {
		old_r->kid[dir] = new_r->kid[!dir];
		set_height(old_r);
		new_r->kid[!dir] = old_r;
	}
	return new_r;
}
 
static void adjust_balance(struct hvr_avl_node **rootp,
        hvr_avl_node_allocator *tracker)
{
	struct hvr_avl_node *root = *rootp;
	int b = balance(root)/2;
	if (b) {
		int dir = (1 - b)/2;
		if (balance(root->kid[dir]) == -b) {
			rotate(&root->kid[dir], !dir, tracker);
        }
		root = rotate(rootp, dir, tracker);
	}
	if (root != nnil) set_height(root);
}
 
// find the node that contains value as payload; or returns 0
static struct hvr_avl_node *query(struct hvr_avl_node *root, uint32_t key)
{
	return root == nnil
		? 0
		: root->key == key
			? root
			: query(root->kid[key > root->key], key);
}
 
int hvr_avl_insert(struct hvr_avl_node **rootp, uint32_t key, uint64_t value,
        hvr_avl_node_allocator *tracker) {
    int result;
	struct hvr_avl_node *root = *rootp;

	if (root == nnil) {
		*rootp = new_node(key, value, tracker);
        result = 1;
    } else if (key != root->key) { // don't allow dup keys
		result = hvr_avl_insert(&root->kid[key > root->key], key, value,
                tracker);
		adjust_balance(rootp, tracker);
	} else {
        // Duplicate key
        result = 0;
    }
    return result;
}
 
int hvr_avl_delete(struct hvr_avl_node **rootp, uint32_t key,
        hvr_avl_node_allocator *tracker)
{
	struct hvr_avl_node *root = *rootp;
	if (root == nnil) return 0; // not found
 
	// if this is the node we want, rotate until off the tree
	if (root->key == key)
		if (nnil == (root = rotate(rootp, balance(root) < 0, tracker)))
			return 1;
 
	int success = hvr_avl_delete(&root->kid[key > root->key], key,
            tracker);
	adjust_balance(rootp, tracker);
    return success;
}

void hvr_avl_delete_all(struct hvr_avl_node *root,
        hvr_avl_node_allocator *tracker) {
    if (root == nnil) return;

    hvr_avl_delete_all(root->kid[0], tracker);
    hvr_avl_delete_all(root->kid[1], tracker);
    hvr_avl_node_allocator_free(root, tracker);
}

static void hvr_avl_serialize_helper(struct hvr_avl_node *root,
        uint64_t *values, int arr_capacity, int *index) {
    if (root == nnil) return;

    assert(*index < arr_capacity);
    values[*index] = root->value;
    *index += 1;

    hvr_avl_serialize_helper(root->kid[0], values, arr_capacity, index);
    hvr_avl_serialize_helper(root->kid[1], values, arr_capacity, index);
}

int hvr_avl_serialize(struct hvr_avl_node *root,
        uint64_t *values, int arr_capacity) {
    int index = 0;
    hvr_avl_serialize_helper(root, values, arr_capacity, &index);
    return index;
}

struct hvr_avl_node *hvr_avl_find(const struct hvr_avl_node *root,
        const uint32_t key) {
    while (root != nnil && root->key != key) {
        root = hvr_avl_find(root->kid[key > root->key], key);
    }
    return (struct hvr_avl_node *)root;
/*
    if (root == nnil) {
        return nnil;
    } else if (root->key == key) {
        return (struct hvr_avl_node *)root;
    } else {
        return hvr_avl_find(root->kid[key > root->key], key);
    }
    */
}

static void hvr_avl_size_helper(struct hvr_avl_node *curr, unsigned *counter) {
    if (curr != nnil) {
        *counter += 1;
        hvr_avl_size_helper(curr->kid[0], counter);
        hvr_avl_size_helper(curr->kid[1], counter);
    }
}

unsigned hvr_avl_size(struct hvr_avl_node *root) {
    unsigned counter = 0;
    hvr_avl_size_helper(root, &counter);
    return counter;
}

void hvr_avl_node_allocator_init(hvr_avl_node_allocator *allocator,
        size_t pool_size, const char *envvar) {
    struct hvr_avl_node *pool = (struct hvr_avl_node *)malloc_helper(
            pool_size * sizeof(*pool));
    assert(pool);

    allocator->head = pool;
    for (size_t i = 0; i < pool_size - 1; i++) {
        pool[i].kid[0] = &pool[i + 1];
    }
    pool[pool_size - 1].kid[0] = NULL;

    allocator->mem = pool;
    allocator->pool_size = pool_size;
    allocator->n_reserved = 0;

    allocator->envvar = (char *)malloc(strlen(envvar) + 1);
    memcpy(allocator->envvar, envvar, strlen(envvar) + 1);
}

struct hvr_avl_node *hvr_avl_node_allocator_alloc(
        hvr_avl_node_allocator *allocator) {
    struct hvr_avl_node *result = allocator->head;
    if (!result) {
        fprintf(stderr, "ERROR failed allocating AVL node. Increase %s "
                "(%lu).\n", allocator->envvar, allocator->pool_size);
        abort();
    }
    allocator->head = result->kid[0];
    allocator->n_reserved += 1;
    return result;
}

void hvr_avl_node_allocator_free(struct hvr_avl_node *node,
        hvr_avl_node_allocator *allocator) {
    node->kid[0] = allocator->head;
    allocator->head = node;
    allocator->n_reserved -= 1;
}

void hvr_avl_node_allocator_bytes_usage(hvr_avl_node_allocator *allocator,
        size_t *out_allocated, size_t *out_used) {
    *out_used = allocator->n_reserved * sizeof(struct hvr_avl_node);
    *out_allocated = allocator->pool_size * sizeof(struct hvr_avl_node);
}
