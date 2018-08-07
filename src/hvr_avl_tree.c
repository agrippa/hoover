/* For license: see LICENSE.txt file at top-level */

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "hvr_avl_tree.h"

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

 
// A utility function to get height of the tree
static inline int height(hvr_avl_tree_node_t *N) {
    if (N == NULL) {
        return 0;
    }
    return N->height;
}
 
/* Helper function that allocates a new node with the given key and
    NULL left and right pointers. */
static hvr_avl_tree_node_t *newNode(hvr_vertex_id_t key) {
    hvr_avl_tree_node_t* node = (hvr_avl_tree_node_t*)malloc(
            sizeof(hvr_avl_tree_node_t));
    assert(node);
    node->key = key;
    node->subtree = NULL;
    node->linearized = NULL;
    node->left = NULL;
    node->right = NULL;
    node->height = 1;  // new node is initially added at leaf
    return(node);
}

static hvr_avl_tree_node_t *rotateWithLeftChild(hvr_avl_tree_node_t *k2) {
    hvr_avl_tree_node_t *k1 = k2->left;
    k2->left = k1->right;
    k1->right = k2;
    k2->height = MAX(height(k2->left), height(k2->right)) + 1;
    k1->height = MAX(height(k1->left), k2->height) + 1;
    return k1;
}

static hvr_avl_tree_node_t *rotateWithRightChild(hvr_avl_tree_node_t *k1) {
    hvr_avl_tree_node_t *k2 = k1->right;
    k1->right = k2->left;
    k2->left = k1;
    k1->height = MAX(height(k1->left), height(k1->right)) + 1;
    k2->height = MAX(height(k2->right), k1->height) + 1;
    return k2;
}

static hvr_avl_tree_node_t *doubleWithLeftChild(hvr_avl_tree_node_t *k3) {
    k3->left = rotateWithRightChild(k3->left);
    return rotateWithLeftChild(k3);
}

static hvr_avl_tree_node_t *doubleWithRightChild(hvr_avl_tree_node_t *k1) {
    k1->right = rotateWithLeftChild(k1->right);
    return rotateWithRightChild(k1);
}

// Recursive function to insert key in subtree rooted
// with node and returns new root of subtree.
hvr_avl_tree_node_t* hvr_tree_insert(hvr_avl_tree_node_t* node,
        hvr_vertex_id_t key) {
    if (node == NULL) {
        node = newNode(key);
    } else if (key == node->key) {
        node = node;
    } else if (key < node->key) {
        node->left  = hvr_tree_insert(node->left, key);
        if (height(node->left) - height(node->right) == 2) {
            if (key < node->left->key) {
                node = rotateWithLeftChild(node);
            } else {
                node = doubleWithLeftChild(node);
            }
        }
    } else if (key > node->key) {
        node->right = hvr_tree_insert(node->right, key);
        if (height(node->right) - height(node->left) == 2) {
            if (key > node->right->key) {
                node = rotateWithRightChild(node);
            } else {
                node = doubleWithRightChild(node);
            }
        }
    } else {
        assert(0);
    }

    node->height = 1 + MAX(height(node->left),
                           height(node->right));
 
    return node;
}

hvr_avl_tree_node_t *hvr_tree_find(hvr_avl_tree_node_t *curr,
        hvr_vertex_id_t key) {
    if (curr == NULL) {
        return NULL;
    }

    if (key < curr->key) {
        return hvr_tree_find(curr->left, key);
    } else if (key > curr->key) {
        return hvr_tree_find(curr->right, key);
    } else {
        return curr;
    }
}

void hvr_tree_destroy(hvr_avl_tree_node_t *curr) {
    if (curr == NULL) {
        return;
    }

    hvr_tree_destroy(curr->left);
    hvr_tree_destroy(curr->right);
    hvr_tree_destroy(curr->subtree);
    if (curr->linearized) free(curr->linearized);
    free(curr);
}

size_t hvr_tree_size(hvr_avl_tree_node_t *curr) {
    if (curr == NULL) {
        return 0;
    }
    return 1 + hvr_tree_size(curr->left) + hvr_tree_size(curr->right);
}

static void hvr_tree_linearize_helper(hvr_vertex_id_t *arr, unsigned *index,
        hvr_avl_tree_node_t *curr) {
    if (curr == NULL) {
        return;
    }

    hvr_tree_linearize_helper(arr, index, curr->left);
    hvr_tree_linearize_helper(arr, index, curr->right);
    arr[*index] = curr->key;
    *index += 1;
}

size_t hvr_tree_linearize(hvr_vertex_id_t **arr, size_t *arr_capacity,
        hvr_avl_tree_node_t *curr) {
    if (curr->linearized == NULL) {
        const size_t tree_size = hvr_tree_size(curr);
        hvr_vertex_id_t *linearized = (hvr_vertex_id_t *)malloc(
                tree_size * sizeof(*linearized));
        assert(linearized);

        unsigned index = 0;
        hvr_tree_linearize_helper(linearized, &index, curr);
        assert(index == tree_size);

        curr->linearized = linearized;
        curr->linearized_length = tree_size;
    }

    if (*arr_capacity < curr->linearized_length) {
        *arr = (hvr_vertex_id_t *)realloc(*arr,
                curr->linearized_length * sizeof(hvr_vertex_id_t));
        assert(*arr);
        *arr_capacity = curr->linearized_length;
    }

    memcpy(*arr, curr->linearized,
            curr->linearized_length * sizeof(hvr_vertex_id_t));

    return curr->linearized_length;
}
