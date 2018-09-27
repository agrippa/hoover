/* For license: see LICENSE.txt file at top-level */

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "hvr_avl_tree.h"

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define INIT_CAPACITY 16

/* Helper function that allocates a new node with the given key and
    NULL left and right pointers. */
static hvr_avl_tree_node_t *newNode(hvr_vertex_id_t key) {
    hvr_avl_tree_node_t* node = (hvr_avl_tree_node_t*)malloc(
            sizeof(hvr_avl_tree_node_t));
    assert(node);
    node->key = key;
    node->linearized_length = 0;
    node->linearized_capacity = INIT_CAPACITY;

    node->linearized = (hvr_vertex_id_t *)malloc(INIT_CAPACITY * sizeof(hvr_vertex_id_t));
    assert(node->linearized);

    node->linearized_edges = (hvr_edge_type_t *)malloc(INIT_CAPACITY * sizeof(hvr_edge_type_t));
    assert(node->linearized_edges);

    node->left = NULL;
    node->right = NULL;
    node->height = 1;  // new node is initially added at leaf
    return(node);
}

// A utility function to get height of the tree
static int height(hvr_avl_tree_node_t *N)
{
    if (N == NULL) {
        return 0;
    }
    return N->height;
}
 
hvr_avl_tree_node_t* hvr_tree_insert(hvr_avl_tree_node_t* node,
        hvr_vertex_id_t key) {
    /* 1.  Perform the normal BST rotation */
    if (node == NULL)
        return(newNode(key));

    if (key < node->key) {
        node->left  = hvr_tree_insert(node->left, key);
    } else if (key > node->key) {
        node->right = hvr_tree_insert(node->right, key);
    } else { // Equal keys not allowed
        return node;
    }

    /* 2. Update height of this ancestor node */
    node->height = 1 + MAX(height(node->left),
            height(node->right));

    /* return the (unchanged) node pointer */
    return node;
}
 
/* Given a non-empty binary search tree, return the
      node with minimum key value found in that tree.
         Note that the entire tree does not need to be
            searched. */
hvr_avl_tree_node_t * minValueNode(hvr_avl_tree_node_t* node)
{
    hvr_avl_tree_node_t* current = node;

    /* loop down to find the leftmost leaf */
    while (current->left != NULL)
        current = current->left;

    return current;
}
 
static hvr_avl_tree_node_t *hvr_tree_find_helper(hvr_avl_tree_node_t *curr,
        hvr_vertex_id_t key) {
    if (curr == NULL) {
        return NULL;
    }

    if (key < curr->key) {
        return hvr_tree_find_helper(curr->left, key);
    } else if (key > curr->key) {
        return hvr_tree_find_helper(curr->right, key);
    } else {
        assert(key == curr->key);
        return curr;
    }
}

hvr_avl_tree_node_t *hvr_tree_find(hvr_avl_tree_node_t *curr,
        hvr_vertex_id_t key) {
    return hvr_tree_find_helper(curr, key);
}

void hvr_tree_destroy(hvr_avl_tree_node_t *curr) {
    if (curr == NULL) {
        return;
    }

    hvr_tree_destroy(curr->left);
    hvr_tree_destroy(curr->right);
    if (curr->linearized) {
        free(curr->linearized);
        free(curr->linearized_edges);
    }
    free(curr);
}

size_t hvr_tree_linearize(hvr_vertex_id_t **arr, hvr_edge_type_t **directions,
        hvr_avl_tree_node_t *curr) {
    *arr = curr->linearized;
    *directions = curr->linearized_edges;

    return curr->linearized_length;
}

static int find_in_node(hvr_vertex_id_t vert, hvr_avl_tree_node_t *curr) {
    for (unsigned i = 0; i < curr->linearized_length; i++) {
        if ((curr->linearized)[i] == vert) {
            return i;
        }
    }
    return -1;
}

void hvr_tree_append_to_node(hvr_vertex_id_t vert, hvr_edge_type_t dir,
        hvr_avl_tree_node_t *curr) {
    int found = find_in_node(vert, curr);
    if (found >= 0) {
        assert(curr->linearized_edges[found] == dir);
        return;
    }

    if (curr->linearized_length == curr->linearized_capacity) {
        curr->linearized_capacity *= 2;
        curr->linearized = (hvr_vertex_id_t *)realloc(curr->linearized,
                curr->linearized_capacity * sizeof(hvr_vertex_id_t));
        assert(curr->linearized);
        curr->linearized_edges = (hvr_edge_type_t *)realloc(
                curr->linearized_edges,
                curr->linearized_capacity * sizeof(hvr_edge_type_t));
        assert(curr->linearized_edges);
    }

    (curr->linearized)[curr->linearized_length] = vert;
    (curr->linearized_edges)[curr->linearized_length] = dir;
    curr->linearized_length += 1;
}

void hvr_tree_remove_from_node(hvr_vertex_id_t vert, hvr_avl_tree_node_t *curr) {
    int found = find_in_node(vert, curr);
    assert(found >= 0);

    (curr->linearized)[found] = (curr->linearized)[curr->linearized_length - 1];
    (curr->linearized_edges)[found] = (curr->linearized_edges)[
        curr->linearized_length - 1];
    curr->linearized_length -= 1;
}

hvr_edge_type_t hvr_tree_lookup_in_node(hvr_vertex_id_t vert,
        hvr_avl_tree_node_t *curr) {
    int found = find_in_node(vert, curr);
    if (found >= 0) {
        return curr->linearized_edges[found];
    } else {
        return NO_EDGE;
    }
}
