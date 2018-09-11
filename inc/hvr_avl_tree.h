/* For license: see LICENSE.txt file at top-level */

#ifndef _HOOVER_AVL_TREE_H
#define _HOOVER_AVL_TREE_H

#include "hvr_common.h"

// An AVL tree node
typedef struct _hvr_avl_tree_node_t {
    hvr_vertex_id_t key;
    struct _hvr_avl_tree_node_t *subtree;

    hvr_vertex_id_t *linearized;
    unsigned linearized_length;

    struct _hvr_avl_tree_node_t *left;
    struct _hvr_avl_tree_node_t *right;
    int height;
} hvr_avl_tree_node_t;

extern hvr_avl_tree_node_t* hvr_tree_insert(hvr_avl_tree_node_t* node,
        hvr_vertex_id_t key);
extern hvr_avl_tree_node_t* hvr_tree_remove(hvr_avl_tree_node_t *curr,
        hvr_vertex_id_t key);
extern hvr_avl_tree_node_t *hvr_tree_find(hvr_avl_tree_node_t *curr,
        hvr_vertex_id_t key);
extern size_t hvr_tree_size(hvr_avl_tree_node_t *curr);
extern size_t hvr_tree_linearize(hvr_vertex_id_t **arr,
        hvr_avl_tree_node_t *curr);
extern void hvr_tree_destroy(hvr_avl_tree_node_t *curr);

#endif
