/* For license: see LICENSE.txt file at top-level */

#ifndef _HOOVER_AVL_TREE_H
#define _HOOVER_AVL_TREE_H

#include "hvr_common.h"

// An AVL tree node
typedef struct _hvr_avl_tree_node_t {
    hvr_vertex_id_t key;

    hvr_vertex_id_t *linearized;
    hvr_edge_type_t *linearized_edges;
    unsigned linearized_length;
    unsigned linearized_capacity;

    struct _hvr_avl_tree_node_t *left;
    struct _hvr_avl_tree_node_t *right;
    int height;
} hvr_avl_tree_node_t;

extern hvr_avl_tree_node_t* hvr_tree_insert(hvr_avl_tree_node_t* node,
        hvr_vertex_id_t key);

extern hvr_avl_tree_node_t *hvr_tree_find(hvr_avl_tree_node_t *curr,
        hvr_vertex_id_t key);

extern size_t hvr_tree_linearize(hvr_vertex_id_t **vertices,
        hvr_edge_type_t **directions, unsigned *capacity,
        hvr_avl_tree_node_t *curr);

extern void hvr_tree_destroy(hvr_avl_tree_node_t *curr);

extern void hvr_tree_remove_from_node(hvr_vertex_id_t vert,
        hvr_avl_tree_node_t *curr);

extern void hvr_tree_append_to_node(hvr_vertex_id_t vert, hvr_edge_type_t dir,
        hvr_avl_tree_node_t *curr);

extern hvr_edge_type_t hvr_tree_lookup_in_node(hvr_vertex_id_t vert,
        hvr_avl_tree_node_t *curr);

#endif
