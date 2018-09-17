#include <stdio.h>
#include <shmem.h>

#include "hvr_edge_set.h"

static inline int valid_vertex_id(const hvr_vertex_id_t id) {
    return id != HVR_INVALID_VERTEX_ID &&
                 VERTEX_ID_PE(id) < shmem_n_pes() &&
                 VERTEX_ID_OFFSET(id) < get_symm_pool_nelements();
}

hvr_edge_set_t *hvr_create_empty_edge_set() {
    hvr_edge_set_t *new_set = (hvr_edge_set_t *)malloc(sizeof(*new_set));
    assert(new_set);
    new_set->tree = NULL;
    return new_set;
}

/*
 * Add an edge from a local vertex local_vertex_id to another vertex (possibly
 * local or remote) global_vertex_id.
 */
void hvr_add_edge(const hvr_vertex_id_t local_vertex_id,
        const hvr_vertex_id_t global_vertex_id, hvr_edge_type_t direction,
        hvr_edge_set_t *set) {
    // If it already exists, just returns existing node in tree
    assert(valid_vertex_id(local_vertex_id));
    assert(valid_vertex_id(global_vertex_id));

    set->tree = hvr_tree_insert(set->tree, local_vertex_id);
    hvr_avl_tree_node_t *inserted = hvr_tree_find(set->tree, local_vertex_id);
    inserted->subtree = hvr_tree_insert(inserted->subtree, global_vertex_id,
            direction);
}

void hvr_remove_edge(const hvr_vertex_id_t local_vertex_id,
        const hvr_vertex_id_t global_vertex_id, hvr_edge_set_t *set) {
    assert(valid_vertex_id(local_vertex_id));
    assert(valid_vertex_id(global_vertex_id));

    hvr_avl_tree_node_t *found = hvr_tree_find(set->tree, local_vertex_id);
    if (found) {
        found->subtree = hvr_tree_remove(found->subtree, global_vertex_id);
    }
}

hvr_edge_type_t hvr_have_edge(const hvr_vertex_id_t local_vertex_id,
        const hvr_vertex_id_t global_vertex_id, hvr_edge_set_t *set) {
    assert(valid_vertex_id(local_vertex_id));
    assert(valid_vertex_id(global_vertex_id));

    hvr_avl_tree_node_t *inserted = hvr_tree_find(set->tree, local_vertex_id);
    if (inserted == NULL) {
        return 0;
    }

    hvr_avl_tree_node_t *found = hvr_tree_find(inserted->subtree,
            global_vertex_id);
    if (found) {
        return found->direction;
    } else {
        return NO_EDGE;
    }
}

size_t hvr_count_edges(const hvr_vertex_id_t local_vertex_id,
        hvr_edge_set_t *set) {
    assert(valid_vertex_id(local_vertex_id));

    hvr_avl_tree_node_t *found = hvr_tree_find(set->tree, local_vertex_id);
    if (found == NULL) return 0;
    else return hvr_tree_size(found->subtree);
}

void hvr_clear_edge_set(hvr_edge_set_t *set) {
    hvr_tree_destroy(set->tree);
    set->tree = NULL;
}

void hvr_release_edge_set(hvr_edge_set_t *set) {
    hvr_tree_destroy(set->tree);
    free(set);
}

static void hvr_print_edge_set_helper(hvr_avl_tree_node_t *tree,
        const int print_colon) {
    if (tree == NULL) {
        return;
    }
    hvr_print_edge_set_helper(tree->left, print_colon);
    hvr_print_edge_set_helper(tree->right, print_colon);
    if (print_colon) {
        printf("%lu: ", tree->key);
        hvr_print_edge_set_helper(tree->subtree, 0);
        printf("\n");
    } else {
        printf("%lu ", tree->key);
    }
}

void hvr_print_edge_set(hvr_edge_set_t *set) {
    if (set->tree == NULL) {
        printf("Empty set\n");
    } else {
        hvr_print_edge_set_helper(set->tree, 1);
    }
}


void hvr_update_edge_type(hvr_vertex_id_t a, hvr_vertex_id_t b,
        hvr_edge_type_t edge, hvr_edge_set_t *set) {
    hvr_avl_tree_node_t *base = hvr_tree_find(set->tree, a);
    assert(base);
    hvr_avl_tree_node_t *leaf = hvr_tree_find(base->subtree, b);
    assert(leaf);
    leaf->direction = edge;
}

