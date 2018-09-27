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
    /*
     * If it already exists, just returns existing node in tree. Direction here
     * doesn't matter.
     */
    set->tree = hvr_tree_insert(set->tree, local_vertex_id);
    hvr_avl_tree_node_t *inserted = hvr_tree_find(set->tree, local_vertex_id);
    assert(inserted && inserted->key == local_vertex_id);

    hvr_tree_append_to_node(global_vertex_id, direction, inserted);
}

void hvr_remove_edge(const hvr_vertex_id_t local_vertex_id,
        const hvr_vertex_id_t global_vertex_id, hvr_edge_set_t *set) {
    hvr_avl_tree_node_t *found = hvr_tree_find(set->tree, local_vertex_id);
    assert(found);
    hvr_tree_remove_from_node(global_vertex_id, found);
}

hvr_edge_type_t hvr_have_edge(const hvr_vertex_id_t local_vertex_id,
        const hvr_vertex_id_t global_vertex_id, hvr_edge_set_t *set) {
    hvr_avl_tree_node_t *inserted = hvr_tree_find(set->tree, local_vertex_id);
    if (inserted == NULL) {
        return NO_EDGE;
    }

    return hvr_tree_lookup_in_node(global_vertex_id, inserted);
}

size_t hvr_count_edges(const hvr_vertex_id_t local_vertex_id,
        hvr_edge_set_t *set) {
    hvr_avl_tree_node_t *found = hvr_tree_find(set->tree, local_vertex_id);
    if (found == NULL) return 0;
    else return found->linearized_length;
}

void hvr_clear_edge_set(hvr_edge_set_t *set) {
    hvr_tree_destroy(set->tree);
    set->tree = NULL;
}

void hvr_release_edge_set(hvr_edge_set_t *set) {
    hvr_tree_destroy(set->tree);
    free(set);
}

void hvr_print_edge_set(hvr_edge_set_t *set) {
    if (set->tree == NULL) {
        printf("Empty set\n");
    } else {
        printf("hvr_print_edge_set unsupported at the moment\n");
        // hvr_print_edge_set_helper(set->tree, 1);
    }
}
