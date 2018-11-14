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
    hvr_map_init(&new_set->map, 64);
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
    hvr_map_val_t val;
    val.edge_info.id = global_vertex_id;
    val.edge_info.edge = direction;
    hvr_map_add(local_vertex_id, val, EDGE_INFO, &set->map);
}

void hvr_remove_edge(const hvr_vertex_id_t local_vertex_id,
        const hvr_vertex_id_t global_vertex_id, hvr_edge_set_t *set) {
    hvr_map_val_t val = {.edge_info = {.id = global_vertex_id}};
    hvr_map_remove(local_vertex_id, val, EDGE_INFO, &set->map);
}

hvr_edge_type_t hvr_have_edge(const hvr_vertex_id_t local_vertex_id,
        const hvr_vertex_id_t global_vertex_id, hvr_edge_set_t *set) {
    return hvr_map_contains(local_vertex_id, global_vertex_id, &set->map);
}

size_t hvr_count_edges(const hvr_vertex_id_t local_vertex_id,
        hvr_edge_set_t *set) {
    return hvr_map_count_values(local_vertex_id, &set->map);
}

void hvr_clear_edge_set(hvr_edge_set_t *set) {
    hvr_map_clear(&set->map);
}

void hvr_release_edge_set(hvr_edge_set_t *set) {
    hvr_map_clear(&set->map);
    free(set);
}

void hvr_print_edge_set(hvr_edge_set_t *set) {
    printf("hvr_print_edge_set unsupported at the moment\n");
}
