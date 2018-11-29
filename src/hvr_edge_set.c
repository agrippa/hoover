#include <stdio.h>
#include <shmem.h>

#include "hvr_edge_set.h"

void hvr_edge_set_init(hvr_edge_set_t *e) {
    hvr_map_init(&e->map, 8, EDGE_INFO);
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
    val.edge_info = construct_edge_info(global_vertex_id, direction);

    hvr_map_add(local_vertex_id, val, &set->map);
}

void hvr_remove_edge(const hvr_vertex_id_t local_vertex_id,
        const hvr_vertex_id_t global_vertex_id, hvr_edge_set_t *set) {
    hvr_map_val_t val = {.edge_info = construct_edge_info(global_vertex_id, BIDIRECTIONAL)};
    hvr_map_remove(local_vertex_id, val, &set->map);
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
