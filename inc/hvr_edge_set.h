/* For license: see LICENSE.txt file at top-level */

#ifndef _HVR_EDGE_SET_H
#define _HVR_EDGE_SET_H

#include "hvr_avl_tree.h"
#include "hvr_map.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Edge set utilities.
 *
 * An edge set is simply a space efficient data structure for storing edges
 * between vertices in the graph.
 */
typedef struct _hvr_edge_set_t {
    hvr_map_t map;
} hvr_edge_set_t;

extern hvr_edge_set_t *hvr_create_empty_edge_set();

extern void hvr_add_edge(const hvr_vertex_id_t local_vertex_id,
        const hvr_vertex_id_t global_vertex_id, hvr_edge_type_t direction,
        hvr_edge_set_t *set);

extern void hvr_remove_edge(const hvr_vertex_id_t local_vertex_id,
        const hvr_vertex_id_t global_vertex_id, hvr_edge_set_t *set);

extern hvr_edge_type_t hvr_have_edge(const hvr_vertex_id_t local_vertex_id,
        const hvr_vertex_id_t global_vertex_id, hvr_edge_set_t *set);

extern size_t hvr_count_edges(const hvr_vertex_id_t local_vertex_id,
        hvr_edge_set_t *set);

extern void hvr_clear_edge_set(hvr_edge_set_t *set);
extern void hvr_release_edge_set(hvr_edge_set_t *set);
extern void hvr_print_edge_set(hvr_edge_set_t *set);

typedef struct _hvr_partition_list_node_t {
    hvr_partition_t part;
    struct _hvr_partition_list_node_t *next;
} hvr_partition_list_node_t;

#ifdef __cplusplus
}
#endif

#endif // _HVR_EDGE_SET_H
