#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hvr_irregular_matrix.h"

void hvr_irr_matrix_init(size_t nvertices, size_t pool_size,
        hvr_irr_matrix_t *m) {
    m->edges = (struct hvr_avl_node **)malloc_helper(
            nvertices * sizeof(m->edges[0]));
    assert(m->edges);
    for (size_t i = 0; i < nvertices; i++) {
        m->edges[i] = nnil;
    }

    m->nvertices = nvertices;
    m->nedges = 0;

    hvr_avl_node_allocator_init(&m->allocator, pool_size,
            "HVR_EDGES_POOL_SIZE");
}

void hvr_irr_matrix_get(const hvr_vertex_id_t i,
        const hvr_vertex_id_t j, const hvr_irr_matrix_t *m,
        hvr_edge_type_t *out_edge_type,
        hvr_edge_create_type_t *out_creation_type) {
    struct hvr_avl_node *root = m->edges[i];
    struct hvr_avl_node *found = hvr_avl_find(root, j);
    if (found != nnil) {
        *out_edge_type = EDGE_INFO_EDGE(found->value);
        *out_creation_type = EDGE_INFO_CREATION(found->value);
    } else {
        *out_edge_type = NO_EDGE;
    }
}

void hvr_irr_matrix_set(hvr_vertex_id_t i, hvr_vertex_id_t j, hvr_edge_type_t e,
        hvr_edge_create_type_t create_type, hvr_irr_matrix_t *m,
        int known_no_edge) {
    struct hvr_avl_node *root = m->edges[i];
    struct hvr_avl_node *found = hvr_avl_find(root, j);
    if (found == nnil) {
        if (e == NO_EDGE) return;

        hvr_avl_insert(&(m->edges[i]), j,
                construct_edge_info(j, e, create_type), &m->allocator);
        m->nedges += 1;
    } else {
        if (e == NO_EDGE) {
            hvr_avl_delete(&(m->edges[i]), j, &m->allocator);
            m->nedges -= 1;
        } else {
            found->value = construct_edge_info(j, e, create_type);
        }
    }
}

unsigned hvr_irr_matrix_row_len(hvr_vertex_id_t i, hvr_irr_matrix_t *m) {
    return hvr_avl_size(m->edges[i]);
}

unsigned hvr_irr_matrix_linearize(hvr_vertex_id_t i,
        hvr_vertex_id_t *out_vals, size_t capacity, hvr_irr_matrix_t *m) {
    return hvr_avl_serialize(m->edges[i], out_vals, capacity);
}

void hvr_irr_matrix_usage(size_t *out_bytes_allocated, size_t *out_bytes_used,
        size_t *out_max_edges, size_t *out_max_edges_index,
        hvr_irr_matrix_t *m) {
    size_t allocator_used, allocator_allocated;
    hvr_avl_node_allocator_bytes_usage(&m->allocator, &allocator_allocated,
            &allocator_used);

    *out_bytes_allocated = m->nvertices * sizeof(m->edges[0]) +
        allocator_allocated;
    *out_bytes_used = m->nvertices * sizeof(m->edges[0]) + allocator_used;

    size_t max_edges = 0;
    size_t max_edges_index = 0;

    for (size_t i = 0; i < m->nvertices; i++) {
        unsigned row_len = hvr_irr_matrix_row_len(i, m);
        if (row_len > max_edges) {
            max_edges = row_len;
            max_edges_index = i;
        }
    }

    *out_max_edges = max_edges;
    *out_max_edges_index = max_edges_index;
}
