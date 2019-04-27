#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hvr_irregular_matrix.h"

void hvr_irr_matrix_init(size_t nvertices, size_t pool_size,
        hvr_irr_matrix_t *m) {
    m->edges = (hvr_edge_info_t **)malloc(nvertices * sizeof(m->edges[0]));
    assert(m->edges);
    m->edges_capacity = (unsigned *)malloc(
            nvertices * sizeof(m->edges_capacity[0]));
    assert(m->edges_capacity);
    m->edges_len = (unsigned *)malloc(nvertices * sizeof(m->edges_len[0]));
    assert(m->edges_len);

    memset(m->edges, 0x00, sizeof(m->edges[0]) * nvertices);
    memset(m->edges_capacity, 0x00, sizeof(m->edges_capacity[0]) * nvertices);
    memset(m->edges_len, 0x00, sizeof(m->edges_capacity[0]) * nvertices);

    m->nvertices = nvertices;

    m->pool = malloc(pool_size);
    assert(m->pool);
    m->pool_size = pool_size;
    m->allocator = create_mspace_with_base(m->pool, pool_size, 0);
    assert(m->allocator);
}

hvr_edge_type_t hvr_irr_matrix_get(hvr_vertex_id_t i,
        hvr_vertex_id_t j, const hvr_irr_matrix_t *m) {
    const unsigned curr_len = m->edges_len[i];
    const hvr_edge_info_t *curr_edges = m->edges[i];

    for (unsigned iter = 0; iter < curr_len; iter++) {
        if (EDGE_INFO_VERTEX(curr_edges[iter]) == j) {
            return (hvr_edge_type_t)EDGE_INFO_EDGE(curr_edges[iter]);
        }
    }
    return NO_EDGE;
}

void hvr_irr_matrix_resize(hvr_vertex_id_t i, unsigned new_capacity,
        hvr_irr_matrix_t *m) {
    const unsigned curr_len = m->edges_len[i];
    const unsigned curr_capacity = m->edges_capacity[i];
    assert(curr_len <= new_capacity);

    if (new_capacity != curr_capacity) {
        m->edges[i] = mspace_realloc(m->allocator, m->edges[i],
                new_capacity * sizeof(m->edges[0][0]));
        assert(m->edges[i]);
        m->edges_capacity[i] = new_capacity;
    }
}

void hvr_irr_matrix_set(hvr_vertex_id_t i, hvr_vertex_id_t j, hvr_edge_type_t e,
        hvr_irr_matrix_t *m) {
    const unsigned curr_len = m->edges_len[i];
    const unsigned curr_capacity = m->edges_capacity[i];
    hvr_edge_info_t *curr_edges = m->edges[i];

    int found = -1;
    for (unsigned iter = 0; iter < curr_len; iter++) {
        hvr_edge_info_t e = curr_edges[iter];
        hvr_vertex_id_t neighbor = EDGE_INFO_VERTEX(e);
        if (neighbor == j) {
            found = iter;
            break;
        }
    }

    if (found >= 0) {
        // Existing neighbor
        if (e == NO_EDGE) {
            // Delete entry
            curr_edges[found] = curr_edges[curr_len - 1];
            m->edges_len[i] = curr_len - 1;
        } else {
            // Overwrite entry
            curr_edges[found] = construct_edge_info(j, e);
        }
    } else {
        // Does not exist already
        if (e == NO_EDGE) return;

        if (curr_len == curr_capacity) {
            // No more room, expand
            unsigned new_capacity;
            if (curr_capacity == 0) {
                new_capacity = 2;
            } else if (curr_capacity <= 128) {
                new_capacity = curr_capacity * 2;
            } else {
                new_capacity = curr_capacity + 16;
            }

            m->edges[i] = mspace_realloc(m->allocator, curr_edges,
                    new_capacity * sizeof(*curr_edges));
            assert(m->edges[i]);
            m->edges_capacity[i] = new_capacity;
        }

        (m->edges[i])[curr_len] = construct_edge_info(j, e);
        m->edges_len[i] += 1;
    }
}

unsigned hvr_irr_matrix_linearize_zero_copy(hvr_vertex_id_t i,
        hvr_edge_info_t **out_edges, hvr_irr_matrix_t *m) {
    const unsigned curr_len = m->edges_len[i];
    hvr_edge_info_t *curr_edges = m->edges[i];

    *out_edges = curr_edges;

    return curr_len;
}

unsigned hvr_irr_matrix_linearize(hvr_vertex_id_t i, hvr_edge_info_t *out_edges,
        size_t capacity, hvr_irr_matrix_t *m) {
    const unsigned curr_len = m->edges_len[i];
    hvr_edge_info_t *curr_edges = m->edges[i];

    assert(curr_len <= capacity);

    memcpy(out_edges, curr_edges, curr_len * sizeof(*out_edges));

    return curr_len;
}

void hvr_irr_matrix_usage(size_t *out_bytes_used, size_t *out_bytes_capacity,
        size_t *out_bytes_allocated, size_t *out_max_edges,
        size_t *out_max_edges_index, hvr_irr_matrix_t *m) {
    *out_bytes_allocated = m->nvertices * sizeof(m->edges[0]) +
        m->nvertices * sizeof(m->edges_capacity[0]) +
        m->nvertices * sizeof(m->edges_len[0]) +
        m->pool_size;

    size_t max_edges = 0;
    size_t max_edges_index = 0;
    size_t bytes_used = 0;
    size_t bytes_capacity = 0;
    bytes_capacity = bytes_used = m->nvertices * sizeof(m->edges[0]) +
        m->nvertices * sizeof(m->edges_capacity[0]) +
        m->nvertices * sizeof(m->edges_len[0]);

    for (size_t i = 0; i < m->nvertices; i++) {
        bytes_capacity += m->edges_capacity[i] * sizeof(m->edges[0]);
        bytes_used += m->edges_len[i] * sizeof(m->edges[0]);
        if (m->edges_len[i] > max_edges) {
            max_edges = m->edges_len[i];
            max_edges_index = i;
        }
    }

    *out_bytes_used = bytes_used;
    *out_bytes_capacity = bytes_capacity;
    *out_max_edges = max_edges;
    *out_max_edges_index = max_edges_index;
}
