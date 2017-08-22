#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <shmem.h>

#include <hoover.h>

int main(int argc, char **argv) {
    shmem_init();
    const int pe = shmem_my_pe();
    const int npes = shmem_n_pes();

    printf("Hello from PE %d / %d\n", pe + 1, npes);

    const int vertex_per_node = 24;
    const int edges_per_vertex = 5;

    vertex_id_t *vertices = (vertex_id_t *)malloc(
            vertex_per_node * sizeof(*vertices));
    assert(vertices);
    for (int i = 0; i < vertex_per_node; i++) {
        vertices[i] = pe * vertex_per_node + i;
    }

    vertex_id_t *edges = (vertex_id_t *)malloc(
            edges_per_vertex * sizeof(*edges));
    assert(edges);
    for (int i = 0; i < edges_per_vertex; i++) {
        edges[i] = (rand() % (npes * vertex_per_node));
    }

    size_t *edge_offsets = (size_t *)malloc(
            (vertex_per_node + 1) * sizeof(*edge_offsets));
    assert(edge_offsets);

    edge_offsets[0] = 0;
    for (int i = 1; i <= vertex_per_node; i++) {
        edge_offsets[i] = edge_offsets[i - 1] + edges_per_vertex;
    }

    hvr_ctx_t hvr_ctx;
    hvr_init(vertex_per_node, vertices, NULL, 0, edges, edge_offsets, &hvr_ctx);
    hvr_finalize(hvr_ctx);

    shmem_finalize();

    return 0;
}
