#include <stdio.h>
#include <assert.h>

#include "hvr_common.h"

static void verify_edge_construction(int pe, uint64_t offset,
        hvr_edge_type_t edge) {
    uint64_t id = construct_vertex_id(pe, offset);
    uint64_t edge_info = construct_edge_info(id, edge);

    uint64_t computed_pe = VERTEX_ID_PE(EDGE_INFO_VERTEX(edge_info));
    uint64_t computed_offset = VERTEX_ID_OFFSET(EDGE_INFO_VERTEX(edge_info));
    hvr_edge_type_t computed_edge = EDGE_INFO_EDGE(edge_info);

    if (computed_pe != pe || computed_offset != offset ||
        computed_edge != edge) {
        fprintf(stderr, "Expected pe=%d offset=%lu edge=%d, got pe=%lu "
                "offset=%lu edge=%d\n",
                pe, offset, edge, computed_pe, computed_offset, computed_edge);
        abort();
    }
}

int main(int argc, char **argv) {
    verify_edge_construction(0, 0, NO_EDGE);
    verify_edge_construction(0, 0, BIDIRECTIONAL);
    verify_edge_construction(0, 0, DIRECTED_IN);
    verify_edge_construction(0, 0, DIRECTED_OUT);

    verify_edge_construction(4, 3, NO_EDGE);
    verify_edge_construction(4, 3, BIDIRECTIONAL);
    verify_edge_construction(4, 3, DIRECTED_IN);
    verify_edge_construction(4, 3, DIRECTED_OUT);

    verify_edge_construction(3, 4, NO_EDGE);
    verify_edge_construction(3, 4, BIDIRECTIONAL);
    verify_edge_construction(3, 4, DIRECTED_IN);
    verify_edge_construction(3, 4, DIRECTED_OUT);

    verify_edge_construction(3000, UINT32_MAX, NO_EDGE);
    verify_edge_construction(3000, UINT32_MAX, BIDIRECTIONAL);
    verify_edge_construction(3000, UINT32_MAX, DIRECTED_IN);
    verify_edge_construction(3000, UINT32_MAX, DIRECTED_OUT);

    verify_edge_construction(2048, 1024, NO_EDGE);
    verify_edge_construction(2048, 1024, BIDIRECTIONAL);
    verify_edge_construction(2048, 1024, DIRECTED_IN);
    verify_edge_construction(2048, 1024, DIRECTED_OUT);

    printf("Success!\n");
    return 0;
}
