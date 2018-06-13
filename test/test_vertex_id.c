#include <stdio.h>
#include <assert.h>

#include "hvr_common.h"

static void verify_id_construction(int pe, uint64_t offset) {
    uint64_t id = construct_vertex_id(pe, offset);

    uint64_t computed_pe = VERTEX_ID_PE(id);
    uint64_t computed_offset = VERTEX_ID_OFFSET(id);

    if (computed_pe != pe || computed_offset != offset) {
        fprintf(stderr, "Expected pe=%d offset=%lu, got pe=%lu offset=%lu\n",
                pe, offset, computed_pe, computed_offset);
        abort();
    }
}

int main(int argc, char **argv) {
    verify_id_construction(0, 0);
    verify_id_construction(4, 3);
    verify_id_construction(3, 4);
    verify_id_construction(INT32_MAX, UINT32_MAX);
    verify_id_construction(2048, 1024);
    printf("Success!\n");
    return 0;
}
