#include "hvr_map.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    hvr_map_t map;
    hvr_map_init(&map);

    assert(hvr_map_contains(3, 3, &map) == NO_EDGE);


    hvr_map_val_t val1 = {.edge_info = {.id = 3, .edge = BIDIRECTIONAL}};
    hvr_map_add(3, val1, 1, &map);
    assert(hvr_map_contains(3, 3, &map) == BIDIRECTIONAL);
    assert(hvr_map_contains(4, 4, &map) == NO_EDGE);

    hvr_map_val_t val2 = {.edge_info = {.id = 4, .edge = DIRECTED_IN}};
    hvr_map_add(3, val2, 1, &map);
    assert(hvr_map_contains(3, 3, &map) == BIDIRECTIONAL);
    assert(hvr_map_contains(3, 4, &map) == DIRECTED_IN);
    assert(hvr_map_contains(4, 4, &map) == NO_EDGE);

    hvr_map_val_t val3 = {.edge_info = {.id = 1500, .edge = DIRECTED_OUT}};
    hvr_map_add(1500, val3, 1, &map);
    assert(hvr_map_contains(1500, 1500, &map) == DIRECTED_OUT);
    assert(hvr_map_contains(3, 3, &map) == BIDIRECTIONAL);
    assert(hvr_map_contains(3, 4, &map) == DIRECTED_IN);
    assert(hvr_map_contains(4, 4, &map) == NO_EDGE);

    hvr_vertex_id_t *tmp_arr = NULL;
    hvr_edge_type_t *tmp_edges = NULL;
    unsigned capacity = 0;
    unsigned len = hvr_map_linearize(3, &tmp_arr, &tmp_edges, &capacity, &map);
    assert(len == 2);
    assert(capacity == 2);
    assert(tmp_arr[0] == 3);
    assert(tmp_arr[1] == 4);
    assert(tmp_edges[0] == BIDIRECTIONAL);
    assert(tmp_edges[1] == DIRECTED_IN);

    hvr_map_val_t val7 = {.edge_info = {.id = 3}};
    hvr_map_remove(3, val7, 1, &map);
    assert(hvr_map_contains(1500, 1500, &map) == DIRECTED_OUT);
    assert(hvr_map_contains(3, 3, &map) == NO_EDGE);
    assert(hvr_map_contains(3, 4, &map) == DIRECTED_IN);
    assert(hvr_map_contains(4, 4, &map) == NO_EDGE);


    // Test that a double add followed by a single remove works
    hvr_map_val_t val4 = {.edge_info = {.id = 3, .edge = BIDIRECTIONAL}};
    hvr_map_add(3, val4, 1, &map);
    assert(hvr_map_contains(3, 3, &map) == BIDIRECTIONAL);

    hvr_map_val_t val5 = {.edge_info = {.id = 3, .edge = BIDIRECTIONAL}};
    hvr_map_add(3, val5, 1, &map);
    assert(hvr_map_contains(3, 3, &map) == BIDIRECTIONAL);

    hvr_map_val_t val8 = {.edge_info = {.id = 3}};
    hvr_map_remove(3, val8, 1, &map);
    assert(hvr_map_contains(3, 3, &map) == NO_EDGE);

    // Test that a double remove is fine
    hvr_map_val_t val6 = {.edge_info = {.id = 3, .edge = BIDIRECTIONAL}};
    hvr_map_add(3, val6, 1, &map);
    assert(hvr_map_contains(3, 3, &map) == BIDIRECTIONAL);

    hvr_map_val_t val9 = {.edge_info = {.id = 3}};
    hvr_map_remove(3, val9, 1, &map);
    assert(hvr_map_contains(3, 3, &map) == NO_EDGE);

    hvr_map_val_t val10 = {.edge_info = {.id = 3}};
    hvr_map_remove(3, val10, 1, &map);
    assert(hvr_map_contains(3, 3, &map) == NO_EDGE);

    printf("Success!\n");

    return 0;
}
