#include "hvr_map.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    hvr_map_t map;
    hvr_map_init(&map, 3, "DUMMY");

    assert(hvr_map_get(3, &map) == NULL);

    hvr_map_add(3, (void*)0x1, 0, &map);
    assert(hvr_map_get(3, &map) == (void*)0x1);
    assert(hvr_map_get(4, &map) == NULL);

    hvr_map_add(4, (void*)0x2, 0, &map);
    assert(hvr_map_get(3, &map) == (void*)0x1);
    assert(hvr_map_get(4, &map) == (void*)0x2);
    assert(hvr_map_get(5, &map) == NULL);

    hvr_map_add(1500, (void*)0x3, 0, &map);
    assert(hvr_map_get(1500, &map) == (void*)0x3);
    assert(hvr_map_get(3, &map) == (void*)0x1);
    assert(hvr_map_get(4, &map) == (void*)0x2);
    assert(hvr_map_get(5, &map) == NULL);

    hvr_map_remove(3, (void*)0x1, &map);
    assert(hvr_map_get(1500, &map) == (void*)0x3);
    assert(hvr_map_get(3, &map) == NULL);
    assert(hvr_map_get(4, &map) == (void*)0x2);
    assert(hvr_map_get(5, &map) == NULL);

    // Test that a double add followed by a single remove works
    hvr_map_add(3, (void *)0x4, 1, &map);
    assert(hvr_map_get(3, &map) == (void *)0x4);

    hvr_map_add(3, (void*)0x5, 1, &map);
    assert(hvr_map_get(3, &map) == (void *)0x5);

    hvr_map_remove(3, (void*)0x5, &map);
    assert(hvr_map_get(3, &map) == NULL);

    // Test that a double remove is fine
    hvr_map_add(3, (void *)0x6, 0, &map);
    assert(hvr_map_get(3, &map) == (void *)0x6);

    hvr_map_remove(3, (void *)0x6, &map);
    assert(hvr_map_get(3, &map) == NULL);

    hvr_map_remove(3, (void *)0x6, &map);
    assert(hvr_map_get(3, &map) == NULL);

    printf("Success!\n");

    return 0;
}
