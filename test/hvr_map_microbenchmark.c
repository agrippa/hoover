#include "hvr_map.h"

int main(int argc, char **argv) {
    hvr_map_t m;
    hvr_map_init(&m,
            40000,/* prealloc segs */
            1073741824, /* prealloc vals */
            65536, /* prealloc nodes */,
            8,
            EDGE_INFO);

    for (unsigned i = 0; i < 10000000; i+) {
        for (unsigned j = 10000000 - 1; j >= 0; j--) {
            hvr_map_add(
        }
    }
    return 0;
}
