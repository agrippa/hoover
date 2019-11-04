#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "graph_generator.h"
#include "utils.h"

/*
 * Problem sizes:
 *
 *   Name     | SCALE
 *   ----------------
 *   - Toy    | 26
 *   - Mini   | 29
 *   - Small  | 32
 *   - Medium | 36
 *   - Large  | 39
 *   - Huge   | 42
 */

int main(int argc, char **argv) {

    int64_t SCALE = 16; // size of graph (# vertices = 2^SCALE)
    int64_t edgefactor = 16; // average # of edges per vertex

    if (argc != 2) {
        fprintf(stderr, "usage: %s <SCALE>\n", argv[0]);
        return 1;
    }

    SCALE = atoi(argv[1]);

    int64_t nedges = edgefactor << SCALE;
    int64_t nvertices = 1 << SCALE;

    printf("# vertices = %ld, # edges = %ld, %f edges per vertex\n", nvertices,
            nedges, (double)nedges / (double)nvertices);

    packed_edge *buf = (packed_edge *)malloc(nedges * sizeof(*buf));
    assert(buf);

	int64_t seed1 = 2, seed2 = 3;
    uint_fast32_t seed[5];
    make_mrg_seed(seed1, seed2, seed);

    generate_kronecker_range(seed, SCALE, 0, nedges, buf);

    printf("finished generating edges\n");

    int64_t *row = (int64_t *)malloc(nedges * sizeof(*row));
    assert(row);
    int64_t *col = (int64_t *)malloc(nedges * sizeof(*col));
    assert(col);

    int64_t min_vertex = get_v0_from_edge(buf + 0);
    int64_t max_vertex = get_v0_from_edge(buf + 0);

    for (int i = 0; i < nedges; i++) {
        row[i] = get_v0_from_edge(buf + i);
        col[i] = get_v1_from_edge(buf + i);

        if (row[i] < min_vertex) min_vertex = row[i];
        if (col[i] < min_vertex) min_vertex = col[i];
        if (row[i] > max_vertex) max_vertex = row[i];
        if (col[i] > max_vertex) max_vertex = col[i];
    }

    printf("min vertex = %ld, max vertex = %ld\n", min_vertex, max_vertex);

    char filename[1024];
    sprintf(filename, "scale=%ld_edgefactor=%ld_edges=%ld_vertices=%ld.bin",
            SCALE, edgefactor, nedges, nvertices);

    size_t n;
    FILE *fp = fopen(filename, "w");
    n = fwrite(&nvertices, sizeof(nvertices), 1, fp);
    assert(n == 1);
    n = fwrite(&nvertices, sizeof(nvertices), 1, fp);
    assert(n == 1);
    n = fwrite(&nedges, sizeof(nedges), 1, fp);
    assert(n == 1);
    n = fwrite(row, sizeof(*row), nedges, fp);
    assert(n == nedges);
    n = fwrite(col, sizeof(*col), nedges, fp);
    assert(n == nedges);
    fclose(fp);

    return 0;
}
