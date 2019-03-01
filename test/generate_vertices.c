#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define MAX_DST_DELTA 2000.0

static double random_double_in_range(double min_val_inclusive,
        double max_val_exclusive) {
    return min_val_inclusive + (((double)rand() / (double)RAND_MAX) *
            (max_val_exclusive - min_val_inclusive));
}

static int random_int_in_range(int max_val) {
    return (rand() % max_val);
}

static void write_wrapper(double *vertex_values, unsigned n_vertex_values,
        FILE *fp) {
    size_t written = fwrite(vertex_values, sizeof(*vertex_values),
            n_vertex_values, fp);
    assert(written == n_vertex_values);
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <output-dir> <nvertices> <max_x> "
                "<max_y> <# infected>\n", argv[0]);
        return 1;
    }

    /*
     * 0: actor id
     * 1: px
     * 2: py
     * 3: dst_x
     * 4: dst_y
     * 5: infected
     */
    double buf[6];

    char *output_dir = argv[1];
    long nvertices = atol(argv[2]);
    long max_x = atol(argv[3]);
    long max_y = atol(argv[4]);
    int n_infected = atoi(argv[5]);

    char output_file[2048];
    sprintf(output_file, "%s/%ld-vert.%ld-y.%ld-x.%d-infected.bin",
            output_dir, nvertices, max_y, max_x, n_infected);
    char output_file_txt[2048];
    sprintf(output_file_txt, "%s/%ld-vert.%ld-y.%ld-x.%d-infected.txt",
            output_dir, nvertices, max_y, max_x, n_infected);

    FILE *out = fopen(output_file, "wb");
    assert(out);
    FILE *out_txt = fopen(output_file_txt, "w");
    assert(out_txt);

    int *initial_infected = (int *)malloc(n_infected *
            sizeof(*initial_infected));
    assert(initial_infected);
    for (int i = 0; i < n_infected; i++) {
        initial_infected[i] = random_int_in_range(nvertices);
        printf("Actor %ld infected\n", initial_infected[i]);
    }

    for (long i = 0; i < nvertices; i++) {
        buf[0] = i;
        buf[1] = random_double_in_range(0.0, max_x);
        buf[2] = random_double_in_range(0.0, max_y);
        buf[3] = buf[1] + random_double_in_range(-MAX_DST_DELTA,
                MAX_DST_DELTA);
        buf[4] = buf[2] + random_double_in_range(-MAX_DST_DELTA,
                MAX_DST_DELTA);

        if (buf[3] > (double)max_x) buf[3] = (double)max_x - 1.0;
        if (buf[4] > (double)max_y) buf[4] = (double)max_y - 1.0;
        if (buf[3] < 0.0) buf[3] = 0.0;
        if (buf[4] < 0.0) buf[4] = 0.0;

        buf[5] = 0;
        for (int j = 0; j < n_infected; j++) {
            if (initial_infected[j] == i) {
                buf[5] = 1;
            }
        }

        write_wrapper(buf, 6, out);

        fprintf(out_txt, "%f %f %f %f %f %f\n", buf[0], buf[1], buf[2], buf[3],
                buf[4], buf[5]);
    }

    fclose(out);
    fclose(out_txt);
    return 0;
}
