#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include <assert.h>

int main(int argc, char **argv) {
    int ret_code;
    if (argc != 4) {
        fprintf(stderr, "usage: %s <mat-market-file> <print-freq> <output>\n",
                argv[1]);
        return 1;
    }

    const char *mat_filename = argv[1];
    int print_freq = atoi(argv[2]);
    const char *output_filename = argv[3];
    FILE *fp = fopen(mat_filename, "r");
    assert(fp);

    MM_typecode matcode;
    if (mm_read_banner(fp, &matcode) != 0) {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    assert(mm_is_matrix(matcode));
    assert(mm_is_coordinate(matcode));
    assert(mm_is_pattern(matcode));
    assert(mm_is_general(matcode));

    int M, N, nz;
    if ((ret_code = mm_read_mtx_crd_size(fp, &M, &N, &nz)) !=0) {
        abort();
    }

    printf("Matrix %s is %d x %d with %d non-zeroes\n", mat_filename, M, N,
            nz);

    int *I = (int *)malloc(nz * sizeof(*I));
    assert(I);
    int *J = (int *)malloc(nz * sizeof(*J));
    assert(J);

    for (int i=0; i<nz; i++) {
        fscanf(fp, "%d %d\n", &I[i], &J[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
        if ((i+1) % print_freq == 0) {
            printf("%d / %d (%f%%)\n", i+1, nz, 100.0 * (i+1) / (double)nz);
        }
    }
    fclose(fp);

    size_t n;
    fp = fopen(output_filename, "w");
    n = fwrite(&M, sizeof(M), 1, fp);
    assert(n == 1);
    n = fwrite(&N, sizeof(N), 1, fp);
    assert(n == 1);
    n = fwrite(&nz, sizeof(nz), 1, fp);
    assert(n == 1);
    n = fwrite(I, sizeof(int), nz, fp);
    assert(n == nz);
    n = fwrite(J, sizeof(int), nz, fp);
    assert(n == nz);
    fclose(fp);

    return 0;
}
