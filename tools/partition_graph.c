#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define TILE_SIZE (2 * 1024 * 1024)

typedef struct _file_buffer_t {
    char filename_0[2048];
    char filename_1[2048];

    int64_t _0[TILE_SIZE];
    int64_t _1[TILE_SIZE];
    unsigned nbuffered;

    long n_partition_edges_offset;
    int64_t count;
} file_buffer_t;

void file_buffer_init(file_buffer_t *buf, const char *mat_filename, int npes,
        int pe, int64_t nedges, int64_t nvertices) {
    sprintf(buf->filename_0, "%s.npes=%d.pe=%d_0", mat_filename, npes, pe);
    FILE *fp0 = fopen(buf->filename_0, "wb");
    assert(fp0);

    sprintf(buf->filename_1, "%s.npes=%d.pe=%d_1", mat_filename, npes, pe);
    FILE *fp1 = fopen(buf->filename_1, "wb");
    assert(fp1);

    int64_t placeholder = 0;

    size_t n = fwrite(&nvertices, sizeof(nvertices), 1, fp0);
    assert(n == 1);
    n = fwrite(&nvertices, sizeof(nvertices), 1, fp0);
    assert(n == 1);
    n = fwrite(&nedges, sizeof(nedges), 1, fp0);
    assert(n == 1);
    fflush(fp0);
    buf->n_partition_edges_offset = ftell(fp0);
    n = fwrite(&placeholder, sizeof(placeholder), 1, fp0);
    assert(n == 1);
    fclose(fp0);

    n = fwrite(&nvertices, sizeof(nvertices), 1, fp1);
    assert(n == 1);
    n = fwrite(&nvertices, sizeof(nvertices), 1, fp1);
    assert(n == 1);
    n = fwrite(&nedges, sizeof(nedges), 1, fp1);
    assert(n == 1);
    fflush(fp1);
    assert(buf->n_partition_edges_offset == ftell(fp1));
    n = fwrite(&placeholder, sizeof(placeholder), 1, fp1);
    assert(n == 1);
    fclose(fp1);

    buf->nbuffered = 0;
    buf->count = 0;
}

void file_buffer_write(file_buffer_t *buf, int64_t _0, int64_t _1) {
    if (buf->nbuffered == TILE_SIZE) {
        FILE *fp0 = fopen(buf->filename_0, "ab");
        assert(fp0);
        FILE *fp1 = fopen(buf->filename_1, "ab");
        assert(fp1);

        size_t n;
        n = fwrite(&buf->_0[0], sizeof(buf->_0[0]), TILE_SIZE, fp0);
        assert(n == TILE_SIZE);

        n = fwrite(&buf->_1[0], sizeof(buf->_1[0]), TILE_SIZE, fp1);
        assert(n == TILE_SIZE);

        fclose(fp0); fclose(fp1);

        buf->nbuffered = 0;
    }

    buf->_0[buf->nbuffered] = _0;
    buf->_1[buf->nbuffered] = _1;
    buf->nbuffered += 1;
    buf->count += 1;
}

void file_buffer_flush(file_buffer_t *buf) {
    if (buf->nbuffered > 0) {
        FILE *fp0 = fopen(buf->filename_0, "ab+");
        assert(fp0);
        FILE *fp1 = fopen(buf->filename_1, "ab+");
        assert(fp1);

        size_t n;
        n = fwrite(&buf->_0[0], sizeof(buf->_0[0]), buf->nbuffered, fp0);
        assert(n == buf->nbuffered);

        n = fwrite(&buf->_1[0], sizeof(buf->_1[0]), buf->nbuffered, fp1);
        assert(n == buf->nbuffered);

        fclose(fp0);
        fclose(fp1);
    }

    FILE *fp0 = fopen(buf->filename_0, "rb+");
    assert(fp0);
    FILE *fp1 = fopen(buf->filename_1, "rb+");
    assert(fp1);

    int err = fseek(fp0, buf->n_partition_edges_offset, SEEK_SET);
    assert(err == 0);
    err = fseek(fp1, buf->n_partition_edges_offset, SEEK_SET);
    assert(err == 0);

    size_t n = fwrite(&buf->count, sizeof(buf->count), 1, fp0);
    assert(n == 1);
    n = fwrite(&buf->count, sizeof(buf->count), 1, fp1);
    assert(n == 1);

    fclose(fp0);
    fclose(fp1);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <mat-file> <npes>\n", argv[0]);
        return 1;
    }

    const char *mat_filename = argv[1];
    int npes = atoi(argv[2]);

    FILE *fp = fopen(mat_filename, "rb");
    assert(fp);

    int64_t M, N, nz;
    size_t n;
    n = fread(&M, sizeof(M), 1, fp);
    assert(n == 1);
    n = fread(&N, sizeof(N), 1, fp);
    assert(n == 1);
    n = fread(&nz, sizeof(nz), 1, fp);
    assert(n == 1);

    assert(M == N);

    file_buffer_t *pe_bufs = (file_buffer_t *)malloc(npes * sizeof(*pe_bufs));
    assert(pe_bufs);
    for (int p = 0; p < npes; p++) {
        file_buffer_init(pe_bufs + p, mat_filename, npes, p, nz, M);
    }

    fprintf(stderr, "Matrix %s is %ld x %ld with %ld non-zeroes.\n",
            mat_filename, M, N, nz);
    fprintf(stderr, "Partitioning for %d PEs\n", npes);

    int64_t vertices_per_pe = (M + npes - 1) / npes;

    const int tile_size = 1024 * 1024 * 1024;
    int64_t *I = (int64_t *)malloc(tile_size * sizeof(*I));
    assert(I);
    int64_t *J = (int64_t *)malloc(tile_size * sizeof(*J));
    assert(J);

    long I_offset = ftell(fp);

    fprintf(stderr, "Seeking...\n");

    int64_t count_edges_to_insert = 0;
    for (int64_t i = 0; i < nz; i += tile_size) {
        int64_t to_read = nz - i;
        if (to_read > tile_size) to_read = tile_size;

        n = fread(I, sizeof(*I), to_read, fp);
        assert(n == to_read);
    }

    long J_offset = ftell(fp);

    for (int64_t i = 0; i < nz; i += tile_size) {
        fprintf(stderr, "Processing %ld / %lu (%f%%)\n", i, nz,
                100.0 * (float)i / (float)nz);

        int64_t to_read = nz - i;
        if (to_read > tile_size) to_read = tile_size;

        fseek(fp, I_offset + i * sizeof(*I), SEEK_SET);
        n = fread(I, sizeof(int64_t), to_read, fp);
        assert(n == to_read);

        fseek(fp, J_offset + i * sizeof(*J), SEEK_SET);
        n = fread(J, sizeof(int64_t), to_read, fp);
        assert(n == to_read);

        for (int64_t j = 0; j < to_read; j++) {
            assert(I[j] < M);
            int64_t I_pe = I[j] / vertices_per_pe;
            assert(I_pe < npes);
            int64_t I_offset = I[j] % vertices_per_pe;

            assert(J[j] < M);
            int64_t J_pe = J[j] / vertices_per_pe;
            assert(J_pe < npes);
            int64_t J_offset = J[j] % vertices_per_pe;

            // Basic load balancing
            if (pe_bufs[I_pe].count < pe_bufs[J_pe].count) {
                file_buffer_write(pe_bufs + I_pe, I[j], J[j]);
            } else {
                file_buffer_write(pe_bufs + J_pe, J[j], I[j]);
            }
        }
    }

    for (int p = 0; p < npes; p++) {
        file_buffer_flush(pe_bufs + p);
    }

    free(I);
    free(J);

    fclose(fp);

    return 0;
}
