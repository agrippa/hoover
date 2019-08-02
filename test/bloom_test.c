#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "hvr_bloom.h"

int main(int argc, char **argv) {
    {
        hvr_counting_bloom_t blm;
        hvr_counting_bloom_init(&blm);

        hvr_counting_bloom_set(3, &blm);
        assert(hvr_counting_bloom_check(3, &blm));
        assert(!hvr_counting_bloom_check(2048, &blm));

        hvr_counting_bloom_remove(3, &blm);
        assert(!hvr_counting_bloom_check(3, &blm));

        printf("Passed!\n");
    }

    {
        hvr_bloom_t blm;
        hvr_bloom_init(&blm, 1024);

        for (int i = 0; i < 4096; i++) {
            for (int j = 0; j < 4096; j++) {
                assert(!hvr_bloom_check(j, &blm));
            }

            hvr_bloom_set(i, &blm);
            assert(hvr_bloom_check(i, &blm));

            hvr_bloom_remove(i, &blm);
            for (int j = 0; j < 4096; j++) {
                assert(!hvr_bloom_check(j, &blm));
            }
        }

        printf("Passed!\n");
    }

    return 0;
}
