#include <stdio.h>
#include <assert.h>
#include <hoover.h>
#include <shmem.h>

#define NPES 100

int main(int argc, char **argv) {
    shmem_init();

    hvr_ctx_t ctx;
    hvr_ctx_create(&ctx);

    ((hvr_internal_ctx_t *)ctx)->npes = 1;
    hvr_pe_set_t *singleton_set =
        hvr_create_empty_pe_set(ctx);
    hvr_pe_set_insert(0, singleton_set);
    assert(hvr_pe_set_contains(0, singleton_set) == 1);
    assert(hvr_pe_set_count(singleton_set) == 1);
    hvr_pe_set_destroy(singleton_set);

    ((hvr_internal_ctx_t *)ctx)->npes = NPES;
    hvr_pe_set_t *set = hvr_create_empty_pe_set(ctx);

    for (int i = 0; i < NPES; i++) {
        hvr_pe_set_insert(i, set);
        assert(hvr_pe_set_contains(i, set) == 1);
        for (int j = 0; j < NPES; j++) {
            if (j != i) assert(hvr_pe_set_contains(j, set) == 0);
        }
        hvr_pe_set_clear(i, set);
        for (int j = 0; j < NPES; j++) {
            assert(hvr_pe_set_contains(j, set) == 0);
        }
    }

    hvr_pe_set_insert(2, set);
    hvr_pe_set_insert(8, set);
    for (int i = 0; i < NPES; i++) {
        if (i == 2 || i == 8) {
            assert(hvr_pe_set_contains(i, set) == 1);
        } else {
            assert(hvr_pe_set_contains(i, set) == 0);
        }
    }
    assert(hvr_pe_set_count(set) == 2);

    char buf[1024];
    hvr_pe_set_to_string(set, buf, 1024);
    printf("%s\n", buf);

    hvr_pe_set_t *or_set = hvr_create_empty_pe_set(ctx);
    hvr_pe_set_insert(0, set);
    hvr_pe_set_insert(1, set);
    hvr_pe_set_merge(set, or_set);

    for (int i = 0; i < NPES; i++) {
        if (i == 2 || i == 8 || i == 0 || i == 1) {
            assert(hvr_pe_set_contains(i, set) == 1);
        } else {
            assert(hvr_pe_set_contains(i, set) == 0);
        }
    }
    assert(hvr_pe_set_count(set) == 4);

    hvr_pe_set_wipe(set);
    for (int i = 0; i < NPES; i++) {
        assert(hvr_pe_set_contains(i, set) == 0);
    }

    hvr_pe_set_destroy(set);

    hvr_pe_set_t *custom_set = hvr_create_empty_pe_set_custom(2000, ctx);
    for (int i = 0; i < 2000; i += 2) {
        hvr_pe_set_insert(i, custom_set);
    }
    for (int i = 0; i < 2000; i++) {
        if (i % 2 == 0) {
            assert(hvr_pe_set_contains(i, custom_set) == 1);
        } else {
            assert(hvr_pe_set_contains(i, custom_set) == 0);
        }
    }

    printf("Passed!\n");

    shmem_finalize();

    return 0;
}
