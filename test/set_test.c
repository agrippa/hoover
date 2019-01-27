/* For license: see LICENSE.txt file at top-level */

#include <stdio.h>
#include <assert.h>
#include <hoover.h>
#include <shmem.h>

#define NPES 100

int main(int argc, char **argv) {
    hvr_set_t *singleton_set = hvr_create_empty_set(1);
    hvr_set_insert(0, singleton_set);
    assert(hvr_set_contains(0, singleton_set) == 1);
    assert(hvr_set_count(singleton_set) == 1);
    hvr_set_destroy(singleton_set);

    hvr_set_t *set = hvr_create_empty_set(NPES);

    for (int i = 0; i < NPES; i++) {
        hvr_set_insert(i, set);
        assert(hvr_set_contains(i, set) == 1);
        for (int j = 0; j < NPES; j++) {
            if (j != i) assert(hvr_set_contains(j, set) == 0);
        }
        hvr_set_wipe(set);
    }

    hvr_set_insert(2, set);
    hvr_set_insert(8, set);
    for (int i = 0; i < NPES; i++) {
        if (i == 2 || i == 8) {
            assert(hvr_set_contains(i, set) == 1);
        } else {
            assert(hvr_set_contains(i, set) == 0);
        }
    }
    assert(hvr_set_count(set) == 2);

    hvr_set_t *or_set = hvr_create_empty_set(NPES);
    hvr_set_insert(0, set);
    hvr_set_insert(1, set);
    hvr_set_merge(set, or_set);

    for (int i = 0; i < NPES; i++) {
        if (i == 2 || i == 8 || i == 0 || i == 1) {
            assert(hvr_set_contains(i, set) == 1);
        } else {
            assert(hvr_set_contains(i, set) == 0);
        }
    }
    assert(hvr_set_count(set) == 4);

    hvr_set_wipe(set);
    for (int i = 0; i < NPES; i++) {
        assert(hvr_set_contains(i, set) == 0);
    }

    hvr_set_destroy(set);

    hvr_set_t *custom_set = hvr_create_empty_set(2000);
    for (int i = 0; i < 2000; i += 2) {
        hvr_set_insert(i, custom_set);
    }
    for (int i = 0; i < 2000; i++) {
        if (i % 2 == 0) {
            assert(hvr_set_contains(i, custom_set) == 1);
        } else {
            assert(hvr_set_contains(i, custom_set) == 0);
        }
    }
    for (int i = 0; i < 2000; i++) {
        if (i % 2 == 0) {
            hvr_set_clear(i, custom_set);
        } else {
            hvr_set_insert(i, custom_set);
        }
    }
    for (int i = 0; i < 2000; i++) {
        if (i % 2 == 0) {
            assert(hvr_set_contains(i, custom_set) == 0);
        } else {
            assert(hvr_set_contains(i, custom_set) == 1);
        }
    }

    printf("Passed!\n");

    return 0;
}
