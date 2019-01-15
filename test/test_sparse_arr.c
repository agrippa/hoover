#include "hvr_sparse_arr.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    hvr_sparse_arr_t arr;
    hvr_sparse_arr_init(&arr, 2000);

    assert(hvr_sparse_arr_contains(3, 3, &arr) == 0);

    hvr_sparse_arr_insert(3, 3, &arr);
    assert(hvr_sparse_arr_contains(3, 3, &arr) == 1);
    assert(hvr_sparse_arr_contains(4, 4, &arr) == 0);
    assert(hvr_sparse_arr_contains(3, 4, &arr) == 0);
    assert(hvr_sparse_arr_contains(3, 1, &arr) == 0);

    hvr_sparse_arr_insert(1500, 1500, &arr);
    assert(hvr_sparse_arr_contains(1500, 1500, &arr) == 1);
    assert(hvr_sparse_arr_contains(3, 3, &arr) == 1);
    assert(hvr_sparse_arr_contains(4, 4, &arr) == 0);
    assert(hvr_sparse_arr_contains(3, 4, &arr) == 0);
    assert(hvr_sparse_arr_contains(3, 1, &arr) == 0);

    int *tmp_arr = NULL;
    unsigned len = hvr_sparse_arr_linearize_row(3, &tmp_arr, &arr);
    assert(len == 1);
    assert(tmp_arr[0] == 3);

    hvr_sparse_arr_remove(3, 3, &arr);
    assert(hvr_sparse_arr_contains(3, 3, &arr) == 0);
    assert(hvr_sparse_arr_contains(4, 4, &arr) == 0);

    printf("Success!\n");

    return 0;
}
