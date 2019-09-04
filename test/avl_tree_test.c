#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include "hvr_avl_tree.h"

int main(int argc, char **argv) {

    size_t pool_size = 1024 * 1024;
    void *pool = malloc(pool_size);
    assert(pool);
    mspace tracker = create_mspace_with_base(pool, pool_size, 0);
    struct node *tree = nnil;

    for (unsigned i = 0; i < 1024; i++) {
        hvr_avl_insert(&tree, i, tracker);

        for (unsigned j = 0; j <= i; j++) {
            struct node *exists = hvr_avl_find(tree, j);
            assert(exists != nnil);
        }
    }

    for (unsigned i = 0; i < 1024; i++) {
        hvr_avl_delete(&tree, i, tracker);

        for (unsigned j = 0; j <= i; j++) {
            struct node *exists = hvr_avl_find(tree, j);
            assert(exists == nnil);
        }

        for (unsigned j = i+1; j < 1024; j++) {
            struct node *exists = hvr_avl_find(tree, j);
            assert(exists != nnil);
        }
    }

    printf("Success!\n");

    return 0;
}
