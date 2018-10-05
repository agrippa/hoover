#include "hvr_vertex_ll.h"

void hvr_vertex_ll_init(hvr_vertex_ll_t *l, size_t capacity) {
    hvr_vertex_ll_node_t *allocated = (hvr_vertex_ll_node_t *)malloc(
            capacity * sizeof(*allocated));
    assert(allocated);

    l->pool = allocated;
    for (int i = 0; i < capacity - 1; i++) {
        allocated[i].next = allocated + (i + 1);
    }
    allocated[capacity - 1].next = NULL;

    l->head = NULL;

    l->capacity = capacity;
    l->length = 0;
    l->allocated = allocated;
}

void hvr_vertex_ll_destroy(hvr_vertex_ll_t *l) {
    free(l->allocated);
}

void hvr_vertex_ll_push(hvr_vertex_t *vert, hvr_vertex_ll_t *l) {
    hvr_vertex_ll_node_t *node = l->pool;
    assert(node);
    node->vert = vert;

    l->pool = node->next;
    node->next = l->head;
    l->head = node;

    l->length += 1;
}

hvr_vertex_t *hvr_vertex_ll_pop(hvr_vertex_ll_t *l) {
    if (l->head == NULL) {
        assert(l->length == 0);
        return NULL;
    } else {
        hvr_vertex_ll_node_t *node = l->head;
        hvr_vertex_t *result = node->vert;

        l->head = node->next;

        node->next = l->pool;
        l->pool = node;

        l->length -= 1;

        return result;
    }
}

