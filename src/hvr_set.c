#include <shmem.h>
#include <stdio.h>
#include <string.h>

#include "hvr_set.h"
#include "hvr_common.h"

static hvr_set_t *hvr_create_empty_set_helper(const uint64_t nelements,
        hvr_set_t *set, bit_vec_element_type *bit_vector) {
    set->bit_vector = bit_vector;
    set->nelements = nelements;
    set->n_contained = 0;

    memset(set->bit_vector, 0x00, nelements * sizeof(bit_vec_element_type));

    return set;
}

hvr_set_t *hvr_create_empty_set_symmetric(const uint64_t nvals) {
    const uint64_t bits_per_ele = sizeof(bit_vec_element_type) *
                BITS_PER_BYTE;
    const uint64_t nelements = (nvals + bits_per_ele - 1) / bits_per_ele;
    hvr_set_t *set = (hvr_set_t *)shmem_malloc(sizeof(*set));
    assert(set);
    bit_vec_element_type *bit_vector = (bit_vec_element_type *)shmem_malloc(
            nelements * sizeof(bit_vec_element_type));
    assert(bit_vector);
    return hvr_create_empty_set_helper(nelements, set, bit_vector);
}

hvr_set_t *hvr_create_empty_set(const unsigned nvals) {
    const uint64_t bits_per_ele = sizeof(bit_vec_element_type) *
                BITS_PER_BYTE;
    const uint64_t nelements = (nvals + bits_per_ele - 1) / bits_per_ele;
    hvr_set_t *set = (hvr_set_t *)malloc(sizeof(*set));
    assert(set);
    bit_vec_element_type *bit_vector = (bit_vec_element_type *)malloc(
            nelements * sizeof(bit_vec_element_type));
    assert(bit_vector);
    return hvr_create_empty_set_helper(nelements, set, bit_vector);
}

void hvr_fill_set(hvr_set_t *s) {
    memset(s->bit_vector, 0xff, s->nelements * sizeof(*(s->bit_vector)));
}

hvr_set_t *hvr_create_full_set(const uint64_t nvals) {
    hvr_set_t *empty_set = hvr_create_empty_set(nvals);
    hvr_fill_set(empty_set);
    return empty_set;
}

static int hvr_set_insert_internal(uint64_t val,
        bit_vec_element_type *bit_vector) {
    const uint64_t element = val / (sizeof(*bit_vector) * BITS_PER_BYTE);
    const uint64_t bit = val % (sizeof(*bit_vector) * BITS_PER_BYTE);
    const bit_vec_element_type old_val = bit_vector[element];
    const bit_vec_element_type new_val =
        (old_val | ((bit_vec_element_type)1 << bit));
    bit_vector[element] = new_val;
    return old_val != new_val;
}

int hvr_set_insert(uint64_t val, hvr_set_t *set) {
    const int changed = hvr_set_insert_internal(val, set->bit_vector);
    if (changed) {
        set->n_contained++;
    }
    return changed;
}

void hvr_set_wipe(hvr_set_t *set) {
    memset(set->bit_vector, 0x00, set->nelements * sizeof(*(set->bit_vector)));
    set->n_contained = 0;
}

int hvr_set_contains(uint64_t val, hvr_set_t *set) {
    bit_vec_element_type *bit_vector = set->bit_vector;
    const uint64_t element = val / (sizeof(*bit_vector) * BITS_PER_BYTE);
    const uint64_t bit = val % (sizeof(*bit_vector) * BITS_PER_BYTE);
    const bit_vec_element_type old_val = bit_vector[element];
    if (old_val & ((bit_vec_element_type)1 << bit)) {
        return 1;
    } else {
        return 0;
    }
}

int hvr_set_contains_atomic(uint64_t val, hvr_set_t *set) {
    bit_vec_element_type *bit_vector = set->bit_vector;
    const uint64_t element = val / (sizeof(*bit_vector) * BITS_PER_BYTE);
    const uint64_t bit = val % (sizeof(*bit_vector) * BITS_PER_BYTE);

    assert(sizeof(unsigned long long) == sizeof(bit_vec_element_type));
    const bit_vec_element_type old_val = shmem_ulonglong_atomic_fetch(
            bit_vector + element, shmem_my_pe());
    if (old_val & ((bit_vec_element_type)1 << bit)) {
        return 1;
    } else {
        return 0;
    }
}

unsigned hvr_set_count(hvr_set_t *set) {
    return set->n_contained;
}

void hvr_set_destroy(hvr_set_t *set) {
    free(set->bit_vector);
    free(set);
}

void hvr_set_to_string(hvr_set_t *set, char *buf, unsigned buflen,
        unsigned *values) {
    int offset = snprintf(buf, buflen, "{");

    const size_t nvals = set->nelements * sizeof(bit_vec_element_type) *
        BITS_PER_BYTE;
    for (unsigned i = 0; i < nvals; i++) {
        if (hvr_set_contains(i, set)) {
            offset += snprintf(buf + offset, buflen - offset - 1, " %u", i);
            assert(offset < buflen);
            if (values) {
                offset += snprintf(buf + offset, buflen - offset - 1, ": %u",
                        values[i]);
                assert(offset < buflen);
            }
        }
    }

    snprintf(buf + offset, buflen - offset - 1, " }");
}

void hvr_set_merge(hvr_set_t *set, hvr_set_t *other) {
    assert(set->nelements == other->nelements);
    assert(sizeof(unsigned long long) == sizeof(bit_vec_element_type));

    for (int i = 0; i < set->nelements; i++) {
        (set->bit_vector)[i] = ((set->bit_vector)[i] | (other->bit_vector)[i]);
    }
}

void hvr_set_copy(hvr_set_t *dst, hvr_set_t *src) {
    memcpy(dst->bit_vector, src->bit_vector,
            src->nelements * sizeof(bit_vec_element_type));
    memcpy(dst, src, offsetof(hvr_set_t, bit_vector));
}
