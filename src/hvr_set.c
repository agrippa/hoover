#include <shmem.h>
#include <stdio.h>
#include <string.h>

#include "hvr_set.h"
#include "hvr_common.h"

static hvr_set_t *hvr_create_empty_set_helper(const uint64_t nelements,
        const uint64_t max_n_contained, hvr_set_t *set,
        bit_vec_element_type *bit_vector) {
    set->n_contained = 0;
    set->max_n_contained = max_n_contained;
    set->bit_vector = bit_vector;
    set->bit_vector_len = nelements;

    memset(set->bit_vector, 0x00, nelements * sizeof(bit_vec_element_type));

    return set;
}

hvr_set_t *hvr_create_empty_set(const unsigned nvals) {
    const uint64_t bits_per_ele = sizeof(bit_vec_element_type) *
                BITS_PER_BYTE;
    const uint64_t nelements = (nvals + bits_per_ele - 1) / bits_per_ele;
    hvr_set_t *set = (hvr_set_t *)malloc_helper(sizeof(*set));
    assert(set);
    bit_vec_element_type *bit_vector = (bit_vec_element_type *)malloc_helper(
            nelements * sizeof(bit_vec_element_type));
    assert(bit_vector);
    return hvr_create_empty_set_helper(nelements, nvals, set, bit_vector);
}

void hvr_set_fill(hvr_set_t *s) {
    memset(s->bit_vector, 0xff, s->bit_vector_len * sizeof(*(s->bit_vector)));
    s->n_contained = s->max_n_contained;
}

hvr_set_t *hvr_create_full_set(const uint64_t nvals) {
    hvr_set_t *empty_set = hvr_create_empty_set(nvals);
    hvr_set_fill(empty_set);
    return empty_set;
}

int hvr_set_insert(uint64_t val, hvr_set_t *set) {
    bit_vec_element_type *bit_vector = set->bit_vector;
    const uint64_t element = val / (sizeof(*bit_vector) * BITS_PER_BYTE);
    const uint64_t bit = val % (sizeof(*bit_vector) * BITS_PER_BYTE);
    const bit_vec_element_type old_val = bit_vector[element];
    const bit_vec_element_type new_val =
        (old_val | ((bit_vec_element_type)1 << bit));
    bit_vector[element] = new_val;

    if (old_val != new_val) {
        set->n_contained++;
    }
    return (old_val != new_val);
}

#ifdef HVR_MULTITHREADED
int hvr_set_insert_atomic(uint64_t val, hvr_set_t *set) {
    bit_vec_element_type *bit_vector = set->bit_vector;
    const uint64_t element = val / (sizeof(*bit_vector) * BITS_PER_BYTE);
    const uint64_t bit = val % (sizeof(*bit_vector) * BITS_PER_BYTE);
    const bit_vec_element_type mask = ((bit_vec_element_type)1 << bit);
    const bit_vec_element_type old_val = __sync_fetch_and_or(
            &bit_vector[element], mask);
    const bit_vec_element_type new_val = (old_val | mask);

    if (old_val != new_val) {
        __sync_fetch_and_add(&(set->n_contained), 1);
    }
    return (old_val != new_val);

}
#endif

static int hvr_set_clear_internal(uint64_t val,
        bit_vec_element_type *bit_vector) {
    const uint64_t element = val / (sizeof(*bit_vector) * BITS_PER_BYTE);
    const uint64_t bit = val % (sizeof(*bit_vector) * BITS_PER_BYTE);
    const bit_vec_element_type old_val = bit_vector[element];
    const bit_vec_element_type new_val =
        (old_val & (~((bit_vec_element_type)1 << bit)));
    bit_vector[element] = new_val;
    return old_val != new_val;
}

int hvr_set_clear(uint64_t val, hvr_set_t *set) {
    const int changed = hvr_set_clear_internal(val, set->bit_vector);
    if (changed) {
        set->n_contained--;
    }
    return changed;
}

void hvr_set_wipe(hvr_set_t *set) {
    memset(set->bit_vector, 0x00,
            set->bit_vector_len * sizeof(*(set->bit_vector)));
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

uint64_t hvr_set_count(hvr_set_t *set) {
    return set->n_contained;
}

uint64_t hvr_set_min_contained(hvr_set_t *s) {
    assert(hvr_set_count(s) > 0);
    for (uint64_t i = 0; i < s->max_n_contained; i++) {
        if (hvr_set_contains(i, s)) {
            return i;
        }
    }
    abort();
}

uint64_t hvr_set_max_contained(hvr_set_t *s) {
    assert(hvr_set_count(s) > 0);
    uint64_t i = s->max_n_contained - 1U;
    while (1) {
        if (hvr_set_contains(i, s)) {
            return i;
        }
        if (i == 0) break;
        i--;
    }
    abort();
}

void hvr_set_destroy(hvr_set_t *set) {
    free(set->bit_vector);
    free(set);
}

void hvr_set_to_string(hvr_set_t *set, char *buf, unsigned buflen,
        unsigned *values) {
    int offset = snprintf(buf, buflen, "{");

    const size_t nvals = set->bit_vector_len * sizeof(bit_vec_element_type) *
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
    assert(set->bit_vector_len == other->bit_vector_len);

    for (int i = 0; i < set->bit_vector_len; i++) {
        (set->bit_vector)[i] = ((set->bit_vector)[i] | (other->bit_vector)[i]);
    }

    uint64_t new_count = 0;
    for (uint64_t i = 0; i < set->max_n_contained; i++) {
        if (hvr_set_contains(i, set)) {
            new_count++;
        }
    }
    set->n_contained = new_count;
}

void hvr_set_copy(hvr_set_t *dst, hvr_set_t *src) {
    assert(dst->bit_vector_len == src->bit_vector_len);
    assert(dst->max_n_contained == src->max_n_contained);

    dst->n_contained = src->n_contained;
    memcpy(dst->bit_vector, src->bit_vector,
            src->bit_vector_len * sizeof(bit_vec_element_type));
}

int hvr_set_equal(hvr_set_t *a, hvr_set_t *b) {
    assert(a->max_n_contained == b->max_n_contained);
    if (a->n_contained != b->n_contained) {
        return 0;
    }

    for (uint64_t i = 0; i < a->max_n_contained; i++) {
        if (hvr_set_contains(i, a) != hvr_set_contains(i, b)) {
            return 0;
        }
    }
    return 1;
}

void hvr_set_or(hvr_set_t *dst, hvr_set_t *src) {
    assert(dst->max_n_contained == src->max_n_contained);
    assert(dst->bit_vector_len == src->bit_vector_len);

    for (uint64_t i = 0; i < dst->bit_vector_len; i++) {
        (dst->bit_vector)[i] = ((dst->bit_vector)[i] | (src->bit_vector)[i]);
    }

    unsigned count_contained = 0;
    for (unsigned i = 0; i < dst->max_n_contained; i++) {
        if (hvr_set_contains(i, dst)) {
            count_contained++;
        }
    }
    dst->n_contained = count_contained;
}

void hvr_set_and(hvr_set_t *dst, hvr_set_t *src1, hvr_set_t *src2) {
    assert(dst->max_n_contained == src1->max_n_contained);
    assert(dst->max_n_contained == src2->max_n_contained);

    assert(dst->bit_vector_len == src1->bit_vector_len);
    assert(dst->bit_vector_len == src2->bit_vector_len);

    for (uint64_t i = 0; i < dst->bit_vector_len; i++) {
        (dst->bit_vector)[i] = ((src1->bit_vector)[i] & (src2->bit_vector)[i]);
    }

    unsigned count_contained = 0;
    for (unsigned i = 0; i < dst->max_n_contained; i++) {
        if (hvr_set_contains(i, dst)) {
            count_contained++;
        }
    }
    dst->n_contained = count_contained;
}
