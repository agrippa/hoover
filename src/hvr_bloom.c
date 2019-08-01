#include <string.h>
#include <assert.h>

#include "hvr_bloom.h"

void hvr_counting_bloom_init(hvr_counting_bloom_t *blm) {
    memset(blm, 0x00, sizeof(*blm));
}

void hvr_counting_bloom_set(size_t ele, hvr_counting_bloom_t *blm) {
    const size_t offset = COUNTING_BLOOM_HASH(ele);
    const counting_bloom_ele_t old_val = blm->filter[offset];
    const counting_bloom_ele_t new_val = old_val + 1;
    assert(new_val > old_val);
    blm->filter[offset] = new_val;
}

void hvr_counting_bloom_remove(size_t ele, hvr_counting_bloom_t *blm) {
    const size_t offset = COUNTING_BLOOM_HASH(ele);
    const counting_bloom_ele_t old_val = blm->filter[offset];
    assert(old_val > 0);
    blm->filter[offset] = old_val - 1;
}

int hvr_counting_bloom_check(size_t ele, const hvr_counting_bloom_t *blm) {
    const size_t offset = COUNTING_BLOOM_HASH(ele);
    return (blm->filter[offset] > 0);
}

void hvr_bloom_init(hvr_bloom_t *blm, size_t n) {
    blm->n = n;
    size_t nelements = (n + BLOOM_ELEMENT_BITS - 1) / BLOOM_ELEMENT_BITS;
    blm->filter = (bloom_ele_t *)malloc(nelements * sizeof(bloom_ele_t));
    assert(blm->filter);
    memset(blm->filter, 0x00, nelements * sizeof(bloom_ele_t));
}

void hvr_bloom_set(size_t ele, hvr_bloom_t *blm) {
    const size_t hash = (ele % blm->n);
    const size_t element_index = hash / BLOOM_ELEMENT_BITS;
    const size_t bit_index = hash % BLOOM_ELEMENT_BITS;
    const size_t bit_mask = ((bloom_ele_t)1 << bit_index);
    blm->filter[element_index] = (blm->filter[element_index] | bit_mask);
}

void hvr_bloom_remove(size_t ele, hvr_bloom_t *blm) {
    const size_t hash = (ele % blm->n);
    const size_t element_index = hash / BLOOM_ELEMENT_BITS;
    const size_t bit_index = hash % BLOOM_ELEMENT_BITS;
    size_t bit_mask = ((bloom_ele_t)1 << bit_index);
    bit_mask = (~bit_mask);

    blm->filter[element_index] = (blm->filter[element_index] & bit_mask);
}

int hvr_bloom_check(size_t ele, hvr_bloom_t *blm) {
    const size_t hash = (ele % blm->n);
    const size_t element_index = hash / BLOOM_ELEMENT_BITS;
    const size_t bit_index = hash % BLOOM_ELEMENT_BITS;
    const size_t bit_mask = ((bloom_ele_t)1 << bit_index);

    if ((blm->filter[element_index] & bit_mask) == 0) {
        return 0;
    } else {
        return 1;
    }
}

