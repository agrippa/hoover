#include <string.h>
#include <assert.h>

#include "hvr_bloom.h"

void hvr_bloom_init(hvr_bloom_t *blm) {
    memset(blm, 0x00, sizeof(*blm));
}

void hvr_bloom_set(size_t ele, hvr_bloom_t *blm) {
    const size_t offset = (ele % BLOOM_ELEMENTS);
    const bloom_ele_t old_val = blm->filter[offset];
    const bloom_ele_t new_val = old_val + 1;
    assert(new_val > old_val);
    blm->filter[offset] = new_val;
}

void hvr_bloom_remove(size_t ele, hvr_bloom_t *blm) {
    const size_t offset = (ele % BLOOM_ELEMENTS);
    const bloom_ele_t old_val = blm->filter[offset];
    assert(old_val > 0);
    blm->filter[offset] = old_val - 1;
}

int hvr_bloom_check(size_t ele, const hvr_bloom_t *blm) {
    const size_t offset = (ele % BLOOM_ELEMENTS);
    return (blm->filter[offset] > 0);
}
