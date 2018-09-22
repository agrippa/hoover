#include <shmem.h>
#include <string.h>
#include <limits.h>

#include "hvr_common.h"
#include "hvr_dist_bitvec.h"

#define BITS_PER_ELE (sizeof(hvr_dist_bitvec_ele_t) * BITS_PER_BYTE)

void hvr_dist_bitvec_init(hvr_dist_bitvec_size_t dim0,
        hvr_dist_bitvec_size_t dim1, hvr_dist_bitvec_t *vec) {
    vec->dim0 = dim0;
    vec->dim1 = dim1;

    vec->dim1_length_in_words = (dim1 + BITS_PER_ELE - 1) / BITS_PER_ELE;

    vec->dim0_per_pe = (dim0 + shmem_n_pes() - 1) / shmem_n_pes();

    vec->symm_vec = (hvr_dist_bitvec_ele_t *)shmem_malloc(vec->dim0_per_pe *
            vec->dim1_length_in_words * sizeof(hvr_dist_bitvec_ele_t));
    assert(vec->symm_vec);
    memset(vec->symm_vec, 0x00, vec->dim0_per_pe * vec->dim1_length_in_words *
            sizeof(hvr_dist_bitvec_ele_t));
    shmem_barrier_all();
}

void hvr_dist_bitvec_set(hvr_dist_bitvec_size_t coord0,
        hvr_dist_bitvec_size_t coord1, hvr_dist_bitvec_t *vec) {
    const unsigned coord0_pe = coord0 / vec->dim0_per_pe;
    const unsigned coord0_offset = coord0 % vec->dim0_per_pe;

    const unsigned coord1_word = coord1 / BITS_PER_ELE;
    const unsigned coord1_bit = coord1 % BITS_PER_WORD;
    unsigned coord1_mask = (1U << coord1_bit);

    assert(sizeof(hvr_dist_bitvec_ele_t) == sizeof(unsigned int));
    shmem_uint_atomic_or(
            vec->symm_vec + (coord0_offset * vec->dim1_length_in_words) +
            coord1_word, coord1_mask, coord0_pe);
}

void hvr_dist_bitvec_clear(hvr_dist_bitvec_size_t coord0,
        hvr_dist_bitvec_size_t coord1, hvr_dist_bitvec_t *vec) {
    const unsigned coord0_pe = coord0 / vec->dim0_per_pe;
    const unsigned coord0_offset = coord0 % vec->dim0_per_pe;

    const unsigned coord1_word = coord1 / BITS_PER_ELE;
    const unsigned coord1_bit = coord1 % BITS_PER_WORD;
    unsigned coord1_mask = (1U << coord1_bit);
    coord1_mask = ~coord1_mask;

    assert(sizeof(hvr_dist_bitvec_ele_t) == sizeof(unsigned int));
    shmem_uint_atomic_and(
            vec->symm_vec + (coord0_offset * vec->dim1_length_in_words) +
            coord1_word, coord1_mask, coord0_pe);
}

void hvr_dist_bitvec_local_subcopy_init(hvr_dist_bitvec_t *vec,
        hvr_dist_bitvec_local_subcopy_t *out) {
    out->coord0 = UINT_MAX;
    out->dim1 = vec->dim1;
    out->dim1_length_in_words = vec->dim1_length_in_words;
    out->subvec = (hvr_dist_bitvec_ele_t *)malloc(
            vec->dim1_length_in_words * sizeof(*(out->subvec)));
    assert(out->subvec);
    memset(out->subvec, 0x00,
            vec->dim1_length_in_words * sizeof(*(out->subvec)));
}

void hvr_dist_bitvec_copy_locally(hvr_dist_bitvec_size_t coord0,
        hvr_dist_bitvec_t *vec, hvr_dist_bitvec_local_subcopy_t *out) {
    const unsigned coord0_pe = coord0 / vec->dim0_per_pe;
    const unsigned coord0_offset = coord0 % vec->dim0_per_pe;

    out->coord0 = coord0;
    for (unsigned i = 0; i < vec->dim1_length_in_words; i++) {
        out->subvec[i] = shmem_uint_atomic_fetch(
                vec->symm_vec + (coord0_offset * vec->dim1_length_in_words + i),
                coord0_pe);
    }
}

int hvr_dist_bitvec_local_subcopy_contains(hvr_dist_bitvec_size_t coord1,
        hvr_dist_bitvec_local_subcopy_t *vec) {
    const unsigned coord1_word = coord1 / BITS_PER_ELE;
    const unsigned coord1_bit = coord1 % BITS_PER_WORD;
    unsigned coord1_mask = (1U << coord1_bit);
    if (vec->subvec[coord1_word] & coord1_mask) {
        return 1;
    } else {
        return 0;
    }
}

int hvr_dist_bitvec_owning_pe(hvr_dist_bitvec_size_t coord0,
        hvr_dist_bitvec_t *vec) {
    return coord0 / vec->dim0_per_pe;
}

void hvr_dist_bitvec_my_chunk(hvr_dist_bitvec_size_t *lower,
        hvr_dist_bitvec_size_t *upper, hvr_dist_bitvec_t *vec) {
    *lower = shmem_my_pe() * vec->dim0_per_pe;
    *upper = (shmem_my_pe() + 1) * vec->dim0_per_pe;

    if (*lower > vec->dim0) *lower = vec->dim0;
    if (*upper > vec->dim0) *upper = vec->dim0;
}

void hvr_dist_bitvec_local_subcopy_copy(hvr_dist_bitvec_local_subcopy_t *dst,
        hvr_dist_bitvec_local_subcopy_t *src) {
    assert(dst->dim1 == src->dim1);
    assert(dst->dim1_length_in_words == src->dim1_length_in_words);

    dst->coord0 = src->coord0;
    memcpy(dst->subvec, src->subvec,
            src->dim1_length_in_words * sizeof(hvr_dist_bitvec_ele_t));
}

void hvr_dist_bitvec_local_subcopy_destroy(hvr_dist_bitvec_local_subcopy_t *c) {
    free(c->subvec);
    free(c);
}