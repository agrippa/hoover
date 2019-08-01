#ifndef _HVR_BLOOM_H
#define _HVR_BLOOM_H

#include <stdint.h>
#include <stdlib.h>

typedef uint8_t counting_bloom_ele_t;
#define COUNTING_BLOOM_HASH_BITS 8
#define COUNTING_BLOOM_ELEMENTS (1 << COUNTING_BLOOM_HASH_BITS)

#define COUNTING_BLOOM_HASH(my_ele) ((my_ele) & (COUNTING_BLOOM_ELEMENTS - 1))

typedef struct _hvr_counting_bloom_t {
    counting_bloom_ele_t filter[COUNTING_BLOOM_ELEMENTS];
} hvr_counting_bloom_t;

void hvr_counting_bloom_init(hvr_counting_bloom_t *blm);

void hvr_counting_bloom_set(size_t ele, hvr_counting_bloom_t *blm);

void hvr_counting_bloom_remove(size_t ele, hvr_counting_bloom_t *blm);

int hvr_counting_bloom_check(size_t ele, const hvr_counting_bloom_t *blm);


typedef uint32_t bloom_ele_t;
#define BLOOM_ELEMENT_BYTES sizeof(bloom_ele_t)
#define BLOOM_ELEMENT_BITS (8 * BLOOM_ELEMENT_BYTES)

typedef struct _hvr_bloom_t {
    bloom_ele_t *filter;
    size_t n;
} hvr_bloom_t;

void hvr_bloom_init(hvr_bloom_t *blm, size_t n);

void hvr_bloom_set(size_t ele, hvr_bloom_t *blm);

void hvr_bloom_remove(size_t ele, hvr_bloom_t *blm);

int hvr_bloom_check(size_t ele, hvr_bloom_t *blm);

#endif // _HVR_BLOOM_H
