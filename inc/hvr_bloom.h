#ifndef _HVR_BLOOM_H
#define _HVR_BLOOM_H

#include <stdint.h>
#include <stdlib.h>

typedef uint8_t bloom_ele_t;
#define BLOOM_ELEMENTS 512

typedef struct _hvr_bloom_t {
    bloom_ele_t filter[BLOOM_ELEMENTS];
} hvr_bloom_t;

void hvr_bloom_init(hvr_bloom_t *blm);

void hvr_bloom_set(size_t ele, hvr_bloom_t *blm);

void hvr_bloom_remove(size_t ele, hvr_bloom_t *blm);

int hvr_bloom_check(size_t ele, const hvr_bloom_t *blm);

#endif // _HVR_BLOOM_H
