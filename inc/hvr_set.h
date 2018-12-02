#ifndef _HVR_SET_H
#define _HVR_SET_H

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned long long bit_vec_element_type;

/*
 * Utilities used for storing a set of integers. This class is used for storing
 * sets of PEs, of partitions, and is flexible enough to store anything
 * integer-valued.
 *
 * Under the covers, hvr_set_t at its core is a bit vector.
 *
 * HOOVER user code is never expected to create sets, but may be required to
 * manipulate them.
 */
typedef struct _hvr_set_t {
    // Number of elements in the set
    uint64_t nelements;

    // Total number of elements inserted in this cache
    uint64_t n_contained;

    // Backing bit vector
    bit_vec_element_type *bit_vector;
} hvr_set_t;


extern hvr_set_t *hvr_create_empty_set(const unsigned nvals);

extern hvr_set_t *hvr_create_empty_set_symmetric(const uint64_t nvals);

/*
 * Add a given value to this set.
 */
extern int hvr_set_insert(uint64_t val, hvr_set_t *set);

/*
 * Check if a given value exists in this set.
 */
extern int hvr_set_contains(uint64_t val, hvr_set_t *set);

extern int hvr_set_contains_atomic(uint64_t val, hvr_set_t *set);

/*
 * Count how many elements are in this set.
 */
extern unsigned hvr_set_count(hvr_set_t *set);

/*
 * Remove all elements from this set.
 */
extern void hvr_set_wipe(hvr_set_t *set);

/*
 * Free the memory used by the given set.
 */
extern void hvr_set_destroy(hvr_set_t *set);

/*
 * Create a human-readable string from the provided set.
 */
extern void hvr_set_to_string(hvr_set_t *set, char *buf, unsigned buflen,
        unsigned *values);

/*
 * Return an array containing all values in the provided set.
 */
extern uint64_t *hvr_set_non_zeros(hvr_set_t *set,
        uint64_t *n_non_zeros, int *user_must_free);

/*
 * Create a set that stores up to nvals, with all values initially present.
 */
extern hvr_set_t *hvr_create_full_set(const uint64_t nvals);

/*
 * Make 'set' the union of 'set' and 'other'.
 */
extern void hvr_set_merge(hvr_set_t *set, hvr_set_t *other);

/*
 * Copy src to dst.
 */
extern void hvr_set_copy(hvr_set_t *dst, hvr_set_t *src);

#ifdef __cplusplus
}
#endif

#endif // _HVR_SET_H
