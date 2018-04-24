/* For license: see LICENSE.txt file at top-level */

#ifndef _SHMEM_RW_LOCK_H
#define _SHMEM_RW_LOCK_H

#include <assert.h>

#define BITS_PER_BYTE 8
#define WRITER_BIT (((long)1) << (sizeof(long) * BITS_PER_BYTE - 1))

static inline long nreaders(const long lock_val) {
    const long mask = ~(((long)1) << (sizeof(long) * BITS_PER_BYTE - 1));
    return (lock_val & mask);
}

static inline int has_writer_request(const long lock_val) {
    return (WRITER_BIT & lock_val) != 0;
}

static inline long set_writer(const long lock_val) {
    assert(has_writer_request(lock_val) == 0);
    return (WRITER_BIT | lock_val);
}

static inline long clear_writer(const long lock_val) {
    assert(has_writer_request(lock_val) == 1);
    return (~WRITER_BIT) & lock_val;
}

static inline long add_reader(const long lock_val) {
    assert(has_writer_request(lock_val) == 0);
    return nreaders(lock_val) + 1;
}

static inline long remove_reader(const long lock_val) {
    assert(nreaders(lock_val) > 0);
    return ((has_writer_request(lock_val) ? WRITER_BIT : 0) |
        (nreaders(lock_val) - 1));
}

long *hvr_rwlock_create_n(const int n);
void hvr_rwlock_rlock(long *lock, const int target_pe);
void hvr_rwlock_runlock(long *lock, const int target_pe);
void hvr_rwlock_wlock(long *lock, const int target_pe);
void hvr_rwlock_wunlock(long *lock, const int target_pe);

#endif // _SHMEM_RW_LOCK_H
