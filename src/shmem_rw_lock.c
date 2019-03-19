/* For license: see LICENSE.txt file at top-level */

#include <shmem.h>
#include <string.h>

#include "shmem_rw_lock.h"


/*
 * In the long, the topmost bit is used to indicate a write lock request, the
 * remainder of the bits are used to count readers.
 */
long *hvr_rwlock_create_n(const int n) {
    long *lock = (long *)shmem_malloc(n * sizeof(*lock));
    assert(lock);
    memset(lock, 0x00, n * sizeof(*lock));
    shmem_barrier_all();
    return lock;
}

void hvr_rwlock_rlock(long *lock, const int target_pe) {
    long curr_val;
    shmem_getmem(&curr_val, lock, sizeof(curr_val), target_pe);

    while (1) {
        if (has_writer_request(curr_val)) {
            /*
             * Don't add readers as long as there's a pending write request.
             * Just update our current value until it clears.
             */
            shmem_getmem(&curr_val, lock, sizeof(curr_val), target_pe);
        } else {
            // No write request
            long new_val = add_reader(curr_val);
            long old_val = SHMEM_LONG_CSWAP(lock, curr_val, new_val, target_pe);
            if (old_val == curr_val) {
                return;
            } else {
                // Retry
                curr_val = old_val;
            }
        }
    }
}

void hvr_rwlock_runlock(long *lock, const int target_pe) {
    long curr_val;
    shmem_getmem(&curr_val, lock, sizeof(curr_val), target_pe);

    while (1) {
        long new_val = remove_reader(curr_val);
        long old_val = SHMEM_LONG_CSWAP(lock, curr_val, new_val, target_pe);
        if (old_val == curr_val) {
            return;
        } else {
            // Retry
            curr_val = old_val;
        }
    }
}

void hvr_rwlock_wlock(long *lock, const int target_pe) {
    long curr_val;
    shmem_getmem(&curr_val, lock, sizeof(curr_val), target_pe);

    while (1) {
        if (has_writer_request(curr_val)) {
            /*
             * Can't set while someone else has the writer lock, so just keep
             * polling.
             */
            shmem_getmem(&curr_val, lock, sizeof(curr_val), target_pe);
        } else {
            long new_val = set_writer(curr_val);
            long old_val = SHMEM_LONG_CSWAP(lock, curr_val, new_val, target_pe);
            if (old_val == curr_val) {
                break;
            } else {
                // Retry
                curr_val = old_val;
            }
        }
    }

    // Wait for all readers to exit the critical section
    long n = nreaders(curr_val);
    while (n > 0) {
        shmem_getmem(&curr_val, lock, sizeof(curr_val), target_pe);
        n = nreaders(curr_val);
    }
}

void hvr_rwlock_wunlock(long *lock, const int target_pe) {
    long curr_val;
    shmem_getmem(&curr_val, lock, sizeof(curr_val), target_pe);

    while (1) {
        long new_val = clear_writer(curr_val);
        long old_val = SHMEM_LONG_CSWAP(lock, curr_val, new_val, target_pe);
        if (old_val == curr_val) {
            return;
        } else {
            // Retry
            curr_val = old_val;
        }
    }
}

