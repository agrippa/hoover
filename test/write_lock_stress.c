#include <shmem.h>
#include <unistd.h>
#include <stdlib.h>
#include "shmem_rw_lock.h"

int main(int argc, char **argv) {
    shmem_init();

    long *lock = hvr_rwlock_create_n(1);
    long *val = shmem_malloc(sizeof(long));
    *val = 0;
    shmem_barrier_all();

    long my_val = shmem_my_pe();
    long tmp_val;

    for (int iter = 0; iter < 10; iter++) {
        hvr_rwlock_wlock(lock, 0);
        shmem_putmem(val, &my_val, sizeof(my_val), 0);
        sleep(1);
        shmem_getmem(&tmp_val, val, sizeof(tmp_val), 0);
        assert(tmp_val == my_val);
        hvr_rwlock_wunlock(lock, 0);
    }

    shmem_finalize();
    return 0;
}
