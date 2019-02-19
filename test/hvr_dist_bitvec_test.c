#include <stdio.h>
#include <shmem.h>
#include <assert.h>

#include "hvr_dist_bitvec.h"

int main(int argc, char **argv) {
    shmem_init();

    if (shmem_my_pe() == 0) {
        fprintf(stderr, "Running with %d PEs, sizeof(unsigned)=%lu\n",
                shmem_n_pes(), sizeof(unsigned));
    }

    const int N = 2000;
    hvr_dist_bitvec_t vec;
    hvr_dist_bitvec_init(N, shmem_n_pes(), &vec);

    for (unsigned i = 0; i < N; i += 2) {
        hvr_dist_bitvec_set(i, shmem_my_pe(), &vec);
    }

    shmem_barrier_all();

    hvr_dist_bitvec_local_subcopy_t copy;
    hvr_dist_bitvec_local_subcopy_init(&vec, &copy);

    for (unsigned i = 0; i < N; i += 2) {
        for (int j = 0; j < shmem_n_pes(); j++) {
            hvr_dist_bitvec_copy_locally(i, &vec, &copy);
            assert(hvr_dist_bitvec_local_subcopy_contains(j, &copy));
        }
    }

    shmem_barrier_all();

    for (unsigned i = 0; i < N; i += 2) {
        if (shmem_my_pe() % 2 == 0) {
            hvr_dist_bitvec_clear(i, shmem_my_pe(), &vec);
        }
    }

    shmem_barrier_all();

    for (unsigned i = 0; i < N; i += 2) {
        for (int j = 0; j < shmem_n_pes(); j++) {
            hvr_dist_bitvec_copy_locally(i, &vec, &copy);

            if (j % 2 == 0) {
                // Should have been cleared
                assert(!hvr_dist_bitvec_local_subcopy_contains(j, &copy));
            } else {
                // Still set
                assert(hvr_dist_bitvec_local_subcopy_contains(j, &copy));
            }
        }
    }

    shmem_barrier_all();

    for (unsigned i = 0; i < N; i += 2) {
        if (shmem_my_pe() % 2 == 1) {
            hvr_dist_bitvec_clear(i, shmem_my_pe(), &vec);
        }
    }

    shmem_barrier_all();

    for (unsigned i = 0; i < N; i += 2) {
        for (int j = 0; j < shmem_n_pes(); j++) {
            hvr_dist_bitvec_copy_locally(i, &vec, &copy);

            // Should all have been cleared
            assert(!hvr_dist_bitvec_local_subcopy_contains(j, &copy));
        }
    }

    shmem_barrier_all();

    hvr_dist_bitvec_t vec2;
    hvr_dist_bitvec_init(16000000, shmem_n_pes(), &vec2);

    hvr_dist_bitvec_local_subcopy_t copy2;
    hvr_dist_bitvec_local_subcopy_init(&vec2, &copy2);

    for (unsigned i = 0; i < 16000000; i += 5) {
        hvr_dist_bitvec_set(i, shmem_my_pe(), &vec2);
    }

    for (unsigned i = 0; i < 16000000; i += 20) {
        for (int j = 0; j < shmem_n_pes(); j++) {
            hvr_dist_bitvec_copy_locally(i, &vec2, &copy2);
        }
    }

    shmem_barrier_all();

    if (shmem_my_pe() == 0) {
        printf("Done!\n");
    }

    shmem_finalize();

    return 0;
}
