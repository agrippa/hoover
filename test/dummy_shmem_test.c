#include <stdio.h>
#include <stdlib.h>
#include <shmem.h>

int main(int argc, char **argv) {
    shmem_init();
    int pe = shmem_my_pe();
    int npes = shmem_n_pes();
    if (pe == 0) printf("Running with %d PEs\n", npes);
    shmem_finalize();
    return 0;
}
