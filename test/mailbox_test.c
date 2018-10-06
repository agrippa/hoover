#include <stdio.h>
#include <shmem.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>

#include "hvr_mailbox.h"

int main(int argc, char **argv) {
    shmem_init();
    hvr_mailbox_t mailbox;
    hvr_mailbox_init(&mailbox, 18);

    assert(shmem_n_pes() >= 2);

    void *msg = NULL;
    size_t msg_capacity = 0;
    size_t msg_len = 0;

    if (shmem_my_pe() == 0) {
        int msg = 42;
        fprintf(stderr, "PE 0 sending message...\n");
        hvr_mailbox_send(&msg, sizeof(msg), 1, -1, &mailbox);
        fprintf(stderr, "PE 0 done sending message...\n");
    } else if (shmem_my_pe() == 1) {
        sleep(10);
        int success = hvr_mailbox_recv(&msg, &msg_capacity, &msg_len, &mailbox);
        assert(success);
        assert(msg_len == sizeof(int));
        assert(*((int *)msg) == 42);
    }

    shmem_barrier_all();

    for (unsigned i = 0; i < 100; i++) {
        if (shmem_my_pe() == 0) {
            hvr_mailbox_send(&i, sizeof(i), 1, -1, &mailbox);
        } else if (shmem_my_pe() == 1) {
            int success = hvr_mailbox_recv(&msg, &msg_capacity, &msg_len,
                    &mailbox);
            while (!success) {
                success = hvr_mailbox_recv(&msg, &msg_capacity, &msg_len,
                        &mailbox);
            }
            uint64_t temp;
            memcpy(&temp, mailbox.indices, sizeof(temp));
            assert(msg_len == sizeof(i));
            assert(i == *((unsigned *)msg));
        }
    }

    shmem_barrier_all();

    shmem_finalize();
    return 0;
}
