#include <stdio.h>
#include <shmem.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#include "hvr_mailbox.h"
#include "hoover.h"

#define MAILBOX_SIZE 32

int main(int argc, char **argv) {
    shmem_init();
    hvr_mailbox_t mailbox;
    hvr_mailbox_init(&mailbox, MAILBOX_SIZE);

    assert(shmem_n_pes() >= 2);

    void *msg = NULL;
    size_t msg_capacity = 0;
    size_t msg_len = 0;

    if (shmem_my_pe() == 0) {
        int msg = 42;
        fprintf(stderr, "PE 0 sending message...\n");
        hvr_mailbox_send(&msg, sizeof(msg), 1, -1, &mailbox, NULL);
        fprintf(stderr, "PE 0 done sending message...\n");
    } else if (shmem_my_pe() == 1) {
        sleep(10);
        int success = hvr_mailbox_recv(&msg, &msg_capacity, &msg_len, &mailbox);
        assert(success);
        assert(msg_len == sizeof(int));
        assert(*((int *)msg) == 42);
    }

    shmem_barrier_all();

    for (uint64_t i = 0; i < 100; i++) {
        if (shmem_my_pe() == 0) {
            int success = hvr_mailbox_send(&i, sizeof(i), 1, -1, &mailbox,
                    NULL);
            assert(success);
        } else if (shmem_my_pe() == 1) {
            int success = hvr_mailbox_recv(&msg, &msg_capacity, &msg_len,
                    &mailbox);
            while (!success) {
                success = hvr_mailbox_recv(&msg, &msg_capacity, &msg_len,
                        &mailbox);
            }
            assert(msg_len == sizeof(i));
            assert(i == *((uint64_t *)msg));
        }
    }

    shmem_barrier_all();

    for (size_t msg_size = sizeof(unsigned);
            msg_size < MAILBOX_SIZE - sizeof(unsigned) - sizeof(size_t);
            msg_size += sizeof(unsigned)) {
        if (shmem_my_pe() == 0) {
            unsigned char *buf = (unsigned char *)malloc(msg_size);
            assert(buf);
            for (int i = 0; i < msg_size; i++) buf[i] = i;
            int success = hvr_mailbox_send(buf, msg_size, 1, -1, &mailbox,
                    NULL);
            assert(success);
            free(buf);
        } else if (shmem_my_pe() == 1) {
            int success = hvr_mailbox_recv(&msg, &msg_capacity, &msg_len,
                    &mailbox);
            while (!success) {
                success = hvr_mailbox_recv(&msg, &msg_capacity, &msg_len,
                        &mailbox);
            }
            assert(msg_len == msg_size);

            for (int i = 0; i < msg_size; i++) {
                assert(((unsigned char *)msg)[i] == i);
            }
        }
    }

    shmem_barrier_all();

    hvr_mailbox_t big_mailbox;
    hvr_mailbox_init(&big_mailbox,   256 * 1024 * 1024);

    if (shmem_my_pe() == 0) {
        fprintf(stderr, "sizeof(hvr_vertex_update_t) = %lu\n",
                sizeof(hvr_vertex_update_t));
    }

    unsigned char *buf = (unsigned char *)malloc(sizeof(hvr_vertex_update_t));
    assert(buf);
    for (int i = 0; i < sizeof(hvr_vertex_update_t); i++) {
        buf[i] = i;
    }

    int target = 0;
    unsigned long long start_time = hvr_current_time_us();
    while (hvr_current_time_us() - start_time < 60ULL * 1000ULL * 1000ULL) {
        hvr_mailbox_send(buf, sizeof(hvr_vertex_update_t), target, 100,
                &big_mailbox, NULL);
        target = (target + 1) % shmem_n_pes();

        int success = hvr_mailbox_recv(&msg, &msg_capacity, &msg_len,
                &big_mailbox);
        if (success) {
            assert(msg_len == sizeof(hvr_vertex_update_t));
            for (int i = 0; i < sizeof(hvr_vertex_update_t); i++) {
                assert(((unsigned char *)msg)[i] == (unsigned char)i);
            }
        }
    }

    shmem_barrier_all();

    if (shmem_my_pe() == 0) {
        printf("Success!\n");
    }

    shmem_finalize();
    return 0;
}
