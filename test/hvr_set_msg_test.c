#include <stdio.h>
#include <shmem.h>

#include "hvr_set_msg.h"

int main(int argc, char **argv) {
    shmem_init();
    int pe = shmem_my_pe();
    int npes = shmem_n_pes();
    assert(npes > 1);

#define NSETS 3
    hvr_set_t *my_sets[NSETS];
    hvr_set_msg_t msgs[NSETS];
    for (int i = 0; i < NSETS; i++) {
        my_sets[i] = hvr_create_empty_set(npes);
        hvr_set_insert(pe, my_sets[i]);

        hvr_set_msg_init(my_sets[i], &msgs[i]);
    }

    hvr_set_t *recvd_set = hvr_create_empty_set(npes);

    hvr_mailbox_t mb;
    hvr_mailbox_init(&mb, 16 * 1024 * 1024);

    for (int r = 0; r < 100; r++) {
        hvr_set_msg_t *curr_msg = &msgs[r % NSETS];
        hvr_set_t *curr_set = my_sets[r % NSETS];

        hvr_set_msg_send((pe + 1) % npes, pe, 0, 42, &mb, curr_msg);

        void *buf = malloc(curr_msg->msg_buf_len);
        assert(buf);

        size_t msg_len;
        int success;
        do {
            success = hvr_mailbox_recv(buf, curr_msg->msg_buf_len, &msg_len,
                    &mb);
        } while (!success);

        int expected_src_pe = pe - 1;
        if (expected_src_pe < 0) expected_src_pe = npes - 1;

        assert(msg_len == curr_msg->msg_buf_len);
        hvr_internal_set_msg_t *recvd = (hvr_internal_set_msg_t *)buf;
        assert(recvd->n_contained == 1);
        assert(recvd->max_n_contained == npes);
        assert(recvd->bit_vector_len == curr_set->bit_vector_len);
        assert(recvd->pe == expected_src_pe);
        assert(recvd->iter == 0);
        assert(recvd->metadata == 42);

        hvr_set_msg_copy(recvd_set, recvd);

        for (int p = 0; p < npes; p++) {
            if (p == expected_src_pe) {
                assert(hvr_set_contains(p, recvd_set));
            } else {
                assert(!hvr_set_contains(p, recvd_set));
            }
        }

        free(buf);
    }

    void *buf = malloc(msgs[0].msg_buf_len);
    assert(buf);
    // Assert that we only receive one message
    for (unsigned i = 0; i < 100000; i++) {
        size_t msg_len;
        int success = hvr_mailbox_recv(buf, msgs[0].msg_buf_len, &msg_len, &mb);
        assert(!success);
    }
    free(buf);

    shmem_finalize();
    if (pe == 0) printf("Done!\n");
    return 0;
}
