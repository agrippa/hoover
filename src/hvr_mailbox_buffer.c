#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <shmem.h>

#include "hvr_mailbox_buffer.h"
#include "hoover.h"

void hvr_mailbox_buffer_init(hvr_mailbox_buffer_t *buf, hvr_mailbox_t *mbox,
        int npes, size_t msg_size, size_t buffer_size_per_pe) {
    buf->mbox = mbox;
    buf->npes = npes;
    buf->msg_size = msg_size;
    buf->buffer_size_per_pe = buffer_size_per_pe;

    buf->nbuffered_per_pe = (unsigned *)malloc(npes * sizeof(unsigned));
    assert(buf->nbuffered_per_pe);
    memset(buf->nbuffered_per_pe, 0x00, npes * sizeof(unsigned));

    buf->buffers = (char *)malloc(npes * buffer_size_per_pe * msg_size);
    assert(buf->buffers);
}

int hvr_mailbox_buffer_send(const void *msg, size_t msg_len, int target_pe,
        int max_tries, hvr_mailbox_buffer_t *buf, int multithreaded) {
    assert(msg_len == buf->msg_size);

    unsigned nbuffered = buf->nbuffered_per_pe[target_pe];
    assert(nbuffered <= buf->buffer_size_per_pe);

    char *pe_buf = buf->buffers +
        (target_pe * buf->buffer_size_per_pe * buf->msg_size);

    if (nbuffered == buf->buffer_size_per_pe) {
        // flush
        int success = hvr_mailbox_send(pe_buf, nbuffered * buf->msg_size,
                target_pe, max_tries, buf->mbox, multithreaded);
        if (success) {
            buf->nbuffered_per_pe[target_pe] = 0;
        } else {
            return 0;
        }
    }

    nbuffered = buf->nbuffered_per_pe[target_pe];
    char *dst = pe_buf + (nbuffered * buf->msg_size);
    memcpy(dst, msg, msg_len);
    buf->nbuffered_per_pe[target_pe] = nbuffered + 1;
    return 1;
}

void hvr_mailbox_buffer_flush(hvr_mailbox_buffer_t *buf, int (*cb)(void *, int),
        void *user_data) {
    for (int p = 0; p < buf->npes; p++) {
        const unsigned nbuffered = buf->nbuffered_per_pe[p];
        assert(nbuffered <= buf->buffer_size_per_pe);

        char *pe_buf = buf->buffers + (p * buf->buffer_size_per_pe *
                buf->msg_size);
        if (nbuffered > 0) {
            unsigned count_loops = 0;
            int printed_warning = 0;

            int success = hvr_mailbox_send(pe_buf, nbuffered * buf->msg_size,
                    p, 100, buf->mbox, 0);
            int should_abort_send = 0;
            while (!success && !should_abort_send) {
                if (cb) {
                    should_abort_send = cb(user_data, p);
                }

                success = hvr_mailbox_send(pe_buf, nbuffered * buf->msg_size,
                        p, 100, buf->mbox, 0);

                count_loops++;
                if (count_loops > 100000 && !printed_warning) {
                    fprintf(stderr, "PE %d seems to have gotten wedged sending "
                            "to PE %d\n", shmem_my_pe(), p);
                    printed_warning = 1;
                }
            }
            buf->nbuffered_per_pe[p] = 0;
        }
    }
}
