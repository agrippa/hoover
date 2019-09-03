#include <assert.h>
#include <string.h>

#include "hvr_mailbox_buffer.h"

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

void hvr_mailbox_buffer_send(const void *msg, size_t msg_len, int target_pe,
        int max_tries, hvr_mailbox_buffer_t *buf, int multithreaded) {
    assert(msg_len == buf->msg_size);

    char *pe_buf = buf->buffers + (target_pe * buf->buffer_size_per_pe * buf->msg_size);
    const unsigned nbuffered = buf->nbuffered_per_pe[target_pe];
    char *dst = pe_buf + (nbuffered * buf->msg_size);
    memcpy(dst, msg, msg_len);

    if (nbuffered + 1 == buf->buffer_size_per_pe) {
        // flush
        int success = hvr_mailbox_send(pe_buf,
                buf->buffer_size_per_pe * buf->msg_size,
                target_pe, max_tries, buf->mbox, multithreaded);
        assert(success);

        buf->nbuffered_per_pe[target_pe] = 0;
    } else {
        buf->nbuffered_per_pe[target_pe] = nbuffered + 1;
    }
}

void hvr_mailbox_buffer_flush(hvr_mailbox_buffer_t *buf) {
    for (int p = 0; p < buf->npes; p++) {
        const unsigned nbuffered = buf->nbuffered_per_pe[p];
        char *pe_buf = buf->buffers + (p * buf->buffer_size_per_pe * buf->msg_size);
        if (nbuffered > 0) {
            int success = hvr_mailbox_send(pe_buf, nbuffered * buf->msg_size,
                    p, -1, buf->mbox, 0);
            assert(success);
        }
        buf->nbuffered_per_pe[p] = 0;
    }
}
