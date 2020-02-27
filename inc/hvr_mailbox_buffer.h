#ifndef _HVR_MAILBOX_BUFFER_H
#define _HVR_MAILBOX_BUFFER_H

#include "hvr_mailbox.h"

typedef struct _hvr_mailbox_buffer_t {
    hvr_mailbox_t *mbox;
    int npes;
    size_t msg_size;
    size_t buffer_size_per_pe;

    unsigned *nbuffered_per_pe;

    char *buffers;
} hvr_mailbox_buffer_t;

void hvr_mailbox_buffer_init(hvr_mailbox_buffer_t *buf, hvr_mailbox_t *mbox,
        int npes, size_t msg_size, size_t buffer_size_per_pe);

int hvr_mailbox_buffer_send(const void *msg, size_t msg_len, int target_pe,
        int max_tries, hvr_mailbox_buffer_t *buf);

void hvr_mailbox_buffer_flush(hvr_mailbox_buffer_t *buf, int (*cb)(void *, int),
        void *user_data);

#endif // _HVR_MAILBOX_BUFFER_H
