#ifndef _HVR_MAILBOX
#define _HVR_MAILBOX

typedef struct _hvr_mailbox_t {
    uint64_t *write_index;
    uint64_t *read_index;
    size_t capacity_in_bytes;
    char *buf;
} hvr_mailbox_t;

/*
 * A symmetric call that allocates a remotely accessible mailbox data structure
 * across all PEs.
 */
void hvr_mailbox_init(hvr_mailbox_t *mailbox, size_t capacity_in_bytes);

/*
 * Place msg with length msg_len in bytes into the designated mailbox on the
 * designated PE.
 */
void hvr_mailbox_send(const void *msg, size_t msg_len, int target_pe,
        hvr_mailbox_t *mailbox);

/*
 * Check my local mailbox for a new message. If one is found, a pointer to it is
 * stored in msg, msg_len is updated to reflect the length of the message, and 1
 * is returned. Otherwise, 0 is returned to indicate no message was found.
 */
int hvr_mailbox_recv(void **msg, size_t *msg_capacity, size_t *msg_len,
        hvr_mailbox_t *mailbox);

#endif // _HVR_MAILBOX
