#include <shmem.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include "hvr_mailbox.h"

const static unsigned sentinel = 0xdeed;
const static unsigned clear_sentinel = 0x0;

void hvr_mailbox_init(hvr_mailbox_t *mailbox, size_t capacity_in_bytes) {
    memset(mailbox, 0x00, sizeof(*mailbox));
    mailbox->write_index = (uint64_t *)shmem_malloc(sizeof(uint64_t));
    assert(mailbox->write_index);
    *(mailbox->write_index) = 0;

    mailbox->read_index = (uint64_t *)shmem_malloc(sizeof(uint64_t));
    assert(mailbox->read_index);
    *(mailbox->read_index) = 0;

    mailbox->buf = (char *)shmem_malloc(capacity_in_bytes);
    assert(mailbox->buf);
    memset(mailbox->buf, 0x00, capacity_in_bytes);
    mailbox->capacity_in_bytes = capacity_in_bytes;

    shmem_barrier_all();
}

static void put_in_mailbox_with_rotation(const void *data, size_t data_len,
        uint64_t starting_offset, hvr_mailbox_t *mailbox, int target_pe) {
    if (starting_offset + data_len <= mailbox->capacity_in_bytes) {
        shmem_putmem(mailbox->buf + starting_offset, data, data_len, target_pe);
    } else {
        uint64_t rotate_index = mailbox->capacity_in_bytes - starting_offset;
        shmem_putmem(mailbox->buf + starting_offset, data, rotate_index,
                target_pe);
        shmem_putmem(mailbox->buf, (char *)data + rotate_index,
                data_len - rotate_index, target_pe);
    }
}

static void get_from_mailbox_with_rotation(uint64_t starting_offset, void *data,
        uint64_t data_len, hvr_mailbox_t* mailbox) {
    if (starting_offset + data_len <= mailbox->capacity_in_bytes) {
        shmem_getmem(data, mailbox->buf + starting_offset, data_len,
                shmem_my_pe());
    } else {
        uint64_t rotate_index = mailbox->capacity_in_bytes - starting_offset;
        shmem_getmem(data, mailbox->buf + starting_offset, rotate_index,
                shmem_my_pe());
        shmem_getmem((char *)data + rotate_index, mailbox->buf,
                data_len - rotate_index, shmem_my_pe());
    }
}

void hvr_mailbox_send(const void *msg, size_t msg_len, int target_pe,
        hvr_mailbox_t *mailbox) {
    uint64_t full_msg_len = sizeof(sentinel) + sizeof(msg_len) + msg_len;
    assert(full_msg_len <= mailbox->capacity_in_bytes);

    uint64_t start_send_index = shmem_uint64_atomic_fetch_add(
            mailbox->write_index, full_msg_len, target_pe);

    // Wait for enough space in the mailbox to open up
    uint64_t other_read_index, consumed, left;
    do {
        other_read_index = shmem_uint64_atomic_fetch(mailbox->read_index,
                target_pe);
        consumed = start_send_index - other_read_index;
        left = mailbox->capacity_in_bytes - consumed;
    } while (left < full_msg_len);

    /*
     * Send the actual message, accounting for if the space allocated goes
     * around the circular buffer.
     */
    uint64_t start_send_offset = (start_send_index %
            mailbox->capacity_in_bytes);
    uint64_t msg_len_offset = ((start_send_index + sizeof(sentinel)) %
            mailbox->capacity_in_bytes);
    uint64_t msg_offset = ((start_send_index + sizeof(sentinel) +
                sizeof(msg_len)) % mailbox->capacity_in_bytes);

    put_in_mailbox_with_rotation(&msg_len, sizeof(msg_len), msg_len_offset,
            mailbox, target_pe);
    put_in_mailbox_with_rotation(msg, msg_len, msg_offset, mailbox, target_pe);
    shmem_fence();
    put_in_mailbox_with_rotation(&sentinel, sizeof(sentinel),
            start_send_offset, mailbox, target_pe);
}

int hvr_mailbox_recv(void **msg, size_t *msg_capacity, size_t *msg_len,
        hvr_mailbox_t *mailbox) {
    // Wait for at least one message to arrive
    uint64_t read_index = shmem_uint64_atomic_fetch(mailbox->read_index,
            shmem_my_pe());
    uint64_t write_index = shmem_uint64_atomic_fetch(mailbox->write_index,
            shmem_my_pe());
    uint64_t used = write_index - read_index;

    if (used == 0) return 0;

    // Wait for the sentinel value to appear
    uint64_t start_msg_offset = read_index % mailbox->capacity_in_bytes;
    uint64_t msg_len_offset = (read_index + sizeof(sentinel)) %
        mailbox->capacity_in_bytes;
    uint64_t msg_offset = ((read_index + sizeof(sentinel) + sizeof(*msg_len)) %
            mailbox->capacity_in_bytes);

    fprintf(stderr, "Trying to get sentinel value at byte %lu / %lu\n",
            start_msg_offset, mailbox->capacity_in_bytes);
    unsigned expect_sentinel;
    do {
        get_from_mailbox_with_rotation(start_msg_offset, &expect_sentinel,
                sizeof(expect_sentinel), mailbox);
    } while (expect_sentinel != sentinel);
    fprintf(stderr, "Got sentinel value\n");

    size_t recv_msg_len;
    get_from_mailbox_with_rotation(msg_len_offset, &recv_msg_len,
            sizeof(recv_msg_len), mailbox);
    *msg_len = recv_msg_len;

    if (*msg_capacity < recv_msg_len) {
        *msg = (void *)realloc(*msg, recv_msg_len);
        assert(*msg);
        *msg_capacity = recv_msg_len;
    }

    get_from_mailbox_with_rotation(msg_offset, *msg, recv_msg_len, mailbox);

    /*
     * Once we've finished extracting the message, clear the sentinel value and
     * increment the read index.
     */
    put_in_mailbox_with_rotation(&clear_sentinel, sizeof(clear_sentinel),
            start_msg_offset, mailbox, shmem_my_pe());
    shmem_fence();
    shmem_uint64_atomic_add(mailbox->read_index,
            sizeof(sentinel) + sizeof(recv_msg_len) + recv_msg_len,
            shmem_my_pe());
    shmem_quiet();

    return 1;
}
