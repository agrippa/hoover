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
    mailbox->indices = (uint64_t *)shmem_malloc(sizeof(*(mailbox->indices)));
    assert(mailbox->indices);
    *(mailbox->indices) = 0;

    mailbox->buf = (char *)shmem_malloc(capacity_in_bytes);
    assert(mailbox->buf);
    memset(mailbox->buf, 0x00, capacity_in_bytes);
    mailbox->capacity_in_bytes = capacity_in_bytes;

    shmem_barrier_all();
}

static uint64_t pack_indices(uint32_t read_index, uint32_t write_index) {
    uint64_t packed = read_index;
    packed = (packed << 32);
    packed = (packed & 0xffffffff00000000);
    packed = (packed | (uint64_t)write_index);
    return packed;
}

static void unpack_indices(uint64_t index, uint32_t *read_index,
        uint32_t *write_index) {
    *write_index = (index & 0x00000000ffffffff);
    uint64_t tmp = (index >> 32);
    *read_index = tmp;
}

static uint32_t used_bytes(uint32_t read_index, uint32_t write_index,
        hvr_mailbox_t *mailbox) {
    if (write_index >= read_index) {
        return write_index - read_index;
    } else {
        return write_index + (mailbox->capacity_in_bytes - read_index);
    }
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

int hvr_mailbox_send(const void *msg, size_t msg_len, int target_pe,
        int max_tries, hvr_mailbox_t *mailbox) {
    uint64_t full_msg_len = sizeof(sentinel) + sizeof(msg_len) + msg_len;
    assert(full_msg_len <= mailbox->capacity_in_bytes);

    uint64_t indices = shmem_uint64_atomic_fetch(mailbox->indices, target_pe);
    uint32_t start_send_index = 0;

    unsigned try = 0;
    while (max_tries < 0 || try < max_tries) {
        uint32_t read_index, write_index;
        unpack_indices(indices, &read_index, &write_index);

        // fprintf(stderr, "sender: unpacked read=%u write=%u\n", read_index, write_index);

        uint32_t consumed = used_bytes(read_index, write_index, mailbox);
        uint32_t free_bytes = mailbox->capacity_in_bytes - consumed;
        // fprintf(stderr, "sender: free_bytes=%u\n", free_bytes);
        if (free_bytes >= full_msg_len) {
            // Enough room to try
            uint32_t new_write_index = (write_index + full_msg_len) %
                mailbox->capacity_in_bytes;
            // fprintf(stderr, "sender: packing read=%u write=%u\n", read_index,
            //         new_write_index);
            uint64_t new = pack_indices(read_index, new_write_index);
            // fprintf(stderr, "sender: new=%lu\n", new);
            uint64_t old = shmem_uint64_atomic_compare_swap(mailbox->indices,
                    indices, new, target_pe);
            if (old == indices) {
                // Successful
                start_send_index = write_index;
                break;
            } else {
                indices = old;
            }
        } else {
            indices = shmem_uint64_atomic_fetch(mailbox->indices, target_pe);
        }
        try++;
    }

    if (try == max_tries) {
        // Failed
        return 0;
    }

    /*
     * Send the actual message, accounting for if the space allocated goes
     * around the circular buffer.
     */
    uint32_t start_send_offset = start_send_index;
    uint32_t msg_len_offset = ((start_send_index + sizeof(sentinel)) %
            mailbox->capacity_in_bytes);
    uint32_t msg_offset = ((start_send_index + sizeof(sentinel) +
                sizeof(msg_len)) % mailbox->capacity_in_bytes);

    put_in_mailbox_with_rotation(&msg_len, sizeof(msg_len), msg_len_offset,
            mailbox, target_pe);
    put_in_mailbox_with_rotation(msg, msg_len, msg_offset, mailbox, target_pe);
    shmem_fence();
    put_in_mailbox_with_rotation(&sentinel, sizeof(sentinel),
            start_send_offset, mailbox, target_pe);
    return 1;
}

int hvr_mailbox_recv(void **msg, size_t *msg_capacity, size_t *msg_len,
        hvr_mailbox_t *mailbox) {
    // Wait for at least one message to arrive
    uint64_t indices = shmem_uint64_atomic_fetch(mailbox->indices,
            shmem_my_pe());
    uint32_t read_index, write_index;
    unpack_indices(indices, &read_index, &write_index);
    uint32_t used = used_bytes(read_index, write_index, mailbox);

    if (used == 0) return 0;

    // Wait for the sentinel value to appear
    uint64_t start_msg_offset = read_index;
    uint64_t msg_len_offset = (read_index + sizeof(sentinel)) %
        mailbox->capacity_in_bytes;
    uint64_t msg_offset = ((read_index + sizeof(sentinel) + sizeof(*msg_len)) %
            mailbox->capacity_in_bytes);

    unsigned expect_sentinel;
    do {
        get_from_mailbox_with_rotation(start_msg_offset, &expect_sentinel,
                sizeof(expect_sentinel), mailbox);
    } while (expect_sentinel != sentinel);

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

    uint32_t new_read_index = (read_index + sizeof(sentinel) +
            sizeof(recv_msg_len) + recv_msg_len) % mailbox->capacity_in_bytes;
    uint64_t new_indices = pack_indices(new_read_index, write_index);
    while (1) {
        uint64_t old = shmem_uint64_atomic_compare_swap(mailbox->indices,
                indices, new_indices, shmem_my_pe());
        if (old == indices) break;

        uint32_t this_read_index, this_write_index;
        unpack_indices(old, &this_read_index, &this_write_index);
        assert(read_index == this_read_index);

        indices = old;
        new_indices = pack_indices(new_read_index, this_write_index);
    }

    shmem_quiet();

    return 1;
}
