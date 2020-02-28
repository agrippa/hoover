#include <shmem.h>
#include <shmemx.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include "hvr_mailbox.h"
#include "hvr_common.h"

// #define USE_CRC
#ifdef USE_CRC
#define CRC32
#include "crc.c"
#endif

const static unsigned sentinel = 0xdeed;
const static unsigned clear_sentinel = 0x0;

void hvr_mailbox_init(hvr_mailbox_t *mailbox, size_t capacity_in_bytes) {
    // So that sentinel values are always cohesive
    assert(capacity_in_bytes % sizeof(sentinel) == 0);

    memset(mailbox, 0x00, sizeof(*mailbox));
    mailbox->indices = (uint64_t *)shmem_malloc_wrapper(
            sizeof(*(mailbox->indices)));
    assert(mailbox->indices);
    shmem_uint64_p(mailbox->indices, 0, shmem_my_pe());
    mailbox->indices_curr_val = 0;

    mailbox->capacity_in_bytes = capacity_in_bytes;

    mailbox->buf = (char *)shmem_malloc_wrapper(capacity_in_bytes);
    assert(mailbox->buf);
    memset(mailbox->buf, 0x00, capacity_in_bytes);

    mailbox->pe = shmem_my_pe();

#ifdef USE_CRC
    crcInit();
#endif

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

static void clear_mailbox_with_rotation(size_t data_len,
        uint64_t starting_offset, hvr_mailbox_t *mailbox) {
    if (starting_offset + data_len <= mailbox->capacity_in_bytes) {
        memset(mailbox->buf + starting_offset, 0x00, data_len);
    } else {
        uint64_t rotate_index = mailbox->capacity_in_bytes - starting_offset;
        memset(mailbox->buf + starting_offset, 0x00, rotate_index);
        memset(mailbox->buf, 0x00, data_len - rotate_index);
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
    const int target_pe = mailbox->pe;
    if (starting_offset + data_len <= mailbox->capacity_in_bytes) {
        shmem_getmem(data, mailbox->buf + starting_offset, data_len, target_pe);
    } else {
        uint64_t rotate_index = mailbox->capacity_in_bytes - starting_offset;
        shmem_getmem(data, mailbox->buf + starting_offset, rotate_index,
                target_pe);
        shmem_getmem((char *)data + rotate_index, mailbox->buf,
                data_len - rotate_index, target_pe);
    }
}

int hvr_mailbox_send(const void *msg, size_t msg_len, int target_pe,
        int max_tries, hvr_mailbox_t *mailbox) {
    // So that sentinel values are always cohesive
    assert(msg_len % sizeof(sentinel) == 0);

    uint64_t full_msg_len = sizeof(sentinel) + sizeof(msg_len) + msg_len;
#ifdef USE_CRC
    full_msg_len += 2 * sizeof(crc);
    crc msg_len_crc = crcFast((const unsigned char *)&msg_len,
            sizeof(msg_len));
    crc msg_crc = crcFast(msg, msg_len);
#endif
    assert(full_msg_len < mailbox->capacity_in_bytes);

    uint64_t indices = shmem_uint64_atomic_fetch(mailbox->indices, target_pe);
    uint32_t start_send_index = 0;

    unsigned tries = 0;
    while (max_tries < 0 || tries < max_tries) {
        if (tries > 1000000) {
            fprintf(stderr, "WARNING PE %d hitting many failed tries sending "
                    "to %d\n", shmem_my_pe(), target_pe);
            abort();
        }
        uint32_t read_index, write_index;
        unpack_indices(indices, &read_index, &write_index);

        uint32_t consumed = used_bytes(read_index, write_index, mailbox);
        uint32_t free_bytes = mailbox->capacity_in_bytes - consumed;
        if (free_bytes > full_msg_len) {
            // Enough room to try
            uint32_t new_write_index = (write_index + full_msg_len) %
                mailbox->capacity_in_bytes;
            uint64_t new_val = pack_indices(read_index, new_write_index);
            uint64_t old = shmem_uint64_atomic_compare_swap(mailbox->indices,
                    indices, new_val, target_pe);
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
        tries++;
    }

    if (tries == max_tries) {
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
#ifdef USE_CRC
    uint32_t msg_len_crc_offset = ((start_send_index + sizeof(sentinel)) %
        mailbox->capacity_in_bytes);
    uint32_t msg_crc_offset = ((start_send_index + sizeof(sentinel) +
                sizeof(crc)) % mailbox->capacity_in_bytes);

    msg_len_offset = (msg_len_offset + 2 * sizeof(crc)) %
        mailbox->capacity_in_bytes;
    msg_offset = (msg_offset + 2 * sizeof(crc)) % mailbox->capacity_in_bytes;

    put_in_mailbox_with_rotation(&msg_len_crc, sizeof(msg_len_crc),
            msg_len_crc_offset, mailbox, target_pe);
    put_in_mailbox_with_rotation(&msg_crc, sizeof(msg_crc),
            msg_crc_offset, mailbox, target_pe);
#endif

    put_in_mailbox_with_rotation(&msg_len, sizeof(msg_len), msg_len_offset,
            mailbox, target_pe);
    put_in_mailbox_with_rotation(msg, msg_len, msg_offset, mailbox, target_pe);

    shmem_quiet();

    put_in_mailbox_with_rotation(&sentinel, sizeof(sentinel),
            start_send_offset, mailbox, target_pe);
    return 1;
}

int hvr_mailbox_recv(void *msg, size_t msg_capacity, size_t *msg_len,
        hvr_mailbox_t *mailbox) {
    uint32_t read_index, write_index;
    uint64_t curr_indices;

    unpack_indices(mailbox->indices_curr_val, &read_index, &write_index);
    if (used_bytes(read_index, write_index, mailbox) > 0) {
        /*
         * If the previously saved current value of indices indicates there are
         * pending messages, we can assume that is still the case without
         * actually having to check the mailbox.
         */
        curr_indices = mailbox->indices_curr_val;
    } else {
        /*
         * Otherwise, the last time we checked the mailbox it was empty. We have
         * to check if that's still the case.
         */
        uint64_t new_indices = shmem_uint64_atomic_fetch(mailbox->indices,
                    mailbox->pe);
        if (new_indices != mailbox->indices_curr_val) {
            curr_indices = new_indices;
        } else {
            return 0;
        }
    }

    unpack_indices(curr_indices, &read_index, &write_index);
    uint32_t used = used_bytes(read_index, write_index, mailbox);
    assert(used > 0);

    // Wait for the sentinel value to appear
    uint64_t start_msg_offset = read_index;
    uint64_t msg_len_offset = (read_index + sizeof(sentinel)) %
        mailbox->capacity_in_bytes;
    uint64_t msg_offset = ((read_index + sizeof(sentinel) +
                sizeof(*msg_len)) % mailbox->capacity_in_bytes);
#ifdef USE_CRC
    uint64_t msg_len_crc_offset = (read_index + sizeof(sentinel)) %
        mailbox->capacity_in_bytes;
    uint64_t msg_crc_offset = (read_index + sizeof(sentinel) + sizeof(crc)) %
        mailbox->capacity_in_bytes;

    msg_len_offset = (msg_len_offset + 2*sizeof(crc)) %
        mailbox->capacity_in_bytes;
    msg_offset = (msg_offset + 2*sizeof(crc)) % mailbox->capacity_in_bytes;
#endif

    // Assert that the sentinel value is cohesive
    unsigned expect_sentinel;
    assert(start_msg_offset + sizeof(expect_sentinel) <=
            mailbox->capacity_in_bytes);
    shmem_uint_wait_until((unsigned *)(mailbox->buf + start_msg_offset),
            SHMEM_CMP_EQ, sentinel);

#ifdef USE_CRC
    crc msg_len_crc;
    get_from_mailbox_with_rotation(msg_len_crc_offset, &msg_len_crc,
            sizeof(msg_len_crc), mailbox);

    crc msg_crc;
    get_from_mailbox_with_rotation(msg_crc_offset, &msg_crc,
            sizeof(msg_crc), mailbox);
#endif

    size_t recv_msg_len;
    get_from_mailbox_with_rotation(msg_len_offset, &recv_msg_len,
            sizeof(recv_msg_len), mailbox);
    *msg_len = recv_msg_len;

    assert(msg_capacity >= recv_msg_len);

    get_from_mailbox_with_rotation(msg_offset, msg, recv_msg_len, mailbox);

#ifdef USE_CRC
    crc calc_msg_len_crc = crcFast((const unsigned char *)&recv_msg_len,
            sizeof(recv_msg_len));
    crc calc_msg_crc = crcFast(msg, recv_msg_len);
    assert(calc_msg_len_crc == msg_len_crc);
    assert(calc_msg_crc == msg_crc);
#endif

    /*
     * Once we've finished extracting the message, clear the sentinel value and
     * increment the read index.
     */
#ifdef USE_CRC
    clear_mailbox_with_rotation(
            2*sizeof(crc) + sizeof(recv_msg_len) + recv_msg_len,
            msg_len_crc_offset, mailbox);
#else
    clear_mailbox_with_rotation(
            sizeof(recv_msg_len) + recv_msg_len,
            msg_len_offset, mailbox);
#endif
    shmem_quiet();
    put_in_mailbox_with_rotation(&clear_sentinel, sizeof(clear_sentinel),
            start_msg_offset, mailbox, mailbox->pe);
    // shmem_fence();
    shmem_quiet();

    uint32_t new_read_index = (read_index + sizeof(sentinel) +
            sizeof(recv_msg_len) + recv_msg_len) % mailbox->capacity_in_bytes;
#ifdef USE_CRC
    new_read_index = (new_read_index + 2*sizeof(crc)) %
        mailbox->capacity_in_bytes;
#endif
    uint64_t new_indices = pack_indices(new_read_index, write_index);
    while (1) {
        uint64_t old = shmem_uint64_atomic_compare_swap(mailbox->indices,
                curr_indices, new_indices, mailbox->pe);
        if (old == curr_indices) break;

        uint32_t this_read_index, this_write_index;
        unpack_indices(old, &this_read_index, &this_write_index);
        assert(read_index == this_read_index);

        curr_indices = old;
        new_indices = pack_indices(new_read_index, this_write_index);
    }
    mailbox->indices_curr_val = new_indices;

    shmem_quiet();

    return 1;
}

void hvr_mailbox_destroy(hvr_mailbox_t *mailbox) {
    shmem_free(mailbox->indices);
    shmem_free(mailbox->buf);
}

size_t hvr_mailbox_mem_used(hvr_mailbox_t *mailbox) {
    return sizeof(mailbox->indices[0]) +
        mailbox->capacity_in_bytes;
}
