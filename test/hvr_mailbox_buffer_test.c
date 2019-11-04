#include <stdio.h>
#include <stdlib.h>
#include <shmem.h>
#include <assert.h>
#include <unistd.h>

#include "hvr_mailbox.h"
#include "hvr_mailbox_buffer.h"
#include "hoover.h"

static volatile int finished = 0;
static int ndone = 0;

static void process_msgs(hvr_update_msg_t *recv_buf, size_t n_to_buffer,
        hvr_mailbox_t *big_mailbox, int max_msgs) {
    size_t msg_len;
    int success = hvr_mailbox_recv(recv_buf, n_to_buffer * sizeof(*recv_buf),
            &msg_len, big_mailbox, 0);
    int nmsgs = 1;
    while (success) {
        assert(msg_len % sizeof(hvr_update_msg_t) == 0);
        int nmsgs = msg_len / sizeof(hvr_update_msg_t);
        for (int i = 0; i < nmsgs; i++) {
            assert(recv_buf[i].is_vert_update == 42 ||
                    recv_buf[i].is_vert_update == 43);
            if (recv_buf[i].is_vert_update == 43) {
                ndone++;
            }
        }
        if (nmsgs == max_msgs) break;

        success = hvr_mailbox_recv(recv_buf, n_to_buffer * sizeof(*recv_buf),
                &msg_len, big_mailbox, 0);
        nmsgs++;
    }
}

typedef struct _process_msgs_ctx {
    hvr_update_msg_t *recv_buf;
    size_t n_to_buffer;
    hvr_mailbox_t *big_mailbox;
    int max_msgs;
} process_msgs_ctx;

static int process_msgs_wrapper(void *data, int pe_sending_to) {
    process_msgs_ctx *ctx = (process_msgs_ctx *)data;
    process_msgs(ctx->recv_buf, ctx->n_to_buffer, ctx->big_mailbox,
            ctx->max_msgs);
    return 0;
}

static void *aborting_thread(void *user_data) {
    int nseconds = 300;

    const unsigned long long start = hvr_current_time_us();
    while (hvr_current_time_us() - start < nseconds * 1000000) {
        sleep(10);
    }

    if (!finished) {
        fprintf(stderr, "INFO: forcibly aborting PE %d after %d "
                "seconds.\n", shmem_my_pe(), nseconds);
        abort(); // Get a core dump
    }

    return NULL;
}

int main(int argc, char **argv) {
    shmem_init();

    int pe = shmem_my_pe();
    int npes = shmem_n_pes();

    hvr_mailbox_t big_mailbox;
    hvr_mailbox_init(&big_mailbox,   384 * 1024 * 1024);

    const unsigned n_to_buffer = 256;
    hvr_mailbox_buffer_t mailbox_buf;
    hvr_mailbox_buffer_init(&mailbox_buf, &big_mailbox, npes,
            sizeof(hvr_update_msg_t), n_to_buffer);

    hvr_update_msg_t *recv_buf = (hvr_update_msg_t *)malloc(
            n_to_buffer * sizeof(*recv_buf));
    assert(recv_buf);

    pthread_t aborting_pthread;
    const int pthread_err = pthread_create(&aborting_pthread, NULL,
            aborting_thread, NULL);
    assert(pthread_err == 0);

    process_msgs_ctx ctx;
    ctx.recv_buf = recv_buf;
    ctx.n_to_buffer = n_to_buffer;
    ctx.big_mailbox = &big_mailbox;
    ctx.max_msgs = -1;

    for (int iter = 0; iter < 100; iter++) {
        unsigned long long start_time = hvr_current_time_us();
        int target_pe = pe;
        while (hvr_current_time_us() - start_time < 1ULL * 1000ULL * 1000ULL) {
            hvr_update_msg_t msg;
            msg.is_vert_update = 42;
            int success = hvr_mailbox_buffer_send(&msg, sizeof(msg), target_pe,
                    100, &mailbox_buf, 0);
            while (!success) {
                process_msgs(recv_buf, n_to_buffer, &big_mailbox, 1);

                success = hvr_mailbox_buffer_send(&msg, sizeof(msg), target_pe,
                        100, &mailbox_buf, 0);
            }
            target_pe = (target_pe + 1) % npes;
        }

        hvr_mailbox_buffer_flush(&mailbox_buf, process_msgs_wrapper, &ctx);
    }

    // Send done notification
    for (int p = 0; p < npes; p++) {
        hvr_update_msg_t msg;
        msg.is_vert_update = 43;
        int success = hvr_mailbox_buffer_send(&msg, sizeof(msg), p,
                100, &mailbox_buf, 0);
        while (!success) {
            process_msgs(recv_buf, n_to_buffer, &big_mailbox, -1);

            success = hvr_mailbox_buffer_send(&msg, sizeof(msg), p,
                    100, &mailbox_buf, 0);
        }
    }

    // Flush the done messages
    hvr_mailbox_buffer_flush(&mailbox_buf, process_msgs_wrapper, &ctx);

    while (ndone != npes) {
        process_msgs(recv_buf, n_to_buffer, &big_mailbox, -1);
    }

    finished = 1;

    shmem_finalize();

    if (pe == 0) {
        printf("Success!\n");
    }

    return 0;
}
