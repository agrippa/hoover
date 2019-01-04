#include <assert.h>
#include <shmem.h>
#include <string.h>

#include "hvr_set_msg.h"

void hvr_set_msg_init(hvr_set_t *set, hvr_set_msg_t *msg) {
    msg->set = set;
    msg->msg_buf_len = sizeof(hvr_internal_set_msg_t) +
            set->bit_vector_len * sizeof(bit_vec_element_type);
    msg->msg_buf = (hvr_internal_set_msg_t *)malloc(msg->msg_buf_len);
    assert(msg->msg_buf);
}

void hvr_set_msg_send_with_val(int dst_pe, hvr_time_t iter,
        hvr_vertex_t *optional_val, int updates_on_this_iter,
        hvr_mailbox_t *mailbox, hvr_set_msg_t *msg) {
    msg->msg_buf->n_contained = msg->set->n_contained;
    msg->msg_buf->max_n_contained = msg->set->max_n_contained;
    msg->msg_buf->bit_vector_len = msg->set->bit_vector_len;
    msg->msg_buf->pe = shmem_my_pe();
    msg->msg_buf->iter = iter;
    if (optional_val) {
        msg->msg_buf->updates_on_this_iter = updates_on_this_iter;
        memcpy(&(msg->msg_buf->val), optional_val, sizeof(*optional_val));
        msg->msg_buf->have_nested = 1;
    } else {
        msg->msg_buf->have_nested = 0;
    }
    memcpy(msg->msg_buf + 1, msg->set->bit_vector,
            msg->set->bit_vector_len * sizeof(bit_vec_element_type));

    hvr_mailbox_send(msg->msg_buf, msg->msg_buf_len, dst_pe, -1, mailbox, NULL);
}

void hvr_set_msg_send(int dst_pe, hvr_time_t iter, hvr_mailbox_t *mailbox,
        hvr_set_msg_t *msg) {
    hvr_set_msg_send_with_val(dst_pe, iter, NULL, 0, mailbox, msg);
}

void hvr_set_msg_copy(hvr_set_t *dst, hvr_internal_set_msg_t *msg) {
    dst->n_contained = msg->n_contained;
    assert(dst->max_n_contained == msg->max_n_contained);
    assert(dst->bit_vector_len == msg->bit_vector_len);
    memcpy(dst->bit_vector, msg + 1,
            dst->bit_vector_len * sizeof(bit_vec_element_type));
}

void hvr_set_msg_destroy(hvr_set_msg_t *msg) {
    free(msg->msg_buf);
}
