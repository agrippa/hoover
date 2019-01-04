#ifndef _HVR_SET_MSG
#define _HVR_SET_MSG

#include <stdint.h>
#include <stdlib.h>

#include "hvr_set.h"
#include "hvr_mailbox.h"
#include "hvr_vertex.h"

typedef struct _hvr_internal_set_msg_t {
    uint64_t n_contained;
    uint64_t max_n_contained;
    uint64_t bit_vector_len;
    int pe;
    hvr_time_t iter;

    // Optional fields, only filled in if have_nested = 1
    int updates_on_this_iter;
    hvr_vertex_t val;
    int have_nested;
} hvr_internal_set_msg_t;

typedef struct _hvr_set_msg_t {
    hvr_set_t *set;
    hvr_internal_set_msg_t *msg_buf;
    size_t msg_buf_len;
} hvr_set_msg_t;

extern void hvr_set_msg_init(hvr_set_t *set, hvr_set_msg_t *msg);

extern void hvr_set_msg_send_with_val(int dst_pe, hvr_time_t iter,
        hvr_vertex_t *optional_val, int updates_on_this_iter,
        hvr_mailbox_t *mailbox, hvr_set_msg_t *msg);

extern void hvr_set_msg_send(int dst_pe, hvr_time_t iter,
        hvr_mailbox_t *mailbox, hvr_set_msg_t *msg);

extern void hvr_set_msg_copy(hvr_set_t *dst, hvr_internal_set_msg_t *msg);

extern void hvr_set_msg_destroy(hvr_set_msg_t *msg);

#endif
