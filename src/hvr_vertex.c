#include <stdio.h>
#include <string.h>

#include "hvr_vertex.h"
#include "hoover.h"

hvr_vertex_t *hvr_vertex_create(hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    hvr_vertex_cache_node_t *reserved = hvr_vertex_cache_reserve(
            &ctx->vec_cache, ctx->pe, ctx->iter);
    hvr_vertex_t *allocated = &reserved->vert;

    allocated->next_in_partition = ctx->recently_created;
    ctx->recently_created = allocated;

    return allocated;
}

void hvr_vertex_delete(hvr_vertex_t *vert, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;

    // Notify others of the deletion
    unsigned long long unused;
    hvr_partition_t part = wrap_actor_to_partition(vert, ctx);
    send_updates_to_all_subscribed_pes(vert, part, 0, 1, NULL, &unused, ctx);

    remove_from_partition_list(vert, &ctx->local_partition_lists, ctx);

    hvr_vertex_cache_delete((hvr_vertex_cache_node_t *)vert, &ctx->vec_cache);
}

void hvr_vertex_init(hvr_vertex_t *vert, hvr_vertex_id_t id, hvr_time_t iter) {
    memset(vert, 0x00, sizeof(*vert));

    vert->id = id;
    vert->creation_iter = iter;
    // Should be sent and processed
    vert->needs_processing = 1;
    vert->needs_send = 1;
    vert->curr_part = HVR_INVALID_PARTITION;
    vert->prev_part = HVR_INVALID_PARTITION;
}

void hvr_vertex_dump(hvr_vertex_t *vert, char *buf, const size_t buf_size,
        hvr_ctx_t ctx) {
    char *iter = buf;
    int first = 1;

    for (unsigned feat = 0; feat < HVR_MAX_VECTOR_SIZE; feat++) {
        double val = hvr_vertex_get(feat, vert, ctx);

        const int capacity = buf_size - (iter - buf);
        int written;
        if (first) {
            written = snprintf(iter, capacity, "%u: %f", feat, val);
        } else {
            written = snprintf(iter, capacity, ", %u: %f", feat, val);
        }
        if (written <= 0 || written > capacity) {
            assert(0);
        }

        iter += written;
        first = 0;
    }
}

int hvr_vertex_get_owning_pe(const hvr_vertex_t *vert) {
    assert(vert->id != HVR_INVALID_VERTEX_ID);
    return VERTEX_ID_PE(vert->id);
}

void hvr_vertex_add(hvr_vertex_t *dst, hvr_vertex_t *src, hvr_ctx_t ctx) {
    for (unsigned i = 0; i < HVR_MAX_VECTOR_SIZE; i++) {
        hvr_vertex_set(i,
                hvr_vertex_get(i, src, ctx) + hvr_vertex_get(i, dst, ctx),
                dst, ctx);
    }
}

int hvr_vertex_equal(hvr_vertex_t *a, hvr_vertex_t *b, hvr_ctx_t in_ctx) {
    for (unsigned i = 0; i < HVR_MAX_VECTOR_SIZE; i++) {
        if (hvr_vertex_get(i, a, in_ctx) != hvr_vertex_get(i, b, in_ctx)) {
            return 0;
        }
    }
    return 1;
}

void hvr_vertex_trigger_update(hvr_vertex_t *vert, hvr_ctx_t in_ctx) {
    vert->needs_processing = 1;
}
