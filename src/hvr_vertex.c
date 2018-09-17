#include <stdio.h>
#include <string.h>

#include "hvr_vertex.h"
#include "hoover.h"
#include "hvr_vertex_pool.h"

hvr_vertex_t *hvr_vertex_create(hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    return hvr_alloc_vertices(1, ctx);
}

void hvr_vertex_delete(hvr_vertex_t *vert, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    hvr_free_vertices(vert, 1, ctx);
}

void hvr_vertex_init(hvr_vertex_t *vert, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    memset(vert, 0x00, sizeof(*vert));

    vert->id = construct_vertex_id(ctx->pe, vert - ctx->pool->pool);
    vert->creation_iter = ctx->iter;
    vert->last_modify_iter = ctx->iter;
}

void hvr_vertex_set(const unsigned feature, const double val,
        hvr_vertex_t *vert, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    assert(VERTEX_ID_PE(vert->id) == ctx->pe);
    vert->last_modify_iter = ctx->iter;

    const unsigned size = vert->size;
    for (unsigned i = 0; i < size; i++) {
        if (vert->features[i] == feature) {
            // Replace
            vert->values[i] = val;
            return;
        }
    }

    assert(size < HVR_MAX_VECTOR_SIZE); // Can hold one more feature
    vert->features[size] = feature;
    vert->values[size] = val;
    vert->size += 1;
}

double hvr_vertex_get(const unsigned feature, hvr_vertex_t *vert,
        hvr_ctx_t in_ctx) {
    for (unsigned i = 0; i < vert->size; i++) {
        if (vert->features[i] == feature) {
            return vert->values[i];
        }
    }
    assert(0);
}

static int uint_compare(const void *_a, const void *_b) {
    unsigned a = *((unsigned *)_a);
    unsigned b = *((unsigned *)_b);
    if (a < b) {
        return -1;
    } else if (a > b) {
        return 1;
    } else {
        return 0;
    }
}

void hvr_vertex_unique_features(hvr_vertex_t *vert,
        unsigned *out_features, unsigned *n_out_features) {
    *n_out_features = vert->size;
    memcpy(out_features, vert->features,
            vert->size * sizeof(vert->features[0]));

    qsort(out_features, *n_out_features, sizeof(*out_features), uint_compare);
}

void hvr_vertex_dump(hvr_vertex_t *vert, char *buf, const size_t buf_size,
        hvr_ctx_t ctx) {
    char *iter = buf;
    int first = 1;

    unsigned n_features;
    unsigned features[HVR_MAX_VECTOR_SIZE];
    hvr_vertex_unique_features(vert, features, &n_features);

    for (unsigned i = 0; i < n_features; i++) {
        const unsigned feat = features[i];
        double val;

        const int err = hvr_vertex_get(feat, vert, ctx);
        assert(err == 1);

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

int hvr_vertex_get_owning_pe(hvr_vertex_t *vert) {
    assert(vert->id != HVR_INVALID_VERTEX_ID);
    return VERTEX_ID_PE(vert->id);
}

void hvr_vertex_add(hvr_vertex_t *dst, hvr_vertex_t *src, hvr_ctx_t ctx) {
    unsigned dst_n_features, src_n_features;
    unsigned dst_features[HVR_MAX_VECTOR_SIZE],
            src_features[HVR_MAX_VECTOR_SIZE];
    hvr_vertex_unique_features(dst, dst_features, &dst_n_features);
    hvr_vertex_unique_features(src, src_features, &src_n_features);
    assert(dst_n_features == src_n_features);

    for (unsigned i = 0; i < dst_n_features; i++) {
        const double sum = hvr_vertex_get(dst_features[i], dst, ctx) +
            hvr_vertex_get(src_features[i], src, ctx);
        hvr_vertex_set(dst_features[i], sum, dst, ctx);
    }
}
