#include <stdio.h>
#include <string.h>

#include "hvr_vertex.h"
#include "hoover.h"
#include "hvr_vertex_pool.h"

hvr_vertex_t *hvr_vertex_create(hvr_ctx_t in_ctx) {
    return hvr_alloc_vertices(1, (hvr_internal_ctx_t *)in_ctx);
}

hvr_vertex_t *hvr_vertex_create_n(size_t n, hvr_ctx_t ctx) {
    return hvr_alloc_vertices(n, (hvr_internal_ctx_t *)ctx);
}

void hvr_vertex_delete(hvr_vertex_t *vert, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    hvr_free_vertices(vert, 1, ctx);
}

void hvr_vertex_init(hvr_vertex_t *vert, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    memset(vert, 0x00, sizeof(*vert));

    hvr_vertex_pool_t *pool = ctx->pool;
    if (vert >= pool->pool && vert < pool->pool + pool->pool_size) {
        vert->id = construct_vertex_id(ctx->pe, vert - pool->pool);
    } else {
        vert->id = HVR_INVALID_VERTEX_ID;
    }
    vert->creation_iter = ctx->iter;
    vert->last_modify_iter = ctx->iter;
}

void hvr_vertex_set(const unsigned feature, const double val,
        hvr_vertex_t *vert, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    assert(VERTEX_ID_PE(vert->id) == ctx->pe ||
            vert->id == HVR_INVALID_VERTEX_ID);
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

int hvr_vertex_get_owning_pe(hvr_vertex_t *vert) {
    assert(vert->id != HVR_INVALID_VERTEX_ID);
    return VERTEX_ID_PE(vert->id);
}

static inline int hvr_vertex_have_feature(unsigned feature,
        hvr_vertex_t *vert) {
    for (unsigned i = 0; i < vert->size; i++) {
        if (vert->features[i] == feature) {
            return 1;
        }
    }
    return 0;
}

void hvr_vertex_add(hvr_vertex_t *dst, hvr_vertex_t *src, hvr_ctx_t ctx) {
    unsigned src_n_features;
    unsigned src_features[HVR_MAX_VECTOR_SIZE];
    hvr_vertex_unique_features(src, src_features, &src_n_features);

    for (unsigned i = 0; i < src_n_features; i++) {
        const double src_val = hvr_vertex_get(src_features[i], src, ctx);
        double dst_val = 0.0;
        if (hvr_vertex_have_feature(src_features[i], dst)) {
            // Increment existing feature
            dst_val = hvr_vertex_get(src_features[i], dst, ctx);
        }
        hvr_vertex_set(src_features[i], src_val + dst_val, dst, ctx);
    }
}
