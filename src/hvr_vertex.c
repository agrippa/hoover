#include "hvr_vertex.h"
#include "hvr_sparse_vec_pool.h"

hvr_vertex_t *hvr_vertex_create(hvr_ctx_t ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    return hvr_alloc_sparse_vecs(1, ctx);
}

void hvr_vertex_delete(hvr_vertex_t *vert, hvr_ctx_t ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    hvr_free_sparse_vecs(vert, 1, ctx);
}

void hvr_vertex_init(hvr_vertex_t *vert, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    memset(vert, 0x00, sizeof(*vert));

    vert->id = construct_vertex_id(ctx->pe, vert - pool->pool);
    vert->creation_iter = ctx->timestep;
    vert->last_modify_iter = ctx->timestep;
}

void hvr_vertex_set(const unsigned feature, const double val,
        hvr_vertex_t *vert, hvr_ctx_t in_ctx) {
    hvr_internal_ctx_t *ctx = (hvr_internal_ctx_t *)in_ctx;
    assert(HVR_VERTEX_ID_PE(vert->id) == ctx->pe);
    vert->last_modify_iter = ctx->timestep;

    const unsigned size = vert->size;
    for (unsigned i = 0; i < size; i++) {
        if (vert->features[i] == feature) {
            // Replace
            vert->values[i] = val;
            return;
        }
    }

    assert(size < HVR_BUCKET_SIZE); // Can hold one more feature
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
