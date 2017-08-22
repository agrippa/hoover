#ifndef _HOOVER_H
#define _HOOVER_H

typedef size_t vertex_id_t;

typedef struct _hvr_internal_ctx_t {
    int pe;
    int npes;

    vertex_id_t n_local_nodes;
    long long n_global_nodes;

    vertex_id_t *vertex_ids;
    void *vertex_metadata;
    size_t vertex_metadata_size;

    vertex_id_t *edges;
    size_t *edge_offsets;
} hvr_internal_ctx_t;

typedef hvr_internal_ctx_t *hvr_ctx_t;

extern void hvr_init(const vertex_id_t n_local_nodes,
        vertex_id_t *vertex_ids, void *vertex_metadata,
        const size_t vertex_metadata_size, vertex_id_t *edges,
        size_t *edge_offsets, hvr_ctx_t *out_ctx);

extern void hvr_cleanup(hvr_ctx_t ctx);

#endif
