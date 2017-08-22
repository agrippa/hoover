#ifndef _HOOVER_H
#define _HOOVER_H

/*
 * High-level workflow of a HOOVER program:
 *
 *     hvr_init();
 *
 *     abort = false;
 *
 *     while (!abort) {
 *         hvr_update_vertex_metadata();
 *
 *         hvr_update_edges();
 *
 *         abort = hvr_check_abort();
 *     }
 *
 *     hvr_finalize();  // return to the user code, not a global barrier
 */

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

extern void hvr_finalize(hvr_ctx_t ctx);

#endif
